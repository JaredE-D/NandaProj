#!/usr/bin/env bash
# vast.sh -- rent, connect to, and destroy a single vast.ai GPU box.
#
# Every subcommand is idempotent and safe to re-run. State lives in .state/
# (gitignored): which instance we own, when it started, what it costs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/.state"
PY="$ROOT/.venv/bin/python"
VASTAI="${VASTAI:-vastai}"
mkdir -p "$STATE"

if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
else
  echo "error: no .env -- copy .env.example to .env and fill it in" >&2
  exit 1
fi

GPU_NAME="${GPU_NAME:-RTX_4090}"
MAX_PRICE="${MAX_PRICE:-0.50}"
BUDGET_USD="${BUDGET_USD:-50}"
IDLE_KILL_MIN="${IDLE_KILL_MIN:-45}"
IMAGE="${IMAGE:-vastai/pytorch:cuda-12.4.1-auto}"
DISK_GB="${DISK_GB:-40}"
LOCAL_PORT="${LOCAL_PORT:-8888}"
MIN_INET="${MIN_INET:-500}"

die() { echo "error: $*" >&2; exit 1; }

# vast.ai does not reliably inject the account's ssh key into a new container,
# so `attach ssh` is called explicitly. Idempotent: "already associated" is a
# success for our purposes, not an error.
_attach_key() {
  local pub
  pub="$(ls "$HOME"/.ssh/*.pub 2>/dev/null | head -1)"
  [ -n "$pub" ] || { echo "   no local ssh key to attach" >&2; return 0; }
  $VASTAI attach ssh "$(instance_id)" "$(cat "$pub")" >/dev/null 2>&1 || true
}
have_instance() { [ -s "$STATE/instance_id" ]; }
instance_id() { cat "$STATE/instance_id"; }

# --- offer search --------------------------------------------------------
# Cheapest single-GPU offers matching the filter, with enough disk and a
# reliability floor -- cheap unreliable hosts cost more in wasted time than
# they save in dollars.
# GEO_FILTER keeps the box near you. Latency is paid on every cell execution
# through the tunnel, and the price spread across regions is pennies per hour --
# so region beats price here. Empty GEO_FILTER means "anywhere".
# inet_down is the host's *advertised* figure, not a measurement: machine
# 148361 advertised 216 Mbit/s and delivered about 3. Raising the floor is a
# weak filter but a free one; EXCLUDE_MACHINES is the part that actually works,
# so add a machine_id here whenever a host turns out to be slow.
_query() {
  local excl=""
  for m in ${EXCLUDE_MACHINES:-}; do
    excl="$excl machine_id!=$m"
  done
  echo "gpu_name=$GPU_NAME num_gpus=1 rentable=true reliability>0.98 \
disk_space>$DISK_GB inet_down>$MIN_INET dph<$MAX_PRICE ${GEO_FILTER:-}$excl"
}

cmd_search() {
  $VASTAI search offers "$(_query)" -o 'dph+' --raw \
    | "$PY" "$ROOT/infra/parse.py" offers
}

# --- lifecycle -----------------------------------------------------------
cmd_up() {
  if have_instance; then
    die "instance $(instance_id) already exists -- 'just status', or 'just down' first"
  fi

  echo ">> searching for $GPU_NAME under \$$MAX_PRICE/hr ..."
  local offer dph
  offer="$($VASTAI search offers "$(_query)" -o 'dph+' --raw \
            | "$PY" "$ROOT/infra/parse.py" cheapest-id)"
  [ -n "$offer" ] || die "no matching offer; raise MAX_PRICE or change GPU_NAME in .env"
  dph="$($VASTAI search offers "$(_query)" -o 'dph+' --raw \
            | "$PY" "$ROOT/infra/parse.py" cheapest-price)"
  echo ">> renting offer $offer at \$$dph/hr"

  # vast.ai's own ssh-key injection has been unreliable on this account: the
  # API reports the key as associated with the instance while the container's
  # authorized_keys never receives it, so sshd rejects the key without even
  # asking for a signature. Reboots, detach/reattach, and re-registering the
  # key all failed across three hosts. So write authorized_keys ourselves at
  # container start and stop depending on their injection at all.
  local pub onstart
  pub="$(cat "$(ls "$HOME"/.ssh/*.pub 2>/dev/null | head -1)" 2>/dev/null || true)"
  [ -n "$pub" ] || die "no ssh public key in ~/.ssh -- run 'just doctor'"
  onstart="mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
echo '$pub' >> /root/.ssh/authorized_keys && \
sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys && \
chmod 600 /root/.ssh/authorized_keys"

  $VASTAI create instance "$offer" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --onstart-cmd "$onstart" \
    --env "-e JUPYTER_TOKEN=$JUPYTER_TOKEN -e HF_TOKEN=${HF_TOKEN:-} -p 8888:8888" \
    --raw | "$PY" "$ROOT/infra/parse.py" new-id > "$STATE/instance_id"

  have_instance || die "instance creation failed"
  date +%s > "$STATE/started_at"
  echo "$dph" > "$STATE/dph"
  echo ">> instance $(instance_id) created; waiting for it to come up ..."
  _attach_key
  cmd_wait

  # The search offer's dph excludes the disk we actually asked for -- a 40 GB
  # box quoted at $0.313 bills at $0.355. Record the instance's real rate, or
  # `just burn` under-reports for the whole project.
  local real
  real="$($VASTAI show instance "$(instance_id)" --raw 2>/dev/null \
          | "$PY" "$ROOT/infra/parse.py" dph || true)"
  if [ -n "$real" ]; then
    echo "$real" > "$STATE/dph"
    echo ">> actual rate \$$real/hr (search quoted \$$dph)"
  fi
  echo ">> provisioning ..."
  cmd_provision
  echo
  echo ">> ready. Next:  just tunnel   (in another terminal)"
}

# Take ownership of an instance created elsewhere (the web console), so the
# rest of the tooling -- provision, tunnel, sync, status, down -- works on it.
# Useful when `up` cannot produce a reachable box but the console can.
cmd_adopt() {
  local id="${1:-}"
  [ -n "$id" ] || die "usage: just adopt <instance_id>"
  if have_instance && [ "$(instance_id)" != "$id" ]; then
    die "already tracking instance $(instance_id) -- 'just down' it first"
  fi
  $VASTAI show instance "$id" --raw >/dev/null 2>&1 \
    || die "instance $id not found on this account"
  echo "$id" > "$STATE/instance_id"
  date +%s > "$STATE/started_at"
  local real
  real="$($VASTAI show instance "$id" --raw | "$PY" "$ROOT/infra/parse.py" dph || true)"
  echo "${real:-0}" > "$STATE/dph"
  echo ">> adopted instance $id at \$${real:-?}/hr"
  echo ">> note: elapsed time counts from now, so 'just burn' undercounts any"
  echo "   time it ran before adoption."
  echo ">> next:  just provision"
}

# Boot means pulling a multi-GB CUDA image on the host's connection. Five
# minutes is not enough; 20 is comfortable. The instance bills throughout, so
# this timeout only decides when we stop *watching*, never when we stop paying.
cmd_wait() {
  have_instance || die "no instance"
  local i status tries
  tries=$(( ${BOOT_TIMEOUT_MIN:-20} * 12 ))   # one poll per 5s
  for i in $(seq 1 "$tries"); do
    status="$($VASTAI show instance "$(instance_id)" --raw \
              | "$PY" "$ROOT/infra/parse.py" status || true)"
    if [ "$status" = "running" ]; then
      # SSH takes a while after the status flips, and vast.ai does not reliably
      # inject the account key into the container at creation -- an explicit
      # `attach ssh` is what actually pushes it, and propagation to the box can
      # take a minute or more. So: retry for 5 minutes, re-attaching partway
      # through rather than failing on the first refusal.
      local j
      for j in $(seq 1 60); do
        if cmd_ssh true 2>/dev/null; then echo ">> ssh is up"; return 0; fi
        if [ "$j" = 6 ] || [ "$j" = 24 ]; then
          echo; echo "   ssh refused -- (re)attaching key and waiting for it to propagate"
          _attach_key
        fi
        printf '\r   waiting for ssh (%ds)' "$((j * 5))"
        sleep 5
      done
      die "instance is running but ssh never answered after 5 min.
  The key is registered but the box is not accepting it. Try:
    just ssh true             # see the raw error
    just down --force         # give up on this host and rent another"
    fi
    printf '\r   status=%-12s (%ds)' "${status:-pending}" "$((i * 5))"
    sleep 5
  done
  die "gave up waiting after ${BOOT_TIMEOUT_MIN:-20} min (last status: ${status:-pending}).
  The instance is still alive and still BILLING -- this timeout only stopped the
  watching, not the boot. Nothing is necessarily wrong. Either:
    just status               # is it 'running' now?
    just provision            # finish the setup this wait was going to do
    just down                 # or stop paying for it
  Raise BOOT_TIMEOUT_MIN in .env if slow image pulls are routine on your hosts."
}

cmd_provision() {
  have_instance || die "no instance"
  local host port
  read -r host port <<<"$(cmd_hostport)"
  scp -o StrictHostKeyChecking=accept-new -P "$port" \
      "$ROOT/infra/provision.sh" "$ROOT/infra/requirements-remote.txt" \
      "$ROOT/infra/watchdog.sh" "root@$host:/root/"
  # shellcheck disable=SC2029
  ssh -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" \
    "JUPYTER_TOKEN='$JUPYTER_TOKEN' HF_TOKEN='${HF_TOKEN:-}' \
     VAST_API_KEY='$VAST_API_KEY' INSTANCE_ID='$(instance_id)' \
     IDLE_KILL_MIN='$IDLE_KILL_MIN' bash /root/provision.sh"
}

# Destroying is irreversible and takes results/ with it, so we pull results
# back first and refuse to destroy if that fails. --force skips the sync, for
# when the box is unreachable and destroying it is the whole point.
cmd_down() {
  have_instance || { echo "no instance to destroy"; return 0; }
  local id force=0
  if [ "${1:-}" = "--force" ]; then force=1; fi
  id="$(instance_id)"

  if [ "$force" -eq 1 ]; then
    echo ">> --force: skipping results sync"
  else
    echo ">> syncing results/ off the box before destroying $id ..."
    if ! cmd_sync; then
      die "sync failed -- results on the box are NOT saved, so nothing was destroyed.
  Fix the connection and retry, or 'just down --force' to destroy anyway."
    fi
  fi

  cmd_burn_record

  # `vastai destroy` prompts "[y/N]". With no stdin it aborts *and still exits
  # 0*, so neither the prompt nor the exit code can be trusted. Feed it yes,
  # then confirm against the API before deleting local state -- forgetting the
  # instance id while the box is alive is strictly worse than not destroying
  # it, because `just status` would then report nothing to destroy.
  yes 2>/dev/null | $VASTAI destroy instance "$id" || true

  local i gone=0
  for i in $(seq 1 12); do
    sleep 3
    if ! $VASTAI show instances --raw 2>/dev/null \
         | "$PY" "$ROOT/infra/parse.py" has-instance "$id"; then
      gone=1; break
    fi
  done

  if [ "$gone" -ne 1 ]; then
    die "instance $id is STILL RUNNING and still billing after a destroy attempt.
  Local state has been left intact so 'just down' can be retried.
  Destroy it by hand now:  vastai destroy instance $id
  Or in the web console under Instances."
  fi

  rm -f "$STATE/instance_id" "$STATE/started_at" "$STATE/dph"
  echo ">> destroyed $id -- confirmed gone from the account"
}

# --- connection ----------------------------------------------------------
cmd_hostport() {
  have_instance || die "no instance"
  $VASTAI show instance "$(instance_id)" --raw \
    | "$PY" "$ROOT/infra/parse.py" hostport
}

cmd_ssh() {
  local host port
  read -r host port <<<"$(cmd_hostport)"
  ssh -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" "$@"
}

cmd_tunnel() {
  local host port
  read -r host port <<<"$(cmd_hostport)"
  echo ">> tunnelling local $LOCAL_PORT to jupyter on $host:$port"
  echo ">> ctrl-C closes the tunnel (the instance keeps running and billing)"
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 \
      -N -L "$LOCAL_PORT:127.0.0.1:8888" -p "$port" "root@$host"
}

cmd_sync() {
  local host port
  read -r host port <<<"$(cmd_hostport)"
  mkdir -p "$ROOT/results"
  # Ensure the source exists before rsync runs. Without this, a box that has
  # simply not produced results yet fails the same way an unreachable box does
  # -- and cmd_down keys its refusal on that exit status.
  ssh -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" \
      "mkdir -p /workspace/results"
  rsync -avz -e "ssh -p $port" "root@$host:/workspace/results/" "$ROOT/results/"
}

# --- money ---------------------------------------------------------------
_elapsed_hours() {
  [ -s "$STATE/started_at" ] || { echo 0; return; }
  "$PY" -c "import time,sys;print((time.time()-float(open('$STATE/started_at').read()))/3600)"
}

cmd_status() {
  if ! have_instance; then
    echo "no instance running -- \$0.00/hr"
    cmd_burn
    return 0
  fi
  local hrs dph
  hrs="$(_elapsed_hours)"; dph="$(cat "$STATE/dph")"
  "$PY" - "$hrs" "$dph" "$(instance_id)" <<'PYEOF'
import sys
hrs, dph, iid = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
print(f"instance {iid}: up {hrs:.2f} h at ${dph:.3f}/hr = ${hrs*dph:.2f} this session")
PYEOF
  cmd_burn
}

cmd_burn_record() {
  local hrs dph
  hrs="$(_elapsed_hours)"; dph="$(cat "$STATE/dph" 2>/dev/null || echo 0)"
  "$PY" -c "print(f'{float('$hrs')*float('$dph'):.4f}')" >> "$STATE/spend.log"
}

# What vast.ai says you actually have, as opposed to cmd_burn's local estimate
# of what this repo rented. Checked before `up`, this is what catches a box
# that will die mid-run because the account is empty.
cmd_balance() {
  local dph rate_note raw
  if [ -s "$STATE/dph" ]; then
    dph="$(cat "$STATE/dph")"; rate_note="current instance"
  else
    dph="$MAX_PRICE"; rate_note="MAX_PRICE ceiling; real 4090s run ~\$0.30"
  fi
  # The JSON travels in argv, not a pipe: a heredoc script already occupies
  # this process's stdin, so a pipe into it would be silently discarded.
  raw="$($VASTAI show user --raw)"
  "$PY" - "$dph" "$raw" "$rate_note" <<'PYEOF'
import json, sys
dph = float(sys.argv[1])
d = json.loads(sys.argv[2])
avail = float(d.get("credit", 0)) + float(d.get("balance", 0))
print(f"vast.ai available: ${avail:.2f}   (lifetime spend ${float(d.get('total_spend', 0)):.2f})")
if dph > 0:
    print(f"runway at ${dph:.3f}/hr: {avail / dph:.1f} GPU-hours  ({sys.argv[3]})")
if avail < 5:
    print("  !! under $5 -- top up at cloud.vast.ai/billing/ before 'just up'")
PYEOF
}

cmd_burn() {
  local session=0
  if have_instance; then
    session="$("$PY" -c "print(float('$(_elapsed_hours)')*float('$(cat "$STATE/dph")'))")"
  fi
  "$PY" - "$STATE/spend.log" "$BUDGET_USD" "$session" <<'PYEOF'
import sys, pathlib
log, budget, session = pathlib.Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
past = sum(float(x) for x in log.read_text().split()) if log.exists() else 0.0
total = past + session
pct = 100 * total / budget if budget else 0
print(f"spend: ${total:.2f} of ${budget:.2f} budget ({pct:.0f}%)")
if pct >= 85:
    print("  !! over 85% of budget -- wrap up or lower MAX_PRICE")
elif pct >= 60:
    print("  ! past 60% of budget")
PYEOF
}

case "${1:-}" in
  search)     shift; cmd_search "$@" ;;
  up)         shift; cmd_up "$@" ;;
  wait)       shift; cmd_wait "$@" ;;
  adopt)      shift; cmd_adopt "$@" ;;
  provision)  shift; cmd_provision "$@" ;;
  ssh)        shift; cmd_ssh "$@" ;;
  tunnel)     shift; cmd_tunnel "$@" ;;
  hostport)   shift; cmd_hostport "$@" ;;
  sync)       shift; cmd_sync "$@" ;;
  status)     shift; cmd_status "$@" ;;
  burn)       shift; cmd_burn "$@" ;;
  balance)    shift; cmd_balance "$@" ;;
  down)       shift; cmd_down "$@" ;;
  *) echo "usage: vast.sh {search|up|adopt|wait|provision|ssh|tunnel|hostport|sync|status|burn|balance|down}" >&2; exit 2 ;;
esac
