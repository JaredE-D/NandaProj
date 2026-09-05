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
MIN_CUDA="${MIN_CUDA:-12.4}"

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
#
# cuda_max_good is the highest CUDA the *host driver* supports, and it must
# cover $IMAGE or the rental is dead on arrival. Machine 13863 (driver
# 535.230.02, cuda_max_good 12.2) was rented under the 12.4.1 image: the image
# fell back to its forward-compatibility libs in /usr/local/cuda/compat, those
# are only supported on data-center GPUs, and on a GeForce 4090 CUDA init died
# with error 804 (cudaErrorCompatNotSupportedOnDevice) after provisioning had
# already billed for the box. Filtering at search time is the only cheap place
# to catch it -- everything downstream costs a rental.
_query() {
  local excl=""
  for m in ${EXCLUDE_MACHINES:-}; do
    excl="$excl machine_id!=$m"
  done
  echo "gpu_name=$GPU_NAME num_gpus=1 rentable=true reliability>0.98 \
disk_space>$DISK_GB inet_down>$MIN_INET dph<$MAX_PRICE cuda_max_good>=$MIN_CUDA \
${GEO_FILTER:-}$excl"
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

  # Capture first, write second. `> "$STATE/instance_id"` truncates the file
  # before the pipeline runs, so a failed create used to leave an empty
  # instance_id behind and report only "instance creation failed" -- hiding the
  # API's actual reason, which parse.py had already printed above it.
  local new_id
  new_id="$($VASTAI create instance "$offer" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --onstart-cmd "$onstart" \
    --env "-e JUPYTER_TOKEN=$JUPYTER_TOKEN -e HF_TOKEN=${HF_TOKEN:-} -p 8888:8888" \
    --raw | "$PY" "$ROOT/infra/parse.py" new-id)" \
    || die "create failed -- see the reason above; nothing was rented"
  [ -n "$new_id" ] || die "create returned no instance id; nothing was rented"
  printf '%s\n' "$new_id" > "$STATE/instance_id"
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
      "$ROOT/infra/watchdog.sh" "$ROOT/infra/services.sh" "root@$host:/root/"
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

  rm -f "$STATE/instance_id" "$STATE/started_at" "$STATE/dph" "$STATE/stopped_at"
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

# --- stopped instances ---------------------------------------------------
# A *stopped* instance keeps its disk -- the venv, the HF cache (~8.6 GB of
# gemma), /workspace -- and releases the GPU. Restarting skips the ~9 min image
# pull, the ~4 min pip install and the model download, for roughly 1-7 c/hr of
# storage instead of ~33 c/hr of GPU.
#
# Two things this buys you are not free:
#
#   1. Stopping is NOT a reservation. The host can rent the GPU to someone else
#      while you are stopped, and then `start` fails. cmd_resume reports that
#      and changes nothing -- rent a fresh box yourself if you want one.
#   2. Nothing on a stopped box can run, so nothing on the box can ever restart
#      it, reap it, or sync results off it. Both are local-side jobs.
#
# Hence the reaper below. The watchdog stops an idle box; if nobody resumes it
# within STOPPED_MAX_H, this destroys it, so the bill still reaches zero on its
# own. Its clock starts when a local command first *observes* the stopped state,
# not when the watchdog stopped it -- a stopped box is unreachable and vast does
# not report a stop time. That errs toward paying a few more cents of storage,
# never toward destroying early.
#
# THE CATCH, and it is a real one: results written since the last sync live on
# that disk and die with it. `just stop` syncs first for exactly this reason;
# a watchdog-initiated stop cannot, so `just status` warns while the clock runs.
STOPPED_MAX_H="${STOPPED_MAX_H:-24}"

_instance_state() {
  $VASTAI show instance "$(instance_id)" --raw 2>/dev/null \
    | "$PY" "$ROOT/infra/parse.py" cur-state || true
}

_reap_stopped() {
  have_instance || return 0
  local state now first age_h
  state="$(_instance_state)"
  case "$state" in
    stopped|exited)
      now=$(date +%s)
      if [ ! -s "$STATE/stopped_at" ]; then
        echo "$now" > "$STATE/stopped_at"
        echo ">> instance $(instance_id) is $state; storage-billing clock starts now"
      fi
      first="$(cat "$STATE/stopped_at")"
      age_h="$("$PY" -c "print(($now-$first)/3600)")"
      if "$PY" -c "import sys; sys.exit(0 if float('$age_h') >= float('$STOPPED_MAX_H') else 1)"; then
        echo ">> stopped for ${age_h%%.*}h (limit ${STOPPED_MAX_H}h) -- destroying $(instance_id)."
        echo "   Any results not synced before it stopped are gone with the disk."
        cmd_burn_record
        yes 2>/dev/null | $VASTAI destroy instance "$(instance_id)" || true
        rm -f "$STATE/instance_id" "$STATE/started_at" "$STATE/dph" "$STATE/stopped_at"
        echo ">> reaped."
      else
        "$PY" - "$age_h" "$STOPPED_MAX_H" <<'PYEOF'
import sys
age, limit = float(sys.argv[1]), float(sys.argv[2])
print(f"   stopped {age:.1f}h ago; auto-destroy in {limit-age:.1f}h "
      f"('just resume' to keep it, 'just down' to end it now)")
PYEOF
      fi
      ;;
    *)
      # Running again (or gone): the stop clock no longer applies.
      rm -f "$STATE/stopped_at"
      ;;
  esac
}

# Stop without destroying. Syncs first: the reaper may later destroy this disk
# and nothing can be pulled off a stopped box.
cmd_stop() {
  have_instance || { echo "no instance to stop"; return 0; }
  echo ">> syncing results/ before stopping $(instance_id) ..."
  if ! cmd_sync; then
    die "sync failed -- nothing was stopped.
  A stopped box cannot be synced, and the reaper destroys it after ${STOPPED_MAX_H}h,
  so stopping now would put results on a disk with a deadline and no way off it.
  Fix the connection and retry, or 'just down --force' to give up on the box."
  fi
  $VASTAI stop instance "$(instance_id)" || die "stop failed"
  date +%s > "$STATE/stopped_at"
  echo ">> stopped $(instance_id). Storage still bills (~1-7 c/hr); GPU does not."
  echo ">> 'just resume' to bring it back, or it self-destroys in ${STOPPED_MAX_H}h."
}

# Start a stopped instance and restart its services. The disk survived a stop,
# but every process on it died, so jupyter and the watchdog need relaunching --
# services.sh, not provision.sh, because the packages are already installed.
cmd_resume() {
  have_instance || die "no instance to resume -- 'just up' rents a new one"
  local state
  state="$(_instance_state)"
  if [ "$state" = "running" ]; then
    echo ">> instance $(instance_id) is already running"
  else
    echo ">> starting $(instance_id) ..."
    if ! $VASTAI start instance "$(instance_id)"; then
      die "start failed -- the host has most likely rented the GPU to someone else.
  Stopping is not a reservation. Nothing has been changed or destroyed; the disk
  is still there and still billing storage. Your options:
    just status               # is it still stopped, and how long until it reaps?
    just down                 # give up on this box (results are NOT recoverable
                              #   from a stopped disk -- it must start to sync)
    just up                   # rent a fresh box (only after 'just down')"
    fi
  fi
  rm -f "$STATE/stopped_at"
  # Billing restarts, so the burn clock does too; the stopped hours were storage
  # and are not what `just burn` is estimating.
  date +%s > "$STATE/started_at"
  cmd_wait
  echo ">> restarting jupyter and the watchdog ..."
  cmd_services
  echo ">> resumed. 'just tunnel' to reconnect."
}

cmd_services() {
  have_instance || die "no instance"
  local host port
  read -r host port <<<"$(cmd_hostport)"
  # shellcheck disable=SC2029
  ssh -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" \
    "JUPYTER_TOKEN='$JUPYTER_TOKEN' VAST_API_KEY='$VAST_API_KEY' \
     INSTANCE_ID='$(instance_id)' IDLE_KILL_MIN='$IDLE_KILL_MIN' \
     bash /root/services.sh"
}

cmd_status() {
  _reap_stopped
  if ! have_instance; then
    echo "no instance running -- \$0.00/hr"
    cmd_burn
    return 0
  fi
  local st
  st="$(_instance_state)"
  case "$st" in
    stopped|exited)
      echo "instance $(instance_id): $st (GPU released, storage still billing)"
      echo "   results written since the last sync are on that disk only"
      cmd_burn
      return 0
      ;;
  esac
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
  stop)       shift; cmd_stop "$@" ;;
  resume)     shift; cmd_resume "$@" ;;
  services)   shift; cmd_services "$@" ;;
  down)       shift; cmd_down "$@" ;;
  *) echo "usage: vast.sh {search|up|adopt|wait|provision|ssh|tunnel|hostport|sync|status|burn|balance|down}" >&2; exit 2 ;;
esac
