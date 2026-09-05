#!/usr/bin/env bash
# Idle self-stop. Runs ON the box, in the background.
#
# The single biggest budget risk in this project is an instance left running
# overnight (~$8/day for a 4090). This STOPS it after IDLE_KILL_MIN minutes
# with no activity: the GPU is released and billing drops to storage
# (~1-7 c/hr), while the disk -- venv, HF cache, /workspace -- survives, so
# resuming skips the ~9 min image pull, the ~4 min pip install and the model
# download.
#
# It stops rather than destroys because destroying threw all of that away every
# time. Two consequences, both handled on the LAPTOP, not here:
#
#   - Nothing on a stopped box can restart it. `just resume` does that.
#   - A stopped box bills storage forever if forgotten, so vast.sh's reaper
#     destroys it after STOPPED_MAX_H. The bill still reaches zero on its own.
#
# This cannot sync results/ first -- it has no route to the laptop. Anything
# written since the last `just sync` sits on that disk until you resume, and
# dies if the reaper gets there first. `just stop` syncs; this cannot.
#
# --- what counts as activity, and why it is not just jupyter -----------------
#
# The first version asked jupyter for its kernels and treated an EMPTY kernel
# list as "definitely idle" -- it printed 9999 minutes, which cleared any
# threshold, so the box stopped on the next 120-second cycle. Every one of these
# looked like a random stop mid-session:
#
#   - a kernel that OOMs loading a 4b model (the most likely one here)
#   - restarting the kernel, which briefly leaves zero kernels
#   - closing the notebook in VSCode, which shuts its kernel down
#   - working in a terminal on the box before opening any notebook
#
# An absent kernel is an absence of *evidence*, not evidence of idleness, so it
# no longer shortcuts the timer. And cell execution was never the only way to be
# working: reading output, editing a file and staring at a layer table are all
# work, and 45 minutes of them is an ordinary afternoon in this project.
#
# So activity is the newest of several signals, and the log names which one is
# holding the box alive -- an unexplained stop is exactly what wasted the time
# this comment exists to stop wasting.
set -euo pipefail

IDLE_KILL_MIN="${IDLE_KILL_MIN:-45}"
# An open ssh/tunnel connection says a human is attached but not that they are
# doing anything -- a tunnel forgotten overnight is the failure this whole file
# exists to prevent. So presence buys a longer leash, never an unlimited one.
IDLE_KILL_CONNECTED_MIN="${IDLE_KILL_CONNECTED_MIN:-180}"
CHECK_EVERY=120
GRACE_MIN=20  # never stop within this many minutes of boot

WORKSPACE="${WORKSPACE:-/workspace}"
HOLD_FILE="$WORKSPACE/.nokill"          # `just hold` -- disables stopping outright
STAMP=/var/run/nandaproj_last_active

started=$(date +%s)
date +%s > "$STAMP"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "watchdog up: idle limit ${IDLE_KILL_MIN}m (${IDLE_KILL_CONNECTED_MIN}m while connected), grace ${GRACE_MIN}m"

# Newest kernel activity as an epoch, or empty if jupyter cannot be reached or
# has no kernels. Empty means "no signal", which is not the same as idle.
kernel_activity_epoch() {
  curl -sf -H "Authorization: token $JUPYTER_TOKEN" \
       127.0.0.1:8888/api/kernels 2>/dev/null \
  | python3 -c '
import json, sys, datetime
try:
    kernels = json.load(sys.stdin)
except Exception:
    sys.exit()                      # unreachable: no signal
if not kernels:
    sys.exit()                      # no kernels: no signal, NOT idleness
now = datetime.datetime.now(datetime.timezone.utc)
if any(k.get("execution_state") == "busy" for k in kernels):
    print(int(now.timestamp())); sys.exit()   # something is running right now
best = 0
for k in kernels:
    t = k.get("last_activity")
    if not t:
        continue
    ts = datetime.datetime.fromisoformat(t.replace("Z", "+00:00"))
    best = max(best, int(ts.timestamp()))
if best:
    print(best)
' 2>/dev/null || true
}

# Is the GPU doing anything? Catches compute started outside jupyter -- a script
# under `just ssh`, or a kernel that has since died leaving work running.
gpu_busy() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local util
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
          | sort -rn | head -1)"
  [ -n "$util" ] && [ "$util" -ge 5 ] 2>/dev/null
}

# Anything written under /workspace recently? Catches editing and saving, and a
# long job writing results, neither of which touches a kernel's last_activity.
# hf_cache is excluded: an 8.6 GB model download is real work, but the cache is
# also touched by reads, and walking it is the one expensive part of this check.
files_touched() {
  local mins="$1"
  [ -d "$WORKSPACE" ] || return 1
  find "$WORKSPACE" -xdev \
       \( -path "$WORKSPACE/hf_cache" -o -name '.ipynb_checkpoints' -o -name '*.log' \) -prune \
       -o -type f -newermt "-${mins} min" -print -quit 2>/dev/null | grep -q .
}

# Is a human attached? An `ssh -N -L` tunnel is an established connection to 22.
connected() {
  if command -v ss >/dev/null 2>&1; then
    ss -tn state established 2>/dev/null | awk '{print $3}' | grep -q ':22$' && return 0
  fi
  pgrep -f 'sshd: root@' >/dev/null 2>&1
}

while true; do
  sleep "$CHECK_EVERY"

  now=$(date +%s)
  if [ $(((now - started) / 60)) -lt "$GRACE_MIN" ]; then
    continue
  fi

  if [ -f "$HOLD_FILE" ]; then
    log "hold file present ($HOLD_FILE); not stopping"
    continue
  fi

  reason=""
  if gpu_busy; then
    reason="gpu busy"
  elif files_touched 5; then
    reason="files written under $WORKSPACE"
  fi
  if [ -n "$reason" ]; then
    date +%s > "$STAMP"
  fi

  # Kernel activity is a timestamp rather than a yes/no, so it can be older than
  # now and still be the newest thing we know about.
  kern="$(kernel_activity_epoch)"
  stamp="$(cat "$STAMP" 2>/dev/null || echo "$started")"
  if [ -n "$kern" ] && [ "$kern" -gt "$stamp" ] 2>/dev/null; then
    stamp="$kern"
    echo "$stamp" > "$STAMP"
    reason="${reason:-kernel activity}"
  fi

  idle_min=$(((now - stamp) / 60))

  limit="$IDLE_KILL_MIN"
  if connected; then
    limit="$IDLE_KILL_CONNECTED_MIN"
  fi

  log "idle ${idle_min}m / limit ${limit}m${reason:+ (active: $reason)}$(connected && echo ' [connected]' || true)"

  if [ "$idle_min" -ge "$limit" ]; then
    log "IDLE LIMIT REACHED -- stopping instance $INSTANCE_ID"
    log "  unsynced results stay on the disk; 'just resume' then 'just sync'"
    if vastai stop instance "$INSTANCE_ID" --api-key "$VAST_API_KEY"; then
      log "stopped; this watchdog dies with the box"
      exit 0
    fi
    log "stop failed; will retry next cycle"
  fi
done
