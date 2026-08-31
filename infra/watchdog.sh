#!/usr/bin/env bash
# Idle self-destruct. Runs ON the box, in the background.
#
# The single biggest budget risk in this project is an instance left running
# overnight (~$8/day for a 4090). This kills it after IDLE_KILL_MIN minutes
# with no kernel activity.
#
# "Activity" = the newest `last_activity` across all kernels. A long-running
# training cell keeps its kernel `busy`, which counts as activity, so this
# will not kill a job that is genuinely working.
set -euo pipefail

IDLE_KILL_MIN="${IDLE_KILL_MIN:-45}"
CHECK_EVERY=120
GRACE_MIN=20  # never kill within this many minutes of boot

started=$(date +%s)

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "watchdog up: idle limit ${IDLE_KILL_MIN}m, grace ${GRACE_MIN}m"

while true; do
  sleep "$CHECK_EVERY"

  now=$(date +%s)
  if [ $(((now - started) / 60)) -lt "$GRACE_MIN" ]; then
    continue
  fi

  idle_min=$(curl -sf -H "Authorization: token $JUPYTER_TOKEN" \
               127.0.0.1:8888/api/kernels 2>/dev/null \
             | python3 -c '
import json, sys, datetime
try:
    kernels = json.load(sys.stdin)
except Exception:
    print(-1); sys.exit()          # jupyter unreachable: do not kill on this
if not kernels:
    print(9999); sys.exit()        # no kernel at all: definitely idle
now = datetime.datetime.now(datetime.timezone.utc)
def age(k):
    t = k["last_activity"].replace("Z", "+00:00")
    return (now - datetime.datetime.fromisoformat(t)).total_seconds() / 60
if any(k.get("execution_state") == "busy" for k in kernels):
    print(0)                       # something is actively running
else:
    print(min(age(k) for k in kernels))
' 2>/dev/null || echo -1)

  if [ "$idle_min" = "-1" ]; then
    log "jupyter unreachable; not counting as idle"
    continue
  fi

  log "idle for ${idle_min%%.*} min (limit ${IDLE_KILL_MIN})"

  if [ "${idle_min%%.*}" -ge "$IDLE_KILL_MIN" ]; then
    log "IDLE LIMIT REACHED -- destroying instance $INSTANCE_ID"
    vastai destroy instance "$INSTANCE_ID" --api-key "$VAST_API_KEY" || \
      log "destroy failed; will retry next cycle"
  fi
done
