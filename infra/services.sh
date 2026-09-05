#!/usr/bin/env bash
# Start (or restart) the long-running processes on the box: jupyter and the
# idle watchdog. Runs ON the box.
#
# Split out of provision.sh because a *resumed* instance keeps its disk -- the
# venv, the HF cache, /workspace -- but loses every running process. Re-running
# provision.sh to get jupyter back would pay ~4 minutes of pip install to
# reinstall packages that are already there. This is the part that actually
# needs re-running, and it takes seconds.
#
# Idempotent: both services are pkill'd before being started, so running this
# twice is a restart, not two copies.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
IDLE_KILL_MIN="${IDLE_KILL_MIN:-45}"
: "${JUPYTER_TOKEN:?JUPYTER_TOKEN is required}"
: "${INSTANCE_ID:?INSTANCE_ID is required}"
: "${VAST_API_KEY:?VAST_API_KEY is required}"

# Same resolution as provision.sh: the image keeps torch in a venv that a
# non-login ssh shell does not activate, so the bare `jupyter` on PATH would
# launch a kernel with no torch in it.
VENV_PY=""
for cand in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c 'import torch' 2>/dev/null; then
    VENV_PY="$cand"; break
  fi
done
[ -n "$VENV_PY" ] || { echo "no python with torch found -- wrong image?" >&2; exit 1; }
VENV_BIN="$(dirname "$VENV_PY")"
echo ">> using python: $VENV_PY"

# --- jupyter -------------------------------------------------------------
# Bind to loopback only. The sole way in is the SSH tunnel, so the token is
# never exposed to the public internet.
mkdir -p /root/.jupyter "$WORKSPACE"
cat > /root/.jupyter/jupyter_lab_config.py <<EOF
c.ServerApp.ip = "127.0.0.1"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.root_dir = "$WORKSPACE"
c.IdentityProvider.token = "${JUPYTER_TOKEN}"
c.ServerApp.terminado_settings = {"shell_command": ["/bin/bash"]}
EOF

pkill -f "jupyter-lab" 2>/dev/null || true
sleep 2
nohup "$VENV_BIN/jupyter" lab > /var/log/jupyter.log 2>&1 &

"$VENV_PY" -m ipykernel install --name nandaproj --display-name "NandaProj (GPU)" >/dev/null 2>&1 || true

echo ">> waiting for jupyter"
for _ in $(seq 1 30); do
  if curl -sf -H "Authorization: token $JUPYTER_TOKEN" \
       127.0.0.1:8888/api >/dev/null 2>&1; then
    echo "   jupyter is up"
    break
  fi
  sleep 2
done

# --- idle watchdog -------------------------------------------------------
pkill -f "watchdog.sh" 2>/dev/null || true
JUPYTER_TOKEN="$JUPYTER_TOKEN" VAST_API_KEY="$VAST_API_KEY" \
  INSTANCE_ID="$INSTANCE_ID" IDLE_KILL_MIN="$IDLE_KILL_MIN" \
  nohup bash /root/watchdog.sh > /var/log/watchdog.log 2>&1 &

echo ">> services up. logs: /var/log/jupyter.log /var/log/watchdog.log"
echo ">> watchdog will STOP the instance after ${IDLE_KILL_MIN} min of kernel idleness"
