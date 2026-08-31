#!/usr/bin/env bash
# Runs ON the vast.ai box. Idempotent: safe to re-run on the same instance.
#
# Expects in the environment: JUPYTER_TOKEN, HF_TOKEN, VAST_API_KEY,
# INSTANCE_ID, IDLE_KILL_MIN.
set -euo pipefail

WORKSPACE=/workspace
mkdir -p "$WORKSPACE"/{hf_cache,results,NandaProj}

# Persist env for every future shell and for the Jupyter kernel.
cat > /etc/profile.d/nandaproj.sh <<EOF
export WORKSPACE=$WORKSPACE
export HF_HOME=$WORKSPACE/hf_cache
export HF_TOKEN=${HF_TOKEN:-}
export PYTHONPATH=$WORKSPACE/NandaProj/src
EOF
# shellcheck disable=SC1091
. /etc/profile.d/nandaproj.sh

echo ">> installing python deps (this is the slow part, ~3-4 min)"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r /root/requirements-remote.txt

echo ">> torch / cuda check"
python - <<'PYEOF'
import torch
print(f"   torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   gpu: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA not available -- wrong image or a bad host; destroy and retry")
PYEOF

# --- jupyter -------------------------------------------------------------
# Bind to loopback only. The sole way in is the SSH tunnel, so the token is
# never exposed to the public internet.
mkdir -p /root/.jupyter
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
nohup jupyter lab > /var/log/jupyter.log 2>&1 &

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

echo ">> provisioned. jupyter log: /var/log/jupyter.log"
echo ">> watchdog will self-destroy after ${IDLE_KILL_MIN} min of kernel idleness"
