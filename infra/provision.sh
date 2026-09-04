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

# --- find the image's python ---------------------------------------------
# The vast.ai pytorch image keeps torch in a venv (/venv/main), which a
# non-login ssh shell does not activate. Installing with the bare `pip` on
# PATH therefore targets Debian's system python, which has no torch -- and pip
# then happily downloads a fresh 526 MB torch that does not match the driver.
# Resolve the interpreter explicitly instead of trusting PATH.
VENV_PY=""
for cand in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c 'import torch' 2>/dev/null; then
    VENV_PY="$cand"; break
  fi
done
[ -n "$VENV_PY" ] || { echo "no python with torch found -- wrong image?" >&2; exit 1; }
VENV_BIN="$(dirname "$VENV_PY")"
echo ">> using python: $VENV_PY"

# Pin the torch the image shipped so nothing in requirements can upgrade it.
# Replacing it is the fastest way to break CUDA (SPEC.md 2.4).
"$VENV_PY" - > /root/constraints.txt <<'PYEOF'
import importlib.metadata as md
for pkg in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
PYEOF
echo ">> pinning: $(tr '\n' ' ' < /root/constraints.txt)"

echo ">> installing python deps (this is the slow part, ~3-4 min)"
# No `pip install --upgrade pip`: the image's pip is Debian-managed and cannot
# uninstall itself ("RECORD file not found"). We do not need a newer pip.
"$VENV_PY" -m pip install --no-cache-dir \
  --constraint /root/constraints.txt \
  -r /root/requirements-remote.txt

# If torch moved despite the constraint, fail here rather than letting broken
# CUDA surface halfway through a billed experiment.
echo ">> torch / cuda check"
"$VENV_PY" - <<'PYEOF'
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
# Must be the venv's jupyter: a system jupyter would launch a kernel with no torch.
nohup "$VENV_BIN/jupyter" lab > /var/log/jupyter.log 2>&1 &

# Register the venv as the kernel the notebooks get, by name.
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

echo ">> provisioned. jupyter log: /var/log/jupyter.log"
echo ">> watchdog will self-destroy after ${IDLE_KILL_MIN} min of kernel idleness"
