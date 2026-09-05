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

# --- long-running services ----------------------------------------------
# jupyter + watchdog live in services.sh so that `just resume` can restart them
# on a stopped-and-started box without repeating the pip install above.
WORKSPACE="$WORKSPACE" IDLE_KILL_MIN="$IDLE_KILL_MIN" \
  JUPYTER_TOKEN="$JUPYTER_TOKEN" VAST_API_KEY="$VAST_API_KEY" \
  INSTANCE_ID="$INSTANCE_ID" bash /root/services.sh

echo ">> provisioned."
