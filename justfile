# NandaProj -- every operation is one verb.
# Run `just` with no argument to see this list.

vast := "bash infra/vast.sh"

default:
    @just --list

# --- money-spending verbs ------------------------------------------------

# Show candidate GPU offers without renting anything.
search:
    @{{vast}} search

# Rent the cheapest matching GPU and provision it. Starts billing.
up:
    @{{vast}} up

# Sync results/ back, then destroy the instance. Run whenever you stop working.
# `just down --force` skips the sync (use only if the box is unreachable).
down *ARGS:
    @{{vast}} down {{ARGS}}

# Instance state and dollars spent.
status:
    @{{vast}} status

# Total spend against the budget in .env (local estimate).
burn:
    @{{vast}} burn

# Real vast.ai account balance and how many GPU-hours it buys.
balance:
    @{{vast}} balance

# --- connection ----------------------------------------------------------

# Forward the remote Jupyter to a local port. Foreground; ctrl-C to close.
tunnel:
    @{{vast}} tunnel

# Print the URL to paste into VSCode's "Existing Jupyter Server" prompt.
url:
    @printf 'http://127.0.0.1:%s/lab?token=%s\n' "${LOCAL_PORT:-8888}" "$(grep '^JUPYTER_TOKEN=' .env | cut -d= -f2-)"

# Shell on the box.
ssh *ARGS:
    @{{vast}} ssh {{ARGS}}

# Re-run provisioning on the existing instance.
provision:
    @{{vast}} provision

# Pull results/ back from the box.
sync:
    @{{vast}} sync

# Push local code to the box (notebooks, src).
push:
    #!/usr/bin/env bash
    set -euo pipefail
    read -r host port <<<"$(bash infra/vast.sh hostport)"
    rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'results' \
      -e "ssh -p $port" ./ "root@$host:/workspace/NandaProj/"

# --- local ---------------------------------------------------------------

test:
    @.venv/bin/pytest -q

lint:
    @.venv/bin/ruff check . && .venv/bin/ruff format --check .

# Verify the local half of the setup is sane.
doctor:
    @bash infra/doctor.sh
