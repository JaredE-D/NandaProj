#!/usr/bin/env bash
# Check the local half of the setup. Costs nothing, touches no GPU.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

pass=0; fail=0
ok()   { printf '  \033[32m ok \033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n   -> %s\n' "$1" "$2"; fail=$((fail+1)); }
warn() { printf '  \033[33mwarn\033[0m %s\n   -> %s\n' "$1" "$2"; }

echo "tools"
for t in uv uvx just vastai ssh rsync; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t missing" "see README setup"; fi
done

echo "project"
[ -d .venv ] && ok "local venv" || bad "no .venv" "run: uv venv --python 3.12 && uv pip install -e '.[dev]'"
if .venv/bin/pytest -q >/dev/null 2>&1; then ok "tests pass"; else bad "tests failing" "run: just test"; fi

echo "config"
if [ -f .env ]; then
  ok ".env exists"
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
  [ -n "${VAST_API_KEY:-}" ]  && ok "VAST_API_KEY set"  || bad "VAST_API_KEY empty" "add it to .env"
  [ -n "${JUPYTER_TOKEN:-}" ] && ok "JUPYTER_TOKEN set" || bad "JUPYTER_TOKEN empty" "openssl rand -hex 32"
  [ -n "${HF_TOKEN:-}" ]      && ok "HF_TOKEN set"      || warn "HF_TOKEN empty" "only needed for gated models like Gemma"
else
  bad "no .env" "cp .env.example .env, then fill it in"
fi

echo "vast.ai"
if [ -n "${VAST_API_KEY:-}" ]; then
  if vastai show instances --raw >/dev/null 2>&1; then
    ok "api key works"
    n=$(vastai show instances --raw | .venv/bin/python -c 'import json,sys; print(len(json.load(sys.stdin)))' 2>/dev/null || echo '?')
    if [ "$n" = "0" ]; then ok "no instances running (\$0.00/hr)"
    else warn "$n instance(s) running" "run 'just status' to see the burn rate"; fi
  else
    bad "api key rejected" "check the key at cloud.vast.ai/account"
  fi
fi

echo "mcp"
# Can uvx actually fetch and start the server? This is the real dependency.
if timeout 120 uvx jupyter-mcp-server@latest --help >/dev/null 2>&1; then
  ok "uvx can run jupyter-mcp-server"
else
  bad "uvx cannot run jupyter-mcp-server" "check network, then: uvx jupyter-mcp-server@latest --help"
fi
# .mcp.json expands ${JUPYTER_TOKEN} from Claude Code's own environment, which
# this script cannot inspect. So this is a reminder, not a check.
printf '  \033[36minfo\033[0m .mcp.json reads $JUPYTER_TOKEN from the shell that launched Claude Code.\n'
printf '       fish: set -gx JUPYTER_TOKEN (grep "^JUPYTER_TOKEN=" .env | cut -d= -f2-)\n'
printf '       then restart Claude Code so the MCP server picks it up.\n'

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
