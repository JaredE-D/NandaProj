# NandaProj

Interpretability research environment for a MATS (Neel Nanda stream) application project.
One remote GPU kernel, shared by VSCode and Claude Code.

**Budget: $50. The only real way to overspend is leaving an instance running. Run `just down`.**

## Daily loop

```fish
just up          # rent + provision a GPU box     (~6 min, starts billing)
just tunnel      # in a second terminal; leave it open
just url         # paste this into VSCode -> Existing Jupyter Server
# ... work ...
just sync        # pull results/ back to this machine
just down        # STOP BILLING
```

`just status` shows dollars spent this session; `just burn` shows the total against your budget.
A watchdog on the box self-destroys the instance after 45 minutes of kernel idleness, but treat
that as a safety net, not a habit.

## First-time setup

Already done by the setup pass: `uv`, `just`, and `vastai` are installed in `~/.venvs/tools`
and symlinked into `~/.local/bin`; the local venv is Python 3.12 (your system Python is 3.14,
which TransformerLens does not support).

What you still need to do:

1. **Fill in `.env`** — `VAST_API_KEY` from the vast.ai console's API Keys page, and `HF_TOKEN`
   from your Hugging Face token settings if you want gated models like Gemma.
   `JUPYTER_TOKEN` was generated for you.

2. **Export the Jupyter token** so `.mcp.json` can expand it, then restart Claude Code:
   ```fish
   set -gx JUPYTER_TOKEN (grep '^JUPYTER_TOKEN=' .env | cut -d= -f2-)
   ```

3. **Check everything**: `just doctor`

## Connecting VSCode

With `just tunnel` running:

1. Command palette → **Jupyter: Select Interpreter to Start Jupyter Server**
2. Choose **Existing** → paste the output of `just url`
3. Open a notebook, pick the remote kernel from the kernel picker

The tunnel binds the remote Jupyter to a local port. Jupyter itself listens only on the box's
loopback interface, so the token is never exposed to the internet.

## How Claude Code sees the same kernel

`.mcp.json` points `jupyter-mcp-server` at the same tunneled port. So when you hit a confusing
tensor shape, Claude can inspect the live variables in *your* kernel rather than guessing.
This requires the tunnel to be open and `jupyter-collaboration` on the server (it's in
`infra/requirements-remote.txt`).

## Layout

```
infra/vast.sh         rent / connect / destroy; all vast.ai logic
infra/parse.py        parses `vastai --raw` JSON
infra/provision.sh    runs on the box: deps, jupyter, watchdog
infra/watchdog.sh     idle self-destruct
infra/doctor.sh       local setup check
src/nandaproj/config.py   model presets, paths, device -- switch scale here
notebooks/            00_smoke_test.ipynb proves the stack works
results/              gitignored; `just sync` target
```

## Persistence

`/workspace` on the box is where everything lives: the git checkout, `hf_cache/` (as `HF_HOME`),
and `results/`. Model weights land in the cache so re-renting doesn't mean re-downloading.
**Anything outside `/workspace` dies with the instance**, and `results/` is only on your machine
once you've run `just sync`.

## Switching model scale

`src/nandaproj/config.py` holds the presets. Default is `tiny` (gpt2-small). Set
`NANDA_PRESET=gemma` for Gemma-2-2B, and raise `MAX_PRICE` in `.env` if you need a bigger card.
Start small — iteration speed matters more than model size in a 20-hour project.
