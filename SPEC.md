# NandaProj — Environment Spec

**Goal:** an interp research environment for a MATS (Neel Nanda stream) application project.
Constraints: ~20 hours of human time, <$50 total compute, no local GPU (14 GB RAM machine).

## 0. Guiding decisions

| Decision | Choice | Why |
|---|---|---|
| Kernel location | Remote vast.ai, reached by SSH tunnel | No local GPU; tunnel means VSCode *and* the MCP server share one kernel |
| Model scale | **`gemma-3-4b-it`** on 1x 4090 (~8 GB bf16) | Revised up from ≤2B: verbalized confidence is degenerate in small models, and the project in `PLAN2.md` needs it to exist. Still ~$0.30/hr. Debug on `gemma-3-270m-it`. |
| Libraries | jlens + HF transformers; sae_lens optional | Revised: `PLAN2.md` needs the Jacobian lens and linear probes, not TransformerLens. Keeping TL installed but off the critical path avoids the v3 `TransformerBridge` migration risk. |
| Persistence | git repo + HF cache on a vast.ai volume | Destroying instances must be painless, weight re-download must not be |
| Real cost risk | Idle instances, not GPU tier | 4090 forgotten for a week = $50. Guardrails target this, not price shopping |

**Time budget:** setup ≤2h, leaving ~18h of research. If setup starts exceeding this, we cut features, not research time.

## 1. Architecture

```
Local (Arch, no GPU)                 SSH tunnel            vast.ai 1x4090 (~$0.30/hr)
┌──────────────────────────┐                           ┌────────────────────────────┐
│ VSCode                   │                           │ JupyterLab 4 + RTC         │
│  └ Jupyter: Existing     │──── port 8888 ───────────▶│  └ ipykernel (python 3.11) │
│     Server → :8888       │     -L 8888:local:8888    │     torch/TL/nnsight/SAE   │
│                          │                           │                            │
│ Claude Code              │                           │ /workspace  (vast volume)  │
│  └ jupyter-mcp-server ───┼──── same port 8888 ──────▶│   ├ NandaProj/ (git)       │
│     (uvx, stdio)         │                           │   └ hf_cache/ (HF_HOME)    │
│                          │                           │                            │
│ ./results/ ◀─────────────┼──── rsync over ssh ───────│ results/                   │
└──────────────────────────┘                           └────────────────────────────┘
```

One kernel, two clients. You edit and run cells in VSCode; I read and execute the *same* kernel
through MCP, so we see identical state and outputs.

## 2. Deliverables

### 2.1 Repo scaffold
```
NandaProj/
├── SPEC.md
├── README.md                  # 6-line quickstart: up / tunnel / work / down
├── .env.example               # VAST_API_KEY, HF_TOKEN, JUPYTER_TOKEN
├── .mcp.json                  # jupyter-mcp-server, project-scoped
├── .vscode/settings.json      # Jupyter existing-server + notebook hygiene
├── justfile                   # every operation is one verb
├── infra/
│   ├── vast.sh                # search / create / ssh / tunnel / sync / status / destroy
│   ├── provision.sh           # runs ON the box: env + jupyter + libs
│   ├── requirements-remote.txt
│   └── watchdog.sh            # idle self-destruct on the box
├── notebooks/
│   └── 00_smoke_test.ipynb    # proves GPU, TL, nnsight, SAE load
├── src/nandaproj/
│   ├── __init__.py
│   ├── config.py              # model name, device, paths — one place to switch scale
│   └── viz.py                 # plotting helpers (Neel-style imshow/line)
├── results/                   # gitignored, rsync target
└── tests/
    └── test_config.py
```

### 2.2 Command surface (justfile)
```
just up          # rent cheapest matching 4090, provision, print token      (~6 min)
just tunnel      # ssh -N -L ... (foreground, ctrl-C to stop)
just status      # running instances, $/hr, hours elapsed, spend to date
just sync        # rsync results/ back to local
just down        # sync results/ back, then destroy the instance
just burn        # total $ spent so far this project
```

### 2.3 Cost guardrails (the part that actually protects the $50)
1. `infra/watchdog.sh` on the box: if no kernel execution for 45 min, self-destroy via vast API.
2. `just status` prints elapsed cost in dollars, not hours.
3. `just up` refuses to create a second instance if one is already running.
4. A `$50` ceiling recorded in `.env`; `just burn` warns at 60% and 85%.
5. README's loudest line is "run `just down` when you stop working."

### 2.4 Remote env (`requirements-remote.txt`)
Base: vast.ai PyTorch template image (CUDA + torch preinstalled — we do not build torch).
Added: `jupyterlab jupyter-collaboration ipykernel jupyter-mcp-tools`,
`transformer-lens nnsight sae-lens datasets einops jaxtyping plotly circuitsvis`.
Pinned to a lockfile after the first successful install so re-provisioning is deterministic.

### 2.5 MCP wiring
`.mcp.json` (project-scoped, so it only loads in this repo): runs `uvx jupyter-mcp-server@latest`
over stdio, pointed at the tunneled local port with `JUPYTER_URL` / `JUPYTER_TOKEN` /
`ALLOW_IMG_OUTPUT=true`.

Requires `jupyter-collaboration` on the server (RTC is how the MCP sees notebook state) — hence
it's in the remote requirements. Requires `uv` locally (not currently installed; task 1 installs it).

## 3. Verification (each is a real check, not a claim)
- **V1** `just up` → instance reaches "running", ssh succeeds.
- **V2** On the box: `nvidia-smi` shows the GPU; `torch.cuda.is_available()` → True.
- **V3** Tunnel open → the Jupyter `/api` endpoint returns JSON locally.
- **V4** VSCode connects to the existing server and shows the remote kernel; a cell prints the
  *remote* hostname and GPU name.
- **V5** I execute a cell via MCP and read its output.
- **V6** `00_smoke_test.ipynb` loads `gpt2-small` in TransformerLens, runs a forward pass, caches
  activations, and loads one Gemma Scope SAE.
- **V7** `just down` → `just status` shows zero instances.
- **V8** `pytest` passes on the local package.

## 4. Task decomposition

**Phase A — local, no spend (~30 min)**
1. Install `uv`; create local venv (py3.12 — your system python is 3.14, which TransformerLens
   does not support).
2. Repo scaffold, git init, `.gitignore`, `src/`, `tests/`, `pytest` green.
3. `.mcp.json` + `.vscode/settings.json`.
4. Install + authenticate `vastai` CLI with your API key; `vastai show instances` works.

**Phase B — remote, spend begins (~45 min)**
5. `infra/vast.sh search` — list candidate 4090 offers with price, verify the filter.
6. `just up` on the cheapest → V1.
7. `provision.sh` → V2.
8. Set `HF_HOME` to `/workspace/hf_cache` on the instance disk. **No persistent volume:**
   the disk dies with the instance, so weights re-download (~8 GB, ~5 min) each rental.
   Accepted deliberately — a volume would bill continuously between sessions.

**Phase C — wiring (~30 min)**
9. `just tunnel` → V3.
10. VSCode existing-server connection → V4.
11. MCP connection → V5.
12. Smoke-test notebook → V6.

**Phase D — guardrails (~20 min)**
13. `watchdog.sh` + `just status` / `just burn` → V7.
14. README quickstart.

**Phase E — research scoping** ✅ *done — see [`PLAN2.md`](PLAN2.md)*
15. ~~Shortlist of candidate projects, pick one.~~ Chosen; `PLAN2.md` carries the research plan and
    its own phases, gates, and risks from here.

## 5. Project

**Chosen — see [`PLAN2.md`](PLAN2.md).** *Where does a lie get made?* Under an instructed lie the
honest answer is known by construction, so the J-lens can ask whether the true answer is still in
the **reportable** subspace at the layer where the model says something else — and activation
patching can then name the heads and MLPs that keep it out of the output.

Plan 1 (`misc/plan1old.md`) asked the same availability-vs-routing question about the
confidence–faithfulness gap. It is superseded because its primary arm inferred from absence; the
deception framing turns the same measurement into a positive detection. Its pre-registration is
kept unrun.

The candidate shortlist that preceded it (SAE feature ablation, circuit replication, refusal
direction transfer, SAE seed stability) is superseded and no longer tracked here.

## 6. Explicit non-goals
- No multi-GPU / model-parallel plumbing.
- No local torch/CUDA install (14 GB RAM, no GPU — not worth it).
- No Docker image build; we use vast.ai's template image.
- No experiment tracking service (W&B) unless you ask — one more account, little payoff here.

## 7. Open risks
- **R1** vast.ai offer availability fluctuates; `just up` must handle "no matching offer" gracefully.
- **R2** jupyter-collaboration / JupyterLab version skew is the most common failure mode for this
  MCP. If RTC breaks, the fallback is pinning the versions the MCP docs test against.
- **R3** Some vast.ai hosts have poor network; the first `just up` may need a retry elsewhere.
- **R4** Volume availability is per-datacenter, which constrains which offers we can pick. If it
  bites, the fallback is git-only plus weight re-download (~5 min/rental).
