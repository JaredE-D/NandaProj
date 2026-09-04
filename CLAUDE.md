# NandaProj — project instructions

Research project for a MATS application (Neel Nanda stream). See `SPEC.md` for the
environment and `PLAN2.md` for the research plan (`misc/plan1old.md` is the superseded
first plan, kept for its pre-registration).

## Commits

**Do not add a `Claude-Session:` trailer (or any `claude.ai/code/session_...` link)
to commit messages in this repo.** `Co-Authored-By:` is fine.

This repo is likely to be shared with MATS reviewers. A session link is a durable
identifier pointing at a private transcript, git history is permanent once pushed,
and the link is of no use to anyone reading the repo.

## Secrets

`.env` holds `VAST_API_KEY`, `HF_TOKEN`, and `JUPYTER_TOKEN` and is gitignored.
`.env.example` is the committed template and must keep every secret value blank.

**Never print a file or API response that may contain a secret.** Two HF tokens
have already been leaked into a transcript this way: once from an instance's
`extra_env` via the vast.ai API, once from `/etc/profile.d/nandaproj.sh` on the
box. Both files legitimately hold tokens. Grep for the specific line you need,
or redact before printing -- do not `cat` them.

Before any commit that touches infra, check that no secret value appears in a
tracked file. Search for the literal values from `.env`, not for patterns —
pattern matching misses tokens whose format you guessed wrong.

## Who runs what

**Jared runs the important scripts, not Claude.** Anything that rents, destroys,
provisions, or otherwise spends money or changes remote state is his to execute,
so he can see what is happening as it happens.

That means Claude does **not** run: `just up`, `just down`, `just provision`,
`just tunnel`, `vastai create/destroy/attach/detach`, or long background jobs
that drive them. Instead, print the exact command and let him run it — in Claude
Code he can prefix it with `!` to run it in-session so the output lands in the
conversation.

Claude may still run, without asking: read-only checks (`just status`,
`just balance`, `just search`, `vastai show ...`), local tests and linters, and
edits to files in this repo.

Why: a background `just up` hides a 12-minute failure behind a task
notification, and three GPU rentals were burned on debugging Jared could have
diagnosed in one look at his own terminal. Visibility is worth more than the
round trip.

## Money

Every GPU hour is billed. `just balance` reports the real vast.ai balance;
`just burn` is only a local `dph x elapsed` estimate and can drift.

- `just down` when work stops. Closing the tunnel does **not** stop billing.
- Prefer `just down` over `just down --force`: the plain form syncs `results/`
  off the box first and refuses to destroy if that sync fails.
- Budget is $50 total for the project. Flag it before running anything that
  would spend more than a few dollars in one go.

## Notebooks

**Edit `.ipynb` files with the `NotebookEdit` tool only.** Never `Write`, never
`Edit`, never a `python -c "import json ..."` rewrite of the file.

`NotebookEdit` addresses one cell at a time by id and the change shows up live
in Jared's open notebook, so he can see each cell land and stop a wrong turn
after one cell instead of after twenty. Whole-file writes replace the notebook
under an editor that already has it open, drop execution counts and outputs,
and turn a small change into an unreviewable diff.

Reading is unrestricted: `Read` renders the notebook cell by cell with ids, and
a `json.load` for a quick grep of every cell at once is fine. The rule is about
writes.

## Working style

Follow the global workflow in `~/.claude/CLAUDE.md`: brainstorm, spec, decompose,
then execute — do not jump to implementation code.

Verify claims about this repo by running the thing, not by reading it. Several
bugs here were found only by executing the path (`down` refusing to destroy,
the heredoc that swallowed a pipe's stdin).
