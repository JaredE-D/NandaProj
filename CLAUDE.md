# NandaProj — project instructions

Research project for a MATS application (Neel Nanda stream). See `SPEC.md` for the
environment and `PLAN.md` for the research plan.

## Commits

**Do not add a `Claude-Session:` trailer (or any `claude.ai/code/session_...` link)
to commit messages in this repo.** `Co-Authored-By:` is fine.

This repo is likely to be shared with MATS reviewers. A session link is a durable
identifier pointing at a private transcript, git history is permanent once pushed,
and the link is of no use to anyone reading the repo.

## Secrets

`.env` holds `VAST_API_KEY`, `HF_TOKEN`, and `JUPYTER_TOKEN` and is gitignored.
`.env.example` is the committed template and must keep every secret value blank.

Before any commit that touches infra, check that no secret value appears in a
tracked file. Search for the literal values from `.env`, not for patterns —
pattern matching misses tokens whose format you guessed wrong.

## Money

Every GPU hour is billed. `just balance` reports the real vast.ai balance;
`just burn` is only a local `dph x elapsed` estimate and can drift.

- `just down` when work stops. Closing the tunnel does **not** stop billing.
- Prefer `just down` over `just down --force`: the plain form syncs `results/`
  off the box first and refuses to destroy if that sync fails.
- Budget is $50 total for the project. Flag it before running anything that
  would spend more than a few dollars in one go.

## Working style

Follow the global workflow in `~/.claude/CLAUDE.md`: brainstorm, spec, decompose,
then execute — do not jump to implementation code.

Verify claims about this repo by running the thing, not by reading it. Several
bugs here were found only by executing the path (`down` refusing to destroy,
the heredoc that swallowed a pipe's stdin).
