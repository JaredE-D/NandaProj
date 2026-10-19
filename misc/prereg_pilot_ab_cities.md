# Pre-registration — answers-only pilot: is the D "Yes" skew the word, the framing, or the truth?

Written 2026-10-18, before the pilot is run. Code: `src/nandaproj/pilot.py`,
`scripts/pilot_gate.py`. Model: gemma-3-4b-it. No activations, final-token probabilities only.

## Why

Under D, Gemma says "Yes" on 75/100 v2 items (H: 51, C1: 52). It "lies" on 41/50 items whose
lie is "Yes" and 16/50 whose lie is "No", so D-honest items are almost all Yes-true
(`misc/results_detector_baselines.md`). In a v2 pair, three things flip together: the true
answer, the lie's word, and the question's framing (asks about a problem vs a good property).

## Design

- **v2 bank**, 100 belief items, conditions H / D / C1, three formats:
  `yesno` (the plan-2 prompt, unchanged), `ab_yesA` ("A) Yes / B) No"), `ab_yesB` (swapped).
  Each item runs under both A/B mappings, so a letter bias cancels exactly.
- **Cities** (Geometry of Truth `cities` + `neg_cities`), 30 cities x 4 statements = 120:
  {affirmative, negated} x {true, false} on the same fact. D/C1 persona: a quiz host who wins
  if the contestant believes the statement is `target`, with `target` drawn per statement
  independently of the truth (seed in the results file). "Conflict" items have
  target != truth; only they can carry a lie.
- An item counts in a format only if it is H-correct in that format. "Lied" = D answer != truth.

## Predictions

- **Q1 (word).** If the skew is the word "Yes": in A/B (both mappings pooled), the v2 D lie-rate
  gap between arms (No-true minus Yes-true) is at most half its Yes/No size.
- **Q2 (framing vs truth, cities, conflict items).** Framing predicts the lie rate differs by
  affirmative vs negated statement; truth predicts it differs by true vs false within each.
  Reported as the 2x2 of lie rates with Wilson 95% CIs; no threshold, this is descriptive.
- **Q3 (letter).** Under D in A/B, P(answer = A) is within 0.40-0.60.

## Pass / fail (the decision this pilot exists for)

A (source, format) passes if, among H-correct items whose D answer can be a lie (all v2 items;
cities conflict items), **each polarity arm has >= 25% lied and >= 25% honest**.
- v2 passes in A/B -> the existing bank is usable with a format change; build the activation
  collection on it.
- Only cities passes -> the skew is item-specific; v3 bank needs redesign before collection.
- Nothing passes -> report it; the lie/honest contrast is not available on this model with
  these prompts, and the next step is a prompt search, not a collection.

Also reported: answer-format compliance (probability mass on the two candidates), and the
Yes/No D p_yes against plan-2's `slot_emitted` as a reproduction check.
