# cx-docsync2 — 2026-08-11

## Initial checkpoint

- Scope: docs/research-only batch for the current serve-home row and the
  `research/spec-landscape-20260810/SURVEY.md` addenda.
- Receipts to quote exactly: `research/newbox-bench-20260811/RESULTS.md` and
  `research/newboxgates-20260811/RESULTS.md`.
- Guardrails: preserve generated PERF marker blocks, append survey addenda
  without rewriting ranks, never run `cargo fmt`, and commit progress early.

## Completed ledger

- `b6acfa12`: initial ledger committed before implementation edits.
- `fa2f46fe`: added a rig row for the two-card pair, the
  historical Max-Q/300 W host label, exact receipt links, and the N=5 summary.
- `6dd7a217`: appended six arXiv-linked MoE/spill addenda without changing the
  existing survey ranks.
- Validation: `python3 tools/update-perf-board.py --check` passed; receipt and
  DSpark paths exist; generated PERF marker blocks were not edited; `cargo fmt`
  was not run.
