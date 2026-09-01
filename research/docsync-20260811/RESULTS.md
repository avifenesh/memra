# cx-docsync — final verdict — 2026-08-11

## Verdict

**PASS — documentation sync complete.**

The three confirmed drift findings were reconciled against the merged receipts:

- `docs/FLAGS.md` now records `MEMRA_SWA_RING` as a validated serving-config
  opt-in, with default OFF and the deliberate prefix-cache `serve-smoke`
  SCOPED RED cell plus flip policy as the remaining doors.
- `docs/FLAGS.md` now describes the merged `MEMRA_SPEC_PIPE` TX-ticket seam,
  `MEMRA_SPEC_PIPE_TRACE`, the zero-added-sync serial path, the +13.94% / -47.89%
  attribution, default OFF, and increment-1 state-fork GO.
- `research/kv256-20260809/RESULTS.md` has only its status header amended; its
  historical measurement body remains unchanged and its capacity claims point
  to the 2 -> 12 / 6.0x receipt rows.

Authoritative receipts cited: `research/ringval-20260810/RESULTS.md`,
`research/newboxgates-20260811/RESULTS.md`, and
`research/opti0-20260810/RESULTS.md`.

## Verification

- `git diff --check`: PASS
- `python3 tools/update-perf-board.py --check`: PASS
- No README/PERFORMANCE PERF marker blocks changed.
- No `cargo fmt` run.

Commits: `a68a2d9b` (initial checkpoint), `4da54bcc` (docs sync), and the
commit containing this verdict.
