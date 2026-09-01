# cx-docsync — 2026-08-11

## Initial checkpoint

- Scope: reconcile the three confirmed external-review documentation drifts from
  the merged `ringval-20260810`, `newboxgates-20260811`, and `opti0-20260810`
  receipts.
- Planned edits: the `MEMRA_SWA_RING` and `MEMRA_SPEC_PIPE` rows in
  `docs/FLAGS.md`, plus the status header of
  `research/kv256-20260809/RESULTS.md`.
- Guardrails: docs-only; preserve historical measurement bodies; do not touch
  generated PERF marker blocks; never run `cargo fmt`.

## Completed ledger

- `docs/FLAGS.md` / `MEMRA_SWA_RING`: replaced the stale box1-blocked claim with
  validated serving-config opt-in. The row records ring wrap byte-identity, the
  one-hash golden, `run-spec` K=1..8, the 2 -> 12 / 6.0x capacity result, and
  the reproduction on the new Workstation Edition pair. It keeps default OFF
  and names the deliberate `serve-smoke` prefix-cache SCOPED RED cell plus
  default-flip policy as the remaining doors.
- `docs/FLAGS.md` / `MEMRA_SPEC_PIPE`: updated the pre-seam description to the
  merged `verify_stage0_issue -> VerifyBoundaryTicket ->
  verify_stage1_finish` TX-ticket flow, including `MEMRA_SPEC_PIPE_TRACE`, the
  zero-added-sync serial path, measured +13.94% over serial / -47.89% vs plain,
  default OFF, and increment-1 state-fork GO.
- `research/kv256-20260809/RESULTS.md`: amended only the status header to mark
  flag-ON receipts complete, retain default OFF, name the prefix-cache and
  flip-policy doors, and point capacity claims at the 2 -> 12 / 6.0x rows.

Receipts cited exactly: `research/ringval-20260810/RESULTS.md`,
`research/newboxgates-20260811/RESULTS.md`, and
`research/opti0-20260810/RESULTS.md`. Generated PERF marker blocks and the
historical kv256 measurement body were not edited.
