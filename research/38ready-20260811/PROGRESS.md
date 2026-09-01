# Qwen3.8-27B readiness ledger progress

Date: 2026-08-11
Branch: `lane/cx-38ready`

## Scope

Produce a docs-only readiness ledger for a same-architecture Qwen3.8-27B bring-up. The ledger will
separate already staged preparation from missing exact-artifact evidence, order the future
acquisition and validation steps, and enumerate the measurements required by the beside-math
analysis.

## Constraints

- Read-only research lane: no code changes, GPU work, model downloads, or model acquisition.
- No business, priority, timing, or acquisition-source decision.
- No merge, tag, push, formatting sweep, or performance-board change.
- Repository evidence will be cited by file and line.

## Checklist

- [x] Read the lane inbox and project instructions.
- [x] Inventory the prior 3.8 prep, architecture kit, quantization direction, and beside-plan/math receipts.
- [x] Write `READINESS.md` with STAGED vs MISSING evidence.
- [x] Record the ordered exact-artifact acquisition and validation checklist.
- [x] Record the first-boot measurements needed to resolve the beside-math unknowns.
- [x] Verify citations, scope, and the final diff.

## Completion

`READINESS.md` records the reusable runbook/tooling as staged while keeping the exact Qwen3.8
artifact, literal GGUF, validated drafter pairing, golden-output receipts, and target-specific
memory terms missing. No code, model bytes, GPU work, acquisition, performance board, merge, tag,
push, or formatting action was performed.
