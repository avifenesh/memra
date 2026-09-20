# Day 8 — banked expert residency under pressure

Repository: `avifenesh/memra`, lane `lane/spill-c-20260919`.
Development correctness only: rented RTX 5090, configured 400 W / maximum
600 W, overlay storage (not NVMe spill speed), collector-only GPU cells.
No production/default/model-support promotion.

## Day-seven seal and transfer integration

Day-seven report and recovered final device-publication receipt are committed
at `9ac958fa`; `verify-day7.py` replays all ten collector cells successfully.
The interrupted verifier contained a corrupted token regex; repaired before
sealing and checked against the actual 32-token tapes.

Merged A's final `06fda471` at `a061524f`. Replayed C's banked device delta in
an isolated scratch worktree against that exact import. Linux-target engine lib
and `qwen4exp_gpu_gate` check PASS (`DOCS_RS=1`, compile-only MMQ placeholder).
Native macOS engine check ran and still refuses existing Linux-only
`spill_pread.rs` libc APIs. Initial checks used the wrong hyphenated gate target;
those failures and the corrected-target retries are retained under `raw/day8-cpu/`.
This compile check does not replace the day-seven native GPU receipt.

## Pressure protocol

The 4/8 GiB budgets address the **native GPU SLRU**, using its existing
`MEMRA_MOE_SLOTS` override. The host bank remains bounded to 16 records and
256 MiB; do not describe these as 4/8 GiB host banks. All model cells retain
prompt ids `[55,88,13]`, `MEMRA_NGEN=32`, and `MEMRA_MOE_RESIDENT=0`.

Pressure cells, refusal, frozen fixture and final checks: pending.
