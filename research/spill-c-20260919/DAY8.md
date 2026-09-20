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

## Implementation and CPU evidence

`44f87f18` adds qualification-only lifetime GPU-eviction counts and the complete
host-bank demand trace (ids, bytes, slots, hits and victims). The trace does not
claim to record all GPU hits or model routing decisions. The existing
`--experts-via-tier` installer also accepts `--expert-bank-host-bytes=N`; it
refuses a budget smaller than one exact record, rather than increasing it.
Default is unchanged (256 MiB ceiling, at most 16 host slots). No new MEMRA
read, unsafe Send/Sync implementation, dependency, or numeric program.

`d69f6303` drives all **2,013 frozen SLRU decisions through fake transfers**:
506 submitted transfers, including delayed publication, cancellation, mandatory
consumer-ready fences, and retirement before budget release. Outcome counts:
24 aborted, 350 admitted, 730 capacity, 142 hit, 482 noop, 129 published,
156 reserved. The frozen decisions are unchanged; only their native whole-file
source hash changed for the new diagnostic counter. This is synthetic CPU
lifetime evidence, not the requested model pressure-trace replay.

`verify-day8.py --cpu` ran all nine checks successfully: fmt, macOS tier check,
Linux-target tier check, all tier tests (**189 passed including doc tests**),
strict tier clippy, strict Linux-target engine lib/run-gen/run-spec clippy,
whitespace, flags census and frozen trace regeneration check. Engine Linux
checks use DOCS_RS placeholders and cannot claim native execution.

## Unfinished native pressure cells / access blocker

The native build at exact source `44f87f18` was started in a new isolated C
worktree. It survived the local 180-second command timeout and was observed
still compiling CUDA. Before build completion could be checked, the approved
SSH ControlMaster disappeared: `ssh -O check` returned
`Control socket connect(...): No such file or directory`. No fresh connection
was opened. The lead has been asked to restore it.

The 4 GiB/8 GiB ON/OFF gen/spec cells, native host-budget refusal, and CPU replay
of those model traces remain **unrun**, not passed. `verify-day8.py` fails closed
on the missing receipts. No GPU eviction or re-read counts are yet claimed.
The native binary/source sidecars and build log still need collection after
access returns.
