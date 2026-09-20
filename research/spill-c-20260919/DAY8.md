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
lifetime evidence. The separate model-pressure replay below extends it with
the complete recorded host-bank demand streams.

`verify-day8.py --cpu` ran all ten checks successfully: fmt, macOS tier check,
Linux-target tier check, all tier tests (**193 passed including doc tests**),
strict tier clippy, strict Linux-target engine lib/run-gen/run-spec clippy,
whitespace, flags census, frozen trace regeneration and verifier red probes. Engine Linux
checks use DOCS_RS placeholders and cannot claim native execution.

## Native pressure cells

The lead restored the approved SSH master after the interruption. The completed
native build and its source/binary digests were collected from source
`44f87f181bbbcc75a14d8d9362609f60186f293d`. Commands use these frozen binaries;
receipt-only commits do not rebuild them. Each completed cell is synced,
hash-replayed, committed and pushed before starting the next.

Run order: 8 GiB banked gen/spec, then their native controls; 4 GiB follows.
A budget includes each slot's native eight-byte tail padding: 9,986 slots =
8,589,637,648 bytes; 4,993 slots = 4,294,818,824 bytes. Both round **down** from
the requested GiB capacity.

| Cell | GPU evictions | Host evictions | Physical reads | Re-reads |
|---|---:|---:|---:|---:|
| 8 GiB banked gen | 12,091 | 22,061 | 22,077 | 4,389 |
| 8 GiB banked spec | 63,996 | 73,966 | 73,982 | 53,684 |
| 4 GiB banked gen | 26,990 | 31,967 | 31,983 | 17,553 |
| 4 GiB banked spec | 178,523 | 183,500 | 183,516 | 165,768 |

8 GiB gen ON/OFF and 4 GiB gen ON each report verbatim:

> prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH

8 GiB spec ON/OFF and 4 GiB spec ON each report verbatim:

> === SELF-CONSISTENCY PASS ===

All eight K rows pass. Gen and spec ON/OFF token tapes are byte-identical within
each numeric class, also matching the respective day-seven baseline; ON/OFF
spec acceptance rows are identical. The 4 GiB ON tapes and acceptance rows
also match their 8 GiB counterparts exactly. CPU replay of all four complete
recorded host-bank demand traces passes: **311,558 decisions and fake transfers**. This is host-bank replacement evidence, not a reconstructed
trace of every native GPU hit or a new numerical model fixture.

## Remaining interruption

After syncing and pushing the 4 GiB spec receipt at `d9ab7944`, the approved
ControlMaster disappeared a second time. No fresh connection was opened. Four
GiB OFF controls and the host-budget refusal have not started; native strict
clippy was started before this interruption, but its completion is uncollected.
No claim is made that these pending cells passed.

`pressure-refusal.py` is ready for collector-only execution. It retains native
stdout/stderr and native exit status, emits `REFUSED:` and exits 2 **only** when
run-gen exits 1 with the exact one-record host-budget refusal and no dispatch.
Unexpected errors/signals remain failures; CPU red probes exercise this rule.
This is a host-record minimum: existing GPU slot sizing clamps small requests
to eight slots and must not be described as a GPU-budget refusal.
