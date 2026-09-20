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
read, unsafe Send/Sync implementation, dependency, or numeric program. This
remains the existing default-OFF qualification door; decide-by: 2026-10-04.

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

Both budgets, gen ON and OFF, report verbatim:

> prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH

Both budgets, spec ON and OFF, report verbatim:

> === SELF-CONSISTENCY PASS ===

All eight K rows pass. Gen and spec ON/OFF token tapes are byte-identical within
each numeric class, also matching the respective day-seven baseline; ON/OFF
spec acceptance rows are identical. The 4 GiB ON/OFF tapes and acceptance rows
also match their 8 GiB counterparts exactly. CPU replay of all four complete
recorded host-bank demand traces passes: **311,558 decisions and fake transfers**. This is host-bank replacement evidence, not a reconstructed
trace of every native GPU hit or a new numerical model fixture.

## Refusal and GPU-budget seam note

The one-byte host-budget cell reports verbatim:

> Error: "experts-via-tier host bank budget cannot hold one expert record"
> native_exit_code=1
> REFUSED: experts-via-tier host bank budget cannot hold one expert record

The collector records **status `refused`, exit 2**, not a successful run.
`pressure-refusal.py` retains the entire native output and exit status, and
normalizes only that exact rejection to the gate's exit-2 convention. No host
bank demand was dispatched. Unexpected errors/signals remain failures; the
classifier and receipt-identity red probes pass. This is a **host-record minimum**
refusal (largest required record 860,160 bytes), not a GPU-budget refusal.

**Seam note for the lead:** `MoeSlotCache::new` computes uniform slot count with
`(budget_bytes / (max_block_bytes + 8)).max(8)`. Thus `MEMRA_MOE_SLOTS=0..7`
is clamped to **eight slots**, not refused, and cannot express the intended
below-minimum GPU-budget refusal. This lane records the behavior and does not
change generic native cache sizing. A strict GPU-budget admission seam needs
its own decision and tests; the host refusal must not be relabeled as that gate.

## Final verification and boundaries

- `verify-day8.py`: **PASS**, all nine native captures, raw/telemetry hashes,
  exact commands/prompts/arms, configured/max power, canonical locks, nonzero
  pressure counts, ON/OFF tapes, all K rows, explicit refusal, and source/binary
  identities. Postcheck records the clean native source and unchanged binary
  hashes after all nine cells ended. `verify-day7.py` also replays PASS.
- Native release engine lib + run-gen/run-spec strict clippy: **exit 0**, at
  exact runtime source `44f87f18`; full log and exit sidecar retained.
- Final Mac-side battery: all ten checks pass, including **193 tier tests**,
  four recorded-pressure CPU replays, strict clippy, fmt, flags and whitespace.
  Rechecks preserve their own timestamped directories and tested-file hashes;
  earlier checkpoints remain retained. Mac **engine** compilation still has
  the already-recorded Linux-only libc blocker; Linux-target checks are not
  mislabeled macOS engine execution.
- Every native cell was synced, committed and pushed independently. Receipt
  commits: 8 GiB gen/spec ON `ae0dc8cc`/`623bf70a`, OFF `1c15a4d1`/`161217c6`;
  4 GiB gen/spec ON `5c26f72f`/`d9ab7944`, OFF `ffbdefda`/`a0ae2e79`;
  refusal `7b40bae9`. Unskipped push hooks ran throughout.
- SSH master losses interrupted collection twice; the lead restored access.
  Builds survived and their completed logs were collected. No fresh lane-owned
  connections, new locks, or bare GPU runs were used.

The evidence is checkpoint-faithful Qwen expert byte provenance and SLRU
correctness on the rented RTX 5090, plus the separately banked day-seven device
publication fixture. No performance decision, PP/RPC qualification, production
support-state change, or PRO-pair release qualification is implied. Lead owns
integration of the banked device delta; nothing here is merged to main.

Day-eight effort is approximately **one active agent-hour** (about two elapsed
hours including transport waits), within the three-hour session bound. This
closes the assigned day-eight scope of the eight-day lane; cumulative active
hours for the preceding seven days are not present in this receipt and are not
invented here.
