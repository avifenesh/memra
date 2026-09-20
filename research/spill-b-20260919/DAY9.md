# WP-B day 9 — fixed-VA VMM physical reclaim

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`.
Native VMM source: **f3a3247a027ca2f36bff2cf194669712f564c76b**.
Residual-diagnostic source: **55f82783f74a317c8698bbc3e641fe378c00ea60**.
Development correctness evidence only; **not merged, released, deployed, serving
qualified, or a performance/default promotion**.

## Diagnosis first (verbatim)

```text
RECLAIM-DIAG: freed but not observable
```

The initial explanation that the gate released only prefix subviews is withdrawn.
It transfers whole, independently allocated K/V planes. The diagnostic native
cell at `ef0ebe8c6` observes all **32** source backing Rcs go **1→0**, reaching
their `CudaSlice` destructors; backend device registry occupancy becomes **0**.
Pool used bytes fall by **243,269,856 B**, exactly source allocation bytes
**243,269,888 B** less **32** one-byte cache placeholders. No source remains
owned by the gate, detached layer or materializer.

Checked `cuDeviceGetDefaultMemPool` and `cuMemPoolTrimTo(pool, 0)` both succeed.
Pool reserved remains **15,200,157,696 B**; driver-free VRAM remains
**17,934,516,224 B** before demote, before trim, after trim and after restore.
A trim-only fix is therefore insufficient. The driver reservation's internal
reason is not inferred; fragmentation is a hypothesis, not a measured cause.

The diagnostic control reproduces the frozen day-8 result:

```text
ACTIVE-8K copy/restore bit-identical, no reclaim — not G1 PASS
```

Full evidence and alternatives: [RECLAIM-DESIGN.md](RECLAIM-DESIGN.md).

## VMM result (verbatim)

```text
ACTIVE-8K G1 PASS
ACTIVE-32K physical reclaim/restore bit-identical, one-granule residual unclassified — not G1 PASS
```

| Observation | 8k native VMM result |
| --- | ---: |
| Suspended prompt / generated continuation | 8064 / 128 tokens |
| Demoted / restored planes | 32 / 32 |
| Queried allocation granularity | 2,097,152 B |
| Native plane capacity (including existing tail pads) | 243,269,888 B |
| Rounded VMM physical capacity | 301,989,888 B |
| Logical D2H / pinned host bytes | 239,468,544 B |
| Whole-chunk physical release | **201,326,592 B** |
| Partial-edge and unused-capacity chunks retained | 100,663,296 B |
| Driver-free before demote | 17,613,651,968 B |
| Driver-free after demote | 17,814,978,560 B |
| Driver-free after restore | 17,613,651,968 B |
| Observed reclaim / reacquisition | **201,326,592 / 201,326,592 B** |
| Virtual addresses | Original per-plane VA retained through restore |
| Pool trim release | 0 B |

The observed VRAM rise and fall equal the chunk calculation **exactly**; this is
physical reclamation, not a governor charge or pool-available counter. All 129
logit decision/final rows, 128 generated ids, final logits, prefix/restored-prefix
state, final state, prompt bytes and serialized plan match the frozen native 8k
baseline. The checkpoint's native K q8_0 / V q5_1 encoding and tokenwise trunk-only
`decode_step_h` program are unchanged. No MTP, alternate prefill, external engine,
KV-format substitution, CUDA kernel or new numerical program was introduced.

**32k initial attempt:** `died, cause: host stop`. The raw CELL journal contains
only `event=start`; there is no collector exit record, capture JSON, final state or
verdict. Last progress: `baseline prompt committed=768`. The cause is the lead's
confirmed host-stop/restart report ("container restarted; tmux sessions gone"),
**not** an inference that missing stderr proves a particular failure. The incomplete
capture is retained under `day9-vmm-32768-interrupted/`, including `failure.json`.

The native binary survived the restart: its re-observed SHA-256 is
`8bb1d8981478d349ccfdfca84885a172645dfbfe028ae56c50d2467d60eb935a`,
matching the sealed 8k build and interrupted capture; no rebuild was needed.
The 32k retry uses that exact `f3a3247a` binary and the canonical collector.
The retry completed (exit 0); all seven frozen 32k baseline surfaces and the
restored prefix match exactly. Its physical release/reacquisition is real, but the
strict raw equality check fails by one allocation granule:

| Observation | 32k native VMM result |
| --- | ---: |
| Suspended prompt / generated continuation | 32640 / 128 tokens |
| Demoted / restored planes | 32 / 32 |
| Source capacity / rounded physical capacity | 973,078,784 / 1,040,187,392 B |
| Logical D2H / pinned host bytes | 969,277,440 B |
| Whole chunks released | 905,969,664 B |
| Driver-free before / after demote / after restore | 16,493,772,800 / 17,397,645,312 / 16,493,772,800 B |
| Observed reclaim / reacquisition | **903,872,512 / 903,872,512 B** |
| Residual | **2,097,152 B** (one granule, 0.2315%) |
| Leak | None observed: restored driver-free equals original exactly |

### One bounded residual diagnostic

Before labeling, the lead made `≈` explicit: observed rise ≥ released bytes minus
one queried granule; restored free equals original exactly; demote/restore deltas
match; **a nonzero residual also requires a diagnostic classification**.
The gate records the bounded `reclaim_observed`, raw `reclaim_exact_equal`, residual
bytes/class and separate `g1_reclaim_qualified`. An unclassified residual is not PASS.

One collector cell runs 32k then 16k at 600 W / 600 W, timeout 1700 seconds.
At each size it frees a spare **reserved-but-never-mapped** VA range after demote,
re-reads free VRAM, calls `cuCtxSynchronize`, and re-reads again.

| Diagnostic observation | 32k | 16k |
| --- | ---: | ---: |
| Free before / after spare VA reservation | 16,493,772,800 / 16,493,772,800 B | 17,095,655,424 / 17,095,655,424 B |
| Spare reserved VA size (all cache planes, including dormant ones) | 1,105,199,104 B | 570,425,344 B |
| Free after demote | 17,397,645,312 B | 17,531,863,040 B |
| Free after spare VA release | 17,397,645,312 B | 17,531,863,040 B |
| Free after full-context synchronize | 17,397,645,312 B | 17,531,863,040 B |
| Free after restore | 16,493,772,800 B | 17,095,655,424 B |
| Residual / native class | **2,097,152 B / `unclassified`** | **0 B / `none`** |

The 32k residual reproduces; neither targeted operation returns the granule.
It is absent at 16k, so it is **not a constant one-granule residual across 16k/32k**.
The size dependence does not identify page-table or pinned-mapping metadata.
Residual class, verbatim: **`unclassified`**. The documented non-PASS verdict stands;
no further diagnostic or GPU run is launched. 16k is a diagnostic only, not a new
model-scale qualification context; its restored prefix is checked in-process.

All cells are N=1 RTX 5090 development correctness checks under the canonical
collector lock and 250 ms telemetry. The diagnostic, 8k VMM and interrupted 32k
cells ran at **400 W / 600 W maximum**. After restart the 32k retry runs at
**600 W / 600 W maximum**. The replay verifies each cell's actual power envelope. After the later restart
used only to retrieve these completed receipts, a fresh metadata check read
**400 W / 600 W** again; no new GPU cell was run and no earlier telemetry was rewritten.
**No timing comparison between these captures is permitted**; G1 uses byte-exact
continuation and physical VRAM deltas only. There is no performance/thermal claim.
Host tier only. Filesystem: **overlay, development, not spill speed**.

## Implementation and door

- `crates/memra-kv/src/plane.rs`: `KvPlane` privately owns either ordinary storage
  or a fixed reserved VA, per-chunk handles/mappings and the cudarc operand.
  Device VMM capability/granularity failure produces named `REFUSED:` diagnostics.
  Only whole chunks inside the demoted prefix are unmapped/released; edges stay.
- The narrowly approved unsafe cudarc `upgrade_device_ptr` constructor is isolated
  in that owner. VMM Drop consumes the operand with `CudaSlice::leak` before VMM
  teardown; it never invokes `cudaFreeAsync` for a VMM VA. Failed synchronization
  or cleanup quarantines remaining mappings rather than recycling them.
- No `DerefMut` or owning VMM `CudaSlice` escape. `KvWrite` exposes mutable borrowed
  CUDA views, with the existing pointer/stride/extent/event semantics. KV writer
  shims accept those borrowed views; CUDA kernels and argument layout are unchanged.
- `CudaTransfers` retains the complete typed owner, charges rounded physical VMM
  capacity, retires D2H before unmap and H2D before publication. `take_device`
  rejects VMM; `take_plane` returns the complete owner. The gate prevents token
  execution with a suspended layer and checks the original VA after restore.
- Door: **`--kv-allocator vmm`**, gate-only; default **`pooled`**.
  **decide-by: 2026-10-04**. No new `MEMRA_*` environment read, so no FLAGS row or
  `FLAGS-FRAGMENT.md` is required under the lead's revised instruction.

**Integration boundary:** the gate first creates an ordinary empty Cache, then
replaces its K/V planes with VMM before the first token. Its unused pooled
bootstrap reservation can remain. This establishes physical VMM chunk reclaim,
**not a reduced total process footprint**. Direct VMM allocator injection into
cache construction and serving admission are subsequent integration work.
Recurrent state/counters stay native and resident. Prefix hierarchy, serving
scheduler registration, graph/spec/PP transitions and PRO hardware are unqualified.

## Checks actually run

| Check | Result |
| --- | --- |
| Mac `cargo fmt --all -- --check` | PASS |
| Mac all-target check, memra-kv + memra-tier, offline | PASS |
| Linux-target all-target cross-check, same crates, offline | PASS; compilation only |
| Mac tests, same crates, offline/no-fail-fast | **246 passed** |
| Mac all-target scoped clippy, `-D warnings` | PASS |
| Standalone gate CLI + pure reclaim-criterion tests | **6 + 1 passed** on final source |
| `git diff --check`; `bash tools/check-flags.sh` | PASS |
| Native release gate build, f3a3247a | PASS; 40.69 s |
| Native engine/gate clippy, `-D warnings` | PASS on both f3a3247a and final diagnostic source 55f82783 |
| Final diagnostic native release build | PASS on 55f82783 |
| Native release tests, memra-kv + memra-tier | Initial run: one storage failure; full retry: **246 passed** |
| Native 8k VMM collector + frozen-baseline comparison | **ACTIVE-8K G1 PASS** |
| Native 32k VMM collector + frozen baseline | Bit-identical, real reclaim; one-granule residual unclassified — **not G1 PASS** |
| Single 32k + 16k residual diagnostic collector | Exit 0; 32k residual reproduces, 16k zero; neither VA free nor context sync releases extra bytes |
| Native workspace/all-target clippy | **Not run**; optional tmux launch was lost with host stop |
| Full GPU exactness/serving/PRO-pair battery | Not run |

The initial native CPU failure is preserved, not relabeled as success:
`day4::review_gc_cancel_removes_staging_and_corrupt_staging_is_noop`,
`crates/memra-tier/tests/storage/day4.rs:471`, `unwrap()` on **`Err(Busy)`**.
The full bounded retry passed. That storage source is unchanged in this lane;
the intermittent suite failure's cause remains unresolved and is reported to the
lead. `tests.log` and `tests-retry.log` retain both outcomes.

`verify-day9.py` replays complete CELL and raw-file integrity, source/binary/checkpoint/plan identity,
collector execution and telemetry, source destruction, physical chunk arithmetic,
VRAM deltas and every frozen continuation surface. Six mutated exact-verdict arms reject
incorrect deltas, accounting, engagement or continuation. It does **not** rerun
CUDA. `day9-manifest.json` seals the receipts, CPU check logs, design, report and verifier.
The generated 16k final-logit bytes triggered a false-positive secret-pattern match.
They are archived losslessly as `final-logits.f32le.gz`, with original decoded
length/hash verified on replay; no data was changed, no allowlist modified and no
hook skipped. Only the three **unpublished receipt-archive commits** were consolidated
to keep the rejected raw representation out of the pushed history. Measured runtime
source commits and all previously published commits were unchanged.

## Publication and hygiene

- `ef0ebe8c6`: diagnostic instrumentation, pushed.
- `c59f2b1a2`: measured diagnosis, control receipt and reclaim design, pushed.
- `f3a3247a027ca2f36bff2cf194669712f564c76b`: VMM ownership and native gate, pushed.
- `16695c83`: 8k physical-reclaim receipt, CPU/native logs and replay verifier, pushed.
- `9c32ba18`: interrupted 32k raw capture and strengthened replay, pushed.
- `909944dd`, `f8952487`, `55f82783`: bounded criterion, causal diagnostic and
  accurate incomplete-qualification status; full native source recorded in receipts.
- `fe26da36`: completed 32k retry and bounded diagnostic protocol, pushed.
- `29bac98a`: all three diagnostic/build directories, complete CELL replay and
  lossless verified 16k logits archive, pushed.

All pushes retain `core.hooksPath=tools/hooks`; no hook was skipped. Transient
GitHub failures were followed by bounded retries, not assumed successful pushes.
The final receipt push used the lead-approved loopback SOCKS forward over the existing
ControlMaster after HTTPS failed. Hooks ran normally; no credential or shared config
was changed, and the temporary forward is removed at close.
Only B's branch/worktree and native scratch were changed. The B lane remains open
for lead integration; main, other lanes, artifacts and shared checkouts are untouched.
After access restoration, B's temporary remote bundle/development files were
removed. Before deleting the preserved day-8 duplicates, all **31** preserved files
were byte-compared against their tracked copies; the only extra tracked file was
the subsequently added verdict. The native B worktree was clean afterward. Residual-diagnostic bundle/script scratch
was also removed after all three directories were synced and replayed. Local scratch
is removed at close; the B branch/worktree remains open for lead integration.

The initial day-9 implementation/qualification consumed approximately **3.3 agent-hours**
including bounded transport waits; the lead-authorized residual extension adds
approximately **0.8 agent-hours**. This is a per-session estimate against the
**10-agent-day** lane budget, not a reconstruction of prior sessions' cumulative time.
The diagnostic itself is one bounded collector invocation; no performance campaign
or further search is hidden in that estimate.
