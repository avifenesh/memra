# WP-C queued cells — none are GPU execution receipts

Day-1 CPU cells run via the local harness (RESULTS.md/raw logs). GPU cells below are
**pending access, immutable artifacts and adapter implementation**. No GPU claimed here.
No new numeric program, quantization study, CPU expert assignment, Engram adapter or
DSv4 Flash 0731 use. Existing Hy3 and Qwen4Exp PLE only.

Dates are check-ins: C2 rows 2026-09-25; C1 experts 2026-09-26. Lead must rebook when
access changes. 5090 fitting fixture work is development only; full models and integrated
pre-merge qualification require a designated **non-serving 2× RTX PRO 6000** pair.

## Required preflight for every real run

1. Lead grants one exclusive whole-box campaign; no second-card scored tenant. No live
   serving instance used. Record `nvidia-smi topo -m`, concurrent compute processes,
   CUDA/toolkit/driver, RAM/pin limits, NVMe mount/device class and storage free capacity.
2. Pin qualified Hy3 artifact and PLE checkpoint plus byte manifests, templates/prompts,
   plan/numeric-class hashes and exact binary SHA256. Hy3 mixed/pruned assignment and
   original router IDs/scales must be real; no new scored arm or BF16 fallback.
3. Stage byte-identical sources `/data` → local NVMe `/scratch`; record both manifests.
4. Build **before** taking GPU lock (avoid sccache inheriting the flock). Execute binaries
   under canonical lock. Capture all raw stdout+stderr and preserve the command exit code;
   parser reads saved logs afterward. Failures retain stderr and concurrent-GPU snapshot.
5. Never use a changed numerical configuration as the no-spill oracle. Cache-off can
   select a different qmatvec numerical path today; the adapter gate must pin the same
   existing compute program and compare original checkpoint bytes, logits and token stream.

Shell parameters below (`HY3_ARTIFACT`, `PLE_ARTIFACT`, `PROMPTS`, `OUT`) are caller-supplied
public artifact/receipt paths, **not new MEMRA flags**. Commands are templates until these
paths and immutable manifests are supplied. These are not authorized to execute today.

| Cell | Command / status | Time + memory budget | Rig, lock |
|---|---|---|---|
| C1-baseline | `flock /tmp/memra-gpu.lock target/release/run-gen "$HY3_ARTIFACT" --prompt "Compute 17 times 23 and explain briefly."`; then `flock /tmp/memra-gpu.lock target/release/run-spec "$HY3_ARTIFACT"` with a pinned prompt and supported K=1..8 manifest. Existing commands; baseline only, no new service engagement. | 60 min booking; governor ≤ measured free VRAM minus unchanged model/scratch reserve; no full-bank expansion. Full artifact manifest required. | Non-serving PRO pair; fitting selected-record copies only on 5090 under `/tmp/memra-5090.lock`. |
| C1-layout-cache | Proposed **not-yet-created** test target: `flock /tmp/memra-gpu.lock target/release/deps/banked_residency_gpu-<hash> --nocapture`. Adapter must add this GPU test before cell can run. | 90 min correctness; zero/small/full *cache* regimes, bounded staging independent of cache; full cache only when exact source fits, otherwise explicit capacity refusal + selected-record full fixture. | PRO pair `/tmp/memra-gpu.lock`; fitting development 5090 `/tmp/memra-5090.lock`. |
| C2-history-baseline | `flock /tmp/memra-gpu.lock target/release/qwen4exp_gpu_gate "$OUT/ple-tiny.tsv"` (Cargo auto-discovered underscore binary). Existing gate invokes `gate_ple_ngram_cache` at source line 1860; also exercises GPU fixtures. | 30 min; tiny fixture only, not NVMe row-service exactness. | 5090 fitting: substitute `/tmp/memra-5090.lock`; then PRO pair `/tmp/memra-gpu.lock`. |
| C2-model-baseline | `flock /tmp/memra-gpu.lock target/release/qwen4exp_real_gate "$PLE_ARTIFACT" "$OUT" --label spill-c-ple-baseline --prompts "$PROMPTS" --mtp --verify-bit-gate 24 --rewind-bit-gate 24 --spec-gate 256`. Existing parsed flags; check manifest applicability first. | 90 min; full model's existing native configuration, host load peak accounted separately from service slots. | PRO pair `/tmp/memra-gpu.lock`. |
| C2-host-nvme | Proposed **not-yet-created** `flock /tmp/memra-gpu.lock target/release/deps/table_rows_gpu-<hash> --nocapture`. Host ordinary → pinned/UVA exactness → bounded NVMe. Baseline binaries do not exercise these adapters. | 90 min; initial hot rows ≤64 MiB, slots 1/4/16 ×64 KiB, output ≤4 MiB/request; B may lower caps. Oversized logical table streamed, not allocated wholesale. | PRO pair `/tmp/memra-gpu.lock`; fitting 5090 under its canonical lock. |
| C3-row-pipeline | Future A storage-bench plus C adapter replay; exact command frozen after A interface publication, **not executable yet**. Sparse/packed 264 B ×48 plus real PLE F32/BF16 rows; granularity 512 B/4 KiB/16 KiB. | 120 min; ≥5 AB +5 BA pairs plus 30 min stationary pressure; ≤64 MiB hot cache, 1/4/16 ×64 KiB staging initially. One bounded pipeline with B contention fixtures. | PRO pair `/tmp/memra-gpu.lock`; fitting generic-default cells on 5090 before any generic promotion. |
| C3-expert-pipeline | Future Hy3 bank replay, exact command after GPU target added; paired same-binary baseline/candidate reads with compute program fixed. | 120 min; ≥5 AB +5 BA, cold/warm cache and forced-miss traces; mandatory scratch before hot cache. Feasible hot regimes only, never promise fully cold streaming is interactive. | PRO pair `/tmp/memra-gpu.lock`. |

The standard engine/kernel and server integrated battery is lead-owned; this queue does
not waive it. Actual GPU test registration is frozen with lead module wiring; never put
`<hash>` globs into a shell without resolving the one exact build artifact first.

## Assertions / non-vacuity

**C1:** Q2_K/Q3_K/NVFP4 real layouts, original masked IDs, unequal exact lengths and macro
scales; force nonzero misses/evictions; zero/small/full cache hashes equal checkpoint bytes
before compute; uniform-only adapters refuse mixed records; no renumbered pruned ID.
Poison source/length/scale or attempt masked dispatch and demand a loud error. No Q2_K
kernel promotion or new CPU arithmetic. Compare state/logits/tokens, not two cache copies.

**C2:** real PLE row encodings unchanged; duplicates/reordering/straddles, EOS/history,
ragged chunks, request reuse and speculative rewind. Tiny hot cache smaller than batch,
repeated eviction, all-host versus actual NVMe misses, short reads/failed reads/cancel and
restart. Delayed ready/release fences and tiny pools must never publish/reuse early.
History is adapter-owned; CPU row tests do not prove n-gram algorithm correctness.

**C3:** storage → host → H2D/UVA → unchanged compute → request completion, not storage-only
GB/s. Preserve raw byte hashes, exit status, thermal/cache regime, clocks/power/temperature/
VRAM/PCIe at 250 ms, host/SSD counters, useful/physical bytes, duplicate hits, amplification,
cache hits/misses, prefetch useful/wasted bytes, queue depth, staged bytes, H2D waits,
**p99 demand miss wait** and TTFT/E2E/TPOT/ITL p50/p95/p99 plus request/token throughput.
Percentiles need enough observations; small samples are descriptive only. Separate spill
performance from model-quality study. No default claim until balanced runs and exactness
pass; rejected/flat doors removed if introduced.

## Day-2 migration update — 2026-09-19

The standalone research harness is retired; its day-1 raw receipts remain. Run
`cargo test -p memra-tier --offline --test bank` in the workspace. Native engine
exports/dependency wiring remain lead-owned. Details: `HOSTEXPS-ADAPTER.md`.

| Cell | Day-2 execution | Remaining blocking evidence |
|---|---|---|
| C1-contract | Frozen `BankedResidency` cancellation schedule directly path-imported, not copied; original masks, three independent epochs, duplicate leases, zero/small/full host cache, late release, retained rejected-publication resources and common-budget pressure tested. | Actual A transfer owner/ReadyView, native SLRU and GPU byte/consumer-fence integration. |
| C1-hostexps | Typed HostExps metadata/bridge, native qtype-code pins, authoritative per-record offset/len/row_bytes, split offset-zero, macro/block-scale presence/checksums tested. New UniformLease-only boundary has a compile-fail red. | Bridge is unexported; tests compile it against an API-shaped CPU fixture, not CUDA HostExps. Dispatch sites and planned edits are enumerated in HOSTEXPS-ADAPTER.md. No runtime changes authorized here. |
| C2-contract | Frozen RowService ordered/duplicate schedule runs against BoundedRowService. Multi-plane records, chunked slots smaller than batch, failed completion cardinality, partial-release Busy retry, bounds/corruption/cancel/retire tested. | Async worker, real NVMe/pinned/UVA/device publication and model-scale persistent hot cache. |
| C2-ngram | Recorded **synthetic** six-step trace reproduced by pinned native full/cached CPU ID functions, then replayed through RowService for F32 and BF16 expansion bit identity. The source-body drift test runs in the same target. | Existing full-geometry `gate_ple_ngram_cache` plus real artifact rewind/spec/projection identity and raw native CPU/GPU row-byte capture. |
| C3-policy | Portable 512 B requested granularity /4 KiB slot selected from arithmetic table; 512 B/4 KiB/16 KiB sparse and packed cases rerun. No env flag, no hardware/default promotion. | Actual backend alignment, physical SSD counters, latency/IOPS and complete storage→compute A/B. A native backend must raise granularity when its contract requires it. |
| Inventory | One bounded read-only SSH attempt, exit 255: `Connection closed by UNKNOWN port 65535`. No remote inventory or GPU run succeeded. | Lead supplies reachable designated non-serving rig and immutable Hy3/PLE artifacts. No retry beyond the two-attempt cap; only one attempted this turn. |

All previously queued C1/C2/C3 GPU/model/performance cells remain **pending**;
none is promoted by CPU conformance. The host implementation explicitly rejects
requested device/pinned allocation and never claims `consumer_fenced=true`.
Current stage/gather reads are synchronous; only the ticket/publication/retirement
lifecycle is asynchronous. A's actual asynchronous byte transport remains required.

Native follow-up order: (1) lead engine dependency/export wiring, (2) loader source
and manifest registration with full metadata accounting, (3) A ObjectStore + owner
TransferEngine/ReadyView and B production governor, (4) C1 exact native compute
adapter and C2 unchanged PLE expansion/projection, (5) all queued real gates.
No changed hardware default can land from the portable 512 B policy alone.


## Day-3 CPU integration and rig handoff

Day-2 synchronous stage/gather statements above are historical. Current source
moves all reads into explicit `BankService::progress`, one bounded gather chunk
per call. `stage` and `gather` only validate/reserve/enqueue; `publish` returns
NotReady and never pumps I/O. ObjectReader issues A CpuTransfers tickets against
ObjectStore, checks every accepted outcome, drops the host consumer, requires
`retired`, then acknowledges. Host-ready remains explicitly not GPU-ready.

| Cell | Day-3 result | Still required |
|---|---|---|
| C1/C2 A integration | Forced expert and PLE row misses through A ExtentStore + CpuTransfers. Duplicate/order/scales exact; partial second-chunk error refuses entire batch; cancel between chunks issues no subsequent read. | Native worker/owner/pinned/DMA/SLRU bindings; mid-DMA cancellation is not simulated by a between-chunk CPU cancel. |
| Shared B budget | Same `tier::Governor` instance charges banks, A store leases and pool/queue. Optional ticket exhaustion preserves one mandatory slot; saturated optional NVMe capacity refuses Demand but admits MandatoryActive. | Server queues and model-scale pressure with actual allocations. |
| C1/C2 prediction | RouterTopKHint and NgramLookaheadHint use separate domains, fixed candidate metadata and bounded item/byte/scan caps. No heat from prefetch. | Native router/history hint callsites and usefulness/waste/latency traces. |
| C2 trace | Full local text/gzip search found summaries, no replayable real row-ID trace. New SHA-bound six-step synthetic last-chunk policy table, see TRACE-AUDIT.md. | Capture real F32/BF16 PLE rows/history/rollback and projection identity. |
| C1 runtime patch | Unapplied, rustfmt-parsed and `git apply --check` passed. **Partial mask-guard patch only, NO-GO**; full BankedResidency/UniformLease conversion is not implemented. | PATCH-REVIEW.md enumerates every missing native boundary and exact before/after commands. |
| Rig runner | bash syntax plus 5090/PRO stubs, repeated runs, injected exit-23 failure and missing non-serving confirmation refusal all exercised. | Actual fresh Linux execution and compiled native bank/row test targets. |

### Reproducers

- CPU suite plus cross-target syntax/type checks:
  `python3 research/spill-c-20260919/verify-day3.py`.
- Safe orchestration-only check:
  `bash research/spill-c-20260919/rig-cells-c.sh --dry-run --host-label stub`.
- On an assigned **non-serving** 5090, with nvcc/toolchain ready:
  `bash research/spill-c-20260919/rig-cells-c.sh --non-serving-confirmed --host-label dev5090`.
  Builds outside `/tmp/memra-5090.lock`, acquires that lock nonblocking, refuses
  an occupied GPU, runs tiny PLE first. Hy3 is explicitly PRO-pair-blocked without
  a separate fitting receipt. Missing native targets are **BLOCKED, real exit 3**,
  not an empty-test PASS.
- PRO pair baseline only:
  `bash research/spill-c-20260919/rig-cells-c.sh --non-serving-confirmed --rig pro-pair --host-label pro-pair --hy3-artifact "$HY3_ARTIFACT" --hy3-manifest "$HY3_MANIFEST"`.
  Uses `/tmp/memra-gpu.lock`, validates the caller's standard byte manifest,
  runs native run-gen/run-spec on the unchanged artifact. `kernel-check` accepts
  GGUF only; a directory records that missing kernel-artifact cell explicitly.

Each run creates a new `raw/<public-host-alias>-<utc>/` directory (atomic suffix
on collisions), tees logs before recording JSONL and preserves command failures.
No install/download, artifact conversion, scored timing, serving change, third
lock or skip switch. Do not read baseline success as bank/row service engagement.
Before performance/default decisions, extend the runner with the actual native
engagement targets, original byte/logit/token goldens, balanced AB/BA N>=5,
250 ms telemetry and serving metrics from the protocol above.

## Day-4 queue update — CPU prerequisites, still no GPU receipt

| Cell | CPU status | Rig acceptance still required |
|---|---|---|
| C1-SLRU-policy | 2013 recorded synthetic default-SLRU decisions and queue orders matched; charged host BankedResidency/old-allocation alias test. | Native intrusive/fixed-slot adaptation, pending ready wait, LFU/frozen compatibility and model traces. |
| C1/C2-source-install | Typed BankSource verifies exact expected/supplied source sets, source layouts/generations/lengths; ExtentStore/CpuTransfers expert and PLE tests use installer. | Model-load source registration, original-file/per-extent indexing, metadata/validation-I/O budget, no substituted weights/scales. |
| C1/C2-ownership | READYVIEW-OWNERSHIP.md maps owner thread, actual copy→consumer wait, take-once and final-consumer fence. | CUDA implementation, unknown-drain quarantine, graph/multiple aliases, complete-byte/logit/token tests. |
| C1-guard-v2 | Unapplied patch adds CPU-tested exact source-bound check after cardinality/mask checks; patch context/syntax only. | Native compile and all pre/post cells; full conversion remains NO-GO. |

Runner now begins with source-contract CPU tests and v2 patch-context check
outside the lock, then builds, then **PLE tiny first**. Hy3 remains PRO-pair-only
unless separately hash-bound fitting evidence is granted. Existing explicit
BLOCKED records and real exit 3 remain; no stubs become GPU evidence. Dates above
remain check-ins, no gates advanced by CI compilation or these CPU tests.
