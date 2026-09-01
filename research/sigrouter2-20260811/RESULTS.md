# cx-sigrouter2: zero-DtoH resident sigmoid dispatch (2026-08-11)

Lane `lane/cx-sigrouter2`. Measured checkout: `1808220e` (the runtime last changed in
`82b75811`; later commits contain harness fixes, documentation, and receipts). Target rig: 2x
NVIDIA RTX PRO 6000 Blackwell Server Edition. Every box1 GPU command ran under an exclusive
`/tmp/memra-gpu.lock` hold.

## Verdict: ADOPT increment 2

Keep the Step-3.7 local-resident sigmoid dispatch arm enabled by default. It eliminates the
per-layer decode routing readback and synchronization, is exact across the mandatory synthetic,
served-logit, generation, specification, and fresh-boot gates, and wins all five paired box1
comparisons at both requested loads:

- c1: 101.281657 vs 84.520490 tok/s, **+19.8309%** by arm medians, 5/5 paired wins;
- c8: 169.004876 vs 165.505592 tok/s, **+2.1143%**, 5/5 paired wins.

This is a deliberately narrow default. It engages only for Step-3.7 with uniform q8 expert
layouts, local resident slabs, at most top-8, no per-expert macros, no observation mode, and the
matching CUDA owner device. Mixed layouts, spill/cache staging, remote slabs, macro-scaled
experts, non-Step sigmoid architectures, and explicit rollback settings retain their established
metadata-aware paths. The result therefore claims zero router DtoH for the eligible Step decode
path, not for every sigmoid-router configuration.

## Runtime contract

The device sigmoid selector already returned device-resident selected ids and weights. The new
arm consumes those buffers directly in the established resident q8 expert programs:

- unclamped layers use the fused gate/up/SwiGLU rows kernel followed by slot-ordered weighted
  down/FMA;
- Step's two clamped layers keep separate gate/up, the authoritative clamp expression, down, and
  slot-ordered scatter;
- the shared expert retains the same decode/verify kernel forms as the established path.

No new CUDA arithmetic kernel was introduced. The change is dispatch plumbing plus fail-closed
eligibility.

Three binding review items are now explicit contracts:

1. Host and device launchers call one `active_count >= n_used` validator before slicing or CUDA
   work. The `n_used - 1` cell proves identical rejection text:
   `sigmoid router requires active_count >= n_used: active_count=7, n_used=8`.
2. Model load runs a once-per-process, 24-input host-`expf` byte probe. Polarity (see the
   post-review amendment below, and `hybrid.rs` load path): `MEMRA_SIG_ROUTER=0` selected +
   probe mismatch = boot FATAL (the host-oracle arm is the libm-dependent path); device default +
   probe mismatch = WARN only (the device arm uses vendored scalar `expf` and never calls host
   `expf` at serve time, so host-libm drift cannot corrupt it; the WARN flags host-side
   replay/comparison tooling as invalid on that host). `MEMRA_SIG_ROUTER=0` remains the explicit
   host-oracle rollback.
3. `MEMRA_SIG_ROUTER_LOGIT_TRACE` captures one real t=1 row per layer as raw f32 bits, including
   correction bias, original-id active mask, scale, and normalization. The required replay cell
   rejects malformed or duplicate records and requires identical ids and weight bits.

The byte probe was chosen instead of a second vendored host exponential because the contract is
with the scalar function in the deployed executable. A second implementation could drift away
from that real oracle; the boot probe detects libc/compiler/runtime movement at the boundary that
matters and fails closed. This is also consistent with NVIDIA's CUDA 13.2 documentation that
[host and device floating-point results may differ across platforms](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html).

## Exactness evidence

| Gate | Result | Receipt |
|---|---|---|
| Local RTX 5090 fast kernel-check | `ALL GREEN (77 cells, 22 model-backed skips)`; 24-case host-expf and undersubscribed-active cells pass | `raw/local-kernel-check-fast.log` |
| Local RTX 5090 required-manifest full kernel-check | `ALL GREEN (101 cells, 0 skipped)`; 42 served rows/42 layers, zero id or weight-bit mismatches | `raw/local-kernel-check-manifest.log` |
| Box1 required-manifest kernel-check | `ALL GREEN (83 cells, 20 unavailable-model skips)`; required three cells present; 68-case router corpus has zero mismatches and 0 ULP max weight gap | `raw/box1-kernel-check.log` |
| Box1 Step-3.7 `run-gen` PP-2 | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH; device arm observed on all 42 MoE layers including clamped layers 43/44 | `raw/box1-run-gen-capture.log` |
| Box1 Step-3.7 `run-spec` PP-2 | K=1..8 self-consistency PASS, 8/8 | `raw/box1-run-spec.log` |
| Box1 fresh server boots | 10/10 exact `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` | `raw/box1-golden/driver.log` and per-boot QoS JSON |

The served capture is 42 records from an actual box1 `run-gen`, not reconstructed decimal
inputs. Local and box1 replay both report `records=42 layers=42 idx_mismatch=0
weight_bit_mismatch=0`.

### Standing local-CI settle

The unmodified local `tools/local-ci.sh --perf` correctness stage was green, including its
model-backed kernel, generation, specification, graph, serving, and acceptance gates. Its later
historical perf comparison flagged the unrelated `26b-spec-d1736` cell at acceptance `0.646`
against a rolling `0.880` reference (and 240.14 tok/s against the older performance-profile
history). That alert is real, but it is not a regression introduced by this lane.

To settle causality, the exact lane base (`30418923`) and candidate were run under one local GPU
lock in an alternating N=5 same-window matrix, using the same model, draft, prompt ids, and rank
artifact. Every run in both arms produced the same 47 rounds, 127 drafted tokens, 82 accepted
tokens, `tok/round=2.74`, and acceptance `0.646`. Median throughput was 242.03 tok/s at the base
and 241.91 tok/s for the candidate (`-0.0496%`). The standing reference therefore needs separate
baseline/profile reconciliation; it does not change this Step-only adoption verdict. Full raw
outputs, hashes, thermal snapshots, and the machine reduction are in `raw/local-26b-ab/`.

After the final engine-scope commit, all release binaries were rebuilt and the hook-supported
fresh `tools/local-ci.sh --perf-quick` battery ran under the local GPU lock. The complete
correctness stage is green (including c=64 serving stress and the served-acceptance gate), and
the four 31B cells completed with `0 fail, 0 warn`: plain short 41.21 tok/s, plain depth 38.24
tok/s, spec short 106.08 tok/s at acceptance 0.798, and spec depth 98.26 tok/s at acceptance
0.817. The full output is `raw/local-ci-perf-quick-final.log`; its four rows are retained in the
append-only `research/tune-data/perf-ci.jsonl`.

## Performance evidence

Metric: `decode_window_tok_s`, 512 generated tokens per request. N=5 per arm and concurrency,
under one exclusive lock hold; fresh server and one warmup per arm; default/increment-1 order
alternated by repetition; c1/c8 order reversed between arms; 250 ms NVML samples throughout each
arm. Observed GPU temperature was 31-46 C. All 90 measured requests reached the requested length
with zero errors. Nsight ran afterward and is excluded from the medians.

The increment-1 control is `MEMRA_MOE_DEV=0` with `MEMRA_SIG_ROUTER` unset. This preserves the
increment-1 device sigmoid selector and pinned `[sel,w]` readback; it does not compare against the
older full-logit host rollback.

| Load | Device-resident samples (tok/s) | Increment-1 samples (tok/s) | Arm medians | Delta | Paired wins |
|---|---|---|---|---:|---:|
| c1 | 101.2817, 101.0208, 101.3095, 101.1737, 101.3453 | 84.5205, 84.5787, 84.4873, 84.5266, 84.4713 | 101.2817 vs 84.5205 | **+19.8309%** | 5/5 |
| c8 | 169.1187, 168.7247, 169.0049, 168.9698, 169.1249 | 165.5056, 164.7904, 165.3902, 165.6953, 165.5833 | 169.0049 vs 165.5056 | **+2.1143%** | 5/5 |

Paired-delta medians are +19.8309% at c1 (range +19.4399% to +19.9761%) and +2.1831%
at c8 (range +1.9762% to +2.3875%). `raw/box1-perf/summary.json` is the authoritative
reduction; `points.jsonl` retains every request and summary row.

## Transfer and synchronization receipt

Both Nsight arms use one generated token, the same prompt and PP-2 placement, and the same binary.
The exported CUPTI tables and CUDA API summary show:

| Whole-run observation | Device-resident default | Increment 1 | Change |
|---|---:|---:|---:|
| Decode-sized router DtoH, 32 B | 0 | 18,144 | **100% removed** |
| Prefill-sized DtoH, 2,912 B | 138 | 420 | 282 removed |
| `cuMemcpyDtoHAsync_v2`, all sizes | 359 | 18,785 | **98.09% fewer** |
| `cuStreamSynchronize`, all causes | 298 | 9,511 | **96.87% fewer** |
| Unrelated 515,584-byte DtoH | 221 | 221 | unchanged |

The synchronization delta is exactly 9,213 calls: 9,072 eliminated decode route invocations
plus 141 eliminated prefill route invocations. The whole `run-gen` profile also executes oracle
passes outside the local-resident PP arm. By exact count matching with the dispatch trace, the
remaining 138 prefill-sized copies are 69 route readbacks from three stage-1 reference passes
(23 MoE layers each); this attribution is an inference from the counts, not a sampled stack trace.
It does not weaken the direct observation that every decode-sized router readback disappeared.

Primary `.nsys-rep` files and SQLite exports remain on box1 because those files can embed the
process environment and the repository's pre-push guard forbids committing them. Their hashes,
sizes, extracted memcpy JSON, and CUDA API/memory summaries are retained in
`raw/box1-perf/`; see `raw/README.md`.

## SOL movement

The original `research/solgap-20260811/sol-model.py` now accepts the frozen performance summary as
an optional c1/c8 override; its default historical output is unchanged. Re-running the same
weight-streaming ceiling model gives:

| Load | Increment-1 %SOL | Increment-2 %SOL | Movement |
|---|---:|---:|---:|
| c1 | 28.812% | 34.526% | **+5.714 percentage points** |
| c8 | 27.168% | 27.742% | **+0.574 percentage points** |

The exact invocation and output are in `raw/sol-model.log`. The unchanged c2/c4 historical
anchors are not part of this lane's claim.

## Changed surface and boundaries

- `crates/memra-engine/src/sigrouter_contract.rs`: active-count, host-expf, and served-replay
  contracts.
- `crates/memra-engine/src/lib.rs`: launcher check and device-returning selector API.
- `crates/memra-engine/src/hybrid.rs`: load-time expf gate and active-count authority.
- `crates/memra-engine/src/hybrid_forward.rs`: eligible resident device consumer, trace capture,
  and exact shared-expert forms.
- `crates/memra-engine/src/bin/kernel_check.rs`: mandatory contract and replay cells.
- `docs/FLAGS.md`: default, rollback, and diagnostic behavior.

No published perf board was changed: these are controlled lane measurements rather than a
board-moving merge. No merge, tag, or release was performed from this isolated lane. `cargo fmt`
was not run.

## Post-review amendments (orchestrator, 2026-08-11)

Two external-review fixes applied after the lane's box1 battery (behavior-preserving on the
scored path; local `cargo test -p memra-engine` 61 pass):

1. **Probe polarity inverted.** `verify_host_expf` previously hard-failed the DEVICE default on
   host-libm mismatch while leaving the `=0` HOST-oracle arm bootable unprobed — backwards: the
   device arm never calls host expf at serve time (vendored scalar, rig-deterministic); the host
   oracle is the libm-dependent arm. Now: `=0` selected + probe mismatch = boot FATAL; device
   default + mismatch = WARN (host-side replay/comparison cells flagged invalid on that host).
2. **Eligibility predicate de-hosted.** `sigmoid_resident_dev_eligible` re-read five MEMRA_MOE_*
   env vars per MoE layer per decode step (~300+ environ scans/step). Observation mode is now a
   process-wide `OnceLock<bool>`, matching the file's existing gate idiom. The scored box1
   medians predate this hoist and therefore UNDERSTATE the shipped arm by whatever the env-scan
   tax was; no re-measurement claim is made here.
