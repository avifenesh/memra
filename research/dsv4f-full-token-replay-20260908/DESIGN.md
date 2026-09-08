# Full-token segmented replay: review checkpoint

Status: full-token runtime integration and the one-load model gate are implemented
in source. Real compressor/attention control and device-sampler components pass
on the target development pair. Full-model graph coverage, changing-token/refusal
qualification and the 20-row performance experiment have **not run**. This is a
review checkpoint, not production admission or a speed result.

Base: `2d0271fb3b6021c2c9f1b3b36d2598614a10284f`. Existing issue #4 ownership
and the older window-only/EP/compressor probe guards are preserved.

## Runtime contract

The request owns four retained graphs: forward and commit/sample on each rank.
Forward captures embedding, all 43 complete trunk layers and both collectives
per layer. Both forward graphs launch before either rank is drained. The host
then reads **both** sticky AR refusal words. Only `[0, 0]` allows segment two:
persistent-ring commit on each rank, head and the existing device sampler on the
head rank. Both streams drain before the sampled token is returned. The caller
checks token/EOS before starting another step.

One 24-byte pinned upload per rank/token contains token/position, the f64
position-keyed uniform bits, and a live rank/layer refusal control word. Captured
nodes use stable device pointers. RoPE positions, compressor append/store slots,
C4/C128 emission cadence, selector counts, attention slot bounds and commit
`slot_rows` are live. There is no per-node host update loop or recapture path.

The compressor uses a uniform whole-block emission predicate at kernel entry.
An inactive block returns **before** any barrier or write; active blocks retain
the exact existing pooling, norm, RoPE, quantization, and reduction program.
The selector computes the same live power-of-two sort width. Attention retains
the exact live loop bounds and reduction order; it does not substitute padded
reduction bounds. No new numeric class is introduced.

The initially tested CUDA-IF body implementation failed synccheck in the composed
compressor, despite normal and memcheck equality. Removing/moving its diagnostic
marker, prewarming, and a newer sanitizer did not clear the composed failure.
That implementation is unqualified and its runtime helpers were removed. The
uniform-predicate implementation is the separately gated equivalent. No sanitizer
suppression or numerical-program change was used to obtain its pass. Private
companion receipts retain the failing controls and diagnostic source snapshot.

## Ownership and failure handling

`ReplayPair` owns both ranks' graph handles, pinned/device inputs, counters and
sampler. `MatrixStep` declares it before the captured workspace allocations.
Capture is ended/aborted before drains, and both ranks are drained before graph
or buffer release, including an exception between rank submissions.

If either rank's completion cannot be proved by a successful drain, the runtime
attempts both drains, reports their errors and aborts the process before Rust
destructors run. Logging and returning would let the enclosing workspace, caches
and model pointers be freed while peer access might remain outstanding. Drop
also fails stop if capture abort cannot be established. This is restricted to
the unrecoverable completion failure; ordinary AR refusal with successful drains
still rolls back and returns a quarantined request. Rust subprocess tests inject
failed completion on either rank, during an error return and destructor unwind,
and assert SIGABRT with no captured-storage destructor release. The separate
Rust CUDA lifecycle test covers partial forward/commit submission, capture-body
unwind and the capture-end/instantiate error boundary using the actual owners.

Capture records GPU work but runs Rust bookkeeping. Host compressor high-water
marks are restored after initial capture and on pre-execution capture failures.
Each replay refreshes the host checkpoint marks once/token. On an AR refusal,
the existing zero-commit rollback restores both pending planes/high-water marks;
neither ring is committed and the request remains quarantined. Prefix restoration
rejects failed/open requests and cannot act as a retry bypass.

Refusal injection is read by the existing AR kernel at the captured attention
rank/layer from the live control word. It runs before actual barriers, whose real
timeout errors therefore take precedence. It is not a captured stack-backed host
copy or a frozen per-node error value. Device AR epochs remain model-global and
the existing walk mutex protects their sequence.

The arming API is explicitly unsafe and gate-only: the borrowed model must remain
at a stable address, with immutable weight allocations/kernel configuration, until
the armed state is dropped. The gate owns that lifetime. No serving route invokes
this API. Admission refuses unsupported topology, cache residency, stream override,
shape, numeric program, sampling configuration, DSpark and split-K. Both arms use
attention TP2/expert-ID EP, f32x/RefFp8Round, device sampler and small-kernel diet.
The admitted request has capacity 512..1024 and replay positions below 512.

## Owned implementation

| File under `crates/memra-engine/` | Role |
| --- | --- |
| `src/dsv4_gpu.rs` | Strict request arming, complete token capture/replay, host refusal boundary, capture metadata restoration, enqueue-only successful commit, stable-address prefix row reset |
| `src/dsv4_graph.rs` | Capture-only graph lifetime, paired submission/drains, stable input/sampler ownership, device replay counters, structural census and DOT export |
| `cu/dsv4_gpu.cu`, `src/dsv4_ffi.rs` | Live scalar/offset consumers, uniform compressor predicates and copies, capture lifecycle, controls, counters and census |
| `cu/tp_ar.cu`, `src/tp_ar.rs`, `src/dsv4_ep.rs` | Existing AR with live diagnostic word, capture context binding and authoritative device-epoch readback |
| `cu/dsv4_sampler.cu`, `src/dsv4_sampler.rs` | Existing device sampler with pointer-fed uniform and separate enqueue/readback |
| `src/dsv4_full_token_replay_gate.rs` and sampled gate entry | Changing-token/state/epoch/refusal checks, then one bounded 20-row ABBA |

## Components and model gate

- `tools/dsv4-full-token-control-gate.cu`: paired lifecycle, live controls, actual
  AR widths/epochs, injected and actual missing-peer refusal, failure cleanup,
  quarantine and live rank/layer refusal after capture. The original five setup
  and poison writes are explicitly ordered on the owning nonblocking stream.
- `tools/dsv4-replay-live-kernel-gate.cu`: real pool/norm/RoPE/quantization across
  513 positions for C4, C128 and the rotated indexer; real selector/attention
  intermediates at changing and decreasing bounds. Exact eager comparison on both
  GPUs. The guarded implementation passed normal, memcheck and synccheck.
- `tools/dsv4-replay-sampler-gate.cu`: fixed logits, eight changing f64 uniforms,
  257 and 129280 vocabulary entries, both GPUs. Exact token/canary/prefix/block-sum
  comparison with the existing sampler, eight distinct tokens in each case.

Build/checks run remotely only, with arch `120a`, a dedicated target directory,
and at most two build jobs. `MEMRA_SKIP_PERF_CI=1` is used for push; local git
hooks are disabled because they run rig gates. Hosted CI remains a merge gate.

The sampled gate's `--full-token-replay` arm uses the existing pinned source and
model loader. It first primes 256 tokens and compares every subsequent changing
sample, final logits, both cache/hidden planes and all 72 AR epochs/rank for 256
steps. It requires exactly one capture per segment and a forward census of 86 AR
nodes, one embedding and 86 HC posts per rank, with no unsupported node type.

Six faults are armed **after capture**: each rank at positions 259/383/511 and
layers 0/21/42. Each must leave the cache/position unchanged, increase forward
counters only, and quarantine ordinary retry and prefix-reset retry.

Only then does the same load run `AAAAA BBBBB BBBBB AAAAA`: five rows/block,
256 outputs/row. A new scored graph state includes its first capture in the first
B row's measured wall, then retains the same four graphs for all later B rows.
Rows reset primed bytes into existing allocations outside timing. Both arms use
the same consume/commit/draw order: a common initial carry draw outside timing,
then 256 forward/refusal/commit/head/sample/readback steps inside timing, including
the final next draw. Output/caches/hidden/epochs are validated after each row.
EOS-shortened or looped rows fail eligibility. No profile/hash work is timed.

Every row emits wall time, token/state hashes, live-control hash, replay/capture
counts and eligibility. DOT graphs are exported for SHA binding. On correctness
failure the experiment stops. Flat/negative full-model replay removes the added
door/dispatch/kernels/gate cells in this lane; a win returns to root for full
qualification. No merge, serving claim, DSpark/server/host-C4/1M ladder or automatic
scope expansion is authorized by component results.

CUDA references: [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
Target runtime/headers and failure controls are recorded in the private ops lane;
architecture labels alone are not support evidence.
