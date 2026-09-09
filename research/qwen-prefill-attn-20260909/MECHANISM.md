# Carried Qwen prime launch diet

The Qwen chunk walk now replays the existing numerical program from a CUDA graph.
At 8192, 32768 and 131070 input tokens, three interleaved fresh boots per arm give
median cold TTFT of 2.516772 -> 2.460437 s, 11.323300 -> 10.990342 s and
68.267276 -> 66.496199 s. These are 2.24%, 2.94% and 2.59% reductions.

Scope: the measured 170-SM sm_120a target, Qwen geometry with hidden width 5120,
64 dense-FFN layers, 24 Q heads, 4 KV heads and head dimension 256. Prime chunk
1024 stays unchanged. No new attention numerical class, activation precision,
model format, release or deployment is part of this change. The winner has no
runtime door. The pre-existing three-plane attention door stays at its default
OFF in both measured arms.

## Execution and lifetime

Each replay reads mutable KV, convolution, recurrent-state and tap pointers from
a device table refreshed immediately before launch. Absolute append offsets and
true KV depth are live table values. The graph never captures raw session state
handles. GDN host-side swaps and KV lengths advance once per actual replay.

The graph owns stable input, output, position and FP16 scratch storage. Reuse
checks row count, capacity, slab and dequant-workspace addresses, and tap-layer
identity. A short restored suffix or one-off boundary tail stays eager when it
cannot amortize capture. Its actual rows are never padded into another shape.
PrimeWalker still consumes and fences one existing chunk at a time; streamed
DFlash ingestion and boundary publication remain outside the graph.

Tap output is copied in bulk instead of issuing a D2D call for every row. The
eager bulk-copy helper expresses a destination offset in the device pointer,
so an offset past the first row does not violate the CUDA pitch contract.

Capture refuses host-endpoint memcpy nodes. The first development attempt was
rejected for 16 captured length uploads; length publication now happens in the
table-aware append. Capture bookkeeping and event tracking are restored on
failure. Destruction fences replay, destroys the graph before its buffers, and
trims unused graph memory. Cache declares the graph field first so cancellation
cannot free session buffers before that fence. Normal completion releases graph
storage before prefix publication and decode. The existing low-headroom guard
keeps replay away from the documented driver exhaustion failure.

## Evidence

- `receipts/oracle-comparison.json`: pinned-main and candidate tap, feature,
  position, final-logit, boundary-logit and complete boundary-state hashes agree
  at all three sizes. Interleaved 32k and 8k primes with PrimeWalker yielding
  also match their serial oracles.
- `receipts/cold-summary.json`: all 18 raw TTFT values, three distinct boots per
  arm and size. One request per cold boot, vendor-default sampling, output cap
  128, zero cached tokens and exact input counts. Boot/request/log custody is
  retained in the private companion.
- Twelve direct attention-table cells are byte-identical, including irregular
  tails, causal/noncausal execution and depth 131070. Memcheck and synccheck on
  fitting and irregular cells report zero errors.
- The same candidate's run-spec K=1..8 matches its plain target. The canonical
  prefill/decode margin guard reports flips=0, bad=0. This supplemental guard is
  distinct from the full-buffer oracle that exercises graph replay.
- Remote Rust 1.97.1 checks: Clippy all targets; engine library 467 passed,
  20 explicitly ignored; server library 652 passed, 4 explicitly ignored;
  ModelPlan/reference/runtime/tokenizer suites. A new ignored GPU regression
  test reuses one graph while changing tap destination and stride, preserving
  NaN payloads and signed zero and detecting writes to the old destination.

Measured binary SHA256: `8a5e39febe38f43456a4f92b5a6876334e4058dae8c72e54f7eed744eb48d554`.
Control: `ff0d650eeb120d8ee7d9032c273c62bd7e7a74d0ef452002881af128e8b9b251`.
Both use Rust 1.97.1 and CUDA 13.0. The control runtime is `15820202c`; later
integration changes are recorded separately. Kernel edits were rebuilt with
`RUSTC_WRAPPER=` and symbol census preserves every original CUDA entry.

Earlier diagnostic Nsight idle totals included allocation-trace tap hashing.
Without that hashing, the pilot 131k prime had 1.519 s idle, reduced to 0.267 s
by replay. This is not an 8.8 s serving launch budget. Final binary profiling,
cache/decode controls and hosted CI are recorded at delivery.

No local rig gates or CI ran. Pushes use `MEMRA_SKIP_PERF_CI=1` with local hooks
disabled under the owner's temporary restriction. Hosted CI still gates merge.
