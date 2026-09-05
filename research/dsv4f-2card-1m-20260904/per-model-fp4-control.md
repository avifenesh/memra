# One-load FP4 rewrite qualification control

2026-09-05, based on `5228acc1a`. This is a dispatch/control change, not a new
numeric kernel or a serving default. The block and warp kernel bodies are
unchanged. Default remains `MEMRA_DSV4_FP4_REDUCE=block`.

## Why

The initial process-global switch required reloading the complete checkpoint
between model arms. Native DSV4 now resolves a typed mode before opening weights
and passes it explicitly at all six selected-FP4 sites: decode, batched
prefill/verification and the optional fused DSpark path. The isolated
`set_fp4_reduce_for_gate` method requires exclusive mutable model access and
drains both stage streams before the next arm. It does not mutate environment
variables or expose a request-time switch. There are no owned DSV4 CUDA graphs
to invalidate today; future graph integration must re-capture at this seam.

This avoids the multithreaded environment-mutation hazard documented by
[Rust's current set_var reference](https://doc.rust-lang.org/std/env/fn.set_var.html).
The stream drain follows the synchronization contract in
[NVIDIA's stream API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
The existing C `_sel` and `_sel_g` interfaces retain their legacy behavior;
the new `_sel_g_arm` accepts only explicit mode 0 or 1.

## Component evidence

Local RTX 5090 Laptop; correctness only under `/tmp/memra-5090.lock`. A separate
ColBERT rebuild remained resident (1,390 MiB, over 22 GiB free). It was not
paused, and no performance number is taken from this shared window.

The updated standalone gate checks both templates, the legacy launcher and both
explicit launch modes against the independently retained per-expert kernel and
CPU witnesses. For the small cases it captures the actual explicit launch and
checks the selected kernel pointer, block/grid dimensions and shared-memory
size. Equal output alone cannot prove that an exact rewrite engaged.

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-fp4-reduce-gate.cu -lcublasLt -o target/dsv4-reduce-probe/gate-per-model

# Each GPU invocation below was inside the canonical local lock.
MEMRA_DSV4_FP4_REDUCE=block target/dsv4-reduce-probe/gate-per-model
MEMRA_DSV4_FP4_REDUCE=warp target/dsv4-reduce-probe/gate-per-model --quick
target/dsv4-reduce-probe/gate-per-model --teeth
target/dsv4-reduce-probe/gate-per-model --nan-teeth
target/dsv4-reduce-probe/gate-per-model --dispatch-teeth
MEMRA_DSV4_FP4_REDUCE=warp compute-sanitizer --tool synccheck \
  --error-exitcode 10 target/dsv4-reduce-probe/gate-per-model --quick
```

Full gate: 19,962,180 bit comparisons passed. Quick reverse-env gate: 3,780
passed. All three corruption arms failed as required; synccheck reported zero
errors. CPU-only invalid-mode checks also passed for legacy and explicit C
launchers. Raw: `per-model-fp4-local-correctness.log`,
`per-model-fp4-cuda-build.log` (only the pre-existing shd linkage warning).

SHA256 values:

- CUDA TU: `b6b1072336fc2ffa31daad1210c146eef4718285a4431b4e66ec1debc4a7a6be`
- Standalone source: `84cfc7b9a5d77d35be3e5356de6b5c54f64b1196589b235b5cfad6acc1caca62`
- Standalone binary: `6e21ae8337af1c2d420c6912ac7520ac36a8f2555b76ff68459b10d3eb17c9b4`

## Next target gate, not yet executed

```sh
cargo build --release -p memra-engine --bin dsv4_fp4_reduce_gate
flock -n /tmp/memra-gpu.lock env MEMRA_DSV4_DRAFTER=dspark \
  MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native \
  MEMRA_DSV4_DENSE_ARM=fp8 target/release/dsv4_fp4_reduce_gate \
  <pinned-0731-NVFP4-model-dir> <0731-ref-fixtures.json>
```

The gate refuses incomplete settings before loading, requires the `ref`
activation contract and a real fixture long enough to exercise full width-64
transactions. It checks block/warp/block plain sampled runs at widths 1/32/64,
plus block/warp DSpark with fused dispatch both off and on. It hashes finite
logits, sampled tokens, live trunk cache, DSpark persistent rings, proposals,
confidence and round bookkeeping; timing fields are excluded. Sampling is the
pinned mint's generation configuration (temperature 1, top-p 1, no top-k), with
a fixed correctness seed. The older preview gate's ClampOnly/0.95 settings are
not reused for this 0731 contract.
The pinned [generation configuration](https://huggingface.co/tiyuvta/DeepSeek-V4-Flash-0731-NVFP4/blob/bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38/generation_config.json)
was re-read on 2026-09-05. Built model-gate binary SHA256:
`f578657e3a4855cd1c107966e2715c75668d4d502fc4f6b3641895a43546cea9`.

CPU tests cover mode resolution and exact/poisoned/empty digest behavior.
`cargo test --release -p memra-engine --lib --bin dsv4_fp4_reduce_gate -j8`
passed 390 library tests (three GPU tests ignored) and all three gate-helper
tests. `cargo clippy --release -p memra-engine --all-targets -j8 -- -D warnings`
passed. Logs: `per-model-fp4-final-cpu.log`,
`per-model-fp4-all-targets-clippy.log`. Formatting, flags and diff checks passed.
The compiled gate's actual CLI was also tested with missing settings and an
invalid FP4 mode against a nonexistent model path: both refused before opening
weights (`per-model-fp4-preflight.log`, `per-model-fp4-invalid-load.log`).

Full-model execution, served cache transparency, clean-window serving A/B,
completed 1M prefill and concurrency remain pending. Prior target-card timings
do not qualify the changed integration automatically.
