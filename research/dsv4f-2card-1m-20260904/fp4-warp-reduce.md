# Exact selected-FP4 reduction candidate

2026-09-05. Base `820b9f73eac6383470cd7f2adf29daa4dbda2566` plus this
lane's working patch; `MEMRA_DSV4_FP4_REDUCE=block` remains the default.

The selected NVFP4/MXFP4 projection already computes four output columns per
128-thread block. The old reduction uses nine block barriers per column
(before staging, after staging, and seven tree levels). The candidate stages
all four column planes once and performs the identical tree in warp 0:
`(p[t]+p[t+64])+(p[t+32]+p[t+96])`, then shuffle-down levels 16, 8, 4, 2, 1.
All 32 lanes participate in every shuffle; only the original tree's active
lanes add. Thread/group ownership and within-group multiplication are unchanged.
The extra reduction shared storage is 2 KiB instead of 512 bytes.

This is not FP4 requantization, a format conversion, or reassociated warp-first
summing. NVIDIA's current language-extension reference describes the shuffle
participation requirement:
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

## Local correctness only

GPU: RTX 5090 Laptop, no serving process, serialized with
`flock -n /tmp/memra-5090.lock`. No local timing was collected or used.

Build:

```sh
/usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  --expt-relaxed-constexpr -fmad=false -Xcompiler=-ffp-contract=off \
  tools/dsv4-fp4-reduce-gate.cu -lcublasLt -o target/dsv4-reduce-probe/gate
```

Build emitted the pre-existing `shd` linkage warning at `dsv4_gpu.cu:3268`.
Candidate source SHA256:
`5c23161f62af934c521d2a1ee6e0c9dd8d0b273b9fcccfaf672269a7fe6ff617`.
Gate source SHA256:
`97c38650ade332698c25ac156978d2f42e72942292136aa7fffce70c15a93c6c`.
Binary SHA256:
`5bd2a93e120cc005a5fc0a894fabab0d0266b67bdf19594c2351d34b2b1cd2c7`.

`env -u MEMRA_DSV4_FP4_REDUCE target/dsv4-reduce-probe/gate` and
`MEMRA_DSV4_FP4_REDUCE=warp target/dsv4-reduce-probe/gate` each passed
11,977,308 bit comparisons over ten cells, three projections and three
activation-index modes. Each compares both templates and the actual C launcher
with the independent per-expert kernel. CPU witnesses anchor first/last outputs;
finite checks and tail canaries guard against poisoned/vacuous comparisons.
Logs: `fp4-warp-reduce-local-v2.log`, `fp4-warp-reduce-local-switch-warp.log`.
Earlier two-template-only gate is retained in `fp4-warp-reduce-local.log`.

Additional invocations in the same lock:

```sh
MEMRA_DSV4_FP4_REDUCE=block target/dsv4-reduce-probe/gate --quick
MEMRA_DSV4_FP4_REDUCE= target/dsv4-reduce-probe/gate --quick
MEMRA_DSV4_FP4_REDUCE=invalid target/dsv4-reduce-probe/gate --invalid-switch
target/dsv4-reduce-probe/gate --teeth
target/dsv4-reduce-probe/gate --nan-teeth
MEMRA_DSV4_FP4_REDUCE=warp compute-sanitizer --tool memcheck \
  --error-exitcode 10 target/dsv4-reduce-probe/gate --quick
MEMRA_DSV4_FP4_REDUCE=warp compute-sanitizer --tool synccheck \
  --error-exitcode 10 target/dsv4-reduce-probe/gate --quick
```

Block/empty quick modes passed; both launchers rejected the invalid value with
40021. The corruption arms failed with `bit mismatch` and `non-finite comparison
operand` respectively. Both sanitizer runs reported zero errors (logs beside
this file). Quick cases use K=8192 with odd N=7 and both weight scale encodings,
including multiple group iterations per thread.

## RTX PRO 6000 component gate

Same binary SHA256 as above on two RTX PRO 6000 Blackwell Server Edition cards,
driver 610.43.02. Each passed 11,977,308 bit comparisons. Invocation per device:
`CUDA_VISIBLE_DEVICES=<0 or 1> MEMRA_DSV4_FP4_REDUCE=warp ./gate --perf`,
serialized in one `/tmp/memra-gpu.lock` campaign. Each arm has five CUDA-event
samples of 20 launches, alternating AB/BA order. These are synthetic component
inputs, not a model request. Raw: `fp4-warp-reduce-pro-gpu0.log`,
`fp4-warp-reduce-pro-gpu1.log`; 250 ms telemetry:
`fp4-warp-reduce-pro-telemetry.csv`.

NVFP4 measurements (microseconds per projection):

| GPU | N, K, activation rows, top-k | block median | warp median | time saved | block/warp spread (range / median) |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | 2048, 4096, 1, 6 | 34.605 | 30.115 | 12.98% | 2.26% / 1.85% |
| 1 | 2048, 4096, 1, 6 | 34.515 | 29.954 | 13.21% | 1.14% / 1.33% |
| 0 | 4096, 2048, 6, 6 | 202.464 | 156.306 | 22.80% | 1.96% / 1.58% |
| 1 | 4096, 2048, 6, 6 | 202.824 | 156.682 | 22.75% | 1.89% / 1.50% |

Every pair favored warp in these four cells. The short campaign began cold
(27/29 C); sampled SM clocks rose from idle into the 2025–2370 MHz range. These
are interleaved short-run medians, not heat-soaked serving measurements. The
250 ms samples are too coarse to assign a power or clock limit to individual
launches. No power-cap change was made or is justified by these data.

## Pending whole-model decisions

Microkernel wins alone do not establish TTFT/TPS, sampled cache identity or DSpark acceptance. Those
need the exact pinned model and served path. No power-limit attribution follows
from a card's configured limit or a utilization sample.

## Source regression gate

`tools/local-ci.sh --perf` exited 0 on this source patch. Raw log:
`local-ci-warp-reduce.log`. Clippy, server/engine CPU suites, 107 kernel-check
cells (11 explicitly skipped), graph/session/served-cache correctness controls
and the available Qwen9B perf control passed. Models absent on the local rig
were explicitly skipped; this does not replace DSV4 target-card qualification.
`cargo fmt --all -- --check`, `git diff --check` and `tools/check-flags.sh` also
passed. The built serving binary SHA256 is
`a3237e858600968087bda170ae14fb778c8cef210a2a2444966f06c88bcc29b6`.

The new provider denied Nsight Compute counters with `ERR_NVGPUCTRPERM` on
device 0. Its component timing/identity results remain valid, but no stall or
power-cause classification was obtained from that attempt.
