# Rank-local expert graphs and packed-half2 gates

2026-09-07, native ST DSV4 matrix EP on two SM120 target devices.
No serving default changes. The 120 plain tok/s objective remains unmet.

Frozen candidate CLI at source checkpoint `c95742723`, factorized rather than
stacked (the rejected graph mode is subsequently removed):

- `dsv4_plain_perf_gate half2`: scalar eager GU/M1-down versus packed-half2
  dequant stores. Both rank enqueue counters must equal 86 per decode step.
- `dsv4_plain_perf_gate expert-graph`: identical scalar arithmetic, eager
  versus per-layer rank-local expert graphs. Require 86 retained entries,
  two per actual trunk layer, zero stale fallback and zero eager prepares.
- `all` measures these two modes, 256/8192 prompts, 256 output tokens, one
  warm row per arm and three ABBA repeats (six scored rows per arm).

Every row checks the sampled token tape, final logits and committed KV class
hashes. Capture cost stays in the primary wall rate; post-capture token
intervals are reported separately. Recent C4 sidecar and copy elision are OFF.
No speculative decode rows or repeated 1M capacity tests enter these cells.

Frozen binaries, built with CUDA 13.1 for sm_120a before the source checkpoint:

| binary | sha256 |
|---|---|
| plain gate | d539b3433cae1b61f6d710c21b27912461edc732e94633a334fc54a16a922fdb |
| attribution profile | 601efa7f0dcec226351de3ca5f190e43d6b8b222cad97784794b5d7e64e065e7 |
| component test executable | 0b89e7d6012cc1422b579808b287f3276e022217628735d8d6db02b8db96c1d2 |

Source baseline was `958d79cfe` plus this checkpoint's changes. The measured
CUDA source SHA is `4cb4539cf3bee7260c8f85082e3fb8cfddcb34240f191d2de4d870989eb8e6f5`;
plain gate source SHA is `480db71bb01c9ab39fcd77b91b551a14444515b23307cd8b2427cef91537fb00`.

Completed: 409 CPU engine tests pass (14 GPU tests ignored); strict release
lib/bin clippy passes; flags and registry censuses pass. Exhaustive finite
packed conversion identity passed on both target devices. Corrected full-chain
component test passes on both devices under memcheck: 12,288 H and 24,576
contribution f32 bit patterns, full-bank and partitioned results exact, actual
GU/down half2 enqueue counters engaged, zero sanitizer errors.

The first component run failed only at the fixture's original-slot merge:
it supplied grouped local IDs rather than the original selected expert IDs.
The kernel full-bank and partition H checks had passed. Correcting the fixture
to use `part_a.ids` made the same arithmetic pass on both target devices;
the original failed run is retained in the private operator receipts.

Full-model comparison completed: all 56 rows pass token/final-logit/KV
identity and actual engagement, with no loop exclusions. Six timed rows per
arm in every cell:

| candidate | prompt | baseline tok/s | candidate tok/s | wall change |
|---|---:|---:|---:|---:|
| packed half2 | 256 | 31.41518 | 31.61820 | +0.646% |
| packed half2 | 8192 | 27.33107 | 27.60817 | +1.014% |
| expert graphs | 256 | 31.51095 | 31.36738 | -0.456% |
| expert graphs | 8192 | 27.12690 | 27.16817 | +0.152% |

Every expert-graph candidate row retained 86 entries and 860 kernels, captured
86 times, replayed 21,844 times, and recorded zero stale fallback or eager
prepare. Post-capture changes remained -0.254%/+0.314%. This is a flat result:
the rank-local graph performance API/maps/dispatch are removed, preserving
only the original layer/head/stage0 capture instruments. The old 2026-09-05
0.995x MoE-graph result was chunk-32 prefill, not this plain EP experiment.

Half2 is a small, exact measured gain, not a route to 120 tok/s by itself.
All three ABBA cycles were positive in both contexts. Its process gates stay
OFF outside controlled experiments; production admission has not been run.

Raw model log SHA:
`c9dac294d29b79764d6c2e8de8fa8d72e82fdb975a7c7da562f28023baddff9c`.
Matched plain-baseline Nsight profiling is running separately. The profile
binary above contains the now-rejected graph arm as historical evidence; it
is running `baseline`, not graphs. Host enqueue counters advance on capture,
not replay. No profiled throughput is promoted as an unprofiled rate.
