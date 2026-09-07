# Rank-local expert graphs and packed-half2 gates

2026-09-07, native ST DSV4 matrix EP on two SM120 target devices.
No serving default changes. The 120 plain tok/s objective remains unmet.

The candidates are factorized, not stacked:

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

Full-model performance is in progress, not yet a qualification or speed claim.
The attribution profile now accepts explicit `baseline`, `half2`, and
`expert-graph` arms. Its 32..64 sampled-step range is for Nsight attribution,
not throughput. Host enqueue counters run during capture, not every replay;
half2 and graph arms remain disjoint to avoid that accounting ambiguity.
