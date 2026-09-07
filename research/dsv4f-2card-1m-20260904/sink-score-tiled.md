# Tiled exact sink-attention scores

2026-09-06 UTC. The matrix/EP profile attributes 20.8% of summed kernel time
to sink-attention scores. This rewrite targets that measured operation while
retaining the original attention program. No default is promoted.

## Implementation and contract

The scalar scorer assigns one key and all 64 heads to a block. The new native
kernel stages 8 query heads and 32 selected keys, with a padded transposed KV
tile. Each head/key pair still accumulates dimensions 0..511 in the original
order and multiplies by the same scale. The normal DSV4 build retains
`-fmad=false`; the separate FMA research build is not qualified by these
receipts. Negative indices still emit negative infinity. Softmax, sink handling,
and output accumulation use the unchanged kernels and score layout.

The tile uses 84096 bytes of dynamic shared memory. Its explicit initializer
checks the device limit and sets the attribute before graph capture. Launches
do not lazily mutate attributes. This follows the current CUDA guidance on
[shared-memory layout](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
and [kernel attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html);
no third-party kernel/runtime dependency was added.

`MEMRA_DSV4_SINK_SCORE=scalar|tiled` defaults to scalar. Tiled mode requires
device f32x and 64 heads of width 512; unsupported path/geometry and unknown
or non-Unicode values refuse. `set_sink_score_for_gate` is exclusive, drains
both stages and initializes attributes before switching. Single-query and
batched paths use the new scorer when selected; successful launches are counted.
Persistent graph users must key/rebuild on the arm. Rollback is scalar/unset.

## Component evidence

`tools/dsv4-sink-score-tiled-gate.cu` SHA256 binary:
`75f2992ff440e3d34d4c4a44911c4796f128fbda6465ac5fe7a316b866bbd8bb`.
It passes separately on both PRO cards, including memcheck and synccheck with
zero errors. Coverage is 25 cases x six changing input/mask/index patterns:
queries 1/6/32 and slots 1/31/32/33/128/640/8320, wide-query tails 128/512,
zero/signed-zero scale, random/duplicated/padded/all-masked indices, zero
queries and mixed magnitudes. Scores, exponentials, denominators, outputs and
independent tail guards match bitwise. Single-query cases also match the old
plain-attention entry. A corrupted valid index must be observable. Each case
replays a captured attention graph with six changed inputs/indices; this does
not prove live-scalar full-model graphs.

Warm component ABBA measurements have ten rows per arm and include the entire
three-kernel attention operation, not only scores. On GPU0, scalar/tiled time
ratios are 1.386x / 1.369x for one query at 640/8320 slots, 1.720x / 1.820x
for six queries, and 1.871x / 1.867x for 32 queries. GPU1 has the same direction
(1.367x–1.873x across those cells). These are warm synthetic component results,
not complete-model or serving speedups. Raw target logs are retained in the
companion private lane as `sink-score-tiled-gpu{0,1}-{plain,memcheck,synccheck}.log`.

## Model integration gate

`dsv4_sink_score_gate` performs scalar/tiled/scalar walks on one matrix/EP
model load, comparing complete logits/cache/rings, sampled plain/DSpark output,
warm restoration/continuation, active C4 and confidence/round data. It checks
engagement explicitly and includes width 512. The local engine suite passes
405 tests with ten ignored; the new mode parser test is included. Build and
remaining exact-head gates are recorded alongside this file.

The server library has 602 passing tests; strict engine/server library/bin
clippy, formatting and the runtime-flag census pass. The model gate binary is
`c8cf054f9946daa538f195f91d01cee4926d627659cb64fefd5751c89fb6c364`;
the conditional long/profile binary is
`8410c70353cd8971c4d998c80ad6bf715480b25a45f5f46c2bebbe5602580188`.

The full-model scalar/tiled/scalar gate passed at 04:56:51 UTC: real-source
prompt lengths 1/32/160/1025/4097, widths through 512, complete logits/cache/rings,
41 sampled output tokens per walk, warm suffixes, active C4, confidence and
round data are identical. Tiled call counters are positive and scalar counters
zero. This qualifies the tested storage rewrite, not the matrix program's
broader quality or a serving default.

## Matched phase profile

The subsequent profile passed at 05:05:19 UTC with 16896 real-source input
tokens and the same explicit 16384..16640 capture span, width 32 and matrix/EP
program as the prior control. The observed kernel extent is 1.582341 s versus
1.812260 s, 12.69% shorter. Score kernels consume 182963478 ns versus
410234874 ns, 55.40% less summed time. Both-device overlap is 23.93% of the
kernel-busy union. Expert GEMM still accounts for 42.0% of summed kernel time;
dense FP8 is 27.9%, tiled scores 10.5%, and attention output 4.6%.

The complete prefill took 98.834067 s (170.953 input tokens/s); 16 sampled
plain/spec outputs match in three speculative rounds. These are single-run
mechanism observations, not balanced performance admission. Restore-inclusive
decode timings are not output TPS. Independent CSV audit finds only the owned
PID in 3871 samples, including six teardown name gaps. Nsight retains generic
CUDA/NVTX completeness warnings; no explicit dropped-event count is present.
Raw report SHA256 is
`8b28a27047466aa19fb566976df293c42cb4a8448ef21b0799ac8119caf18749`;
SQLite SHA256 is
`afcdc0da7dacbb42991c2c6b406d7cced0c8e9b7b44fc7118dd1cfcb3e119234`.
Raw artifacts and the read-only interval audit are banked in the companion lane.

Balanced end-to-end measurements, actual 1M serving, direct host-C4 working-set
admission, concurrency/fairness, full live graphs, broader quality and complete
TP2 remain required. The default remains scalar.
