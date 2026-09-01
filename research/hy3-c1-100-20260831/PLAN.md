# HY3 plain c1 — 100 tok/s program

Status: active. Success is not a speculative or aggregate-throughput substitute.

## Bound target

- Artifact index:
  `0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`.
- Hardware: four NVIDIA RTX PRO 6000 Blackwell Server Edition cards.
- Runtime: native Memra automatic placement, one request, `MEMRA_SERVE_SPEC=0`.
- Product-shape row: request carries no sampling fields; the server resolves the artifact's
  configured sampling defaults.
- Pass: at least 128 non-looped output tokens at more than 100 output tok/s on c1, x3
  interleaved fresh-boot repetitions; escalate to x5 under the standing spread rule.
- Correctness: finite logits, plain greedy tape/argmax gate where the numeric class is exact,
  bounded full-logit and symmetric teacher-forcing gates for any new numeric class, tools and
  reasoning surfaces, cache-on multi-turn twin, context/admission, disconnect rollback, and
  `tools/local-ci.sh --perf`.

The merged sampled baseline is 35.73 tok/s, or 27.99 ms/token. The target is below
10.00 ms/token: 17.99 ms must leave the critical path.

## Evidence already banked

The only saved Nsight capture is a K=1 speculative round window, not a plain-decode capture.
It cannot price the whole c1 target. It does contain exactly 40 target-trunk automatic-EP calls,
so the routed-expert kernels inside that shared target forward remain useful:

| target-trunk automatic-EP phase | measured max-rank kernel class |
|---|---:|
| gate/up Q8 | about 22 us/layer |
| BF16 down | about 65 us/layer |
| activation | about 4 us/layer |
| input Q8 quantization | about 4 us/layer |
| route mirror | about 2 us/layer |

The capture's 316 blocking DtoH calls belong to the MTP/draft host-visible route, not plain
automatic EP. They must not be charged to the c1 target.

Source inspection of the plain automatic-EP path establishes:

- shared expert execution was queued only after the routed-rank join;
- each layer issues the same input quantization independently on all four ranks;
- each layer performs one route-mirror launch per rank;
- BF16 down is the dominant routed-expert kernel and loops every top-k slot inside each CTA;
- output allocation, slot-row zeroing, four rank events, and root slot reduction repeat per layer;
- automatic EP keeps attention, dense layer 0, shared MLP, and the LM head on the root rather than
  composing rank-local TP attention with whole-expert EP.

## Ranked attack ledger

Predictions are hypotheses until a same-window box A/B prices them.

| priority | mechanism | predicted saving | acceptance / kill gate |
|---:|---|---:|---|
| 1 | automatic-EP PREJOIN shared-expert overlap | 1–3 ms/token | exact tape/full-logit gate; retain only with ≥1.03x c1 |
| 2 | steady-state plain phase profile: Nsight + EP event timing + CPU perf/sched/NUMA | diagnostic | must account for at least 95% of 27.99 ms before the next wide rewrite |
| 3 | BF16-down kernel campaign: NCU roofline, warp-packed rows, owner-local FMA/direct join | 2–5 ms/token | full-logit gate; candidate must beat current down by ≥1.5x in isolation and ≥1.08x c1 |
| 4 | fuse Q8 input quantization + route mirror; capture/replay the four-rank EP chain | 2–4 ms/token | current graph arm is W4A16-only; new arm must beat eager by ≥1.05x and prove engagement |
| 5 | quantize input once, distribute its Q8 row/scales rather than quantizing four times | 0.5–1.2 ms/token | bit-identical Q8 inputs; retain only with ≥1.02x c1 |
| 6 | generic TP-attention/nonexpert + whole-expert EP composition derived from `ModelPlan` | 4–8 ms/token | no architecture-name recipe; 2/3/4-card placement + full serve gates |
| 7 | served-HY3 owner statistics and co-activation placement | 0–4 ms/token | c1 objective is min critical-rank selected experts, not peer-count alone; reject maps that worsen max-rank load |
| 8 | persistent scratch/output rows and allocator diet | 0–2 ms/token | count reduction is not evidence; retain only if the wall is allocation/launch-bound |
| 9 | CPU worker affinity/NUMA locality, priority, allocator, spin/wakeup policy | 0–2 ms/token | paired runs on the target host; no global scheduler change without isolated evidence |
| 10 | cache/prefetch: slot-major readers, L2 policy, activation residency, cp.async where the
  profiler shows scoreboard stalls | 0–4 ms/token | NCU counters must name the stall; no speculative prefetch rewrite |

The Step and GLM campaigns provide two strong boundary conditions:

1. removing host boundaries must happen in whole groups; leaving a few blocking reads can make
   each survivor drain a deeper queue and lose;
2. expert placement that halves peer dispatches can still lose when it raises the maximum
   selected-expert count on the critical rank. HY3 placement optimizes max-rank work first.

## First implementation rung

This lane first extends the existing, qualified `MEMRA_SHEXP_OVERLAP` PREJOIN hook to automatic
W4A16 whole-expert EP at `t=1`. The off arm is unchanged. The same patch adds
`[nvfp4-ep-q8-phases]` diagnostics under the existing `MEMRA_STEP_TP_TIMING=1` instrument:
copy/stage, gate/up, activation, down, rank span, issue, and join.

No GPU is rented until this rung builds and its local gates pass. The first box window runs:

1. exact current-main c1 baseline and decode-only Nsight capture;
2. overlap OFF/ON exactness and x3 sampled A/B;
3. NCU on BF16 down and Q8 gate/up when counters are permitted;
4. CPU `perf stat`, `perf sched`, affinity/NUMA census, launch/memcpy/alloc counts;
5. route capture outside the timed window to price owner imbalance and co-activation.

## Live plain-c1 findings

These are development measurements, not a support-state or product claim. The request is one
vendor-default sampled stream, 128 output tokens, explicit `MEMRA_SERVE_SPEC=0` /
`MEMRA_SPEC_K=0`, on four RTX PRO 6000 Blackwell Server Edition cards.

| arm | sampled c1 tok/s | reading |
|---|---:|---|
| automatic EP, overlap off | 33.40 | true plain baseline; the older 35.73 row was K=1 speculative |
| `MEMRA_SHEXP_OVERLAP=1` | 35.84 | +7.3%, retained implementation candidate |
| overlap + CPU NUMA-node-1 binding | 35.89 | neutral; socket placement is not the wall |
| automatic TP-attention sidecars, batched B=1 path | 35.85 | sidecars loaded but serving stayed on the batched trunk |
| eager B=1 TP4, F32 mirror | 26.23 | rejected |
| eager B=1 TP4, BF16 sidecars | 26.65 | rejected |
| BF16 TP4 + device counters/no local shadow/lazy length mirrors | 27.98 | positive within TP, still rejected versus root-local |
| same arm, rank-done-fenced raw P2P gathers | 35.00 | +25.1% versus the prior TP arm; still below retained root-local |

The root-local Nsight trace accounts for the placement problem:

- root GPU busy 84.7%, peers 22.3–22.9%;
- root-only BF16 projections about 10.0 ms/token;
- root attention kernels about 2.7 ms/token;
- routed EP peers about 4.7–5.0 ms/token and already concurrent;
- the 4-byte output-token read waits about 11.2 ms, but exposes queued root work rather than
  copying cost.

The first TP4 trace proved that distribution did reduce projection compute, but its choreography
created 184,557 peer copies, 286,565 event waits, and 123,117 event creates in one 128-token
request. `cudarc::CudaStream::memcpy_dtod` creates a fresh source event for every cross-context
copy. The v2 finish already waits one persistent rank-done event, so replacing its fenced O/K/V
gathers with raw UVA copies removes a redundant event layer without changing operands or order.
After that change, a structurally comparable trace contained only 237 event creates (prompt-side
work); its wall was tracer-perturbed and is excluded from performance rows.

The next topology rung decouples the groups: TP2 attention (where the existing fused QKV/O path is
qualified and has much lower collective fanout) with EP4 whole-expert ownership. No family or
layer recipe should enter that design; the ModelPlan supplies every attention and routed-MLP
scope.

## Shared-branch gate

The rebased generic checkpoint `dd183f2f362ba06e8a808fd24f6bc8946c960f1e` ran
`tools/local-ci.sh --perf` on RTX PRO 6000 Blackwell with the required 9B NVFP4 fixture
SHA-256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.

- memra-server: 477 passed;
- kernel-check: 106 green, 13 unavailable-model skips;
- sampled accept-walk oracle: green;
- 9B NVFP4 decode-batch config B=8 and strict B=4: green;
- qwen spec-on-cache-hit normal and teeth arms: green;
- perf: qwen9b plain 203.93 tok/s, 0 fail, 0 warn.

Q35, Gemma, Ornith, and 27B cells skipped because those artifacts were not installed on the
isolated box; this receipt does not claim their gates ran. The HY3 TP+EP arm remains default off
and below the retained root-local c1 result, so this is a draft/shared substrate checkpoint, not a
`NativeTuned` claim.
