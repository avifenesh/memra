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

## 2026-09-01 exact-head profiling and migrated-repo continuation

The repository was replaced by a new migrated root while this program was active. The canonical
continuation base is `49d1d6f6594a7df52d3fed3268c5355026e3fae7`. Its HY3, TP/EP, Q8 paired,
flags, kernels, and prior receipt blobs are byte-identical to the profiled source tree; provider
history and old pull-request numbering are deliberately not part of the new repository.

Fresh decode-only Nsight Systems capture on the pinned artifact and four 600 W RTX PRO 6000
Blackwell cards, with vendor-default sampling (`temperature=0.9`, `top_p=1.0`) and speculation
off, found:

- root automatic EP4: device 0 was 91.9% kernel-busy over the captured span; peers were
  16.6-17.7%; root-only `matvec_bf16_f32acc_x4_rows` cost 8.96 ms/token;
- per token: 3,180 kernel launches, 1,205 async allocations, 1,205 async frees, and about
  7.50 ms of launch API time; the physical D2H copy cost about 0.010 ms while its host API call
  exposed queued work;
- TP2-attention + EP4 + sampled device chaining reduced the uninstrumented row to 57.43 tok/s;
  the corresponding profile left device 0 at 14.97 ms/token of summed kernels and device 1 at
  9.71 ms/token. Expert down was 26.0% of aggregate kernel time, TP QKV/O 24.7%, and attention
  about 12%;
- paired all-Q8 experts on that topology were fluent at 66.32 tok/s, but remain a new numeric
  class pending full-logit and symmetric teacher-forcing gates;
- TP4 attention was slower at 55.32 tok/s. Shared-expert split plus route prestage were flat at
  66.38 tok/s; the larger inherited-door stack regressed to 64.55 tok/s. Those compositions are
  closed for this topology.

The chosen topology is TP2 attention + EP4 whole experts + sampled device chaining. The missing
mechanism is a coarse persistent schedule, not another rank degree or flag stack. A new generic
two-rank replicated-row join provides the required 16 KiB collective: two direct peer pushes, two
reusable events, ping-pong staging, no per-call allocation, no host synchronization, and the same
`(rank0 + rank1)` operand order on both cards. Its 10,000-repetition hardware gate passed at
9.461 us/join; two joins across 80 layers price at about 1.51 ms/token before overlap. Receipt:
`receipts/tp2-replicated-row-join-20260901.txt`.

This is substrate evidence, not the 100 tok/s result. Next implementation: retain the residual on
the TP pair, use the persistent join at the attention and FFN ownership boundaries, keep EP4
experts whole, and profile each integrated rung. The pass condition at the top of this file is
unchanged.

## 2026-09-01 four-card continuation: profile-driven boundary diet

The migrated-repo continuation used a non-production 4x RTX PRO 6000 Blackwell Workstation host
at 600 W/card. The pinned public artifact revision and index hash above were downloaded once. Raw
logs, five Nsight reports/SQLite exports, telemetry, forced tapes, and hardware gates were copied
and hash-checked into the private receipt namespace before the host was destroyed. Provider,
instance, price, and placement evidence stays in the private deployment repository rather than
this public engine plan.

The strengthened primitive gate used a non-periodic fixed PRNG matrix. On the exact cards:

- persistent 16 KiB TP2 join: **9.383 us** over 10,000 joins, rank outputs bit-identical;
- 4096x1536 BF16 row-parallel partial: same argmax, max-abs **9.5e-7**, max-rel
  **5.6515e-4** (relative maximum is on a near-zero reference; the mixed bound passed);
- the later Q8-preparation fusion probe also proved q8 bytes/scales/ids/weights bit-identical, but
  that dispatch was removed after its wall result failed the retention bar.

The exact 66-token prompt and BF16-fused TP recipe were required to compare with the prior 66.32
row. A first F32-mirror diagnostic measured 32.94 tok/s and is excluded: it was the wrong TP
numeric/residency arm. Same-host vendor-default sampled c1 results on the correct recipe:

| rung | sampled tok/s | verdict |
|---|---:|---|
| reconstructed TP2-attention + EP4 + paired all-Q8 + chain8 | 52.18 | same engagement markers and 1.466 s TTFT as the prior receipt, but a slower host window |
| coarse TP2 shared expert (`MEMRA_PARALLEL_TP_SHEXP`) | 49.92 | **rejected**, -4.3%; dispatch removed |
| root shared-expert PREJOIN overlap | 57.69 | retained recipe component, +10.6% |
| overlap + native Q8 TP-attention mirrors | 64.51 | retained recipe component, +11.8% |
| plus TP-attention launch diet (DCW fused rope/append, no unused local shadow, lazy length mirrors, direct O join) | **71.93** | current measured best, +11.5% |
| root-light EP, 1 / 64 / 64 / 63 experts | 69.84 | rejected; TTFT 1.605 s |
| root-light EP, 24 / 56 / 56 / 56 experts | 71.90 | flat; TTFT 1.536 s; dispatch removed |
| fused Q8 input-quantize + route mirror | 72.22 | +0.4%, below 3% bar; dispatch removed |
| multi-device Q8 EP rank-chain graph, overlap excluded | 54.09 | **rejected**; capture/replay worked but serialized dependency latency; dispatch removed |

The coarse-shared profiles explain that negative instead of merely timing it. OFF shared work was
about 2.14 ms/token aggregate. The split reduced rank-local matrix duration, but added 50,165
kernel launches and 20,066 each of event record, wait, and copy operations over the 128-token
window. Profiled throughput moved 9.01 -> 8.22 tok/s, in the same direction as uninstrumented wall.

The retained attention diet has a non-vacuous profile receipt. Against W8+overlap, profiled
throughput moved 8.86 -> **10.34 tok/s** while D2D calls fell **50,927 -> 20,447**, H2D calls
**51,054 -> 9,694**, and kernel launches fell by **25,840**. Its fused
`qk_norm_rope_append_inc_dcw` and `fa_decode_vec_q_v3_dcw` kernels appear in the capture. GPU0
still summed 1,458.5 ms of kernels over the decode window versus 906.9 / 270.4 / 274.1 ms on the
other cards. The best uninstrumented row is 13.90 ms/token, so the 100 tok/s target still needs
about **3.90 ms/token** removed.

Correctness so far: every real-artifact rung above kept the prompt prefill/decode argmax at 40129.
The current-best max-diff is 8.912e-1, the already-declared W8 numeric class. Forcing its own
128-token sampled tape yielded 0/128 argmax disagreements and total NLL 9.4495. Forcing the older
base tape through the current best yielded 5/128 disagreements and total NLL 15.5025; the reverse
arm and paired full-logit comparison remain required before a serving/default decision. These are
engine development rows, not product or support-state claims.

The negative dispatch implementations were reverted after their receipts were banked. The branch
retains the generic, hardware-gated TP2 replicated-row join and compact BF16 K-range primitive.
Next work must remove a whole dependency group without graph serialization. The live profile says
the four-rank expert preparation/gate-up/activation/down issue stream and the remaining TP
attention boundaries are the candidates; deleting launch count alone is not a wall result.

## 2026-09-01 exact-geometry sm120 NCU closure

A standalone profiler instrument now constructs the exact automatic-EP c1 geometry without a
model load: hidden 4096, expert width 1536, top-8 slots, two contiguous owner-local experts, and
the shipped paired gate/up, q8 activation, and down kernels. It is intentionally not a serving
path. On the rig's 175 W RTX 5090 Laptop GPU, the unprofiled baseline was:

| phase | us/call |
|---|---:|
| paired gate/up | 12.070 |
| q8 activation | 1.720 |
| down | 8.638 |
| queued chain | 25.857 |

NCU replay durations are clock-perturbed and excluded from wall rows. The counters still identify
the mechanism:

- gate/up: 1,536 x 128-thread CTAs, 48 registers/thread, 1.87 waves/SM, 47.45% DRAM,
  48.32% compute, 16.49% L2 hit, 43.80% cycles with no eligible warp, and 42.84%
  long-scoreboard stalls;
- down: 2,048 x 64-thread CTAs, 40 registers/thread, 1.04 waves/SM, 32.03% DRAM,
  37.22% compute, 25.56% L2 hit, 48.34% cycles with no eligible warp, and 51.21%
  long-scoreboard stalls.

The counter-selected experiment grouped two gate/up output rows and four down rows per CTA to
raise independent weight-load ILP while retaining each output's reduction tree. It was bit-exact
against the shipped kernels: zero gate, up, or down bit mismatches. Five interleaved 3,000-call
rounds nevertheless rejected it:

| phase | shipped median us | multi-row median us | verdict |
|---|---:|---:|---|
| paired gate/up | 11.929 | 12.142 | -1.8% |
| down | 8.583 | 8.291 | +3.5% |
| full chain | 26.110 | 25.427 | +2.7% |

This misses the 1.5x isolated-kernel retention bar by a wide margin. The candidate kernels and
wrappers were removed; only the reusable probe and raw NCU reports remain. Reports:

`/home/avifenesh/projects/runpod-receipts/hy3-c1-100-20260901/local-sm120-probe/`

The exact PRO 6000 profile independently bounds this arc. On root device 0, paired gate/up plus
down totaled 223.809 ms in the captured request, while the adjacent root costs were scalar FA
178.442 ms, shared BF16 dual-SiLU 173.599 ms, Q8 QKV 161.035 ms, Q8 output projection
127.764 ms, and vector FA 70.289 ms. Expert-row ILP cannot supply the remaining 3.90 ms/token.
The next paid cell profiles and attacks the attention/shared-dense dependency groups; another
expert-row layout is not justified by these counters.

### Shared-expert BF16 closure

The same probe covers HY3's root shared expert at its exact per-layer shapes: gate/up
4096->1536 and down 1536->4096. An existing unwired interleaved gate/up kernel was bit-exact,
but five interleaved 2,000-call rounds were flat: both the shipped and interleaved medians were
8.350 us. `MEMRA_DOWN_X4=1`, already a documented off-by-default adjacent arm, regressed the down
median from 7.570 to 8.356 us (-9.4%). The full shipped chain was 16.282 us; interleaved gate/up
with the normal down was 16.164 us (+0.7%, noise and far below retention).

NCU explains why another BF16 schedule is not the commercial lever:

- dual gate/up already reaches 80.52% DRAM throughput; 84.21% long-scoreboard stalls reflect a
  streamed 24 MiB weight pair, not missing arithmetic parallelism. The interleaved twin did not
  move its wall;
- down reaches 62.93% DRAM with 4.16 waves/SM. Its remaining stalls are 47.94%
  long-scoreboard and 22.55% barrier; serial four-row grouping makes both worse in wall time.

The unwired interleaved wrapper was removed and `DOWN_X4` remains off. The retained next rung is
the existing q8 mirror numeric class (`MEMRA_W8_HYBRID`) for shared down, dense layer 0, and the
LM head, combined with attention profiling. Halving streamed bytes is supported by adjacent
receipts; rescheduling the same BF16 bytes is now falsified for HY3's exact shape.

## 2026-09-02 sampled endpoint and TP predictor closure

The exact artifact then ran through the native OpenAI-compatible server on four RTX PRO 6000
Blackwell Server Edition cards. Every row below used the same 66-token real prompt, no sampling
parameters, no prompt-cache hit, 128 streamed output tokens, one warmup, and three repetitions.
The number shown is median post-first-token throughput; TTFT and total-throughput fields remain in
the private raw receipts. These are development measurements, not a support-state or product
claim.

| server arm | sampled tok/s | verdict |
|---|---:|---|
| TP2 attention + EP4 + all-Q8, W8 hybrid off | 62.88 | valid plain control |
| TP2 attention + EP4 + all-Q8, W8 hybrid on | **64.75** | fastest serving row; +3.0% |
| exact EP4 plain | 32.36 | exact numeric control |
| exact EP4 predictor K=3 | 26.00 | rejected; 24.9-31.3% acceptance |
| EP4 Q8 gate/up plain | 41.94 | matched predictor control |
| EP4 Q8 gate/up predictor K=1 | 46.30 | +10.4%, but below TP2 plain |
| TP2 predictor K=1, host-staged cache repair | 14.15 | correct, structurally too slow |
| TP2 predictor K=1, native peer-device cache repair | 13.99 | correct, no improvement |

Before the repair, TP2 prediction failed closed on the next decode because the exact verify walk
advanced only the canonical full-width K/V cache; the rank-local TP caches retained their old
length and bytes. The repair copies only the accepted quantized row slices into each rank cache,
then publishes the matching length mirror. A 16-token warmup and all three 128-token sampled rows
completed without cache divergence, CUDA failure, or Xid. The peer-device binary was hash-bound
as `d252b61c93a383dfb33dfb7f02bb8d96cbdf3e0d57f64eed7f3f1c01bfdb0e8e`.

The performance result is the important profile: replacing host staging with direct native P2P
did not move the wall (14.15 -> 13.99 tok/s), despite 55.0-65.3% K=1 acceptance in the three
peer-device rows. PCIe byte volume is therefore not the dominant regression. The per-layer repair
and synchronization topology is. Predictor remains off for this placement. Any follow-up must
publish verified rows during the verify walk itself or batch the whole 80-layer repair into a
constant-number dependency group; another per-layer copy variant is rejected.

The serving result also corrects the earlier `71.93 tok/s` interpretation: that number came from
the generation harness with `MEMRA_ASYNC_CHAIN=8`, a mechanism the server request path did not
engage. The current comparable server baseline is **64.75 tok/s**, leaving 5.44 ms/token to reach
100 tok/s. The next plain-path cell profiles server-side dependency groups and either integrates a
coarse persistent chain into the server or rejects it on measured wall time; expert-row and
shared-BF16 reschedules are already closed by the counters above.

Post-rebase `tools/local-ci.sh --perf` on exact head `6acc5a6d17063760127bc29f6265b7bf75e7791f`
passed the workspace clippy gate, 550 server tests, flags and drafter-wiring checks, 107 available
kernel cells, the sampled-spec distribution oracle, both installed decode-batch numeric classes,
graph warmup/canary stress, plain serve/cache accounting, 64 concurrent streams, and both normal
and teeth arms of the installed Qwen spec-on-cache-hit gate. Q35, Gemma, and 27B arms named their
missing local artifacts and skipped; this receipt makes no claim for them.

The sole red was the cross-day `qwen9b-plain-short` timing tripwire at 131.29 tok/s. Its required
same-window settle used the immediate pre-fix parent `a6c9aaad` as A and this rebased head as B,
interleaved A/B five times under one exclusive GPU lock. Medians were 132.90 and 132.89 tok/s
respectively (**-0.008%**). The TP cache repair does not regress the plain Qwen cell; the historic
median comparison moved with machine state.

The next implementation reuses the existing eager device chain under `MEMRA_SERVE_BATCH=0`, with
no speculative, grammar, or penalty program. The opt-in arm emits each returned id through the
ordinary stop/stream/accounting battery, retains the chain boundary token for the next tick, and
falls back to the original one-token step if the model declines. Every eligible legacy session
keeps this path when round-robin width changes, avoiding a mid-stream device-draw to stale-host-RNG
transition. A stop, EOS, or disconnect inside the submitted chunk taints the overshot cache, which
makes retirement drop it instead of publishing a hidden suffix through whole-session affinity
reuse. Default remains 0 until the exact sampled endpoint runs OFF/ON with stop, disconnect,
TTFT/ITL, c2 arrival isolation, and three 128-token rows.

### Dense server-chain qualification

The first local 9B run was excluded because its server log carried no
`[serve-async-chain] engaged` line: the plan was `Generic`, and the original engine door admitted
only `SlidingGatedMoe`, so both arms silently used the fallback. The gate now fails on missing
engagement, and engine eligibility is derived from the actual execution shape: standard residual,
no Gemma/E4B, HyperConnections or PP cut, and only Full/Linear mixers. MLA and KDA stay refused.

On the local RTX 5090 Laptop GPU, exact dense artifact
`52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`, binary
`74a2b0b86754b60624ba928a5943de7422d0a86c8f998b24f9ce24838e230dc2`, one 61-token real
prompt, no sampling fields, no cache hit, and five fresh boots per arm in A/B, B/A order:

| arm | sampled post-first tok/s | TTFT | verdict |
|---|---:|---:|---|
| chain off | 120.40 | 36.4 ms | control; 120.10-121.29 spread |
| chain K=8 | **131.90** | 38.0 ms | **+9.55%**; 131.70-132.05 spread; engaged 5/5 |

The explicit-greedy OFF/ON outputs were byte-identical. Seed 7 and seed 1234 each reproduced
byte-for-byte between solo and c2 arrival. A newline stop returned `finish_reason=stop` and the log
proved the overshot cache was refused from reuse; a client disconnect billed two generated tokens,
left the worker alive, and a following eight-token request completed. No CUDA error, panic, or Xid
appeared. This qualifies the server mechanism on that dense model and hardware only. The global
default remains off; the HY3 PRO 6000 sampled cell still decides whether the mechanism advances
the 64.75 tok/s MoE baseline toward 100 tok/s.

The second dense plan confirms compatibility but rejects a family-wide speed conclusion. Q38 27B,
artifact `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`, used the same
binary, request shape, and five-boot A/B, B/A gate. OFF was 44.70 tok/s and K=8 was 45.98 tok/s
(+2.86%), below the 3% retention bar; TTFT was 107.7 vs 108.3 ms. Engagement fired 5/5, greedy
identity and both seeded c2 isolation rows passed, and stop-taint plus disconnect recovery stayed
green. The chain remains a qualified opt-in mechanism whose performance decision is per model and
hardware, not a generic dense default.

## 2026-09-02 TP verified-prefix dispatch collapse

A current primary-source sweep did not justify an all-expert dense GEMM for this c1 target.
TensorRT-LLM's [Blackwell DENSEGEMM report](https://github.com/NVIDIA/TensorRT-LLM/blob/181f726d10f713836ad3d19df4016c2d3f5ab631/docs/source/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.md)
puts its measured sweet spot at 64-208 input tokens and states that the grouped path wins below
that range; HY3 c1 presents one token. Its [min-latency report](https://github.com/NVIDIA/TensorRT-LLM/blob/181f726d10f713836ad3d19df4016c2d3f5ab631/docs/source/blogs/tech_blog/blog01_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.md)
instead identifies MTP and whole-boundary fusion as the large levers. Memra's selected-slot
paired expert kernels already remove the batch-1 permutation shape that "sparse experts as GEMMs"
targets. The actionable adjacent mechanism is therefore the measured TP predictor repair wall,
not multiplying all 192 experts for one token.

The existing TP predictor repair issued K copy, V copy, and length publication independently for
every layer and rank. A new native primitive batches uniform layers through one pointer table and
one kernel per rank. Each block gathers one or more full-width canonical rows with their source
stride into the rank-local contiguous K/V rows, then publishes the layer's device length after the
copy. Uniform runtime, target length, row count, packed geometry, physical contiguity, and native
P2P are all required; any mismatch takes the existing per-layer fallback before a batch launch.

The local exact-geometry gate models a full-accept K=1 round: 80 layers, two accepted rows, rank
K/V bytes 544/384, canonical strides 1088/768, and 200 timed rounds per arm. Five A/B, B/A rows
were byte-exact with every one of the 80 length mirrors equal to 130:

| repair boundary | median us/round | verdict |
|---|---:|---|
| per-layer K + V + len dispatches | 629.04 | control |
| one batched kernel | **1.86** | **338x**, exact |

This is an isolated dispatch-boundary receipt, not an end-to-end predictor claim. The live PRO 6000
cell must show `[tp-kv-verify-batch] engaged`, preserve sampled correctness and acceptance, and beat
the 13.99 tok/s peer-repair row before the predictor path can advance.
