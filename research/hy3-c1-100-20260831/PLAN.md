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

## 2026-09-01 Vast continuation: profile-driven boundary diet

The migrated-repo continuation rented Vast instance `49548335`, a non-production 4x RTX PRO 6000
Blackwell Workstation host at 600 W/card and $6.8056/hour including 500 GB disk. The pinned public
artifact revision and index hash above were downloaded once. Raw logs, five Nsight reports/SQLite
exports, telemetry, forced tapes, and hardware gates were copied and hash-checked before the
instance was destroyed:

`/home/avifenesh/projects/runpod-receipts/hy3-c1-100-20260901/vast-49548335/raw/`

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
