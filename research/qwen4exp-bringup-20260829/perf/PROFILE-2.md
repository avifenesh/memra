# qwen4_exp decode PROFILE-2 — round 2: bf16 trunk, workspace + graphs, sel v2, gate micro (2026-08-29)

Same box, same artifact, same prompts as PROFILE-0/1 (read those first). memra
qwen4exp-bringup-20260829; final binary sha256 `e6c57048db587b34…` (perf8). Greedy is the
instrument. Every change: interleaved ×5 same-run A/B (fresh state + probe prefill + 4
warmup steps per arm, non-overlapping ranges), tiny gate + real-checkpoint gate re-run
green, receipts banked before the next item.

## Headline

| | ms/token (warm) | tok/s | vs PROFILE-1 | vs PROFILE-0 |
|---|---|---|---|---|
| PROFILE-1 (grouped sel + fused gates) | 28.8 | 34.67 | — | 2.72× |
| **PROFILE-2 (this round)** | **17.2** | **58.18** | **1.68×** | **4.56×** |

Owner target ~90 tok/s = 11.1 ms/token: **1.55× still to find** (single card). The TP2
projection below covers the remaining gap and what it cannot cover.

Host-side launches per token fell **2,932 → 531** (84 graph replays + 447 eager launches,
`profile4-nsys-decode8_cuda_api_sum.csv`); pooled allocations 2,234 → 39, memsets ~0.

## Per-change interleaved A/B (each row = its own box battery, both arms in ONE run)

| item | change | A/B (mean of 5 means) | win | rep-0 chains | receipt |
|---|---|---|---|---|---|
| 1 | **bf16 trunk residency** — `qmatvec_bf16w_f32` (batched, f32 accumulate, exact-widening + in_f%8 guards) for gdn/qsa/lm_head mats; read-gate down/up STACKED to one batched launch per projection; inject bf16 twin | 28.84 → 22.58 | **1.28×** | fork @12 (accumulation class) | `ab-trunk-nvfp4.tsv` |
| 2a | **step workspace** — named-slot take/put (`StepPool`), address-stable transients | 22.59 → 22.29 | 1.013× | identical (−1) | `ab-ws-nvfp4.tsv` |
| 2b | **decode CUDA graphs** — per-layer GDN interior + MoE-tail + exit graphs, no-warmup capture after one eager warm step | 22.24 → 21.96 | 1.013× | identical (−1) | `ab-graph-nvfp4.tsv` |
| 3 | **sel matvec v2** — uint4 code loads + 2 rows/warp on the modelopt bank | 21.95 → 20.37 | **1.08×** | identical (−1) | `ab-selv2-nvfp4.tsv` |
| 4 | **hcmicro bundle** — batched plane norms (ptr table), two-stage inject, slab write gate, shared-expert bf16 | 20.38 → 17.07 | **1.19×** | fork @0 (accumulation class) | `ab-hcmicro-nvfp4.tsv` |

The chain telescopes: each battery's off-arm reproduces the previous battery's on-arm to
≤0.03 ms (28.84 ↔ PROFILE-1's 28.84; 22.58 ↔ 22.59; 21.96 ↔ 21.95; 20.37 ↔ 20.38).

Items 2a/2b bought little wall time on their own — the step was already GPU-execution
bound (PROFILE-1 nsys: 24.7 ms GPU busy under a 28.8 ms wall) — but 2a is the address
stability 2b requires, and 2b is what makes the later small-kernel batching nearly free
to launch. Graph replay is bit-identical to the ws-eager path by construction (same
kernels, same baked addresses, same order), and the A/B chains confirm it.

## What the graphs could and could not take

MoE routing is a HOST twin by lane doctrine (reference top-k + renorm floor + tie rule),
so a whole-step graph is **structurally impossible**: the step has one host boundary per
MoE layer (router logits dtoh). The shipped structure: 35 GDN-interior graphs (attn gate →
GDN → write → mlp gate), 48 MoE-tail graphs (sel matvecs → shared → write), 1 exit graph
(exit mixer + lm_head); QSA and PLE interiors stay eager (indexer host twin, mask h2d,
n-gram host hashing live there). 84 replays/token, captured lazily on the second decode
step of a state (the first graph-eligible step runs eager to warm/park every workspace
slot — an allocation inside a capture region becomes a graph mem node). The decode-timing
warm mean now excludes the first TWO steps (step 1 carries the captures).

## The perf7 incident (and the oracle it bought)

The hcmicro bundle first shipped tiny-gate-green and **broke the real model from layer 0**
(argmax 0/10, greedy divergence at step 0 — `run-perf8-hcmicro-nvfp4.log` is the FIXED
run; the broken battery was perf7). Root cause: the two-stage inject's stage-1 kernels
stored **warp 0's partial (`acc`) instead of the block sum (`v`)** in the final write. At
tiny geometry a chunk spans 2 elements — every element lands in warp 0 and the bug is
arithmetically invisible; at real geometry a chunk spans 640 elements across 8 warps and
7/8 of the dot product vanished. Caught in minutes on the rig by the new permanent tiny
gate arm 0c, `gate_hc_micro_kernels`: the micro kernels vs the classic composition at the
ARTIFACT's read-gate shape (streams 4, hidden 2560, t 10) — worst rel 1.652e-6 after the
fix. Lesson (card-keyed-defaults class): a stream-batched kernel needs an oracle at the
real stream/width geometry, not only the tiny plan's.

## Correctness after all five changes (perf8, re-run vs the banked baseline)

| gate | banked baseline | PROFILE-2 re-run |
|---|---|---|
| logits argmax vs transformers | 10/10 | **10/10** |
| greedy 64-token chains (prompts 0-3) | none, 8, none, 48 | **none, 8, none, 48** |
| layer0 / layer47 max_abs | 7.258e-3 / 1.014e0 | 7.269e-3 / 1.022e0 |
| cross-arm KL row 9 (worst row) | 0.293 | 0.17303 (top-1 true) |

Envelope moves are the documented accumulation class (bf16 residency is value-exact — the
checkpoint IS bf16; only reduction trees changed). Tiny gate: seven arms green including
the three kernel oracles (`gpu-eager/tiny-fixture-gate.tsv`).

VRAM: post-load 89.9 GiB of 95.6 (81.9 baseline + ~7.5 GiB bf16 trunk twins + ~0.5 GiB
shared-expert twins); the f32 originals stay resident as A/B twins and guarded fallbacks —
drop them if a future lane needs the headroom.

## Residual: per-section profile after (64 warm steps, eager under the profiler)

Profiled wall 23.47, attributed 21.59, unprofiled 17.19 ms/token — shares are the signal
(`profile4-nvfp4-after-micro.tsv`; graphs disable themselves under the section profiler,
so this is the ws-eager composition of the same kernels).

| section | ms/tok | % attr | physics |
|---|---|---|---|
| hyper.read | 3.69 | 17.1 | 96 × (batched norm + 2 batched bf16 GEMVs + reduce/epilogue + 2-stage inject); the down/up bytes (96 × 13 MB bf16) floor it at ~0.9 ms — the rest is small-kernel latency |
| moe.sel_grouped | 3.46 | 16.0 | v2 at ~340-420 GB/s on 1.18 GB/token — still 3-4× under the card; sel v3 (4 rows/warp, wider blocks) is a real lever |
| gdn.proj | 2.87 | 13.3 | 3.0 GB bf16/token ≈ 1.9 ms bandwidth floor — near it |
| gdn.conv_scan | 1.84 | 8.5 | `gdn_scan_naive_f32` runs 32 blocks at t=1 — latency-bound; a decode-step twin is unwritten |
| gdn.norm_gate_out | 1.60 | 7.4 | one-block norms + elementwise + out-proj (bf16); batching candidate |
| moe.router | 1.37 | 6.4 | 48 f32 Lt GEMVs — left f32 DELIBERATELY: router logits feed the routing decision, and a reduction-tree change there can flip near-tie selections |
| moe.shared | 1.33 | 6.2 | bf16 now; sff is small so this is mostly launch/latency |
| qsa.sdpa | 1.10 | 5.1 | dense masked naive SDPA — the gather/compact QSA kernel is a named deferred lane |
| hyper.write | 1.01 | 4.7 | one slab launch per gate; profiled-sync inflation (nsys shows ~0.1) |
| qsa.proj | 0.94 | 4.4 | bf16; near floor |
| lm_head | 0.85 | 3.9 | 1.27 GB bf16 ≈ bandwidth floor |
| rest (qsa.gate_wo/idx, ple, entry/exit, dtoh) | ~1.5 | 7.0 | flat |

## TP2 projection (NOT implemented — projection only)

The box's second 96 GB card is idle. Slices that split cleanly across 2 cards (row/head/
expert parallel, Megatron seams: all-reduce after the attn out-proj and after the MoE
down):

| slice (real ≈ attributed × 17.19/21.59) | 1-card ms | TP2 ms | saved |
|---|---|---|---|
| sel_grouped (experts 5/5) | 2.76 | 1.38 | 1.38 |
| gdn.proj (row split) | 2.29 | 1.15 | 1.14 |
| hyper.read GEMV portion (~60%, stacked rows split) | 1.76 | 0.88 | 0.88 |
| gdn.conv_scan + norm_gate_out (value heads 16/16) | 2.74 | 1.37 | 1.37 |
| moe.shared (row split) | 1.06 | 0.53 | 0.53 |
| qsa.proj + sdpa + gate_wo (heads split) | 2.02 | 1.01 | 1.01 |
| lm_head (vocab split + local argmax merge) | 0.68 | 0.34 | 0.34 |
| NOT split: norms/inject/write/router/PLE/host/dtoh | ~3.9 | ~3.9 | 0 |

Compute-side total ≈ 17.2 − 6.6 ≈ **10.6 ms**, plus the PCIe join tax: no NVLink on this
box, 2 all-reduces per layer × 48 layers ≈ 96 joins of a 10 KB [hidden] row — at the
measured PCIe P2P latency class (~10-20 µs/join, tp2-join-diet playbook) that is
**+1.0-2.0 ms**, and the graph structure survives only if the joins are captured
(NCCL-in-graph or the direct-join pattern).

**Projected TP2: ≈ 11.6-12.6 ms/token ≈ 79-86 tok/s.** The 90 tok/s target needs TP2
PLUS one more single-card lever, in measured order: sel v3 (headroom ≥2× on 2.8 ms),
a gdn scan/step decode twin (1.5 ms latency-bound), gdn/qsa small-norm batching, and a
quantized lm_head. A W4A4 expert path (the parked `input_scale` consumer) would cut the
sel bytes themselves.

Receipts: `ab-{trunk,ws,graph,selv2,hcmicro}-nvfp4.tsv`, `profile2a/3/4-*.tsv`,
`profile4-nsys-decode8_*.csv`, `hidden/greedy/logits-compare-perf8-nvfp4.tsv`,
`run-perf3-trunk-nvfp4.log`, `run-perf8-hcmicro-nvfp4.log`, box `~/realgate/perf3..8`.
