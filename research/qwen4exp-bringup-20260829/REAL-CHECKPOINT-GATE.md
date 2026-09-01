# qwen4_exp REAL-CHECKPOINT gate — first engine run of the real 360 GB artifacts (2026-08-29)

**Verdict: PASS on both artifacts.** The GPU eager arm loaded the pinned BF16 export
(fused-3D dialect) AND the per-expert NVFP4 mint through the pack/plan/contract loader
and tracked the banked transformers goldens end to end: per-layer wide-stream envelopes
grow smoothly with no layer-class discontinuity, **final-logits argmax agrees 10/10 on
both arms**, and the 64-token greedy chains match the HF bf16 chains for 64/64 steps on
prompt 0 (both arms) and 64/64 on prompt 2 (NVFP4), with all divergences late and
coherent. One real finding was hit and fixed in-lane (non-pow2 NVFP4 macros, below).

Gate binary: `qwen4exp_real_gate` (crates/memra-engine/src/bin/qwen4exp_real_gate.rs);
goldens prep shim: `gpu-eager/prep-real-gate.py`. Receipts:
`gpu-eager/real-checkpoint/` (tsv/log/json committed; `probe-logits-*.bin` and the
30 MB `hidden-goldens.pt` mirrored there UNCOMMITTED — rig copy is the reclaim
insurance for the spot box; public-repo hygiene keeps binaries out of git).

## Pins

| what | value |
|---|---|
| memra commits under test | aa9b9e67eb (gate seams) + 0c57fc75ea (NVFP4 macro fix) on qwen4exp-bringup-20260829; box HEAD = 0c57fc75ea |
| BF16 artifact | ~/data/q48fn-bf16 (Qwen/Qwen3.8-Flash-Next @ de4b8e4d, 131 shards, 336 GiB) |
| NVFP4 artifact | ~/data/q48fn-nvfp4 (Avifenesh/Qwen3.8-Flash-Next-NVFP4 mint, 9 shards + mtp bf16 graft, 174 GB) |
| goldens | ~/goldens/{hidden-goldens.pt, greedy-goldens.json} — transformers main, bf16 forward on the BF16 (make-goldens.py) |
| box | sbox-eval Frankfurt: 2× RTX PRO 6000 Blackwell 96 GB (sm_120a), 48 vCPU, 499 GB RAM, driver 595.91.07 |
| toolchain | rustc 1.98.0, nvcc auto-picked CUDA 13.2 (/usr/local/cuda-13.2), MEMRA_CUDA_ARCH auto 120a, `cargo build --release` |
| binaries (sha256 prefix) | bf16 run 4d87daf0c12bbfab (pre-fix build @ aa9b9e67eb); nvfp4 run 950052cf339019f3 (@ 0c57fc75ea) |
| probe | "The quick brown fox jumps over the lazy dog." → T=10 ids [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13] (from hidden-goldens.pt, never re-tokenized) |

## Dialect binding (deliverable: verify which dialect each dir probes to)

- **BF16 dir → FusedBanks** (probe key `…experts.0.gate_proj.weight` absent). Full
  contract walk bound clean: zero name/shape/dtype refusals across 48 trunk layers,
  128 n-gram shards concatenated to [320001536, 160], I64 buffers loaded. No census
  finding — the dir binds as pinned.
- **NVFP4 dir → PerExpertModelopt** (probe key present). Names/shapes/dtypes bound
  clean; the VALUE domain produced the lane's one real finding:

### FINDING → FIX: the real mint's `weight_scale_2` is NOT pow2

First load refused at
`model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_2 = 5.9945243e-5`
— the dsv4-inherited pow2-macro law (`assert_pow2_macro`) assumed the macro multiplies
inside the bf16-emitting dequant kernel, where only pow2 is exact. modelopt's
`weight_scale_2` is amax-derived (arbitrary float); the rig census gate checked
names/dtypes/shapes, so this was the first VALUE-level contact with the mint.

Fix (0c57fc75ea, qwen4exp module only — dsv4 keeps its own law): the kernel now runs
with macro 1.0 (e2m1 × e4m3 products carry ≤ 6 significand bits → EXACT in bf16) and
the macro multiplies AFTER the exact f32 upcast — bit-identical to the host decoder's
`(code * scale) * scale_2` single-rounding chain for any finite macro.
`validate_macro` still refuses non-finite/non-positive. Tiny gate arm D
(dir-nvfp4-perexpert) now synthesizes the measured non-pow2 macro and re-ran GREEN on
the rig (receipt gpu-eager/tiny-fixture-gate.tsv re-banked), so the new chain is gated
kernel-vs-host-decoder at both macro classes (arm C keeps pow2).

## Per-layer envelope vs transformers bf16 (hidden-gate-{bf16,nvfp4}.tsv)

Reference = HF bf16 FORWARD (activations bf16), candidate = our f32 eager on the same
weights — the envelope measures bf16 accumulation drift as much as our math. Wide-state
magnitudes (ref_absmax) grow 0.32 → 25.75 over 48 layers; at magnitude ~26 the bf16
quantum is ~0.25/element, which is the scale the deep-layer max_abs must be read
against. rel = |diff| / max(1, |ref|).

Measured envelope (full tables in the receipts):

| record | BF16 arm max_abs / mean_abs | NVFP4 arm max_abs / mean_abs | ref_absmax |
|---|---|---|---|
| layer0 | 4.879e-3 / 6.0e-5 | 7.258e-3 / 2.1e-4 | 0.32 |
| layer1 (PLE) | 4.612e-3 / 1.2e-4 | 7.216e-3 / 3.4e-4 | 0.46 |
| layer11 | 4.541e-2 / 8.5e-4 | 4.517e-2 / 1.7e-3 | 3.0 |
| layer23 | 3.042e-2 / 1.9e-3 | 4.141e-2 / 3.3e-3 | 3.0 |
| layer35 | 8.114e-2 / 5.3e-3 | 2.175e-1 / 9.2e-3 | 3.9 |
| layer47 (QSA) | 7.938e-1 / 3.5e-2 | 1.014e0 / 5.8e-2 | 25.8 |
| exit_mixer | 1.097e1 / 2.5e-1 | 1.702e1 / 4.4e-1 | 188.0 |
| logits | 3.711e0 / 1.6e-1 | 6.437e0 / 2.7e-1 | 19.5 |
| **logits argmax** | **10/10 rows** | **10/10 rows** | |

Reading:

- Growth is smooth (~15%/layer in mean_abs) and magnitude-tracking; the task's starting
  threshold (max_abs/rel ≤ 3e-2) holds through ~layer 23 on the BF16 arm and the
  exceedances beyond are the accumulation class, not a fault: no jump at any layer-class
  boundary. QSA layers (3, 7, …, 47) sit in-family with their GDN neighbors; the PLE
  layer (1) is in-family → the 102 GB host n-gram table + host hashing + gather + PLE
  conv path are correct against HF.
- The NVFP4 arm's envelope is a uniform ~1.5-2× the BF16 arm's at every depth —
  quantization noise stacked on the same accumulation shape, no localized damage.

## Greedy continuation gate (greedy-gate-{bf16,nvfp4}.tsv, decoded twins banked)

64-token argmax chains vs HF bf16 `generate(do_sample=False)`. Early divergence
(step 1-2 everywhere) would have meant a bug; measured:

| prompt | BF16 arm first_div | NVFP4 arm first_div | coherence past fork |
|---|---|---|---|
| 0 merge linked lists | **none (64/64)** | **none (64/64)** | identical |
| 1 capital of Australia | 8 | 8 | equivalent phrasing ("has stated" vs "has made a statement"), both correct Canberra answer |
| 2 def fib(n) | 7 | **none (64/64)** | BF16 arm forks on indentation width, stays valid python |
| 3 translate to French | 34 | 48 | same "Il fait beau aujourd'hui" reasoning, phrasing fork |

All four continuations remain fluent and semantically equivalent after the fork
(greedy-decoded-{bf16,nvfp4}.json) — the bf16-vs-f32 argmax-tie class, exactly the
expected shape. Note the NVFP4 arm tracks the HF chain as long or LONGER than our own
BF16 arm on 2/4 prompts: at these fork points argmax tie sensitivity dominates
quantization error.

## NVFP4 vs BF16 cross-arm logits (logits-compare-nvfp4.tsv; ours vs ours, same probe)

Per-row over the 10 probe positions, reference = our BF16-arm logits:

- top-1: **10/10 match**.
- top-20 overlap: 13-20 of 20 (median 18).
- KL(bf16‖nvfp4): rows 0-8 in **8.3e-4 … 4.7e-2**; **mean over rows 0.0373** — inside
  the mint-gate expectation (≤ ~0.04). The one above-envelope value is the FINAL row
  (position 9, after the sentence-ending "."): **KL 0.293**, top-20 13/20 — a single
  high-entropy continuation position where the BF16 arm itself is farthest from HF
  (row-9 max_abs 3.7 vs HF). Top-1 still agrees and the greedy chains above track HF
  through real decode. A serving-qualification KL claim needs a real prompt battery
  (many rows), which is that lane's job, not this probe's.

## Memory reality + residency (deliverable 5)

- **The eager arm is single-device.** Anchor: `Qwen4ExpGpu` holds `CudaSlice`s from ONE
  `Engine` (crates/memra-engine/src/qwen4exp_gpu.rs); the gate constructs
  `Engine::new(0)`. GPU1 sat at 3 MiB for the whole task. Multi-GPU sharding is engine
  work for a later lane, not a gate flag.
- **NVFP4 fits one 96 GB card**: 81.9 GiB used post-load / 82.0 GiB post-decode
  (~68 GB as-stored expert banks + ~14.7 GB trunk f32 + 2.5 GB lm_head f32 + state).
  ~16 GB headroom at T≈42; long-context KV/state budget is a serving-lane question.
- **BF16 cannot be device-resident on any single card**: f32 banks ≈ 483 GB, raw bf16
  banks ≈ 242 GB > 96 GB (and > 192 GB with trunk in f32 across two cards even if the
  arm could shard). Gate residency used: `LoadOptions::host_bf16_banks` keeps banks
  host-resident raw bf16 and uploads+upcasts per ROUTED expert per forward (~9.8 MB ×
  10 experts × 48 layers ≈ 4.7 GB H2D per decode token). Device: 17.2 GiB. Host peak
  ~331 GB of 499 (banks 242 + table 102 + trunk f32 ~20). Gate-mode residency only.
- **n-gram table**: HOST-resident bf16 (102.4 GB) in both arms, by construction
  (NgramTable::Bf16; gather is host math). The consuming loader
  (`from_loaded_checkpoint`) moves it — the previous clone would have doubled it.
- Load wall-clock (disk-bound, ~360 MB/s effective): BF16 1013 s, NVFP4 873 s.

## Timing SIGNAL — untuned eager, correctness-arm residency, NEVER a perf claim

- NVFP4 (device-resident banks, per-routed-expert kernel dequant): decode
  **77.6 ms/token mean (12.9 tok/s)**, p90 77.8 ms, at T≈10-75; prefill T=10 0.66 s.
- BF16 host-bank arm: decode ~554 ms/token (H2D-dominated by design); prefill T=10 2.8 s.
- Untuned: per-expert loop MoE, dense masked attention, host indexer/routing twins,
  synchronous ngram gather, no graphs/batching. The perf lanes own real numbers.

## Indexer layernorm (1+w) — the SEMANTICS.md VERIFY, settled

Two facts close it:

1. **tinyparity already pinned the fold against transformers with REAL block dropping**:
   tiny geometry (budget 2 blocks) prunes for real past position 11, probes a24/c32
   passed at worst 2.015e-5 abs / 1.566e-3 rel with the (1+w) fold applied to the
   indexer q/k layernorms (TINY-PARITY.md) — transformers' own module on one side,
   the folded reference on the other.
2. **On real geometry, this gate's probes are structurally INSENSITIVE to the fold**:
   budget 512 blocks × block 4 = 2048 tokens, so any prompt shorter than ~2051 tokens
   selects EVERY complete block regardless of scores — the indexer norms only order
   scores. T here is 10-75. A real-geometry discriminating arm needs a >2051-token
   prompt (and an HF long-prompt golden to arbitrate); that belongs to the long-context
   lane. The `--indexer-norm-raw` two-arm knob is built and wired for it.

Consistent with both: QSA layers show no envelope discontinuity in either arm's
per-layer table. The loader fold stays (1+w) on the indexer layernorms.

## How to re-run (box)

```
~/venv/bin/python ~/memra/research/qwen4exp-bringup-20260829/gpu-eager/prep-real-gate.py \
    ~/data/q48fn-bf16 ~/goldens ~/realgate/dump
cd ~/memra && cargo build --release -p memra-engine --bin qwen4exp_real_gate
./target/release/qwen4exp_real_gate ~/data/q48fn-bf16 ~/realgate/out --label bf16 \
    --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
    --decode-timing 16 --host-bf16-banks
./target/release/qwen4exp_real_gate ~/data/q48fn-nvfp4 ~/realgate/out --label nvfp4 \
    --goldens ~/realgate/dump --prompts ~/realgate/dump/prompts.tsv \
    --decode-timing 32 --compare-logits ~/realgate/out/probe-logits-bf16.bin
```

## What this phase does NOT certify

Sampled serving behavior (greedy is the instrument), MTP/spec execution, vision, long
context (>2051-token indexer pruning on real geometry), batching, multi-GPU, tokenizer/
template serving surface, or any perf/context product claim. Production admission still
requires the NativeQualified gate set.
