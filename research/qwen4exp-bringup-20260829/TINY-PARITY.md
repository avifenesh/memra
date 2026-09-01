# qwen4_exp tiny cross-oracle parity: transformers vs memra-reference (2026-08-29)

**Verdict: PASS.** All four probes, every trunk layer, final logits, argmax, and the MTP
block agree within fp32 op-order noise: worst **max_abs 2.015e-5**, worst
**max_rel 1.566e-3** (denominator floor 1e-2). Gate thresholds: max_abs 1e-4,
max_rel 3e-3 (~2-5x headroom over the measured envelope).

This is the checkpoint-parity pattern scaled to a CPU-only rig: a tiny random BF16
checkpoint whose SAME config.json both transformers main and memra-gguf parse, opened
through the REAL loading path: `HfConfig::parse` -> `ModelConfig::from_hf` -> qwen4_exp
pack -> `ModelPlan` -> census-gated `TensorContract::bind` against the safetensors header
(175 requirements over 176 tensors, bound clean), then executed by `memra-reference`
and compared per layer against transformers.

## Pins

| what | value |
|---|---|
| memra commit (reference arms under test) | d061e96450 (branch qwen4exp-bringup-20260829) |
| transformers | 5.16.0.dev0 @ git 805a9e939fa8c1bff8d8ffdf041c051b71a914aa |
| torch | 2.13.0+cpu (venv /tmp/q48fn-tinyparity-venv, python 3.14.4) |
| checkpoint seed | 525293533 (`make-tiny-checkpoint.py` SEED; weights random fp32 -> BF16-rounded) |
| tiny config | `tinyparity/tiny-config.json` (the exact saved config.json) |
| HF run dtype | float32, `attn_implementation="eager"` (BF16 weights upcast exactly; both oracles compute on identical values) |

Tiny geometry: hidden 32, hc_count 4 (wide 128), hc_lowrank 8, 4 layers (3 GDN + 1 QSA,
interval 4), head_dim 16 partial-rotary 0.25 (rope 4 dims, theta 1e4), GDN 2 K-heads /
4 V-heads x 8, indexer 4+1 heads x 8 / ratio 4 / budget 8 (=2 blocks), MoE 8 experts
top-2 (ff 16) + gated shared 16, PLE on layer 2 (one-indexed -> module layers.1) with
ngram base 97 / 2 heads-per-ngram / ngram_size 3 / 2 shards (4 primes 97+101+103+107 =
402 -> padded 408 rows), vocab 64, EOS 63, 1 MTP block, tiny 2-block vision tower
(censused, not executed).

## How to re-run

```
/tmp/q48fn-tinyparity-venv/bin/python research/qwen4exp-bringup-20260829/tinyparity/make-tiny-checkpoint.py /tmp/q48fn-tinyparity/ckpt
/tmp/q48fn-tinyparity-venv/bin/python research/qwen4exp-bringup-20260829/tinyparity/dump-hf-goldens.py /tmp/q48fn-tinyparity/ckpt /tmp/q48fn-tinyparity
cargo run -p memra-reference --bin qwen4exp_tiny_parity -- /tmp/q48fn-tinyparity/ckpt /tmp/q48fn-tinyparity/goldens-{a24,b20eos,c32,d8}.bin
```

Venv: python3 -m venv, torch CPU wheel, `pip install git+https://github.com/huggingface/transformers.git`
(needs qwen4_exp: `"qwen4_exp" in CONFIG_MAPPING_NAMES`).

## Probes

| probe | T | shape exercised |
|---|---|---|
| a24 | 24 | plain prompt; queries past position 12 see >2 complete indexer blocks: top-k prunes for real |
| b20eos | 20 | EOS(63) mid-sequence: n-gram EOS-segment reset + PLE fresh segment |
| c32 | 32 | token repeats (n-gram collisions) + two EOS resets; longest selection rows |
| d8 | 8 | degenerate control: <= budget blocks everywhere -> indexer selects all, QSA == plain causal attention |

## Parity numbers

Per-layer table for probe a24 (post-layer WIDE stream [24,128]; layers 0-2 GDN with PLE
on layer 1, layer 3 QSA):

| record | max_abs | max_rel |
|---|---|---|
| layer_hidden.0 | 1.431e-6 | 7.702e-5 |
| layer_hidden.1 (PLE) | 3.338e-6 | 1.524e-4 |
| layer_hidden.2 | 4.202e-6 | 2.088e-4 |
| layer_hidden.3 (QSA) | 3.874e-6 | 2.228e-4 |
| logits [24,64] | 1.788e-6 | 1.155e-4 |
| mtp_hidden [24,128] | 3.815e-6 | 1.743e-4 |
| mtp_logits [24,64] | 1.401e-6 | 7.506e-5 |

All probes (worst record per probe; argmax(last token) matched on every probe):

| probe | worst max_abs | worst max_rel |
|---|---|---|
| a24 | 4.202e-6 | 2.228e-4 |
| b20eos | 1.144e-5 | 3.045e-4 |
| c32 | 2.015e-5 | 1.566e-3 |
| d8 | 3.099e-6 | 1.766e-4 |

Full transcript banked in the lane; regenerate with the commands above (deterministic:
seeded checkpoint + fixed probes).

## What the pass certifies

Gated residual read/write + grouped (1+w) hc_norm, entry repeat + exit mixer downmix,
GDN (V-head reorder transforms, sigmoid z-gate, -exp(A_log), l2-normed qk delta rule,
conv-silu), QSA fused [q|gate] attention with partial rope + (1+w) q/k norms + sigmoid
output gate, the micro-block indexer (raw-key pooling, k_layernorm, block-start rope,
relu-sum scores, top-k + tail), MoE (softmax top-k renormalized router, fused-3D expert
split gate-first, sigmoid-gated shared expert), PLE (splitmix ngram hash via checkpoint
I64 buffers, EOS-segment reset, signed-sqrt sigmoid stream gates, dilation-3 causal
dwconv), and the MTP block (SGLang-fusion: GemmaRMSNorm glue, per-stream fc_hidden +
broadcast fc_embedding, QSA draft layer, own mixer, shared lm_head).

The transformers side of the MTP comparison is a twin built in `dump-hf-goldens.py`
from transformers' own Qwen4ExpTextDecoderLayer/GatedResidual modules wired per the
banked SGLang implementation (raw/sglang_qwen4_exp_mtp.py): transformers modeling has
no MTP module (SEMANTICS.md §MTP), so this is a banked-Python oracle, not transformers
proper.

## Findings (divergences hit and their resolutions)

1. **Indexer relu score ties are STRUCTURAL, not improbable**: the first full-probe run
   diverged only at the QSA layer (layers 0-2 already at ~1e-6). Root cause: with
   `indexer_n_heads=1`, `score = relu(q.k)` is EXACTLY 0.0 for every all-negative block
   (~half of them), and torch.topk breaks the giant 0.0 tie class in
   implementation-defined order (measured: HF picked block 4 over block 0 among equal
   zeros) while the reference pins (score desc, index asc). NOT a reference bug: this is
   precisely the tie rule the in-code pin and SEMANTICS.md §QSA anticipate. Resolution:
   tie-FREE fixtures per the pin: 4 indexer query heads (the artifact's head count;
   P(zero-score block) ~ 2^-4) and a `TieAudit` in dump-hf-goldens.py that recomputes
   every selection row and refuses any top-k boundary gap <= 1e-4. Consequence for later
   lanes: any qwen4_exp indexer parity fixture MUST audit the boundary gap; relu
   guarantees a zero-tie class exists in real checkpoints too, so kernel-vs-reference
   gates need the same audit (or an argsort-stable kernel pin).

2. **save_pretrained default un-fuses the experts**: transformers main's
   `save_original_format=True` (default) reverse-runs the qwen2_moe expert-fusion
   converter inherited by qwen4_exp_text and writes per-expert 2D
   `mlp.experts.{e}.gate_proj.weight` tensors, NOT the fused 3D
   `mlp.experts.gate_up_proj` the real artifact ships. The artifact layout is the
   RUNTIME format (`save_original_format=False`) plus shard-split ngram tables. The
   script saves runtime-format and re-shards the ngram table itself (even dim-0 chunks in
   shard order: the exact inverse of the load-side `Concatenate(dim=0)` converter).
   The census-derived contract is CORRECT for the pinned artifact; but a sibling
   checkpoint re-saved by transformers default settings would present per-expert names
   and refuse at the census gate (correctly: different layout, different program).

3. **vision_config.model_type spelling drifted upstream**: current transformers
   serializes `"qwen4_exp_vision"`; the pinned artifact (5.8.0.dev0 era) ships
   `"qwen4_exp"`, and `config.rs`'s qwen4exp_vision arm keys on exactly that spelling
   (crates/memra-gguf/src/config.rs:2563). The script restores the artifact spelling in
   the saved config.json. Flagging for the engine lane: a future re-exported sibling with
   the new spelling would silently parse as text-only and then FAIL the census bind on
   unclaimed `model.visual.*` tensors (loud, but the error would point at the census, not
   the config arm). Accepting both spellings in config.rs is a one-line engine-lane
   decision; not taken here (contract work stays in its own lane).

4. Two binding squeezes the harness (not the contract) owns, mirrored from the
   reference's fixture layout: `shared_expert_gate` ships [1, hidden] and is consumed
   [hidden]; `ple.conv1d` ships [wide, 1, kernel] and is consumed squeezed
   [wide, kernel] (the GDN conv row gets this via Conv1dSqueezeReorder; the PLE row's
   contract transform is deliberately Identity: family rows fail closed).

## What changed outside tinyparity/

* `crates/memra-reference/src/lib.rs`: `ReferenceOutput` gains
  `layer_hidden: Vec<Vec<f32>>` (post-layer trunk residual states: wide for
  gated-residual trunks). Pure additive debug/parity surface; nothing else constructs
  the struct. This is what makes per-layer localization practical (the QSA divergence
  above was pinned to one layer in one run because of it).
* `crates/memra-reference/src/bin/qwen4exp_tiny_parity.rs`: the gate binary (CPU-only;
  real SafetensorsSource-header census -> contract bind -> transform + (1+w)-fold
  binding -> execute -> compare). The (1+w) fold covers every RMSNorm row EXCEPT
  `linear_attn.norm.weight` (Qwen4ExpTextRMSNormGated is plain-weight, ones-init
  verified in transformers source; same carve-out as llama.cpp qwen.py:302-303).

## Scope

Prefill-shape only (whole prompt, one pass), text-only (vision tower censused and
loaded-past, never executed), no cache-resume shapes, tiny geometry. This gate proves
the reference implements the same MATH as transformers on every qwen4_exp-specific
component; it does not certify the real artifact (checkpoint-parity on real weights is
the next gate in the pack's gate list) nor any engine kernel.
