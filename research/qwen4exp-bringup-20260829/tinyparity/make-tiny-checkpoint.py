#!/usr/bin/env python3
"""Build the tiny random qwen4_exp checkpoint for the cross-oracle parity gate.

Produces an HF directory (config.json + model.safetensors, BF16) that BOTH
transformers main (Qwen4ExpForConditionalGeneration) and memra-gguf's
census-gated qwen4_exp contract accept:

  * trunk + vision namespaces come from transformers save_pretrained with
    save_original_format=False. MEASURED 2026-08-29: the DEFAULT
    save_original_format=True reverse-runs the qwen2_moe expert-fusion
    converter inherited by qwen4_exp_text and emits per-expert 2D
    `mlp.experts.{e}.gate_proj.weight` tensors: NOT the fused 3D
    `mlp.experts.gate_up_proj` layout the real artifact ships. The runtime
    (False) layout matches the artifact census except the ngram table, which
    this script re-shards into split_ngram_parts itself (the load-side
    Concatenate(dim=0) converter's exact inverse: even row chunks in shard
    order);
  * the mtp.* namespace is appended by THIS script: transformers modeling has
    no MTP module (SEMANTICS.md §MTP: the namespace is checkpoint-only,
    semantics banked from SGLang), so the draft tensors are seeded here with
    the exact names/shapes the contract derives.

Weights are drawn in fp32 from a seeded generator, then rounded through BF16
(the contract requires ExactFloat(Bf16)); both oracles upcast BF16->f32
exactly, so parity tolerances stay fp32-tight.

Usage: make-tiny-checkpoint.py <out_dir>   (run inside the tinyparity venv)
"""

import json
import math
import sys

import torch

SEED = 48_2026_0829 % (2**31)  # recorded in TINY-PARITY.md


def tiny_config_dict() -> dict:
    """The one config.json both parsers accept (mirrors the artifact's key set,
    scaled down; every memra-required field present: config.rs refuses by name)."""
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "image_token_id": 60,
        "language_model_only": False,
        "model_type": "qwen4_exp",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 62,
            "dtype": "bfloat16",
            "eos_token_id": 63,
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 8,
            "head_dim": 16,
            "heads_per_ngram": 2,
            "hidden_act": "silu",
            "hidden_size": 32,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 8,
            "indexer_kv_heads": 1,
            # 4 query heads (the artifact's head count), NOT 1: the indexer
            # score is sum_h relu(q_h . k), so with 1 head every all-negative
            # block scores EXACTLY 0.0 and torch.topk breaks the tie in
            # implementation-defined order while the reference pins
            # (score desc, index asc). Parity fixtures must be tie-free
            # (SEMANTICS.md §QSA, dsv4-lane lesson); 4 heads make zero-score
            # blocks rare and dump-hf-goldens.py audits the top-k boundary.
            "indexer_n_heads": 4,
            "initializer_range": 0.02,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 8,
            "make_ngram_vocab_size_divisible_by": 8,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 256,
            "model_type": "qwen4_exp_text",
            "moe_intermediate_size": 16,
            "mtp": {
                "hybrid": True,
                "layer_types": ["full_attention"],
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1,
                "rope_theta": 10000.0,
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "ngram_size": 3,
            "ngram_vocab_size_base": 97,
            "num_attention_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "output_gate_type": "sigmoid",
            "output_router_logits": False,
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 32,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 0],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
            "router_aux_loss_coef": 0.001,
            "seed": 1234,
            "shared_expert_intermediate_size": 16,
            "split_ngram_parts": 2,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 64,
            "intermediate_size": 64,
        },
        "tie_word_embeddings": False,
        "video_token_id": 61,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "model_type": "qwen4_exp",
            "num_heads": 2,
            "num_position_embeddings": 4,
            "out_hidden_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 59,
        "vision_start_token_id": 58,
    }


def randomize_(model: torch.nn.Module, generator: torch.Generator) -> None:
    """Seeded random weights, iterated in sorted-name order for determinism.

    Norm weights are drawn NON-ZERO around 0 (the family's zero-centered (1+w)
    convention) so a missing +1 fold on either side breaks parity loudly
    instead of hiding behind an all-zeros init. I64 hash buffers
    (layer_multipliers / vocab_sizes / offsets) are nn.Buffers, not parameters,
    and keep their config-derived values: LOAD, never re-derive.
    """
    norm_suffixes = (
        "hc_norm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    )
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            data = parameter.data
            if name.endswith("linear_attn.A_log"):
                fresh = torch.empty_like(data).uniform_(0.1, 4.0, generator=generator).log_()
            elif name.endswith("linear_attn.dt_bias"):
                fresh = torch.empty_like(data).uniform_(0.0, 1.0, generator=generator)
            elif name.endswith(norm_suffixes):
                # zero-centered: effective weight = 1 + w
                fresh = torch.empty_like(data).uniform_(-0.4, 0.4, generator=generator)
            elif name.endswith("linear_attn.norm.weight"):
                # RMSNormGated is PLAIN weight (no +1): center near 1.
                fresh = 1.0 + torch.empty_like(data).uniform_(-0.4, 0.4, generator=generator)
            elif name.endswith((".bias",)):
                fresh = torch.empty_like(data).uniform_(-0.05, 0.05, generator=generator)
            else:
                fan_in = data.shape[-1] if data.dim() >= 2 else max(data.numel(), 1)
                std = 1.0 / math.sqrt(fan_in)
                fresh = torch.empty_like(data).normal_(0.0, std, generator=generator)
            data.copy_(fresh)


def mtp_tensors(text: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Synthetic mtp.* namespace with the contract-derived names and shapes
    (pack tensor_schema: one QSA gated-residual block + glue + own mixer)."""
    h = text["hidden_size"]
    hc = text["hc_count"]
    wide = hc * h
    rank = text["hc_lowrank"]
    nh = text["num_attention_heads"]
    kv = text["num_key_value_heads"]
    hd = text["head_dim"]
    idx_out = (text["indexer_n_heads"] + text["indexer_kv_heads"]) * text["indexer_head_dim"]
    experts = text["num_experts"]
    mff = text["moe_intermediate_size"]
    sff = text["shared_expert_intermediate_size"]

    def normal(*shape: int) -> torch.Tensor:
        fan_in = shape[-1]
        return torch.empty(*shape).normal_(0.0, 1.0 / math.sqrt(fan_in), generator=generator)

    def zero_centered_norm(width: int) -> torch.Tensor:
        return torch.empty(width).uniform_(-0.4, 0.4, generator=generator)

    tensors: dict[str, torch.Tensor] = {
        # glue (SGLang _fuse_residual_linear_shared; GemmaRMSNorm = zero-centered (1+w))
        "mtp.fc_embedding.weight": normal(h, h),
        "mtp.fc_hidden.weight": normal(h, h),
        "mtp.pre_fc_norm_embedding.weight": zero_centered_norm(h),
        "mtp.pre_fc_norm_hidden.weight": zero_centered_norm(wide),
        # own exit mixer (read half only, use_combine=False)
        "mtp.hyper_connection_mixer.hc_norm.weight": zero_centered_norm(wide),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": normal(rank, wide),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": normal(wide, rank),
    }
    p = "mtp.layers.0."
    for sub in ("attn_hyper_connection.", "mlp_hyper_connection."):
        tensors[f"{p}{sub}block_inject_weight.weight"] = normal(hc, wide)
        tensors[f"{p}{sub}hc_norm.weight"] = zero_centered_norm(wide)
        tensors[f"{p}{sub}input_mix_weight_down.weight"] = normal(rank, wide)
        tensors[f"{p}{sub}input_mix_weight_up.weight"] = normal(wide, rank)
    tensors.update(
        {
            f"{p}self_attn.q_proj.weight": normal(2 * nh * hd, h),
            f"{p}self_attn.k_proj.weight": normal(kv * hd, h),
            f"{p}self_attn.v_proj.weight": normal(kv * hd, h),
            f"{p}self_attn.o_proj.weight": normal(h, nh * hd),
            f"{p}self_attn.q_norm.weight": zero_centered_norm(hd),
            f"{p}self_attn.k_norm.weight": zero_centered_norm(hd),
            f"{p}self_attn.indexer.index_qk_proj.weight": normal(idx_out, h),
            f"{p}self_attn.indexer.q_layernorm.weight": zero_centered_norm(
                text["indexer_head_dim"]
            ),
            f"{p}self_attn.indexer.k_layernorm.weight": zero_centered_norm(
                text["indexer_head_dim"]
            ),
            f"{p}mlp.gate.weight": normal(experts, h),
            f"{p}mlp.experts.gate_up_proj": normal(experts, 2 * mff, h),
            f"{p}mlp.experts.down_proj": normal(experts, h, mff),
            f"{p}mlp.shared_expert.gate_proj.weight": normal(sff, h),
            f"{p}mlp.shared_expert.up_proj.weight": normal(sff, h),
            f"{p}mlp.shared_expert.down_proj.weight": normal(h, sff),
            f"{p}mlp.shared_expert_gate.weight": normal(1, h),
        }
    )
    return tensors


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make-tiny-checkpoint.py <out_dir>")
    out_dir = sys.argv[1]

    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    config_dict = tiny_config_dict()
    config = Qwen4ExpConfig(**config_dict)
    torch.manual_seed(SEED)
    model = Qwen4ExpForConditionalGeneration(config)
    model = model.float()

    generator = torch.Generator().manual_seed(SEED)
    randomize_(model, generator)

    # Smoke-forward in fp32 before saving: a config the modeling code refuses
    # must fail HERE, not in the parity harness.
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3, 63, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        out = model(input_ids=ids)
        assert out.logits.shape == (1, 16, config_dict["text_config"]["vocab_size"])
        assert torch.isfinite(out.logits).all(), "tiny model produced non-finite logits"

    model = model.to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True, save_original_format=False)

    # Post-process the saved file: re-shard the ngram table into
    # split_ngram_parts (the artifact layout) and append the checkpoint-only
    # mtp.* namespace (bf16, same seed stream).
    from safetensors import safe_open
    from safetensors.torch import save_file

    st_path = f"{out_dir}/model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(st_path, framework="pt") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)

    parts = config_dict["text_config"]["split_ngram_parts"]
    for key in [k for k in tensors if k.endswith("ple.ple_embedding.ngram_embedding.weight")]:
        table = tensors.pop(key)
        rows = table.shape[0]
        assert rows % parts == 0, f"{key}: {rows} rows not divisible into {parts} shards"
        stem = key.removesuffix(".weight")
        for shard, chunk in enumerate(table.chunk(parts, dim=0)):
            tensors[f"{stem}.shard_{shard}.weight"] = chunk.contiguous()

    fused_leftovers = [k for k in tensors if ".mlp.experts." in k and k.endswith("proj.weight")]
    assert not fused_leftovers, (
        "per-expert 2D tensors leaked into the save (save_original_format took the "
        f"qwen2_moe reverse conversion): {fused_leftovers[:4]}"
    )

    draft = {k: v.to(torch.bfloat16) for k, v in mtp_tensors(config_dict["text_config"], generator).items()}
    overlap = set(draft) & set(tensors)
    assert not overlap, f"mtp namespace collided with saved tensors: {sorted(overlap)}"
    tensors.update(draft)
    save_file(tensors, st_path, metadata={"format": "pt"})

    # Current transformers main serializes vision_config.model_type as
    # "qwen4_exp_vision" (the config class rewrote it); the pinned artifact
    # (transformers 5.8.0.dev0 era) ships "qwen4_exp", and memra-gguf's parse
    # keys the vision census on that exact spelling (config.rs qwen4exp_vision
    # arm). Restore the artifact spelling: and see TINY-PARITY.md for the
    # fragility note on re-saved siblings.
    config_path = f"{out_dir}/config.json"
    with open(config_path) as fh:
        saved_config = json.load(fh)
    saved_config["vision_config"]["model_type"] = "qwen4_exp"
    with open(config_path, "w") as fh:
        json.dump(saved_config, fh, indent=2, sort_keys=True)
        fh.write("\n")

    shard_names = sorted(k for k in tensors if ".ngram_embedding.shard_" in k)
    print(f"saved {len(tensors)} tensors to {st_path} (seed {SEED})")
    print(f"ngram shards: {shard_names}")
    for name in sorted(tensors):
        t = tensors[name]
        print(f"  {name}\t{t.dtype}\t{list(t.shape)}")


if __name__ == "__main__":
    main()
