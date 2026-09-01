#!/usr/bin/env python3
"""Dump transformers goldens for the qwen4_exp tiny cross-oracle parity gate.

Runs the tiny checkpoint (make-tiny-checkpoint.py) in float32 eager attention on
fixed token-id probes and dumps, per probe:

  token_ids        i64 [T]
  layer_hidden.{i} f32 [T, hc_count*hidden]   (forward hook on decoder layer i)
  logits           f32 [T, vocab]
  mtp_hidden       f32 [T, hc_count*hidden]   (post-MTP-layer wide state)
  mtp_logits       f32 [T, vocab]

The trunk goldens come from Qwen4ExpTextModel itself. transformers has no MTP
module (SEMANTICS.md §MTP), so the MTP goldens come from a twin built HERE out
of the same transformers classes wired per the banked SGLang implementation
(raw/sglang_qwen4_exp_mtp.py _fuse_residual_linear_shared + a 1-layer QSA
Qwen4ExpModel + own hyper_connection_mixer + shared lm_head), loaded with the
checkpoint's mtp.* tensors.

Container: magic "Q48FNTP1", u32 count, then per record u32 name_len, name,
u8 dtype (0=f32|1=i64), u32 ndim, ndim*u64 dims, payload (little-endian).

Usage: dump-hf-goldens.py <ckpt_dir> <out_dir>
"""

import copy
import struct
import sys

import torch
import torch.nn.functional as F

# EOS = 63 in the tiny config; PLE resets n-gram segments at it.
PROBES: dict[str, list[int]] = {
    # long enough that queries past position 12 see >budget_blocks complete
    # indexer blocks (ratio 4, budget 8 -> 2 blocks) so top-k actually binds
    "a24": [7, 3, 11, 42, 19, 5, 23, 8, 31, 2, 47, 13, 29, 55, 17, 40, 9, 26, 51, 4, 36, 21, 44, 15],
    # EOS mid-sequence: n-gram EOS-segment reset + a fresh PLE segment
    "b20eos": [12, 34, 6, 27, 50, 18, 41, 3, 22, 63, 9, 45, 30, 14, 57, 25, 38, 11, 48, 20],
    # repeats (n-gram collisions) + two EOS resets, longest probe
    "c32": [5, 9, 5, 9, 5, 63, 33, 12, 33, 12, 46, 7, 28, 52, 16, 39, 63, 24, 10, 43, 5, 9, 5, 61, 35, 19, 49, 2, 56, 30, 13, 37],
    # degenerate control: <= budget_blocks complete blocks everywhere, so the
    # indexer selects everything and QSA reduces to plain causal attention
    "d8": [7, 3, 11, 42, 19, 5, 23, 8],
}

# Top-k boundary tie audit threshold: the reference pins (score desc, index
# asc) while torch.topk tie order is implementation-defined, so a golden is
# only valid when the budget_blocks-th and next scores are strictly separated
# for EVERY query (SEMANTICS.md §QSA indexer, dsv4-lane lesson).
TIE_GAP = 1e-4


class TieAudit:
    """Recomputes indexer block scores from hooked inputs and rejects any
    probe whose top-k boundary is a tie (including the relu zero-score class
    reaching the boundary)."""

    def __init__(self, indexer, label: str):
        self.indexer = indexer
        self.label = label
        self.calls: list[tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = []
        indexer.register_forward_hook(self._hook)

    def _hook(self, _module, args, _output):
        hidden, position_embeddings = args[0], args[1]
        self.calls.append((hidden.detach(), tuple(t.detach() for t in position_embeddings)))

    @torch.no_grad()
    def check_last(self, probe: str) -> None:
        import math

        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

        indexer = self.indexer
        hidden, (full_cos, full_sin) = self.calls[-1]
        tokens = hidden.shape[1]
        heads, dim = indexer.index_n_heads, indexer.index_head_dim
        ratio, budget = indexer.compress_ratio, indexer.block_topk

        qk = indexer.index_qk_proj(hidden)
        q, token_k = torch.split(qk, [heads * dim, indexer.index_kv_heads * dim], dim=-1)
        q = indexer.q_layernorm(q.reshape(1, tokens, -1, dim))
        q = apply_rotary_pos_emb(
            q, cos=full_cos[:, -tokens:, :], sin=full_sin[:, -tokens:, :], unsqueeze_dim=2
        )
        raw_keys = token_k.reshape(1, tokens, -1, dim).squeeze(2)

        for query in range(tokens):
            complete = (query + 1) // ratio
            if complete <= budget:
                continue
            blocks = torch.arange(complete * ratio).view(complete, ratio)
            pooled = raw_keys[0].index_select(0, blocks.flatten()).view(complete, ratio, dim)
            pooled = indexer.k_layernorm(pooled.float().mean(dim=1))
            starts = blocks[:, 0]
            block_keys = apply_rotary_pos_emb(
                pooled.unsqueeze(1),
                cos=full_cos[0].index_select(0, starts),
                sin=full_sin[0].index_select(0, starts),
            ).squeeze(1)
            scores = torch.relu(
                torch.matmul(q[0, query].float(), block_keys.float().transpose(-1, -2))
            ).sum(dim=0) / math.sqrt(dim)
            ranked = scores.sort(descending=True).values
            gap = float(ranked[budget - 1] - ranked[budget])
            if gap <= TIE_GAP:
                raise SystemExit(
                    f"TIE-AUDIT FAIL [{self.label} probe {probe} query {query}]: "
                    f"top-k boundary gap {gap:.3e} <= {TIE_GAP} "
                    f"(scores {scores.tolist()}): torch.topk tie order is "
                    "implementation-defined; reseed or change the probe"
                )


def write_goldens(path: str, records: dict[str, torch.Tensor]) -> None:
    payload = bytearray()
    payload += b"Q48FNTP1"
    payload += struct.pack("<I", len(records))
    for name, tensor in records.items():
        tensor = tensor.detach().contiguous().cpu()
        if tensor.dtype == torch.int64:
            tag, raw = 1, tensor.numpy().tobytes()
        else:
            tag, raw = 0, tensor.to(torch.float32).numpy().tobytes()
        encoded = name.encode()
        payload += struct.pack("<I", len(encoded)) + encoded
        payload += struct.pack("<B", tag)
        payload += struct.pack("<I", tensor.dim())
        for dim in tensor.shape:
            payload += struct.pack("<Q", dim)
        payload += raw
    with open(path, "wb") as fh:
        fh.write(payload)


def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """SGLang GemmaRMSNorm: zero-centered (1+w), fp32 accumulation."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return normed * (1.0 + weight.float())


class MtpTwin:
    """The SGLang Qwen4ExpForCausalLMMTP math on transformers modules."""

    def __init__(self, model, checkpoint: dict[str, torch.Tensor]):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextDecoderLayer,
            Qwen4ExpTextGatedResidual,
        )

        text_config = model.config.text_config
        draft_config = copy.deepcopy(text_config)
        draft_config.num_hidden_layers = 1
        draft_config.layer_types = ["full_attention"]
        draft_config.full_attention_interval = 1
        draft_config.ple_layer_ids = []

        self.eps = text_config.rms_norm_eps
        self.hc_count = text_config.hc_count
        self.hidden = text_config.hidden_size
        self.embed = model.model.language_model.embed_tokens
        self.lm_head_weight = model.lm_head.weight.float()
        self.rotary = model.model.language_model.rotary_emb

        def block(prefix: str) -> dict[str, torch.Tensor]:
            return {
                key.removeprefix(prefix): value.float()
                for key, value in checkpoint.items()
                if key.startswith(prefix)
            }

        self.layer = Qwen4ExpTextDecoderLayer(draft_config, 0).float()
        self.layer.load_state_dict(block("mtp.layers.0."), strict=True)
        self.mixer = Qwen4ExpTextGatedResidual(draft_config, use_combine=False).float()
        self.mixer.load_state_dict(block("mtp.hyper_connection_mixer."), strict=True)
        self.fc_embedding = checkpoint["mtp.fc_embedding.weight"].float()
        self.fc_hidden = checkpoint["mtp.fc_hidden.weight"].float()
        self.norm_embedding = checkpoint["mtp.pre_fc_norm_embedding.weight"].float()
        self.norm_hidden = checkpoint["mtp.pre_fc_norm_hidden.weight"].float()
        self.layer.eval()
        self.mixer.eval()

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor, wide_trunk: torch.Tensor):
        """token_ids [1, T]; wide_trunk [1, T, hc*hidden] (post last trunk layer).

        Returns (mtp_hidden [1, T, wide], mtp_logits [1, T, vocab]).
        """
        tokens = token_ids.shape[1]
        # _fuse_residual_linear_shared (sglang_qwen4_exp_mtp.py L105-115)
        embeds = self.embed(token_ids).float()
        fused_embed = F.linear(
            gemma_rms_norm(embeds, self.norm_embedding, self.eps), self.fc_embedding
        )
        hidden = gemma_rms_norm(wide_trunk.float(), self.norm_hidden, self.eps)
        per_stream = hidden.view(*hidden.shape[:-1], self.hc_count, self.hidden)
        fused = (fused_embed.unsqueeze(-2) + F.linear(per_stream, self.fc_hidden)).flatten(-2)

        # eager float causal mask (0 visible / min masked) + full-position rope
        min_value = torch.finfo(torch.float32).min
        mask = torch.full((tokens, tokens), min_value).triu(1)[None, None]
        position_ids = torch.arange(tokens)[None]
        cos, sin = self.rotary(fused, position_ids)

        wide = self.layer(
            fused,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            conv_mask=None,
            past_key_values=None,
            ple_input_ids=None,
        )
        mixed = self.mixer(wide)
        logits = F.linear(mixed, self.lm_head_weight)
        return wide, logits


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: dump-hf-goldens.py <ckpt_dir> <out_dir>")
    ckpt_dir, out_dir = sys.argv[1], sys.argv[2]

    from safetensors import safe_open
    from transformers import Qwen4ExpForConditionalGeneration

    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        ckpt_dir, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    checkpoint: dict[str, torch.Tensor] = {}
    with safe_open(f"{ckpt_dir}/model.safetensors", framework="pt") as handle:
        for key in handle.keys():
            if key.startswith("mtp."):
                checkpoint[key] = handle.get_tensor(key)
    twin = MtpTwin(model, checkpoint)

    language_model = model.model.language_model
    audits = [
        TieAudit(layer.self_attn.indexer, f"trunk layer {index}")
        for index, layer in enumerate(language_model.layers)
        if hasattr(layer, "self_attn")
    ]
    audits.append(TieAudit(twin.layer.self_attn.indexer, "mtp layer 0"))
    layer_outputs: dict[int, torch.Tensor] = {}

    def capture(index: int):
        def hook(_module, _inputs, output):
            layer_outputs[index] = output.detach().float()

        return hook

    for index, layer in enumerate(language_model.layers):
        layer.register_forward_hook(capture(index))

    for probe, ids in PROBES.items():
        layer_outputs.clear()
        token_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = language_model(input_ids=token_ids, use_cache=False)
            logits = F.linear(out.last_hidden_state.float(), model.lm_head.weight.float())
        wide_trunk = layer_outputs[len(language_model.layers) - 1]
        mtp_hidden, mtp_logits = twin.forward(token_ids, wide_trunk)
        for audit in audits:
            audit.check_last(probe)

        records: dict[str, torch.Tensor] = {"token_ids": token_ids[0]}
        for index in sorted(layer_outputs):
            records[f"layer_hidden.{index}"] = layer_outputs[index][0]
        records["logits"] = logits[0]
        records["mtp_hidden"] = mtp_hidden[0]
        records["mtp_logits"] = mtp_logits[0]

        path = f"{out_dir}/goldens-{probe}.bin"
        write_goldens(path, records)
        print(
            f"{probe}: T={len(ids)} logits[{tuple(logits[0].shape)}] "
            f"argmax(last)={int(logits[0, -1].argmax())} -> {path}"
        )


if __name__ == "__main__":
    main()
