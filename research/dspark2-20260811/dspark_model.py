"""Exact bounded DSpark pilot model for the Qwen3.5-9B single-tap corpus.

The frozen embedding/head tables are exported from the deployed GGUF in draft-vocabulary
order. They are non-persistent buffers: checkpoints contain only trainable pilot state and are
therefore always paired with the hashed shared artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DSparkConfig:
    d_model: int = 4096
    n_layers: int = 2
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 256
    ffn_dim: int = 8192
    draft_vocab: int = 32768
    target_vocab: int = 248320
    markov_rank: int = 256
    block_size: int = 5
    rms_eps: float = 1.0e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    initializer_range: float = 0.02

    def validate(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv_heads == 0
        assert self.block_size == 5
        rope_dim = int(self.head_dim * self.partial_rotary_factor)
        assert rope_dim > 0 and rope_dim % 2 == 0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value_f32 = value.float()
        normalized = value_f32 * torch.rsqrt(value_f32.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


def _apply_partial_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    theta: float,
) -> torch.Tensor:
    """Apply Qwen3.5 text RoPE to the configured prefix and leave the suffix unchanged."""
    frequencies = 1.0 / (
        theta
        ** (torch.arange(0, rope_dim, 2, device=value.device, dtype=torch.float32) / rope_dim)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    cos = angles.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = angles.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
    rotated = value[..., :rope_dim]
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    rotate_half = torch.stack((-odd, even), dim=-1).flatten(-2)
    prefix = rotated * cos.to(rotated.dtype) + rotate_half * sin.to(rotated.dtype)
    return torch.cat((prefix, value[..., rope_dim:]), dim=-1)


class DSparkAttention(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        target_context: torch.Tensor,
        noise_positions: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, noise_len, _ = hidden.shape
        context_len = target_context.shape[1]
        q = self.q_proj(hidden).view(
            batch, noise_len, self.config.n_heads, self.config.head_dim
        )
        k_context = self.k_proj(target_context)
        k_noise = self.k_proj(hidden)
        v_context = self.v_proj(target_context)
        v_noise = self.v_proj(hidden)
        k = torch.cat((k_context, k_noise), dim=1).view(
            batch,
            context_len + noise_len,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        v = torch.cat((v_context, v_noise), dim=1).view(
            batch,
            context_len + noise_len,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        rope_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        q = _apply_partial_rope(q, noise_positions, rope_dim, self.config.rope_theta)
        kv_positions = torch.cat((context_positions, noise_positions), dim=1)
        k = _apply_partial_rope(k, kv_positions, rope_dim, self.config.rope_theta)
        groups = self.config.n_heads // self.config.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        # One predecessor carrier plus five mutually visible noise tokens is wholly inside SWA-128.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().view(batch, noise_len, -1)
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class DSparkLayer(nn.Module):
    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attention = DSparkAttention(config)
        self.post_attention_norm = RMSNorm(config.d_model, config.rms_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden: torch.Tensor,
        target_context: torch.Tensor,
        noise_positions: torch.Tensor,
        context_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.attention(
            self.input_norm(hidden), target_context, noise_positions, context_positions
        )
        return hidden + self.mlp(self.post_attention_norm(hidden))


class FrozenSharedWeights(nn.Module):
    def __init__(self, embedding: torch.Tensor, lm_head: torch.Tensor):
        super().__init__()
        assert embedding.ndim == 2 and lm_head.shape == embedding.shape
        self.register_buffer("embedding", embedding, persistent=False)
        self.register_buffer("lm_head", lm_head, persistent=False)

    def embed(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(draft_ids, self.embedding)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head)


@dataclass
class DSparkOutput:
    base_logits: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    confidence_logits: torch.Tensor
    prev_draft_ids: torch.Tensor
    prev_in_trim: torch.Tensor
    anchor_in_trim: torch.Tensor


class DSparkModel(nn.Module):
    def __init__(
        self,
        config: DSparkConfig,
        embedding_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ):
        super().__init__()
        config.validate()
        assert embedding_weight.shape == (config.draft_vocab, config.d_model)
        assert lm_head_weight.shape == embedding_weight.shape
        self.config = config
        self.shared = FrozenSharedWeights(embedding_weight, lm_head_weight)
        self.mask_embedding = nn.Parameter(torch.empty(config.d_model))
        self.hidden_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.hidden_norm = RMSNorm(config.d_model, config.rms_eps)
        self.layers = nn.ModuleList(DSparkLayer(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.markov_w1 = nn.Embedding(config.draft_vocab, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.draft_vocab, bias=False)
        self.confidence_head = nn.Linear(config.d_model + config.markov_rank, 1, bias=True)
        self.apply(self._initialize)
        with torch.no_grad():
            self.mask_embedding.copy_(embedding_weight.float().mean(dim=0))

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _markov_embedding(
        self, prev_draft_ids: torch.Tensor, prev_in_trim: torch.Tensor
    ) -> torch.Tensor:
        safe = prev_draft_ids.clamp_min(0)
        latent = self.markov_w1(safe)
        return latent * prev_in_trim.unsqueeze(-1).to(latent.dtype)

    def forward(
        self,
        target_hidden: torch.Tensor,
        tokens_target: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_to_draft: torch.Tensor,
    ) -> DSparkOutput:
        assert tokens_target.shape[-1] == self.config.block_size + 1
        mapped = target_to_draft[tokens_target.long()].long()
        anchor_draft = mapped[:, 0]
        anchor_in_trim = anchor_draft.ge(0)
        anchor_embedding = self.shared.embed(anchor_draft.clamp_min(0))
        mask = self.mask_embedding.to(anchor_embedding.dtype).expand_as(anchor_embedding)
        anchor_embedding = torch.where(anchor_in_trim.unsqueeze(-1), anchor_embedding, mask)
        noise = torch.cat(
            (
                anchor_embedding.unsqueeze(1),
                self.mask_embedding.to(anchor_embedding.dtype)
                .view(1, 1, -1)
                .expand(anchor_embedding.shape[0], self.config.block_size - 1, -1),
            ),
            dim=1,
        )
        context = self.hidden_norm(self.hidden_projection(target_hidden)).unsqueeze(1)
        context_positions = (anchor_positions.long() - 1).clamp_min(0).unsqueeze(1)
        offsets = torch.arange(self.config.block_size, device=noise.device).unsqueeze(0)
        noise_positions = anchor_positions.long().unsqueeze(1) + offsets
        hidden = noise
        for layer in self.layers:
            hidden = layer(hidden, context, noise_positions, context_positions)
        hidden = self.final_norm(hidden)
        base_logits = self.shared.project(hidden)
        prev_draft_ids = mapped[:, :-1]
        prev_in_trim = prev_draft_ids.ge(0)
        markov_latent = self._markov_embedding(prev_draft_ids, prev_in_trim)
        logits = base_logits + self.markov_w2(markov_latent)
        confidence_features = torch.cat((hidden, markov_latent.to(hidden.dtype)), dim=-1)
        confidence_logits = self.confidence_head(confidence_features).squeeze(-1).float()
        return DSparkOutput(
            base_logits=base_logits,
            logits=logits,
            hidden=hidden,
            confidence_logits=confidence_logits,
            prev_draft_ids=prev_draft_ids,
            prev_in_trim=prev_in_trim,
            anchor_in_trim=anchor_in_trim,
        )

    def recursive_logits_and_confidence(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        anchor_draft_ids: torch.Tensor,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposals = []
        corrected = []
        confidence = []
        previous = anchor_draft_ids
        for position in range(self.config.block_size):
            previous_valid = previous.ge(0)
            latent = self._markov_embedding(previous, previous_valid)
            step_logits = base_logits[:, position].float() + self.markov_w2(latent).float()
            confidence_features = torch.cat(
                (hidden[:, position], latent.to(hidden.dtype)), dim=-1
            )
            confidence.append(self.confidence_head(confidence_features).squeeze(-1).float())
            if temperature == 0.0:
                proposal = step_logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(step_logits / temperature, dim=-1)
                proposal = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            proposals.append(proposal)
            corrected.append(step_logits)
            previous = proposal
        return (
            torch.stack(proposals, dim=1),
            torch.stack(corrected, dim=1),
            torch.stack(confidence, dim=1),
        )


def load_bf16_matrix(path: Path, rows: int, columns: int) -> torch.Tensor:
    expected = rows * columns * 2
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} has {actual} bytes, expected {expected}")
    return torch.from_file(
        str(path), shared=False, size=rows * columns, dtype=torch.bfloat16
    ).reshape(rows, columns)


def load_shared_artifact(path: Path, config: DSparkConfig) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        load_bf16_matrix(path / "embedding.bf16", config.draft_vocab, config.d_model),
        load_bf16_matrix(path / "lm_head.bf16", config.draft_vocab, config.d_model),
    )


def expected_trainable_parameters(config: DSparkConfig) -> int:
    d = config.d_model
    h = config.n_heads
    kv = config.n_kv_heads
    hd = config.head_dim
    ffn = config.ffn_dim
    per_layer = (
        d * h * hd
        + 2 * d * kv * hd
        + h * hd * d
        + 2 * hd
        + 2 * d
        + 3 * d * ffn
    )
    return (
        d
        + d * d
        + d
        + config.n_layers * per_layer
        + d
        + 2 * config.draft_vocab * config.markov_rank
        + d
        + config.markov_rank
        + 1
    )
