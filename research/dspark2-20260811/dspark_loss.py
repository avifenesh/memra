"""Full-trim DSpark loss with a non-winnable teacher escape bucket."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def build_target_to_draft(d2t: torch.Tensor, target_vocab: int) -> torch.Tensor:
    d2t = d2t.long()
    if d2t.ndim != 1 or d2t.unique().numel() != d2t.numel():
        raise ValueError("d2t must be a unique one-dimensional target-id list")
    if d2t.min().item() < 0 or d2t.max().item() >= target_vocab:
        raise ValueError("d2t contains a target id outside target_vocab")
    inverse = torch.full((target_vocab,), -1, dtype=torch.long, device=d2t.device)
    inverse[d2t] = torch.arange(d2t.numel(), device=d2t.device)
    return inverse


def sparse_teacher_distribution(
    top_target_ids: torch.Tensor,
    top_probabilities: torch.Tensor,
    tail_probability: torch.Tensor,
    target_to_draft: torch.Tensor,
    draft_vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map target top-k probabilities into the trim and aggregate every miss as escape."""
    mapped = target_to_draft[top_target_ids.long()]
    retained = mapped.ge(0)
    dense_shape = (*mapped.shape[:-1], draft_vocab)
    dense = torch.zeros(dense_shape, dtype=torch.float32, device=top_probabilities.device)
    safe_indices = mapped.clamp_min(0)
    dense.scatter_add_(-1, safe_indices, top_probabilities.float() * retained.float())
    escape = tail_probability.float() + (
        top_probabilities.float() * (~retained).float()
    ).sum(dim=-1)
    mass_error = (dense.sum(dim=-1) + escape - 1.0).abs().max()
    if mass_error.item() > 2.0e-4:
        raise ValueError(f"sparse teacher mass error {mass_error.item():.6g}")
    return dense, escape


@dataclass
class DSparkLoss:
    total: torch.Tensor
    ce: torch.Tensor
    tvd: torch.Tensor
    confidence: torch.Tensor
    acceptance: torch.Tensor
    escape: torch.Tensor
    ce_in_trim: torch.Tensor

    def metrics(self) -> dict[str, float | list[float]]:
        acceptance = self.acceptance.detach().float()
        return {
            "loss": self.total.detach().item(),
            "ce": self.ce.detach().item(),
            "tvd": self.tvd.detach().item(),
            "confidence_bce": self.confidence.detach().item(),
            "escape_mean": self.escape.detach().float().mean().item(),
            "ce_in_trim_rate": self.ce_in_trim.detach().float().mean().item(),
            "acceptance_by_slot": acceptance.mean(dim=0).cpu().tolist(),
            "q2_analytical": acceptance[:, :2].prod(dim=-1).mean().item(),
        }


def compute_dspark_loss(
    draft_logits: torch.Tensor,
    confidence_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    top_target_ids: torch.Tensor,
    top_probabilities: torch.Tensor,
    tail_probability: torch.Tensor,
    target_to_draft: torch.Tensor,
    *,
    temperature: float = 0.7,
    gamma: float = 5.0,
) -> DSparkLoss:
    if temperature <= 0:
        raise ValueError("training temperature must be positive")
    draft_vocab = draft_logits.shape[-1]
    teacher, escape = sparse_teacher_distribution(
        top_target_ids,
        top_probabilities,
        tail_probability,
        target_to_draft,
        draft_vocab,
    )
    draft_log_probabilities = F.log_softmax(draft_logits.float() / temperature, dim=-1)
    draft_probabilities = draft_log_probabilities.exp()
    l1 = (draft_probabilities - teacher).abs().sum(dim=-1) + escape
    tvd_by_token = 0.5 * l1
    # For two normalized distributions, speculative acceptance is 1 - TVD = 1 - 0.5*L1.
    # This follows the current DeepSpec implementation and resolves the SPEC's duplicated factor.
    acceptance = (1.0 - tvd_by_token).clamp(0.0, 1.0)

    mapped_targets = target_to_draft[target_tokens.long()]
    ce_in_trim = mapped_targets.ge(0)
    ce_by_token = F.nll_loss(
        draft_log_probabilities.reshape(-1, draft_vocab),
        mapped_targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).reshape_as(mapped_targets)
    weights = torch.exp(
        -torch.arange(draft_logits.shape[1], device=draft_logits.device).float() / gamma
    ).unsqueeze(0)
    weights = weights.expand_as(tvd_by_token)
    ce_weights = weights * ce_in_trim.float()
    ce = (ce_by_token * ce_weights).sum() / ce_weights.sum().clamp_min(1.0)
    tvd = (tvd_by_token * weights).sum() / weights.sum()
    confidence_by_token = F.binary_cross_entropy_with_logits(
        confidence_logits.float(), acceptance.detach(), reduction="none"
    )
    confidence = (confidence_by_token * weights).sum() / weights.sum()
    total = 0.1 * ce + 0.9 * tvd + confidence
    return DSparkLoss(
        total=total,
        ce=ce,
        tvd=tvd,
        confidence=confidence,
        acceptance=acceptance,
        escape=escape,
        ce_in_trim=ce_in_trim,
    )
