"""Held-out DSpark q trajectory, rejection sampling, and sequential calibration."""

from __future__ import annotations

import math

import torch

from dspark_loss import compute_dspark_loss, sparse_teacher_distribution


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> float:
    probabilities = probabilities.detach().double().flatten().cpu()
    labels = labels.detach().double().flatten().cpu()
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = (probabilities >= low) & (
            probabilities <= high if index == bins - 1 else probabilities < high
        )
        if selected.any():
            weight = selected.double().mean().item()
            error += weight * abs(
                probabilities[selected].mean().item() - labels[selected].mean().item()
            )
    return error


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().double().flatten().cpu()
    labels = labels.detach().bool().flatten().cpu()
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    sorted_scores = scores[order]
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels].sum().item()
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def fit_sequential_temperature_scaling(
    confidence_logits: torch.Tensor,
    accept_events: torch.Tensor,
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
    steps: int = 376,
) -> dict:
    """Left-to-right 1D grid fit minimizing cumulative prefix-survival ECE."""
    logits = confidence_logits.detach().double().cpu()
    events = accept_events.detach().double().cpu()
    prefix_labels = events.cumprod(dim=-1)
    temperatures = torch.ones(logits.shape[1], dtype=torch.double)
    raw_probabilities = logits.sigmoid()
    calibrated = raw_probabilities.clone()
    raw_ece = []
    calibrated_ece = []
    grid = torch.linspace(minimum, maximum, steps, dtype=torch.double)
    for position in range(logits.shape[1]):
        raw_prefix = raw_probabilities[:, : position + 1].prod(dim=-1)
        raw_ece.append(expected_calibration_error(raw_prefix, prefix_labels[:, position]))
        best_temperature = 1.0
        best_ece = math.inf
        fixed_prefix = (
            calibrated[:, :position].prod(dim=-1)
            if position > 0
            else torch.ones(logits.shape[0], dtype=torch.double)
        )
        for temperature in grid:
            candidate = fixed_prefix * torch.sigmoid(logits[:, position] / temperature)
            error = expected_calibration_error(candidate, prefix_labels[:, position])
            if error < best_ece:
                best_ece = error
                best_temperature = float(temperature)
        temperatures[position] = best_temperature
        calibrated[:, position] = torch.sigmoid(logits[:, position] / best_temperature)
        calibrated_ece.append(best_ece)
    calibrated_prefix = calibrated.cumprod(dim=-1)
    auc = [
        binary_auc(calibrated_prefix[:, position], prefix_labels[:, position])
        for position in range(logits.shape[1])
    ]
    return {
        "temperatures": temperatures.tolist(),
        "raw_cumulative_ece": raw_ece,
        "calibrated_cumulative_ece": calibrated_ece,
        "calibrated_cumulative_auc": auc,
        "raw_cumulative_ece_mean": float(torch.tensor(raw_ece).mean()),
        "calibrated_cumulative_ece_mean": float(torch.tensor(calibrated_ece).mean()),
        "calibrated_cumulative_auc_mean": float(
            torch.tensor([value for value in auc if math.isfinite(value)]).mean()
        ),
    }


def _conditional_rates(events: torch.Tensor) -> tuple[list[float], list[float]]:
    events = events.float()
    prefix = events.cumprod(dim=-1)
    rates = []
    alive = torch.ones(events.shape[0])
    for position in range(events.shape[1]):
        denominator = alive.sum().clamp_min(1.0)
        rates.append(float((alive * events[:, position].cpu()).sum() / denominator))
        alive = prefix[:, position].cpu()
    return rates, prefix.mean(dim=0).tolist()


def _analytical_rates(acceptance: torch.Tensor) -> tuple[list[float], list[float]]:
    acceptance = acceptance.float().cpu()
    prefix = acceptance.cumprod(dim=-1)
    rates = []
    alive = torch.ones(acceptance.shape[0])
    for position in range(acceptance.shape[1]):
        rates.append(float((alive * acceptance[:, position]).sum() / alive.sum().clamp_min(1e-9)))
        alive = prefix[:, position]
    return rates, prefix.mean(dim=0).tolist()


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    target_to_draft: torch.Tensor,
    *,
    device: torch.device,
    seed: int = 20260811,
) -> dict:
    model.eval()
    analytical = []
    sampled_events = []
    greedy_events = []
    confidence_logits = []
    loss_sums = {"loss": 0.0, "ce": 0.0, "tvd": 0.0, "confidence_bce": 0.0}
    examples = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                batch["hidden"], batch["tokens"], batch["anchor_position"], target_to_draft
            )
        loss = compute_dspark_loss(
            output.logits,
            output.confidence_logits,
            batch["tokens"][:, 1:],
            batch["top_ids"],
            batch["top_probs"],
            batch["tail_probs"],
            target_to_draft,
        )
        count = batch["tokens"].shape[0]
        metrics = loss.metrics()
        for key in loss_sums:
            loss_sums[key] += float(metrics[key]) * count
        examples += count
        analytical.append(loss.acceptance.cpu())
        teacher, _ = sparse_teacher_distribution(
            batch["top_ids"],
            batch["top_probs"],
            batch["tail_probs"],
            target_to_draft,
            model.config.draft_vocab,
        )
        anchor_draft = target_to_draft[batch["tokens"][:, 0].long()]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            proposals, corrected, recursive_confidence = model.recursive_logits_and_confidence(
                output.base_logits, output.hidden, anchor_draft, 0.7, generator
            )
            greedy, _, _ = model.recursive_logits_and_confidence(
                output.base_logits, output.hidden, anchor_draft, 0.0
            )
        q_probabilities = torch.softmax(corrected.float() / 0.7, dim=-1)
        q_selected = q_probabilities.gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
        p_selected = teacher.gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
        ratio = (p_selected / q_selected.clamp_min(1e-30)).clamp(max=1.0)
        uniform = torch.rand(ratio.shape, generator=generator, device=device)
        sampled_events.append(uniform.le(ratio).cpu())
        target_greedy = target_to_draft[batch["top_ids"][:, :, 0].long()]
        greedy_events.append((greedy == target_greedy).logical_and(target_greedy.ge(0)).cpu())
        confidence_logits.append(recursive_confidence.cpu())

    analytical_tensor = torch.cat(analytical)
    sampled_tensor = torch.cat(sampled_events)
    greedy_tensor = torch.cat(greedy_events)
    confidence_tensor = torch.cat(confidence_logits)
    analytical_conditional, analytical_prefix = _analytical_rates(analytical_tensor)
    sampled_conditional, sampled_prefix = _conditional_rates(sampled_tensor)
    greedy_conditional, greedy_prefix = _conditional_rates(greedy_tensor)
    calibration = fit_sequential_temperature_scaling(confidence_tensor, sampled_tensor)
    model.train()
    return {
        "examples": examples,
        "loss": {key: value / examples for key, value in loss_sums.items()},
        "temperature_0_7": {
            "analytical_conditional_acceptance": analytical_conditional,
            "analytical_prefix_survival": analytical_prefix,
            "q2_analytical": analytical_prefix[1],
            "sampled_conditional_acceptance": sampled_conditional,
            "sampled_prefix_survival": sampled_prefix,
            "q2_sampled": sampled_prefix[1],
            "seed": seed,
            "sparse_teacher_policy": "top64 exact; tail and out-of-trim mass are non-winnable escape",
        },
        "temperature_0_greedy": {
            "conditional_acceptance": greedy_conditional,
            "prefix_survival": greedy_prefix,
            "q2": greedy_prefix[1],
        },
        "confidence_sts": calibration,
    }
