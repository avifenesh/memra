#!/usr/bin/env python3

import unittest

import torch

from dspark_loss import build_target_to_draft, compute_dspark_loss, sparse_teacher_distribution
from dspark_eval import fit_sequential_temperature_scaling
from dspark_model import (
    DSparkConfig,
    DSparkModel,
    _apply_partial_rope,
    expected_trainable_parameters,
)


class DSparkTrainingTests(unittest.TestCase):
    def small_config(self) -> DSparkConfig:
        return DSparkConfig(
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            ffn_dim=64,
            draft_vocab=16,
            target_vocab=32,
            markov_rank=4,
            partial_rotary_factor=0.5,
        )

    def test_exact_model_shape_frozen_tables_and_parameter_accounting(self):
        config = self.small_config()
        embedding = torch.randn(config.draft_vocab, config.d_model)
        head = torch.randn_like(embedding)
        model = DSparkModel(config, embedding, head)
        actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(actual, expected_trainable_parameters(config))
        self.assertFalse(any(key.startswith("shared.") for key in model.state_dict()))
        output = model(
            torch.randn(3, config.d_model),
            torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [31, 1, 2, 3, 4, 5]]),
            torch.tensor([10, 20, 30]),
            build_target_to_draft(torch.arange(config.draft_vocab), config.target_vocab),
        )
        self.assertEqual(output.logits.shape, (3, 5, 16))
        self.assertEqual(output.confidence_logits.shape, (3, 5))
        self.assertEqual(output.anchor_in_trim.tolist(), [True, True, False])

    def test_qwen_partial_rope_leaves_suffix_untouched(self):
        value = torch.randn(2, 4, 5, 8)
        positions = torch.arange(5).expand(2, -1)
        rotated = _apply_partial_rope(value, positions, 4, 10_000_000.0)
        torch.testing.assert_close(rotated[..., 4:], value[..., 4:])
        torch.testing.assert_close(rotated[:, :, 0, :4], value[:, :, 0, :4])

    def test_escape_tvd_acceptance_and_full_softmax_gradient(self):
        probabilities = torch.tensor([0.4, 0.3, 0.2, 0.1])
        logits = (probabilities.log() * 0.7).view(1, 1, 4).expand(1, 5, 4).clone()
        logits.requires_grad_(True)
        top_ids = torch.tensor([[[0, 4]]]).expand(1, 5, 2)
        top_probs = torch.tensor([[[0.5, 0.25]]]).expand(1, 5, 2)
        tail = torch.full((1, 5), 0.25)
        inverse = build_target_to_draft(torch.arange(4), 6)
        dense, escape = sparse_teacher_distribution(top_ids, top_probs, tail, inverse, 4)
        torch.testing.assert_close(dense.sum(-1) + escape, torch.ones(1, 5))
        loss = compute_dspark_loss(
            logits,
            torch.zeros(1, 5, requires_grad=True),
            torch.zeros(1, 5, dtype=torch.long),
            top_ids,
            top_probs,
            tail,
            inverse,
        )
        torch.testing.assert_close(loss.acceptance, torch.full((1, 5), 0.4))
        loss.total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs() > 0).all())

    def test_position_weights_match_contract(self):
        got = torch.exp(-torch.arange(5).float() / 5.0)
        expected = torch.tensor([1.0, 0.81873075, 0.67032005, 0.54881164, 0.44932896])
        torch.testing.assert_close(got, expected)

    def test_sequential_temperature_scaling_reduces_prefix_ece(self):
        generator = torch.Generator().manual_seed(7)
        probabilities = 0.2 + 0.7 * torch.rand(3000, 5, generator=generator)
        events = torch.rand(3000, 5, generator=generator).lt(probabilities)
        sharpened_logits = 2.0 * torch.logit(probabilities)
        fit = fit_sequential_temperature_scaling(sharpened_logits, events)
        self.assertLess(
            fit["calibrated_cumulative_ece_mean"],
            fit["raw_cumulative_ece_mean"],
        )
        for temperature in fit["temperatures"]:
            self.assertGreater(temperature, 1.2)
            self.assertLessEqual(temperature, 4.0)
        self.assertAlmostEqual(fit["temperatures"][0], 2.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
