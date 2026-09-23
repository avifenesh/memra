"""Exact two-token laws for post-offer versus chosen-token censoring."""

import unittest


def one_step_law(target, proposal, *, cutoff, discard_chosen):
    residual = [max(0.0, p - q) for p, q in zip(target, proposal)]
    total = sum(residual)
    if total:
        residual = [value / total for value in residual]
    output = [0.0] * len(target)
    for pick, q in enumerate(proposal):
        if discard_chosen and q < cutoff:
            for token, p in enumerate(target):
                output[token] += q * p
            continue
        accept = min(1.0, target[pick] / q)
        output[pick] += q * accept
        if not accept:
            for token, p in enumerate(residual):
                output[token] += q * (1 - accept) * p
    return output


class SampledCutoffLawTest(unittest.TestCase):
    def test_discarding_chosen_token_changes_target_law(self):
        target = proposal = [0.8, 0.2]
        old = one_step_law(target, proposal, cutoff=0.5, discard_chosen=True)
        fixed = one_step_law(target, proposal, cutoff=0.5, discard_chosen=False)
        self.assertAlmostEqual(old[0], 0.96)
        self.assertAlmostEqual(old[1], 0.04)
        for observed, expected in zip(fixed, target):
            self.assertAlmostEqual(observed, expected)

    def test_post_offer_correction_preserves_mismatched_target(self):
        target, proposal = [0.5, 0.5], [0.9, 0.1]
        old = one_step_law(target, proposal, cutoff=0.7, discard_chosen=True)
        fixed = one_step_law(target, proposal, cutoff=0.7, discard_chosen=False)
        self.assertAlmostEqual(old[0], 0.55)
        self.assertAlmostEqual(old[1], 0.45)
        for observed, expected in zip(fixed, target):
            self.assertAlmostEqual(observed, expected)

    def test_pre_draw_zero_offer_is_separately_exact(self):
        target, proposal = [0.35, 0.65], [0.8, 0.2]
        offered = one_step_law(target, proposal, cutoff=0.0, discard_chosen=False)
        zero_offer = target
        for mix in (0.0, 0.25, 0.5, 1.0):
            output = [
                mix * no_offer + (1 - mix) * offer
                for no_offer, offer in zip(zero_offer, offered)
            ]
            for observed, expected in zip(output, target):
                self.assertAlmostEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
