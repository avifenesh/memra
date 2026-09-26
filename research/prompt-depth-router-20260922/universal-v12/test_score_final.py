"""Keep the final claim dependent on every domain and every action."""

import copy
import unittest

import score_final


def fixture():
    return {
        "pooled_comparisons": {
            "vs_own_noop": {
                "status": "paired",
                "bootstrap_95_percent": [0.1, 3.0],
            },
            "vs_global_fixed": {
                "status": "paired",
                "bootstrap_95_percent": [0.1, 3.0],
            },
        },
        "domains": {
            domain: {
                "vs_validation_best_fixed": {
                    "status": "paired",
                    "bootstrap_95_percent": [0.0, 3.0],
                },
                "vs_own_noop": {
                    "status": "paired",
                    "bootstrap_95_percent": [0.0, 3.0],
                },
                "vs_global_fixed": {
                    "status": "paired",
                    "bootstrap_95_percent": [0.0, 3.0],
                },
                "quality": {
                    "fixed-control": {"eligible": True},
                },
            }
            for domain in score_final.DOMAINS
        },
        "observed_behavior": {
            "adaptive_k_observed": True,
            "adaptive_d_observed": True,
            "adaptive_c_observed": True,
        },
    }


class FinalPolicyTest(unittest.TestCase):
    def test_every_domain_and_action_is_required(self):
        good = fixture()
        self.assertEqual(
            score_final.decide(good), "bounded-one-policy-win",
        )
        prose_loss = copy.deepcopy(good)
        prose_loss["domains"]["prose"]["quality"][
            "fixed-control"
        ]["eligible"] = False
        self.assertEqual(
            score_final.decide(prose_loss), "global-no-go",
        )
        no_k_switch = copy.deepcopy(good)
        no_k_switch["observed_behavior"][
            "adaptive_k_observed"
        ] = False
        self.assertEqual(
            score_final.decide(no_k_switch),
            "bounded-one-policy-win",
        )
        self.assertEqual(
            score_final.adaptive_components(
                no_k_switch["observed_behavior"]
            ),
            {
                "k": False, "d": True,
                "c": True, "all_three": False,
            },
        )
        code_rate_loss = copy.deepcopy(good)
        code_rate_loss["domains"]["code"][
            "vs_validation_best_fixed"
        ]["bootstrap_95_percent"][0] = -0.01
        self.assertEqual(
            score_final.decide(code_rate_loss), "global-no-go",
        )
        prose_noop_loss = copy.deepcopy(good)
        prose_noop_loss["domains"]["prose"][
            "vs_own_noop"
        ]["bootstrap_95_percent"][0] = -0.01
        self.assertEqual(
            score_final.decide(prose_noop_loss), "global-no-go",
        )
        math_global_loss = copy.deepcopy(good)
        math_global_loss["domains"]["math"][
            "vs_global_fixed"
        ]["bootstrap_95_percent"][0] = -0.01
        self.assertEqual(
            score_final.decide(math_global_loss), "global-no-go",
        )


if __name__ == "__main__":
    unittest.main()
