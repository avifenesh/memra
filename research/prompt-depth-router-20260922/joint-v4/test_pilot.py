"""Checks the two failure boundaries in the observational pilot."""

import unittest
import json
from pathlib import Path

from pilot import features, make_labels
from quality import CASES
from static_oracle import optimal_actions, pooled


class PilotBoundaryTest(unittest.TestCase):
    def test_rejection_never_labels_a_later_proposal(self):
        offered = [0.1, 0.8, 0.9]
        self.assertEqual(make_labels(offered, 0, True), [(0, 0.1, 0)])
        self.assertEqual(
            make_labels(offered, 1, True),
            [(0, 0.1, 1), (1, 0.8, 0)],
        )
        self.assertEqual(make_labels(offered, 3, False), [])

    def test_prior_round_feedback_enters_only_its_ablation(self):
        row = {
            "q": 0.6, "position": 1, "last": 7,
            "history": [2, 3, 7], "prior_accept": 1 / 3,
            "prior_ms": 14,
        }
        other = {**row, "prior_accept": 1.0, "prior_ms": 22}
        for arm in ("q", "token", "history"):
            self.assertEqual(features(row, arm, {}, [7]), features(other, arm, {}, [7]))
        self.assertNotEqual(
            features(row, "prior_round", {}, [7]),
            features(other, "prior_round", {}, [7]),
        )

    def test_best_per_input_rate_can_reduce_pooled_throughput(self):
        conversations = [
            {"controls": {
                "A": {"tokens": 100, "seconds": 1.0},
                "B": {"tokens": 200, "seconds": 2.1},
            }},
            {"controls": {
                "A": {"tokens": 1, "seconds": 1.0},
                "B": {"tokens": 5, "seconds": 2.0},
            }},
        ]
        self.assertEqual(optimal_actions(conversations, ("A", "B")), ("B", "A"))
        self.assertGreater(
            pooled(conversations, ("B", "A")),
            pooled(conversations, ("A", "B")),
        )

    def test_fresh_code_requests_all_have_functional_probes(self):
        manifest = json.loads(
            (Path(__file__).with_name("workloads-v4") / "manifest.json").read_text()
        )
        names = {
            turn["function"]
            for group in manifest["groups"].values()
            for item in group
            for turn in item["turns"]
        }
        self.assertEqual(set(CASES), names)


if __name__ == "__main__":
    unittest.main()
