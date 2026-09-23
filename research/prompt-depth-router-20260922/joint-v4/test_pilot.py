"""Checks the two failure boundaries in the observational pilot."""

import unittest

from pilot import features, make_labels


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


if __name__ == "__main__":
    unittest.main()
