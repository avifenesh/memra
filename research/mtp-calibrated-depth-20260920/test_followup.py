import collections
import unittest

from controller import ARMS, schedule, select_depth


class FollowupTests(unittest.TestCase):
    def test_five_arm_schedule_balances_positions_and_predecessors(self):
        rows = schedule()
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(set(row), set(ARMS))
        for position in range(5):
            self.assertEqual(set(collections.Counter(row[position] for row in rows).values()), {2})
        pairs = collections.Counter((row[i], row[i + 1]) for row in rows for i in range(4))
        self.assertEqual(len(pairs), 20)
        self.assertEqual(set(pairs.values()), {2})

    def test_calibration_uses_pooled_throughput_and_complete_depth_sets(self):
        def row(k, tokens, seconds):
            return dict(arm=f'k{k}', tokens=tokens, elapsed_s=seconds, correctness_only=False)
        groups = [[row(1, 1200, 60), row(2, 660, 60)], [row(1, 6000, 600), row(2, 6600, 600)]]
        winner, rates = select_depth(groups, 2)
        self.assertEqual(winner, 2)
        self.assertAlmostEqual(rates[1], 7200 / 660)
        with self.assertRaises(AssertionError):
            select_depth([groups[0][:-1]], 2)
        with self.assertRaises(AssertionError):
            select_depth([[row(1, 1, 1)]], 1)

    def test_exact_calibration_tie_prefers_smaller_depth(self):
        groups = [[dict(arm=f'k{k}', tokens=600, elapsed_s=60, correctness_only=False) for k in (1, 2)]]
        self.assertEqual(select_depth(groups, 2)[0], 1)


if __name__ == '__main__':
    unittest.main()
