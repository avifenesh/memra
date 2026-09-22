import unittest
from collections import Counter
from scored_sets import matched_report


ARMS = ("calibrated", "native", "context", "learned", "measured")


def counts(groups, planned):
    return {
        "sets": len(groups),
        "runs": sum(len(g["records"]) for g in groups),
        "policy_counts": dict(Counter(r["arm"] for g in groups for r in g["records"])),
        "complete_planned_matrix": len(groups) == planned,
    }


class MatchedExclusionTests(unittest.TestCase):
    def setUp(self):
        self.groups = [
            {"cycle": i, "records": [{"arm": arm, "turns": 8} for arm in ARMS]}
            for i in range(10)
        ]
        self.flag = {"4": [{"arm": "calibrated", "turn": 8, "period": 3}]}

    def test_one_bad_policy_excludes_all_five_policies_in_its_set(self):
        report = matched_report(self.groups, self.flag, self.flag, counts, True)
        self.assertEqual(report["policy_counts"], dict.fromkeys(ARMS, 9))
        self.assertEqual(report["runs"], 45)
        self.assertEqual(report["recorded_runs"], 50)
        self.assertEqual(report["excluded_cycles"], [4])
        self.assertFalse(report["complete_planned_matrix"])
        self.assertTrue(report["original_schedule_completed"])
        self.assertEqual(len(self.groups[4]["records"]), 5)

    def test_an_unflagged_set_cannot_be_removed(self):
        with self.assertRaisesRegex(ValueError, "observed repetition"):
            matched_report(self.groups, self.flag, {}, counts, True)

    def test_a_flagged_set_cannot_enter_the_scores(self):
        with self.assertRaisesRegex(ValueError, "observed repetition"):
            matched_report(self.groups, {}, self.flag, counts, True)

    def test_an_excluded_set_cannot_hide_a_missing_control(self):
        self.groups[4]["records"].pop()
        with self.assertRaisesRegex(ValueError, "all five eight-turn policies"):
            matched_report(self.groups, self.flag, self.flag, counts, True)


if __name__ == "__main__":
    unittest.main()
