import unittest
from fit_profile import fit_profile


def matrix(values):
    return {kind: values for kind in ("prose", "code", "numeric")}


class ProfileFitTests(unittest.TestCase):
    def test_pooled_time_wins_over_average_of_rates(self):
        profile, detail = fit_profile(3, matrix({
            3: [(100, 2), (100, 2), (100, 2)],
            4: [(100, 1), (100, 1), (100, 10)],
        }), ["a", "b", "c"])
        self.assertEqual(detail["prose"][4]["confirmations"], 2)
        self.assertEqual(profile["prose"], 3)

    def test_one_conversation_cannot_promote_a_depth(self):
        profile, _ = fit_profile(3, matrix({
            3: [(10000, 10), (10, 1), (10, 1)],
            4: [(10000, 1), (10, 2), (10, 2)],
        }), ["a", "b", "c"])
        self.assertEqual(profile["numeric"], 3)

    def test_supported_class_change_preserves_fallback(self):
        profile, _ = fit_profile(3, matrix({
            2: [(100, 1), (100, 1), (100, 3)],
            3: [(100, 2), (100, 2), (100, 2)],
        }), ["a", "b", "c"])
        self.assertEqual(profile, {"prose": 2, "code": 2, "numeric": 2, "fallback": 3})

    def test_missing_or_duplicated_groups_are_not_confirmation(self):
        for ids, values in [
            (["a", "a"], {3: [(1, 1), (1, 1)]}),
            (["a", "b"], {3: [(1, 1)]}),
            (["a", "b"], {3: [(1, 1), (1, 1)], 4: [(0, 0), (1, 1)]}),
        ]:
            with self.assertRaises(ValueError):
                fit_profile(3, matrix(values), ids)


if __name__ == "__main__":
    unittest.main()
