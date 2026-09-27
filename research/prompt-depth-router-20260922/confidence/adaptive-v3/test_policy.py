"""Behavioral checks for live, evidence-derived C and its cost oracle."""

from pathlib import Path
import tempfile
import unittest

from costs import cost_table
from learn import eligible_labels, isotonic_blocks, loop_candidate, update
from oracle import best_fixed, hindsight_depths, pooled


def round_row(q, accepted, emitted, elapsed=14, number=1):
    return {
        "round": number, "q": q, "accepted": accepted,
        "drafted": len(q), "emitted": emitted,
        "elapsed_ns": elapsed, "eligible": True,
    }


class LearnedConfidenceTest(unittest.TestCase):
    def test_rejection_censors_later_labels(self):
        rows = [round_row([0.1, 0.8, 0.9], 0, 1)]
        self.assertEqual(eligible_labels(rows, 0), [(0.1, 0)])
        self.assertEqual(eligible_labels(rows, 1), [])
        self.assertEqual(eligible_labels(rows, 2), [])

    def test_cutoff_moves_from_observed_values_then_back_to_zero(self):
        costs = {1: 10, 2: 12, 3: 14}
        first = [
            round_row([0.1, 0.8, 0.8], 0, 1, number=i)
            for i in range(1, 11)
        ] + [
            round_row([0.9, 0.8, 0.8], 3, 4, number=i)
            for i in range(11, 21)
        ]
        state, decision = update(None, first, costs, "a" * 64, 1, False)
        self.assertEqual(state["cutoffs"], [0.5, 0.0])
        self.assertEqual(decision["eligible_rounds_added"], 20)
        second = [
            round_row([0.1, 0.8, 0.8], 3, 4, number=i)
            for i in range(1, 21)
        ]
        state, decision = update(state, second, costs, "a" * 64, 2, False)
        self.assertEqual(state["cutoffs"], [0.0, 0.0])
        self.assertEqual(decision["cutoffs_before"], [0.5, 0.0])
        self.assertFalse(decision["probe"])

    def test_monotone_fit_and_exact_loop_screen(self):
        blocks = isotonic_blocks([(0.1, 0), (0.2, 1), (0.3, 0), (0.4, 1)])
        rates = [block["success"] / block["count"] for block in blocks]
        self.assertEqual(rates, sorted(rates))
        self.assertIsNotNone(loop_candidate([1, 2] * 256))
        self.assertIsNone(loop_candidate(list(range(512))))

    def test_cost_oracle_prices_short_rejections(self):
        rows = [
            round_row([0.1, 0.8, 0.8], 0, 1),
            round_row([0.9, 0.8, 0.8], 3, 4),
        ]
        costs = {1: 10, 2: 12, 3: 14}
        upper, depths = hindsight_depths(rows, costs)
        self.assertEqual(depths, [1, 3])
        self.assertGreater(upper, pooled(rows, [3, 3], costs))
        fixed = best_fixed(rows, costs)
        self.assertEqual(fixed["cutoffs"][0], 0.9)
        self.assertEqual(fixed["depth_counts"], {"1": 1, "2": 0, "3": 1})

    def test_cost_table_reads_matched_native_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {}
            for k in (1, 2, 3):
                root = Path(directory) / f"k{k}"
                root.mkdir()
                (root / "turns.tsv").write_text(
                    "turn\toutput_tokens\n" + "".join(
                        f"{turn}\t128\n" for turn in range(1, 9)
                    )
                )
                (root / "rounds.tsv").write_text(
                    "turn\tround\tdraft_depth\temitted\telapsed_ns\teligible_for_learning\n"
                    + "".join(
                        f"{turn}\t1\t{k}\t1\t{10 + 2 * k}\ttrue\n"
                        for turn in range(1, 9)
                    )
                )
                for turn in range(1, 9):
                    (root / f"turn-{turn}.output.ids").write_text(
                        " ".join(str(i) for i in range(512))
                    )
                roots[k] = root
            report = cost_table(roots, "s" * 64, "m" * 64)
            self.assertEqual(
                report["round_ns"], {"1": 12.0, "2": 14.0, "3": 16.0}
            )
            self.assertEqual(report["excluded_turns"], [])

    def test_cost_table_matches_loop_exclusion_across_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = {k: [] for k in (1, 2, 3)}
            for session in range(2):
                for k in (1, 2, 3):
                    root = Path(directory) / f"session{session}-k{k}"
                    root.mkdir()
                    (root / "turns.tsv").write_text(
                        "turn\toutput_tokens\n" + "".join(
                            f"{turn}\t128\n" for turn in range(1, 9)
                        )
                    )
                    (root / "rounds.tsv").write_text(
                        "turn\tround\tdraft_depth\temitted\telapsed_ns\teligible_for_learning\n"
                        + "".join(
                            f"{turn}\t1\t{k}\t1\t{10 + 2 * k}\ttrue\n"
                            for turn in range(1, 9)
                        )
                    )
                    for turn in range(1, 9):
                        ids = (
                            [1, 2] * 256 if (session, k, turn) == (1, 2, 4)
                            else list(range(512))
                        )
                        (root / f"turn-{turn}.output.ids").write_text(
                            " ".join(map(str, ids))
                        )
                    roots[k].append(root)
            report = cost_table(roots, "s" * 64, "m" * 64)
            self.assertEqual(report["excluded_turns"], [{"session": 1, "turn": 4}])
            self.assertEqual(report["detail"]["1"]["rounds"], 15)
            self.assertEqual(report["round_ns"], {"1": 12.0, "2": 14.0, "3": 16.0})


if __name__ == "__main__":
    unittest.main()
