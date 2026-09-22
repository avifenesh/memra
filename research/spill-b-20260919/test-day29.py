#!/usr/bin/env python3
"""CPU-only tests of the twin gate's V3-premise detector and typed refusal (spill-b day 29).

Replays A's day-18 local RTX 5090 receipts (`research/spill-a-20260919/rtx5090-day18/twin27-off/{calibration,measured}/
server.log` on origin/lane/spill-a-20260919) as verbatim `[admit-oom] reclaim-on-defer` lines through the gate's own
parser and premise rows, and pins the arithmetic behind the constant `-410352980` B V3 state error.
GATE=<path> overrides the module under test (default: tools/prefix-newest-turn-fits-gate.py of this tree).
"""
import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = Path(os.environ.get("GATE", ROOT / "tools" / "prefix-newest-turn-fits-gate.py"))
spec = importlib.util.spec_from_file_location("gate", GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

MIB = 1 << 20
# A's day-18 receipts, verbatim: the measured (cache-on) boot's reclaim lines by turn (turns 1..8).
MEASURED = {
    2: "[admit-oom] reclaim-on-defer: evicted 2 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3744MB -> 4651MB",
    3: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3603MB -> 4443MB",
    4: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 0 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3976MB -> 4414MB",
    5: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 0 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3938MB -> 4385MB",
    6: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3899MB -> 4983MB",
    7: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3861MB -> 4965MB",
    8: "[admit-oom] reclaim-on-defer: evicted 1 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 3824MB -> 4944MB",
}
# The calibration (cache-off) boot's, same receipts.
CALIBRATION = {
    3: "[admit-oom] reclaim-on-defer: evicted 0 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 4060MB -> 4471MB",
    8: "[admit-oom] reclaim-on-defer: evicted 0 prefix entries + 1 plain + 0 spec + 0 dspark parked sessions (global LRU); effective free 4370MB -> 4780MB",
}
# The settle line beside turn 2's reclaim: the prefix bytes it returned.
TURN2_EVICTED_PREFIX_BYTES = 497188864
V3_ERROR_DAY18 = -410352980  # turns 2..7 of rtx5090-day18/twin27-off/TURNS.md; 0 on turns 1 and 8


def boots(measured: dict, calibration: dict, turns: int = 8, cohort=(2800, 3000, 3200)):
    """cal/rec dicts of the shape the gate hands v3_premise_rows, with windows parsed from the lines."""
    empty = gate.parse_window([])
    cal = {"cohort": [{"tokens": n, "windows": [gate.parse_window([]), gate.parse_window([])]} for n in cohort],
           "turns": [{"turn": k, "window": gate.parse_window([calibration[k]] if k in calibration else [])} for k in range(1, turns + 1)]}
    rec = {"cohort": [{"tokens": n, "window_first": gate.parse_window([]), "window_second": gate.parse_window([])} for n in cohort],
           "turns": [{"turn": k, "window": gate.parse_window([measured[k]] if k in measured else [])} for k in range(1, turns + 1)],
           "boot": {"budget_bytes": 1073741824}}
    assert empty["parked_releases"] == []
    return cal, rec


class ParseReclaimLine(unittest.TestCase):
    def test_turn2_line_is_parsed_with_its_counts_and_free_move(self):
        w = gate.parse_window([MEASURED[2]])
        self.assertEqual(w["reclaims"], 1)
        self.assertEqual(len(w["parked_releases"]), 1)
        r = w["parked_releases"][0]
        self.assertEqual((r["prefix"], r["plain"], r["spec"], r["dspark"]), (2, 1, 0, 0))
        self.assertEqual((r["effective_free_before_mb"], r["effective_free_after_mb"]), (3744, 4651))
        self.assertEqual(gate.parked_released(w), (1, 0, 0))

    def test_a_window_without_reclaim_releases_nothing(self):
        w = gate.parse_window(["[prefix-cache] hit: 9184 of 9500 prompt tokens from cache (model gate)"])
        self.assertEqual(w["reclaims"], 0)
        self.assertEqual(gate.parked_released(w), (0, 0, 0))

    def test_prefix_only_reclaim_releases_no_parked_session(self):
        self.assertEqual(gate.parked_released(gate.parse_window([MEASURED[4]])), (0, 0, 0))


class PremiseRows(unittest.TestCase):
    def test_day18_receipts_break_the_premise_on_turns_2_6_7_only(self):
        cal, rec = boots(MEASURED, CALIBRATION)
        rows = gate.v3_premise_rows(cal, rec)
        self.assertEqual(len(rows), 6 + 8)  # 3 cohort lengths x 2 sends, then 8 turns
        broken = [r["who"] for r in rows if not r["equal"]]
        self.assertEqual(broken, ["turn 2", "turn 6", "turn 7"])
        # turn 3 and turn 8 released one plain session in BOTH boots: the premise holds there.
        by_who = {r["who"]: r for r in rows}
        self.assertEqual((by_who["turn 3"]["calibration"], by_who["turn 3"]["measured"]), ((1, 0, 0), (1, 0, 0)))
        self.assertEqual((by_who["turn 8"]["calibration"], by_who["turn 8"]["measured"]), ((1, 0, 0), (1, 0, 0)))

    def test_day17_receipts_no_reclaim_in_either_boot_hold_the_premise(self):
        cal, rec = boots({}, {})
        self.assertTrue(all(r["equal"] for r in gate.v3_premise_rows(cal, rec)))

    def test_identical_reclaims_in_both_boots_hold_the_premise(self):
        cal, rec = boots(CALIBRATION, CALIBRATION)
        self.assertTrue(all(r["equal"] for r in gate.v3_premise_rows(cal, rec)))

    def test_a_release_in_the_calibration_boot_alone_breaks_it_too(self):
        cal, rec = boots({}, CALIBRATION)
        self.assertEqual([r["who"] for r in gate.v3_premise_rows(cal, rec) if not r["equal"]], ["turn 3", "turn 8"])

    def test_a_cohort_send_release_is_a_window_of_its_own(self):
        cal, rec = boots({}, {})
        rec["cohort"][1]["window_second"] = gate.parse_window([CALIBRATION[3]])
        self.assertEqual([r["who"] for r in gate.v3_premise_rows(cal, rec) if not r["equal"]], ["cohort 3000 send 2"])


class Refusal(unittest.TestCase):
    def test_refusal_is_typed_and_names_the_windows_and_the_card(self):
        cal, rec = boots(MEASURED, CALIBRATION)
        rows = gate.v3_premise_rows(cal, rec)
        cal_card = {"at": "t0", "memory_free_total_mib": "22572, 24463", "compute_apps": ["4125675, /x/colbert-2/.venv/bin/python, 1390"]}
        meas_card = {"at": "t1", "memory_free_total_mib": "22572, 24463", "compute_apps": []}
        text = gate.v3_premise_refusal(rows, cal_card, meas_card, 1073741824)
        self.assertTrue(text.startswith("V3 premise: "))
        for who in ("turn 2:", "turn 6:", "turn 7:"):
            self.assertIn(who, text)
        self.assertNotIn("turn 3:", text)
        self.assertIn("3 window(s)", text)
        self.assertIn("python 1390 MiB", text)
        self.assertIn("compute-apps 0 []", text)
        self.assertIn("budget 1073741824 B", text)
        self.assertNotIn("PASS", text)
        self.assertNotIn("/x/colbert-2", text)  # the brief carries basenames; the listing itself is in summary.json

    def test_card_brief_without_a_sample(self):
        self.assertEqual(gate.card_brief(None), "card not sampled")


class Arithmetic(unittest.TestCase):
    def test_the_constant_is_one_cohort_shaped_parked_plain_session(self):
        # ctx_cap 3272 (3200 prompt + 72) x 29696 B/token, plus the admission line's 313MB fixed for that ctx.
        kv = 3272 * 29696
        self.assertEqual(kv, 97165312)
        fixed = -V3_ERROR_DAY18 - kv
        self.assertEqual(fixed, 313187668)
        self.assertEqual(round(fixed / 1e6), 313)

    def test_turn2_reclaim_free_move_minus_its_prefix_bytes_is_the_constant_within_the_gate_slack(self):
        r = gate.parse_window([MEASURED[2]])["parked_releases"][0]
        moved = (r["effective_free_after_mb"] - r["effective_free_before_mb"]) * 1_000_000
        plain_released = moved - TURN2_EVICTED_PREFIX_BYTES
        self.assertLessEqual(abs(plain_released - (-V3_ERROR_DAY18)), gate.V3_SLACK)
        self.assertLess(abs(plain_released - (-V3_ERROR_DAY18)), 1_000_000)  # the MB rounding of the log line

    def test_v3_slack_unchanged(self):
        self.assertEqual(gate.V3_SLACK, 64 * MIB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
