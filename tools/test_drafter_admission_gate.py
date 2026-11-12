#!/usr/bin/env python3
"""tools/drafter-admission-gate.py's verdicts on recorded steps, no GPU (memra#1173).

`verdicts()` is the gate's teeth: it must pass the fixed tree's steps, fail the unfixed tree's
on V2, V3, V4, V7 and V8, and fail when one arm was never taken, when a receipt's numbers
disagree with its arm, when a refused request speculated, and when a park replay speculated on
a plain charge. The MTP route keeps its eager arm and must show the capture's refusal.
Run: python3 tools/test_drafter_admission_gate.py
"""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "drafter_admission_gate", Path(__file__).resolve().parent / "drafter-admission-gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
MB = gate.MB
PLANE = 1197 * MB
REFS = {"spec_greedy": "G", "plain_greedy": "G", "spec_sampled": "S", "plain_sampled": "P"}
FULL = {"peak_pool_used": 2449 * MB, "pool_used_before": 0}


def row(step, kind, **over):
    r = {
        "step": step, "kind": kind, "status": 200, "completion_tokens": 48, "speculated": True, "sampled": False,
        "text": "G", "pool_used_before": 0, "peak_pool_used": 2449 * MB, "eager_lines": [], "refused_lines": [],
        "receipt": None, "declined_lines": [], "capture_skipped_lines": [], "fault_lines": [], "cost_lines": [],
        "last_cost_path": None, "served_plain_delta": 0, "drafter_refusals_delta": 0,
    }
    r.update(over)
    return r


def refused(step, kind, arm, available, sampled=False, **over):
    r = row(step, kind, speculated=False, sampled=sampled, text="P" if sampled else "G",
            peak_pool_used=1229 * MB, refused_lines=["r"], declined_lines=["d"], served_plain_delta=1,
            drafter_refusals_delta=1,
            receipt={"plane_mb": 1197, "available_mb": available, "with_mb": 5170, "without_mb": 3973, "arm": arm})
    r.update(over)
    return r


def fixed_rows():
    return [
        refused(1, "idle-window", "idle-fits", 4570),
        refused(2, "idle-window", "idle-fits", 4570, sampled=True),
        refused(3, "idle-band", "idle-short", 3823),
        refused(4, "busy-window", "busy-fits", 4570, text="other"),
        refused(5, "busy-band", "busy-unfundable", 3673, text="other"),
    ]


def unfixed_rows():
    return [
        row(1, "idle-window", eager_lines=["e"], drafter_refusals_delta=None),
        row(2, "idle-window", eager_lines=["e"], sampled=True, text="S", drafter_refusals_delta=None),
        row(3, "idle-band", status=400, speculated=False, completion_tokens=None, text=None,
            drafter_refusals_delta=None),
        row(4, "busy-window", eager_lines=["e"], drafter_refusals_delta=None),
        row(5, "busy-band", drafter_refusals_delta=None),
    ]


def replay(speculated, last_path):
    return row("replay", "replay", speculated=speculated, fault_lines=["f"],
               cost_lines=[["spec", 1816], [last_path, 1491]], last_cost_path=last_path)


def run(rows, log="", dflash=True, rep=None, replay_log=""):
    v = gate.verdicts(rows, PLANE, log, True, 200, REFS, FULL, dflash, rep, replay_log)
    return {k.split()[0]: ok for k, (ok, _) in v.items()}


class Verdicts(unittest.TestCase):
    def test_the_fixed_tree_passes_every_verdict(self):
        v = run(fixed_rows(), rep=replay(False, "plain"))
        self.assertEqual(v, {k: True for k in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")})

    def test_the_unfixed_tree_fails_the_charge_the_refusal_the_plane_the_arms_and_the_replay(self):
        v = run(unfixed_rows(), log=gate.EAGER * 3, rep=replay(True, "plain"))
        self.assertTrue(v["V1"] and v["V5"] and v["V6"])
        for k in ("V2", "V3", "V4", "V7", "V8"):
            self.assertFalse(v[k], k)

    def test_an_arm_never_taken_fails_v7(self):
        for arm in gate.ARMS:
            rows = [r for r in fixed_rows() if r["receipt"]["arm"] != arm]
            if arm == "idle-fits":
                continue  # that is V1's refusal: the window was not reached
            self.assertFalse(run(rows)["V7"], arm)
        self.assertFalse(run([r for r in fixed_rows() if r["receipt"]["arm"] != "idle-fits"])["V1"])

    def test_a_receipt_whose_numbers_disagree_with_its_arm_fails_v7(self):
        for i, available in ((0, 3900), (2, 4000), (3, 5200), (4, 4100)):
            rows = fixed_rows()
            rows[i]["receipt"]["available_mb"] = available
            self.assertFalse(run(rows)["V7"], rows[i]["receipt"])
        rows = fixed_rows()
        rows[3]["receipt"]["arm"] = "idle-fits"  # an idle arm on a busy step
        self.assertFalse(run(rows)["V7"])

    def test_a_refused_request_that_speculated_or_held_the_plane_fails(self):
        rows = fixed_rows()
        rows[0]["speculated"] = True
        self.assertFalse(run(rows)["V3"])
        rows = fixed_rows()
        rows[2]["peak_pool_used"] = 2449 * MB
        self.assertFalse(run(rows)["V4"])
        rows = fixed_rows()
        rows[1]["text"] = "S"  # a refused sampled request must write the plain stream
        self.assertFalse(run(rows)["V6"])
        rows = fixed_rows()
        rows[0]["text"] = "x"
        self.assertFalse(run(rows)["V6"])

    def test_the_replay_may_speculate_only_on_a_spec_charge(self):
        self.assertTrue(run(fixed_rows(), rep=replay(True, "spec"))["V8"])
        self.assertTrue(run(fixed_rows(), rep=replay(False, "plain"))["V8"])
        self.assertFalse(run(fixed_rows(), rep=replay(True, "plain"))["V8"])
        unfired = replay(False, "plain")
        unfired["fault_lines"] = []
        self.assertFalse(run(fixed_rows(), rep=unfired)["V8"])
        self.assertFalse(run(fixed_rows(), rep=replay(False, "plain"),
                             replay_log="CUDA_ERROR_OUT_OF_MEMORY (real)")["V8"])
        self.assertTrue(run(fixed_rows(), rep=replay(False, "plain"),
                            replay_log="CUDA_ERROR_OUT_OF_MEMORY (synthetic: MEMRA_STEP_OOM_FAULT)")["V8"])

    def test_a_bad_line_or_a_5xx_fails_v5(self):
        self.assertFalse(run(fixed_rows(), log="CUDA_ERROR_OUT_OF_MEMORY")["V5"])
        rows = fixed_rows()
        rows[4]["status"] = 503
        self.assertFalse(run(rows)["V5"])

    def test_the_mtp_route_keeps_its_eager_arm_and_must_show_the_capture_refusal(self):
        rows = [row(1, "idle-window", eager_lines=["e"], capture_skipped_lines=["c"]),
                row(2, "idle-window", eager_lines=["e"], capture_skipped_lines=["c"], sampled=True, text="S")]
        v = run(rows, log=gate.EAGER * 2, dflash=False)
        self.assertEqual(v, {"V1": True, "M2": True, "M3": True, "INFO": True, "V5": True, "V6": True})
        rows[1]["capture_skipped_lines"] = []
        self.assertFalse(run(rows, log=gate.EAGER * 2, dflash=False)["M2"])

    def test_receipt_parsing(self):
        line = ('[admission] drafter refused: model="gate" ctx=29222 draft plane 1197MB does not fit (available '
                '4967MB, required 5170MB with it, 3973MB without; arm idle-fits); the request is priced again as '
                'PLAIN, no drafter session is created')
        self.assertEqual(gate.parse_receipt(["x", line]),
                         {"plane_mb": 1197, "available_mb": 4967, "with_mb": 5170, "without_mb": 3973,
                          "arm": "idle-fits"})
        self.assertIsNone(gate.parse_receipt(["[admission] eager-arm admit: ..."]))


if __name__ == "__main__":
    unittest.main()
