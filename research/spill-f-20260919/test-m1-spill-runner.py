#!/usr/bin/env python3
"""OWED 12 CPU gates for m1-spill-runner.py, driven by m1-stub-run-gen.py (no GPU, no lock).

Covers: a full registered dry run (10 rounds x 6 arms, cold regime), order and pairing, the
verdict rule including sign-agreement and contamination edges, identity-mismatch refusal before
any visit, a corrupted-token arm and a misreported direct over-read refused as arms, raw-log-first
reparse (and its red control on a tampered log), stub-only lock bypass, and one bounded-regime
run with a small balloon. The scratch filesystem is the repo's own; the synthetic PRIVATE proof
carries its live identity, like the real receipt the lead's box will produce.
"""
import collections
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())
RUNNER = HERE / "m1-spill-runner.py"
STUB = HERE / "m1-stub-run-gen.py"
spec = importlib.util.spec_from_file_location("runner", RUNNER)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
LOCK = HERE / "m1-prereg/b3-arms.lock.json"
TPS = {"worker16": 10.0, "mmap-random": 9.0, "mmap-normal": 10.2, "pread16": 8.0,
       "worker2": 11.0, "direct16": 12.0}


def device_of(path):
    """Top block device name and its leaves, from sysfs (the stub proof's block graph)."""
    st = os.stat(path)
    top = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
    leaves, node = [], top
    if (node / "partition").exists():
        node = node.parent
    slaves = sorted((node / "slaves").glob("*")) if (node / "slaves").is_dir() else []
    leaves = [s.resolve().name for s in slaves] or [node.name]
    return top.name, leaves


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".m1-runner-test-"))
        self.scratch = self.tmp / "scratch"
        self.scratch.mkdir()
        self.artifact = self.scratch / "artifact.gguf"
        self.artifact.write_bytes(os.urandom(4 << 20))
        top, leaves = device_of(self.scratch)
        identity = R.B.filesystem_identity(self.scratch)
        self.proof = self.tmp / "proof.private.json"
        self.proof.write_text(json.dumps({
            "schema": "m1-nvme-proof-v1", "verdict": "PASS", "class": "nvme-local-direct", "reasons": [],
            "tool_sha256": "stub", "A8_identity": identity,
            "A4_block_graph": {"nodes": [{"name": top}], "leaves": [{"name": n} for n in leaves]}}))
        self.stub = self.tmp / "m1-stub-run-gen"
        self.stub.write_text(f"#!/bin/sh\nexec {sys.executable} {STUB} \"$@\"\n")
        self.stub.chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_runner(self, name, *extra, env=None, regime="cold", rounds="10", arms=None):
        out = self.tmp / name
        argv = [sys.executable, str(RUNNER), "run", "--arms-lock", str(LOCK), "--regime", regime,
                "--binary", str(self.stub), "--artifact", str(self.artifact), "--proof", str(self.proof),
                "--out", str(out), "--rounds", rounds, "--stub-no-lock", "--contamination-limit", "1.0",
                *(["--arms", arms] if arms else []), *extra]
        e = dict(os.environ, M1_STUB_TPS=json.dumps(TPS), **(env or {}))
        proc = subprocess.run(argv, capture_output=True, text=True, env=e)
        return proc, out


class DryRun(Harness):
    def test_full_registered_dry_run(self):
        proc, out = self.run_runner("full")
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        visits = [json.loads(p.read_text()) for p in sorted(out.glob("*/visit.json"))]
        self.assertEqual(len(visits), 60)
        names = [a["name"] for a in json.loads(LOCK.read_text())["arms"]]
        for r in range(10):
            order = [v["arm"] for v in sorted((v for v in visits if v["round"] == r + 1),
                                               key=lambda v: v["position"])]
            self.assertEqual(order, names if r % 2 == 0 else names[::-1])
        meets = collections.Counter()
        for r in range(10):
            order = names if r % 2 == 0 else names[::-1]
            for i, a in enumerate(order):
                for b in order[i + 1:]:
                    meets[(a, b)] += 1
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertEqual((meets[(a, b)], meets[(b, a)]), (5, 5))
        self.assertTrue(all(v["scored"] for v in visits), [v["correctness_problems"] for v in visits])
        summary = json.loads((out / "summary.json").read_text())
        verdicts = {a: s["verdict"] for a, s in summary["arms"].items()}
        self.assertEqual(verdicts, {"mmap-random": "loser", "mmap-normal": "flat", "pread16": "loser",
                                    "worker2": "winner", "direct16": "winner"})
        self.assertEqual(summary["refused_arms"], [])
        self.assertTrue(all(v["proc_io"]["read_bytes"] >= (1 << 20) for v in visits),
                        [v["proc_io"] for v in visits][:3])
        self.assertTrue(all(v["telemetry_ok"] for v in visits))
        reparse = subprocess.run([sys.executable, str(RUNNER), "reparse", str(out)], capture_output=True, text=True)
        self.assertEqual(reparse.returncode, 0, reparse.stdout)
        victim = next(out.glob("*/run.log"))
        victim.write_text(victim.read_text().replace("MATCH", "MISMATCH"))
        red = subprocess.run([sys.executable, str(RUNNER), "reparse", str(out)], capture_output=True, text=True)
        self.assertEqual(red.returncode, 3)
        self.assertIn("REPARSE MISMATCH", red.stdout)

    def test_corrupt_arm_and_bad_overread_are_refused(self):
        proc, out = self.run_runner("refuse", env={"M1_STUB_CORRUPT": "worker2",
                                                   "M1_STUB_BAD_OVERREAD": "direct16"})
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["refused_arms"], ["direct16", "worker2"])
        self.assertNotIn("worker2", summary["arms"])
        problems = json.loads(next(out.glob("*-direct16/visit.json")).read_text())["correctness_problems"]
        self.assertTrue(any("overread_bytes" in p for p in problems), problems)

    def test_identity_mismatch_refuses_before_any_visit(self):
        proof = json.loads(self.proof.read_text())
        proof["A8_identity"]["mount_id"] += 1
        self.proof.write_text(json.dumps(proof))
        proc, out = self.run_runner("identity")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not on the proven filesystem", proc.stderr)
        self.assertFalse(any(out.glob("r*")))

    def test_stub_bypass_refuses_a_real_binary_name(self):
        real = self.tmp / "run-gen"
        shutil.copy(self.stub, real)
        out = self.tmp / "real"
        proc = subprocess.run([sys.executable, str(RUNNER), "run", "--arms-lock", str(LOCK), "--regime",
                               "cold", "--binary", str(real), "--artifact", str(self.artifact), "--proof",
                               str(self.proof), "--out", str(out), "--stub-no-lock"],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("only for the stub binary", proc.stderr)
        missing = subprocess.run([sys.executable, str(RUNNER), "run", "--arms-lock", str(LOCK), "--regime",
                                  "cold", "--binary", str(real), "--artifact", str(self.artifact), "--proof",
                                  str(self.proof), "--out", str(self.tmp / "nolock")],
                                 capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("--lock-fd", missing.stderr)

    def test_bounded_regime_holds_and_releases_a_balloon(self):
        proc, out = self.run_runner("bounded", "--balloon-bytes", str(4 << 20), "--floor-bytes", str(1 << 30),
                                    regime="bounded", rounds="2", arms="worker16,mmap-random")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        log = (out / "balloon.log").read_text()
        self.assertIn('"state": "LOCKED"', log)
        self.assertIn('"state": "RELEASED"', log)
        visits = [json.loads(p.read_text()) for p in sorted(out.glob("*/visit.json"))]
        self.assertEqual(len(visits), 4)
        self.assertTrue(all(v["regime_ok"] for v in visits))


class VerdictRule(unittest.TestCase):
    def visits(self, ratios_fwd, ratios_rev, dirty=0):
        vs = []
        for i, (f, r) in enumerate(zip(ratios_fwd, ratios_rev)):
            for rnd, order, ratio in ((2 * i + 1, "forward", f), (2 * i + 2, "reverse", r)):
                vs.append({"arm": "base", "round": rnd, "order": order, "scored": True, "clean_timing": True, "tok_s": 10.0})
                vs.append({"arm": "x", "round": rnd, "order": order, "scored": True, "clean_timing": True, "tok_s": 10.0 * ratio})
        for v in vs[:dirty]:
            v["clean_timing"] = v["scored"] = False
        return vs

    def verdict(self, *a, **k):
        return R.verdicts(self.visits(*a, **k), ["base", "x"], "base")["arms"]["x"]["verdict"]

    def test_rule(self):
        self.assertEqual(self.verdict([1.1] * 5, [1.1] * 5), "winner")
        self.assertEqual(self.verdict([1.1, 1.1, 1.1, 0.99, 0.99], [1.1] * 5), "flat")  # 3/5 forward
        self.assertEqual(self.verdict([1.1, 1.1, 1.1, 1.1, 0.99], [1.1] * 5), "winner")  # 4/5 each
        self.assertEqual(self.verdict([0.9] * 5, [0.9] * 5), "loser")
        self.assertEqual(self.verdict([1.03] * 5, [1.03] * 5), "flat")
        self.assertEqual(self.verdict([1.1] * 3, [1.1] * 3), "insufficient")

    def test_contamination_edges(self):
        # two dirty baseline visits: scored regime, but the pair rounds shrink
        self.assertEqual(self.verdict([1.1] * 5, [1.1] * 5, dirty=1), "winner")
        v = self.visits([1.1] * 5, [1.1] * 5)
        for x in [x for x in v if x["arm"] == "x"][:3]:
            x["clean_timing"] = x["scored"] = False
        self.assertEqual(R.verdicts(v, ["base", "x"], "base")["arms"]["x"]["verdict"], "unscored-regime")


if __name__ == "__main__":
    unittest.main(verbosity=2)
