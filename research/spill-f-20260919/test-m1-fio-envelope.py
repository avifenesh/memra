#!/usr/bin/env python3
"""B0 runner CPU gates: plan shape, the B5 screen rule, and an end-to-end run with a stub fio."""
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
spec = importlib.util.spec_from_file_location("fio_env", HERE / "m1-fio-envelope.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class Plan(unittest.TestCase):
    def test_shape(self):
        steps = F.plan(Path("/x/f"))
        self.assertEqual(len(steps), 88)
        self.assertEqual(steps[0]["name"], "prepare")
        grid = [s for s in steps if s["name"].startswith("grid-")]
        self.assertEqual(len(grid), 45)
        for cell in range(15):
            engines = [s["engine"] for s in grid[3 * cell:3 * cell + 3]]
            self.assertEqual(engines, F.ENGINES if cell % 2 == 0 else F.ENGINES[::-1])
        screen = [s for s in steps if s["name"].startswith("screen-")]
        self.assertEqual(len(screen), 40)
        psync16 = next(s for s in screen if s["engine"] == "psync" and s["depth"] == 16)
        self.assertIn("--numjobs=16", psync16["argv"])
        self.assertIn("--iodepth=1", psync16["argv"])
        uring16 = next(s for s in screen if s["engine"] == "io_uring" and s["depth"] == 16)
        self.assertIn("--iodepth=16", uring16["argv"])
        self.assertIn("--numjobs=1", uring16["argv"])
        self.assertIn("--direct=0", steps[-1]["argv"])
        self.assertNotIn("--direct=1", steps[-1]["argv"])


def rows(ab, ba, cpu=1.0, depth_values=(2, 16)):
    out = {}
    for depth in depth_values:
        for rnd in range(1, 6):
            for order, ratio in (("AB", ab[rnd - 1]), ("BA", ba[rnd - 1])):
                out[f"screen-d{depth}-r{rnd}-{order}-psync"] = {"bw_bytes": 1e9, "cpu_s_per_gib": 1.0}
                out[f"screen-d{depth}-r{rnd}-{order}-io_uring"] = {"bw_bytes": 1e9 * ratio, "cpu_s_per_gib": cpu}
    return out


class Screen(unittest.TestCase):
    def v(self, *a, **k):
        return F.screen_verdict(rows(*a, **k))["depth16"]["verdict"]

    def test_rule(self):
        self.assertEqual(self.v([1.1] * 5, [1.1] * 5), "io_uring-justified")
        self.assertEqual(self.v([1.1, 1.1, 1.1, 0.99, 0.99], [1.1] * 5), "io_uring-not-justified")
        self.assertEqual(self.v([1.0] * 5, [0.99] * 5, cpu=0.85), "io_uring-justified")
        self.assertEqual(self.v([1.0] * 5, [1.0] * 5, cpu=0.95), "io_uring-not-justified")
        self.assertEqual(self.v([0.95] * 5, [0.95] * 5, cpu=0.5), "io_uring-not-justified")
        partial = rows([1.1] * 5, [1.1] * 5)
        del partial["screen-d16-r3-BA-psync"]
        self.assertEqual(F.screen_verdict(partial)["depth16"]["verdict"], "incomplete")


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".m1-fio-test-"))
        self.dir = self.tmp / "scratch"
        self.dir.mkdir()
        st = os.stat(self.dir)
        top = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        leaf = top.parent.name if (top / "partition").exists() else top.name
        self.proof = self.tmp / "proof.json"
        self.proof.write_text(json.dumps({"verdict": "PASS", "class": "nvme-local-direct",
                                          "A8_identity": F.B.filesystem_identity(self.dir),
                                          "A4_block_graph": {"nodes": [{"name": top.name}], "leaves": [{"name": leaf}]}}))
        self.fio = self.tmp / "m1-stub-fio"
        self.fio.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'm1-stub-fio.py'} \"$@\"\n")
        self.fio.chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_env(self, name, fio=None, proof=None):
        argv = [sys.executable, str(HERE / "m1-fio-envelope.py"), "run", "--dir", str(self.dir),
                "--proof", str(proof or self.proof), "--out", str(self.tmp / name), "--fio", str(fio or self.fio),
                "--size", "64M", "--stub-no-lock"]
        env = dict(os.environ, M1_STUB_FIO_BW=json.dumps({"psync": 1_000_000_000, "io_uring": 1_120_000_000,
                                                          "libaio": 1_050_000_000}))
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600)

    def test_run(self):
        p = self.run_env("ok")
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-2000:])
        out = self.tmp / "ok"
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["steps"], 88)
        self.assertEqual({k: v["verdict"] for k, v in summary["screen"].items()},
                         {"depth2": "io_uring-justified", "depth16": "io_uring-justified"})
        self.assertEqual(summary["sustained_read_bytes_per_s_max"], 1_120_000_000)
        self.assertEqual(len((out / "rows.jsonl").read_text().splitlines()), 88)
        self.assertFalse((self.dir / "m1-fio-envelope.bin").exists(), "envelope file must be removed")
        self.assertTrue((out / "grid-557056-16-io_uring.host.jsonl").exists())

    def test_refusals(self):
        proof = json.loads(self.proof.read_text())
        proof["A8_identity"]["device"] += 1
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps(proof))
        p = self.run_env("identity", proof=bad)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("not on the proven filesystem", p.stderr)
        real = self.tmp / "fio"
        shutil.copy(self.fio, real)
        p = self.run_env("names", fio=real)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("only for the stub fio", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
