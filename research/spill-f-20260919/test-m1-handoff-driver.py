#!/usr/bin/env python3
"""OWED 13 CPU gates for m1-handoff-driver.py against m1-stub-kv-server.py (no GPU, no lock).

The stub prints the real handoff log formats (export line with write_ms/fsync_ms, import armed
at boot, import DONE) and serves /v1/models, /v1/completions and /metrics. Green: two cycles pass
with a real file written, hashed and made cold. Red: a refused export, an import with skips, a
probe that misses the restored prefix, a probe whose text drifts, prompts exhausted before the
size, a scratch directory off the proven filesystem, and the stub-only lock bypass.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())
DRIVER = HERE / "m1-handoff-driver.py"
STUB = HERE / "m1-stub-kv-server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Harness(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drv", DRIVER)
        self.D = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.D)
        self.tmp = Path(tempfile.mkdtemp(dir=ROOT, prefix=".m1-b2-test-"))
        self.scratch = self.tmp / "b2"
        self.scratch.mkdir()
        identity = self.D.B.filesystem_identity(self.scratch)
        self.proof = self.tmp / "proof.private.json"
        st = os.stat(self.scratch)
        top = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}").resolve()
        leaf = top.parent.name if (top / "partition").exists() else top.name
        self.proof.write_text(json.dumps({"verdict": "PASS", "class": "nvme-local-direct", "A8_identity": identity,
                                          "A4_block_graph": {"nodes": [{"name": top.name}], "leaves": [{"name": leaf}]}}))
        self.prompts = self.tmp / "prompts.jsonl"
        self.prompts.write_text("".join(json.dumps({"id": i, "prompt": f"stub prompt number {i} body"}) + "\n"
                                        for i in range(8)))
        for name, kind in (("m1-stub-kv-gate", "gate"), ("m1-stub-kv-server", "server")):
            w = self.tmp / name
            w.write_text(f"#!/bin/sh\nM1_STUB_KIND={kind} exec {sys.executable} {STUB} \"$@\"\n")
            w.chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_driver(self, name, size=1 << 29, cycles="2", env=None, gate="m1-stub-kv-gate",
                   server="m1-stub-kv-server", scratch=None):
        out = self.tmp / name
        argv = [sys.executable, str(DRIVER), "run", "--gate", str(self.tmp / gate), "--server", str(self.tmp / server),
                "--artifact", "/nonexistent-stub-artifact", "--prompts", str(self.prompts), "--proof", str(self.proof),
                "--scratch", str(scratch or self.scratch), "--out", str(out), "--size-bytes", str(size),
                "--cycles", cycles, "--port", str(free_port()), "--stub-no-lock"]
        e = dict(os.environ, M1_STUB_ENTRY_BYTES=str(128 << 20), **(env or {}))
        proc = subprocess.run(argv, capture_output=True, text=True, env=e, timeout=600)
        return proc, out

    def cycles(self, out):
        return [json.loads(p.read_text()) for p in sorted(out.glob("cycle-*/cycle.json"))]


class Green(Harness):
    def test_two_cycles_pass(self):
        proc, out = self.run_driver("green")
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        cs = self.cycles(out)
        self.assertEqual(len(cs), 2)
        for c in cs:
            self.assertTrue(c["passed"], c["problems"])
            self.assertEqual(c["prompts_sent"], 4)  # 4 x 128 MiB reaches 512 MiB
            self.assertEqual(c["export"]["entries"], 4)
            self.assertIn("fsync_ms", c["export"])
            self.assertEqual(c["import"]["done"]["skipped"], 0)
            self.assertTrue(c["cold_ok"])
            self.assertEqual(len(c["file_sha256"]), 64)
            self.assertTrue(all(p["identical"] and p["cached_tokens"] > 0 for p in c["probes"]))
            self.assertTrue(c["export_window"]["telemetry_ok"] and c["import_window"]["telemetry_ok"])
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual((summary["cycles"], summary["passed"], summary["n"]), (2, 2, 2))
        self.assertFalse((self.scratch / "handoff.bin").exists(), "import must consume the file")


class Red(Harness):
    def failed(self, name, needle, **kw):
        proc, out = self.run_driver(name, cycles="1", **kw)
        self.assertEqual(proc.returncode, 3, proc.stdout[-2000:] + proc.stderr[-2000:])
        c = self.cycles(out)[0]
        self.assertFalse(c["passed"])
        self.assertTrue(any(needle in p for p in c["problems"]), c["problems"])
        return c

    def test_refused_export(self):
        c = self.failed("refuse", "export refused", env={"M1_STUB_REFUSE_EXPORT": "1"})
        self.assertIn("no handoff file after the export", c["problems"])
        self.assertNotIn("import", c)

    def test_import_skips(self):
        self.failed("skips", "import skipped", env={"M1_STUB_IMPORT_SKIPS": "2"})

    def test_probe_misses_cache(self):
        self.failed("miss", "missed the restored prefix", env={"M1_STUB_MISS_PROBE": "1"})

    def test_probe_text_drift(self):
        self.failed("drift", "differs from the cold reference", env={"M1_STUB_TEXT_DRIFT": "1"})

    def test_prompts_exhausted(self):
        self.failed("exhausted", "prompts exhausted", size=16 << 30)

    def test_scratch_off_the_proven_filesystem(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as other:
            proc, out = self.run_driver("offfs", scratch=other)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not on the proven filesystem", proc.stderr)
        self.assertFalse((out / "reference-boot.log").exists())

    def test_stub_bypass_needs_stub_names(self):
        shutil.copy(self.tmp / "m1-stub-kv-gate", self.tmp / "kv-handoff-gate")
        proc, _ = self.run_driver("names", gate="kv-handoff-gate")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("only for stub servers", proc.stderr)


class Parse(unittest.TestCase):
    def test_real_line_formats(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("drv2", DRIVER)
        D = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(D)
        export = ("[prefix-host] handoff export: 61 entries / 8712.3MB to /scratch/spill-f/b2/handoff.bin in 5321ms "
                  "write_ms=4102.7 fsync_ms=1188.4 (drain-demoted 7 device entries first; 0 skipped over the "
                  "MEMRA_KV_HOST_HANDOFF_MB cap)")
        self.assertEqual(D.parse_export(export), {"entries": 61, "mb": 8712.3, "ms": 5321, "write_ms": 4102.7,
                                                  "fsync_ms": 1188.4, "drain_demoted": 7, "skipped_over_cap": 0})
        text = ("[prefix-host] handoff import armed at boot: 61 entries / 8712.3MB from /scratch/x (file age 12s); "
                "re-materializing one per tick\n[prefix-host] handoff import DONE: 61 entries / 8712.3MB "
                "re-materialized, 0 skipped, in 9.4s from /scratch/x\n")
        self.assertEqual(D.parse_import(text), {"armed": [61, 8712.3], "aborted": False,
                                                "done": {"entries": 61, "mb": 8712.3, "skipped": 0, "seconds": 9.4}})
        self.assertIsNone(D.parse_export("[prefix-host] handoff export: 61 entries / 1.0MB to /x in 5ms (old format)"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
