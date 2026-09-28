#!/usr/bin/env python3
"""Unit cells for m1-step-runner.py (OWED 20, M1-PREREG.md section H). CPU only."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("step", HERE / "m1-step-runner.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

LINE = ("[spill-pread] reads={r} bytes={b} errors=0 short_reads=0 fallbacks={f} buffer_waits=1 ring_full=2 "
        "overread_bytes={o} worker_read_ns=10 demand_read_ns=0 wait_ns=5 h2d_submits={r}")


class StepRunner(unittest.TestCase):
    def setUp(self):
        # A real filesystem: tmpfs pages cannot be dropped, so a cold check there always fails.
        base = HERE.parents[2] / "target"
        base.mkdir(exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=base, prefix="step-runner-test-"))
        self.shards = []
        for i, n in enumerate((8192, 12288, 4096)):
            p = self.tmp / f"s-0000{i + 1}-of-00003.gguf"
            with p.open("wb") as f:
                f.write(os.urandom(n))
                f.flush()
                os.fsync(f.fileno())  # clean pages: DONTNEED cannot drop dirty ones
            self.shards.append(p)
        self.lock = {"artifact": {"expert_bank_bytes": 10, "files": [
            {"path": f"IQ4_XS/{p.name}", "bytes": p.stat().st_size, "sha256": S.R.sha(p), "shard": True}
            for p in self.shards] + [{"path": "mtp.gguf", "bytes": 1, "sha256": "x", "shard": False}]}}
        self.saved = (S.R.CACHE, S.R.parse_log, S.R.GPU.GpuSampler)

    def tearDown(self):
        S.R.CACHE, S.R.parse_log, S.R.GPU.GpuSampler = self.saved
        for p in self.tmp.iterdir():
            p.unlink()
        self.tmp.rmdir()

    def test_totals_lines_of_both_stages_are_summed(self):
        S.install(self.lock, self.tmp)
        text = LINE.format(r=10, b=100, f=0, o=4096 * 10) + "\n" + LINE.format(r=5, b=50, f=2, o=4096 * 5) + "\n"
        out = S.R.parse_log(text)
        self.assertEqual(out["drop_lines"], 2)
        self.assertEqual(out["drop"][0], "15")
        self.assertEqual(out["drop"][4], "2")
        self.assertEqual(out["drop"][7], str(4096 * 15))
        single = S.R.parse_log(LINE.format(r=3, b=30, f=0, o=0))
        self.assertEqual(single["drop_lines"], 1)
        self.assertEqual(single["drop"][0], "3")

    def test_cold_and_residency_cover_every_shard(self):
        shards = S.install(self.lock, self.tmp)
        self.assertEqual(shards, [str(p) for p in self.shards])
        for p in self.shards:
            p.read_bytes()
        self.assertTrue(S.R.CACHE.cold(shards[:1]))
        resident, pages = S.R.CACHE.residency(shards[0])
        self.assertEqual(pages, (8192 + 12288 + 4096) // 4096)
        self.assertEqual(resident, 0, "cold must have dropped every shard")

    def test_shard_identity_and_ram_gates(self):
        shards = S.install(self.lock, self.tmp)
        S.check_shards(self.lock, shards)
        with self.shards[1].open("wb") as f:
            f.write(os.urandom(12288))
            os.fsync(f.fileno())
        with self.assertRaises(Exception):
            S.check_shards(self.lock, shards)
        with self.assertRaises(Exception):
            S.check_ram(self.lock)  # any real host has more than 10 bytes of RAM
        big = {"artifact": {"expert_bank_bytes": 1 << 62}}
        self.assertLess(S.check_ram(big)["cache_ceiling_fraction"], 1)

    def test_gpu_sampler_covers_both_cards(self):
        S.install(self.lock, self.tmp)
        self.assertEqual(S.R.GPU.GpuSampler(self.tmp / "g.csv").device, "0,1")


if __name__ == "__main__":
    unittest.main()
