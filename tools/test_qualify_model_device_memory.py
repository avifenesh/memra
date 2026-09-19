#!/usr/bin/env python3
"""CPU controls for native #544 qualification admission; never touches CUDA."""
import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gate544", Path(__file__).with_name("qualify-model-device-memory.py"))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
A = "GPU-00000000-0000-0000-0000-000000000001"
B = "GPU-00000000-0000-0000-0000-000000000002"


def lock_row(path, pid, kind="FLOCK", mode="WRITE", waiter=False):
    stat = path.stat()
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    return f"1: {'-> ' if waiter else ''}{kind} ADVISORY {mode} {pid} {device} 0 EOF\n"


class QualificationControls(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "live flock ownership proof requires Linux /proc/locks")
    def test_live_linux_exclusive_lock_is_owned_until_release(self):
        import fcntl
        with tempfile.NamedTemporaryFile() as lock:
            path = Path(lock.name)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(gate.exclusive_lock_held(path.stat(), gate.ancestors(),
                                                    Path("/proc/locks").read_text()))
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            self.assertFalse(gate.exclusive_lock_held(path.stat(), gate.ancestors(),
                                                     Path("/proc/locks").read_text()))

    def test_two_distinct_owned_locks_admit_and_an_unrelated_holder_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "card-a", Path(tmp) / "card-b"
            a.touch()
            b.touch()
            specs = [f"{A}={a}", f"{B}={b}"]
            rows = lock_row(a, 123) + lock_row(b, 123)
            self.assertEqual(set(gate.lock_mapping([A, B], specs, {123, 124}, rows)), {A, B})
            with self.assertRaisesRegex(ValueError, "no exclusive flock"):
                gate.lock_mapping([A, B], specs, {124}, rows)
            with self.assertRaisesRegex(ValueError, "no exclusive flock"):
                gate.lock_mapping([A, B], specs, {123}, lock_row(a, 123))

    def test_read_posix_and_waiting_locks_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            path.touch()
            for row in [lock_row(path, 7, mode="READ"), lock_row(path, 7, kind="POSIX"),
                        lock_row(path, 7, waiter=True)]:
                self.assertFalse(gate.exclusive_lock_held(path.stat(), {7}, row))

    def test_one_file_or_hardlinked_files_cannot_cover_two_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "card-a", Path(tmp) / "card-b"
            a.touch()
            os.link(a, b)
            for specs in [[f"{A}={a}", f"{B}={a}"], [f"{A}={a}", f"{B}={b}"]]:
                with self.assertRaises(ValueError):
                    gate.lock_mapping([A, B], specs, {7}, lock_row(a, 7))

    def test_gpu_list_must_be_exact_full_uuids(self):
        for gpus, specs in [([A, A], []), (["0"], []), (["GPU-a"], []), ([A], []),
                            ([A], [f"{A}=/one", f"{B}=/two"])]:
            with self.assertRaises(ValueError):
                gate.lock_mapping(gpus, specs, {7}, "")
        self.assertEqual(gate.STAGES["same-device"][2], 1)
        self.assertEqual(gate.STAGES["pair"][2], 2)
        self.assertEqual(gate.STAGES["worker"][2], 2)

    def test_lease_binds_uuid_order_canonical_files_and_live_ancestor(self):
        lease = {"wrapper_pid": 7, "child_pid": 8, "requested_uuids": [B, A],
                 "lock_order": [A, B], "lock_files": {
                     uuid: f"/tmp/memra-gpu-locks/{uuid}.lock" for uuid in [A, B]}}
        specs, holders = gate.lease_locks([B, A], lease, {7, 8, 9})
        self.assertEqual(holders, {7})
        self.assertEqual(specs[0], f"{B}=/tmp/memra-gpu-locks/{B}.lock")
        for bad in [dict(lease, lock_order=[B, A]), dict(lease, requested_uuids=[A, B]),
                    dict(lease, wrapper_pid=1), dict(lease, child_pid=77),
                    dict(lease, lock_files={A: "/tmp/unshared", B: "/tmp/also-unshared"})]:
            with self.assertRaises(ValueError):
                gate.lease_locks([B, A], bad, {7, 8, 9})

    def test_no_pass_from_missing_filtered_or_skipped_test(self):
        good = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 600 filtered out;"
        gate.test_verdict(good)
        for text in ["", good.replace("1 passed", "0 passed"),
                     good.replace("0 ignored", "1 ignored"), "SKIP: absent hardware\n" + good]:
            with self.assertRaises(ValueError):
                gate.test_verdict(text)

    def test_model_flags_and_documentation_stubs_cannot_leak_into_native_build(self):
        with patch.dict(os.environ, {"DOCS_RS": "1", "MEMRA_GLM5_TP": "all@0,1", "PATH": "/bin"}, clear=True):
            self.assertEqual(gate.clean_env(), {"PATH": "/bin"})


if __name__ == "__main__":
    unittest.main()
