#!/usr/bin/env python3
"""CPU contract for tools/memra-gpu-run, the physical-card lease wrapper release capture runs under.

No GPU touches this file. A planted nvidia-smi answers the two queries the wrapper makes (the device list and
the compute processes), and --lock-dir points the per-UUID lock at a temporary directory. Each case runs the
real wrapper as a subprocess and checks what qualify-release.py capture and seal rely on: the child sees its
own pid, its parent and its card in the lease and the environment, the exclusive flock is held while it runs,
and the finished lease records exit, timeout and lingering compute the way release_qualification.validate_lease
judges them.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import release_qualification as q  # noqa: E402

WRAPPER = TOOLS / "memra-gpu-run"
UUID = "GPU-9f8ecd14-35a9-f104-1dd9-c4757538bb34"
OTHER = "GPU-875678d1-165c-23e1-ba0e-65298ed4a996"
NAME = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"

FAKE_SMI = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, sys
    args = " ".join(sys.argv[1:])
    if "--query-gpu=" in args:
        sys.stdout.write(open(os.environ["FAKE_SMI_GPUS"]).read())
    elif "--query-compute-apps=" in args:
        sys.stdout.write(open(os.environ["FAKE_SMI_APPS"]).read())
    else:
        sys.exit("unexpected nvidia-smi call: " + args)
    """)

CHILD = textwrap.dedent("""\
    import fcntl, json, os, sys
    lease = json.load(open(os.environ["MEMRA_GPU_LEASE_FILE"]))
    assert lease["child_pid"] == os.getpid(), lease
    assert lease["wrapper_pid"] == os.getppid(), lease
    assert os.environ["CUDA_VISIBLE_DEVICES"] == lease["requested_uuids"][0]
    lock = lease["lock_files"][lease["requested_uuids"][0]]
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit("the per-UUID lock was not held during the child")
    except BlockingIOError:
        pass
    st = os.stat(lock)
    held = [l for l in open("/proc/locks") if " FLOCK " in l and l.split()[4] == str(lease["wrapper_pid"])
            and l.split()[5].split(":")[2] == str(st.st_ino)]
    assert held, "no FLOCK owned by the wrapper on the lock inode"
    open(sys.argv[1], "w").write("ran")
    sys.exit(int(sys.argv[2]))
    """)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="memra-gpu-run-test-"))
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        smi = bin_dir / "nvidia-smi"
        smi.write_text(FAKE_SMI)
        smi.chmod(0o755)
        self.gpus = self.tmp / "gpus.csv"
        self.gpus.write_text(f"2, {UUID}, {NAME}, 97887, 00000000:C1:00.0\n3, {OTHER}, {NAME}, 97887, 00000000:F1:00.0\n")
        self.apps = self.tmp / "apps.csv"
        self.apps.write_text("")
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        FAKE_SMI_GPUS=str(self.gpus), FAKE_SMI_APPS=str(self.apps))
        self.child = self.tmp / "child.py"
        self.child.write_text(CHILD)
        self.marker = self.tmp / "ran"
        self.receipt = self.tmp / "lease"

    def tearDown(self):
        subprocess.run(["rm", "-rf", str(self.tmp)], check=False)

    def wrap(self, child_code=0, extra=(), child=None):
        cmd = [sys.executable, str(WRAPPER), "--gpus", UUID, "--receipt", str(self.receipt),
               "--lock-dir", str(self.tmp / "locks"), "--wait", "5", *extra, "--",
               *(child or [sys.executable, str(self.child), str(self.marker), str(child_code)])]
        return subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=120)

    def lease(self):
        return json.loads((self.receipt / "lease.json").read_text())

    def test_child_sees_its_lease_card_and_held_lock(self):
        r = self.wrap()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.marker.read_text(), "ran")
        lease = self.lease()
        self.assertEqual(lease["state"], "finished")
        self.assertEqual((lease["exit_code"], lease["child_exit_code"]), (0, 0))
        self.assertIs(lease["timed_out"], False)
        self.assertIsNone(lease["interrupted_signal"])
        self.assertEqual(lease["lingering_compute"], [])
        self.assertEqual(lease["requested_uuids"], [UUID])
        self.assertEqual(lease["devices"], [{"index": 2, "uuid": UUID, "name": NAME, "memory_mib": 97887,
                                             "pci_bus_id": "00000000:C1:00.0"}])
        self.assertEqual(len(lease["all_devices"]), 2)
        self.assertLessEqual(lease["started_unix"], lease["finished_unix"])

    def test_finished_lease_passes_the_validator(self):
        self.assertEqual(self.wrap().returncode, 0)
        lease = self.lease()
        # The only seam: the validator requires the canonical lock directory, the test uses a temporary one.
        lease["lock_files"] = {UUID: f"/tmp/memra-gpu-locks/{UUID}.lock"}
        device = {"index": "2", "uuid": UUID, "name": NAME, "compute_cap": "12.0", "driver_version": "615.71.09"}
        run = {"lease_owner": {k: lease[k] for k in ("wrapper_pid", "child_pid", "requested_uuids")},
               "started_unix": lease["started_unix"], "finished_unix": lease["finished_unix"],
               "hardware": {"devices": [device],
                            "headroom_query": {"selector": "cuda-visible-uuid", "uuid": UUID, "nvml_index": 2}},
               "numeric_environment": {"CUDA_VISIBLE_DEVICES": q.digest(UUID.encode())}}
        q.validate_lease(lease, run)

    def test_refuses_a_card_that_runs_compute(self):
        self.apps.write_text(f"{UUID}, 4242\n")
        r = self.wrap()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already runs compute", r.stderr)
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.receipt / "lease.json").exists())

    def test_failed_child_is_not_a_clean_lease(self):
        r = self.wrap(child_code=3)
        self.assertEqual(r.returncode, 1)
        lease = self.lease()
        self.assertEqual((lease["exit_code"], lease["child_exit_code"]), (1, 3))
        lease["lock_files"] = {UUID: f"/tmp/memra-gpu-locks/{UUID}.lock"}
        with self.assertRaisesRegex(q.GateError, "did not finish cleanly"):
            q.validate_lease(lease, {})

    def test_timeout_is_recorded(self):
        r = self.wrap(extra=("--timeout", "1"), child=[sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertEqual(r.returncode, 1)
        lease = self.lease()
        self.assertIs(lease["timed_out"], True)
        self.assertEqual(lease["exit_code"], 1)

    def test_refuses_a_reused_receipt_and_an_unknown_card(self):
        self.assertEqual(self.wrap().returncode, 0)
        again = self.wrap()
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("already holds a lease", again.stderr)
        self.gpus.write_text(f"3, {OTHER}, {NAME}, 97887, 00000000:F1:00.0\n")
        self.receipt = self.tmp / "lease2"
        missing = self.wrap()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("no such GPU", missing.stderr)


if __name__ == "__main__":
    unittest.main()
