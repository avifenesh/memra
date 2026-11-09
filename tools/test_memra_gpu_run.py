#!/usr/bin/env python3
"""CPU contract for tools/memra-gpu-run, the physical-card lease wrapper release capture runs under.

No GPU touches this file. A planted nvidia-smi answers the two queries the wrapper makes (the device list and
the compute processes, read from files the test and the child write), and --lock-dir points the per-UUID lock at
a temporary directory. Each case runs the real wrapper as a subprocess and checks what qualify-release.py capture
and seal rely on: the child passes capture's own live_lease (its lock-dir seam pointed at the test's directory),
sees its own pid, its parent and its card in the lease, runs in its own process group with the box lock held and
the default SIGPIPE/SIGXFSZ dispositions, with an inherited ignored SIGHUP/SIGINT left ignored; the command never
runs without a lease (a receipt dir that refuses it, a lease write that fails, a wrapper killed or signalled
before the go byte); a missing command says why; a timeout or a signal stops the command's whole process group
(SIGKILL for one that ignores SIGTERM) before the lease is finished and the locks released; the lock outlives a
killed wrapper while the command runs, reaches a descendant only through inherited descriptors, and is held until
the lease is finished; a descendant left in the group fails the lease; and the finished lease records exit,
timeout, signal and lingering compute the way release_qualification.validate_lease judges them (a dirty run must
be refused).
"""
import fcntl
import importlib.machinery
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import release_qualification as q  # noqa: E402

# GPU_RUN_WRAPPER_UNDER_TEST runs the same contract against another copy (the before/after check of a fix).
WRAPPER = Path(os.environ.get("GPU_RUN_WRAPPER_UNDER_TEST", TOOLS / "memra-gpu-run"))
UUID = "GPU-00000000-1111-2222-3333-444444444444"
OTHER = "GPU-55555555-6666-7777-8888-999999999999"
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

# The child: passes capture's own live-lease check, checks what that check does not (exact pids, its own process
# group, the box lock), then follows its mode. (A Python child ignores SIGPIPE and SIGXFSZ itself at startup, so
# the dispositions are checked with a shell command instead.)
CHILD = textwrap.dedent("""\
    import fcntl, importlib.util, json, os, subprocess, sys, time
    from pathlib import Path
    marker, mode, arg = sys.argv[1], sys.argv[2], sys.argv[3]
    tools = Path(os.environ["GPU_RUN_TEST_TOOLS"])
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("qualify_release", tools / "qualify-release.py")
    qr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qr)
    # one card, CUDA_VISIBLE_DEVICES, wrapper/child ancestry, the lock file and its FLOCK owned by the wrapper
    lease = qr.live_lease(lock_dir=Path(os.environ["GPU_RUN_TEST_LOCKS"]))
    assert lease["child_pid"] == os.getpid(), lease
    assert lease["wrapper_pid"] == os.getppid(), lease
    assert os.getpgid(0) == os.getpid(), "the command runs in its own process group"
    if lease.get("box_lock"):
        fd = os.open(lease["box_lock"], os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sys.exit("the box lock was not held during the child")
        except BlockingIOError:
            pass
        os.close(fd)
    open(marker, "w").write("ran")
    if mode == "exit":
        sys.exit(int(arg))
    if mode == "grandchild":  # a descendant in the same group, then wait
        g = subprocess.Popen(["sleep", "300"])
        open(arg, "w").write(str(g.pid))
        time.sleep(300)
    if mode == "linger":  # leave a compute process on the card behind
        open(arg, "a").write(lease["requested_uuids"][0] + ", " + str(os.getpid()) + "\\n")
        sys.exit(0)
    if mode == "sleep":
        time.sleep(float(arg))
        sys.exit(0)
    if mode == "orphan":  # leave a descendant in the group and exit cleanly
        g = subprocess.Popen(["sleep", "300"])
        open(arg, "w").write(str(g.pid))
        sys.exit(0)
    if mode == "trap":  # exit 0 on SIGTERM: a signalled lease must fail even when the command exits cleanly
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        open(arg, "w").write("armed")
        time.sleep(300)
    """)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat").read_text() if Path(f"/proc/{pid}/stat").exists() else ""
    return bool(stat) and stat.split(") ")[-1][:1] != "Z"


def held(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def children(pid):
    try:
        return [int(c) for c in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def drain_fifo(path, timeout=30.0):
    """Read a FIFO the wrapper writes; None when no writer came within TIMEOUT (a bounded wait, never a hang)."""
    out = {}

    def read():
        with open(path) as fifo:
            out["data"] = fifo.read()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))  # a writer that closes at once ends the read
        reader.join(5)
        return None
    return out.get("data")


def wrapper_module():
    loader = importlib.machinery.SourceFileLoader("memra_gpu_run", str(WRAPPER))
    spec = importlib.util.spec_from_loader("memra_gpu_run", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def wait_until(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


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
        self.locks = self.tmp / "locks"
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        FAKE_SMI_GPUS=str(self.gpus), FAKE_SMI_APPS=str(self.apps),
                        GPU_RUN_TEST_TOOLS=str(TOOLS), GPU_RUN_TEST_LOCKS=str(self.locks))
        self.child = self.tmp / "child.py"
        self.child.write_text(CHILD)
        self.marker = self.tmp / "ran"
        self.receipt = self.tmp / "lease"
        self.lock = self.locks / f"{UUID}.lock"

    def tearDown(self):
        subprocess.run(["chmod", "-R", "u+w", str(self.tmp)], check=False)
        subprocess.run(["rm", "-rf", str(self.tmp)], check=False)

    def argv(self, mode="exit", arg="0", extra=(), linger="1"):
        return [sys.executable, str(WRAPPER), "--gpus", UUID, "--receipt", str(self.receipt),
                "--lock-dir", str(self.locks), "--wait", "5", "--linger-wait", linger, *extra, "--",
                sys.executable, str(self.child), str(self.marker), mode, arg]

    def wrap(self, mode="exit", arg="0", extra=()):
        return subprocess.run(self.argv(mode, arg, extra), env=self.env, capture_output=True, text=True, timeout=180)

    def bare(self, *cmd, extra=(), receipt=None):
        """argv for a plain command (no lease checks of its own, so nothing but the wrapper stops it)."""
        return [sys.executable, str(WRAPPER), "--gpus", UUID, "--receipt", str(receipt or self.receipt),
                "--lock-dir", str(self.locks), "--wait", "5", "--linger-wait", "1", *extra, "--", *cmd]

    def blocked_in_the_lease_write(self, cmd):
        """Start the wrapper with lease.json.tmp planted as a FIFO: it forks, then blocks writing the lease
        (no reader), so the child waits for a go byte that has not come. Returns (wrapper, child pid)."""
        self.receipt.mkdir()
        os.mkfifo(self.receipt / "lease.json.tmp")
        p = subprocess.Popen(self.bare(*cmd), env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.addCleanup(p.stderr.close)
        # lease.json is created after the wrapper's last nvidia-smi call, so its only child from then on is the fork
        self.assertTrue(wait_until(lambda: (self.receipt / "lease.json").exists()), "the wrapper never created the lease")
        kids = []
        self.assertTrue(wait_until(lambda: kids.extend(children(p.pid)) or kids), "the wrapper never forked")
        child = kids[0]

        def stop():
            for pid in (p.pid, child):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.wait(timeout=30)
        self.addCleanup(stop)
        return p, child

    def start(self, mode, arg, extra=()):
        p = subprocess.Popen(self.argv(mode, arg, extra), env=self.env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        self.assertTrue(wait_until(self.marker.exists), "the child never ran")
        return p

    def lease(self):
        return json.loads((self.receipt / "lease.json").read_text())

    def canonical(self, lease):
        # The only seam: the validator requires the canonical lock directory; the test uses a temporary one.
        return dict(lease, lock_files={UUID: f"/tmp/memra-gpu-locks/{UUID}.lock"})

    def run_for(self, lease):
        device = {"index": "2", "uuid": UUID, "name": NAME, "compute_cap": "12.0", "driver_version": "615.71.09"}
        return {"lease_owner": {k: lease[k] for k in ("wrapper_pid", "child_pid", "requested_uuids")},
                "started_unix": lease["started_unix"], "finished_unix": lease["finished_unix"],
                "hardware": {"devices": [device],
                             "headroom_query": {"selector": "cuda-visible-uuid", "uuid": UUID, "nvml_index": 2}},
                "numeric_environment": {"CUDA_VISIBLE_DEVICES": q.digest(UUID.encode())}}

    def test_child_sees_its_lease_card_locks_and_default_signals(self):
        box = self.tmp / "box.lock"
        r = self.wrap(extra=("--box-lock", str(box)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.marker.read_text(), "ran")
        lease = self.lease()
        self.assertEqual(lease["state"], "finished")
        self.assertEqual((lease["exit_code"], lease["child_exit_code"]), (0, 0))
        self.assertIs(lease["timed_out"], False)
        self.assertIsNone(lease["interrupted_signal"])
        self.assertEqual(lease["lingering_compute"], [])
        self.assertEqual(lease["box_lock"], str(box))
        self.assertEqual(lease["devices"], [{"index": 2, "uuid": UUID, "name": NAME, "memory_mib": 97887,
                                             "pci_bus_id": "00000000:C1:00.0"}])
        self.assertEqual(len(lease["all_devices"]), 2)
        self.assertFalse(held(self.lock), "the lock outlived a clean lease")
        self.assertFalse(held(box), "the box lock outlived a clean lease")

    def test_the_command_starts_with_default_sigpipe_and_sigxfsz(self):
        r = subprocess.run([sys.executable, str(WRAPPER), "--gpus", UUID, "--receipt", str(self.receipt),
                            "--lock-dir", str(self.locks), "--wait", "5", "--linger-wait", "1", "--",
                            "sh", "-c", f"grep SigIgn: /proc/$$/status > {self.marker}"],
                           env=self.env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr)
        ign = int(self.marker.read_text().split()[1], 16)
        self.assertFalse(ign & (1 << (signal.SIGPIPE - 1)), f"SIGPIPE left ignored: {ign:x}")
        self.assertFalse(ign & (1 << (signal.SIGXFSZ - 1)), f"SIGXFSZ left ignored: {ign:x}")

    def test_finished_lease_passes_the_validator(self):
        self.assertEqual(self.wrap().returncode, 0)
        lease = self.lease()
        q.validate_lease(self.canonical(lease), self.run_for(lease))

    def test_refuses_a_card_that_runs_compute(self):
        self.apps.write_text(f"{UUID}, 4242\n")
        r = self.wrap()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already runs compute", r.stderr)
        self.assertFalse(self.marker.exists())
        self.assertFalse((self.receipt / "lease.json").exists())

    def test_failed_child_is_not_a_clean_lease(self):
        r = self.wrap("exit", "3")
        self.assertEqual(r.returncode, 1)
        lease = self.lease()
        self.assertEqual((lease["exit_code"], lease["child_exit_code"]), (1, 3))
        with self.assertRaisesRegex(q.GateError, "did not finish cleanly"):
            q.validate_lease(self.canonical(lease), self.run_for(lease))

    def test_lingering_compute_fails_the_lease(self):
        r = self.wrap("linger", str(self.apps))
        self.assertEqual(r.returncode, 1, r.stderr)
        lease = self.lease()
        self.assertEqual(lease["child_exit_code"], 0)
        self.assertTrue(lease["lingering_compute"], "the compute process left on the card was not recorded")
        self.assertEqual(lease["exit_code"], 1)
        with self.assertRaisesRegex(q.GateError, "did not finish cleanly"):
            q.validate_lease(self.canonical(lease), self.run_for(lease))

    def test_lock_held_until_the_lease_is_finished(self):
        p = subprocess.Popen(self.argv("linger", str(self.apps), linger="3"), env=self.env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertTrue(wait_until(self.marker.exists), "the child never ran")
        time.sleep(1)  # the child has exited; the wrapper is polling the card for lingering compute
        self.assertIsNone(p.poll(), "the wrapper finished before its lingering check")
        self.assertTrue(held(self.lock), "the lock was released before the lease was finished")
        p.wait(timeout=60)

    def test_a_descendant_left_in_the_group_fails_the_lease_and_is_stopped(self):
        gfile = self.tmp / "grandchild"
        r = self.wrap("orphan", str(gfile))
        self.assertEqual(r.returncode, 1, r.stderr)
        lease = self.lease()
        self.assertEqual(lease["child_exit_code"], 0)
        self.assertIs(lease["process_group_lingered"], True)
        self.assertEqual(lease["exit_code"], 1)
        self.assertFalse(alive(int(gfile.read_text())), "the orphaned descendant outlived the lease")
        self.assertFalse(held(self.lock))

    def test_a_signal_fails_the_lease_even_when_the_command_exits_cleanly(self):
        armed = self.tmp / "armed"
        p = self.start("trap", str(armed))
        self.assertTrue(wait_until(armed.exists))
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=120)
        lease = self.lease()
        self.assertEqual(lease["child_exit_code"], 0)
        self.assertEqual(lease["interrupted_signal"], signal.SIGTERM)
        self.assertEqual(lease["exit_code"], 1)
        self.assertEqual(p.returncode, 1)

    def test_timeout_stops_the_whole_group_before_the_lock_is_released(self):
        gfile = self.tmp / "grandchild"
        r = subprocess.run(self.argv("grandchild", str(gfile), ("--timeout", "2")), env=self.env,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 1, r.stderr)
        lease = self.lease()
        self.assertIs(lease["timed_out"], True)
        self.assertEqual(lease["child_exit_code"], -signal.SIGTERM)
        self.assertEqual(lease["exit_code"], 1)
        self.assertFalse(alive(int(gfile.read_text())), "the grandchild outlived the lease")
        self.assertFalse(held(self.lock))

    def test_sigterm_to_the_wrapper_stops_the_group_and_is_recorded(self):
        gfile = self.tmp / "grandchild"
        p = self.start("grandchild", str(gfile))
        self.assertTrue(wait_until(gfile.exists))
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=120)
        self.assertEqual(p.returncode, 1)
        lease = self.lease()
        self.assertEqual(lease["interrupted_signal"], signal.SIGTERM)
        self.assertEqual(lease["child_exit_code"], -signal.SIGTERM)
        self.assertFalse(alive(int(gfile.read_text())), "the grandchild outlived the signalled lease")
        with self.assertRaisesRegex(q.GateError, "did not finish cleanly"):
            q.validate_lease(self.canonical(lease), self.run_for(lease))

    def test_lock_outlives_a_killed_wrapper_while_the_command_runs(self):
        p = self.start("sleep", "4")
        p.send_signal(signal.SIGKILL)
        p.wait(timeout=30)
        self.assertTrue(held(self.lock), "a killed wrapper freed the card while its command still ran")
        self.assertTrue(wait_until(lambda: not held(self.lock), 30), "the lock stayed held after the command ended")

    @unittest.skipIf(os.geteuid() == 0, "root ignores the read-only mode this case relies on")
    def test_the_command_never_runs_without_its_lease(self):
        self.receipt.mkdir()
        self.receipt.chmod(0o555)  # the lease cannot be created
        # a bare command that would leave the marker if it ever ran (it reads no lease, so nothing else stops it)
        r = subprocess.run([sys.executable, str(WRAPPER), "--gpus", UUID, "--receipt", str(self.receipt),
                            "--lock-dir", str(self.locks), "--wait", "5", "--linger-wait", "1", "--",
                            sys.executable, "-c", f"open({str(self.marker)!r}, 'w').write('ran')"],
                           env=self.env, capture_output=True, text=True, timeout=180)
        self.assertNotEqual(r.returncode, 0)
        time.sleep(1)
        self.assertFalse(self.marker.exists(), "the command ran without a lease")
        self.assertFalse(held(self.lock))

    def test_a_wrapper_killed_before_the_go_byte_never_runs_the_command(self):
        p, child = self.blocked_in_the_lease_write(["sh", "-c", f"echo ran > {self.marker}"])
        p.send_signal(signal.SIGKILL)
        p.wait(timeout=30)
        self.assertTrue(wait_until(lambda: not alive(child), 10), "the child outlived its wrapper without a go byte")
        self.assertTrue(wait_until(lambda: not held(self.lock), 10), "a child without a go byte kept the card locked")
        self.assertFalse(self.marker.exists(), "the command ran although the lease never named it")

    def test_a_signal_before_the_go_byte_keeps_the_command_from_starting(self):
        p, _ = self.blocked_in_the_lease_write(["sh", "-c", f"echo ran > {self.marker}"])
        p.send_signal(signal.SIGTERM)
        running = drain_fifo(self.receipt / "lease.json.tmp")  # let the lease write finish
        self.assertIsNotNone(running, "the wrapper did not survive the signal to write its lease")
        self.assertEqual(json.loads(running)["state"], "running")
        p.wait(timeout=60)
        self.assertEqual(p.returncode, 1, p.stderr.read())
        lease = self.lease()
        self.assertEqual(lease["interrupted_signal"], signal.SIGTERM)
        self.assertEqual(lease["child_exit_code"], 127, "the child did not exit before the command (it ran it or was stopped mid-start)")
        self.assertFalse(self.marker.exists())
        self.assertFalse(held(self.lock))

    def test_a_lease_that_cannot_be_written_never_starts_the_command(self):
        self.receipt.mkdir()
        (self.receipt / "lease.json.tmp").mkdir()  # the lease is created, but its content cannot land
        r = subprocess.run(self.bare("sh", "-c", f"echo ran > {self.marker}"), env=self.env, capture_output=True,
                           text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot write the lease", r.stderr)
        time.sleep(1)
        self.assertFalse(self.marker.exists(), "the command ran without a written lease")
        self.assertFalse(held(self.lock))

    def test_a_missing_command_fails_the_lease_and_says_why(self):
        r = subprocess.run(self.bare(str(self.tmp / "no-such-command")), env=self.env, capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot run " + str(self.tmp / "no-such-command"), r.stderr)
        lease = self.lease()
        self.assertEqual((lease["child_exit_code"], lease["exit_code"]), (127, 1))

    def test_a_command_that_ignores_sigterm_is_killed_after_the_grace(self):
        start = time.monotonic()
        r = subprocess.run(self.bare("sh", "-c", f"trap '' TERM; echo ran > {self.marker}; sleep 120",
                                     extra=("--timeout", "1", "--grace", "1")),
                           env=self.env, capture_output=True, text=True, timeout=90)
        self.assertEqual(r.returncode, 1, r.stderr)
        lease = self.lease()
        self.assertIs(lease["timed_out"], True)
        self.assertEqual(lease["child_exit_code"], -signal.SIGKILL)
        self.assertLess(time.monotonic() - start, 60, "the SIGKILL never came")
        self.assertFalse(held(self.lock))

    def test_signals_inherited_as_ignored_stay_ignored(self):
        def ignore():
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        p = subprocess.Popen(self.bare("sh", "-c", f"grep SigIgn: /proc/$$/status > {self.marker}; sleep 3"),
                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             preexec_fn=ignore)
        self.assertTrue(wait_until(lambda: self.marker.exists() and self.marker.read_text()))
        p.send_signal(signal.SIGHUP)  # nohup: a dropped connection must not stop the capture
        p.send_signal(signal.SIGINT)
        p.wait(timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr.read())
        lease = self.lease()
        self.assertIsNone(lease["interrupted_signal"])
        self.assertEqual(lease["child_exit_code"], 0)
        ign = int(self.marker.read_text().split()[1], 16)
        self.assertTrue(ign & (1 << (signal.SIGHUP - 1)) and ign & (1 << (signal.SIGINT - 1)),
                        f"the command lost the inherited ignore: {ign:x}")

    def test_a_relative_receipt_reaches_a_command_in_another_directory(self):
        r = subprocess.run(self.bare("sh", "-c", f'cd / && test -f "$MEMRA_GPU_LEASE_FILE" && echo ran > {self.marker}',
                                     receipt="lease"),
                           env=self.env, cwd=self.tmp, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.marker.exists(), "MEMRA_GPU_LEASE_FILE was not absolute")

    def test_the_lock_reaches_a_descendant_only_through_inherited_descriptors(self):
        # The docs' claim, pinned: the command holds the lock; a Python subprocess child holds it only with
        # close_fds=False (capture's battery is started with the default, so it does not).
        probe = (f"import os, subprocess, sys; lock = {str(self.lock)!r}; "
                 "check = 'for f in /proc/self/fd/*; do [ \"$(readlink $f)\" = \"$1\" ] && echo held; done; :'; "
                 "out = lambda **kw: subprocess.run(['sh', '-c', check, 'sh', lock], capture_output=True, text=True, **kw).stdout; "
                 f"open({str(self.marker)!r}, 'w').write(out() + '|' + out(close_fds=False))")
        r = subprocess.run(self.bare(sys.executable, "-c", probe), env=self.env, capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        default, kept = self.marker.read_text().split("|")
        self.assertEqual(default.strip(), "", "a default subprocess child kept the lock descriptor")
        self.assertEqual(kept.strip(), "held", "close_fds=False did not pass the lock descriptor")

    def test_signalling_a_group_of_another_account_does_not_crash_the_wrapper(self):
        module = wrapper_module()
        real = os.killpg

        def refuse(_pgid, _sig):
            raise PermissionError(1, "Operation not permitted")

        os.killpg = refuse
        try:
            module.signal_group(424242, signal.SIGTERM)  # must not raise
            self.assertTrue(module.group_alive(424242), "an unsignallable group must count as alive")
        finally:
            os.killpg = real

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
