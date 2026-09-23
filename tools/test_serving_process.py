"""Real, network/GPU-free process tests. Linux subreaping tests are explicit skips elsewhere."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import serving_process
from serving_process import (OwnedServer, ReadinessEvidence, ServingProcessError,
                             process_identity)


FIXTURE = r'''
import json, os, signal, subprocess, sys, time
mode = sys.argv[1]
child = None
if mode in ("tree", "orphan", "escaped", "stuck-tree", "orphan-drain"):
    child = subprocess.Popen([sys.executable, __file__, "leaf-default" if mode == "orphan-drain" else "leaf"],
                             start_new_session=mode == "escaped")
    print("DESCENDANT " + str(child.pid), flush=True)
if mode == "drain-worker":
    child = subprocess.Popen([sys.executable, __file__, "work-needed"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    print("DESCENDANT " + str(child.pid), flush=True)
    assert child.stdout.readline().strip() == "WORKER_READY"
if mode == "work-needed":
    def premature_term(_signum, _frame):
        print("PREMATURE_TERM", flush=True)
        raise SystemExit(41)
    signal.signal(signal.SIGTERM, premature_term)
    print("WORKER_READY", flush=True)
    assert sys.stdin.readline().strip() == "finish"
    time.sleep(0.25)
    print("INFLIGHT_COMPLETE", flush=True)
    raise SystemExit(0)
if mode == "leaf":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode in ("stuck", "stuck-tree"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
elif mode not in ("leaf", "orphan", "escaped"):
    def drain(_signum, _frame):
        print("DRAIN", flush=True)
        if mode == "drain-worker":
            try:
                child.stdin.write("finish\n")
                child.stdin.flush()
            except BrokenPipeError:
                pass
            result = child.stdout.read().strip()
            code = child.wait(timeout=2)
            print("WORKER_RESULT " + result + " exit=" + str(code), flush=True)
            if code != 0 or result != "INFLIGHT_COMPLETE":
                print("INFLIGHT_LOST", flush=True)
                raise SystemExit(43)
        elif child is not None and mode != "orphan-drain":
            child.kill()
            child.wait(timeout=2)
            print("CHILD_REAPED", flush=True)
        time.sleep(0.05)
        print("DRAINED", flush=True)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, drain)
print("STDERR", file=sys.stderr, flush=True)
print("CWD " + os.getcwd(), flush=True)
print("AMBIENT " + str(os.environ.get("SERVING_PROCESS_TEST_AMBIENT")), flush=True)
print("READY", flush=True)
if mode == "delayed-signal":
    def delivered(_number, _frame):
        print("SIGNAL_DELIVERED", flush=True)
        raise SystemExit(0)
    signal.signal(signal.SIGUSR1, delivered)
    print("SIGNAL_READY", flush=True)
if mode == "exit7":
    raise SystemExit(7)
if mode == "signal":
    os.kill(os.getpid(), signal.SIGUSR1)
if mode == "normal":
    time.sleep(0.3)
    print("COMPLETE", flush=True)
    raise SystemExit(0)
if mode in ("orphan", "escaped"):
    time.sleep(0.15)
    raise SystemExit(7)
while True:
    time.sleep(0.02)
'''


class OwnedServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="serving-process-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.script = self.root / "fixture.py"
        self.script.write_text(FIXTURE)
        self.servers = []
        self.addCleanup(self.clean_servers)

    def clean_servers(self):
        for server in self.servers:
            try:
                server.close(reason="test_cleanup")
            except ServingProcessError:
                pass

    def server(self, mode="idle", **limits):
        options = dict(startup_timeout=3, overall_timeout=6, drain_timeout=0.2, kill_timeout=1)
        options.update(limits)
        server = OwnedServer(argv=[sys.executable, str(self.script), mode],
                             env={"PYTHONUNBUFFERED": "1", "PRIVATE_VALUE": "not-in-receipt"},
                             cwd=str(self.root), evidence_dir=str(self.root / f"run-{len(self.servers)}"),
                             **options)
        self.servers.append(server)
        return server.start()

    def ready(self, server):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server.output_path.exists() and "READY" in server.output_path.read_text():
                return server.mark_ready(ReadinessEvidence(
                    server.identity, "owned_output", {"marker": "READY", "path": str(server.output_path)}))
            time.sleep(0.01)
        self.fail("owned stdout did not become ready")

    def assert_clean(self, server, receipt):
        self.assertTrue(receipt["cleanup"]["complete"], receipt)
        self.assertEqual(receipt["cleanup"]["remaining"], [])
        self.assertTrue(receipt["cleanup"]["raw_final"])
        self.assertIsNotNone(server._supervisor.returncode)
        self.assertIsNone(process_identity(server.identity.pid))
        self.assertEqual(json.loads((server.directory / "receipt.json").read_text()), receipt)
        self.assertNotIn("not-in-receipt", json.dumps(receipt))
        key = lambda identity: (identity["pid"], identity["start_time"])
        known = {key(row) for row in receipt["observed_processes"]}
        reaped = {key(row["identity"]) for row in receipt["descendant_exits"]}
        retired = {key(row["identity"]) for row in receipt["descendant_retirements"]}
        self.assertFalse(reaped & retired)
        self.assertEqual(known, reaped | retired | {key(receipt["server_exit"]["identity"])})
        finish = receipt["cleanup"]["finished_monotonic"]
        self.assertGreaterEqual(finish, receipt["server_exit"]["observed_monotonic"])
        self.assertEqual(receipt["cleanup"]["elapsed_s"], finish - receipt["stop"]["monotonic"])
        for row in receipt["signals"]:
            self.assertLessEqual(row["monotonic"], row["finished_monotonic"])
            self.assertLessEqual(row["finished_monotonic"], finish)
            if row["result"] == "sent":
                self.assertLessEqual(row["monotonic"], row["sent_monotonic"])
                self.assertLessEqual(row["sent_monotonic"], row["finished_monotonic"])
        for row in receipt["descendant_exits"]:
            self.assertLessEqual(row["observed_monotonic"], finish)
        for row in receipt["descendant_retirements"]:
            self.assertLessEqual(row["last_seen_monotonic"], row["started_monotonic"])
            self.assertLessEqual(row["started_monotonic"], row["finished_monotonic"])
            self.assertLessEqual(row["finished_monotonic"], finish)
            self.assertIsNone(row["exit_status"])

    def test_normal_exit_and_combined_raw_output(self):
        server = self.server("normal")
        self.ready(server)
        receipt = server.wait()
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["stop"]["reason"], "server_exit")
        self.assertEqual(receipt["server_exit"]["returncode"], 0)
        self.assertTrue(receipt["server_exit"]["before_cleanup"])
        self.assertFalse(receipt["signals"])
        self.assertIn("STDERR", server.output_path.read_text())
        self.assertIn("COMPLETE", server.output_path.read_text())
        self.assertIn("CWD " + str(self.root), server.output_path.read_text())
        self.assertEqual(receipt["supervisor_exit"], {"returncode": 0, "reaped": True})
        self.assertEqual(server.close(), receipt)

    def test_launch_failure_is_recorded_and_private_supervisor_reaped(self):
        server = OwnedServer(argv=[str(self.root / "missing-executable")], env={},
                             cwd=str(self.root), evidence_dir=str(self.root / "launch-failure"),
                             startup_timeout=1, overall_timeout=2, drain_timeout=0.1, kill_timeout=0.5)
        with self.assertRaises(ServingProcessError):
            server.start()
        receipt = server.receipt()
        self.assertIsNone(receipt["server"])
        self.assertIsNone(receipt["server_exit"])
        self.assertTrue(any("FileNotFoundError" in error for error in receipt["errors"]))
        self.assertFalse(receipt["cleanup"]["complete"])
        self.assertIsNotNone(server._supervisor.returncode)

    def test_launch_environment_is_explicit_not_ambient(self):
        previous = os.environ.get("SERVING_PROCESS_TEST_AMBIENT")
        os.environ["SERVING_PROCESS_TEST_AMBIENT"] = "must-not-leak"
        try:
            server = self.server()
            self.ready(server)
            receipt = server.cancel()
            self.assert_clean(server, receipt)
            self.assertIn("AMBIENT None", server.output_path.read_text())
            self.assertNotIn("SERVING_PROCESS_TEST_AMBIENT", receipt["env_keys"])
        finally:
            if previous is None:
                os.environ.pop("SERVING_PROCESS_TEST_AMBIENT", None)
            else:
                os.environ["SERVING_PROCESS_TEST_AMBIENT"] = previous

    def test_sigterm_drains_real_child_and_grandchild(self):
        server = self.server("tree")
        self.ready(server)
        descendant = int(next(line.split()[1] for line in server.output_path.read_text().splitlines()
                              if line.startswith("DESCENDANT ")))
        receipt = server.close(reason="drain_cell")
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["server_exit"]["returncode"], 0)
        self.assertFalse(receipt["server_exit"]["before_cleanup"])
        self.assertFalse(receipt["cleanup"]["escalated"])
        self.assertIsNone(process_identity(descendant))
        self.assertIn("CHILD_REAPED", server.output_path.read_text())
        self.assertIn("DRAINED", server.output_path.read_text())
        self.assertEqual(receipt["descendant_exits"], [])
        self.assertEqual([row["pid"] for row in receipt["descendant_retirements"]], [descendant])
        self.assertEqual(receipt["descendant_retirements"][0]["observation"], "absent")

    def test_stuck_server_escalates_and_preserves_signal(self):
        server = self.server("stuck")
        self.ready(server)
        started = time.monotonic()
        receipt = server.close()
        self.assertLess(time.monotonic() - started, 3)
        self.assert_clean(server, receipt)
        self.assertTrue(receipt["cleanup"]["escalated"])
        self.assertEqual(receipt["server_exit"]["returncode"], -signal.SIGKILL)
        self.assertEqual(receipt["server_exit"]["signal"], signal.SIGKILL)

    def test_graceful_drain_keeps_needed_worker_alive(self):
        server = self.server("drain-worker", drain_timeout=1.5)
        self.ready(server)
        descendant = int(next(line.split()[1] for line in server.output_path.read_text().splitlines()
                              if line.startswith("DESCENDANT ")))
        receipt = server.close(reason="drain_cell")
        self.assert_clean(server, receipt)
        raw = server.output_path.read_text()
        self.assertEqual(receipt["server_exit"]["returncode"], 0, raw)
        self.assertFalse(receipt["server_exit"]["before_cleanup"])
        self.assertIn("WORKER_RESULT INFLIGHT_COMPLETE exit=0", raw)
        self.assertIn("DRAINED", raw)
        self.assertNotIn("PREMATURE_TERM", raw)
        self.assertNotIn("INFLIGHT_LOST", raw)
        self.assertFalse(receipt["cleanup"]["escalated"])
        self.assertEqual([(e["pid"], e["signal"]) for e in receipt["signals"]
                          if e["result"] == "sent"], [(server.identity.pid, signal.SIGTERM)])
        self.assertIsNone(process_identity(descendant))
        self.assertEqual([row["pid"] for row in receipt["descendant_retirements"]], [descendant])
        print(json.dumps({"cpu_control": "primary_reaped_child",
                          "server_exit": receipt["server_exit"],
                          "descendant_retirements": receipt["descendant_retirements"],
                          "cleanup": receipt["cleanup"]}, sort_keys=True), flush=True)

    def test_original_nonzero_exit_and_signal_are_not_cleanup_status(self):
        for mode, code in [("exit7", 7), ("signal", -signal.SIGUSR1)]:
            with self.subTest(mode=mode):
                server = self.server(mode)
                receipt = server.wait()
                self.assert_clean(server, receipt)
                self.assertEqual(receipt["server_exit"]["returncode"], code)
                self.assertEqual(receipt["stop"]["reason"], "exit_before_ready")
                self.assertEqual(receipt["stop"]["server_returncode_before_cleanup"], code)

    def test_startup_and_autonomous_overall_deadlines(self):
        server = self.server(startup_timeout=0.3)
        receipt = server.wait()
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["stop"]["reason"], "startup_timeout")
        server = self.server(overall_timeout=0.6)
        self.ready(server)
        # No library call drives the deadline: the watchdog must act independently.
        time.sleep(1)
        receipt = server.wait()
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["stop"]["reason"], "overall_timeout")

    def test_cancel_does_not_signal_unrelated_child(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            server = self.server()
            self.ready(server)
            self.assertNotEqual(server.identity.pgid, os.getpgrp())
            receipt = server.cancel()
            self.assert_clean(server, receipt)
            self.assertEqual(receipt["stop"]["reason"], "cancelled")
            self.assertIsNone(foreign.poll())
            self.assertNotIn(foreign.pid, [event["pid"] for event in receipt["signals"]])
        finally:
            foreign.terminate()
            foreign.wait(timeout=2)

    def test_http_boolean_and_foreign_identity_cannot_mark_ready(self):
        server = self.server()
        with self.assertRaises(ValueError):
            server.mark_ready(True)
        with self.assertRaises(ValueError):
            server.mark_ready(ReadinessEvidence(process_identity(os.getpid()), "listener_identity",
                                                {"http_status": 200}))
        with self.assertRaises(ValueError):
            server.mark_ready(ReadinessEvidence(server.identity, "http_200", {"http_status": 200}))
        receipt = server.cancel()
        self.assert_clean(server, receipt)
        self.assertIsNone(receipt["ready"])

    def test_context_exception_and_wait_timeout_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "cell failed"):
            server = OwnedServer(argv=[sys.executable, str(self.script), "idle"], env={},
                                 cwd=str(self.root), evidence_dir=str(self.root / "context"),
                                 drain_timeout=0.2, kill_timeout=1)
            with server:
                raise RuntimeError("cell failed")
        self.assert_clean(server, server.receipt())
        self.assertEqual(server.receipt()["stop"]["reason"], "scope_exception")
        server = self.server()
        self.ready(server)
        receipt = server.wait(timeout=0.05)
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["stop"]["reason"], "caller_wait_timeout")

    @unittest.skipUnless(sys.platform == "linux", "actual subreaper adoption requires Linux")
    def test_linux_reaps_orphan_and_session_escaped_grandchild(self):
        for mode in ["orphan", "escaped", "stuck-tree"]:
            with self.subTest(mode=mode):
                server = self.server(mode)
                if mode == "stuck-tree":
                    self.ready(server)
                    receipt = server.cancel()
                else:
                    receipt = server.wait()
                self.assert_clean(server, receipt)
                self.assertTrue(receipt["linux_subreaper"])
                self.assertEqual(receipt["server_exit"]["returncode"],
                                 -signal.SIGKILL if mode == "stuck-tree" else 7)
                self.assertEqual(len(receipt["descendant_exits"]), 1)
                child = receipt["descendant_exits"][0]
                self.assertEqual(receipt["descendant_retirements"], [])
                self.assertEqual(child["returncode"], -signal.SIGKILL)
                self.assertIsNone(process_identity(child["pid"]))
                if mode == "stuck-tree":
                    term_pids = [e["pid"] for e in receipt["signals"]
                                 if e["signal"] == signal.SIGTERM and e["result"] == "sent"]
                    self.assertEqual(term_pids, [server.identity.pid])
                    killed = {e["pid"] for e in receipt["signals"]
                              if e["signal"] == signal.SIGKILL and e["result"] == "sent"}
                    self.assertEqual(killed, {server.identity.pid, child["pid"]})

    def test_controller_death_triggers_owned_cleanup(self):
        directory = self.root / "abandoned"
        code = ("from serving_process import OwnedServer; import os; "
                f"s=OwnedServer(argv={[sys.executable, str(self.script), 'idle']!r},env={{}},"
                f"cwd={str(self.root)!r},evidence_dir={str(directory)!r},"
                "startup_timeout=3,overall_timeout=5,drain_timeout=.2,kill_timeout=1); "
                "s.start(); os._exit(23)")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, timeout=4)
        self.assertEqual(result.returncode, 23)
        deadline = time.monotonic() + 4
        receipt = {}
        while time.monotonic() < deadline:
            path = directory / "receipt.json"
            if path.exists():
                receipt = json.loads(path.read_text())
                if receipt.get("state") == "finished":
                    break
            time.sleep(0.02)
        self.assertEqual(receipt["stop"]["reason"], "controller_eof")
        self.assertTrue(receipt["cleanup"]["complete"], receipt)
        self.assertIsNone(process_identity(receipt["server"]["pid"]))

    def test_killed_supervisor_fails_closed_and_stops_known_server(self):
        server = self.server()
        self.ready(server)
        os.kill(server._supervisor.pid, signal.SIGKILL)
        with self.assertRaises(ServingProcessError):
            server.close()
        receipt = server.receipt()
        self.assertIn("controller_error", receipt)
        self.assertFalse(receipt["cleanup"]["complete"])
        current = process_identity(server.identity.pid)
        self.assertTrue(current is None or current.state.startswith("Z"), receipt)
        self.assertTrue(any(e["pid"] == server.identity.pid and e["result"] == "sent"
                            for e in receipt["emergency_signals"]))

    def test_explicit_inputs_and_existing_evidence_refuse(self):
        options = dict(argv=[sys.executable, "-c", "pass"], env={}, cwd=str(self.root),
                       evidence_dir=str(self.root / "input"))
        for change in [dict(argv="echo bad"), dict(argv=["python", "-c", "pass"]),
                       dict(env=None), dict(overall_timeout=float("inf")), dict(kill_timeout=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                OwnedServer(**(options | change))
        (self.root / "input").mkdir()
        with self.assertRaises(FileExistsError):
            OwnedServer(**options).start()

    def test_actual_delayed_lookup_and_signal_syscall_are_bracketed(self):
        server = self.server("delayed-signal")
        self.ready(server)
        deadline = time.monotonic() + 2
        while "SIGNAL_READY" not in server.output_path.read_text():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.01)
        identity = server.identity
        original_lookup = serving_process.process_identity
        stamps = {}

        def lookup(pid):
            time.sleep(.08)
            answer = original_lookup(pid)
            stamps["lookup_finished"] = time.monotonic()
            return answer

        use_pidfd = (sys.platform == "linux" and hasattr(os, "pidfd_open")
                     and hasattr(signal, "pidfd_send_signal"))
        name = "pidfd_send_signal" if use_pidfd else "kill"
        namespace = serving_process.signal if use_pidfd else serving_process.os
        syscall = getattr(namespace, name)

        def delayed_send(*args, **kwargs):
            time.sleep(.08)
            stamps["syscall_started"] = time.monotonic()
            result = syscall(*args, **kwargs)  # Sends a real signal to our child.
            stamps["syscall_returned"] = time.monotonic()
            return result

        with patch.object(serving_process, "process_identity", side_effect=lookup), \
                patch.object(namespace, name, side_effect=delayed_send):
            event = serving_process._signal_process(identity, signal.SIGUSR1)
        self.assertEqual(event["result"], "sent")
        self.assertLess(event["monotonic"], stamps["lookup_finished"])
        self.assertLess(stamps["lookup_finished"], stamps["syscall_started"])
        self.assertLessEqual(stamps["syscall_returned"], event["sent_monotonic"])
        self.assertLessEqual(event["sent_monotonic"], event["finished_monotonic"])
        self.assertGreaterEqual(event["sent_monotonic"] - event["monotonic"], .16)
        receipt = server.wait()
        self.assert_clean(server, receipt)
        self.assertEqual(receipt["server_exit"]["returncode"], 0)
        self.assertIn("SIGNAL_DELIVERED", server.output_path.read_text())
        print(json.dumps({"cpu_control": "delayed_real_signal", "signal": event,
                          "actual_call_stamps": stamps, "server_exit": receipt["server_exit"]},
                         sort_keys=True), flush=True)

    def test_retirement_requires_actual_absence_not_lookup_error(self):
        server = self.server()
        self.ready(server)
        identity = server.identity
        self.assertIsNone(serving_process._observe_retirement(identity, time.monotonic()))
        receipt = server.cancel()
        self.assert_clean(server, receipt)
        retired = serving_process._observe_retirement(identity, receipt["stop"]["monotonic"])
        self.assertEqual(retired["observation"], "absent")
        self.assertIsNone(retired["exit_status"])
        with patch.object(serving_process.sys, "platform", "linux"), \
                patch.object(Path, "read_text", side_effect=PermissionError("unreadable proc stat")):
            with self.assertRaises(PermissionError):
                serving_process._observe_retirement(identity, time.monotonic())

    @unittest.skipUnless(sys.platform == "linux", "real Linux producer/policy census integration")
    def test_linux_policy_accepts_primary_reaped_and_supervisor_reaped_children(self):
        from serving_policy import _drain_lifecycle
        from serving_release import ServingGateError
        import copy

        for mode in ("drain-worker", "orphan-drain"):
            with self.subTest(mode=mode):
                server = self.server(mode, drain_timeout=1.5)
                self.ready(server)
                receipt = server.close(reason="drain_cell")
                self.assert_clean(server, receipt)
                config = {"server_identity": {"pid": server.identity.pid,
                          "start_identity": receipt["boot_id"] + ":" + server.identity.start_time},
                          "stop_reason": "drain_cell", "drain_timeout_ns": 1_500_000_000}
                result = _drain_lifecycle(config, receipt)  # Unmodified actual producer receipt.
                print(json.dumps({"cpu_control": "linux_producer_policy_" + mode,
                                  "lifecycle_result": result, "signals": receipt["signals"],
                                  "server_exit": receipt["server_exit"],
                                  "descendant_exits": receipt["descendant_exits"],
                                  "descendant_retirements": receipt["descendant_retirements"],
                                  "cleanup": receipt["cleanup"]}, sort_keys=True), flush=True)
                if mode == "drain-worker":
                    self.assertEqual(len(receipt["descendant_retirements"]), 1)
                    self.assertEqual(len(result["retired_without_wait_status"]), 1)
                    broken = copy.deepcopy(receipt)
                    broken["descendant_retirements"] = []
                    broken["cleanup"]["descendants_retired"] = 0
                else:
                    self.assertEqual(len(receipt["descendant_exits"]), 1)
                    self.assertEqual(result["retired_without_wait_status"], [])
                    broken = copy.deepcopy(receipt)
                    broken["descendant_exits"] = []
                    broken["cleanup"]["descendants_reaped"] = 0
                with self.assertRaisesRegex(ServingGateError, "lacks exit or identity retirement"):
                    _drain_lifecycle(config, broken)


if __name__ == "__main__":
    unittest.main()
