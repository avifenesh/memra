#!/usr/bin/env python3
"""CPU-only controller regressions. No engine build, driver load or GPU APIs.

Portable checks: python3 -B test_native_env_controller.py -v
Linux qualification: python3 -B test_native_env_controller.py --require-linux -v
The latter refuses non-Linux hosts. Linux integration failures (including proc
policy denial or missing C compiler) are failures, never skips. Linux fixtures
use real libc getenv, four pthreads, SIGSTOP and the real owned-child controller.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_env_controller as control


def frame(phase="mutate", pid=123, address=456, before="0", after="1"):
    return f"MEMRA_ENV_CONTROL\t{phase}\t{pid}\t{address}\t{before}\t{after}\n".encode()


class ProtocolTests(unittest.TestCase):
    def test_exact_order_and_fixed_values(self):
        first = control.parse_handshake(frame(), 123, 0)
        self.assertEqual(first, {"phase": "mutate", "pid": 123, "address": 456,
                                 "from": "0", "to": "1"})
        second = frame("restore", before="1", after="0")
        self.assertEqual(control.parse_handshake(second, 123, 1, 456)["to"], "0")
        for line, index, address in [(second, -1, None), (second, 0, None), (frame(), 1, 456),
                                     (frame(), 2, 456), (second, 1, 999)]:
            with self.subTest(line=line, index=index), self.assertRaises(control.ControlError):
                control.parse_handshake(line, 123, index, address)

    def test_malformed_or_ambiguous_protocol_is_rejected(self):
        bad = [b" " + frame(), b"log: " + frame(), frame().rstrip(b"\n"),
               frame().replace(b"\n", b"\r\n"), frame() + b"extra",
               frame().replace(b"\t", b" "), frame().replace(b"mutate", b"MUTATE"),
               frame().replace(b"456", b"0456"), frame().replace(b"123", b"0123"),
               frame().replace(b"456", b"0x123"), frame(address=0), frame(address=-1),
               frame(address=1 << 63), frame(before="1"), frame(after="0"),
               frame(before="secret"), frame(after="secret"), frame(pid=124),
               frame().replace(b"MEMRA_ENV_CONTROL", b"MEMRA_OTHER_CONTROL"),
               frame().replace(b"mutate", b"mutate\tMEMRA_OTHER"), frame() + frame()]
        for line in bad:
            with self.subTest(line=line), self.assertRaises(control.ControlError):
                control.parse_handshake(line, 123, 0)

    def test_invalid_timeout_command_and_existing_output_never_spawn(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "_OwnedChild") as spawn:
            root = Path(directory)
            for timeout in (0, -1, float("nan"), float("inf"), 86401):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    control.run_controlled(["unused"], root / "new", timeout)
            for command in ([], "unused", [""], ["bad\0arg"], ["unused", None]):
                with self.subTest(command=command), self.assertRaises(ValueError):
                    control.run_controlled(command, root / "new", 1)
            with self.assertRaises(FileExistsError):
                control.run_controlled(["unused"], root, 1)
            spawn.assert_not_called()

    def test_unsupported_platform_is_failure_with_evidence_and_no_spawn(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control.sys, "platform", "darwin"), \
                patch.object(control, "_OwnedChild") as spawn:
            out = Path(directory) / "new"
            result = control.run_controlled(["unused"], out, 1)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["controller_returncode"], 125)
            self.assertIsNone(result["child_pid"])
            self.assertIn("unsupported platform", result["error"])
            self.assertEqual(json.loads((out / "result.json").read_text()), result)
            for name in ("stdout.log", "stderr.log", "protocol.log"):
                self.assertEqual((out / name).read_bytes(), b"")
            spawn.assert_not_called()

    def test_missing_proc_is_failure(self):
        with patch.object(control.sys, "platform", "linux"), patch.object(control.Path, "is_dir", return_value=False):
            with self.assertRaisesRegex(control.ControlError, "/proc"):
                control.require_linux()

    def test_monotonic_timeout_and_cancellation(self):
        with patch.object(control.time, "monotonic", return_value=10):
            with self.assertRaises(control.ControlTimeout):
                control.check_deadline(10)
            control.check_deadline(11)
        with patch.object(control, "_INTERRUPTED_SIGNAL", signal.SIGTERM):
            with self.assertRaisesRegex(control.ControlError, "interrupted"):
                control.check_deadline(float("inf"))


class StopAndMemoryTests(unittest.TestCase):
    def child(self, stopped=True):
        # Construct without launching solely for syscall-order fault injection.
        child = object.__new__(control._OwnedChild)
        child.process = SimpleNamespace(pid=123, returncode=None)
        child.stopped = stopped
        return child

    def test_waitpid_requires_wuntraced_sigstop(self):
        child = self.child(False)
        with patch.object(control.os, "waitpid", return_value=(123, (signal.SIGSTOP << 8) | 0x7f)) as wait:
            child.observe()
            self.assertTrue(child.stopped)
            wait.assert_called_once_with(123, os.WNOHANG | os.WUNTRACED)
        child = self.child(False)
        with patch.object(control.os, "waitpid", return_value=(123, (signal.SIGTSTP << 8) | 0x7f)):
            with self.assertRaises(control.ControlError):
                child.observe()

    def test_no_stop_or_incomplete_group_never_opens_memory(self):
        first = control.parse_handshake(frame(), 123, 0)
        for stopped in (False, True):
            with self.subTest(stopped=stopped), patch.object(control.os, "open") as opening, \
                    patch.object(control, "validate_stopped_group", side_effect=control.ControlError("running task")):
                with self.assertRaises(control.ControlError):
                    self.child(stopped).change_and_resume(first, float("inf"))
                opening.assert_not_called()

    def test_only_known_byte_is_written_after_full_stop_and_verified_before_continue(self):
        calls = []
        child = self.child()
        first = control.parse_handshake(frame(), 123, 0)
        def stopped(pid):
            calls.append(("stopped", pid))
            return [123, 124]
        def read(fd, length, address):
            calls.append(("read", fd, length, address))
            return b"1\0" if any(call[0] == "write" for call in calls) else b"0\0"
        def write(fd, value, address):
            calls.append(("write", fd, value, address))
            return 1
        with patch.object(control, "validate_stopped_group", side_effect=stopped), \
                patch.object(control.os, "open", return_value=7) as opening, \
                patch.object(control.os, "pread", side_effect=read, create=True), \
                patch.object(control.os, "pwrite", side_effect=write, create=True), \
                patch.object(control.os, "close") as closing, \
                patch.object(control.os, "kill", side_effect=lambda *args: calls.append(("signal", *args))):
            event = child.change_and_resume(first, float("inf"))
        opening.assert_called_once_with("/proc/123/mem", os.O_RDWR | os.O_CLOEXEC)
        closing.assert_called_once_with(7)
        self.assertEqual(calls, [("stopped", 123), ("read", 7, 2, 456), ("stopped", 123),
                                 ("write", 7, b"1", 456), ("read", 7, 2, 456),
                                 ("stopped", 123), ("signal", 123, signal.SIGCONT)])
        self.assertEqual(event["stopped_threads"], 2)
        self.assertTrue(event["readback_verified"])

    def test_invalid_value_nul_short_read_and_group_change_never_write(self):
        first = control.parse_handshake(frame(), 123, 0)
        for raw in (b"", b"0", b"1\0", b"x\0", b"0x"):
            with self.subTest(raw=raw), patch.object(control, "validate_stopped_group", return_value=[123]), \
                    patch.object(control.os, "open", return_value=7), patch.object(control.os, "close"), \
                    patch.object(control.os, "pread", return_value=raw, create=True), \
                    patch.object(control.os, "pwrite", create=True) as write, \
                    patch.object(control.os, "kill") as kill:
                with self.assertRaises(control.ControlError) as error:
                    self.child().change_and_resume(first, float("inf"))
                self.assertNotIn(repr(raw), str(error.exception))
                write.assert_not_called()
                kill.assert_not_called()
        with patch.object(control, "validate_stopped_group", side_effect=[[123], [123, 124]]), \
                patch.object(control.os, "open", return_value=7), patch.object(control.os, "close"), \
                patch.object(control.os, "pread", return_value=b"0\0", create=True), \
                patch.object(control.os, "pwrite", create=True) as write:
            with self.assertRaises(control.ControlError):
                self.child().change_and_resume(first, float("inf"))
            write.assert_not_called()

    def test_partial_write_bad_readback_and_access_denial_never_resume(self):
        first = control.parse_handshake(frame(), 123, 0)
        for written, readback in ((0, b"1\0"), (1, b"0\0"), (1, b"1x"), (1, b"")):
            with self.subTest(written=written, readback=readback), \
                    patch.object(control, "validate_stopped_group", return_value=[123]), \
                    patch.object(control.os, "open", return_value=7), patch.object(control.os, "close"), \
                    patch.object(control.os, "pread", side_effect=[b"0\0", readback], create=True), \
                    patch.object(control.os, "pwrite", return_value=written, create=True), \
                    patch.object(control.os, "kill") as kill:
                with self.assertRaises(control.ControlError):
                    self.child().change_and_resume(first, float("inf"))
                kill.assert_not_called()
        with patch.object(control, "validate_stopped_group", return_value=[123]), \
                patch.object(control.os, "open", side_effect=PermissionError(13, "denied")), \
                patch.object(control.os, "pwrite", create=True) as write, patch.object(control.os, "kill") as kill:
            with self.assertRaises(PermissionError):
                self.child().change_and_resume(first, float("inf"))
            write.assert_not_called()
            kill.assert_not_called()

    def test_task_scan_checks_every_thread_and_identity(self):
        def status(tid, **overrides):
            values = {"State": "T (stopped)", "Tgid": "123", "Pid": str(tid),
                      "PPid": str(os.getpid()), "TracerPid": "0", "Threads": "2", **overrides}
            return "\n".join(f"{key}:\t{value}" for key, value in values.items())
        for overrides in ({"State": "R (running)"}, {"State": "t (tracing stop)"},
                          {"State": "D"}, {"TracerPid": "99"}, {"Threads": "1"},
                          {"Tgid": "999"}, {"Pid": "999"}, {"PPid": "999"}, {}):
            with self.subTest(overrides=overrides), \
                    patch.object(control.Path, "iterdir", return_value=iter([Path("123"), Path("124")])), \
                    patch.object(control.Path, "read_text", side_effect=[status(123), status(124, **overrides)]):
                if overrides:
                    with self.assertRaises(control.ControlError):
                        control._task_snapshot(123)
                else:
                    self.assertEqual(control._task_snapshot(123), [123, 124])
        with patch.object(control, "_task_snapshot", side_effect=[[123], [123, 124]]):
            with self.assertRaises(control.ControlError):
                control.validate_stopped_group(123)


@unittest.skipUnless(os.name == "posix", "pipe/process CPU tests require POSIX")
class PortableProcessTests(unittest.TestCase):
    """Actual pipe pressure, malformed protocol, timeout, killing and reaping.

    Only platform preflight is mocked on macOS; no memory-operation success is
    mocked into a Linux qualification. These tests all expect controller failure.
    """
    def run_child(self, source, timeout=2):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "result"
        with patch.object(control, "require_linux"):
            return control.run_controlled([sys.executable, "-B", "-c", source], self.out, timeout)

    def assert_reaped(self, result):
        self.assertIsNotNone(result["child_returncode"])
        with self.assertRaises(ChildProcessError):
            os.waitpid(result["child_pid"], os.WNOHANG)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["child_pid"], 0)

    def test_running_timeout_preserves_pipes_and_reaps(self):
        result = self.run_child("import os,time; os.write(1,b'child-out\\n'); os.write(2,b'child-err\\n'); time.sleep(60)", .3)
        self.assertEqual(result["controller_returncode"], 124)
        self.assertEqual(result["child_returncode"], -signal.SIGKILL)
        self.assertEqual((self.out / "stdout.log").read_bytes(), b"child-out\n")
        self.assertEqual((self.out / "stderr.log").read_bytes(), b"child-err\n")
        self.assert_reaped(result)

    def test_stopped_child_without_protocol_times_out_and_is_reaped(self):
        result = self.run_child("import os,signal; os.kill(os.getpid(), signal.SIGSTOP)", .3)
        self.assertEqual(result["controller_returncode"], 124)
        self.assertEqual(result["child_returncode"], -signal.SIGKILL)
        self.assert_reaped(result)

    def test_protocol_without_stop_times_out_without_opening_mem(self):
        source = "import os,time; print(f'MEMRA_ENV_CONTROL\\tmutate\\t{os.getpid()}\\t456\\t0\\t1',flush=True); time.sleep(60)"
        with patch.object(control, "validate_stopped_group") as stopped:
            result = self.run_child(source, .3)
            stopped.assert_not_called()
        self.assertEqual(result["controller_returncode"], 124)
        self.assertIn(b"MEMRA_ENV_CONTROL", (self.out / "protocol.log").read_bytes())
        self.assert_reaped(result)

    def test_malformed_protocol_kills_stopped_child_and_preserves_raw(self):
        raw = b"prefix MEMRA_ENV_CONTROL\tmutate\t123\t456\t0\t1\n"
        result = self.run_child(f"import os,signal,time; os.write(1,{raw!r}); os.kill(os.getpid(),signal.SIGSTOP); time.sleep(60)")
        self.assertEqual(result["controller_returncode"], 125)
        self.assertEqual((self.out / "protocol.log").read_bytes(), raw)
        self.assert_reaped(result)

    def test_flooded_pipes_are_preserved_without_deadlock(self):
        # Far larger than either OS pipe buffer; the parent must service both.
        source = "import os; [(os.write(2,b'e'*8192),os.write(1,b'o'*8191+b'\\n')) for _ in range(128)]"
        result = self.run_child(source, 5)
        self.assertEqual(result["controller_returncode"], 125)  # missing handshake
        self.assertEqual(result["child_returncode"], 0)
        self.assertEqual((self.out / "stderr.log").read_bytes(), b"e" * 8192 * 128)
        self.assertEqual((self.out / "stdout.log").read_bytes(), (b"o" * 8191 + b"\n") * 128)
        self.assert_reaped(result)

    def test_fragmented_and_unterminated_protocol_is_preserved(self):
        raw = b"MEMRA_ENV_CONTROL\tmutate\t123\t456\t0\t1"
        result = self.run_child(f"import os,time; os.write(1,{raw[:8]!r}); time.sleep(.03); os.write(1,{raw[8:]!r})")
        self.assertEqual(result["controller_returncode"], 125)
        self.assertEqual((self.out / "stdout.log").read_bytes(), raw)
        self.assertEqual((self.out / "protocol.log").read_bytes(), raw)
        self.assert_reaped(result)

    def test_unbounded_stdout_line_fails_and_reaps(self):
        result = self.run_child("import os,time; os.write(1,b'x'*131072); time.sleep(60)")
        self.assertEqual(result["controller_returncode"], 125)
        self.assertIn("bounded", result["error"])
        self.assert_reaped(result)

    def test_spawn_failure_has_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "require_linux"):
            result = control.run_controlled(["/nonexistent/memra-cpu-fixture"], Path(directory) / "result", 1)
        self.assertEqual(result["controller_returncode"], 125)
        self.assertIsNone(result["child_pid"])

    def test_memory_access_denial_reaps_a_real_stopped_child(self):
        source = "import os,signal,time; print(f'MEMRA_ENV_CONTROL\\tmutate\\t{os.getpid()}\\t456\\t0\\t1',flush=True); os.kill(os.getpid(),signal.SIGSTOP); time.sleep(60)"
        # Skip only Linux proc validation here; never allow a mocked write.
        # Fault injection tests the controller's actual cleanup on denied open.
        real_open = os.open
        def denied(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith("/proc/") and path.endswith("/mem"):
                raise PermissionError(13, "denied")
            return real_open(path, *args, **kwargs)
        with patch.object(control, "validate_stopped_group", return_value=[123]), \
                patch.object(control.os, "open", side_effect=denied), \
                patch.object(control.os, "pwrite", create=True) as write:
            result = self.run_child(source)
            write.assert_not_called()
        self.assertEqual(result["controller_returncode"], 125)
        self.assertEqual(result["child_returncode"], -signal.SIGKILL)
        self.assertIn("errno=13", result["error"])
        self.assert_reaped(result)

    def test_cleanup_does_not_signal_an_already_reaped_pid(self):
        child = control._OwnedChild([sys.executable, "-B", "-c", "pass"])
        try:
            child.process.wait(timeout=5)
            with patch.object(control.os, "killpg") as killpg, patch.object(control.os, "kill") as kill:
                child.kill_and_reap()
                killpg.assert_not_called()
                kill.assert_not_called()
        finally:
            child.kill_and_reap()
            child.process.stdout.close()
            child.process.stderr.close()


# Compiled and executed only on Linux. Reads from getenv use copied bytes; there
# is no setenv/putenv, driver, engine or CUDA dependency even in failure modes.
LINUX_CHILD = r'''
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static atomic_int done = 0, ready = 0, seen_one = 0;
static void *reader(void *unused) {
    (void)unused;
    atomic_fetch_add(&ready, 1);
    while (!atomic_load(&done)) {
        const volatile char *p = getenv("MEMRA_FAST");
        char value = p ? *p : 'x';
        if (value != '0' && value != '1') _exit(81);
        if (value == '1') atomic_store(&seen_one, 1);
        usleep(100);
    }
    return NULL;
}
static uintptr_t snapshot(char expected) {
    const volatile char *p = getenv("MEMRA_FAST");
    if (!p || *p != expected || p[1] != '\0') _exit(82);
    return (uintptr_t)p;
}
static void send_frame(const char *phase, long pid, uintptr_t address, char from, char to) {
    printf("\nMEMRA_ENV_CONTROL\t%s\t%ld\t%lu\t%c\t%c\n", phase, pid,
           (unsigned long)address, from, to);
    fflush(stdout);
}
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "ok";
    snapshot('0');
    pthread_t threads[4];
    for (int i=0; i<4; ++i) if (pthread_create(&threads[i], NULL, reader, NULL)) return 83;
    while (atomic_load(&ready) != 4) usleep(100);
    // A retained CPU object stands in for lifecycle placement, not GPU proof.
    int *retained = malloc(sizeof(int));
    if (!retained) return 84;
    *retained = 42;
    puts("RETAINED_CPU_OBJECT"); fflush(stdout);
    fputs("cpu stderr\n", stderr); fflush(stderr);
    uintptr_t address = snapshot('0');
    char invalid[2] = {'0', 'x'};
    if (!strcmp(mode, "bad_nul")) address = (uintptr_t)invalid;
    long pid = (long)getpid() + (!strcmp(mode, "bad_pid") ? 100000 : 0);
    if (!strcmp(mode, "stop_only")) { raise(SIGSTOP); return 85; }
    send_frame("mutate", pid, address, '0', '1');
    if (!strcmp(mode, "duplicate")) send_frame("mutate", pid, address, '0', '1');
    if (!strcmp(mode, "no_stop")) { for (;;) pause(); }
    raise(SIGSTOP);
    if (snapshot('1') != address || *retained != 42) return 86;
    while (!atomic_load(&seen_one)) usleep(100);
    puts("REAL_ENV=1"); fflush(stdout);
    if (!strcmp(mode, "missing_restore")) return 0;
    send_frame("restore", pid, address, '1', '0');
    raise(SIGSTOP);
    if (snapshot('0') != address || *retained != 42) return 87;
    puts("REAL_ENV=0"); fflush(stdout);
    if (!strcmp(mode, "extra")) send_frame("restore", pid, address, '1', '0');
    atomic_store(&done, 1);
    for (int i=0; i<4; ++i) pthread_join(threads[i], NULL);
    free(retained);
    return !strcmp(mode, "exit7") ? 7 : 0;
}
'''


@unittest.skipUnless(sys.platform == "linux", "UNSUPPORTED here: actual Linux /proc child execution required")
class LinuxChildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="memra-env-cpu-")
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        source = root / "child.c"
        source.write_text(LINUX_CHILD)
        cls.binary = root / "child"
        subprocess.run(["cc", "-std=c11", "-D_DEFAULT_SOURCE", "-Wall", "-Wextra", "-Werror",
                        "-pthread", str(source), "-o", str(cls.binary)],
                       check=True, capture_output=True, timeout=60)

    def run_mode(self, mode, timeout=5):
        directory = tempfile.TemporaryDirectory(dir=self.tmp.name)
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name) / "out"
        return control.run_controlled([str(self.binary), mode], self.out, timeout)

    def assert_reaped(self, result):
        with self.assertRaises(ChildProcessError):
            os.waitpid(result["child_pid"], os.WNOHANG)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["child_pid"], 0)

    def test_real_getenv_mutation_and_restore_with_background_threads(self):
        result = self.run_mode("ok")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["child_returncode"], 0)
        self.assertEqual(result["controller_returncode"], 0)
        self.assertEqual([event["phase"] for event in result["events"]], ["mutate", "restore"])
        self.assertTrue(all(event["stopped_threads"] == 5 for event in result["events"]))
        self.assertEqual(result["events"][0]["address"], result["events"][1]["address"])
        output = (self.out / "stdout.log").read_bytes()
        self.assertLess(output.index(b"RETAINED_CPU_OBJECT"), output.index(b"REAL_ENV=1"))
        self.assertLess(output.index(b"REAL_ENV=1"), output.index(b"REAL_ENV=0"))
        self.assertEqual((self.out / "stderr.log").read_bytes(), b"cpu stderr\n")
        self.assertEqual(len((self.out / "protocol.log").read_bytes().splitlines()), 2)
        self.assert_reaped(result)

    def test_real_failures_kill_and_reap_without_skip(self):
        for mode in ("bad_pid", "bad_nul", "duplicate", "missing_restore", "extra"):
            with self.subTest(mode=mode):
                result = self.run_mode(mode)
                self.assertEqual(result["controller_returncode"], 125, result)
                self.assertEqual(result["status"], "failed")
                self.assert_reaped(result)

    def test_real_stopped_and_running_timeouts(self):
        for mode in ("stop_only", "no_stop"):
            with self.subTest(mode=mode):
                result = self.run_mode(mode, .5)
                self.assertEqual(result["controller_returncode"], 124, result)
                self.assertEqual(result["child_returncode"], -signal.SIGKILL)
                self.assertEqual(result["events"], [])
                self.assert_reaped(result)

    def test_child_nonzero_status_is_preserved_after_valid_protocol(self):
        result = self.run_mode("exit7")
        self.assertEqual(len(result["events"]), 2, result)
        self.assertEqual(result["child_returncode"], 7)
        self.assertEqual(result["controller_returncode"], 7)
        self.assertEqual(result["status"], "failed")
        self.assert_reaped(result)

    def test_cli_sigterm_kills_owned_stopped_child(self):
        with tempfile.TemporaryDirectory(dir=self.tmp.name) as directory:
            out = Path(directory) / "out"
            process = subprocess.Popen([sys.executable, "-B", control.__file__, "--out", str(out),
                                        "--timeout-seconds", "5", "--", str(self.binary), "stop_only"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if (out / "stdout.log").exists() and b"RETAINED_CPU_OBJECT" in (out / "stdout.log").read_bytes():
                        break
                    time.sleep(.01)
                else:
                    self.fail("CPU child did not start")
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 125)
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["child_returncode"], -signal.SIGKILL)
                with self.assertRaises(ProcessLookupError):
                    os.kill(result["child_pid"], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=6)


if __name__ == "__main__":
    if "--require-linux" in sys.argv:
        sys.argv.remove("--require-linux")
        if sys.platform != "linux":
            sys.exit("UNSUPPORTED: Linux CPU child qualification was not executed on this host")
    unittest.main()
