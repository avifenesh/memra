#!/usr/bin/env python3
"""Own a Linux probe and change only its live MEMRA_FAST byte, 0 -> 1 -> 0.

CLI: python3 native_env_controller.py --out NEW_DIR --timeout-seconds N -- COMMAND...
Python: run_controlled(command, out, timeout_seconds) -> result dictionary.
The fresh private directory contains stdout.log, stderr.log, protocol.log (raw
bytes, including rejected protocol lines), and result.json. Success requires both
rendezvous and exit 0. The exact child return code is always recorded separately;
CLI exit is 124 on timeout, 125 on controller/unsupported failure, otherwise the
child's exit code (128 + signal number for signal termination).

Only Linux, a mounted /proc in the same PID namespace, ordinary writable libc
exec-time environment storage, and permission to access our own child's mem are
supported. Yama/LSM/container denial is failure, never a successful skip. No PID,
address or environment-key targeting option exists. The only memory operations
are two-byte reads (the advertised value and its NUL) and a one-byte write at the
advertised getenv address; there is no memory or environ scan.
If host policy refuses this access, the concrete alternative is a dedicated Linux
VM/CI job permitting parent-to-child mem access, first qualified by the CPU tests
with --require-linux. Dependency injection or an in-process setter is not an
equivalent real-environment-drift gate. Native execution still needs owner review.

Trust boundary: COMMAND must be the cooperative probe itself, not a shell, test
driver that forks, or hostile executable. The child attests that the pointer is
libc.getenv("MEMRA_FAST"); without scanning memory the parent cannot independently
prove that attestation. No environment mutators, fork/exec after the handshake,
shared-VM processes outside the thread group, other debugger, or external SIGCONT
sender are allowed. The restore pointer must equal the mutate pointer. SIGSTOP
does not stop submitted GPU work; the caller must establish an idle boundary.

One monotonic deadline covers the child, protocol, stops and pipe drain. Both
pipes are drained concurrently to disk with bounded RAM. On failure SIGKILL is
sent to the owned session's process group, including a stopped child, and the
direct child is reaped. CLI SIGINT/SIGTERM/SIGHUP follow the same path. As with any
userspace supervisor, parent SIGKILL/crash, kernel hangs/uninterruptible sleep or
descendants escaping the owned group cannot be made recoverable here. Use an
external job/cgroup supervisor if those failures must also be covered. Never use
this controller as evidence of GPU qualification by itself.
"""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time


PREFIX = b"MEMRA_ENV_CONTROL"
MAX_LINE_BYTES = 64 * 1024
CHUNK_BYTES = 64 * 1024
PHASES = (("mutate", "0", "1"), ("restore", "1", "0"))
_INTERRUPTED_SIGNAL = None
FRAME = re.compile(
    rb"MEMRA_ENV_CONTROL\t(mutate|restore)\t([1-9][0-9]{0,9})"
    rb"\t([1-9][0-9]{0,18})\t([01])\t([01])\n"
)


class ControlError(RuntimeError):
    """A refused/failed control operation; messages never contain memory bytes."""


class ControlTimeout(ControlError):
    pass


def parse_handshake(line, owned_pid, index, original_address=None):
    """Strict ASCII, LF-only, standalone protocol. No caller-supplied env key."""
    match = FRAME.fullmatch(line)
    if match is None:
        raise ControlError("malformed environment control protocol")
    phase, pid, address, before, after = match.groups()
    phase, before, after = phase.decode(), before.decode(), after.decode()
    pid, address = int(pid), int(address)
    if not 0 <= index < len(PHASES) or (phase, before, after) != PHASES[index]:
        raise ControlError("expected exactly mutate 0->1 then restore 1->0")
    if pid != owned_pid:
        raise ControlError("protocol PID is not the owned direct child")
    # pread/pwrite offsets are signed off_t, including the terminator byte.
    if address > min(sys.maxsize, (1 << 63) - 1) - 1:
        raise ControlError("protocol address is outside the supported offset range")
    if index == 1 and address != original_address:
        raise ControlError("MEMRA_FAST address changed before restore")
    return {"phase": phase, "pid": pid, "address": address, "from": before, "to": after}


def require_linux():
    if sys.platform != "linux":
        raise ControlError("unsupported platform: Linux /proc environment control is required")
    if not Path("/proc/self/task").is_dir() or not Path("/proc/self/mem").exists():
        raise ControlError("unsupported Linux environment: /proc task and mem are required")


def check_deadline(deadline):
    if _INTERRUPTED_SIGNAL is not None:
        raise ControlError(f"environment controller interrupted by signal {_INTERRUPTED_SIGNAL}")
    if time.monotonic() >= deadline:
        raise ControlTimeout("environment controller deadline exceeded")


def _task_snapshot(pid):
    tasks = Path(f"/proc/{pid}/task")
    tids = sorted(int(entry.name) for entry in tasks.iterdir() if entry.name.isdecimal())
    if not tids or pid not in tids:
        raise ControlError("owned child task group is missing")
    for tid in tids:
        fields = {}
        for line in (tasks / str(tid) / "status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
        try:
            valid = (
                fields["State"].split()[0] == "T"
                and int(fields["Tgid"]) == pid
                and int(fields["Pid"]) == tid
                and int(fields["PPid"]) == os.getpid()
                and int(fields["TracerPid"]) == 0
                and int(fields["Threads"]) == len(tids)
            )
        except (KeyError, ValueError, IndexError):
            valid = False
        if not valid:
            raise ControlError("whole owned thread group is not in an untraced SIGSTOP state")
    return tids


def validate_stopped_group(pid):
    """Used only after waitpid(WUNTRACED) reports SIGSTOP for the owned child."""
    first = _task_snapshot(pid)
    if first != _task_snapshot(pid):
        raise ControlError("owned thread group changed during stop validation")
    return first


class _OwnedChild:
    """There is intentionally no constructor accepting a PID or memory address."""

    def __init__(self, command, environment=None):
        env = dict(os.environ if environment is None else environment)
        env["MEMRA_FAST"] = "0"
        self.process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        self.stopped = False

    def observe(self):
        if self.process.returncode is not None:
            return
        pid, status = os.waitpid(self.process.pid, os.WNOHANG | os.WUNTRACED)
        if pid == 0:
            return
        if os.WIFSTOPPED(status):
            if os.WSTOPSIG(status) != signal.SIGSTOP or self.stopped:
                raise ControlError("unexpected owned child stop")
            self.stopped = True
        elif os.WIFEXITED(status) or os.WIFSIGNALED(status):
            self.process.returncode = os.waitstatus_to_exitcode(status)
            self.stopped = False
        else:
            raise ControlError("unexpected owned child wait status")

    def change_and_resume(self, frame, deadline):
        if not self.stopped or self.process.returncode is not None:
            raise ControlError("no completed owned child SIGSTOP; refusing memory access")
        if frame["pid"] != self.process.pid:
            raise ControlError("protocol PID is not the owned direct child")
        check_deadline(deadline)
        pid = self.process.pid
        tasks = validate_stopped_group(pid)
        # Never read /proc/environ or search another address. Keep the mem fd
        # local to this stop and close it before resuming the process.
        fd = os.open(f"/proc/{pid}/mem", os.O_RDWR | os.O_CLOEXEC)
        try:
            address = frame["address"]
            expected = frame["from"].encode() + b"\0"
            replacement = frame["to"].encode()
            if os.pread(fd, 2, address) != expected:
                raise ControlError("MEMRA_FAST byte or terminating NUL did not match")
            if validate_stopped_group(pid) != tasks:
                raise ControlError("owned thread group changed before write")
            check_deadline(deadline)
            if os.pwrite(fd, replacement, address) != 1:
                raise ControlError("MEMRA_FAST single-byte write was incomplete")
            if os.pread(fd, 2, address) != replacement + b"\0":
                raise ControlError("MEMRA_FAST write readback did not match")
            if validate_stopped_group(pid) != tasks:
                raise ControlError("owned thread group changed before resume")
            check_deadline(deadline)
        finally:
            os.close(fd)
        os.kill(pid, signal.SIGCONT)
        self.stopped = False
        return {**frame, "stopped_threads": len(tasks), "readback_verified": True}

    def kill_and_reap(self):
        if self.process.poll() is not None:
            # Once reaped, the PID/PGID can be reused. Never signal it again.
            return
        # start_new_session makes the child's PID the owned process-group ID.
        # SIGKILL works on stopped tasks without briefly resuming GPU work.
        # Darwin may report EPERM for a group whose last member just exited;
        # direct-child kill() polls again and safely handles that race.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.kill()
        self.process.wait()


class _Capture:
    def __init__(self, child, stdout, stderr, protocol):
        self.selector = selectors.DefaultSelector()
        self.stdout = stdout
        self.protocol = protocol
        self.buffer = bytearray()
        self.lines = []
        for pipe, output in ((child.process.stdout, stdout), (child.process.stderr, stderr)):
            os.set_blocking(pipe.fileno(), False)
            self.selector.register(pipe, selectors.EVENT_READ, output)

    def pump(self, timeout, parse=True):
        # One bounded chunk per ready pipe, so a log flood cannot starve checks.
        for key, _ in self.selector.select(timeout):
            try:
                chunk = os.read(key.fileobj.fileno(), CHUNK_BYTES)
            except BlockingIOError:
                continue
            if not chunk:
                self.selector.unregister(key.fileobj)
                if key.data is self.stdout and self.buffer:
                    if PREFIX in self.buffer:
                        self.protocol.write(self.buffer)
                        self.buffer.clear()
                        if parse:
                            raise ControlError("unterminated environment control protocol")
                    self.buffer.clear()
                continue
            key.data.write(chunk)
            if key.data is self.stdout:
                self.buffer.extend(chunk)
                while b"\n" in self.buffer:
                    end = self.buffer.index(b"\n") + 1
                    line = bytes(self.buffer[:end])
                    del self.buffer[:end]
                    if PREFIX in line:
                        self.protocol.write(line)
                        if parse:
                            self.lines.append(line)
                    if parse and len(line) > MAX_LINE_BYTES:
                        raise ControlError("child stdout line exceeded the bounded protocol buffer")
                if len(self.buffer) > MAX_LINE_BYTES:
                    if PREFIX in self.buffer:
                        self.protocol.write(self.buffer)
                    self.buffer.clear()
                    if parse:
                        raise ControlError("child stdout line exceeded the bounded protocol buffer")

    def drain_after_kill(self):
        # The owned child is dead; retain queued bytes without parsing again.
        # A forbidden descendant retaining the pipe cannot extend this forever.
        end = time.monotonic() + 1.0
        while self.selector.get_map() and time.monotonic() < end:
            self.pump(0.01, parse=False)

    def close(self):
        self.selector.close()


def _monitor(child, capture, deadline, events, check_invariants=None):
    pending = None
    address = None
    while True:
        if check_invariants is not None:
            try:
                check_invariants()
            except Exception as error:
                raise ControlError(f"external invariant failed: {error}") from error
        check_deadline(deadline)
        capture.pump(min(0.01, max(0, deadline - time.monotonic())))
        for line in capture.lines:
            if pending is not None:
                raise ControlError("multiple handshakes before the preceding stop completed")
            pending = parse_handshake(line, child.process.pid, len(events), address)
        capture.lines.clear()
        child.observe()
        if child.process.returncode is not None:
            if pending is not None or len(events) != 2:
                raise ControlError("child exited without exactly two completed environment handshakes")
            if not capture.selector.get_map():
                return
        elif child.stopped and pending is not None:
            events.append(child.change_and_resume(pending, deadline))
            address = pending["address"]
            pending = None


def run_controlled(command, out, timeout_seconds, *, environment=None, check_invariants=None):
    """Run a direct child; return evidence, including failures. Never raises to skip.

    Invalid arguments or an existing output directory raise before launching.
    Importing callers must arrange interruption handling (the CLI handles it).
    """
    if (not isinstance(command, (list, tuple)) or not command
            or not all(isinstance(arg, str) and "\0" not in arg for arg in command)
            or not command[0]):
        raise ValueError("a nonempty direct child command is required")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
        raise ValueError("timeout-seconds must be finite and in (0, 86400]")
    out = Path(out)
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    deadline = started + timeout_seconds
    result = {"schema": "memra-native-env-control-v1", "status": "failed",
              "child_pid": None, "child_returncode": None, "controller_returncode": 125,
              "timeout_seconds": timeout_seconds, "events": [], "error": None}
    child = capture = None
    with contextlib.ExitStack() as stack:
        stdout, stderr, protocol = [stack.enter_context((out / name).open("xb", buffering=0))
                                    for name in ("stdout.log", "stderr.log", "protocol.log")]
        try:
            require_linux()
            check_deadline(deadline)
            child = _OwnedChild(command) if environment is None else _OwnedChild(command, environment)
            result["child_pid"] = child.process.pid
            capture = _Capture(child, stdout, stderr, protocol)
            _monitor(child, capture, deadline, result["events"], check_invariants)
            code = child.process.returncode
            result["controller_returncode"] = code if code >= 0 else 128 - code
            result["status"] = "passed" if code == 0 else "failed"
        except ControlTimeout as exc:
            result["error"] = str(exc)
            result["controller_returncode"] = 124
        except ControlError as exc:
            result["error"] = str(exc)
        except OSError as exc:
            # No raw exception path/content or memory bytes in controller errors.
            result["error"] = f"OS operation failed (errno={exc.errno}); Linux/proc/access required"
        except KeyboardInterrupt:
            result["error"] = "environment controller interrupted"
        finally:
            if child is not None:
                try:
                    if result["status"] != "passed":
                        child.kill_and_reap()
                    if capture is not None:
                        try:
                            capture.drain_after_kill()
                        except OSError:
                            result["status"] = "failed"
                            result["controller_returncode"] = 125
                            result["error"] = "failed to preserve child output during cleanup"
                finally:
                    result["child_returncode"] = child.process.returncode
                    if capture is not None:
                        capture.close()
                    child.process.stdout.close()
                    child.process.stderr.close()
            result["elapsed_seconds"] = time.monotonic() - started
            (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


@contextlib.contextmanager
def cleanup_signals():
    global _INTERRUPTED_SIGNAL
    def interrupted(signum, _frame):
        global _INTERRUPTED_SIGNAL
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        _INTERRUPTED_SIGNAL = signum
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {sig: signal.signal(sig, interrupted) for sig in signals}
    _INTERRUPTED_SIGNAL = None
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _INTERRUPTED_SIGNAL = None


def main(argv=None):
    global _INTERRUPTED_SIGNAL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        parser.error("use -- COMMAND... to specify the direct owned child")

    try:
        with cleanup_signals():
            result = run_controlled(args.command[1:], args.out, args.timeout_seconds)
    except (ValueError, OSError, ControlError) as exc:
        parser.exit(125, f"environment controller setup failed: {type(exc).__name__}\n")
    print(json.dumps(result, sort_keys=True))
    return result["controller_returncode"]


if __name__ == "__main__":
    sys.exit(main())
