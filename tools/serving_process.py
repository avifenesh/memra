"""Owned POSIX server lifecycle; no HTTP, GPU or release-verdict policy.

Use OwnedServer as a context manager (or start/close explicitly). start() proves
process creation, not readiness. mark_ready() requires PID/start-time-bound
evidence collected by the caller from the owned output or listener. The caller
must validate the protocol/model response separately; health HTTP 200 is not an
ownership proof. A separate supervisor enforces deadlines even if the caller
blocks, and cleans up when its control pipe closes.

overall_timeout bounds the running lifetime, including startup. Cleanup then has
drain_timeout + kill_timeout, plus process observation/scheduling overhead.
Graceful TERM targets the primary first so its workers can finish in-flight work;
after its exit, orphans receive TERM. The deadline escalates all remaining owners.
Linux uses a private child subreaper and pidfds when available. Other POSIX hosts can
only reap direct children and track observed descendants; this is recorded, not
presented as Linux descendant-reaping proof. This is not a hostile-process sandbox.

Receipts preserve the server's wait status independently of cleanup. A zero exit
or cleanup.complete alone never establishes a successful serving cell. Environment
values are passed explicitly but only their canonical digest and keys are recorded.
Signal events bracket lookup/send with monotonic and finished_monotonic; sent_monotonic
is recorded only after a successful syscall. Cleanup has a direct finished_monotonic.
Every observed descendant has either a supervisor wait record or a timed identity
disappearance record. Disappearance proves retirement, not its exit status or reaper:
the primary can legitimately wait for its own child before the supervisor sees it.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


class ServingProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    pgid: int
    start_time: str
    identity_source: str
    state: str

    @property
    def key(self):
        return self.pid, self.start_time


@dataclass(frozen=True)
class ReadinessEvidence:
    owner: ProcessIdentity
    method: str  # "owned_output" or "listener_identity"; never just an HTTP status
    details: dict


def _parse_linux_identity(pid, raw):
    # comm can contain spaces and parentheses; fields after its final ')' are stable.
    fields = raw.rsplit(")", 1)[1].split()
    return ProcessIdentity(pid, int(fields[1]), int(fields[2]), fields[19],
                           "linux_proc_start_ticks", fields[0])


def _linux_identity(pid):
    try:
        return _parse_linux_identity(pid, Path(f"/proc/{pid}/stat").read_text())
    except (OSError, ValueError, IndexError):
        return None


def _process_table():
    if sys.platform == "linux":
        return {p.pid: p for entry in Path("/proc").iterdir()
                if entry.name.isdecimal() and (p := _linux_identity(int(entry.name)))}
    # ps is our own short-lived helper, explicitly excluded from server descendants.
    helper = subprocess.Popen(["/bin/ps", "-axo", "pid=,ppid=,pgid=,lstart=,stat="],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, error = helper.communicate(timeout=0.5)
    except subprocess.TimeoutExpired:
        helper.kill()
        helper.communicate(timeout=0.5)
        raise ServingProcessError("process inventory timed out")
    if helper.returncode:
        raise ServingProcessError(f"process inventory failed: {error.decode(errors='replace')}")
    result = {}
    for line in output.decode().splitlines():
        fields = line.split()
        if len(fields) >= 9 and int(fields[0]) != helper.pid:
            p = ProcessIdentity(int(fields[0]), int(fields[1]), int(fields[2]),
                                " ".join(fields[3:8]), "ps_lstart_seconds", fields[8])
            result[p.pid] = p
    return result


def process_identity(pid):
    """Observe an identity without signalling it; None means absent/unobservable."""
    return _linux_identity(pid) if sys.platform == "linux" else _process_table().get(pid)


def _owned(table, supervisor_pid, known):
    by_parent = {}
    for item in table.values():
        by_parent.setdefault(item.ppid, []).append(item)
    found, stack = {}, [supervisor_pid]
    while stack:
        for item in by_parent.get(stack.pop(), []):
            if item.pid not in found:
                found[item.pid] = item
                stack.append(item.pid)
    # Retain observed descendants that were reparented on a non-subreaper host.
    for key, old in known.items():
        current = table.get(old.pid)
        if current and current.key == key:
            found[current.pid] = current
    # An unreaped/observed member anchors the private group even when an orphan
    # is already reparented to init on non-Linux hosts. Never adopt our own group.
    groups = {item.pgid for item in found.values() if item.pgid not in (0, 1, os.getpgrp())}
    for item in table.values():
        if item.pgid in groups:
            found[item.pid] = item
    return found


def _signal_process(identity, signum):
    """Never signal an unverified PID; use kernel identity handles on Linux."""
    event = {"pid": identity.pid, "start_time": identity.start_time,
             "signal": int(signum), "monotonic": time.monotonic()}
    fd = None
    try:
        if sys.platform == "linux" and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(identity.pid)
            except OSError as error:
                if error.errno not in (errno.ENOSYS, errno.EINVAL):
                    raise
        current = process_identity(identity.pid)
        if current is None:
            event["result"] = "already_exited"
        elif current.key != identity.key:
            event["result"] = "identity_changed_refused"
        elif fd is not None:
            signal.pidfd_send_signal(fd, signum)
            event.update(method="pidfd", result="sent", sent_monotonic=time.monotonic())
        else:
            os.kill(identity.pid, signum)
            event.update(method="start_time_checked_pid", result="sent", sent_monotonic=time.monotonic())
    except ProcessLookupError:
        event["result"] = "already_exited"
    except OSError as error:
        event.update(result="error", error=str(error))
    finally:
        if fd is not None:
            os.close(fd)
        event["finished_monotonic"] = time.monotonic()
    return event


def _observe_retirement(identity, last_seen):
    """Check the old birth's absence; do not invent wait status for primary-reaped children."""
    began = time.monotonic()
    if sys.platform == "linux":
        try:
            current = _parse_linux_identity(identity.pid, Path(f"/proc/{identity.pid}/stat").read_text())
        except FileNotFoundError:
            current = None
        # Permission/read/parse failure is not absence: let it fail supervision.
    else:
        current = _process_table().get(identity.pid)
    ended = time.monotonic()
    if current is not None and current.key == identity.key:
        return None
    return {"pid": identity.pid, "identity": asdict(identity),
            "kind": "identity_disappeared", "exit_status": None,
            "last_seen_monotonic": last_seen, "started_monotonic": began,
            "finished_monotonic": ended,
            "observation": "absent" if current is None else "different_birth",
            "replacement_start_time": current.start_time if current is not None else None,
            "source": "linux_proc_stat" if sys.platform == "linux" else "ps_inventory"}


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _supervise(config):
    directory = Path(config["evidence_dir"])
    started = time.monotonic()
    record = {"schema": "memra-owned-server-v1", "state": "starting",
              "argv": config["argv"], "cwd": config["cwd"],
              "env_keys": sorted(config["env"]),
              "env_sha256": hashlib.sha256(json.dumps(config["env"], sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest(),
              "started_unix": time.time(), "started_monotonic": started,
              "timeouts": config["timeouts"], "supervisor": asdict(process_identity(os.getpid())),
              "linux_subreaper": False, "server": None, "ready": None, "stop": None,
              "server_exit": None, "descendant_exits": [], "descendant_retirements": [],
              "signals": [], "errors": [],
              "output_path": str(directory / "output.log")}
    events = (directory / "events.jsonl").open("x", buffering=1)

    def publish(event):
        record["observed_processes"] = [asdict(item) for item in known.values()]
        record["ownership_observations"] = [{"identity": asdict(known[key]), **times}
                                             for key, times in sightings.items()]
        events.write(json.dumps({"event": event, "monotonic": time.monotonic(),
                                 "state": record["state"]}) + "\n")
        _atomic_json(directory / "receipt.json", record)

    server, known, sightings = None, {}, {}
    resolved = set()
    stop_at = None
    stop_reason = None
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, _frame: interrupted.append(number))
    try:
        if sys.platform == "linux":
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER, private supervisor only
                raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
            record["linux_subreaper"] = True
            record["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        with (directory / "output.log").open("xb", buffering=0) as raw:
            server = subprocess.Popen(config["argv"], cwd=config["cwd"], env=config["env"],
                                      stdin=subprocess.DEVNULL, stdout=raw, stderr=subprocess.STDOUT,
                                      start_new_session=True, close_fds=True)
            owner = process_identity(server.pid)
            if owner is None or owner.pgid != server.pid:
                raise ServingProcessError("cannot establish isolated server identity")
            record["server"] = asdict(owner)
            known[owner.key] = owner
            seen_at = time.monotonic()
            sightings[owner.key] = {"first_seen_monotonic": seen_at, "last_seen_monotonic": seen_at}
            record["state"] = "running"
            publish("spawned")
            selector = selectors.DefaultSelector()
            os.set_blocking(sys.stdin.fileno(), False)
            selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
            pending = b""
            sent = set()
            remaining = {}
            try:
                while True:
                    now = time.monotonic()
                    remaining = _owned(_process_table(), os.getpid(), known)
                    known.update((item.key, item) for item in remaining.values())
                    seen_at = time.monotonic()
                    for item in remaining.values():
                        sightings.setdefault(item.key, {"first_seen_monotonic": seen_at})["last_seen_monotonic"] = seen_at
                    while True:
                        try:
                            pid, status = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            break
                        if not pid:
                            break
                        code = os.waitstatus_to_exitcode(status)
                        observed = remaining.get(pid)
                        if observed is None:
                            record["errors"].append(f"reaped PID {pid} without an observed birth identity")
                        else:
                            resolved.add(observed.key)
                        exit_row = {"pid": pid, "wait_status": status, "returncode": code,
                                    "exit_code": code if code >= 0 else None,
                                    "signal": -code if code < 0 else None,
                                    "identity": asdict(observed) if observed else None,
                                    "observed_monotonic": time.monotonic(),
                                    "before_cleanup": stop_at is None}
                        if pid == server.pid:
                            server.returncode = code
                            record["server_exit"] = exit_row
                        else:
                            record["descendant_exits"].append(exit_row)
                        remaining.pop(pid, None)
                        publish("reaped")
                    # waitpid(-1) cannot report children the primary already reaped.
                    # Cover each previously observed identity with a direct absence
                    # observation; never turn an unreadable stat into retirement.
                    for key, item in known.items():
                        if (item.pid == server.pid or key in resolved
                                or (item.pid in remaining and remaining[item.pid].key == key)):
                            continue
                        retired = _observe_retirement(item, sightings[key]["last_seen_monotonic"])
                        if retired is None:
                            remaining[item.pid] = item  # observation raced; keep cleanup pending
                        else:
                            record["descendant_retirements"].append(retired)
                            resolved.add(key)
                            publish("identity_retired")
                    if interrupted and stop_reason is None:
                        stop_reason = f"supervisor_signal_{interrupted[0]}"
                    if server.returncode is not None and stop_reason is None:
                        stop_reason = "server_exit" if record["ready"] else "exit_before_ready"
                    if now - started >= config["timeouts"]["overall"] and stop_reason is None:
                        stop_reason = "overall_timeout"
                    if (not record["ready"] and now - started >= config["timeouts"]["startup"]
                            and stop_reason is None):
                        stop_reason = "startup_timeout"
                    for _key, _mask in selector.select(timeout=0):
                        chunk = os.read(sys.stdin.fileno(), 65536)
                        if not chunk:
                            selector.unregister(sys.stdin.fileno())
                            stop_reason = stop_reason or "controller_eof"
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            command = json.loads(line)
                            if command["op"] == "stop":
                                stop_reason = stop_reason or command["reason"]
                            elif command["op"] == "ready" and stop_reason is None:
                                observed = ProcessIdentity(**command["proof"]["owner"])
                                current = process_identity(owner.pid)
                                if current is None or current.key != owner.key or observed.key != owner.key:
                                    record["errors"].append("readiness owner identity differs or exited")
                                    stop_reason = "readiness_identity_error"
                                else:
                                    record["ready"] = {**command["proof"], "monotonic": now}
                                    record["state"] = "ready"
                                    publish("ready")
                    if stop_reason is not None and stop_at is None:
                        stop_at = time.monotonic()
                        record["stop"] = {"reason": stop_reason, "monotonic": stop_at,
                                          "server_returncode_before_cleanup": server.returncode}
                        record["state"] = "stopping"
                        publish("stopping")
                    if stop_at is not None:
                        elapsed = time.monotonic() - stop_at
                        sig = signal.SIGTERM if elapsed < config["timeouts"]["drain"] else signal.SIGKILL
                        for item in remaining.values():
                            # Let the primary coordinate its workers during drain. Once
                            # it exits, terminate orphans; at the deadline kill everyone.
                            if (sig == signal.SIGTERM and server.returncode is None
                                    and item.pid != server.pid):
                                continue
                            key = (item.key, sig)
                            if key not in sent:
                                record["signals"].append(_signal_process(item, sig))
                                sent.add(key)
                                # Consumers must be able to observe the actual
                                # post-syscall result while the primary still drains.
                                # Publication failure follows the existing supervisor
                                # error/cleanup path; it cannot mint a sent receipt.
                                publish("signal_attempt")
                        if any(e["result"] == "error" for e in record["signals"]):
                            record["errors"] = list(dict.fromkeys(
                                record["errors"] + [e["error"] for e in record["signals"]
                                                    if e["result"] == "error"]))
                        if not remaining and server.returncode is not None and set(known) <= resolved:
                            break
                        if elapsed >= config["timeouts"]["drain"] + config["timeouts"]["kill"]:
                            break
                    time.sleep(0.02)
            finally:
                selector.close()
            cleanup_finished = time.monotonic()
            record["cleanup"] = {
                "complete": (not remaining and server.returncode is not None
                             and set(known) <= resolved and not record["errors"]),
                "escalated": any(e["signal"] == signal.SIGKILL and e["result"] == "sent"
                                 for e in record["signals"]),
                "remaining": [asdict(item) for item in remaining.values()],
                "descendants_reaped": len(record["descendant_exits"]),
                "descendants_retired": len(record["descendant_retirements"]),
                "finished_monotonic": cleanup_finished,
                "elapsed_s": cleanup_finished - stop_at,
                "raw_final": not remaining and set(known) <= resolved,
                "reaping_scope": "linux_subreaper" if record["linux_subreaper"] else "direct_children_only"}
    except BaseException as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
        # Even a supervisor exception gets bounded cleanup; no global waitpid in the caller.
        deadline = time.monotonic() + config["timeouts"]["kill"]
        remaining = {}
        while server is not None and time.monotonic() < deadline:
            try:
                remaining = _owned(_process_table(), os.getpid(), known)
            except Exception as inventory_error:
                record["errors"].append(f"cleanup inventory: {inventory_error}")
                # An unreaped direct child anchors its new session: its PID cannot
                # have been reused. Use only that private group if observation broke.
                if server.returncode is None:
                    try:
                        if os.getpgid(server.pid) == server.pid:
                            os.killpg(server.pid, signal.SIGKILL)
                            record["signals"].append({"pgid": server.pid, "signal": int(signal.SIGKILL),
                                                      "method": "unreaped_child_group", "result": "sent"})
                    except OSError as kill_error:
                        record["errors"].append(str(kill_error))
                remaining = {}
            for item in remaining.values():
                record["signals"].append(_signal_process(item, signal.SIGKILL))
            try:
                while (result := os.waitpid(-1, os.WNOHANG))[0]:
                    pid, status = result
                    code = os.waitstatus_to_exitcode(status)
                    if pid == server.pid:
                        server.returncode = code
                        record["server_exit"] = {"pid": pid, "wait_status": status,
                                                 "returncode": code, "exit_code": code if code >= 0 else None,
                                                 "signal": -code if code < 0 else None}
                    else:
                        record["descendant_exits"].append({"pid": pid, "wait_status": status,
                                                          "returncode": code})
            except ChildProcessError:
                break
            time.sleep(0.02)
        record["cleanup"] = {"complete": False, "remaining": [asdict(p) for p in remaining.values()],
                             "raw_final": False, "reason": "supervisor_error"}
    finally:
        record["state"] = "finished"
        record["finished_unix"] = time.time()
        output = directory / "output.log"
        record["output_bytes_observed"] = output.stat().st_size if output.exists() else 0
        publish("finished")
        events.close()


class OwnedServer:
    """An independently watched server with explicit argv/env/cwd and new receipt directory.

    mark_ready(ReadinessEvidence(...)) is a caller assertion with a checked process
    identity, not an HTTP probe. wait(timeout) initiates cleanup if its caller wait
    expires; close/cancel return the final raw lifecycle receipt. No method converts
    cancellation, escalation, missing readiness or process failure into serving success.
    """

    def __init__(self, *, argv, env, cwd, evidence_dir, startup_timeout=60,
                 overall_timeout=600, drain_timeout=10, kill_timeout=3):
        if os.name != "posix":
            raise ValueError("OwnedServer requires POSIX process sessions")
        if (not isinstance(argv, (list, tuple)) or not argv
                or any(not isinstance(v, str) or "\0" in v for v in argv)
                or not Path(argv[0]).is_absolute()):
            raise ValueError("argv must be strings with an absolute executable; no shell")
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          or not k or "=" in k or "\0" in k + v for k, v in env.items()):
            raise ValueError("env must be an explicit string mapping")
        cwd = Path(cwd)
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("cwd must name an existing absolute directory")
        self.directory = Path(evidence_dir)
        if not self.directory.is_absolute():
            raise ValueError("evidence_dir must be absolute and new")
        timeouts = {"startup": startup_timeout, "overall": overall_timeout,
                    "drain": drain_timeout, "kill": kill_timeout}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or v < 0 or (v == 0 and k != "drain") for k, v in timeouts.items()):
            raise ValueError("timeouts must be finite and positive (drain may be zero)")
        self._config = {"argv": list(argv), "env": dict(env), "cwd": str(cwd),
                        "evidence_dir": str(self.directory), "timeouts": timeouts}
        self._supervisor = None
        self._supervisor_identity = None
        self._started_at = None

    def start(self):
        if self._supervisor is not None:
            raise ServingProcessError("server scope already started")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._started_at = time.monotonic()
        with (self.directory / "supervisor.log").open("xb") as log:
            self._supervisor = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--supervise"],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        self._send(self._config)
        try:
            self._supervisor_identity = process_identity(self._supervisor.pid)
            result = self._await(lambda r: r.get("server") or r.get("state") == "finished",
                                 self._config["timeouts"]["startup"] + 2)
            if result.get("server") is None:
                raise ServingProcessError(f"server launch failed; see {self.directory}")
        except BaseException:
            self.close(reason="start_error")
            raise
        return self

    def receipt(self):
        failure = self.directory / "controller-error.json"
        if failure.exists():
            return json.loads(failure.read_text())
        try:
            return json.loads((self.directory / "receipt.json").read_text())
        except FileNotFoundError:
            return {}

    @property
    def identity(self):
        value = self.receipt().get("server")
        if value is None:
            raise ServingProcessError("server identity is not available")
        return ProcessIdentity(**value)

    @property
    def output_path(self):
        return self.directory / "output.log"

    def _send(self, value):
        if self._supervisor is None:
            raise ServingProcessError("start the server scope first")
        payload = (json.dumps(value) + "\n").encode()
        try:
            self._supervisor.stdin.write(payload)
            self._supervisor.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # The receipt/child exit, not a pipe write, decides the outcome.

    def _await(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while True:
            receipt = self.receipt()
            if predicate(receipt):
                return receipt
            if self._supervisor.poll() is not None:
                # The final atomic publication may race the preceding read.
                receipt = self.receipt()
                if predicate(receipt):
                    return receipt
                raise ServingProcessError(f"supervisor exited without the required receipt; see {self.directory}")
            if time.monotonic() >= deadline:
                raise TimeoutError("owned-server wait deadline expired")
            time.sleep(0.02)

    def mark_ready(self, proof):
        if (not isinstance(proof, ReadinessEvidence) or not isinstance(proof.owner, ProcessIdentity)
                or proof.method not in {"owned_output", "listener_identity"}
                or not isinstance(proof.details, dict) or not proof.details):
            raise ValueError("readiness requires owned process identity and nonempty protocol/output evidence")
        if proof.owner.key != self.identity.key:
            raise ValueError("readiness evidence belongs to another process")
        if self.receipt().get("ready"):
            raise ServingProcessError("readiness evidence is already recorded")
        value = asdict(proof)
        json.dumps(value, allow_nan=False)  # Validate before sending a partial command.
        self._send({"op": "ready", "proof": value})
        result = self._await(lambda r: r.get("ready") or r.get("state") == "finished",
                             self._config["timeouts"]["startup"] + 2)
        if result.get("state") == "finished" or not result.get("ready"):
            raise ServingProcessError("server stopped before readiness was accepted")
        return result

    def _close_control(self):
        try:
            self._supervisor.stdin.close()
        except BrokenPipeError:
            pass  # A buffered final command may race a completed supervisor.

    def _finish(self):
        limits = self._config["timeouts"]
        try:
            result = self._await(lambda r: r.get("state") == "finished",
                                 limits["drain"] + limits["kill"] + 2)
            self._close_control()
            code = self._supervisor.wait(timeout=2)
            if "supervisor_exit" not in result:
                result["supervisor_exit"] = {"returncode": code, "reaped": True}
                # The supervisor has exited, so no receipt writer remains.
                _atomic_json(self.directory / "receipt.json", result)
            if code != 0:
                raise ServingProcessError(f"supervisor exited with {code}")
            return result
        except (TimeoutError, ServingProcessError, subprocess.TimeoutExpired) as error:
            self._emergency_cleanup(error)
            raise ServingProcessError(f"supervisor failed; inspect {self.directory / 'controller-error.json'}") from error

    def _emergency_cleanup(self, error):
        """Best effort on known identities; lost supervision can NEVER be a clean receipt."""
        last = self.receipt()
        if last.get("controller_error"):
            return
        known = {}
        for row in last.get("observed_processes", []):
            item = ProcessIdentity(**row)
            known[item.key] = item
        if last.get("server"):
            item = ProcessIdentity(**last["server"])
            known[item.key] = item
        started = time.monotonic()
        limits = self._config["timeouts"]
        actions, errors, sent, remaining = [], [], set(), {}
        while time.monotonic() - started < limits["drain"] + limits["kill"]:
            try:
                supervisor = process_identity(self._supervisor.pid)
                same = (supervisor is not None and self._supervisor_identity is not None
                        and supervisor.key == self._supervisor_identity.key)
                remaining = _owned(_process_table(), supervisor.pid if same else -1, known)
                known.update((item.key, item) for item in remaining.values())
                sig = signal.SIGTERM if time.monotonic() - started < limits["drain"] else signal.SIGKILL
                for item in remaining.values():
                    if (item.key, sig) not in sent:
                        actions.append(_signal_process(item, sig))
                        sent.add((item.key, sig))
                if not remaining:
                    break
            except Exception as cleanup_error:
                errors.append(str(cleanup_error))
            time.sleep(0.02)
        if self._supervisor_identity is not None:
            actions.append(_signal_process(self._supervisor_identity, signal.SIGKILL))
        self._close_control()
        try:
            self._supervisor.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            errors.append("supervisor not reaped within emergency bound")
        failure = {**last, "state": "finished", "controller_error": str(error),
                   "supervisor_returncode": self._supervisor.returncode,
                   "emergency_signals": actions, "emergency_errors": errors,
                   "cleanup": {"complete": False, "raw_final": False,
                               "remaining_observed": [asdict(item) for item in remaining.values()],
                               "reason": "supervision lost; original exit/reaping may be unobservable"}}
        _atomic_json(self.directory / "controller-error.json", failure)

    def close(self, *, reason="scope_exit"):
        if self._supervisor is None:
            raise ServingProcessError("server scope was never started")
        if not isinstance(reason, str) or not reason:
            raise ValueError("stop reason must be a nonempty string")
        if self.receipt().get("controller_error"):
            raise ServingProcessError("server scope has a recorded supervisor failure")
        self._send({"op": "stop", "reason": reason})
        return self._finish()

    def cancel(self):
        return self.close(reason="cancelled")

    def wait(self, timeout=None):
        if self._supervisor is None:
            raise ServingProcessError("start the server scope first")
        limits = self._config["timeouts"]
        if timeout is None:
            timeout = max(0, limits["overall"] - (time.monotonic() - self._started_at))
            timeout += limits["drain"] + limits["kill"] + 2
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("wait timeout must be finite and nonnegative")
        try:
            self._await(lambda r: r.get("state") == "finished", timeout)
        except TimeoutError:
            return self.close(reason="caller_wait_timeout")
        except BaseException:
            self.close(reason="caller_interrupted")
            raise
        return self._finish()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, _value, _traceback):
        self.close(reason="scope_exception" if exc_type else "scope_exit")


if __name__ == "__main__":
    if sys.argv[1:] != ["--supervise"]:
        raise SystemExit("Import OwnedServer; the supervisor entry point is private")
    _supervise(json.loads(sys.stdin.buffer.readline()))
