"""Owned-server, phase-driven cancellation capture. Never native/C4 qualification.

collect_phase_cancel_cell(required, program, *, server: OwnedServer, output)
borrows a live ready server; it never starts/stops/signals that server. required
is the trusted manifest cell. program is frozen before any HTTP and has EXACT keys:

  schema="memra-phase-cancel-program-v1", mode="phase_cancel", cell_id, scope,
  server_identity={pid,start_identity}, client_trace_key=<32 lowercase hex>,
  trace={worker_generation,worker_route,quantum_routes:[...]},
  endpoint={host:<literal loopback>,port},
  http={connect_timeout,read_timeout,wall_timeout,max_body_bytes},
  timing={trigger_timeout_s,retirement_timeout_s,poll_s,overall_s}, requests=[...].

Requests use serving_cancel's exact role/id/model/wire/path/payload/usage shape:
one target, nonempty recovery, and peers for queued/decode. Only the target gets
x-memra-trace-id. X-Request-Id remains a client label, never a minted server ID.
Requests are sent once. Peers+target run concurrently; recovery follows complete
pressure captures and validated target retirement. Fatal errors cancel owned
clients; unlaunched requests remain explicit in the full planned denominator.
Poll waits wake on cancellation and are clipped to the phase deadline. Client-group
budgets include the complete trigger/retirement watcher, not just its HTTP capture.
All owned invocations and target HTTP threads settle before the log is closed or
capture.json is finalized. Cancellation cleanup has a two-second settlement budget;
an overrun is recorded as failure, and finalization still waits for actual exit
rather than returning an index that live workers can mutate. This cannot impose a
hard return deadline on an OS/filesystem stall in an unkillable Python thread.

The target client's owning thread watches server.output_path while one bounded
HTTP thread captures its response. Only a matching *latest* queued, actual open
quantum-start or decode-start record can set its private cancellation Event.
Prospective selection uses strict 7b syntax/binding checks, not a phase callback
or timer. Full validate_cancel_trace must subsequently prove the target history;
its recorded drop/quantum can expose a stale or false prospective trigger.

Log reads pin a regular, nonsymlink file, recheck path/FD inode and full previously
observed prefix, retain every appended byte (including unfinished lines), and parse
only complete LF prefixes through the reviewed framing parser. No-record-yet is
the ONLY parser refusal treated as a wait condition. Changed observed bytes, inode,
truncation (including shrinkage from a post-read stat), malformed lines or binding
drift fail. This is sampled filesystem evidence, not a claim that unseen transient rewrites are impossible. Local regular
file operations are bounded by bytes; OS/filesystem stalls are not killable Python
threads. The owner must provide its local evidence filesystem.
Before any HTTP, the baseline must end at a complete, stable LF prefix within
trigger_timeout_s; an unfinished old row must not conceal reuse of the target key.

Host monotonic read/set/finish timestamps and byte watermarks stay distinct from
producer trace_elapsed_ns. Trigger prefix and final full-log hashes/refs are kept.
The fixed log bound is 64MiB and at most10000 read observations; the frozen poll/
overall budget must fit. Raw append chunks reconstruct the initial-to-final log.
Failures retain available raw bytes, bad snapshot refs, process/listener/probe
observations, errors and planned/captured/unattempted IDs, not synthetic successes.

Result schema memra-phase-cancel-capture-v1 uses EvidenceStore BlobRefs, including
raw HTTP observation/body refs. state is captured or failed; qualification is
ALWAYS False. Full target trace, wire/terminal/error, peer/recovery and generation
checks run before captured. The parent still authenticates required/program/source/
binary/controller/process/model/lease/whole-C4 evidence. Site and host quantum
return do not prove GPU quiescence. Synthetic producer tests are not native proof.
"""

from dataclasses import asdict
import hashlib
import ipaddress
import math
import os
import stat
import threading
import time

from serving_cancel import _REQUIREMENTS, _positive, evaluate_cancel_wire, validate_cancel_requests
from serving_capture import EvidenceStore, collect_clients, encoded
from serving_completion import _keys, _same
from serving_http import capture_request
from serving_listener import ListenerOwnershipError, prove_listener
from serving_policy import _identity
from serving_process import OwnedServer, process_identity
from serving_release import ServingGateError, account_attempts, json_object, require
from serving_trace import _KEY, decode_lifecycle_log, validate_cancel_trace


_MAX_LOG = 64 * 1024 * 1024
_MAX_READS = 10000
_U64 = 2**64 - 1


def validate_phase_program(required, program):
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required phase cell")
    scenario = required["scenario"]
    require(isinstance(scenario, str) and scenario in _REQUIREMENTS
            and _same(required["requirements"], _REQUIREMENTS[scenario]), "unknown phase requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "required phase scope")
    require(all(type(v) is str and v.strip() for v in scope.values())
            and required["id"] == scope["id"] + "/" + scenario, "invalid phase scope")
    _keys(program, {"schema", "mode", "cell_id", "scope", "server_identity", "client_trace_key",
                    "trace", "endpoint", "http", "timing", "requests"}, "phase program")
    require(program["schema"] == "memra-phase-cancel-program-v1" and program["mode"] == "phase_cancel"
            and program["cell_id"] == required["id"] and _same(program["scope"], scope), "phase program differs")
    _identity(program["server_identity"])
    require(type(program["client_trace_key"]) is str and _KEY.fullmatch(program["client_trace_key"]),
            "invalid target client trace key")
    trace = program["trace"]
    _keys(trace, {"worker_generation", "worker_route", "quantum_routes"}, "expected trace")
    require(type(trace["worker_generation"]) is int and 0 <= trace["worker_generation"] <= _U64,
            "invalid expected worker epoch")
    routes = trace["quantum_routes"]
    require(type(routes) is list and len(routes) <= 32, "invalid expected quantum routes")
    for value in [trace["worker_route"], *routes]:
        require(type(value) is str and value, "invalid expected route")
        try:
            require(len(value.encode("utf-8")) <= 256, "invalid expected route")
        except UnicodeError as error:
            raise ServingGateError("invalid expected route Unicode") from error
    require(len(set(routes)) == len(routes) and (scenario != "cancel_prime" or routes), "missing/duplicate quantum routes")
    _keys(program["endpoint"], {"host", "port"}, "endpoint")
    require(type(program["endpoint"]["host"]) is str, "endpoint must be literal loopback")
    try:
        address = ipaddress.ip_address(program["endpoint"]["host"])
    except ValueError as error:
        raise ServingGateError("endpoint must be literal loopback") from error
    require(address.is_loopback, "endpoint must be literal loopback")
    require(type(program["endpoint"]["port"]) is int and 1 <= program["endpoint"]["port"] <= 65535, "invalid endpoint port")
    _keys(program["http"], {"connect_timeout", "read_timeout", "wall_timeout", "max_body_bytes"}, "HTTP limits")
    require(all(_positive(program["http"][n]) for n in ("connect_timeout", "read_timeout", "wall_timeout"))
            and type(program["http"]["max_body_bytes"]) is int and 0 < program["http"]["max_body_bytes"] <= _MAX_LOG,
            "invalid bounded HTTP limits")
    timing = program["timing"]
    _keys(timing, {"trigger_timeout_s", "retirement_timeout_s", "poll_s", "overall_s"}, "phase timing")
    require(all(_positive(v) for v in timing.values()) and timing["overall_s"] <= 3600
            and .01 <= timing["poll_s"] <= min(timing["trigger_timeout_s"], timing["retirement_timeout_s"])
            and timing["trigger_timeout_s"] + timing["retirement_timeout_s"] < timing["overall_s"]
            and math.ceil(timing["overall_s"] / timing["poll_s"]) + 16 <= _MAX_READS,
            "invalid phase timing/read budget")
    roles = validate_cancel_requests(program["requests"], scope=scope, scenario=scenario)
    encoded(program)  # Finite canonical JSON, no caller-owned mutable request objects.
    return roles


class _LogFailure(ServingGateError):
    def __init__(self, message, observed=None, metadata=None):
        super().__init__(message)
        self.observed, self.metadata = observed, metadata or {}


class _PinnedLog:
    def __init__(self, path):
        self.path, self.fd, self.raw = path, None, b""
        self.reads = 0
        opened_before = time.monotonic_ns()
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode), "owned log is not a regular file")
            self.identity = info.st_dev, info.st_ino
            self.observed_size = info.st_size
            self.descriptor = {"path": str(path), "device": info.st_dev, "inode": info.st_ino,
                "controller_pid": os.getpid(), "fd": self.fd, "regular_file": True,
                "opened_before_ns": opened_before, "opened_after_ns": time.monotonic_ns()}
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _observe_size(self, size):
        previous = self.observed_size
        self.observed_size = max(previous, size)
        require(size >= previous, "owned log truncated below observed size")
        require(size <= _MAX_LOG, "owned log exceeds byte bound")

    def read(self):
        metadata = {"started_ns": time.monotonic_ns()}
        data = None
        try:
            self.reads += 1
            require(self.reads <= _MAX_READS, "log read observation limit exceeded")
            before, named = os.fstat(self.fd), os.stat(self.path, follow_symlinks=False)
            metadata.update(device=before.st_dev, inode=before.st_ino, size_before=before.st_size,
                            named_device=named.st_dev, named_inode=named.st_ino)
            require(stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == self.identity
                    and (before.st_dev, before.st_ino) == self.identity, "owned log inode/replacement differs")
            self._observe_size(before.st_size)
            self._observe_size(named.st_size)
            # Full-prefix reread is deliberate: stat/append offsets alone miss in-place edits.
            data = os.pread(self.fd, before.st_size, 0)
            after, named_after = os.fstat(self.fd), os.stat(self.path, follow_symlinks=False)
            metadata.update(size_after=named_after.st_size, named_after_device=named_after.st_dev,
                            named_after_inode=named_after.st_ino)
            require((after.st_dev, after.st_ino) == self.identity
                    and stat.S_ISREG(named_after.st_mode)
                    and (named_after.st_dev, named_after.st_ino) == self.identity, "owned log replaced during read")
            self._observe_size(after.st_size)
            self._observe_size(named_after.st_size)
            require(len(data) == before.st_size, "owned log truncated during read")
            require(len(data) >= len(self.raw), "owned log truncated")
            require(data.startswith(self.raw), "owned log observed prefix mutated")
            metadata.update(finished_ns=time.monotonic_ns(), bytes=len(data),
                            complete_bytes=data.rfind(b"\n") + 1,
                            sha256=hashlib.sha256(data).hexdigest())
            offset, delta = len(self.raw), data[len(self.raw):]
            self.raw = data
            return metadata, offset, delta
        except (OSError, ServingGateError) as error:
            metadata["finished_ns"] = time.monotonic_ns()
            raise _LogFailure(f"owned log read failed: {error}", data, metadata) from error


def _decode_prefix(raw):
    length = raw.rfind(b"\n") + 1
    if length == 0:
        return None
    try:
        return decode_lifecycle_log(raw[:length])
    except ServingGateError as error:
        if error.code == "invalid_lifecycle" and str(error) == "no lifecycle records":
            return None  # Framing/UTF-8 checks already ran over all complete ordinary lines.
        raise


def _target_prefix(decoded, required, program):
    """Prospective syntax/bindings only. Never catches final-history refusals."""
    if decoded is None:
        return []
    records = decoded["records"]
    keys = {(x["record"]["pid"], x["record"]["trace_id"]) for x in records
            if x["record"]["client_trace_key"] == program["client_trace_key"]}
    require(len(keys) <= 1, "ambiguous target key in observed log")
    if not keys:
        return []
    key = next(iter(keys))
    require(key[0] == program["server_identity"]["pid"], "target log belongs to foreign PID")
    selected = [x for x in records if (x["record"]["pid"], x["record"]["trace_id"]) == key]
    target = next(r for r in program["requests"] if r["role"] == "target")
    minted = set()
    for item in selected:
        row = item["record"]
        require(row["sequence_valid"] and row["first_error"] is None and row["suppressed_events"] == 0
                and row["event"] not in ("overflow", "requeued"), "invalid target prefix: " + str(row["first_error"]))
        require(row["http_route"] == target["path"], "target HTTP route mismatch")
        require(row["client_trace_key"] in (None, program["client_trace_key"]), "target correlation key drift")
        if row["request_id"] is not None:
            minted.add(row["request_id"])
            require(row["model"] == required["scope"]["model"], "target model mismatch")
        if row["worker_generation"] is not None:
            require(row["worker_generation"] == program["trace"]["worker_generation"]
                    and row["worker_route"] == program["trace"]["worker_route"], "target worker binding mismatch")
        if row["quantum"] is not None:
            require(row["quantum"]["route"] in program["trace"]["quantum_routes"], "target quantum route mismatch")
    require(len(minted) <= 1, "target minted ID changed")
    return selected


def _prospective_phase(selected, scenario):
    if not selected:
        return None
    require(not any(x["record"]["event"] in ("http_pending_drop", "http_body_drop", "http_body_eof",
                    "receiver_closed", "retired", "trace_end") for x in selected), "target ended/closed before phase trigger")
    last = selected[-1]
    row = last["record"]
    if not row["bindings_complete"] or row["request_id"] is None or row["worker_generation"] is None:
        return None
    if scenario == "cancel_queued":
        ready = row["event"] == "queued" and row["phase"] == "queued" and not row["quantum_active"]
    elif scenario == "cancel_prime":
        ready = row["event"] == "prime_quantum_start" and row["phase"] == "prime" and row["quantum_active"]
    else:
        ready = row["event"] == "decode_start" and row["phase"] == "decode" and not row["quantum_active"]
    return last if ready else None


def collect_phase_cancel_cell(required, program, *, server, output):
    roles = validate_phase_program(required, program)
    require(isinstance(server, OwnedServer), "phase capture requires an actual OwnedServer")
    required, program = json_object(encoded(required)), json_object(encoded(program))
    roles = validate_phase_program(required, program)
    target = roles["target"][0]
    store, store_lock = EvidenceStore(output), threading.RLock()
    started = time.monotonic_ns()
    deadline = time.monotonic() + program["timing"]["overall_s"]
    state = {"schema": "memra-phase-cancel-capture-v1", "state": "collecting", "qualification": False,
             "clock": "monotonic_ns",
             "started_ns": started, "program": store.obj(program), "required": store.obj(required),
             "observations": [], "probes": [], "listeners": [], "process_observations": [],
             "request_invocations": [],
             "log_chunks": [], "log_observations": [], "errors": [], "client_errors": {},
             "trigger_failures": [],
             "unattempted_ids": [r["id"] for r in program["requests"]]}
    attempts, health, triggers, target_facts = [], [], [], []
    reader, probe_count = None, 0
    cached_prefix_length, cached_decoded = -1, None
    invoked_ids = set()
    after_listener_done = False
    workers, worker_lock = [], threading.Lock()
    stopping = threading.Event()

    def register_worker(thread, done, cancel):
        with worker_lock:
            workers.append((thread, done, cancel))
            if stopping.is_set():
                cancel.set()

    def settle_workers():
        # No final index/descriptor closure while an invocation can still write.
        # Keep own done events: interrupted Thread.join can falsely mark a thread
        # stopped on older CPython. No daemon/summary flag substitutes for exit.
        stop_deadline = time.monotonic() + 2
        overdue = False
        while True:
            with worker_lock:
                current = list(workers)
            pending = []
            for thread, done, cancel in current:
                if stopping.is_set():
                    cancel.set()
                if not done.is_set():
                    pending.append((thread, done))
                elif thread.ident is not None:
                    try:
                        thread.join(timeout=.05)
                    except BaseException as error:
                        stopping.set()
                        state["errors"].append(f"owned client join interrupted: {type(error).__name__}: {error}")
                        state["state"] = "failed"
                    if thread.is_alive():
                        pending.append((thread, done))
            if not pending:
                break
            if time.monotonic() >= stop_deadline and not overdue:
                overdue = True
                stopping.set()
                state["errors"].append("owned clients exceeded two-second cancellation settlement budget")
                state["state"] = "failed"
            for _, done in pending:
                try:
                    done.wait(.05)
                except BaseException as error:
                    stopping.set()
                    state["errors"].append(f"owned client settlement interrupted: {type(error).__name__}: {error}")
                    state["state"] = "failed"

    def wait_poll(cancel, limit, done=None):
        end = min(limit, time.monotonic() + program["timing"]["poll_s"])
        observe_completion = done is not None and not done.is_set()
        while not cancel.is_set():
            left = end - time.monotonic()
            if left <= 0 or (observe_completion and done.is_set()):
                break
            cancel.wait(min(.05, left))

    def save():
        with store_lock:
            store.index(state)

    def remaining():
        left = deadline - time.monotonic()
        require(left > 0, "overall phase capture deadline exceeded")
        return left

    def live():
        receipt, owner = server.receipt(), server.identity
        actual = process_identity(owner.pid)
        require(actual is not None and actual.key == owner.key and actual.pgid == owner.pgid
                and actual.ppid == owner.ppid and actual.state not in {"Z", "X", "x"}
                and receipt.get("state") != "finished" and not receipt.get("stop")
                and receipt.get("ready") and not receipt.get("errors"), "owned server not live/ready")
        require(isinstance(receipt.get("boot_id"), str) and receipt["boot_id"]
                and _same(program["server_identity"], {"pid": owner.pid,
                   "start_identity": receipt["boot_id"] + ":" + owner.start_time}), "owned server birth mismatch")
        return owner

    def process_snapshot(label):
        before = time.monotonic_ns()
        receipt = server.receipt()
        actual = process_identity(server.identity.pid)
        with store_lock:
            state["process_observations"].append(store.obj({"label": label, "started_ns": before,
                "finished_ns": time.monotonic_ns(), "receipt": receipt,
                "observed_identity": asdict(actual) if actual is not None else None}))

    def ownership(label):
        nonlocal after_listener_done
        end = time.monotonic() + min(3.0, remaining())
        while True:
            owner = live()
            try:
                proof = prove_listener(owner, **program["endpoint"], timeout=max(.001, end - time.monotonic()))
                with store_lock:
                    state["listeners"].append(store.obj({"label": label, "proof": asdict(proof)}))
                save()
                if label == "after":
                    after_listener_done = True
                return
            except ListenerOwnershipError as error:
                with store_lock:
                    state["listeners"].append(store.obj({"label": label, "error": str(error), "observed_ns": time.monotonic_ns()}))
                save()
                if time.monotonic() >= end:
                    raise
                time.sleep(max(0, min(.02, end - time.monotonic())))

    def read_log():
        nonlocal cached_prefix_length, cached_decoded
        live()
        with store_lock:
            try:
                metadata, offset, delta = reader.read()
            except _LogFailure as error:
                failure = {**error.metadata, "error": str(error)}
                if error.observed is not None:
                    failure["observed_raw"] = store.put(error.observed)
                state["log_observations"].append(store.obj(failure))
                save()
                raise
            if delta:
                state["log_chunks"].append({"offset": offset, "bytes": len(delta), "raw": store.put(delta)})
            state["log_observations"].append(store.obj(metadata))
            save()
        # The same reviewed7b decoder checks EVERY complete line. Partial tail is retained.
        if metadata["complete_bytes"] != cached_prefix_length:
            cached_decoded = _decode_prefix(reader.raw)
            cached_prefix_length = metadata["complete_bytes"]
        return metadata, cached_decoded

    def trace_args():
        return dict(scenario=required["scenario"], client_trace_key=program["client_trace_key"],
            expected_server_identity=program["server_identity"], expected_model=required["scope"]["model"],
            expected_http_route=target["path"], expected_worker_route=program["trace"]["worker_route"],
            expected_worker_generation=program["trace"]["worker_generation"],
            expected_quantum_routes=program["trace"]["quantum_routes"])

    def http(label, path, body, method, cancelled=None, is_target=False):
        live()
        headers = {"X-Request-Id": label, "Content-Type": "application/json"}
        if is_target:
            headers["x-memra-trace-id"] = program["client_trace_key"]
        if method == "POST":
            with store_lock:
                invoked_ids.add(label)
                state["unattempted_ids"].remove(label)
                state["request_invocations"].append(store.obj({"id": label, "started_ns": time.monotonic_ns(),
                    "stage": "before_capture_request", "path": path, "server_identity": program["server_identity"]}))
                save()
        result = capture_request(request_id=label, path=path, method=method, body=body,
            headers=headers, cancel_event=cancelled, **program["endpoint"],
            **{**program["http"], "wall_timeout": min(program["http"]["wall_timeout"], remaining())})
        result.update(server_identity=dict(program["server_identity"]), method=method, path=path)
        return result

    def probe():
        nonlocal probe_count
        probe_count += 1
        value = http("cancel-probe-" + str(probe_count), "/health", b"", "GET")
        health.append(value)
        with store_lock:
            state["probes"].append(store.observation(value))
        save()

    def invoke(request, group_cancel):
        if request["role"] != "target":
            return http(request["id"], request["path"], encoded(request["payload"]), "POST", group_cancel)
        cancelled, done = threading.Event(), threading.Event()
        box, problems = [], []
        invoke_start = time.monotonic_ns()
        trigger_deadline = min(deadline, time.monotonic() + program["timing"]["trigger_timeout_s"])

        def target_http():
            try:
                box.append(http(request["id"], request["path"], encoded(request["payload"]), "POST", cancelled, True))
            except BaseException as error:
                problems.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=target_http, name="phase-http-" + request["id"])
        unstarted = False
        register_worker(thread, done, cancelled)
        try:
            try:
                thread.start()
            except RuntimeError:
                unstarted = thread.ident is None
                if unstarted:
                    done.set()
                raise
            while True:
                require(not group_cancel.is_set(), "phase capture interrupted")
                now = time.monotonic()
                limit = trigger_deadline if not triggers else min(deadline,
                    triggers[0]["cancellation"]["set_after_ns"] / 1e9 + program["timing"]["retirement_timeout_s"])
                require(now < limit, "target phase trigger deadline exceeded" if not triggers else "target retirement observation deadline exceeded")
                metadata, decoded = read_log()
                selected = _target_prefix(decoded, required, program)
                require(time.monotonic() < limit, "phase/retirement read exceeded its frozen deadline")
                if not triggers:
                    require(not done.is_set(), "target HTTP finished before phase trigger")
                    candidate = _prospective_phase(selected, required["scenario"])
                    if candidate is not None:
                        before = time.monotonic_ns()
                        cancelled.set()
                        after = time.monotonic_ns()
                        cancellation = dict(id=request["id"], server_identity=dict(program["server_identity"]),
                            invoke_started_ns=invoke_start, set_before_ns=before, set_after_ns=after)
                        prefix = reader.raw[:metadata["complete_bytes"]]
                        trigger = {"clock": "monotonic_ns", "log_read": metadata,
                                   "complete_byte_watermark": len(prefix), "observed_record": candidate,
                                   "cancellation": cancellation}
                        triggers.append(trigger)
                        with store_lock:
                            trigger["log_prefix"] = store.put(prefix)
                            state["trigger"] = store.obj(trigger)
                        save()
                if triggers and selected and selected[-1]["record"]["event"] == "trace_end":
                    prefix = reader.raw[:metadata["complete_bytes"]]
                    facts = validate_cancel_trace(prefix, **trace_args())
                    require(time.monotonic() < limit, "target retirement validation exceeded its frozen deadline")
                    trigger_row = triggers[0]["observed_record"]["record"]
                    require(facts["target"]["trace_id"] == trigger_row["trace_id"], "trigger/final trace identity differs")
                    if required["scenario"] == "cancel_prime":
                        require(facts["drop_spanning_quantum"]["id"] == trigger_row["quantum"]["id"],
                                "trigger quantum differs from actual drop-spanning quantum")
                    target_facts.append(facts)
                    with store_lock:
                        state["target_end_observation"] = store.obj({"log_read": metadata,
                            "complete_byte_watermark": len(prefix), "log_prefix": store.put(prefix), "facts": facts})
                    break
                wait_poll(group_cancel, limit, done)
        except BaseException as error:
            cancelled.set()
            group_cancel.set()
            with store_lock:
                state["errors"].append(f"phase watcher: {type(error).__name__}: {error}")
                state["trigger_failures"].append(store.obj({"clock": "monotonic_ns",
                    "observed_ns": time.monotonic_ns(), "error": f"{type(error).__name__}: {error}",
                    "triggered": bool(triggers), "complete_byte_watermark": reader.raw.rfind(b"\n") + 1,
                    "last_log_observation": state["log_observations"][-1] if state["log_observations"] else None}))
            save()
        finally:
            if not target_facts:
                cancelled.set()
            if not unstarted:
                # capture_request has finite connect/read/wall deadlines and honours cancellation.
                done.wait(program["http"]["wall_timeout"] + 2)
                if thread.ident is not None:
                    thread.join(timeout=program["http"]["wall_timeout"] + 2)
                require(done.is_set() and not thread.is_alive(), "target HTTP failed bounded join")
        if problems:
            raise problems[0]
        require(box, "target HTTP has no captured observation")
        return box[0]

    def persist(row):
        with store_lock:
            if any(old["id"] == row["id"] for old in attempts):
                return  # The shared group also flushes each returned observation.
            attempts.append(row)
            state["observations"].append(store.observation(row))
            save()

    def tracked_invoke(request, group_cancel):
        done = threading.Event()
        register_worker(threading.current_thread(), done, group_cancel)
        try:
            require(not stopping.is_set() and not group_cancel.is_set(), "phase capture interrupted before invocation")
            row = invoke(request, group_cancel)
            # Persist before completion, including if the shared group's bounded
            # cancellation join expired. Never lose that late raw observation.
            persist(row)
            return row
        except BaseException as error:
            with store_lock:
                state["client_errors"][request["id"]] = f"{type(error).__name__}: {error}"
            raise
        finally:
            done.set()

    def clients(requests, mode):
        budget = program["http"]["wall_timeout"]
        if any(request["role"] == "target" for request in requests):
            budget = max(budget, program["timing"]["trigger_timeout_s"] + program["timing"]["retirement_timeout_s"])
        try:
            collect_clients(requests, mode, tracked_invoke, persist, min(remaining(), budget), state["client_errors"])
        except BaseException:
            stopping.set()
            raise
        finally:
            # Normally collect_clients already joined these threads. Its failure
            # path is bounded, so retain ownership through any remaining unwind.
            settle_workers()
        require(not state["errors"], "owned client settlement or phase watcher failed")

    save()
    try:
        process_snapshot("before")
        ownership("before")
        reader = _PinnedLog(server.output_path)
        state["log_descriptor"] = store.obj(reader.descriptor)
        baseline_deadline = min(deadline, time.monotonic() + program["timing"]["trigger_timeout_s"])
        while True:
            metadata, decoded = read_log()
            require(time.monotonic() < baseline_deadline, "prelaunch log prefix deadline exceeded")
            if metadata["complete_bytes"] == len(reader.raw) == metadata["size_after"]:
                break
            # An unfinished pre-existing row could conceal a reused target key.
            # Complete the bounded baseline before sending any request or probe.
            wait_poll(stopping, baseline_deadline)
        require(not _target_prefix(decoded, required, program), "target key already exists before launch")
        probe()
        clients([r for r in program["requests"] if r["role"] != "recovery"], "concurrent")
        require(not state["errors"] and len(triggers) == len(target_facts) == 1, "phase trigger/final trace failed")
        probe()
        clients(roles["recovery"], "serial")
        probe()
        ownership("after")
        while True:
            metadata, _ = read_log()
            if metadata["complete_bytes"] == len(reader.raw) == metadata["size_after"]:
                break
            remaining()
            wait_poll(stopping, deadline)
        facts = validate_cancel_trace(reader.raw, **trace_args())
        wire_facts = evaluate_cancel_wire(required, program, attempts=attempts, health_samples=health,
                                         cancellation=triggers[0]["cancellation"])
        minted = wire_facts["server_request_ids"][target["id"]]
        require(minted is None or minted == facts["target"]["request_id"], "wire minted ID differs from lifecycle")
        remaining()
        with store_lock:
            state["log_final"] = store.put(reader.raw)
            state["trace_facts"] = store.obj(facts)
            state["wire_facts"] = store.obj(wire_facts)
            state["state"] = "captured"
    except BaseException as error:
        state["state"] = "failed"
        state["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        stopping.set()
        settle_workers()
        if reader is not None:
            state["log_observed"] = store.put(reader.raw)
            reader.close()
        try:
            process_snapshot("after")
            live()
            if not after_listener_done:
                ownership("after")
        except Exception as error:
            state["state"] = "failed"
            state["errors"].append(f"after process check: {error}")
            state["listeners"].append(store.obj({"label": "after", "error": str(error),
                                                  "observed_ns": time.monotonic_ns()}))
        seen = {r["id"] for r in attempts}
        if seen:
            state["wire_accounting"] = store.obj(account_attempts([r for r in program["requests"] if r["id"] in seen], attempts))
        state["planned"] = len(program["requests"])
        state["captured_attempts"] = len(attempts)
        state["unobserved_invoked_ids"] = sorted(invoked_ids - seen)
        state["request_denominator"] = [{"id": r["id"], "role": r["role"],
            "status": "captured" if r["id"] in seen else
                      "invoked_without_observation" if r["id"] in invoked_ids else "not_invoked"}
            for r in program["requests"]]
        state["finished_ns"] = time.monotonic_ns()
        if time.monotonic() > deadline:
            state["state"] = "failed"
            state["errors"].append("overall phase capture deadline exceeded")
        save()
    return state
