"""Bounded cancellation diagnostics; current Memra cannot qualify phase cancellation.

``collect_cancel_cell(required, program, *, server, host, port, http, output)``
borrows a live, ready OwnedServer. It never starts/stops the server. The caller
owns the outer lifecycle, source/binary/lease bindings, and independently trusted
manifest/program. Listener proof is required before HTTP and after collection.
All HTTP uses serving_http, including a TARGET-ONLY cancellation Event. Requests
are never retried. A watchdog on each HTTP capture bounds reads; collect_clients
joins/cancels its own client threads on interruption. No global process signals.

Program example (all limits frozen before capture)::

    {"cell_id": "tiny/cancel_decode", "scope": <required scope>,
     "server_identity": {"pid": 123, "start_identity": "<boot-id>:<start-ticks>"},
     "mode": "timed_disconnect_diagnostic",
     "timing": {"cancel_after_s": 0.2, "observe_after_s": 0.2,
                "poll_s": 0.05, "overall_s": 10},
     "requests": [
       {"id": "peer", "role": "peer", "model": "gate", "wire": "chat_json",
        "path": "/v1/chat/completions", "payload": <frozen request JSON>,
        "prompt_tokens": {"min": 1, "max": 32},
        "completion_tokens": {"min": 1, "max": 16}},
       {"id": "target", "role": "target", <same wire/path/payload fields>},
       {"id": "recovery", "role": "recovery", <same fields as peer>}]}

Exactly one target, at least one recovery, and peers for queued/decode are required.
Peers and target run concurrently; recovery runs serially after those clients and
a bounded diagnostic observation window. ``cancel_after_s`` is measured from the
target client invocation, NOT a server phase or token count. Health/metrics polls
are global diagnostics. The observation window is NOT a server cleanup deadline.
http has connect_timeout/read_timeout/wall_timeout/max_body_bytes (as in capture).
Generated probe IDs use the reserved ``cancel-probe-`` namespace.

The pure ``evaluate_cancel_cell`` consumes raw attempts, health_samples and the
client cancellation interval. It rederives the full wire denominator and worker
generation brackets and validates clean peer/recovery output and token usage.
It rejects a target that finishes before cancellation, a complete wire terminal
or positively decoded server error hidden by a client cancellation flag, transport
failure, or an absent peer. Raw client-cancelled accounting remains unchanged when
a server error fails the diagnostic; the two observations are not interchangeable.
Passing these *diagnostic* controls returns state=unqualified, qualification=False:
no currently supported input can make this module return a phase/cleanup pass.
Malformed/failed controls raise ServingGateError; the collector persists failure.
Client/server IDs are distinct: completion handlers mint x-request-id; a sent
X-Request-Id is not assumed to be echoed. Missing response headers remain unknown.

Missing native seam at 90d696461ba8d7b6ba62ece57a5ca1b438ef2205:
/health worker.phase and prime_progress are global. TTFT logs are emitted only at
first SSE output or trace drop, not live phase/retirement notifications. Queued
and active [abort] lines omit request identity and precede resource retirement.
A reviewed server-owned diagnostic must link the client correlation to the minted
request ID, stamp worker generation and ordered phase transitions, observe the
actual receiver cancellation and confirm completed request retirement. Prime also
needs request-specific chunk progress/bound evidence. A client disconnect, idle
health, recovery output, ledger settlement or process exit cannot substitute.
This module deliberately has no adapter for invented phase/cleanup markers.
"""

from dataclasses import asdict
import math
import threading
import time

from serving_capture import EvidenceStore, collect_clients, encoded
from serving_completion import _keys, _same
from serving_http import capture_request
from serving_lifecycle import account_generations
from serving_listener import ListenerOwnershipError, prove_listener
from serving_policy import _identity, _range, evaluate_scenario
from serving_process import OwnedServer, process_identity
from serving_release import ServingGateError, WIRE_VALIDATORS, account_attempts, json_object, require


_REQUIREMENTS = {
    "cancel_queued": {"observed_phase": "queued", "server_cleanup": True,
                      "peer_completion": True, "recovery_completion": True},
    "cancel_prime": {"observed_phase": "prime", "server_cleanup": True,
                     "bounded_by_prime_chunk": True, "recovery_completion": True},
    "cancel_decode": {"observed_phase": "decode", "server_cleanup": True,
                      "peer_completion": True, "recovery_completion": True},
}
_PROBE_PREFIX = "cancel-probe-"


def _positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def validate_cancel_program(required, program):
    """Validate trusted configuration before any IO; return per-role request lists."""
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required cancel cell")
    scenario = required["scenario"]
    require(isinstance(scenario, str) and scenario in _REQUIREMENTS
            and _same(required["requirements"], _REQUIREMENTS[scenario]), "unknown cancellation requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "cancel scope")
    require(all(isinstance(v, str) and v.strip() for v in scope.values())
            and required["id"] == scope["id"] + "/" + scenario, "invalid cancel scope")
    _keys(program, {"cell_id", "scope", "server_identity", "mode", "timing", "requests"}, "cancel program")
    require(program["cell_id"] == required["id"] and _same(program["scope"], scope)
            and program["mode"] == "timed_disconnect_diagnostic", "cancel program scope/mode differs")
    _identity(program["server_identity"])
    timing = program["timing"]
    _keys(timing, {"cancel_after_s", "observe_after_s", "poll_s", "overall_s"}, "cancel timing")
    require(all(_positive(v) for v in timing.values()) and timing["overall_s"] <= 3600
            and timing["poll_s"] >= .01 and timing["poll_s"] <= timing["observe_after_s"]
            and timing["cancel_after_s"] + timing["observe_after_s"] < timing["overall_s"],
            "invalid bounded cancellation timing")
    return validate_cancel_requests(program["requests"], scope=scope, scenario=scenario)


def validate_cancel_requests(requests, *, scope, scenario):
    """Shared request denominator/route/usage rules; caller validates the trigger program."""
    require(isinstance(requests, list) and 2 <= len(requests) <= 256, "cancel program needs 2..256 requests")
    roles, seen = {"target": [], "peer": [], "recovery": []}, set()
    for request in requests:
        require(isinstance(request, dict) and isinstance(request.get("role"), str)
                and request["role"] in roles, "unknown cancel role")
        role = request["role"]
        names = {"id", "role", "wire", "model", "path", "payload"}
        if role != "target":
            names |= {"prompt_tokens", "completion_tokens"}
        _keys(request, names, "cancel request")
        key, wire, payload = request["id"], request["wire"], request["payload"]
        require(isinstance(key, str) and key.strip() and key not in seen
                and not key.startswith(_PROBE_PREFIX), "empty/duplicate/reserved request ID")
        require(isinstance(wire, str) and wire in WIRE_VALIDATORS, "unknown response wire")
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        require(request["model"] == scope["model"] and request["path"] == path
                and isinstance(payload, dict) and payload.get("model") == scope["model"]
                and type(payload.get("stream")) is bool
                and payload["stream"] == wire.endswith("_sse"), "request model/route/stream differs")
        encoded(payload)
        if role != "target":
            _range(request["prompt_tokens"], "prompt tokens")
            _range(request["completion_tokens"], "completion tokens")
        seen.add(key)
        roles[role].append(request)
    require(len(roles["target"]) == 1 and roles["recovery"], "need one target and nonempty recovery")
    require(scenario == "cancel_prime" or roles["peer"], "this cancellation scenario requires a peer")
    return roles


def _server_request_id(attempt):
    headers = attempt.get("headers")
    require(isinstance(headers, list), "missing raw headers")
    values = []
    for pair in headers:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(v, str) for v in pair), "invalid raw headers")
        if pair[0].lower() == "x-request-id":
            values.append(pair[1])
    require(len(values) <= 1 and all(v.strip() for v in values), "ambiguous server request ID")
    return values[0] if values else None


def evaluate_cancel_cell(required, program, *, attempts, health_samples, cancellation):
    """Return unqualified diagnostic evidence, or raise for invalid/failed controls."""
    roles = validate_cancel_program(required, program)
    facts = _cancel_wire_facts(required, program, roles, attempts, health_samples, cancellation,
                               int(program["timing"]["cancel_after_s"] * 1e9))
    target = roles["target"][0]
    gaps = ["request_bound_phase_at_cancellation_unavailable", "request_bound_server_cleanup_unavailable"]
    if facts["server_request_ids"][target["id"]] is None:
        gaps.append("target_server_request_id_not_observed")
    if required["scenario"] == "cancel_prime":
        gaps.append("request_bound_prime_chunk_cancellation_bound_unavailable")
    return {"id": required["id"], "scenario": required["scenario"], "scope": dict(required["scope"]),
            "state": "unqualified", "qualification": False, "missing_evidence": gaps,
            **facts, "server_cleanup": None, "observed_target_phase": None}


def evaluate_cancel_wire(required, program, *, attempts, health_samples, cancellation):
    """Wire/peer/recovery facts only; not a timer, phase trigger or qualification.

    The phase collector validates its frozen program and actual log trigger separately.
    No timing-policy waiver is accepted from the program. The legacy public evaluator
    still enforces its original minimum delay at the same observation check below.
    """
    roles = validate_cancel_requests(program["requests"], scope=required["scope"], scenario=required["scenario"])
    return _cancel_wire_facts(required, program, roles, attempts, health_samples, cancellation, 0)


def _cancel_wire_facts(required, program, roles, attempts, health_samples, cancellation, minimum_delay_ns):
    accounting = account_attempts(program["requests"], attempts)
    observed = {r["id"]: r for r in attempts}
    identity = program["server_identity"]
    server_ids = {}
    for request in program["requests"]:
        row = observed[request["id"]]
        require(_same(row.get("server_identity"), identity) and row.get("method") == "POST"
                and row.get("path") == request["path"] and isinstance(row.get("body"), bytes),
                "foreign/incomplete request observation")
        minted = _server_request_id(row)
        require(minted is None or minted not in server_ids.values(), "server request ID reused across attempts")
        server_ids[row["id"]] = minted
    # completed_group checks full health records, unique IDs and ranges, while the
    # full census here includes the cancelled target and any failure in the denominator.
    for row in health_samples:
        require(isinstance(row.get("id"), str) and row["id"] not in observed
                and row.get("method") == "GET" and type(row.get("status")) is int
                and row["status"] == 200 and json_object(row["body"]).get("status") == "ok",
                "invalid healthy cancellation bracket")
    generations = account_generations(accounting, attempts, health_samples)
    require(_same(generations["server_identity"], identity)
            and generations["respawns_observed"] == 0
            and generations["generation_affected_requests"] == 0, "owner/worker generation changed")
    clean_plan = [{k: r[k] for k in ("id", "wire", "model", "prompt_tokens", "completion_tokens")}
                  for r in program["requests"] if r["role"] != "target"]
    clean_attempts = [observed[r["id"]] for r in clean_plan]
    clean_wire = account_attempts(clean_plan, clean_attempts)
    clean_generations = account_generations(clean_wire, clean_attempts, health_samples)
    completed = evaluate_scenario({"scenario": "completed_group", "server_identity": identity,
                                  "requests": clean_plan}, accounting=clean_wire,
                                 generations=clean_generations, attempts=clean_attempts,
                                 health_samples=health_samples)["completed"]
    target = roles["target"][0]
    row = observed[target["id"]]
    _keys(cancellation, {"id", "server_identity", "invoke_started_ns", "set_before_ns", "set_after_ns"},
          "client cancellation interval")
    require(cancellation["id"] == target["id"] and _same(cancellation["server_identity"], identity),
            "cancellation belongs to another attempt/owner")
    invoke, before, after = [cancellation[k] for k in ("invoke_started_ns", "set_before_ns", "set_after_ns")]
    require(all(type(v) is int and v >= 0 for v in (invoke, before, after))
            and invoke <= row["started_ns"] < before <= after
            and before <= row["finished_ns"]
            and before - invoke >= minimum_delay_ns,
            "target did not span the actual client cancellation interval")
    target_result = next(r for r in accounting["requests"] if r["id"] == target["id"])
    require(target_result["outcome"] == "client_cancelled"
            and row.get("status") in (None, 200), "target was not a client-cancelled inference attempt")
    # capture_request checks this actual Event under its final capture lock before
    # sampling finished_ns. The HTTP thread may finish before the cancelling thread
    # resumes to sample set_after_ns. Both are upper bounds on the real set, so use
    # their minimum; never require the target to outlive the caller's post-set sample.
    # The lower bound and actual cancelled result remain mandatory, not a tolerance.
    observed_set_upper = min(after, row["finished_ns"])
    # HTTP capture can classify a complete buffered response as cancelled if the
    # event races its final sample. The retained terminal must independently veto.
    if row.get("status") == 200:
        options = {"model": target["model"], "require_output": False}
        if target["wire"] == "chat_sse":
            options["require_usage"] = True
        try:
            WIRE_VALIDATORS[target["wire"]](row["body"], **options)
        except ServingGateError as error:
            # An incomplete cancelled body is expected. A positively decoded
            # backend error is a failed control even if transport also cancelled.
            if error.code == "typed_error":
                raise
        else:
            raise ServingGateError("target contains a complete response; cancellation may be too late")
    peers = [r["id"] for r in roles["peer"] if observed[r["id"]]["started_ns"] < before
             and observed[r["id"]]["finished_ns"] > observed_set_upper]
    require(required["scenario"] == "cancel_prime" or peers, "no clean peer spans cancellation")
    pressure_end = max(observed[r["id"]]["finished_ns"] for r in roles["target"] + roles["peer"])
    require(all(observed[r["id"]]["started_ns"] > pressure_end for r in roles["recovery"]),
            "recovery did not follow every pressure attempt")
    return {"accounting": accounting, "generations": generations, "completed": completed,
            "client_cancellation": dict(cancellation), "server_request_ids": server_ids,
            "spanning_peer_ids": peers,
            "timing_scope": "client observations only; no server scheduling or performance claim"}


def collect_cancel_cell(required, program, *, server, host, port, http, output):
    """Borrow a ready OwnedServer and persist a diagnostic capture in a NEW directory.

    Capture failures return state=failed plus all available raw rows, unattempted
    IDs and per-client errors. Valid diagnostic controls return state=unqualified.
    Neither can satisfy a required native cancellation cell. Server shutdown and
    its final raw logs remain the caller's job; this function owns only its clients.
    """
    roles = validate_cancel_program(required, program)
    require(isinstance(server, OwnedServer), "collector requires an actual OwnedServer")
    _keys(http, {"connect_timeout", "read_timeout", "wall_timeout", "max_body_bytes"}, "HTTP limits")
    require(all(_positive(http[k]) for k in ("connect_timeout", "read_timeout", "wall_timeout"))
            and type(http["max_body_bytes"]) is int and 0 < http["max_body_bytes"] <= 64 * 1024 * 1024,
            "invalid HTTP limits")
    # Freeze our own JSON copy; caller mutation cannot change a running schedule.
    program, required, http = [json_object(encoded(v)) for v in (program, required, http)]
    roles = validate_cancel_program(required, program)
    store = EvidenceStore(output)
    started = time.monotonic_ns()
    deadline = time.monotonic() + program["timing"]["overall_s"]
    state = {"schema": "memra-cancel-diagnostic-v1", "state": "collecting", "qualification": False,
             "started_ns": started, "plan": store.obj(program), "required": store.obj(required),
             "http": store.obj(http), "endpoint": {"host": host, "port": port},
             "observations": [], "probes": [], "listeners": [], "errors": [], "client_errors": {},
             "unattempted_ids": [r["id"] for r in program["requests"]]}
    attempts, health, cancellations, probe_counter = [], [], [], 0

    def save():
        store.index(state)

    def remaining():
        value = deadline - time.monotonic()
        require(value > 0, "overall cancellation capture deadline exceeded")
        return value

    def live():
        receipt, owner = server.receipt(), server.identity
        actual = process_identity(owner.pid)
        require(actual is not None and actual.key == owner.key and actual.pgid == owner.pgid
                and actual.ppid == owner.ppid and actual.state not in {"Z", "X", "x"}
                and receipt.get("state") != "finished" and not receipt.get("stop")
                and receipt.get("ready") and not receipt.get("errors"), "owned server is not live and ready")
        require(isinstance(receipt.get("boot_id"), str) and receipt["boot_id"],
                "native cancellation capture requires the Linux boot identity")
        require(_same(program["server_identity"], {"pid": owner.pid,
                    "start_identity": receipt["boot_id"] + ":" + owner.start_time}), "owned server birth differs")
        return owner

    def ownership():
        end = time.monotonic() + min(3.0, remaining())
        while True:
            owner = live()
            try:
                proof = prove_listener(owner, host, port, timeout=max(.001, end - time.monotonic()))
                state["listeners"].append(store.obj(asdict(proof)))
                save()
                return
            except ListenerOwnershipError as error:
                state["listeners"].append(store.obj({"error": str(error), "observed_ns": time.monotonic_ns()}))
                save()
                if time.monotonic() >= end:
                    raise
                time.sleep(max(0, min(.02, end - time.monotonic())))

    def capture(label, path, body, method, cancelled=None):
        live()
        options = {**http, "wall_timeout": min(http["wall_timeout"], remaining())}
        value = capture_request(request_id=label, host=host, port=port, path=path, method=method,
                                body=body, headers={"X-Request-Id": label, "Content-Type": "application/json"},
                                cancel_event=cancelled, **options)
        value.update(server_identity=dict(program["server_identity"]), method=method, path=path)
        return value

    def probe(path):
        nonlocal probe_counter
        probe_counter += 1
        value = capture(_PROBE_PREFIX + str(probe_counter), path, b"", "GET")
        state["probes"].append(store.observation(value))
        if path == "/health":
            health.append(value)
        save()

    def invoke(request, group_cancel):
        if request["role"] != "target":
            return capture(request["id"], request["path"], encoded(request["payload"]), "POST", group_cancel)
        own_cancel, done = threading.Event(), threading.Event()
        lock = threading.Lock()
        invoke_start = time.monotonic_ns()
        at = time.monotonic() + program["timing"]["cancel_after_s"]

        def timer():
            while not done.wait(min(.01, max(0, at - time.monotonic()))):
                if group_cancel.is_set() or time.monotonic() >= at:
                    with lock:
                        if done.is_set():
                            return
                        before = time.monotonic_ns()
                        own_cancel.set()
                        after = time.monotonic_ns()
                        # Group cancellation is not the requested experiment.
                        if not group_cancel.is_set():
                            cancellations.append({"id": request["id"], "server_identity": dict(program["server_identity"]),
                                                  "invoke_started_ns": invoke_start,
                                                  "set_before_ns": before, "set_after_ns": after})
                    return

        watcher = threading.Thread(target=timer, name="serving-cancel-timer")
        unstarted = False
        try:
            try:
                watcher.start()
            except RuntimeError:
                unstarted = watcher.ident is None
                raise
            return capture(request["id"], request["path"], encoded(request["payload"]), "POST", own_cancel)
        finally:
            with lock:
                done.set()
            if not unstarted:
                watcher.join(timeout=1)
            require(not watcher.is_alive(), "cancellation timer did not stop")

    def persist(row):
        attempts.append(row)
        state["observations"].append(store.observation(row))
        state["unattempted_ids"].remove(row["id"])
        save()

    def clients(requests, mode):
        return collect_clients(requests, mode, invoke, persist,
                               min(http["wall_timeout"], remaining()), state["client_errors"])

    save()
    try:
        ownership()
        probe("/health")
        clients([r for r in program["requests"] if r["role"] != "recovery"], "concurrent")
        # Global observations are preserved, never treated as request retirement.
        observe_end = time.monotonic() + program["timing"]["observe_after_s"]
        while True:
            probe("/health")
            probe("/metrics")
            if time.monotonic() >= observe_end:
                break
            time.sleep(max(0, min(program["timing"]["poll_s"], remaining(), observe_end - time.monotonic())))
        clients(roles["recovery"], "serial")
        probe("/health")
        ownership()
        remaining()
        require(len(cancellations) == 1, "target finished before requested cancellation or capture was interrupted")
        evidence = evaluate_cancel_cell(required, program, attempts=attempts, health_samples=health,
                                        cancellation=cancellations[0])
        # Framing failures in diagnostics are retained but cannot yield a valid capture.
        for ref in state["probes"]:
            raw = json_object((store.root / ref["path"]).read_bytes())
            require(raw.get("transport_error") is None, "incomplete diagnostic probe")
        state["evaluation"] = store.obj(evidence)
        state["state"] = "unqualified"
    except BaseException as error:
        state["state"] = "failed"
        state["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        state["cancellation_events"] = [store.obj(row) for row in cancellations]
        # Partial census is explicitly partial, never fabricated attempt timestamps.
        attempted_ids = {r["id"] for r in attempts}
        if attempted_ids:
            schedule = [r for r in program["requests"] if r["id"] in attempted_ids]
            state["wire_accounting"] = store.obj(account_attempts(schedule, attempts))
        state["planned"] = len(program["requests"])
        state["captured_attempts"] = len(attempts)
        state["borrowed_server_receipt"] = store.obj(server.receipt())
        state["finished_ns"] = time.monotonic_ns()
        if time.monotonic() > deadline:
            state["state"] = "failed"
            state["errors"].append("overall cancellation capture deadline exceeded")
        save()
    return state
