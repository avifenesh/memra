"""Owned-server drain capture. No source/build/lease or native qualification.

collect_drain_cell(required, program, *, server: OwnedServer, output) borrows an
already ready server and OWNS its requested stop and client cleanup. The caller
must authorize that stop. Invalid input/foreign ownership refuses before signals.

The immutable program has schema memra-drain-program-v1, mode drain, cell_id,
scope, server_identity, endpoint={host,port}, identities={server_binary:path,...},
http={connect_timeout,read_timeout,wall_timeout,max_body_bytes},
timing={trigger_timeout_s,overall_s,drain_timeout_ns,retry_after_s}, and requests.
Every request has id/role/model/wire/path/payload; inflight requests additionally
have inclusive prompt_tokens/completion_tokens ranges. Inflight is SSE; new
admission may request any supported wire. Both roles are nonempty. Source/model/
controller artifact paths may be added to identities by the trusted outer plan.

The replay caller supplies that exact program and required cell, capture digest,
expected artifact hashes and actual launch. No receipt chooses its own oracle.
The server log's source-defined SIGTERM line confirms the in-flight HTTP count;
client first-body observations alone are not server-active-work proof. Request
captures, including buffered reads after exit, retain their own wire deadlines.
No post-exit health or listener proof is manufactured or required.
"""

import ipaddress
import math
from pathlib import Path

from serving_capture import encoded
from serving_completion import _keys, _same
from serving_policy import _identity, _range
from serving_release import ServingGateError, require


_REQUIREMENTS = {"health_draining": 200, "ready_unavailable": 503,
                 "new_request_refused": 503, "retry_headers": True,
                 "inflight_completion": True, "owned_exit_code": 0, "no_escalation": True}
_MAX_BODY = 16 * 1024 * 1024
_MAX_LOG = 64 * 1024 * 1024
_MAX_PROBES = 128
_PROBE_PREFIX = "drain-probe-"


def validate_drain_program(required, program):
    """Validate frozen scope before launching requests or requesting a stop."""
    _keys(required, {"id", "scope", "scenario", "requirements"}, "required drain cell")
    require(required["scenario"] == "drain" and _same(required["requirements"], _REQUIREMENTS),
            "unknown drain requirements")
    scope = required["scope"]
    _keys(scope, {"id", "model", "route", "profile"}, "drain scope")
    require(all(type(v) is str and v.strip() for v in scope.values())
            and required["id"] == scope["id"] + "/drain", "invalid drain scope")
    _keys(program, {"schema", "mode", "cell_id", "scope", "server_identity", "endpoint",
                    "identities", "http", "timing", "requests"}, "drain program")
    require(program["schema"] == "memra-drain-program-v1" and program["mode"] == "drain"
            and program["cell_id"] == required["id"] and _same(program["scope"], scope),
            "drain program differs from required cell")
    _identity(program["server_identity"])
    _keys(program["endpoint"], {"host", "port"}, "drain endpoint")
    host, port = program["endpoint"]["host"], program["endpoint"]["port"]
    require(type(host) is str, "drain endpoint must be literal loopback")
    try:
        require(ipaddress.ip_address(host).is_loopback, "drain endpoint must be literal loopback")
    except ValueError as error:
        raise ServingGateError("drain endpoint must be literal loopback") from error
    require(type(port) is int and 1 <= port <= 65535, "invalid drain port")
    identities = program["identities"]
    require(type(identities) is dict and "server_binary" in identities and 1 <= len(identities) <= 32
            and all(type(k) is str and k and type(v) is str and Path(v).is_absolute()
                    for k, v in identities.items()), "invalid drain artifact paths")
    _keys(program["http"], {"connect_timeout", "read_timeout", "wall_timeout", "max_body_bytes"}, "HTTP limits")
    for key in ("connect_timeout", "read_timeout", "wall_timeout"):
        value = program["http"][key]
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 3600,
                "invalid drain HTTP deadline")
    require(type(program["http"]["max_body_bytes"]) is int
            and 0 < program["http"]["max_body_bytes"] <= _MAX_BODY, "invalid drain HTTP body cap")
    timing = program["timing"]
    _keys(timing, {"trigger_timeout_s", "overall_s", "drain_timeout_ns", "retry_after_s"}, "drain timing")
    for key in ("trigger_timeout_s", "overall_s"):
        require(type(timing[key]) in (int, float) and math.isfinite(timing[key])
                and 0 < timing[key] <= 3600, "invalid drain collector deadline")
    require(type(timing["drain_timeout_ns"]) is int and 0 < timing["drain_timeout_ns"] <= 3600_000_000_000
            and timing["drain_timeout_ns"] % 1_000_000_000 == 0,
            "drain budget must match whole-second MEMRA_DRAIN_S")
    require(type(timing["retry_after_s"]) is int and 1 <= timing["retry_after_s"] <= 60,
            "invalid drain Retry-After")
    require(timing["trigger_timeout_s"] < timing["overall_s"], "trigger deadline exceeds overall budget")
    requests = program["requests"]
    require(type(requests) is list and 2 <= len(requests) <= 32, "drain needs 2..32 planned requests")
    roles, ids = {"inflight": [], "new_admission": []}, set()
    for request in requests:
        require(type(request) is dict and type(request.get("role")) is str and request["role"] in roles, "unknown drain role")
        role = request["role"]
        _keys(request, {"id", "role", "model", "wire", "path", "payload"}
              | ({"prompt_tokens", "completion_tokens"} if role == "inflight" else set()), "drain request")
        name = request["id"]
        require(type(name) is str and name.strip() and name not in ids and not name.startswith(_PROBE_PREFIX),
                "empty, duplicate or reserved drain request ID")
        ids.add(name)
        wire = request["wire"]
        require(type(wire) is str and wire in {"chat_json", "chat_sse", "native_json", "native_sse"}
                and (role != "inflight" or wire.endswith("_sse")), "inflight drain must use an actual stream")
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        payload = request["payload"]
        require(request["path"] == path and request["model"] == scope["model"] and type(payload) is dict
                and payload.get("model") == request["model"] and type(payload.get("stream")) is bool
                and payload["stream"] == wire.endswith("_sse"), "drain request route/model/stream differs")
        if role == "inflight":
            _range(request["prompt_tokens"], "prompt tokens")
            _range(request["completion_tokens"], "completion tokens")
        roles[role].append(request)
    require(all(roles.values()), "drain requires inflight and new-admission denominators")
    encoded(program)
    return roles


def policy_config(program):
    """Only the fields the existing pure drain predicate owns."""
    return {"scenario": "drain", "server_identity": program["server_identity"],
            "stop_reason": "drain_cell", "drain_timeout_ns": program["timing"]["drain_timeout_ns"],
            "retry_after_s": program["timing"]["retry_after_s"],
            "requests": [{k: v for k, v in request.items() if k not in {"path", "payload"}}
                         for request in program["requests"]]}


def require_inflight_prefix(request, prefix):
    """Apply the same incomplete, non-error prefix rule during capture and replay."""
    from serving_release import WIRE_VALIDATORS
    try:
        WIRE_VALIDATORS[request["wire"]](prefix, model=request["model"], require_output=False)
    except ServingGateError as error:
        require(error.code != "typed_error", "inflight prefix is a typed server error")
    else:
        raise ServingGateError("inflight prefix already contains a complete response")


def collect_drain_cell(required, program, *, server, output):
    """Capture and stop one authorized borrowed server; always qualification=False."""
    from dataclasses import asdict
    import queue
    import threading
    import time
    from serving_capture import EvidenceStore, file_identity
    from serving_cancel_phase import _PinnedLog, _LogFailure
    from serving_http import capture_request
    from serving_listener import prove_listener, ListenerOwnershipError
    from serving_process import OwnedServer, process_identity
    from serving_release import account_attempts, json_object
    from serving_policy import _seconds, _signal_interval, evaluate_scenario

    validate_drain_program(required, program)
    required, program = json_object(encoded(required)), json_object(encoded(program))
    roles = validate_drain_program(required, program)
    require(isinstance(server, OwnedServer), "drain needs an actual OwnedServer")
    owner, initial = server.identity, server.receipt()
    # Non-Linux POSIX collection can retain honest CPU evidence, but the unchanged
    # Linux lifecycle/listener predicate cannot qualify that receipt.
    birth = initial.get("boot_id", owner.identity_source) + ":" + owner.start_time
    identity = {"pid": owner.pid, "start_identity": birth}
    require(_same(identity, program["server_identity"]), "drain owner differs from frozen program")
    require(initial.get("ready") and initial.get("state") == "ready" and not initial.get("stop")
            and not initial.get("errors"), "drain server is not ready")
    require(program["identities"]["server_binary"] == initial["argv"][0], "server binary path differs")
    require(not any(k.startswith("MEMRA_API_KEY") for k in initial.get("env_keys", [])),
            "drain capture requires its isolated unauthenticated loopback server")
    require(_seconds(initial["timeouts"]["drain"], "owned drain limit")
            == (program["timing"]["drain_timeout_ns"],) * 2, "owned drain deadline differs")
    store = EvidenceStore(output)
    started = time.monotonic_ns()
    deadline = time.monotonic() + program["timing"]["overall_s"]
    state = {"schema":"memra-drain-capture-v1", "state":"collecting", "qualification":False,
             "clock":"monotonic_ns", "started_ns":started, "finished_ns":None,
             "required":store.obj(required), "program":store.obj(program),
             "identities_before":{}, "identities_after":{}, "process_before":None, "lifecycle":None,
             "listeners":[], "invocations":[], "observations":[], "http_events":[],
             "probes":[{"role":role,"observation":None} for role in ("before","draining","ready")],
             "trigger":None, "stop_call":None, "stop_observation":None, "log_before":None,
             "log":None, "log_identity":None, "log_reads":[], "errors":[], "client_errors":{},
             "request_denominator":[], "unattempted_ids":[], "unobserved_invoked_ids":[],
             "accounting":None, "facts":None}
    lock = threading.RLock()
    notifications = queue.Queue(maxsize=3 * (len(program["requests"])+3))
    cancelled = threading.Event()
    workers, rows, first_bodies, headers, invoked = {}, {}, {}, {}, set()
    stop_thread, stop_done, stop_box = None, threading.Event(), {}
    pinned = None

    def save():
        with lock: store.index(state)

    def remaining():
        left = deadline - time.monotonic()
        require(left > 0, "overall drain capture deadline exceeded")
        return left

    def live():
        begin = time.monotonic_ns()
        actual = process_identity(owner.pid)
        receipt = server.receipt()
        require(actual is not None and actual.key == owner.key and actual.ppid == owner.ppid
                and actual.pgid == owner.pgid and actual.state not in ("Z","X","x")
                and receipt.get("server_exit") is None and receipt.get("state") != "finished",
                "owned drain server exited or changed identity")
        return {"started_ns":begin,"finished_ns":time.monotonic_ns(),"identity":asdict(actual)}

    def ownership(stage, request_id):
        limit = time.monotonic() + min(3, remaining())
        while True:
            begin = time.monotonic_ns()
            live()
            try:
                proof = prove_listener(owner, **program["endpoint"], timeout=max(.001, limit-time.monotonic()))
                reference = store.obj(asdict(proof))
                state["listeners"].append(store.obj({"stage":stage,"id":request_id,"proof":reference}))
                return reference
            except ListenerOwnershipError as error:
                state["listeners"].append(store.obj({"stage":stage,"id":request_id,
                    "started_ns":begin,"finished_ns":time.monotonic_ns(),"error":str(error)}))
                if time.monotonic() >= limit: raise
                cancelled.wait(min(.02, max(0,limit-time.monotonic())))

    def observe(snapshot):
        # The HTTP callback never performs filesystem/procfs IO, waits for the
        # controller or writes capture.json. Payload values are immutable copies.
        notifications.put_nowait(dict(snapshot))

    def record_event(event, *, verify=True):
        event = dict(event)
        if event["event"] == "first_body":
            event["body"] = store.put(event.pop("data"))
        try:
            if verify and event["event"] == "connected":
                event["owner_check"] = store.obj(live())
                event["listener"] = ownership("connected", event["id"])
        except BaseException as error:
            event["ownership_error"] = f"{type(error).__name__}: {error}"
            state["http_events"].append(store.obj(event))
            raise
        reference = store.obj(event)
        state["http_events"].append(reference)
        if event["event"] == "headers": headers[event["id"]] = event
        if event["event"] == "first_body":
            require(event["id"] not in first_bodies, "duplicate first-body observation")
            first_bodies[event["id"]] = reference

    def pump(wait=.01, *, verify=True):
        try: event = notifications.get(timeout=wait)
        except queue.Empty: return
        with lock:
            record_event(event, verify=verify)
            while True:
                try: event = notifications.get_nowait()
                except queue.Empty: break
                record_event(event, verify=verify)
            save()

    def start_http(name, path, payload, method):
        require(name not in workers, "duplicate HTTP invocation")
        live()
        done = threading.Event()
        body = encoded(payload) if method == "POST" else b""
        wall = min(program["http"]["wall_timeout"], remaining())

        def invoke():
            try:
                with lock:
                    invoked.add(name)
                    state["invocations"].append(store.obj({"id":name,"method":method,"path":path,
                        "body":store.put(body),"started_ns":time.monotonic_ns()}))
                result = capture_request(request_id=name, method=method, path=path, body=body,
                    headers={"X-Request-Id":name,"Content-Type":"application/json"},
                    cancel_event=cancelled, observer=observe, **program["endpoint"],
                    **{**program["http"],"wall_timeout":wall})
                result.update(server_identity=dict(identity), method=method, path=path)
                with lock:
                    rows[name] = result
                    reference = store.observation(result)
                    if name.startswith(_PROBE_PREFIX):
                        next(p for p in state["probes"] if _PROBE_PREFIX+p["role"] == name)["observation"] = reference
                    else: state["observations"].append(reference)
            except BaseException as error:
                with lock: state["client_errors"][name] = f"{type(error).__name__}: {error}"
            finally: done.set()

        thread = threading.Thread(target=invoke, name="drain-http-"+name)
        workers[name] = (thread, done)
        try: thread.start()
        except RuntimeError:
            if thread.ident is None: done.set()
            raise
        return done

    def wait_http(name):
        done = workers[name][1]
        while not done.is_set():
            remaining(); pump()
        # Flush callbacks that were enqueued before the final HTTP observation.
        while not notifications.empty(): pump(0)
        require(name in rows and name not in state["client_errors"], "HTTP attempt lacks final observation: "+name)
        return rows[name]

    def start_stop(reason):
        nonlocal stop_thread
        require(stop_thread is None, "owned stop already requested")
        def close_owner():
            stop_box["call"] = {"started_ns":time.monotonic_ns(),"finished_ns":None,"reason":reason,"error":None}
            try: stop_box["receipt"] = server.close(reason=reason)
            except BaseException as error:
                stop_box["call"]["error"] = f"{type(error).__name__}: {error}"
                stop_box["receipt"] = server.receipt()
            finally:
                stop_box["call"]["finished_ns"] = time.monotonic_ns()
                stop_done.set()
        stop_thread = threading.Thread(target=close_owner, name="drain-owned-close")
        try: stop_thread.start()
        except RuntimeError:
            if stop_thread.ident is None:
                stop_thread = None
            raise

    save()
    try:
        before = live()
        state["process_before"] = store.obj({"started_ns":before["started_ns"],
            "finished_ns":time.monotonic_ns(), "receipt":server.receipt(),"observed_identity":before["identity"]})
        state["identities_before"] = {k:file_identity(path) for k,path in program["identities"].items()}
        ownership("initial", None)
        pinned = _PinnedLog(server.output_path)
        state["log_identity"] = store.obj(pinned.descriptor)
        metadata, _, _ = pinned.read()
        state["log_reads"].append(store.obj(metadata)); state["log_before"] = store.put(pinned.raw)
        require(not pinned.raw or pinned.raw.endswith(b"\n"), "owned log baseline has an unfinished line")
        start_http(_PROBE_PREFIX+"before", "/health", None, "GET")
        before_health = wait_http(_PROBE_PREFIX+"before")
        require(before_health["transport_error"] is None and before_health["status"] == 200
                and json_object(before_health["body"]).get("status") == "ok", "initial health was not complete and healthy")
        trigger_started = time.monotonic_ns()
        for request in roles["inflight"]:
            start_http(request["id"], request["path"], request["payload"], "POST")
        trigger_limit = min(deadline, trigger_started/1e9+program["timing"]["trigger_timeout_s"])
        while not all(request["id"] in first_bodies for request in roles["inflight"]):
            require(time.monotonic() < trigger_limit, "inflight first-body trigger deadline exceeded")
            require(not any(done.is_set() for _,done in workers.values() if
                            done is not workers[_PROBE_PREFIX+"before"][1]), "an inflight capture ended before drain trigger")
            pump()
        for request in roles["inflight"]:
            name = request["id"]
            require(not workers[name][1].is_set() and headers[name]["status"] == 200,
                    "inflight request completed/refused before trigger")
            first = json_object((store.root/first_bodies[name]["path"]).read_bytes())
            prefix = (store.root/first["body"]["path"]).read_bytes()
            require_inflight_prefix(request, prefix)
            # The server SIGTERM gauge also checks that the body slot remained
            # in flight; client-buffer lifetime alone cannot supply that proof.
        state["trigger"] = store.obj({"started_ns":trigger_started,"observed_ns":time.monotonic_ns(),
            "inflight_ids":[r["id"] for r in roles["inflight"]],
            "first_body_events":{r["id"]:first_bodies[r["id"]] for r in roles["inflight"]}})
        start_stop("drain_cell")
        while True:
            remaining(); pump()
            began = time.monotonic_ns(); receipt = server.receipt(); ended = time.monotonic_ns()
            signals = [s for s in receipt.get("signals", []) if s.get("pid") == owner.pid and s.get("signal") == 15]
            require(not any(s.get("result") not in ("sent","already_exited") for s in signals), "owned TERM failed")
            sent = [s for s in signals if s.get("result") == "sent"]
            if sent:
                require(len(sent) == 1, "multiple primary TERM observations")
                _signal_interval(sent[0]); live()
                state["stop_observation"] = store.obj({"started_ns":began,"finished_ns":ended,"signal":sent[0]})
                break
            require(not stop_done.is_set(), "no successful TERM record observed before owned exit")
        for role,path in (("draining","/health"),("ready","/readyz")):
            name = _PROBE_PREFIX+role
            start_http(name,path,None,"GET"); wait_http(name)
        for request in roles["new_admission"]:
            start_http(request["id"],request["path"],request["payload"],"POST"); wait_http(request["id"])
        for request in roles["inflight"]: wait_http(request["id"])
        while not stop_done.is_set(): remaining(); pump()
        require(stop_box["call"]["error"] is None, "owned stop failed")
        state["state"] = "captured"
    except BaseException as error:
        state["state"] = "failed"
        state["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if state["state"] != "captured": cancelled.set()
        if stop_thread is None:
            try: start_stop("drain_capture_error")
            except BaseException as error:
                state["errors"].append(f"stop launch: {error}")
                # A true thread allocation failure still closes only this verified
                # owned server synchronously, retaining the failure and receipt.
                begin = time.monotonic_ns()
                try: stop_box["receipt"] = server.close(reason="drain_capture_error")
                except BaseException as cleanup_error:
                    state["errors"].append(f"owned fallback close: {cleanup_error}")
                    stop_box["receipt"] = server.receipt()
                stop_box["call"] = {"started_ns":begin,"finished_ns":time.monotonic_ns(),
                    "reason":"drain_capture_error","error":"stop thread could not start"}
                stop_done.set()
        join_limit = time.monotonic()+max(program["http"]["connect_timeout"],3)+initial["timeouts"]["drain"]+initial["timeouts"]["kill"]+4
        warned = False
        while not stop_done.is_set() or any(not done.is_set() for _,done in workers.values()):
            try: pump(.02, verify=False)
            except BaseException as error: state["errors"].append(f"final HTTP event: {error}")
            if time.monotonic() > join_limit and not warned:
                cancelled.set(); warned = True; state["errors"].append("owned drain workers exceeded settlement deadline")
            # Never seal/return an index while an owned worker can still write it.
        for thread,done in workers.values():
            if thread.ident is not None: thread.join()
        if stop_thread is not None and stop_thread.ident is not None: stop_thread.join()
        while not notifications.empty():
            try: pump(0,verify=False)
            except BaseException as error: state["errors"].append(f"final HTTP event: {error}")
        state["stop_call"] = store.obj(stop_box["call"])
        state["lifecycle"] = store.obj(stop_box.get("receipt",server.receipt()))
        if pinned is not None:
            try:
                metadata,_,_ = pinned.read(); state["log_reads"].append(store.obj(metadata))
                state["log"] = store.put(pinned.raw)
            except _LogFailure as error:
                state["errors"].append(str(error))
                if error.observed is not None: state["log"] = store.put(error.observed)
                state["log_reads"].append(store.obj({**error.metadata,"error":str(error)}))
            finally: pinned.close()
        try: state["identities_after"] = {k:file_identity(path) for k,path in program["identities"].items()}
        except BaseException as error: state["errors"].append(f"artifact postcheck: {error}")
        require_names = {r["id"] for r in program["requests"]}
        attempted = {name for name in rows if name in require_names}
        state["request_denominator"] = [{"id":r["id"],"role":r["role"],"status":
            "captured" if r["id"] in attempted else "invoked_without_observation" if r["id"] in invoked else "not_invoked"}
            for r in program["requests"]]
        state["unattempted_ids"] = [r["id"] for r in program["requests"] if r["id"] not in invoked]
        state["unobserved_invoked_ids"] = sorted((invoked & require_names)-attempted)
        try:
            attempts = [rows[r["id"]] for r in program["requests"] if r["id"] in attempted]
            accounting = account_attempts([r for r in program["requests"] if r["id"] in attempted], attempts)
            state["accounting"] = store.obj(accounting)
            require(attempted == require_names and not state["client_errors"], "incomplete drain request observations")
            state["facts"] = store.obj(evaluate_scenario(policy_config(program), accounting=accounting,
                attempts=attempts,health_samples=[rows[_PROBE_PREFIX+role] for role in ("before","draining")],
                status_samples=[rows[_PROBE_PREFIX+"ready"]],lifecycle=stop_box["receipt"]))
            require(state["identities_before"] == state["identities_after"], "artifact identity changed during drain")
            from serving_drain_evidence import drain_log
            drain_log((store.root/state["log_before"]["path"]).read_bytes(),
                (store.root/state["log"]["path"]).read_bytes(),len(roles["inflight"]),program["timing"]["drain_timeout_ns"])
        except BaseException as error: state["errors"].append(f"drain validation: {type(error).__name__}: {error}")
        state["finished_ns"] = time.monotonic_ns()
        if time.monotonic() > deadline: state["errors"].append("overall drain capture deadline exceeded")
        if state["errors"] or state["client_errors"]: state["state"] = "failed"
        save()
        if state["state"] == "captured":
            try:
                import hashlib
                from serving_drain_evidence import read_drain_capture
                index = (store.root/"capture.json").read_bytes()
                launch = {"argv":initial["argv"], "cwd":initial["cwd"], "env":dict(server._config["env"]),
                          "timeouts":initial["timeouts"], "output_path":str(server.output_path)}
                read_drain_capture(index, lambda name:(store.root/name).read_bytes(),
                    expected_capture_sha256=hashlib.sha256(index).hexdigest(), expected_required=required,
                    expected_program=program, expected_server=launch, expected_identities=state["identities_before"])
            except BaseException as error:
                state["errors"].append(f"immutable drain replay: {type(error).__name__}: {error}")
                state["state"] = "failed"
            # Verification is part of this bounded collector call too. The
            # returned index must not retain a pre-verification finish timestamp.
            state["finished_ns"] = time.monotonic_ns()
            if time.monotonic() > deadline:
                state["errors"].append("overall drain capture deadline exceeded during immutable replay")
                state["state"] = "failed"
            save()
    return state
