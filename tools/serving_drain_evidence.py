"""Immutable replay of one complete drain cell; returns qualification=False.

External required/program/launch/artifact expectations and the index digest are
mandatory. They come from the outer source/build/controller/lease binding layer,
never from the captured run. All blobs, including failed diagnostics, are read.
The existing drain predicate owns lifecycle/deadline/terminal/generation semantics.
This module binds that predicate to the actual planned HTTP and process census.
"""

import hashlib
import ipaddress
import json
from pathlib import PurePosixPath
import re

from serving_cancel_evidence import _Bundle
from serving_completion import _keys, _same
from serving_drain_capture import (_MAX_LOG, _MAX_PROBES, _PROBE_PREFIX,
                                   policy_config, require_inflight_prefix,
                                   validate_drain_program)
from serving_evidence import _listener
from serving_policy import _seconds, _signal_interval, evaluate_scenario
from serving_release import ServingGateError, account_attempts, json_object, require


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOT = frozenset("schema state qualification clock started_ns finished_ns required program "
    "identities_before identities_after process_before lifecycle listeners invocations observations probes "
    "http_events trigger stop_call stop_observation log_before log log_identity log_reads errors client_errors "
    "request_denominator unattempted_ids unobserved_invoked_ids accounting facts payloads".split())
_DRAIN_LINE = re.compile(rb"\[server\] SIGTERM: draining \(([0-9]+) in flight, deadline ([0-9]+)s\)\Z")


def span(row):
    start, finish = row.get("started_ns"), row.get("finished_ns")
    require(type(start) is int and type(finish) is int and 0 <= start <= finish < 2**64,
            "invalid drain evidence interval")
    return start, finish


def validate_launch(receipt, expected):
    _keys(expected, {"argv", "cwd", "env", "timeouts", "output_path"}, "expected drain launch")
    require(type(expected["argv"]) is list and expected["argv"]
            and all(type(arg) is str and "\0" not in arg for arg in expected["argv"])
            and PurePosixPath(expected["argv"][0]).is_absolute(), "invalid expected drain argv")
    require(all(type(expected[k]) is str and PurePosixPath(expected[k]).is_absolute()
                for k in ("cwd", "output_path")), "invalid expected drain paths")
    require(type(expected["env"]) is dict and all(type(k) is str and type(v) is str
            for k, v in expected["env"].items()), "invalid expected drain environment")
    _keys(expected["timeouts"], {"startup", "overall", "drain", "kill"}, "expected server timeouts")
    for name, value in expected["timeouts"].items():
        require(_seconds(value, "expected " + name + " timeout")[0] > 0, "invalid expected lifecycle timeout")
    env_hash = hashlib.sha256(json.dumps(expected["env"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(_same(receipt.get("argv"), expected["argv"]) and receipt.get("cwd") == expected["cwd"]
            and receipt.get("output_path") == expected["output_path"]
            and receipt.get("env_keys") == sorted(expected["env"])
            and receipt.get("env_sha256") == env_hash
            and _same(receipt.get("timeouts"), expected["timeouts"]), "server launch differs from external expectation")


def drain_log(before, raw, count, timeout_ns):
    """Source-defined HTTP in-flight count at TERM; not a GPU-work observation."""
    require(type(before) is bytes and type(raw) is bytes and len(raw) <= _MAX_LOG
            and raw.startswith(before) and (not before or before.endswith(b"\n"))
            and raw.endswith(b"\n"), "drain log is truncated or baseline changed")
    suffix = raw[len(before):].split(b"\n")
    selected = [line for line in suffix if line.startswith(b"[server] SIGTERM:")]
    require(len(selected) == 1, "missing or duplicate owned drain start record")
    match = _DRAIN_LINE.fullmatch(selected[0])
    require(match is not None and int(match[1]) == count
            and int(match[2]) * 1_000_000_000 == timeout_ns, "server drain count/deadline differs from planned inflight")
    require(sum(line.startswith(b"[server] drain complete in ") for line in suffix) == 1
            and not any(line.startswith(b"[server] drain deadline") for line in suffix),
            "server drain did not complete without its deadline path")
    require(suffix.index(selected[0]) < next(i for i,line in enumerate(suffix)
            if line.startswith(b"[server] drain complete in ")), "drain completion precedes start")
    return {"inflight_at_signal": count, "server_deadline_ns": timeout_ns,
            "start_line": selected[0].decode(), "scope": "server HTTP gauge at SIGTERM; not GPU concurrency"}


def read_drain_capture(capture_bytes, evidence_reader, *, expected_capture_sha256,
                       expected_required, expected_program, expected_server, expected_identities):
    """Recompute one bound drain result; failed captures refuse without rewriting."""
    require(type(capture_bytes) is bytes and len(capture_bytes) <= 8 * 1024 * 1024
            and type(expected_capture_sha256) is str and _SHA.fullmatch(expected_capture_sha256)
            and hashlib.sha256(capture_bytes).hexdigest() == expected_capture_sha256,
            "drain index differs from external digest")
    roles = validate_drain_program(expected_required, expected_program)
    capture = json_object(capture_bytes)
    _keys(capture, _ROOT, "drain capture")
    require(capture["schema"] == "memra-drain-capture-v1" and capture["state"] == "captured"
            and capture["qualification"] is False and capture["clock"] == "monotonic_ns"
            and capture["errors"] == [] and capture["client_errors"] == {}, "drain capture failed or is unqualified")
    bounds = span(capture)
    require(bounds[1] - bounds[0] <= int(expected_program["timing"]["overall_s"] * 1e9),
            "capture exceeded frozen overall deadline")
    bundle = _Bundle(capture["payloads"], evidence_reader)
    require(_same(bundle.obj(capture["required"]), expected_required)
            and _same(bundle.obj(capture["program"]), expected_program), "captured drain program/required cell substituted")
    require(type(expected_identities) is dict and expected_identities.keys() == expected_program["identities"].keys(),
            "external artifact scope differs")
    for value in expected_identities.values():
        _keys(value, {"bytes", "sha256"}, "artifact identity")
        require(type(value["bytes"]) is int and value["bytes"] >= 0 and type(value["sha256"]) is str
                and _SHA.fullmatch(value["sha256"]), "invalid external artifact identity")
    require(_same(capture["identities_before"], expected_identities)
            and _same(capture["identities_after"], expected_identities), "drain artifact/source/binary identity differs")
    lifecycle = bundle.obj(capture["lifecycle"])
    validate_launch(lifecycle, expected_server)
    require(expected_program["identities"]["server_binary"] == expected_server["argv"][0],
            "launched executable differs from bound artifact")
    identity = expected_program["server_identity"]
    before = bundle.obj(capture["process_before"])
    _keys(before, {"started_ns", "finished_ns", "receipt", "observed_identity"}, "before process observation")
    before_span = span(before)
    require(bounds[0] <= before_span[0] <= before_span[1] <= bounds[1], "before process outside capture")
    require(type(before["receipt"]) is dict and type(before["observed_identity"]) is dict,
            "before process observation lacks receipt/identity")
    validate_launch(before["receipt"], expected_server)
    require(before["receipt"].get("state") == "ready" and before["receipt"].get("stop") is None
            and before["receipt"].get("server_exit") is None and before["receipt"].get("errors") == [],
            "drain did not borrow a ready live server")
    require(before["receipt"].get("ready") and lifecycle.get("ready")
            and _seconds(lifecycle.get("started_monotonic"), "server start")[1]
                <= _seconds(before["receipt"]["ready"].get("monotonic"), "server ready")[0]
            and _seconds(before["receipt"]["ready"]["monotonic"], "server ready")[1] <= bounds[0],
            "drain process readiness does not precede capture")
    owner = before["receipt"].get("server")
    require(type(owner) is dict and owner.get("pid") == identity["pid"]
            and before["receipt"].get("boot_id", "") + ":" + owner.get("start_time", "") == identity["start_identity"]
            and _same({k:v for k,v in before["observed_identity"].items() if k != "state"},
                      {k:v for k,v in owner.items() if k != "state"})
            and before["observed_identity"].get("state") not in ("Z", "X", "x"), "before observation belongs to another birth")

    require(_same({k:v for k,v in owner.items() if k != "state"},
                  {k:v for k,v in lifecycle["server"].items() if k != "state"}), "final server identity differs from borrowed owner")

    requests = expected_program["requests"]
    ids = [r["id"] for r in requests]
    require(capture["unattempted_ids"] == [] and capture["unobserved_invoked_ids"] == []
            and _same(capture["request_denominator"], [{"id":r["id"], "role":r["role"], "status":"captured"} for r in requests]),
            "drain request denominator is incomplete")
    attempts = [bundle.observation(ref, expected_program["http"]["max_body_bytes"]) for ref in capture["observations"]]
    require(len(attempts) == len(ids) and {r["id"] for r in attempts} == set(ids), "drain attempt census differs")
    # Storage follows actual completion order (a refusal normally precedes the
    # draining stream). Derived lists use the collector's frozen program order;
    # do not rewrite the raw capture or infer dispatch order from completion.
    indexed = {row["id"]: row for row in attempts}
    attempts = [indexed[name] for name in ids]
    probes = capture["probes"]
    require(type(probes) is list and len(probes) <= _MAX_PROBES, "invalid drain probe list")
    for probe in probes:
        _keys(probe, {"role", "observation"}, "drain probe entry")
    require([r.get("role") for r in probes] == ["before", "draining", "ready"], "drain probe census differs")
    probe_rows = [bundle.observation(p["observation"], expected_program["http"]["max_body_bytes"]) for p in probes]
    require([p["id"] for p in probe_rows] == [_PROBE_PREFIX+name for name in ("before", "draining", "ready")],
            "probe IDs differ from frozen schedule")
    observed = {row["id"]:row for row in attempts + probe_rows}
    require(len(observed) == len(attempts) + 3, "probe/request IDs collide")
    invocations = bundle.objects(capture["invocations"], 256)
    require(len(invocations) == len(observed) and {r.get("id") for r in invocations} == set(observed),
            "HTTP invocation denominator differs")
    schedule = {r["id"]:r for r in requests}
    for p in probe_rows:
        schedule[p["id"]] = {"id":p["id"], "path":"/readyz" if p["id"].endswith("ready") else "/health"}
    for invocation in invocations:
        row = observed[invocation["id"]]
        expected = schedule[row["id"]]
        method = "POST" if row["id"] in ids else "GET"
        require(row.get("server_identity") == identity and row.get("method") == method
                and row.get("path") == expected["path"] and invocation.get("method") == method
                and invocation.get("path") == expected["path"], "HTTP owner/route/method differs")
        payload = bundle.raw(invocation["body"])
        from serving_capture import encoded
        require(payload == (encoded(expected["payload"]) if method == "POST" else b""), "sent request body substituted")
        begin, end = span(row)
        require(bounds[0] <= invocation["started_ns"] <= begin <= end <= bounds[1], "HTTP outside invocation/capture interval")

    accounting = account_attempts(requests, attempts)
    require(_same(accounting, bundle.obj(capture["accounting"])), "wire accounting differs from raw attempts")
    facts = evaluate_scenario(policy_config(expected_program), accounting=accounting, attempts=attempts,
        health_samples=probe_rows[:2], status_samples=probe_rows[2:], lifecycle=lifecycle)
    require(_same(facts, bundle.obj(capture["facts"])), "drain facts differ from raw evidence")
    stop = bundle.obj(capture["stop_call"])
    _keys(stop, {"started_ns", "finished_ns", "reason", "error"}, "owned stop call")
    stop_span = span(stop)
    require(stop["reason"] == "drain_cell" and stop["error"] is None
            and bounds[0] <= stop_span[0] <= facts["stop_requested_ns"]
            and facts["cleanup_finished_ns"] <= stop_span[1] <= bounds[1], "owned stop call does not enclose lifecycle")
    signal = bundle.obj(capture["stop_observation"])
    _keys(signal, {"started_ns", "finished_ns", "signal"}, "post-send observation")
    signal_span = span(signal)
    primary = [r for r in lifecycle["signals"] if r["pid"] == identity["pid"] and r["result"] == "sent"]
    require(len(primary) == 1 and _same(signal["signal"], primary[0])
            and _signal_interval(primary[0])[1] <= signal_span[1]
            and stop_span[0] <= signal_span[0] <= signal_span[1] < facts["exit_before_ns"],
            "post-send observation is absent, foreign or after exit")
    require(all(observed[name]["started_ns"] > signal_span[1] for name in
                [r["id"] for r in roles["new_admission"]] + [p["id"] for p in probe_rows[1:]]),
            "post-drain HTTP began before observed TERM delivery")

    # Connection/first-body evidence is validated below by its fixed callback
    # contract. A pre-exit capture start alone never proves a pre-exit connection.
    _http_evidence(capture, bundle, observed, roles, identity, expected_program, facts)
    raw, baseline = bundle.raw(capture["log"]), bundle.raw(capture["log_before"])
    log_facts = drain_log(baseline, raw, len(roles["inflight"]), expected_program["timing"]["drain_timeout_ns"])
    descriptor = bundle.obj(capture["log_identity"])
    require(descriptor.get("path") == expected_server["output_path"], "owned log path differs")
    reads = bundle.objects(capture["log_reads"], 10000)
    require(len(reads) == 2, "drain log observation census differs")
    for read, data in zip(reads, (baseline, raw)):
        a,b = span(read)
        require(bounds[0] <= a <= b <= bounds[1] and read.get("bytes") == len(data)
                and read.get("sha256") == hashlib.sha256(data).hexdigest()
                and read.get("device") == read.get("named_device") == read.get("named_after_device") == descriptor.get("device")
                and read.get("inode") == read.get("named_inode") == read.get("named_after_inode") == descriptor.get("inode")
                and read.get("size_before") == len(data) <= read.get("size_after", -1), "owned log identity/read differs")
    require(reads[0]["size_after"] <= reads[1]["size_before"]
            and reads[0]["finished_ns"] < probe_rows[0]["started_ns"]
            and max(stop_span[1], max(row["finished_ns"] for row in attempts + probe_rows)) <= reads[1]["started_ns"],
            "final log snapshot precedes owned stop/client settlement")
    return {"scope":"drain_capture_consistency_only", "qualification":False,
            "capture_sha256":expected_capture_sha256, "required":expected_required,
            "program":expected_program, "attempts":attempts, "health_samples":probe_rows[:2],
            "status_samples":probe_rows[2:], "lifecycle":lifecycle, "facts":facts,
            "log_facts":log_facts, "verified_payloads":len(bundle.blobs)}


def _http_evidence(capture, bundle, observed, roles, identity, program, facts):
    """Bind actual connect/header/first-body callbacks to final captured responses."""
    events = bundle.objects(capture["http_events"], 1024)
    by_id = {name:[] for name in observed}
    for event in events:
        require(type(event.get("id")) is str and event["id"] in by_id, "HTTP event belongs to unknown request")
        by_id[event["id"]].append(event)
    proof_refs = []
    first_bodies = {}
    for name, rows in by_id.items():
        require([r.get("event") for r in rows] == ["connected", "headers", "first_body"],
                "HTTP connect/header/body observation denominator differs")
        connected, headers, first_body = rows
        _keys(connected, {"event", "id", "observed_ns", "local", "peer", "owner_check", "listener"}, "connection event")
        _keys(headers, {"event", "id", "observed_ns", "status", "headers"}, "headers event")
        _keys(first_body, {"event", "id", "observed_ns", "end_offset", "body"}, "first-body event")
        row = observed[name]
        start, end = span(row)
        times = [e["observed_ns"] for e in rows]
        require(all(type(t) is int and 0 <= t < 2**64 for t in times)
                and start <= times[0] <= times[1] <= times[2] <= end,
                "HTTP observation order differs from actual request interval")
        require(times[0] < facts["exit_before_ns"], "HTTP connected at/after owned process exit")
        for address in (connected["local"], connected["peer"]):
            require(type(address) is list and len(address) in (2,4) and type(address[0]) is str
                    and type(address[1]) is int and 1 <= address[1] <= 65535, "invalid connected socket address")
        try:
            require(ipaddress.ip_address(connected["local"][0]).is_loopback
                    and ipaddress.ip_address(connected["peer"][0]) == ipaddress.ip_address(program["endpoint"]["host"])
                    and connected["peer"][1] == program["endpoint"]["port"], "actual TCP peer differs from owned endpoint")
        except ValueError as error:
            raise ServingGateError("invalid connected loopback address") from error
        owner = bundle.obj(connected["owner_check"])
        _keys(owner, {"started_ns", "finished_ns", "identity"}, "connection owner check")
        owner_span = span(owner)
        actual = owner["identity"]
        require(type(actual) is dict and actual.get("pid") == identity["pid"]
                and actual.get("start_time") == identity["start_identity"].rsplit(":",1)[-1]
                and actual.get("identity_source") == "linux_proc_start_ticks"
                and actual.get("state") not in ("Z", "X", "x"), "TCP connection owner birth differs")
        proof = bundle.obj(connected["listener"])
        proof_span = _listener(proof, identity, program["endpoint"])
        require(times[0] <= owner_span[0] <= owner_span[1] <= proof_span[0]
                <= proof_span[1] < facts["exit_before_ns"],
                "connection ownership was not verified after connect and before exit")
        proof_refs.append(connected["listener"])
        require(_same(headers["headers"], row["headers"]) and headers["status"] == row["status"],
                "response headers differ from actual observation")
        body = row["body"]
        require(type(row["headers"]) is list and all(type(pair) is list and len(pair) == 2
                and all(type(value) is str for value in pair) for pair in row["headers"]), "invalid raw HTTP header pairs")
        lengths = [value for key,value in row["headers"] if key.lower() == "content-length"]
        transfer = [value for key,value in row["headers"] if key.lower() == "transfer-encoding"]
        require(len(lengths) <= 1 and len(transfer) <= 1 and not (lengths and transfer), "ambiguous captured HTTP framing headers")
        if lengths:
            require(lengths[0].isascii() and lengths[0].isdecimal() and int(lengths[0]) == len(body),
                    "captured body differs from Content-Length")
        if transfer:
            require(transfer[0].strip().lower() == "chunked", "unsupported captured transfer framing")
        require(row.get("transport_error") is None, "HTTP capture recorded a transport/observer failure")
        n = first_body["end_offset"]
        require(type(n) is int and 0 < n <= min(len(body),65536)
                and bundle.raw(first_body["body"]) == body[:n]
                and row.get("first_body_byte_ns") == times[2], "first-body capture prefix/timing differs")
        chunks = row.get("chunks")
        require(type(chunks) is list and chunks and chunks[0] == {"end_offset":n,"observed_ns":times[2]},
                "first body lacks its actual chunk record")
        offset, previous = 0, times[1]
        for chunk in chunks:
            _keys(chunk, {"end_offset", "observed_ns"}, "HTTP body chunk")
            require(type(chunk["end_offset"]) is int and offset < chunk["end_offset"] <= len(body)
                    and type(chunk["observed_ns"]) is int and previous <= chunk["observed_ns"] <= end,
                    "HTTP body chunks are missing/reordered")
            offset, previous = chunk["end_offset"], chunk["observed_ns"]
        require(offset == len(body), "body extends beyond final captured chunk")
        first_bodies[name] = first_body
    listeners = bundle.objects(capture["listeners"], 4096)
    initial = []
    connected_refs = []
    for listener in listeners:
        require(listener.get("stage") in ("initial", "connected"), "unknown listener observation stage")
        if "proof" in listener:
            _keys(listener, {"stage", "id", "proof"}, "successful listener observation")
            _listener(bundle.obj(listener["proof"]), identity, program["endpoint"])
            if listener["stage"] == "initial": initial.append(listener["proof"])
            else:
                require(listener["id"] in observed, "listener belongs to unknown request")
                connected_refs.append(listener["proof"])
        else:
            _keys(listener, {"stage", "id", "started_ns", "finished_ns", "error"}, "listener retry")
            begin, end = span(listener)
            require(type(listener["error"]) is str and listener["error"]
                    and capture["started_ns"] <= begin <= end <= capture["finished_ns"], "invalid listener failure record")
    require(len(initial) == 1 and len(connected_refs) == len(proof_refs)
            and sorted(r["sha256"] for r in connected_refs) == sorted(r["sha256"] for r in proof_refs),
            "listener proof census differs from connected sockets")
    require(_listener(bundle.obj(initial[0]), identity, program["endpoint"])[1]
            < observed[_PROBE_PREFIX+"before"]["started_ns"], "initial listener proof is late")
    trigger = bundle.obj(capture["trigger"])
    _keys(trigger, {"started_ns", "observed_ns", "inflight_ids", "first_body_events"}, "drain trigger")
    require(trigger["inflight_ids"] == [r["id"] for r in roles["inflight"]]
            and type(trigger["first_body_events"]) is dict
            and set(trigger["first_body_events"]) == set(trigger["inflight_ids"]), "trigger inflight denominator differs")
    require(type(trigger["observed_ns"]) is int and type(trigger["started_ns"]) is int
            and capture["started_ns"] <= trigger["started_ns"] <= trigger["observed_ns"] < facts["sigterm_before_ns"]
            and trigger["observed_ns"] - trigger["started_ns"] <= int(program["timing"]["trigger_timeout_s"]*1e9),
            "inflight trigger outside frozen wait interval or not before TERM")
    for request in roles["inflight"]:
        name = request["id"]
        require(trigger["started_ns"] <= observed[name]["started_ns"]
                and _same(bundle.obj(trigger["first_body_events"][name]), first_bodies[name])
                and first_bodies[name]["observed_ns"] <= trigger["observed_ns"]
                < observed[name]["finished_ns"], "trigger did not observe this still-open capture")
        require_inflight_prefix(request, bundle.raw(first_bodies[name]["body"]))
