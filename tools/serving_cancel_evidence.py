"""Replay a complete phase-cancellation capture against external expectations.

The caller supplies the immutable index digest, required cell, request program,
server launch and log identity. The capture cannot select those expectations.
Every declared blob is checked, all append bytes are reconstructed, and target,
wire, peer/recovery and timing predicates are rerun from raw observations.

This binds a single captured cell to its sampled process/listener/file evidence.
It does not authenticate the caller's expectations, prove unsampled filesystem
history, actual CUDA completion, or source/build/model/physical-GPU lease scope.
It returns qualification=False. Failed captures refuse; their original bytes and
the complete campaign attempt denominator must remain in the outer run record.
"""

import hashlib
import json
from pathlib import PurePosixPath
import re

from serving_cancel import evaluate_cancel_wire
from serving_cancel_phase import (_decode_prefix, _prospective_phase, _target_prefix,
                                  validate_phase_program)
from serving_completion import _keys, _same
from serving_evidence import _listener
from serving_policy import _seconds
from serving_release import account_attempts, json_object, require
from serving_trace import validate_cancel_trace


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_LOG = 64 * 1024 * 1024
_ROOT = frozenset("schema state qualification clock started_ns finished_ns program required observations probes "
    "listeners process_observations request_invocations log_chunks log_observations errors client_errors "
    "trigger_failures unattempted_ids unobserved_invoked_ids request_denominator planned captured_attempts "
    "payloads log_descriptor log_observed wire_accounting trigger target_end_observation log_final "
    "trace_facts wire_facts".split())
_READ = frozenset("started_ns finished_ns device inode size_before named_device named_inode size_after "
    "named_after_device named_after_inode bytes complete_bytes sha256".split())


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _uint(value):
    return type(value) is int and 0 <= value < 2**64


def _span(value):
    a, b = value.get("started_ns"), value.get("finished_ns")
    require(_uint(a) and _uint(b) and a <= b, "invalid phase evidence interval")
    return a, b


class _Bundle:
    def __init__(self, payloads, reader):
        require(type(payloads) is dict and 0 < len(payloads) <= 32768, "invalid phase payload manifest")
        self.blobs, self.payloads, self.decoded = {}, payloads, {}
        total = 0
        for name, digest in payloads.items():
            require(type(digest) is str and _SHA.fullmatch(digest) and name == "blobs/" + digest,
                    "phase blob path is not canonical")
            data = reader(name)
            require(type(data) is bytes and _hash(data) == digest, "phase payload hash differs")
            total += len(data)
            require(total <= 512 * 1024 * 1024, "phase decoded payload bound exceeded")
            self.blobs[name] = data

    def raw(self, ref):
        _keys(ref, {"path", "sha256"}, "phase blob reference")
        require(type(ref["path"]) is str and ref["path"] in self.blobs
                and ref["sha256"] == self.payloads[ref["path"]], "phase reference missing or substituted")
        return self.blobs[ref["path"]]

    def obj(self, ref):
        raw = self.raw(ref)
        if ref["path"] not in self.decoded:
            self.decoded[ref["path"]] = json_object(raw)
        return self.decoded[ref["path"]]

    def objects(self, refs, limit):
        require(type(refs) is list and len(refs) <= limit, "phase record list exceeds bound")
        return [self.obj(ref) for ref in refs]

    def observation(self, ref, maximum_bytes):
        value = dict(self.obj(ref))
        value["body"] = self.raw(value.get("body"))
        require(len(value["body"]) <= maximum_bytes, "response body exceeds trusted byte cap")
        return value


def _process(records, identity, launch, descriptor, bounds):
    _keys(launch, {"argv", "cwd", "env", "timeouts", "output_path"}, "expected server launch")
    require(type(launch["argv"]) is list and launch["argv"]
            and all(type(v) is str and "\0" not in v for v in launch["argv"])
            and PurePosixPath(launch["argv"][0]).is_absolute(), "invalid expected server argv")
    require(type(launch["cwd"]) is str and PurePosixPath(launch["cwd"]).is_absolute()
            and type(launch["output_path"]) is str and PurePosixPath(launch["output_path"]).is_absolute(),
            "invalid expected server paths")
    require(type(launch["env"]) is dict and all(type(k) is str and type(v) is str
            for k, v in launch["env"].items()), "invalid expected environment")
    _keys(launch["timeouts"], {"startup", "overall", "drain", "kill"}, "expected server timeouts")
    for name, value in launch["timeouts"].items():
        lo, _ = _seconds(value, "server " + name + " timeout")
        require(lo >= 0 and (name == "drain" or lo > 0), "invalid expected server timeout")
    env_hash = _hash(json.dumps(launch["env"], sort_keys=True, separators=(",", ":")).encode())
    require(len(records) == 2 and [r.get("label") for r in records] == ["before", "after"],
            "process snapshot denominator differs")
    previous_end, previous_owner = bounds[0], None
    for record in records:
        _keys(record, {"label", "started_ns", "finished_ns", "receipt", "observed_identity"}, "process snapshot")
        start, end = _span(record)
        require(previous_end <= start <= end <= bounds[1], "process snapshots outside capture or reordered")
        previous_end = end
        receipt = record["receipt"]
        require(type(receipt) is dict and receipt.get("schema") == "memra-owned-server-v1"
                and receipt.get("state") == "ready" and receipt.get("stop") is None
                and receipt.get("errors") == [] and receipt.get("server_exit") is None,
                "borrowed server was not live and ready")
        require(_same(receipt.get("argv"), launch["argv"]) and receipt.get("cwd") == launch["cwd"]
                and receipt.get("output_path") == launch["output_path"] == descriptor["path"]
                and receipt.get("env_keys") == sorted(launch["env"])
                and receipt.get("env_sha256") == env_hash
                and _same(receipt.get("timeouts"), launch["timeouts"]), "server launch differs from trusted scope")
        ready = receipt.get("ready")
        require(type(ready) is dict and _seconds(receipt.get("started_monotonic"), "server start")[1]
                <= _seconds(ready.get("monotonic"), "server ready")[0]
                <= bounds[0], "server readiness does not precede capture")
        supervisor = receipt.get("supervisor")
        require(type(supervisor) is dict and _uint(supervisor.get("pid")) and _uint(supervisor.get("ppid"))
                and supervisor["pid"] > 1 and supervisor.get("ppid") == descriptor["controller_pid"],
                "server supervisor is not owned by the expected controller")
        owner_keys = ("pid", "ppid", "pgid", "start_time", "identity_source")
        owners = [receipt.get("server"), record["observed_identity"], ready.get("owner")]
        for owner in owners:
            require(type(owner) is dict and all(_uint(owner.get(k)) for k in ("pid", "ppid", "pgid"))
                    and owner["pid"] == identity["pid"] and owner.get("pgid") == identity["pid"]
                    and owner.get("ppid") == supervisor["pid"]
                    and owner.get("identity_source") == "linux_proc_start_ticks"
                    and type(owner.get("start_time")) is str and owner["start_time"].isdigit()
                    and type(receipt.get("boot_id")) is str
                    and receipt["boot_id"] + ":" + owner["start_time"] == identity["start_identity"]
                    and owner.get("state") in ("R", "S", "D", "T", "t", "I", "W", "P"),
                    "foreign, dead or non-Linux process evidence")
        own = {k: owners[0][k] for k in owner_keys}
        require(all(_same(own, {k: o[k] for k in owner_keys}) for o in owners)
                and (previous_owner is None or _same(own, previous_owner)), "process ownership drift")
        previous_owner = own
    return records


def _log_capture(capture, bundle, expected, bounds):
    _keys(expected, {"path", "device", "inode", "controller_pid"}, "expected log identity")
    require(type(expected["path"]) is str and PurePosixPath(expected["path"]).is_absolute()
            and all(_uint(expected[k]) for k in ("device", "inode", "controller_pid"))
            and expected["inode"] > 0 and expected["controller_pid"] > 1, "invalid expected log identity")
    descriptor = bundle.obj(capture["log_descriptor"])
    _keys(descriptor, set(expected) | {"fd", "regular_file", "opened_before_ns", "opened_after_ns"}, "log descriptor")
    require(_same({k: descriptor[k] for k in expected}, expected) and descriptor["regular_file"] is True
            and _uint(descriptor["fd"]), "log descriptor differs from expected ownership")
    opened = _span({"started_ns": descriptor["opened_before_ns"], "finished_ns": descriptor["opened_after_ns"]})
    require(bounds[0] <= opened[0] <= opened[1] <= bounds[1], "log opened outside capture")
    require(type(capture["log_chunks"]) is list and len(capture["log_chunks"]) <= 10000,
            "invalid log chunk census")
    chunks, size = [], 0
    for item in capture["log_chunks"]:
        _keys(item, {"offset", "bytes", "raw"}, "log append chunk")
        data = bundle.raw(item["raw"])
        require(_uint(item["offset"]) and item["offset"] == size and _uint(item["bytes"])
                and item["bytes"] == len(data) and data, "log append missing, reordered or empty")
        size += len(data)
        require(size <= _MAX_LOG, "log byte bound exceeded")
        chunks.append(data)
    raw = b"".join(chunks)
    require(raw and raw.endswith(b"\n") and raw == bundle.raw(capture["log_final"])
            == bundle.raw(capture["log_observed"]), "final log is incomplete or differs from append bytes")
    reads = bundle.objects(capture["log_observations"], 10000)
    require(reads, "missing log read history")
    previous_end, previous_size, previous_stat_size, chunk_index = opened[1], 0, 0, 0
    for read in reads:
        _keys(read, _READ, "log read")
        start, end = _span(read)
        count, complete = read["bytes"], read["complete_bytes"]
        require(previous_end <= start <= end <= bounds[1] and _uint(count) and _uint(complete)
                and previous_size <= count <= len(raw) and complete == raw[:count].rfind(b"\n") + 1,
                "log read chronology or watermark differs")
        require(all(_uint(read[k]) for k in ("size_before", "size_after"))
                and previous_stat_size <= read["size_before"] == count <= read["size_after"] <= _MAX_LOG
                and read["sha256"] == _hash(raw[:count]), "log read byte/size/hash differs")
        for a, b in (("device", "inode"), ("named_device", "named_inode"),
                     ("named_after_device", "named_after_inode")):
            require(_same([read[a], read[b]], [expected["device"], expected["inode"]]), "log inode changed")
        if count > previous_size:
            require(chunk_index < len(capture["log_chunks"]), "log growth lacks its append chunk")
            chunk = capture["log_chunks"][chunk_index]
            require(chunk["offset"] == previous_size and chunk["bytes"] == count - previous_size,
                    "append chunk does not match its read watermark")
            chunk_index += 1
        previous_end, previous_size, previous_stat_size = end, count, read["size_after"]
    require(chunk_index == len(capture["log_chunks"])
            and reads[-1]["bytes"] == reads[-1]["complete_bytes"] == reads[-1]["size_after"] == len(raw),
            "final read is not the complete stable log")
    return descriptor, raw, reads


def read_phase_capture(capture_bytes, evidence_reader, *, expected_capture_sha256,
                       expected_required, expected_program, expected_server, expected_log_identity):
    """Return rederived capture facts; source/binary/lease qualification is external."""
    require(type(capture_bytes) is bytes and 0 < len(capture_bytes) <= 16 * 1024 * 1024
            and type(expected_capture_sha256) is str and _SHA.fullmatch(expected_capture_sha256)
            and _hash(capture_bytes) == expected_capture_sha256, "phase index differs from external digest")
    roles = validate_phase_program(expected_required, expected_program)
    capture = json_object(capture_bytes)
    _keys(capture, _ROOT, "phase capture")
    require(capture["schema"] == "memra-phase-cancel-capture-v1" and capture["state"] == "captured"
            and capture["qualification"] is False and capture["clock"] == "monotonic_ns"
            and capture["errors"] == capture["trigger_failures"] == [] and capture["client_errors"] == {},
            "phase capture failed or contains errors")
    bounds = _span(capture)
    require(bounds[1] - bounds[0] <= int(expected_program["timing"]["overall_s"] * 1e9), "capture exceeded deadline")
    bundle = _Bundle(capture["payloads"], evidence_reader)
    require(_same(bundle.obj(capture["program"]), expected_program)
            and _same(bundle.obj(capture["required"]), expected_required), "capture chose a different program or cell")
    descriptor, raw, reads = _log_capture(capture, bundle, expected_log_identity, bounds)
    identity = expected_program["server_identity"]
    processes = _process(bundle.objects(capture["process_observations"], 2), identity, expected_server, descriptor, bounds)
    require(_span(processes[0])[1] <= descriptor["opened_before_ns"]
            and reads[-1]["finished_ns"] <= _span(processes[1])[0], "process snapshots do not enclose log capture")
    requests = expected_program["requests"]
    count = len(requests)
    require(type(capture["planned"]) is int and type(capture["captured_attempts"]) is int
            and capture["planned"] == capture["captured_attempts"] == count
            and capture["unattempted_ids"] == capture["unobserved_invoked_ids"] == []
            and _same(capture["request_denominator"], [{"id": r["id"], "role": r["role"], "status": "captured"}
                                                      for r in requests]), "incomplete request denominator")
    require(type(capture["observations"]) is list and len(capture["observations"]) == count,
            "request observation denominator differs")
    body_limit = expected_program["http"]["max_body_bytes"]
    attempts = [bundle.observation(ref, body_limit) for ref in capture["observations"]]
    accounting = account_attempts(requests, attempts)
    require(_same(accounting, bundle.obj(capture["wire_accounting"])), "recorded accounting differs from raw attempts")
    indexed = {row["id"]: row for row in attempts}
    for request in requests:
        row = indexed[request["id"]]
        require(_same(row.get("server_identity"), identity) and row.get("method") == "POST"
                and row.get("path") == request["path"], "request owner or method/path differs")
    invocations = bundle.objects(capture["request_invocations"], count)
    require(len(invocations) == count and all(type(r.get("id")) is str for r in invocations)
            and len({r.get("id") for r in invocations}) == count
            and {r.get("id") for r in invocations} == set(indexed), "missing/duplicate/foreign invocation")
    for invocation in invocations:
        _keys(invocation, {"id", "started_ns", "stage", "path", "server_identity"}, "request invocation")
        row = indexed[invocation["id"]]
        a, b = _span(row)
        require(_uint(invocation["started_ns"]) and bounds[0] <= invocation["started_ns"] <= a <= b <= bounds[1]
                and invocation["stage"] == "before_capture_request" and invocation["path"] == row["path"]
                and _same(invocation["server_identity"], identity), "invocation does not bind the observed request")
    require(type(capture["probes"]) is list and len(capture["probes"]) == 3, "health probe denominator differs")
    health = [bundle.observation(ref, body_limit) for ref in capture["probes"]]
    require([h.get("id") for h in health] == ["cancel-probe-1", "cancel-probe-2", "cancel-probe-3"],
            "health probe denominator differs")
    for h in health:
        a, b = _span(h)
        require(bounds[0] <= a <= b <= bounds[1] and h.get("path") == "/health"
                and h.get("method") == "GET" and h.get("transport_error") is None
                and _same(h.get("server_identity"), identity), "foreign or incomplete health probe")
    pressure = [indexed[r["id"]] for r in requests if r["role"] != "recovery"]
    recovery = [indexed[r["id"]] for r in roles["recovery"]]
    require(health[0]["finished_ns"] <= min(r["started_ns"] for r in pressure)
            and max(r["finished_ns"] for r in pressure) <= health[1]["started_ns"]
            and health[1]["finished_ns"] <= recovery[0]["started_ns"]
            and all(a["finished_ns"] <= b["started_ns"] for a, b in zip(recovery, recovery[1:]))
            and recovery[-1]["finished_ns"] <= health[2]["started_ns"], "health/recovery scheduling differs")
    listeners = bundle.objects(capture["listeners"], 10000)
    proofs = {}
    for item in listeners:
        label = item.get("label")
        require(label in ("before", "after"), "unknown listener stage")
        if "proof" in item:
            _keys(item, {"label", "proof"}, "listener observation")
            require(label not in proofs, "duplicate successful listener proof")
            proofs[label] = _listener(item["proof"], identity, expected_program["endpoint"])
        else:
            _keys(item, {"label", "error", "observed_ns"}, "listener retry")
            require(type(item["error"]) is str and item["error"] and _uint(item["observed_ns"])
                    and bounds[0] <= item["observed_ns"] <= bounds[1] and label not in proofs,
                    "invalid or late listener retry")
    require(set(proofs) == {"before", "after"}
            and _span(processes[0])[1] <= proofs["before"][0] <= proofs["before"][1] <= descriptor["opened_before_ns"]
            and health[2]["finished_ns"] <= proofs["after"][0] <= proofs["after"][1] <= reads[-1]["started_ns"],
            "listener evidence does not bracket the capture")

    target = roles["target"][0]
    args = dict(scenario=expected_required["scenario"], client_trace_key=expected_program["client_trace_key"],
        expected_server_identity=identity, expected_model=expected_required["scope"]["model"],
        expected_http_route=target["path"], expected_worker_route=expected_program["trace"]["worker_route"],
        expected_worker_generation=expected_program["trace"]["worker_generation"],
        expected_quantum_routes=expected_program["trace"]["quantum_routes"])
    trigger, end = bundle.obj(capture["trigger"]), bundle.obj(capture["target_end_observation"])
    _keys(trigger, {"clock", "log_read", "complete_byte_watermark", "observed_record", "cancellation", "log_prefix"}, "phase trigger")
    _keys(end, {"log_read", "complete_byte_watermark", "log_prefix", "facts"}, "target end observation")
    positions = []
    for point in (trigger, end):
        require(type(point["log_read"]) is dict, "missing point read")
        matches = [i for i, read in enumerate(reads) if _same(point["log_read"], read)]
        require(len(matches) == 1, "trigger/end read is absent or ambiguous")
        positions.append(matches[0])
        n = point["complete_byte_watermark"]
        require(_uint(n) and n == point["log_read"]["complete_bytes"]
                and bundle.raw(point["log_prefix"]) == raw[:n], "trigger/end prefix differs from complete log")
    require(positions[0] < positions[1], "target end observation does not follow trigger")
    baselines = [r for r in reads[:positions[0]] if r["bytes"] == r["complete_bytes"] == r["size_after"]
                 and r["finished_ns"] <= health[0]["started_ns"]]
    require(baselines and not _target_prefix(_decode_prefix(raw[:baselines[-1]["bytes"]]), expected_required, expected_program),
            "missing stable fresh-key baseline before HTTP")
    prospective = _prospective_phase(_target_prefix(_decode_prefix(bundle.raw(trigger["log_prefix"])),
                                    expected_required, expected_program), expected_required["scenario"])
    require(prospective is not None and _same(prospective, trigger["observed_record"]), "trigger did not follow the selected phase")
    cancellation = trigger["cancellation"]
    _keys(cancellation, {"id", "server_identity", "invoke_started_ns", "set_before_ns", "set_after_ns"}, "cancellation observation")
    require(trigger["clock"] == "monotonic_ns" and all(_uint(cancellation[k]) for k in
            ("invoke_started_ns", "set_before_ns", "set_after_ns")), "invalid cancellation clock")
    target_attempt = indexed[target["id"]]
    target_invocation = next(row for row in invocations if row["id"] == target["id"])
    require(cancellation["invoke_started_ns"] <= target_invocation["started_ns"]
            and target_attempt["started_ns"] <= trigger["log_read"]["finished_ns"],
            "phase observation predates target HTTP capture or its invocation")
    # A read can straddle HTTP start, so its start is not a valid refusal bound.
    # But an earlier completed read exposing this same trace is a contradiction,
    # even if the selected trigger later rereads the unchanged phase.
    earlier = [read for read in reads if read["finished_ns"] < target_attempt["started_ns"]]
    if earlier:
        decoded = _decode_prefix(raw[:earlier[-1]["complete_bytes"]])
        selected = prospective["record"]
        require(not any(row["record"]["pid"] == selected["pid"]
                        and row["record"]["trace_id"] == selected["trace_id"]
                        for row in (decoded or {}).get("records", [])),
                "target trace was observed before HTTP capture")
    require(trigger["log_read"]["finished_ns"] <= cancellation["set_before_ns"] <= cancellation["set_after_ns"]
            <= end["log_read"]["started_ns"] and bounds[0] <= cancellation["invoke_started_ns"]
            and cancellation["set_after_ns"] - cancellation["invoke_started_ns"]
                < int(expected_program["timing"]["trigger_timeout_s"] * 1e9)
            and end["log_read"]["finished_ns"] - cancellation["set_after_ns"]
                < int(expected_program["timing"]["retirement_timeout_s"] * 1e9), "phase trigger/end exceeded timing scope")
    end_facts = validate_cancel_trace(bundle.raw(end["log_prefix"]), **args)
    facts = validate_cancel_trace(raw, **args)
    require(_same(end_facts, end["facts"]) and _same(facts, bundle.obj(capture["trace_facts"])), "stored trace facts differ from raw log")
    require(facts["target"]["trace_id"] == prospective["record"]["trace_id"], "trigger/final target differs")
    if expected_required["scenario"] == "cancel_prime":
        require(facts["drop_spanning_quantum"]["id"] == prospective["record"]["quantum"]["id"], "stale prime trigger")
    require(end["log_read"]["finished_ns"] <= health[1]["started_ns"], "recovery bracket precedes target retirement observation")
    wire = evaluate_cancel_wire(expected_required, expected_program, attempts=attempts,
                                health_samples=health, cancellation=cancellation)
    require(_same(wire, bundle.obj(capture["wire_facts"])), "stored wire facts differ from raw responses")
    minted = wire["server_request_ids"][target["id"]]
    require(minted is None or minted == facts["target"]["request_id"], "wire and trace minted identities differ")
    return {"scope": "phase_capture_consistency_only", "qualification": False,
            "capture_sha256": expected_capture_sha256, "program": expected_program,
            "required": expected_required, "trace_facts": facts, "wire_facts": wire,
            "attempts": attempts, "health_samples": health, "log": raw,
            "process_observations": processes, "listener_observations": listeners,
            "verified_payloads": len(bundle.blobs)}
