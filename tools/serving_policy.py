"""Pure, narrow serving-scenario predicates; no capture, process or release I/O.

Example frozen configuration (inclusive token bounds, monotonic nanoseconds)::

    {"scenario": "completed_group",
     "server_identity": {"pid": 100, "start_identity": "<boot UUID>:1234"},
     "requests": [{"id": "r1", "model": "gate", "wire": "chat_sse",
                   "prompt_tokens": {"min": 1, "max": 32},
                   "completion_tokens": {"min": 1, "max": 64}}]}

For "drain", each request also has role "inflight" or "new_admission".
Both roles need nonempty planned IDs; only inflight needs the token bounds above.
Add stop_reason="drain_cell", drain_timeout_ns=30_000_000_000 and retry_after_s=30.
All planned requests, including new-admission refusals, stay in the account_attempts
denominator. completed_group also requires account_generations with full brackets.
For drain pass generations=None (the default): the listener may close immediately
after the last response, so no post-completion or post-exit health call is required.
Drain instead returns generation_observations limited to the captured health
interval, with uncovered request tails marked unknown, generation_affected=None
and clean_latency_ns=None. Lifecycle exit closes the drain, NOT the generation
interval. Observed counter changes still refuse drain. No shutdown worker counter
is inferred from exit 0. The wire/model fields form the schedule; the caller must
bind those IDs to its actual sent request configuration.

evaluate_scenario(config, accounting=..., generations=..., attempts=...,
health_samples=..., status_samples=..., lifecycle=...) returns derived evidence or
raises ServingGateError. No supplied "passed"/"skip" flag grants acceptance.

attempts are capture_request records (id, started_ns, finished_ns, status, body
bytes, optional transport_error, response headers as a list of [name,value]) with
server_identity added by the owner-verifying collector. health_samples have that
same identity/timing, a unique id, path="/health", status and raw body bytes.
status_samples is a separate list of captured /readyz probes for drain. These
must have unique IDs too. completed_group requires health before and after every
attempt. Drain requires a before sample and observed draining health while alive;
its last health call may finish before the last response. All attempts/probes must
START before the recorded process-exit observation. Their finished_ns timestamps
are client read completion, NOT the server's last write: buffered terminal bodies,
typed refusals and probe replies may finish reading after server exit/cleanup.
Every client finish must still meet the frozen stop+drain_timeout_ns allowance.
For example: before-health, request-start, SIGTERM, draining-health/readyz503 and
new-admission requests, server terminal writes, server exit, client finishes reads.
No new post-exit captures are admitted, and no last-write timestamp is invented.
The collector must retain its verified connection/owned-identity binding; a capture
start timestamp alone does not prove a connection preceded the actual process exit.

lifecycle is the original Linux OwnedServer receipt. Its server ProcessIdentity
plus boot_id must map exactly to config.server_identity using "boot_id:start_time".
Its floating monotonic-second timestamps are converted conservatively to ns:
before tests use floor, after/deadline tests use ceiling. drain_timeout_ns bounds
primary exit AND cleanup, starting at the recorded stop request (not wall clock).
The same frozen deadline bounds client reads independently: it is NOT extended by
their finish times. The primary SIGTERM send must precede all drain probes/new
admissions and every inflight client completion. Exactly one such send is required;
no inferred signal or claim that server work continued until the client's last read.
The process receipt must include post-syscall sent_monotonic and action-finish
timestamps; the old pre-send monotonic field alone cannot prove delivery. Every
known descendant needs either a supervisor wait record or an identity-retirement
observation (a primary-reaped child is not waitable by the supervisor). Retirement
proves the old birth disappeared, not its exit status or who reaped it. All reap,
retirement and signal actions must precede cleanup.finished_monotonic, which must
meet the unchanged frozen deadline. Older receipts missing these facts refuse.

This module rederives accounting with existing parsers to reject stale/tampered
derived rows and then checks their per-request observations. It makes no statement
about authenticity of the supplied bytes, hashes, leases, manifest completeness,
listener ownership or protocol routing beyond those observations. Those bindings
belong to the collector/release layer. No benchmark or other scenario is supported.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import math
import signal
import uuid

from serving_lifecycle import account_generations
from serving_release import (ServingGateError, WIRE_VALIDATORS, account_attempts,
                             json_object, require, usage_counts)


def _mapping(value, name):
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def _keys(value, required, name):
    _mapping(value, name)
    require(set(value) == set(required), f"unknown or missing {name} fields")


def _integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def _text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"invalid {name}")
    return value


def _identity(value):
    _keys(value, {"pid", "start_identity"}, "server identity")
    _integer(value["pid"], "server PID", 2)
    _text(value["start_identity"], "server birth identity")
    return value


def _by_id(rows, name):
    require(isinstance(rows, list) and rows, f"empty or missing {name}")
    result = {}
    for row in rows:
        _mapping(row, name)
        key = _text(row.get("id"), f"{name} ID")
        require(key not in result, f"duplicate {name} ID")
        result[key] = row
    return result


def _interval(row):
    start = _integer(row.get("started_ns"), "observation start")
    end = _integer(row.get("finished_ns"), "observation end")
    require(start <= end, "inverted observation interval")
    return start, end


def _seconds(value, name):
    # Bound conversion before decimal/string work; this exceeds centuries of uptime.
    require(type(value) in (int, float) and 0 <= value <= (2**63 - 1) / 1_000_000_000,
            f"invalid {name}")
    try:
        ns = Decimal(str(value)) * 1_000_000_000
        require(ns.is_finite() and ns >= 0, f"invalid {name}")
        return int(ns.to_integral_value(rounding=ROUND_FLOOR)), int(ns.to_integral_value(rounding=ROUND_CEILING))
    except InvalidOperation as error:
        raise ServingGateError(f"invalid {name}") from error


def _range(value, name):
    _keys(value, {"min", "max"}, name)
    low = _integer(value["min"], name + " minimum", 1)
    high = _integer(value["max"], name + " maximum", low)
    return low, high


def _owned_observations(rows, identity):
    for row in rows:
        require(_identity(row.get("server_identity")) == identity,
                "observation belongs to another owned server")
        _interval(row)


def _completed_response(request, attempt, wire):
    require(wire.get("outcome") == "clean_success", "planned completion was not clean")
    require(type(attempt.get("status")) is int and attempt["status"] == 200,
            "clean completion lacks HTTP 200")
    require(attempt.get("transport_error") is None, "completion has a transport interruption")
    start, end = _interval(attempt)
    response = _mapping(wire.get("response"), "completion response")
    require(any(isinstance(response.get(field), str) and response[field]
                for field in ("content", "reasoning")), "completion emitted no output")
    usage = usage_counts(response.get("usage"))
    for field in ("prompt_tokens", "completion_tokens"):
        low, high = _range(request.get(field), field)
        require(low <= usage[field] <= high, f"{field} outside frozen scenario range")
    return {"id": request["id"], "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "started_ns": start, "finished_ns": end, "wall_ns": end - start}


def _clean(request, attempt, wire, generation):
    result = _completed_response(request, attempt, wire)
    require(generation.get("wire_outcome") == "clean_success"
            and generation.get("generation_affected") is False,
            "completion crosses a worker generation")
    before = _integer(generation.get("generation_before"), "generation before")
    require(type(generation.get("generation_after")) is int
            and generation["generation_after"] == before, "worker generation changed")
    require(type(generation.get("clean_latency_ns")) is int
            and generation["clean_latency_ns"] == result["wall_ns"], "clean timing differs")
    return {**result, "generation": before}


def _drain_capture_interval(row, times):
    """Client capture starts before exit observation; read completion has its own bound."""
    start, end = _interval(row)
    require(start < times["exit_before_ns"], "capture started at or after owned process exit observation")
    require(end <= times["deadline_ns"], "client capture exceeded frozen drain allowance")
    return start, end


def _drain_generations(identity, attempts, health_samples, wire, times):
    """Observe counter changes without manufacturing a post-response health sample."""
    samples = []
    previous_end, previous_generation = -1, -1
    for sample in health_samples:
        start, end = _drain_capture_interval(sample, times)
        require(start > previous_end, "drain health intervals must be ordered and disjoint")
        require(type(sample.get("status")) is int and sample["status"] in (200, 503)
                and isinstance(sample.get("body"), bytes), "invalid captured drain health")
        payload = json_object(sample["body"])
        generation = _integer(_mapping(payload.get("worker"), "health worker").get("generation"),
                              "observed worker generation")
        require(generation >= previous_generation, "decreasing worker generation")
        samples.append({"id": sample["id"], "started_ns": start, "finished_ns": end,
                        "generation": generation,
                        # The reply was produced by the owned server before it exited;
                        # its later client read is not a post-exit generation observation.
                        "generation_observation_latest_ns": min(end, times["exit_after_ns"]),
                        "client_read_after_exit_observation": end > times["exit_after_ns"]})
        previous_end, previous_generation = end, generation
    require(len(samples) >= 2, "drain needs before and during-drain health observations")
    results = []
    for attempt in attempts:
        start, end = _interval(attempt)
        before = [sample for sample in samples if sample["finished_ns"] <= start]
        after = [sample for sample in samples if sample["started_ns"] >= end]
        require(before, "drain request lacks an initial health boundary")
        left = before[-1]
        right = after[0] if after else None
        affected = left["generation"] != right["generation"] if right else None
        outcome = wire[attempt["id"]]["outcome"]
        results.append({"id": attempt["id"], "wire_outcome": outcome,
                        "scope": "health_bracketed" if right else "unobserved_tail",
                        "before_sample_id": left["id"], "after_sample_id": right["id"] if right else None,
                        "generation_before": left["generation"],
                        "generation_after": right["generation"] if right else None,
                        "generation_affected": affected,
                        "clean_latency_ns": end - start if right and not affected and outcome == "clean_success" else None})
    return {"scope": "captured_health_interval_only", "server_identity": dict(identity),
            "samples": samples,
            "observed_generation_changes": sum(a["generation"] != b["generation"]
                                               for a, b in zip(samples, samples[1:])),
            "observed_generation_increments": samples[-1]["generation"] - samples[0]["generation"],
            "last_health_finished_ns": samples[-1]["finished_ns"],
            "last_generation_observation_latest_ns": samples[-1]["generation_observation_latest_ns"],
            "terminal_generation_evidence": None, "lifecycle_exit_proves_final_generation": False,
            "requests": results,
            "generation_unqualified_ids": [r["id"] for r in results if r["after_sample_id"] is None],
            "clean_latency_ns": [r["clean_latency_ns"] for r in results if r["clean_latency_ns"] is not None]}


def _probe(row, identity, path, status):
    require(row.get("server_identity") == identity, "probe belongs to another server")
    require(row.get("path") == path, "probe endpoint differs")
    require(type(row.get("status")) is int and row["status"] == status, "probe status differs")
    require(row.get("transport_error") is None, "probe has a transport interruption")
    require(isinstance(row.get("body"), bytes), "probe body must be captured bytes")
    _interval(row)
    return json_object(row["body"])


def _retry_after(headers, expected):
    require(isinstance(headers, list), "raw response header list is missing")
    found = []
    for pair in headers:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(isinstance(part, str) for part in pair), "invalid captured header pair")
        if pair[0].lower() == "retry-after":
            found.append(pair[1])
    require(len(found) == 1, "missing or duplicate Retry-After")
    value = found[0].strip(" \t")
    require(value.isascii() and value.isdecimal() and len(value) <= 10
            and int(value) == expected, "Retry-After differs from frozen drain policy")


def _process_identity(row, name):
    _mapping(row, name)
    _integer(row.get("pid"), name + " PID", 2)
    _integer(row.get("ppid"), name + " PPID")
    _integer(row.get("pgid"), name + " PGID", 2)
    require(row.get("identity_source") == "linux_proc_start_ticks", "drain requires Linux birth identity")
    start = _text(row.get("start_time"), name + " start time")
    require(start.isascii() and start.isdecimal(), "invalid Linux start ticks")
    return row["pid"], start


def _exit(row, name):
    _mapping(row, name)
    require(type(row.get("returncode")) is int and row["returncode"] == 0
            and type(row.get("wait_status")) is int and row["wait_status"] == 0
            and type(row.get("exit_code")) is int and row["exit_code"] == 0
            and row.get("signal", "missing") is None, f"{name} did not exit normally with status zero")


def _signal_interval(event):
    begin, _ = _seconds(event.get("monotonic"), "signal action start")
    finish_start, finish_end = _seconds(event.get("finished_monotonic"), "signal action finish")
    require(begin <= finish_start and event["monotonic"] <= event["finished_monotonic"],
            "inverted signal action interval")
    sent_end = None
    if event.get("result") == "sent":
        sent_start, sent_end = _seconds(event.get("sent_monotonic"), "post-syscall signal bound")
        require(begin <= sent_start <= sent_end <= finish_end
                and event["monotonic"] <= event["sent_monotonic"] <= event["finished_monotonic"],
                "post-syscall signal bound outside action interval")
    else:
        require("sent_monotonic" not in event, "unsent signal claims a delivery timestamp")
    return begin, sent_end, finish_end


def _drain_lifecycle(config, lifecycle):
    record = _mapping(lifecycle, "lifecycle receipt")
    require(record.get("schema") == "memra-owned-server-v1" and record.get("state") == "finished",
            "missing finished owned-server lifecycle")
    require(record.get("errors") == [] and not any(k in record for k in
            ("controller_error", "emergency_errors", "emergency_signals")),
            "lifecycle records errors or emergency cleanup")
    owner = _process_identity(record.get("server"), "lifecycle server")
    boot = _text(record.get("boot_id"), "lifecycle boot")
    try:
        require(str(uuid.UUID(boot)) == boot, "invalid boot UUID")
    except ValueError:
        require(False, "invalid boot UUID")
    identity = {"pid": owner[0], "start_identity": f"{boot}:{owner[1]}"}
    require(identity == config["server_identity"], "lifecycle and observations have different server births")
    require(record.get("linux_subreaper") is True, "drain lacks Linux descendant reaping")
    stop = _mapping(record.get("stop"), "stop record")
    require(stop.get("reason") == config["stop_reason"], "stop cause differs from planned drain")
    require(stop.get("server_returncode_before_cleanup", "missing") is None, "server exited before drain")
    stop_start, stop_end = _seconds(stop.get("monotonic"), "stop time")
    timeout = _integer(config["drain_timeout_ns"], "drain deadline", 1)
    limits = _mapping(record.get("timeouts"), "lifecycle timeouts")
    require(_seconds(limits.get("drain"), "recorded drain budget") == (timeout, timeout),
            "lifecycle drain budget differs from frozen scenario")
    # Use the earlier possible stop instant for an upper-bound deadline.
    deadline = stop_start + timeout
    ended = _mapping(record.get("server_exit"), "server exit")
    require(_process_identity(ended.get("identity"), "exited server") == owner
            and type(ended.get("pid")) is int and ended["pid"] == owner[0],
            "exit record belongs to another process")
    _exit(ended, "server")
    require(ended.get("before_cleanup") is False, "server did not exit during drain")
    exit_start, exit_end = _seconds(ended.get("observed_monotonic"), "server exit time")
    require(stop_end <= exit_start <= exit_end <= deadline, "server exit outside drain deadline")
    cleanup = _mapping(record.get("cleanup"), "cleanup")
    require(cleanup.get("complete") is True and cleanup.get("raw_final") is True
            and cleanup.get("escalated") is False and cleanup.get("remaining") == []
            and cleanup.get("reaping_scope") == "linux_subreaper", "incomplete or escalated cleanup")
    cleanup_start, cleanup_end = _seconds(cleanup.get("finished_monotonic"), "actual cleanup finish")
    require(exit_end <= cleanup_start <= cleanup_end <= deadline,
            "actual cleanup finish is before primary reap or after deadline")
    duration_start, duration_end = _seconds(cleanup.get("elapsed_s"), "cleanup elapsed")
    # elapsed_s is a floating subtraction in the producer. Allow only the source
    # floats' rounding error when cross-checking it, never widen the hard deadline.
    precision = max(1, math.ceil(sum(math.ulp(value) for value in
        (stop["monotonic"], cleanup["finished_monotonic"], cleanup["elapsed_s"])) * 1_000_000_000))
    require(stop_start + duration_start <= cleanup_end + precision
            and stop_end + duration_end >= cleanup_start - precision and duration_end <= timeout,
            "cleanup elapsed disagrees with actual finish or frozen deadline")
    known_rows = record.get("observed_processes")
    require(isinstance(known_rows, list) and known_rows, "missing owned process inventory")
    known = [_process_identity(row, "observed process") for row in known_rows]
    require(len(set(known)) == len(known) and owner in known, "ambiguous owned process inventory")
    observations = record.get("ownership_observations")
    require(isinstance(observations, list), "missing timed ownership observations")
    sightings = {}
    for row in observations:
        _mapping(row, "ownership observation")
        key = _process_identity(row.get("identity"), "observed identity")
        require(key in known and key not in sightings, "unknown or duplicate timed ownership")
        first = _seconds(row.get("first_seen_monotonic"), "first ownership observation")
        last = _seconds(row.get("last_seen_monotonic"), "last ownership observation")
        require((first == last or first[1] <= last[0]) and last[1] <= cleanup_start,
                "invalid ownership observation chronology")
        sightings[key] = (first, last)
    require(set(sightings) == set(known), "timed ownership census is incomplete")
    require(sightings[owner][1][1] <= exit_start, "primary reaped before last ownership observation")
    signals = record.get("signals")
    require(isinstance(signals, list) and signals, "no observed drain signal")
    primary_terms = []
    for event in signals:
        _mapping(event, "signal record")
        key = (event.get("pid"), event.get("start_time"))
        require(type(key[0]) is int and key in known, "signal did not target an observed owner")
        require(type(event.get("signal")) is int and event["signal"] == int(signal.SIGTERM),
                "drain escalated or used a non-TERM signal")
        require(event.get("result") in ("sent", "already_exited"), "signal error or identity refusal")
        if event["result"] == "sent":
            require(event.get("method") in ("pidfd", "start_time_checked_pid"),
                    "signal lacks process-identity-checked delivery")
        begin, sent_end, action_end = _signal_interval(event)
        require(stop_start <= begin and sightings[key][0][1] <= begin and action_end <= cleanup_start,
                "signal action outside ownership/cleanup interval")
        if key == owner and event["result"] == "sent":
            require(sent_end <= exit_start, "TERM send completed after recorded server reap")
            primary_terms.append((begin, sent_end))
        elif key != owner:
            require(begin >= exit_end, "worker was signalled before primary finished drain")
    require(len(primary_terms) == 1, "drain needs exactly one owned primary SIGTERM send")
    descendants = record.get("descendant_exits")
    require(isinstance(descendants, list), "missing descendant exit records")
    seen = set()
    for event in descendants:
        _mapping(event, "descendant exit")
        key = _process_identity(event.get("identity"), "exited descendant")
        require(key in known and key != owner and key not in seen
                and type(event.get("pid")) is int and event["pid"] == key[0],
                "unknown or duplicate descendant exit")
        seen.add(key)
        start, end = _seconds(event.get("observed_monotonic"), "descendant exit time")
        require(sightings[key][1][1] <= start and end <= cleanup_start,
                "descendant reap is before last observation or after actual cleanup finish")
        if event.get("returncode") == -int(signal.SIGTERM):
            # An orphan terminated by our post-primary cleanup may exit by TERM;
            # that is distinct from escalation or an unexplained worker failure.
            require(type(event.get("returncode")) is int
                    and type(event.get("wait_status")) is int and event["wait_status"] == 15
                    and event.get("exit_code", "missing") is None
                    and type(event.get("signal")) is int and event["signal"] == 15
                    and any((s.get("pid"), s.get("start_time")) == key and s.get("result") == "sent"
                            and _signal_interval(s)[1] <= start for s in signals),
                    "descendant signal exit has no matching owned cleanup TERM")
        else:
            _exit(event, "descendant")
    require(type(cleanup.get("descendants_reaped")) is int
            and cleanup["descendants_reaped"] == len(descendants), "descendant reap count differs")
    retirements = record.get("descendant_retirements")
    require(isinstance(retirements, list), "missing identity-retirement census")
    retired = []
    for row in retirements:
        _mapping(row, "retirement")
        key = _process_identity(row.get("identity"), "retired identity")
        require(key in known and key != owner and key not in seen
                and type(row.get("pid")) is int and row["pid"] == key[0],
                "unknown, duplicate or reaped retirement identity")
        require(row.get("kind") == "identity_disappeared" and row.get("source") == "linux_proc_stat"
                and row.get("exit_status", "missing") is None, "invalid nonwaited retirement observation")
        last = _seconds(row.get("last_seen_monotonic"), "retirement last sighting")
        begin = _seconds(row.get("started_monotonic"), "retirement check start")
        end = _seconds(row.get("finished_monotonic"), "retirement check finish")
        require(last == sightings[key][1] and last[1] <= begin[0] <= begin[1] <= end[0]
                and end[1] <= cleanup_start, "retirement outside ownership/cleanup chronology")
        replacement = row.get("replacement_start_time", "missing")
        if row.get("observation") == "absent":
            require(replacement is None, "absent PID has a replacement birth")
        else:
            require(row.get("observation") == "different_birth" and isinstance(replacement, str)
                    and replacement.isascii() and replacement.isdecimal() and replacement != key[1],
                    "retirement does not establish the old birth disappeared")
        seen.add(key)
        retired.append({"pid": key[0], "start_time": key[1], "exit_status": None})
    require(seen == set(known) - {owner}, "known owned descendant lacks exit or identity retirement")
    require(type(cleanup.get("descendants_retired")) is int
            and cleanup["descendants_retired"] == len(retirements), "descendant retirement count differs")
    supervisor = _mapping(record.get("supervisor_exit"), "supervisor exit")
    require(type(supervisor.get("returncode")) is int and supervisor["returncode"] == 0
            and supervisor.get("reaped") is True, "supervisor did not finish and reap cleanly")
    return {"stop_requested_ns": stop_start, "sigterm_before_ns": primary_terms[0][0],
            "sigterm_after_ns": primary_terms[0][1], "exit_before_ns": exit_start,
            "exit_after_ns": exit_end, "cleanup_finished_ns": cleanup_end,
            "retired_without_wait_status": retired, "deadline_ns": deadline}


def evaluate_scenario(config, *, accounting, generations=None, attempts, health_samples,
                      status_samples=None, lifecycle=None):
    """Validate one frozen completed_group/drain cell, without mutating any input."""
    _mapping(config, "scenario config")
    scenario = config.get("scenario")
    require(scenario in ("completed_group", "drain"), "unknown scenario (skip is not a scenario)")
    common = {"scenario", "server_identity", "requests"}
    _keys(config, common | ({"stop_reason", "drain_timeout_ns", "retry_after_s"}
                           if scenario == "drain" else set()), "scenario config")
    identity = _identity(config["server_identity"])
    planned = _by_id(config["requests"], "planned requests")
    observed = _by_id(attempts, "attempts")
    require(planned.keys() == observed.keys(), "planned and observed request IDs differ")
    healthy = _by_id(health_samples, "health samples")
    require(not (planned.keys() & healthy.keys()), "health and request IDs overlap")
    _owned_observations(attempts + health_samples, identity)
    for sample in health_samples:
        require(sample.get("path") == "/health" and sample.get("transport_error") is None,
                "health samples must be complete /health captures")
    schedule = []
    for request in planned.values():
        _text(request.get("model"), "model")
        require(isinstance(request.get("wire"), str) and request["wire"] in WIRE_VALIDATORS,
                "unknown request wire format")
        fields = {"id", "model", "wire", "prompt_tokens", "completion_tokens"}
        if scenario == "drain":
            require(request.get("role") in ("inflight", "new_admission"), "invalid drain request role")
            fields.add("role")
            if request["role"] == "new_admission":
                fields -= {"prompt_tokens", "completion_tokens"}
        _keys(request, fields, "planned request")
        schedule.append({key: request[key] for key in ("id", "model", "wire")})
    # These existing parsers are the authority. Aggregate passed/count booleans
    # are not consulted; supplied per-request derivations must agree with raw data.
    wire = account_attempts(schedule, attempts)
    supplied_wire = _by_id(_mapping(accounting, "accounting").get("requests"), "wire results")
    require(supplied_wire == _by_id(wire["requests"], "derived wire results"), "wire accounting differs from observations")
    completed = []
    if scenario == "completed_group":
        worker = account_generations(wire, attempts, health_samples)
        supplied_worker = _by_id(_mapping(generations, "generations").get("requests"), "generation results")
        require(supplied_worker == _by_id(worker["requests"], "derived generation results")
                and generations.get("server_identity") == identity,
                "generation accounting differs from observations")
        for key, request in planned.items():
            completed.append(_clean(request, observed[key], supplied_wire[key], supplied_worker[key]))
        require(status_samples in (None, []) and lifecycle is None,
                "completed_group does not evaluate lifecycle or status probes")
        return {"scenario": scenario, "server_identity": dict(identity),
                "planned": len(planned), "completed": completed}

    require(generations is None, "drain derives interval-only generation evidence; pass generations=None")
    _text(config["stop_reason"], "planned stop reason")
    require(config["stop_reason"] not in {"cancelled", "scope_exception", "startup_timeout",
            "overall_timeout", "caller_wait_timeout", "controller_eof", "server_exit", "exit_before_ready"}
            and not config["stop_reason"].startswith("supervisor_signal_"), "not a requested graceful drain")
    retry = _integer(config["retry_after_s"], "Retry-After seconds", 1)
    require(retry <= 60, "drain retry bound exceeds protocol contract")
    times = _drain_lifecycle(config, lifecycle)
    worker = _drain_generations(identity, attempts, health_samples, supplied_wire, times)
    require(worker["observed_generation_increments"] == 0, "worker generation changed in observed drain interval")
    worker_rows = {row["id"]: row for row in worker["requests"]}
    inflight = [key for key, request in planned.items() if request["role"] == "inflight"]
    new = [key for key, request in planned.items() if request["role"] == "new_admission"]
    require(inflight and new, "drain needs inflight and new-admission requests")
    for key in inflight:
        completion = _completed_response(planned[key], observed[key], supplied_wire[key])
        generation = worker_rows[key]
        completed.append({**completion, "generation_scope": generation["scope"],
                          "generation_affected": generation["generation_affected"],
                          "clean_latency_ns": generation["clean_latency_ns"]})
        start, end = _drain_capture_interval(observed[key], times)
        require(start < times["sigterm_before_ns"] and end > times["sigterm_after_ns"],
                "inflight capture did not span TERM")
    for key in new:
        row = observed[key]
        start, _ = _drain_capture_interval(row, times)
        require(start > times["sigterm_after_ns"], "new admission started before drain signal")
        require(row.get("transport_error") is None and type(row.get("status")) is int
                and row["status"] == 503, "new admission was not HTTP 503")
        error = _mapping(json_object(row["body"]).get("error"), "drain error")
        require(error.get("code") == "draining" and error.get("type") == "server_error",
                "new admission was not a typed drain refusal")
        require(supplied_wire[key].get("outcome") == "refused", "new admission was not refused")
        _retry_after(row.get("headers"), retry)
    probes = _by_id(status_samples, "readiness probes")
    require(not (probes.keys() & (planned.keys() | healthy.keys())), "duplicate observation IDs")
    _owned_observations(status_samples, identity)
    for probe in probes.values():
        payload = _probe(probe, identity, "/readyz", 503)
        require(payload.get("status") == "not_ready", "readiness probe did not report not_ready")
        start, _ = _drain_capture_interval(probe, times)
        require(start > times["sigterm_after_ns"], "readiness probe started before drain signal")
        _retry_after(probe.get("headers"), retry)
    draining = []
    for key, row in healthy.items():
        payload = json_object(row["body"])
        if payload.get("status") == "draining":
            _probe(row, identity, "/health", 200)
            start, _ = _drain_capture_interval(row, times)
            require(start > times["sigterm_after_ns"], "draining health started before drain signal")
            draining.append(key)
    require(draining, "no captured HTTP 200 draining health")
    return {"scenario": scenario, "server_identity": dict(identity), "planned": len(planned),
            "completed": completed, "refused_ids": new, "draining_health_ids": draining,
            "readiness_probe_ids": list(probes), "wire_counts": dict(Counter(r["outcome"] for r in wire["requests"])),
            "terminal_boundary": "owned_process_exit_and_cleanup",
            "capture_deadline_ns": times["deadline_ns"],
            "client_reads_finished_after_exit_observation": [row["id"] for row in
                attempts + health_samples + status_samples if row["finished_ns"] > times["exit_after_ns"]],
            "timing_scope": "client read completion is not server last-write time",
            "generation_observations": worker,
            **times}
