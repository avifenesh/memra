"""Read an immutable group-capture bundle for the serving policy layer.

The expected capture digest, request plan and artifact identities come from the
external reviewed release record, never from the bundle being checked. The reader
callback accepts only canonical content-addressed blob paths and returns bytes.
Every declared blob is hash checked, including diagnostics. Request observations
are decoded and wire/generation accounting is independently recomputed.
Optional metrics are decoded only when the external plan requests them. Their
owner, endpoint and before/after order are checked, but statuses and body bytes
remain raw policy inputs; a metrics capture does not establish a cache verdict.

This is a consistency/binding boundary for serving_capture's group format. It
does not authenticate the external release record, infer source/build/GPU leases,
or qualify the required scenario manifest. A complete capture can contain failed
requests; those failures remain in its denominator. Listener observations remain
sampled producer evidence, not an atomic socket lease or a fresh procfs scan.
"""

import hashlib
import json
from pathlib import PurePosixPath
import re

from serving_capture import validate_plan
from serving_completion import _keys, _same
from serving_lifecycle import account_generations
from serving_policy import _drain_lifecycle, _seconds
from serving_release import account_attempts, json_object, require


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOT_KEYS = {"schema", "state", "qualification", "plan", "groups", "startup_observations",
              "listener_observations", "errors", "clock", "identities_before", "identities_after",
              "lifecycle", "output.log", "events.jsonl", "supervisor.log", "payloads", "startup_listener"}
_GROUP_KEYS = {"id", "schedule", "observations", "state", "errors", "listener_before",
               "health_before", "wire_accounting", "health_after", "listener_after", "generation_accounting"}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _interval(value):
    start, end = value.get("started_ns"), value.get("finished_ns")
    require(type(start) is int and type(end) is int and 0 <= start <= end,
            "invalid evidence interval")
    return start, end


def _listener(proof, identity, endpoint):
    """Bind the saved producer's proof to this owner/endpoint and clock interval."""
    require(isinstance(proof, dict) and proof.get("method") == "listener_identity",
            "missing listener ownership evidence")
    details = proof.get("details")
    require(isinstance(details, dict) and details.get("schema") == "memra-linux-listener-v1"
            and details.get("clock") == "monotonic_ns" and _same(details.get("endpoint"), endpoint),
            "listener endpoint/clock differs from plan")
    for owner in [proof.get("owner"), details.get("owner_before"), details.get("owner_after")]:
        require(isinstance(owner, dict) and owner.get("pid") == identity["pid"]
                and isinstance(owner.get("start_time"), str)
                and details.get("boot_id", "") + ":" + owner.get("start_time", "") == identity["start_identity"]
                and owner.get("identity_source") == "linux_proc_start_ticks"
                and owner.get("state") not in ("Z", "X", "x"), "listener owner differs from lifecycle")
    require(isinstance(details.get("samples"), list) and len(details["samples"]) == 2,
            "listener requires both producer snapshots")
    first, second = details["samples"]
    for sample in (first, second):
        require(isinstance(sample, dict) and isinstance(sample.get("inodes"), list)
                and len(sample["inodes"]) == 1 and type(sample["inodes"][0]) is int
                and sample["inodes"][0] > 0 and isinstance(sample.get("primary_fds"), list)
                and sample["primary_fds"] and all(type(fd) is int and fd >= 0 for fd in sample["primary_fds"]),
                "listener snapshot lacks its socket or primary descriptors")
    require(_same(first["inodes"], second["inodes"])
            and _same(first["primary_fds"], second["primary_fds"]), "listener snapshots disagree")
    return _interval(details)


def read_group_capture(capture_bytes, evidence_reader, *, expected_capture_sha256,
                       expected_plan, expected_identities):
    """Decode a captured run; return raw groups and rederived accounting, not a GO."""
    require(isinstance(capture_bytes, bytes) and isinstance(expected_capture_sha256, str)
            and _SHA.fullmatch(expected_capture_sha256) and _digest(capture_bytes) == expected_capture_sha256,
            "capture index differs from externally expected digest")
    capture = json_object(capture_bytes)
    _keys(capture, _ROOT_KEYS, "group capture")
    require(capture["schema"] == "memra-serving-capture-v1" and capture["state"] == "captured"
            and capture["qualification"] is False and capture["errors"] == []
            and capture["clock"] == "monotonic_ns", "capture did not complete cleanly")
    validate_plan(expected_plan)
    require(isinstance(expected_identities, dict)
            and expected_identities.keys() == expected_plan["identities"].keys(),
            "external artifact identity scope differs from plan")
    for identity in expected_identities.values():
        _keys(identity, {"bytes", "sha256"}, "artifact identity")
        require(type(identity["bytes"]) is int and identity["bytes"] >= 0
                and isinstance(identity["sha256"], str) and _SHA.fullmatch(identity["sha256"]),
                "invalid external artifact identity")
    require(_same(capture["identities_before"], expected_identities)
            and _same(capture["identities_after"], expected_identities), "captured artifact identity differs")
    payloads = capture["payloads"]
    require(isinstance(payloads, dict) and 0 < len(payloads) <= 16384, "invalid payload manifest")
    blobs, total = {}, 0
    for name, digest in payloads.items():
        require(isinstance(digest, str) and _SHA.fullmatch(digest) and name == "blobs/" + digest,
                "payload path must be canonical and content addressed")
        data = evidence_reader(name)
        require(isinstance(data, bytes) and _digest(data) == digest, "payload bytes differ from manifest")
        total += len(data)
        require(total <= 512 * 1024 * 1024, "group capture exceeds decoded byte bound")
        blobs[name] = data

    def raw(ref):
        _keys(ref, {"path", "sha256"}, "blob reference")
        require(isinstance(ref["path"], str) and ref["path"] in blobs
                and ref["sha256"] == payloads[ref["path"]], "reference is absent or differs from manifest")
        return blobs[ref["path"]]

    def obj(ref):
        return json_object(raw(ref))

    def observation(ref):
        value = obj(ref)
        value["body"] = raw(value.get("body"))
        require(len(value["body"]) <= expected_plan["http"]["max_body_bytes"],
                "captured HTTP body exceeds the reviewed byte limit")
        return value

    require(_same(obj(capture["plan"]), expected_plan), "captured request plan differs from trusted plan")
    lifecycle = obj(capture["lifecycle"])
    owner = lifecycle.get("server")
    require(isinstance(owner, dict) and isinstance(lifecycle.get("boot_id"), str)
            and isinstance(owner.get("start_time"), str), "lifecycle owner is missing")
    identity = {"pid": owner.get("pid"), "start_identity": lifecycle["boot_id"] + ":" + owner["start_time"]}
    server = expected_plan["server"]
    env_hash = _digest(json.dumps(server["env"], sort_keys=True, separators=(",", ":")).encode())
    require(_same(lifecycle.get("argv"), server["argv"])
            and lifecycle.get("cwd") == str(PurePosixPath(server["cwd"]))
            and lifecycle.get("env_keys") == sorted(server["env"])
            and lifecycle.get("env_sha256") == env_hash, "actual server launch differs from trusted plan")
    limits = {name: server[name + "_timeout"] for name in ("startup", "overall", "drain", "kill")}
    require(_same(lifecycle.get("timeouts"), limits), "actual lifecycle limits differ")
    cleanup = _drain_lifecycle({"server_identity": identity, "stop_reason": "capture_complete",
                                "drain_timeout_ns": int(server["drain_timeout"] * 1_000_000_000)}, lifecycle)
    endpoint = {"host": server["host"], "port": server["port"]}
    startup = obj(capture["startup_listener"])
    startup_interval = _listener(startup, identity, endpoint)
    started = _seconds(lifecycle.get("started_monotonic"), "supervisor start")
    require(started[1] <= startup_interval[0], "startup proof predates process scope")
    require(isinstance(lifecycle.get("ready"), dict)
            and _same({k: v for k, v in lifecycle["ready"].items() if k != "monotonic"}, startup),
            "readiness proof differs from captured startup listener")
    require(isinstance(capture["groups"], list)
            and len(capture["groups"]) == len(expected_plan["groups"]), "group denominator differs")
    # ready.monotonic is the supervisor's loop-entry sample, not a post-acceptance
    # timestamp. The completed startup proof is a sound lower bound for groups.
    previous_group_end = startup_interval[1]
    groups = []
    for group, plan in zip(capture["groups"], expected_plan["groups"]):
        metrics_keys = {"metrics_before", "metrics_after"} if plan.get("metrics", False) else set()
        _keys(group, _GROUP_KEYS | metrics_keys, "captured group")
        require(group["id"] == plan["id"] and group["state"] == "captured" and group["errors"] == {}
                and _same(obj(group["schedule"]), plan), "group schedule or capture state differs")
        require(isinstance(group["observations"], list), "group observations are missing")
        attempts = [observation(ref) for ref in group["observations"]]
        accounting = account_attempts(plan["requests"], attempts)
        indexed = {r["id"]: r for r in attempts}
        attempts = [indexed[r["id"]] for r in plan["requests"]]
        if plan["mode"] == "serial":
            require(all(a["finished_ns"] <= b["started_ns"] for a, b in zip(attempts, attempts[1:])),
                    "serial request intervals overlap or differ from trusted order")
        health = [observation(group[k]) for k in ("health_before", "health_after")]
        for request, attempt in zip(plan["requests"], attempts):
            require(_same(attempt.get("server_identity"), identity) and attempt.get("method") == "POST"
                    and attempt.get("path") == request["path"], "captured request owner/route differs")
        for sample in health:
            require(_same(sample.get("server_identity"), identity) and sample.get("method") == "GET"
                    and sample.get("path") == "/health" and sample.get("transport_error") is None,
                    "incomplete or foreign health sample")
        metrics = []
        if metrics_keys:
            for key in ("metrics_before", "metrics_after"):
                sample = observation(group[key])
                require(_same(sample.get("server_identity"), identity)
                        and sample.get("id") == plan["id"] + "-" + key.replace("_", "-")
                        and sample.get("method") == "GET" and sample.get("path") == "/metrics"
                        and sample.get("transport_error") is None,
                        "incomplete, substituted or foreign metrics sample")
                _interval(sample)
                metrics.append(sample)
            require(_interval(health[0])[1] <= metrics[0]["started_ns"]
                    and metrics[0]["finished_ns"] <= min(r["started_ns"] for r in attempts)
                    and max(r["finished_ns"] for r in attempts) <= metrics[1]["started_ns"]
                    and metrics[1]["finished_ns"] <= _interval(health[1])[0],
                    "metrics do not bracket requests inside the health interval")
        generations = account_generations(accounting, attempts, health)
        require(_same(obj(group["wire_accounting"]), accounting)
                and _same(obj(group["generation_accounting"]), generations), "derived group accounting differs")
        before, after = [obj(group[k]) for k in ("listener_before", "listener_after")]
        before_interval = _listener(before, identity, endpoint)
        after_interval = _listener(after, identity, endpoint)
        require(before_interval[1] <= _interval(health[0])[0]
                and _interval(health[1])[1] <= after_interval[0],
                "listener evidence does not bracket group health observations")
        require(previous_group_end <= before_interval[0]
                and after_interval[1] <= cleanup["stop_requested_ns"],
                "group is outside startup/shutdown interval or overlaps a previous group")
        previous_group_end = after_interval[1]
        groups.append({"id": group["id"], "plan": plan, "attempts": attempts,
                       "health_samples": health, "accounting": accounting, "generations": generations,
                       "listener_before": before, "listener_after": after})
        if metrics_keys:
            groups[-1]["metrics_samples"] = metrics
    for key in ("startup_observations", "listener_observations"):
        require(isinstance(capture[key], list), "missing diagnostic observations")
        for ref in capture[key]:
            value = obj(ref)
            if "body" in value:
                require(len(raw(value["body"])) <= expected_plan["http"]["max_body_bytes"],
                        "diagnostic HTTP body exceeds the reviewed byte limit")
    logs = {name: raw(capture[name]) for name in ("output.log", "events.jsonl", "supervisor.log")}
    return {"capture_sha256": expected_capture_sha256, "plan": expected_plan, "groups": groups,
            "lifecycle": lifecycle, "cleanup": cleanup, "server_identity": identity,
            "logs": logs, "qualification": False, "verified_payloads": len(blobs)}
