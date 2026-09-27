"""Sealed-byte consistency controls. Synthetic fixtures do not prove live ownership."""

import copy
import hashlib
import json
import unittest

from serving_capture import encoded
from serving_evidence import read_group_capture
from serving_lifecycle import account_generations
from serving_release import ServingGateError, account_attempts
from test_serving_policy import BOOT, OWNER, PROCESS, drain, health, observation, chat


def fixture(change=None):
    request = {"id": "r1", "model": "gate", "wire": "chat_json", "path": "/v1/chat/completions",
               "payload": {"model": "gate", "stream": False, "messages": [], "temperature": 0}}
    group = {"id": "g1", "mode": "serial", "requests": [request]}
    plan = {"schema": "memra-serving-capture-plan-v1", "server": {
        "argv": ["/fixture/server", "--model", "/fixture/model"], "env": {}, "cwd": "/fixture",
        "host": "127.0.0.1", "port": 18001, "startup_timeout": 1, "overall_timeout": 10,
        "drain_timeout": 3, "kill_timeout": 1},
        "identities": {"server_binary": "/fixture/server", "model": "/fixture/model"},
        "groups": [group], "http": {"connect_timeout": 1, "read_timeout": 1,
                                      "wall_timeout": 2, "max_body_bytes": 16384}}
    identities = {k: {"bytes": 1, "sha256": hashlib.sha256(k.encode()).hexdigest()} for k in plan["identities"]}
    lifecycle = drain()["lifecycle"]
    lifecycle.update(argv=plan["server"]["argv"], cwd="/fixture", env_keys=[], started_monotonic=.1,
                     env_sha256=hashlib.sha256(b"{}").hexdigest())
    lifecycle["stop"]["reason"] = "capture_complete"

    def proof(start, end):
        return {"method": "listener_identity", "owner": dict(PROCESS), "details": {
            "schema": "memra-linux-listener-v1", "endpoint": {"host": "127.0.0.1", "port": 18001},
            "boot_id": BOOT, "owner_before": dict(PROCESS), "owner_after": dict(PROCESS),
            "clock": "monotonic_ns", "started_ns": int(start * 1e9), "finished_ns": int(end * 1e9),
            "samples": [{"inodes": [77], "primary_fds": [5]}, {"inodes": [77], "primary_fds": [5]}]}}

    startup, before, after = proof(.5, .6), proof(.8, .9), proof(1.92, 1.99)
    lifecycle["ready"] = {**copy.deepcopy(startup), "monotonic": .7}
    attempts = [observation("r1", 1.2, 1.8, 200, chat(), method="POST", path=request["path"])]
    samples = [health("before", 1), health("after", 1.9)]
    for sample in samples: sample["method"] = "GET"
    values = dict(plan=copy.deepcopy(plan), identities=copy.deepcopy(identities), group=copy.deepcopy(group),
                  attempts=attempts, health=samples, lifecycle=lifecycle,
                  startup=startup, before=before, after=after)
    if change: change(values)
    blobs = {}

    def put(value):
        raw = value if isinstance(value, bytes) else encoded(value)
        digest = hashlib.sha256(raw).hexdigest();name = "blobs/" + digest;blobs[name] = raw
        return {"path": name, "sha256": digest}

    def observed(value):
        return put({**value, "body": put(value["body"])})

    accounting = account_attempts(group["requests"], values["attempts"])
    generations = account_generations(accounting, values["attempts"], values["health"])
    if values.get("forge_accounting"):accounting["counts"] = {"clean_success": 1000}
    entry = {"id": "g1", "state": "captured", "errors": {}, "schedule": put(values["group"]),
             "observations": [observed(r) for r in values["attempts"]],
             "health_before": observed(values["health"][0]), "health_after": observed(values["health"][1]),
             "wire_accounting": put(accounting), "generation_accounting": put(generations),
             "listener_before": put(values["before"]), "listener_after": put(values["after"])}
    capture = {"schema": "memra-serving-capture-v1", "state": "captured", "errors": [],
        "qualification": False, "clock": "monotonic_ns", "plan": put(values["plan"]), "groups": [entry],
        "identities_before": values["identities"], "identities_after": values["identities"],
        "startup_listener": put(values["startup"]), "startup_observations": [observed(values["health"][0])],
        "listener_observations": [], "lifecycle": put(values["lifecycle"]),
        **{name: put((name + "\n").encode()) for name in ("output.log", "events.jsonl", "supervisor.log")}}
    capture["payloads"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in blobs.items()}
    return capture, blobs, plan, identities


def read(data, reader=None, **overrides):
    capture, blobs, plan, identities = data
    raw = encoded(capture)
    args = dict(expected_capture_sha256=hashlib.sha256(raw).hexdigest(), expected_plan=plan,
                expected_identities=identities)
    args.update(overrides)
    return read_group_capture(raw, reader or blobs.__getitem__, **args)


def replace_blob(data, value):
    raw = value if isinstance(value, bytes) else encoded(value)
    digest = hashlib.sha256(raw).hexdigest();path = "blobs/" + digest
    data[1][path] = raw;data[0]["payloads"][path] = digest
    return {"path": path, "sha256": digest}


def metrics_fixture():
    data = fixture();capture, _, plan, _ = data
    group = capture["groups"][0];plan["groups"][0]["metrics"] = True
    capture["plan"] = replace_blob(data, plan)
    group["schedule"] = replace_blob(data, plan["groups"][0])
    for key, start, end in (("metrics_before", 1.1, 1.11), ("metrics_after", 1.81, 1.82)):
        sample = observation("g1-" + key.replace("_", "-"), start, end, 200,
                             {"prompt_tokens_in": 1}, method="GET", path="/metrics")
        sample["body"] = replace_blob(data, sample["body"])
        group[key] = replace_blob(data, sample)
    return data


class EvidenceTests(unittest.TestCase):
    def test_every_http_body_obeys_external_cap_including_diagnostics_and_metrics(self):
        for location in ("request", "health_before", "health_after", "startup_observations",
                         "metrics_before", "metrics_after"):
            for excess in (0, 1):
                with self.subTest(location=location, excess=excess):
                    data = metrics_fixture() if location.startswith("metrics_") else fixture()
                    capture, blobs, plan, _ = data
                    group = capture["groups"][0]
                    if location == "request":
                        container, key = group["observations"], 0
                    elif location == "startup_observations":
                        container, key = capture[location], 0
                    else:
                        container, key = group, location
                    value = json.loads(blobs[container[key]["path"]])
                    body = blobs[value["body"]["path"]]
                    limit = plan["http"]["max_body_bytes"]
                    value["body"] = replace_blob(data, body + b" " * (limit + excess - len(body)))
                    container[key] = replace_blob(data, value)
                    if excess:
                        with self.assertRaisesRegex(ServingGateError, "body.*limit"):
                            read(data)
                    else:
                        self.assertFalse(read(data)["qualification"])

    def test_metrics_roundtrip_retains_raw_values_without_claiming_cache_success(self):
        for status, body in ((200, b'{"prompt_tokens_in":1}'), (503, b'{"error":"busy"}'),
                             (200, b'{')):
            data = metrics_fixture();group = data[0]["groups"][0]
            sample = json.loads(data[1][group["metrics_after"]["path"]])
            sample.update(status=status, body=replace_blob(data, body))
            group["metrics_after"] = replace_blob(data, sample)
            original = copy.deepcopy(data);result = read(data)
            self.assertEqual(data, original)
            self.assertFalse(result["qualification"])
            decoded = result["groups"][0]
            self.assertEqual(decoded["accounting"]["attempted"], 1)
            self.assertEqual(decoded["metrics_samples"][1]["body"], body)
            self.assertEqual(decoded["metrics_samples"][1]["status"], status)
        self.assertNotIn("metrics_samples", read(fixture())["groups"][0])

    def test_metrics_refs_must_match_the_external_plan_and_payload_manifest(self):
        data = metrics_fixture();del data[0]["groups"][0]["metrics_after"]
        with self.assertRaises(ServingGateError):read(data)
        data = metrics_fixture();data[2]["groups"][0]["metrics"] = False
        data[0]["plan"] = replace_blob(data, data[2])
        data[0]["groups"][0]["schedule"] = replace_blob(data, data[2]["groups"][0])
        with self.assertRaises(ServingGateError):read(data)
        data = metrics_fixture();ref = data[0]["groups"][0]["metrics_after"]
        del data[0]["payloads"][ref["path"]]
        with self.assertRaises(ServingGateError):read(data)
        data = metrics_fixture();data[0]["groups"][0]["metrics_after"] = data[0]["groups"][0]["metrics_before"]
        with self.assertRaises(ServingGateError):read(data)

    def test_metrics_owner_endpoint_identity_and_completion_are_bound(self):
        changes = ({"server_identity": dict(OWNER, pid=101)}, {"method": "POST"},
                   {"path": "/health"}, {"id": "another-metrics-before"},
                   {"transport_error": "truncated chunk"})
        for mutation in changes:
            data = metrics_fixture();group = data[0]["groups"][0]
            sample = json.loads(data[1][group["metrics_before"]["path"]]);sample.update(mutation)
            group["metrics_before"] = replace_blob(data, sample)
            with self.subTest(mutation=mutation), self.assertRaises(ServingGateError):read(data)

    def test_metrics_must_bracket_all_requests_within_health_samples(self):
        changes = (("metrics_before", {"started_ns": 999_000_000}),
                   ("metrics_before", {"finished_ns": 1_200_000_001}),
                   ("metrics_after", {"started_ns": 1_799_999_999}),
                   ("metrics_after", {"finished_ns": 1_900_000_001}),
                   ("metrics_after", {"finished_ns": 1}),
                   ("metrics_after", {"started_ns": True}))
        for key, mutation in changes:
            data = metrics_fixture();group = data[0]["groups"][0]
            sample = json.loads(data[1][group[key]["path"]]);sample.update(mutation)
            group[key] = replace_blob(data, sample)
            with self.subTest(key=key, mutation=mutation), self.assertRaises(ServingGateError):read(data)

    def test_replays_all_raw_groups_without_changing_inputs_or_claiming_qualification(self):
        data = fixture();original = copy.deepcopy(data)
        result = read(data)
        self.assertEqual(data, original)
        self.assertFalse(result["qualification"])
        self.assertEqual(result["groups"][0]["accounting"]["counts"], {"clean_success": 1})
        self.assertEqual(result["server_identity"], OWNER)

    def test_capture_blob_and_external_identity_bindings(self):
        for overrides in ({"expected_capture_sha256": "0" * 64}, {"expected_identities": {}}):
            with self.assertRaises(ServingGateError):read(fixture(), **overrides)
        with self.assertRaises(ServingGateError):read(fixture(), reader=lambda _: b"changed")
        data = fixture();data[2]["groups"][0]["mode"] = "concurrent"
        with self.assertRaises(ServingGateError):read(data)

    def test_paths_are_checked_before_reader_access_and_references_cannot_escape_manifest(self):
        data = fixture();data[0]["payloads"] = {"../escape": "0" * 64};called = []
        with self.assertRaises(ServingGateError):read(data, reader=lambda p: called.append(p))
        self.assertEqual(called, [])
        data = fixture();del data[0]["payloads"][data[0]["lifecycle"]["path"]]
        with self.assertRaises(ServingGateError):read(data)

    def test_launch_limits_owner_endpoint_and_summaries_cannot_be_substituted(self):
        mutations = [lambda v: v["lifecycle"]["argv"].append("--different"),
                     lambda v: v["lifecycle"].update(env_sha256="0" * 64),
                     lambda v: v["lifecycle"]["timeouts"].update(drain=4),
                     lambda v: v["before"]["details"]["endpoint"].update(port=18002),
                     lambda v: v["before"]["owner"].update(pid=101),
                     lambda v: v["before"]["details"].update(samples=[]),
                     lambda v: v["after"]["details"].update(started_ns=1),
                     lambda v: v["group"]["requests"][0]["payload"].update(temperature=1),
                     lambda v: v.update(forge_accounting=True)]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ServingGateError):read(fixture(change))

    def test_collection_success_keeps_failed_requests_in_denominator(self):
        def malformed(values):values["attempts"][0]["body"] = b"{"
        result = read(fixture(malformed))
        group = result["groups"][0]
        self.assertEqual(group["accounting"]["attempted"], 1)
        self.assertEqual(group["accounting"]["counts"], {"invalid_response": 1})
        self.assertIsNone(group["generations"]["requests"][0]["clean_latency_ns"])
        self.assertFalse(result["qualification"])

    def test_incomplete_or_unknown_capture_and_missing_group_refuse(self):
        for mutation in ({"state": "failed"}, {"qualification": True}, {"errors": ["timeout"]},
                         {"groups": []}, {"extra": True}):
            data = fixture();data[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ServingGateError):read(data)

    def test_group_must_follow_startup_and_finish_before_capture_complete_stop(self):
        def shift(values):
            for row in values["attempts"] + values["health"]:
                row["started_ns"] += 100_000_000_000
                row["finished_ns"] += 100_000_000_000
            for name in ("before", "after"):
                values[name]["details"]["started_ns"] += 100_000_000_000
                values[name]["details"]["finished_ns"] += 100_000_000_000
        with self.assertRaisesRegex(ServingGateError, "startup/shutdown"):
            read(fixture(shift))
        def late_startup(values):
            values["startup"]["details"].update(started_ns=850_000_000, finished_ns=950_000_000)
            values["lifecycle"]["ready"] = {**copy.deepcopy(values["startup"]), "monotonic": .7}
        with self.assertRaisesRegex(ServingGateError, "startup/shutdown"):
            read(fixture(late_startup))

    def test_serial_and_sequential_group_chronology(self):
        def rebuild(data, obj):
            raw = encoded(obj);digest = hashlib.sha256(raw).hexdigest();path = "blobs/" + digest
            data[1][path] = raw;data[0]["payloads"][path] = digest
            return {"path": path, "sha256": digest}
        data = fixture();group = data[0]["groups"][0]
        plan = data[2]["groups"][0]
        plan["requests"].append(dict(plan["requests"][0], id="r2"))
        data[0]["plan"] = rebuild(data, data[2]);group["schedule"] = rebuild(data, plan)
        first = json.loads(data[1][group["observations"][0]["path"]])
        second = {**first, "id": "r2", "started_ns": first["started_ns"] + 1,
                  "finished_ns": first["finished_ns"] + 1}
        group["observations"].append(rebuild(data, second))
        attempts = [{**a, "body": data[1][a["body"]["path"]]} for a in (first, second)]
        samples = [json.loads(data[1][group[k]["path"]]) for k in ("health_before", "health_after")]
        samples = [{**r, "body": data[1][r["body"]["path"]]} for r in samples]
        accounting = account_attempts(plan["requests"], attempts)
        group["wire_accounting"] = rebuild(data, accounting)
        group["generation_accounting"] = rebuild(data, account_generations(accounting, attempts, samples))
        with self.assertRaisesRegex(ServingGateError, "serial request"):
            read(data)
        # A second individually consistent group may not replay the first group's
        # time window; all request/group IDs and derived summaries are distinct.
        data = fixture();first = data[0]["groups"][0];plan2 = copy.deepcopy(data[2]["groups"][0])
        plan2["id"] = "g2";plan2["requests"][0]["id"] = "r2"
        data[2]["groups"].append(plan2);data[0]["plan"] = rebuild(data, data[2])
        second = copy.deepcopy(first);second["id"] = "g2";second["schedule"] = rebuild(data, plan2)
        attempt = json.loads(data[1][first["observations"][0]["path"]]);attempt["id"] = "r2"
        second["observations"] = [rebuild(data, attempt)]
        for key in ("wire_accounting", "generation_accounting"):
            value = json.loads(data[1][first[key]["path"]]);value["requests"][0]["id"] = "r2"
            second[key] = rebuild(data, value)
        data[0]["groups"].append(second)
        with self.assertRaisesRegex(ServingGateError, "previous group"):
            read(data)


if __name__ == "__main__":
    unittest.main()
