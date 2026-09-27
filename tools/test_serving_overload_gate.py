"""Synthetic sealed-byte controls; these do not establish native qualification."""

import copy
import hashlib
import json
import unittest

from serving_capture import encoded
from serving_lifecycle import account_generations
from serving_overload_gate import evaluate_overload_capture
from serving_release import ServingGateError, account_attempts
from test_serving_evidence import fixture as evidence_fixture, replace_blob
from test_serving_policy import OWNER
from test_serving_recovery import fixture as policy_fixture


def fixture(wire="chat_json", mutate_attempts=None):
    cell = policy_fixture(wire)
    cell["program"]["server_identity"] = dict(OWNER)
    data = evidence_fixture()
    capture, blobs, plan, identities = data
    template = copy.deepcopy(capture["groups"][0])
    attempts = cell["attempts"]
    for row in attempts:
        row["server_identity"] = dict(OWNER)
        for key in ("started_ns", "finished_ns"):
            row[key] = 1_000_000_000 + row[key] * 1_000_000
    if mutate_attempts:
        mutate_attempts(attempts)

    def put(value):
        return replace_blob(data, value)

    def observation(row):
        return put({**row, "body": put(row["body"])})

    def listener(start, end):
        value = json.loads(blobs[template["listener_before"]["path"]])
        value["details"].update(started_ns=int(start * 1e9), finished_ns=int(end * 1e9))
        return put(value)

    plan["groups"] = []
    capture["groups"] = []
    for index, (suffix, mode, before, after, lb, la) in enumerate([
            ("pressure", "concurrent", 1.01, 1.45, (.8, .9), (1.48, 1.49)),
            ("recovery", "serial", 1.52, 1.9, (1.50, 1.51), (1.92, 1.99))]):
        selected = [r for r in cell["program"]["requests"] if (r["role"] == "recovery") == bool(index)]
        group = {"id": cell["required"]["id"] + "/" + suffix, "mode": mode,
                 "requests": [{k: r[k] for k in ("id", "model", "wire", "path", "payload")} for r in selected]}
        plan["groups"].append(group)
        observed = [r for r in attempts if r["id"] in {s["id"] for s in selected}]
        health = []
        for label, when in (("before", before), ("after", after)):
            value = copy.deepcopy(cell["health_samples"][0])
            value.update(server_identity=dict(OWNER), id=group["id"] + "-" + label, started_ns=int(when * 1e9),
                         finished_ns=int(when * 1e9) + 1_000_000)
            health.append(value)
        accounting = account_attempts(group["requests"], observed)
        capture["groups"].append({"id": group["id"], "state": "captured", "errors": {},
            "schedule": put(group), "observations": [observation(r) for r in observed],
            "health_before": observation(health[0]), "health_after": observation(health[1]),
            "wire_accounting": put(accounting),
            "generation_accounting": put(account_generations(accounting, observed, health)),
            "listener_before": listener(*lb), "listener_after": listener(*la)})
    capture["plan"] = put(plan)
    raw = encoded(capture)
    return {"required": cell["required"], "program": cell["program"], "capture_bytes": raw,
            "evidence_reader": blobs.__getitem__, "expected_capture_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_plan": plan, "expected_identities": identities}


class OverloadCaptureTests(unittest.TestCase):
    def test_all_wires_replay_complete_pressure_and_recovery(self):
        for wire in ("chat_json", "chat_sse", "native_json", "native_sse"):
            with self.subTest(wire=wire):
                args = fixture(wire)
                before = copy.deepcopy({k: v for k, v in args.items() if k != "evidence_reader"})
                result = evaluate_overload_capture(**args)
                self.assertFalse(result["qualification"])
                self.assertEqual(result["cell"]["planned"], 3)
                self.assertEqual(result["cell"]["wire_counts"], {"clean_success": 2, "refused": 1})
                self.assertEqual(len(result["cell"]["generation_observations"]["requests"]), 3)
                self.assertEqual(before, {k: v for k, v in args.items() if k != "evidence_reader"})

    def test_scope_subsets_duplicates_modes_and_order_refuse_before_reading(self):
        mutations = [lambda g: g.pop(0), lambda g: g.append(copy.deepcopy(g[0])),
                     lambda g: g.reverse(), lambda g: g[0].update(mode="serial"),
                     lambda g: g[1].update(mode="concurrent"),
                     lambda g: g[0]["requests"].pop(),
                     lambda g: g[0]["requests"].reverse(),
                     lambda g: g[0]["requests"].append(copy.deepcopy(g[0]["requests"][0])),
                     lambda g: g[1].update(id="another-cell/recovery")]
        def forbidden(_):
            self.fail("invalid schedule reached capture reader")
        for mutate in mutations:
            args = fixture(); mutate(args["expected_plan"]["groups"]); args["evidence_reader"] = forbidden
            with self.assertRaisesRegex(ServingGateError, "complete pressure and recovery"):
                evaluate_overload_capture(**args)

    def test_sent_payload_and_required_roles_cannot_be_substituted(self):
        for mutate in (lambda p: p["requests"][0]["payload"].update(max_tokens=100),
                       lambda p: p["requests"][1].update(role="peer"),
                       lambda p: p["requests"][0].update(role="unknown"),
                       lambda p: p.update(mode="serial"),
                       lambda p: p.update(cell_id="another/overload_recovery"),
                       lambda p: p["scope"].update(route="another"),
                       lambda p: p["server_identity"].update(pid=999)):
            args = fixture(); mutate(args["program"])
            with self.assertRaises(ServingGateError): evaluate_overload_capture(**args)

    def test_self_consistent_captured_failure_still_fails_policy(self):
        mutations = [lambda a: a[1].update(status=503),
                     lambda a: a[1].update(headers=[]),
                     lambda a: a[0].update(body=b'{"error":{"message":"failed"}}'),
                     lambda a: a[2].update(body=b'{"error":{"message":"failed"}}')]
        for mutate in mutations:
            args = fixture(mutate_attempts=mutate)
            with self.assertRaises(ServingGateError): evaluate_overload_capture(**args)

    def test_external_capture_and_artifact_bindings_remain_required(self):
        for field, value in (("expected_capture_sha256", "0" * 64), ("expected_identities", {})):
            args = fixture(); args[field] = value
            with self.assertRaises(ServingGateError): evaluate_overload_capture(**args)

    def test_exact_body_cap_is_valid_but_self_consistent_over_cap_capture_refuses(self):
        def pad(size):
            def mutate(attempts):
                body = attempts[0]["body"]
                attempts[0]["body"] = body + b" " * (size - len(body))
            return mutate
        self.assertFalse(evaluate_overload_capture(**fixture(mutate_attempts=pad(16384)))["qualification"])
        with self.assertRaisesRegex(ServingGateError, "body.*limit"):
            evaluate_overload_capture(**fixture(mutate_attempts=pad(16385)))


if __name__ == "__main__":
    unittest.main()
