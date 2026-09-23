"""The sent schedule and independently reviewed policy must describe one cell.

These sealed synthetic bundles test consistency, not model or GPU provenance.
The collector/decoder's real HTTP-process tests provide its separate I/O checks.
"""

import copy
import hashlib
import json
import unittest

from serving_capture import encoded
from serving_group_gate import evaluate_group_capture
from serving_lifecycle import account_generations
from serving_release import ServingGateError, account_attempts
from test_serving_cache import fixture as cache_fixture
from test_serving_completion import fixture as completion_fixture
from test_serving_evidence import fixture as evidence_fixture, replace_blob
from test_serving_policy import OWNER


def bound_fixture(scenario, *, metrics=True):
    cell = cache_fixture() if scenario == "cache_restore" else completion_fixture(scenario)
    data = evidence_fixture();capture, blobs, plan, identities = data
    program = cell["program"];program["server_identity"] = dict(OWNER)
    schedule = {"id": cell["required"]["id"], "mode": program["mode"],
                "requests": [{k: r[k] for k in ("id", "model", "wire", "path", "payload")}
                             for r in program["requests"]]}
    if scenario == "cache_restore" and metrics:schedule["metrics"] = True
    plan["groups"] = [schedule];capture["plan"] = replace_blob(data, plan)
    group = capture["groups"][0];group.update(id=schedule["id"], schedule=replace_blob(data, schedule))
    attempts = copy.deepcopy(cell["attempts"])
    for a in attempts:
        a["server_identity"] = dict(OWNER)
        for k in ("started_ns", "finished_ns"):a[k] = 1_200_000_000 + a[k] * 100_000

    def put_observation(a):
        return replace_blob(data, {**a, "body": replace_blob(data, a["body"])})

    group["observations"] = [put_observation(a) for a in attempts]
    health = [json.loads(blobs[group[k]["path"]]) for k in ("health_before", "health_after")]
    health = [{**h, "body": blobs[h["body"]["path"]]} for h in health]
    accounting = account_attempts(schedule["requests"], attempts)
    group["wire_accounting"] = replace_blob(data, accounting)
    group["generation_accounting"] = replace_blob(data, account_generations(accounting, attempts, health))
    if scenario == "cache_restore" and metrics:
        for i, (key, start) in enumerate((("metrics_before", 1_100_000_000),
                                           ("metrics_after", 1_800_000_000))):
            sample = dict(cell["metrics_samples"][i], server_identity=dict(OWNER),
                          id=schedule["id"] + "-" + key.replace("_", "-"),
                          started_ns=start, finished_ns=start + 1_000_000)
            group[key] = put_observation(sample)
    raw = encoded(capture)
    return {"required": cell["required"], "program": program, "capture_bytes": raw,
            "evidence_reader": blobs.__getitem__, "expected_capture_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_plan": plan, "expected_identities": identities}


class GroupGateTests(unittest.TestCase):
    def test_complete_bound_scenarios_keep_their_scope_and_never_qualify_a_release(self):
        for scenario in ("short_prompt", "long_prompt", "wire_completion", "concurrent_completion", "cache_restore"):
            with self.subTest(scenario=scenario):
                bundle = bound_fixture(scenario)
                before = copy.deepcopy({k: v for k, v in bundle.items() if k != "evidence_reader"})
                result = evaluate_group_capture(**bundle)
                self.assertEqual(result["cell"]["scenario"], scenario)
                self.assertEqual(result["cell"]["id"], bundle["required"]["id"])
                self.assertFalse(result["qualification"])
                self.assertGreater(result["verified_payloads"], 0)
                self.assertEqual(before, {k: v for k, v in bundle.items() if k != "evidence_reader"})

    def test_a_passing_response_cannot_hide_a_different_sent_payload(self):
        for field, value in (("temperature", False), ("temperature", 0.5),
                             ("max_tokens", 999), ("messages", [{"role": "user", "content": "different"}])):
            bundle = bound_fixture("short_prompt")
            bundle["program"]["requests"][0]["payload"] = dict(
                bundle["program"]["requests"][0]["payload"], **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ServingGateError):
                evaluate_group_capture(**bundle)

    def test_request_order_and_cardinality_cannot_be_selected_from_a_successful_capture(self):
        for mutation in (lambda r: r.reverse(), lambda r: r.pop(), lambda r: r.append(copy.deepcopy(r[0]))):
            bundle = bound_fixture("wire_completion");mutation(bundle["program"]["requests"])
            with self.assertRaises(ServingGateError):evaluate_group_capture(**bundle)

    def test_whole_capture_must_be_single_group_before_any_payload_is_read(self):
        bundle = bound_fixture("short_prompt")
        bundle["expected_plan"]["groups"].append(copy.deepcopy(bundle["expected_plan"]["groups"][0]))
        def forbidden_reader(_):raise AssertionError("must reject broader scope before reading it")
        bundle["evidence_reader"] = forbidden_reader
        with self.assertRaisesRegex(ServingGateError, "single-group"):
            evaluate_group_capture(**bundle)

    def test_owner_mode_cell_and_scope_cannot_be_reassigned(self):
        mutations = (lambda b: b["program"]["server_identity"].update(pid=101),
                     lambda b: b["program"].update(mode="concurrent"),
                     lambda b: b["program"].update(cell_id="another/short_prompt"),
                     lambda b: b["program"]["scope"].update(route="another"))
        for mutation in mutations:
            bundle = bound_fixture("short_prompt");mutation(bundle)
            with self.assertRaises(ServingGateError):evaluate_group_capture(**bundle)

    def test_independent_answer_and_usage_requirements_still_run(self):
        for key, value in (("answer_oracle", "43"), ("prompt_tokens", {"min": 1, "max": 7})):
            bundle = bound_fixture("short_prompt");bundle["program"]["requests"][0][key] = value
            with self.assertRaises(ServingGateError):evaluate_group_capture(**bundle)

    def test_external_digest_and_artifact_identity_remain_required(self):
        for name, value in (("expected_capture_sha256", "0" * 64), ("expected_identities", {})):
            bundle = bound_fixture("short_prompt");bundle[name] = value
            with self.assertRaises(ServingGateError):evaluate_group_capture(**bundle)

    def test_cache_hit_responses_cannot_replace_metrics_evidence(self):
        with self.assertRaisesRegex(ServingGateError, "captured metrics brackets"):
            evaluate_group_capture(**bound_fixture("cache_restore", metrics=False))

    def test_lifecycle_scenarios_cannot_be_claimed_from_a_completion_capture(self):
        for scenario in ("cancel_queued", "cancel_prime", "cancel_decode", "drain",
                         "overload_recovery", "worker_failure_recovery"):
            bundle = bound_fixture("short_prompt");bundle["required"]["scenario"] = scenario
            with self.subTest(scenario=scenario), self.assertRaisesRegex(ServingGateError, "no supported"):
                evaluate_group_capture(**bundle)


if __name__ == "__main__":unittest.main()
