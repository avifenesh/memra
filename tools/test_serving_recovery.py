"""Raw protocol contract fixtures; no process, socket, GPU or benchmark execution."""

import copy
import json
from pathlib import Path
import unittest

from serving_lifecycle import account_generations
from serving_manifest import required_cells
from serving_recovery import evaluate_overload_recovery_cell
from serving_release import ServingGateError, account_attempts


OWNER = {"pid": 17, "start_identity": "boot-id:71"}
SCOPE = {"id": "tiny", "model": "gate", "route": "normal", "profile": "text-generation-v1"}


def encode(value):
    return json.dumps(value).encode()


def success(wire, prompt=8, count=1, text="42"):
    usage = {"prompt_tokens": prompt, "completion_tokens": count, "total_tokens": prompt + count}
    if wire == "chat_json":
        return encode({"model": "gate", "choices": [{"index": 0, "finish_reason": "stop",
                       "message": {"role": "assistant", "content": text}}], "usage": usage})
    if wire == "chat_sse":
        return (b"data: " + encode({"model": "gate", "choices": [{"index": 0,
            "finish_reason": "stop", "delta": {"content": text}}], "usage": usage})
            + b"\n\ndata: [DONE]\n\n")
    terminal = {"stop_reason": "MaxNew", "n_tokens": count, "prompt_tokens": prompt,
                "cached_tokens": 0, "elapsed_s": .01}
    if wire == "native_json":
        return encode({**terminal, "model": "gate", "text": text, "tokens": [4] * count})
    return (b"event: message\ndata: " + encode({"model": "gate", "id": 4, "text": text})
            + b"\n\nevent: done\ndata: " + encode(terminal) + b"\n\n")


def refusal(code="shed_queue", error_type="rate_limit_error"):
    return encode({"error": {"type": error_type, "code": code, "param": None,
                             "message": "request was not admitted; retry later"}})


def health(name, start, generation=7):
    return {"id": name, "server_identity": dict(OWNER), "method": "GET", "path": "/health",
            "started_ns": start, "finished_ns": start + 1, "status": 200,
            "body": encode({"status": "ok", "worker": {"generation": generation}})}


def fixture(wire="chat_json", code="shed_queue", retry=2):
    required = next(c for c in required_cells(
        Path(__file__).with_name("serving-release.cells.json").read_bytes(), [SCOPE])
                    if c["scenario"] == "overload_recovery")
    program = {"cell_id": required["id"], "scope": dict(SCOPE), "server_identity": dict(OWNER),
               "mode": "overload_then_recovery", "requests": []}
    attempts = []
    for role, start, end in [("peer", 100, 400), ("refused", 200, 220), ("recovery", 600, 800)]:
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        payload = {"model": "gate", "stream": wire.endswith("_sse")}
        payload.update({"messages": [{"role": "user", "content": "6 * 7?"}]}
                       if wire.startswith("chat_") else {"prompt_ids": [1, 2, 3]})
        request = {"id": role, "role": role, "model": "gate", "wire": wire,
                   "path": path, "payload": payload}
        request.update({"error_code": code} if role == "refused" else
                       {"prompt_tokens": {"min": 1, "max": 32},
                        "completion_tokens": {"min": 1, "max": 16}})
        program["requests"].append(request)
        attempts.append({"id": role, "server_identity": dict(OWNER), "method": "POST", "path": path,
                         "started_ns": start, "finished_ns": end, "transport_error": None,
                         "status": 429 if role == "refused" else 200,
                         "body": refusal(code) if role == "refused" else success(wire),
                         "headers": [["Retry-After", str(retry)], ["retry-after-ms", str(retry * 1000)]]
                         if role == "refused" else []})
    return {"required": required, "program": program, "attempts": attempts,
            "health_samples": [health("before", 1), health("between", 500), health("after", 1000)]}


def add_request(bundle, role, suffix, start, end):
    request = copy.deepcopy(next(r for r in bundle["program"]["requests"] if r["role"] == role))
    attempt = copy.deepcopy(next(r for r in bundle["attempts"] if r["id"] == role))
    request["id"] += suffix
    attempt.update(id=request["id"], started_ns=start, finished_ns=end)
    bundle["program"]["requests"].append(request)
    bundle["attempts"].append(attempt)


class OverloadRecoveryTests(unittest.TestCase):
    def refuses(self, bundle, message=None):
        with self.assertRaises(ServingGateError) as error:
            evaluate_overload_recovery_cell(**bundle)
        if message:
            self.assertIn(message, str(error.exception))

    def test_each_wire_and_queue_code_with_source_retry_clamp_endpoints(self):
        for wire in ("chat_json", "chat_sse", "native_json", "native_sse"):
            for code in ("shed_queue", "shed_deadline", "shed_queue_wait"):
                for seconds in (1, 60):
                    with self.subTest(wire=wire, code=code, seconds=seconds):
                        bundle = fixture(wire, code, seconds)
                        original = copy.deepcopy(bundle)
                        result = evaluate_overload_recovery_cell(**bundle)
                        self.assertEqual(bundle, original)
                        self.assertEqual(result["planned"], 3)
                        self.assertEqual(result["wire_counts"], {"clean_success": 2, "refused": 1})
                        self.assertEqual(result["refused"][0]["error_code"], code)
                        self.assertEqual(result["refused"][0]["retry_after_ms"], seconds * 1000)
                        self.assertEqual(result["refused"][0]["spanning_peer_ids"], ["peer"])
                        self.assertEqual({r["role"] for r in result["completed"]}, {"peer", "recovery"})
                        self.assertEqual(len(result["generation_observations"]["requests"]), 3)
                        self.assertNotIn("qualification", result)
                        self.assertNotIn("passed", result)

    def test_every_request_in_multiple_role_denominators_is_retained(self):
        bundle = fixture()
        add_request(bundle, "peer", "-2", 105, 450)
        add_request(bundle, "refused", "-2", 250, 270)
        add_request(bundle, "recovery", "-2", 810, 900)
        # Observation/program list order is not chronology; timestamp checks decide.
        bundle["attempts"].reverse()
        result = evaluate_overload_recovery_cell(**bundle)
        self.assertEqual(result["planned"], 6)
        self.assertEqual(result["wire_counts"], {"clean_success": 4, "refused": 2})
        self.assertEqual(len(result["completed"]), 4)
        self.assertEqual(result["pressure_finished_ns"], 450)
        for role in ("peer", "refused", "recovery"):
            self.assertEqual(len(result["planned_by_role"][role]), 2)
            missing = copy.deepcopy(bundle)
            missing["attempts"] = [r for r in missing["attempts"] if r["id"] != role + "-2"]
            self.refuses(missing, "missing or unknown")

    def test_minimal_outer_health_brackets_cover_the_entire_program(self):
        bundle = fixture()
        bundle["health_samples"].pop(1)
        result = evaluate_overload_recovery_cell(**bundle)
        self.assertEqual(result["generation_observations"]["respawns_observed"], 0)
        self.assertEqual(len(result["generation_observations"]["requests"]), 3)

    def test_queue_code_is_frozen_not_selected_after_observation(self):
        bundle = fixture(code="shed_queue_wait")
        bundle["attempts"][1]["body"] = refusal("shed_queue")
        self.refuses(bundle, "frozen queue error")

    def test_nonqueue_errors_are_not_overload_even_with_valid_retry_headers(self):
        for status, code, error_type in [
                (400, "context_length_exceeded", "invalid_request_error"),
                (400, "model_not_found", "invalid_request_error"),
                (429, "rate_limit_exceeded", "rate_limit_error"),
                (503, "overloaded", "server_error"),
                (503, "server_draining", "server_error"),
                (500, "engine_error", "server_error"),
                (429, "shed_queue", "server_error")]:
            with self.subTest(status=status, code=code):
                bundle = fixture()
                bundle["attempts"][1].update(status=status, body=refusal(code, error_type))
                self.refuses(bundle)
                # A capture-selected expected code must not expand the source contract.
                if code != "shed_queue":
                    bundle["program"]["requests"][1]["error_code"] = code
                    self.refuses(bundle, "source queue refusal")

    def test_correct_queue_body_with_wrong_status_or_success_in_refusal_slot_fails(self):
        for status in (200, 400, 403, 500, 503):
            bundle = fixture(); bundle["attempts"][1]["status"] = status
            self.refuses(bundle)
        bundle = fixture()
        bundle["attempts"][1].update(status=200, body=success("chat_json"))
        self.refuses(bundle)

    def test_client_cancel_or_partial_read_cannot_qualify_any_role(self):
        for index in range(3):
            for kind in ("cancelled", "timeout", "read", "connect", "disconnect"):
                with self.subTest(index=index, kind=kind):
                    bundle = fixture()
                    # Even complete-looking bytes cannot override transport failure.
                    bundle["attempts"][index]["transport_error"] = {"kind": kind, "message": "cut short"}
                    self.refuses(bundle)

    def test_malformed_error_or_duplicate_json_keys_cannot_be_refusal(self):
        bodies = [b"", b"{", b'{"error":"full"}', b'{"error":{"code":"shed_queue"}}',
                  b'{"error":{"message":"full","type":"rate_limit_error","param":null,'
                  b'"code":"context_length_exceeded","code":"shed_queue"}}']
        for value in ("", None, [], "wrong"):
            error = json.loads(refusal())
            error["error"]["message" if value != "wrong" else "param"] = value
            bodies.append(encode(error))
        for raw in bodies:
            with self.subTest(raw=raw):
                bundle = fixture(); bundle["attempts"][1]["body"] = raw
                self.refuses(bundle)

    def test_header_names_case_insensitive_and_raw_tuple_pairs_are_supported(self):
        bundle = fixture()
        bundle["attempts"][1]["headers"] = [("rEtRy-AfTeR", "2"), ("RETRY-AFTER-MS", "2000"),
                                             ("Content-Type", "application/json")]
        self.assertEqual(evaluate_overload_recovery_cell(**bundle)["refused"][0]["retry_after_s"], 2)

    def test_missing_duplicate_mismatched_or_contradictory_retry_headers_refuse(self):
        valid = [["Retry-After", "2"], ["retry-after-ms", "2000"]]
        for headers in (None, {}, [], valid[:1], valid[1:], valid + [valid[0]],
                        valid + [["RETRY-AFTER", "2"]], valid + [["Retry-After-Ms", "2000"]],
                        valid + [["x-should-retry", "false"]], valid + [["x-should-retry", "true"]],
                        [["Retry-After", "2"], ["retry-after-ms", "2001"]], [["Retry-After"]]):
            with self.subTest(headers=headers):
                bundle = fixture(); bundle["attempts"][1]["headers"] = headers
                self.refuses(bundle)

    def test_source_retry_seconds_must_be_canonical_and_bounded(self):
        for seconds in ("0", "61", "02", "2.0", "-2", "+2", " 2", "2 ", "٢", "2, 2", "", "9" * 10000,
                        "Tue, 21 Sep 2026 00:00:00 GMT", 2, True):
            bundle = fixture()
            bundle["attempts"][1]["headers"][0][1] = seconds
            with self.subTest(seconds=str(seconds)[:30]): self.refuses(bundle)
        for milliseconds in ("02000", "2000.0", "2e3", "2000 ", 2000):
            bundle = fixture(); bundle["attempts"][1]["headers"][1][1] = milliseconds
            self.refuses(bundle)

    def test_all_roles_need_full_planned_denominator_and_nonempty_unique_ids(self):
        for index in range(3):
            bundle = fixture(); bundle["attempts"].pop(index)
            self.refuses(bundle)
            bundle = fixture(); bundle["attempts"].append(copy.deepcopy(bundle["attempts"][index]))
            self.refuses(bundle)
            for value in (None, "", " ", [], "other"):
                bundle = fixture(); bundle["attempts"][index]["id"] = value
                self.refuses(bundle)
            bundle = fixture()
            bundle["program"]["requests"][index]["id"] = ("refused" if index == 0 else "peer")
            self.refuses(bundle)
        for role in ("peer", "refused", "recovery"):
            bundle = fixture()
            bundle["program"]["requests"] = [r for r in bundle["program"]["requests"] if r["role"] != role]
            bundle["attempts"] = [r for r in bundle["attempts"] if r["id"] != role]
            self.refuses(bundle)
        bundle = fixture(); bundle["program"]["requests"] = []; bundle["attempts"] = []
        self.refuses(bundle)

    def test_total_request_count_does_not_replace_required_roles(self):
        for missing in ("peer", "refused", "recovery"):
            bundle = fixture()
            replacement = "peer" if missing != "peer" else "recovery"
            add_request(bundle, replacement, "-2", 105 if replacement == "peer" else 810,
                        450 if replacement == "peer" else 900)
            bundle["program"]["requests"] = [r for r in bundle["program"]["requests"] if r["role"] != missing]
            bundle["attempts"] = [r for r in bundle["attempts"] if r["id"] != missing]
            self.assertEqual(len(bundle["attempts"]), 3)
            self.refuses(bundle, "denominators must all be nonempty")

    def test_clean_wire_counts_alone_do_not_prove_peer_overlap(self):
        for start, end in ((1, 50), (220, 400), (100, 200), (200, 400), (100, 220), (210, 400)):
            bundle = fixture(); bundle["attempts"][0].update(started_ns=start, finished_ns=end)
            # Use valid brackets for the before-refusal case as well.
            bundle["health_samples"][0].update(started_ns=0, finished_ns=0)
            accounting = account_attempts(bundle["program"]["requests"], bundle["attempts"])
            self.assertEqual(accounting["counts"], {"clean_success": 2, "refused": 1})
            with self.subTest(interval=(start, end)): self.refuses(bundle, "spanned")

    def test_one_good_refusal_cannot_hide_an_unrelated_late_refusal(self):
        bundle = fixture()
        add_request(bundle, "refused", "-late", 460, 480)
        self.refuses(bundle, "spanned")

    def test_recovery_is_after_every_pressure_attempt_not_just_first_refusal(self):
        for start in (50, 219, 220, 399, 400):
            bundle = fixture(); bundle["attempts"][2]["started_ns"] = start
            self.refuses(bundle, "after all pressure")
        bundle = fixture()
        add_request(bundle, "peer", "-slow", 101, 700)
        self.refuses(bundle, "after all pressure")

    def test_every_peer_and_recovery_requires_terminal_usage_nonempty_output(self):
        for index in (0, 2):
            for wire in ("chat_json", "chat_sse", "native_json", "native_sse"):
                for raw in (b"", b"{", success(wire)[:-1], success(wire, text="")):
                    with self.subTest(role=index, wire=wire, raw=raw[:20]):
                        bundle = fixture(wire); bundle["attempts"][index]["body"] = raw
                        self.refuses(bundle)
            bundle = fixture()
            value = json.loads(bundle["attempts"][index]["body"]); del value["usage"]
            bundle["attempts"][index]["body"] = encode(value)
            self.refuses(bundle)

    def test_sse_missing_terminal_or_usage_and_error_under_http200_refuse(self):
        bodies = [b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"42"}}]}\n\n',
                  b'data: {"error":{"code":"shed_queue","message":"full"}}\n\ndata: [DONE]\n\n',
                  b'event: error\ndata: {"error":"out of memory"}\n\n']
        for wire in ("chat_sse", "native_sse"):
            for index in range(3):
                for raw in bodies:
                    bundle = fixture(wire); bundle["attempts"][index].update(status=200, body=raw)
                    self.refuses(bundle)

    def test_usage_must_fit_explicit_frozen_prompt_and_output_ranges(self):
        for index in (0, 2):
            for prompt, count in ((0, 1), (33, 1), (8, 0), (8, 17)):
                bundle = fixture(); bundle["attempts"][index]["body"] = success("chat_json", prompt, count)
                self.refuses(bundle)
            for field in ("prompt_tokens", "completion_tokens"):
                for value in (None, {"min": 0, "max": 32}, {"min": True, "max": 32},
                              {"min": 9, "max": 8}, {"min": 1}):
                    bundle = fixture(); bundle["program"]["requests"][index][field] = value
                    self.refuses(bundle)

    def test_health_brackets_cover_refusals_too_not_only_clean_survivors(self):
        bundle = fixture()
        # Overlap is already invalid here, but the full denominator must fail at
        # generation bracketing FIRST, not silently omit the out-of-window refusal.
        bundle["attempts"][1].update(started_ns=1100, finished_ns=1200)
        self.refuses(bundle, "bracket every request")
        for index in (0, -1):
            bundle = fixture(); bundle["health_samples"].pop(index)
            self.refuses(bundle, "bracket every request")

    def test_generation_change_between_phases_fails_even_if_each_request_is_unaffected(self):
        bundle = fixture()
        bundle["health_samples"] = [health("before", 1), health("pressure-end", 450),
                                     health("recovery-start", 550, 8), health("after", 1000, 8)]
        raw = account_attempts(bundle["program"]["requests"], bundle["attempts"])
        census = account_generations(raw, bundle["attempts"], bundle["health_samples"])
        self.assertEqual(census["generation_affected_requests"], 0)
        self.assertEqual(census["respawns_observed"], 1)
        self.refuses(bundle, "generation changed")

    def test_health_worker_or_owner_changes_refuse(self):
        for field, value in (("status", 503), ("status", True), ("path", "/readyz"),
                             ("method", "POST"), ("transport_error", {"kind": "read", "message": "partial"}),
                             ("body", b'{"status":"unhealthy","worker":{"generation":7}}'),
                             ("body", b'{"status":"ok","worker":{"generation":8}}'),
                             ("body", b'{"status":"ok","worker":{"generation":true}}'),
                             ("body", b"{"), ("body", b'{"status":"ok"}')):
            bundle = fixture(); bundle["health_samples"][-1][field] = value
            with self.subTest(field=field, value=value): self.refuses(bundle)
        for index in range(3):
            for owner in ({"pid": 18, "start_identity": "boot-id:71"},
                          {"pid": 17, "start_identity": "boot-id:72"}, None):
                bundle = fixture(); bundle["health_samples"][index]["server_identity"] = owner
                self.refuses(bundle)

    def test_health_ids_cannot_duplicate_or_alias_any_request_including_refusal(self):
        for value in ("before", "peer", "refused", "recovery", "", [], None):
            bundle = fixture(); bundle["health_samples"][-1]["id"] = value
            self.refuses(bundle)
        bundle = fixture(); bundle["health_samples"].reverse()
        self.refuses(bundle)

    def test_owner_and_route_binding_include_refused_requests(self):
        for index in range(3):
            for key, value in (("server_identity", {"pid": 18, "start_identity": "boot-id:71"}),
                               ("server_identity", {"pid": 17, "start_identity": "other-boot:71"}),
                               ("server_identity", None), ("method", "GET"), ("path", "/v1/other")):
                bundle = fixture(); bundle["attempts"][index][key] = value
                self.refuses(bundle)

    def test_unknown_skipped_or_weakened_requirements_do_not_pass(self):
        for scenario in ("drain", "worker_failure_recovery", "skip", "", None, []):
            bundle = fixture(); bundle["required"]["scenario"] = scenario
            self.refuses(bundle)
        for key in fixture()["required"]["requirements"]:
            for value in (False, 1, None, "true"):
                bundle = fixture(); bundle["required"]["requirements"][key] = value
                self.refuses(bundle)
            bundle = fixture(); del bundle["required"]["requirements"][key]
            self.refuses(bundle)
        for location in ("required", "program"):
            bundle = fixture(); bundle[location]["passed"] = True
            self.refuses(bundle)

    def test_frozen_program_cannot_select_another_scope_mode_or_request_shape(self):
        for field, value in (("cell_id", "other/overload_recovery"), ("mode", "serial"),
                             ("scope", {**SCOPE, "route": "different"}), ("server_identity", {})):
            bundle = fixture(); bundle["program"][field] = value
            self.refuses(bundle)
        for index in range(3):
            for field, value in (("role", "skip"), ("role", []), ("id", " "), ("id", []),
                                 ("wire", "unknown"), ("wire", []), ("path", "/other"),
                                 ("model", "other"), ("payload", {})):
                bundle = fixture(); bundle["program"]["requests"][index][field] = value
                self.refuses(bundle)
            for field, value in (("stream", "false"), ("stream", True), ("model", "other")):
                bundle = fixture(); bundle["program"]["requests"][index]["payload"][field] = value
                self.refuses(bundle)
        bundle = fixture(); bundle["program"]["requests"] *= 86
        self.refuses(bundle, "3..256")

    def test_timestamp_corruption_does_not_prove_overlap_or_recovery(self):
        for index in range(3):
            for start, end in ((0, -1), (True, 400), (100, 100), (800, 799), ("100", 400)):
                bundle = fixture(); bundle["attempts"][index].update(started_ns=start, finished_ns=end)
                self.refuses(bundle)

    def test_summary_pass_flags_cannot_cover_a_missing_terminal(self):
        bundle = fixture("chat_sse")
        bundle["attempts"][2].update(body=b"", passed=True, complete=True, outcome="clean_success")
        self.refuses(bundle)


if __name__ == "__main__":
    unittest.main()
