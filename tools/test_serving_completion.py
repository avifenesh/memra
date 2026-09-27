"""Completion requirements tested with raw protocol fixtures, without server/GPU I/O."""

import copy
import json
from pathlib import Path
import unittest

from serving_completion import evaluate_completion_cell
from serving_manifest import required_cells
from serving_release import ServingGateError


OWNER = {"pid": 17, "start_identity": "boot:71"}
SCOPE = {"id": "tiny", "model": "gate", "route": "normal", "profile": "text-generation-v1"}


def encode(value):
    return json.dumps(value).encode()


def body(wire, prompt=8, answer="42"):
    usage = {"prompt_tokens": prompt, "completion_tokens": 1, "total_tokens": prompt + 1}
    if wire == "chat_json":
        return encode({"model": "gate", "choices": [{"index": 0, "finish_reason": "stop",
                       "message": {"role": "assistant", "content": answer}}], "usage": usage})
    if wire == "chat_sse":
        return b"data: " + encode({"model": "gate", "choices": [{"index": 0,
            "finish_reason": "stop", "delta": {"content": answer}}], "usage": usage}) + b"\n\ndata: [DONE]\n\n"
    terminal = {"stop_reason": "MaxNew", "n_tokens": 1, "prompt_tokens": prompt,
                "cached_tokens": 0, "elapsed_s": .01}
    if wire == "native_json":
        return encode({**terminal, "model": "gate", "text": answer, "tokens": [4]})
    return (b"event: message\ndata: " + encode({"model": "gate", "id": 4, "text": answer})
            + b"\n\nevent: done\ndata: " + encode(terminal) + b"\n\n")


def fixture(scenario):
    required = next(c for c in required_cells(
        Path(__file__).with_name("serving-release.cells.json").read_bytes(), [SCOPE])
                    if c["scenario"] == scenario)
    wires = (["chat_json", "chat_sse", "native_json", "native_sse"]
             if scenario == "wire_completion" else ["chat_json"] *
             (3 if scenario == "concurrent_completion" else 1))
    program = {"cell_id": required["id"], "scope": dict(SCOPE), "server_identity": dict(OWNER),
               "mode": "concurrent" if len(wires) > 1 else "serial", "requests": []}
    attempts = []
    for i, wire in enumerate(wires):
        path = "/v1/chat/completions" if wire.startswith("chat_") else "/v1/completions"
        prompt = 512 if scenario == "long_prompt" else 8
        request = {"id": f"r{i}", "model": "gate", "wire": wire, "path": path,
                   "payload": {"model": "gate", "stream": wire.endswith("_sse"), "temperature": 0},
                   "prompt_tokens": {"min": 1, "max": 4096},
                   "completion_tokens": {"min": 1, "max": 32}}
        if scenario == "short_prompt":
            request["answer_oracle"] = "42"
        program["requests"].append(request)
        attempts.append({"id": request["id"], "server_identity": dict(OWNER),
                         "method": "POST", "path": path, "started_ns": 100 + i,
                         "finished_ns": 200 + i, "status": 200, "body": body(wire, prompt)})
    health = [{"id": name, "server_identity": dict(OWNER), "path": "/health", "status": 200,
               "started_ns": start, "finished_ns": start + 1,
               "body": encode({"status": "ok", "worker": {"generation": 2}})}
              for name, start in [("before", 1), ("after", 1000)]]
    return {"required": required, "program": program, "attempts": attempts, "health_samples": health}


class CompletionCellsTests(unittest.TestCase):
    def refuses(self, bundle):
        with self.assertRaises(ServingGateError):
            evaluate_completion_cell(**bundle)

    def test_each_supported_required_scenario_uses_raw_evidence_without_mutation(self):
        for scenario in ("short_prompt", "long_prompt", "wire_completion", "concurrent_completion"):
            bundle = fixture(scenario)
            before = copy.deepcopy(bundle)
            result = evaluate_completion_cell(**bundle)
            self.assertEqual(bundle, before)
            self.assertEqual(result["scenario"], scenario)
            self.assertEqual(result["planned"], len(bundle["attempts"]))
            self.assertNotIn("qualification", result)
            if scenario == "concurrent_completion":
                self.assertEqual(result["peak_client_intervals"], 3)

    def test_short_oracle_is_exact_and_prompt_bounds_are_manifest_owned(self):
        for prompt, answer in [(0, "42"), (33, "42"), (8, "42\n"), (8, " 42"), (8, "43")]:
            bundle = fixture("short_prompt")
            bundle["attempts"][0]["body"] = body("chat_json", prompt, answer)
            self.refuses(bundle)
        for temperature in (True, 1, None, "0"):
            bundle = fixture("short_prompt")
            bundle["program"]["requests"][0]["payload"]["temperature"] = temperature
            self.refuses(bundle)
        bundle = fixture("short_prompt")
        del bundle["program"]["requests"][0]["answer_oracle"]
        self.refuses(bundle)

    def test_long_prompt_requires_actual_usage_not_a_large_config_limit(self):
        bundle = fixture("long_prompt")
        bundle["attempts"][0]["body"] = body("chat_json", 511)
        self.refuses(bundle)
        bundle["attempts"][0]["body"] = body("chat_json", 512, "")
        self.refuses(bundle)

    def test_all_four_complete_wires_required_no_missing_terminal_or_usage(self):
        for index in range(4):
            bundle = fixture("wire_completion")
            bundle["attempts"].pop(index)
            bundle["program"]["requests"].pop(index)
            self.refuses(bundle)
        for index in (1, 3):
            bundle = fixture("wire_completion")
            bundle["attempts"][index]["body"] = bundle["attempts"][index]["body"][:-1]
            self.refuses(bundle)
        bundle = fixture("wire_completion")
        value = json.loads(bundle["attempts"][0]["body"])
        del value["usage"]
        bundle["attempts"][0]["body"] = encode(value)
        self.refuses(bundle)

    def test_concurrency_needs_three_simultaneously_overlapping_intervals(self):
        for intervals in ([(100, 150), (150, 200), (200, 250)],
                          [(100, 200), (150, 250), (200, 300)],
                          [(100, 200), (101, 201), (102, 102)]):
            bundle = fixture("concurrent_completion")
            for row, (start, end) in zip(bundle["attempts"], intervals):
                row.update(started_ns=start, finished_ns=end)
            self.refuses(bundle)
        bundle = fixture("concurrent_completion")
        bundle["program"]["mode"] = "serial"
        self.refuses(bundle)
        bundle = fixture("concurrent_completion")
        bundle["attempts"].pop()
        bundle["program"]["requests"].pop()
        self.refuses(bundle)

    def test_all_attempts_and_generation_health_are_required(self):
        for change in ("missing", "duplicate", "truncated", "foreign", "generation", "partial_health"):
            bundle = fixture("concurrent_completion")
            if change == "missing": bundle["attempts"].pop()
            elif change == "duplicate": bundle["attempts"].append(copy.deepcopy(bundle["attempts"][0]))
            elif change == "truncated": bundle["attempts"][0]["body"] = b"{"
            elif change == "foreign": bundle["attempts"][0]["server_identity"]["pid"] = 18
            elif change == "generation":
                bundle["health_samples"][-1]["body"] = encode({"worker": {"generation": 3}})
            else: bundle["health_samples"][-1]["transport_error"] = "cut short"
            with self.subTest(change=change): self.refuses(bundle)

    def test_policy_scope_routes_and_unknown_scenarios_cannot_be_substituted(self):
        for location, key, value in [
            ("requirements", "prompt_tokens_max", 1000),
            ("requirements", "prompt_tokens_min", True),
            ("program", "passed", True), ("scope", "model", "other"),
            ("request", "model", "other"), ("request", "path", "/different"),
            ("attempt", "path", "/different"), ("attempt", "method", "GET"),
            ("payload", "stream", True), ("payload", "model", "other")]:
            bundle = fixture("short_prompt")
            target = {"requirements": bundle["required"]["requirements"], "program": bundle["program"],
                      "scope": bundle["program"]["scope"], "request": bundle["program"]["requests"][0],
                      "attempt": bundle["attempts"][0], "payload": bundle["program"]["requests"][0]["payload"]}[location]
            target[key] = value
            with self.subTest(location=location, key=key): self.refuses(bundle)
        for scenario in ("cache_restore", "cancel_decode", "drain", "worker_failure_recovery", []):
            bundle = fixture("short_prompt"); bundle["required"]["scenario"] = scenario
            self.refuses(bundle)


if __name__ == "__main__":
    unittest.main()
