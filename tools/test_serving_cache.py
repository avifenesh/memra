"""Raw native twins and metrics arithmetic, including misleading cache-hit receipts."""

import copy
import json
from pathlib import Path
import unittest

from serving_cache import evaluate_cache_cell
from serving_manifest import required_cells
from serving_release import ServingGateError
from test_serving_completion import OWNER, SCOPE, body, encode


def fixture():
    required = next(c for c in required_cells(
        Path(__file__).with_name("serving-release.cells.json").read_bytes(), [SCOPE])
                    if c["scenario"] == "cache_restore")
    program = {"cell_id": required["id"], "scope": dict(SCOPE), "server_identity": dict(OWNER),
               "mode": "serial", "requests": []}
    attempts = []
    prefix = list(range(2000, 2064))
    for i, (role, ids, cached, salt) in enumerate([
        ("cold", prefix + [3000] * 16, 0, "control"),
        ("seed", prefix, 0, "restored"),
        ("restore", prefix + [3000] * 16, 64, "restored")]):
        program["requests"].append({"id": role, "role": role, "model": "gate", "wire": "native_json",
            "path": "/v1/completions", "payload": {"model": "gate", "stream": False,
                "prompt_ids": ids, "cache_salt": salt, "temperature": 0, "max_tokens": 1},
            "prompt_tokens": {"min": 1, "max": 4096}, "completion_tokens": {"min": 1, "max": 8}})
        value = json.loads(body("native_json", len(ids)))
        value["cached_tokens"] = cached
        attempts.append({"id": role, "server_identity": dict(OWNER), "method": "POST",
            "path": "/v1/completions", "status": 200, "started_ns": 100 + i * 200,
            "finished_ns": 200 + i * 200, "body": encode(value)})
    health = [{"id": name, "server_identity": dict(OWNER), "path": "/health", "status": 200,
               "started_ns": start, "finished_ns": start + 1,
               "body": encode({"status": "ok", "worker": {"generation": 2}})}
              for name, start in [("before", 1), ("after", 1000)]]
    delta = {"prompt_tokens_in": 224, "cached_tokens_in": 64, "computed_tokens_in": 160,
             "prefix_cache_hit_tokens": 64, "prefix_cache_hits": 1, "prefix_cache_misses": 2}
    metrics = [{"id": name, "server_identity": dict(OWNER), "method": "GET", "path": "/metrics",
                "status": 200, "started_ns": start, "finished_ns": start + 1,
                "body": encode({k: 71 + increment * v for k, v in delta.items()})}
               for name, start, increment in [("metrics-before", 20, 0), ("metrics-after", 900, 1)]]
    return dict(required=required, program=program, attempts=attempts,
                health_samples=health, metrics_samples=metrics)


def alter(bundle, index, field, value):
    raw = json.loads(bundle["attempts"][index]["body"])
    raw[field] = value
    bundle["attempts"][index]["body"] = encode(raw)


class CacheCellsTests(unittest.TestCase):
    def refuses(self, bundle):
        with self.assertRaises(ServingGateError):
            evaluate_cache_cell(**bundle)

    def test_actual_native_twin_and_complete_counter_delta(self):
        bundle = fixture(); original = copy.deepcopy(bundle)
        result = evaluate_cache_cell(**bundle)
        self.assertEqual(bundle, original)
        self.assertEqual(result["restored_prefix_tokens"], 64)
        self.assertEqual(result["metrics_delta"]["computed_tokens_in"], 160)
        self.assertEqual(len(result["completed"]), 3)

    def test_cache_hit_cannot_replace_token_text_stop_or_usage_parity(self):
        for field, value in [("tokens", [5]), ("text", "43"), ("stop_reason", "Eos"),
                             ("n_tokens", 2), ("prompt_tokens", 79)]:
            bundle = fixture(); alter(bundle, 2, field, value)
            with self.subTest(field=field): self.refuses(bundle)

    def test_cold_and_seed_are_cold_and_restored_hit_covers_whole_prefix(self):
        for index, cached in [(0, 1), (1, 1), (2, 0), (2, 63)]:
            bundle = fixture(); alter(bundle, index, "cached_tokens", cached)
            self.refuses(bundle)
        bundle = fixture(); value = json.loads(bundle["attempts"][2]["body"])
        del value["cached_tokens"]; bundle["attempts"][2]["body"] = encode(value)
        self.refuses(bundle)

    def test_same_program_and_distinct_namespaces_are_mandatory(self):
        for role, field, value in [(0, "cache_salt", "restored"), (2, "cache_salt", "other"),
                                  (2, "max_tokens", 2), (2, "temperature", .1),
                                  (0, "temperature", True), (2, "prompt", "ambiguous"),
                                  (2, "prompt_ids", list(range(2001, 2081)))]:
            bundle = fixture(); bundle["program"]["requests"][role]["payload"][field] = value
            self.refuses(bundle)
        bundle = fixture(); bundle["program"]["requests"][1]["payload"]["prompt_ids"] += [3000] * 16
        self.refuses(bundle)

    def test_reordering_overlap_foreign_and_missing_requests_refuse(self):
        for change in ("missing", "duplicate", "overlap", "reorder", "foreign", "route", "generation"):
            bundle = fixture()
            if change == "missing": bundle["attempts"].pop()
            elif change == "duplicate": bundle["attempts"].append(copy.deepcopy(bundle["attempts"][0]))
            elif change == "overlap": bundle["attempts"][1]["started_ns"] = 199
            elif change == "reorder": bundle["program"]["requests"].reverse()
            elif change == "foreign": bundle["attempts"][0]["server_identity"]["pid"] = 18
            elif change == "route": bundle["attempts"][0]["method"] = "GET"
            else: bundle["health_samples"][-1]["body"] = encode({"worker": {"generation": 3}})
            with self.subTest(change=change): self.refuses(bundle)

    def test_metrics_require_exact_all_request_arithmetic_and_same_owner_brackets(self):
        for field in json.loads(fixture()["metrics_samples"][-1]["body"]):
            for value in (None, True, -1, 70):
                bundle = fixture(); raw = json.loads(bundle["metrics_samples"][-1]["body"])
                raw[field] = value; bundle["metrics_samples"][-1]["body"] = encode(raw)
                with self.subTest(field=field, value=value): self.refuses(bundle)
        for change in ({"status": 503}, {"transport_error": "timeout"}, {"body": b"{"},
                       {"started_ns": 599}, {"method": "POST"}, {"path": "/health"},
                       {"server_identity": {"pid": 18, "start_identity": OWNER["start_identity"]}}):
            bundle = fixture(); bundle["metrics_samples"][-1].update(change); self.refuses(bundle)
        bundle = fixture(); bundle["metrics_samples"][0]["finished_ns"] = 101
        self.refuses(bundle)

    def test_continuation_hit_does_not_fabricate_prefix_probe_counters(self):
        bundle = fixture()
        # worker.rs increments n_cached_in from every session's n_cached, while
        # continuation reuse increments its own counter and skips prefix probing.
        before = json.loads(bundle["metrics_samples"][0]["body"])
        after = json.loads(bundle["metrics_samples"][1]["body"])
        for field in ("prefix_cache_hits", "prefix_cache_hit_tokens"):
            after[field] = before[field]
        bundle["metrics_samples"][1]["body"] = encode(after)
        result = evaluate_cache_cell(**bundle)
        self.assertEqual(result["metrics_delta"]["cached_tokens_in"], 64)
        self.assertEqual(result["prefix_diagnostics_delta"]["prefix_cache_hits"], 0)
        self.assertEqual(result["prefix_diagnostics_delta"]["prefix_cache_hit_tokens"], 0)
        self.assertIsNone(result["cache_tier_attribution"])
        # Global accounting remains exact even though tier attribution is unknown.
        after["computed_tokens_in"] += 1
        bundle["metrics_samples"][1]["body"] = encode(after)
        self.refuses(bundle)

    def test_unknown_policy_scope_or_capture_as_oracle_refuses(self):
        bundle = fixture(); bundle["required"]["requirements"]["cold_cached_tokens"] = False
        self.refuses(bundle)
        bundle = fixture(); bundle["program"]["scope"]["route"] = "other"
        self.refuses(bundle)
        bundle = fixture(); bundle["program"]["requests"][0]["passed"] = True
        self.refuses(bundle)
        bundle = fixture(); bundle["program"]["mode"] = "concurrent"
        self.refuses(bundle)


if __name__ == "__main__":
    unittest.main()
