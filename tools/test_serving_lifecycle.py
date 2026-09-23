"""Worker-generation evidence controls. No live server or GPU required."""

import copy
import json
import unittest

from serving_lifecycle import account_generations
from serving_release import ServingGateError, account_attempts


class GenerationAccountingTests(unittest.TestCase):
    def fixture(self):
        response = json.dumps({"model": "gate", "choices": [{"index": 0,
            "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}).encode()
        attempts = [dict(id=i, started_ns=start, finished_ns=end, status=200, body=response)
                    for i, start, end in (("old", 10, 20), ("crossing", 40, 60), ("new", 80, 90))]
        schedule = [dict(id=a["id"], model="gate", wire="chat_json") for a in attempts]
        accounting = account_attempts(schedule, attempts)
        samples = [dict(started_ns=start, finished_ns=start + 1, status=200,
                        server_identity={"pid": 100, "start_identity": "boot:1234"},
                        body=json.dumps({"worker": {"generation": generation}}).encode())
                   for start, generation in ((0, 0), (30, 0), (70, 1), (100, 1))]
        return accounting, attempts, samples

    def test_one_respawn_does_not_erase_wire_success_or_enter_clean_latency(self):
        accounting, attempts, samples = self.fixture()
        result = account_generations(accounting, attempts, samples)
        self.assertEqual(accounting["counts"], {"clean_success": 3})
        self.assertEqual(result["respawns_observed"], 1)
        self.assertEqual(result["generation_affected_requests"], 1)
        self.assertEqual(result["clean_latency_ns"], [10, 10])
        crossing = result["requests"][1]
        self.assertEqual(crossing["wire_outcome"], "clean_success")
        self.assertIsNone(crossing["clean_latency_ns"])

    def test_failures_remain_counted_with_or_without_a_generation_change(self):
        accounting, attempts, samples = self.fixture()
        for result in accounting["requests"]:
            result["outcome"] = "truncated_200"
        result = account_generations(accounting, attempts, samples)
        self.assertEqual(len(result["requests"]), 3)
        self.assertEqual(result["clean_latency_ns"], [])
        self.assertEqual(result["generation_affected_requests"], 1)

    def test_missing_or_overlapping_boundary_sample_refuses(self):
        accounting, attempts, samples = self.fixture()
        for wrong in (samples[1:], samples[:-1]):
            with self.assertRaises(ServingGateError):
                account_generations(accounting, attempts, wrong)
        # A health call that only *began* before the request is not a before sample.
        samples[0]["finished_ns"] = 11
        with self.assertRaises(ServingGateError):
            account_generations(accounting, attempts, samples)

    def test_process_replacement_counter_reset_and_malformed_health_refuse(self):
        accounting, attempts, samples = self.fixture()
        mutations = [dict(server_identity={"pid": 100, "start_identity": "boot:9999"}),
                     dict(body=b'{"worker":{"generation":-1}}'),
                     dict(body=b'{"worker":{"generation":true}}'),
                     dict(body=b'{"worker":{"generation":0.5}}'),
                     dict(body=b'{"status":"ok"}'), dict(status=500)]
        for mutation in mutations:
            wrong = copy.deepcopy(samples); wrong[-1].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ServingGateError):
                account_generations(accounting, attempts, wrong)

    def test_omitted_duplicate_reordered_and_stale_timing_refuse(self):
        accounting, attempts, samples = self.fixture()
        for wrong in (attempts[:1], attempts + attempts[:1]):
            with self.assertRaises(ServingGateError):
                account_generations(accounting, wrong, samples)
        with self.assertRaises(ServingGateError):
            account_generations(accounting, attempts, list(reversed(samples)))
        attempts[1]["finished_ns"] += 1
        with self.assertRaises(ServingGateError):
            account_generations(accounting, attempts, samples)


if __name__ == "__main__":
    unittest.main()
