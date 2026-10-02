"""Collector controls only; native model evidence comes from the recorded GPU cell."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("sampled", Path(__file__).with_name("collect-sampled-mtp.py"))
sampled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampled)


class CollectorControls(unittest.TestCase):
    def test_distribution_null_and_old_bias_red(self):
        a = sampled.np.array([0] * 410 + [1] * 102).reshape(-1, 1)
        b = sampled.np.array([0] * 492 + [1] * 20).reshape(-1, 1)
        self.assertEqual(sampled.permutation_rows(a, a)[0]["p_value"], 1)
        self.assertLessEqual(sampled.permutation_rows(a, b)[0]["p_value"], 0.00125)

    def test_native_accounting_and_early_stop(self):
        row = {"model": "gate", "tokens": [1, 2], "n_tokens": 2, "prompt_tokens": 27,
               "cached_tokens": 0, "stop_reason": "Eos"}
        self.assertEqual(sampled.native_tokens(row), [1, 2, -1, -1, -1, -1, -1, -1])
        for key, value in (("n_tokens", 3), ("cached_tokens", 1), ("error", "failed"), ("stop_reason", "Deadline")):
            with self.assertRaises(ValueError):
                sampled.native_tokens({**row, key: value})
        with self.assertRaises(ValueError):
            sampled.native_tokens({**row, "tokens": [1], "n_tokens": True})

    def test_matched_context_cutoff_and_diverged_prefix_refusal(self):
        control = [dict(round=0, pos=10, out_len=1, last=4, draft=[5, 6, 7], accepted=1, bonus=9),
                   dict(round=1, pos=12, out_len=3, last=9, draft=[2, 3, 4], accepted=1, bonus=8)]
        shortened = copy.deepcopy(control)
        shortened[0]["draft"] = [5]
        self.assertIsNotNone(sampled.first_matched_cutoff(control, shortened))
        coincidental = copy.deepcopy(control)
        coincidental[0]["draft"] = [6, 7, 8]
        coincidental[1]["draft"] = [2]
        self.assertIsNone(sampled.first_matched_cutoff(control, coincidental))

    def test_repeated_probe_walk_is_counted_once(self):
        line = "[R0] pos=10 out_len=1 last_tok=4 draft=[5, 6] n_acc=1 bonus=9 t_pred0=0\n"
        self.assertEqual(len(sampled.probe_rounds(line + line)), 1)

    def test_native_dialect_is_explicit_when_auth_is_enabled(self):
        args = SimpleNamespace(port=18121, model="fixture.gguf", server="unused", external_lock=9,
                               private_cache="unused-cache")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sampled.socket, "socket"), \
                mock.patch.object(sampled.subprocess, "Popen") as popen, \
                mock.patch.object(sampled.threading, "Thread"):
            sampled.ExperimentalServer(args, Path(tmp), "positive_graph")
            env = popen.call_args.kwargs["env"]
            self.assertTrue(env["MEMRA_API_KEY"])
            self.assertEqual(env["MEMRA_COMPAT"], "native")

    def test_vendor_profile_rejects_generic_fallback_and_startup_only_trace(self):
        profile = dict(default_temperature=1.0, default_top_k=20, default_top_p=0.95,
                       default_min_p=0.0, default_presence_penalty=1.5)
        good = "[skey] burst sampled=1 temp=1 top_k=20 top_p=0.95 min_p=0 pen_on=1 k=3\n"
        fallback = "[skey] burst sampled=1 temp=1 top_k=0 top_p=1 min_p=0 pen_on=0 k=3\n"
        ready = "[server] listening on http://127.0.0.1:18121\n"
        self.assertEqual(sampled.vendor_trace(fallback + ready + good, profile)["burst_count"], 1)
        for bad in (good, good + ready, ready + fallback):
            with self.assertRaises(ValueError):
                sampled.vendor_trace(bad, profile)

    def test_chat_mechanics_preserve_reasoning_without_claiming_a_final_answer(self):
        document = {"id": "chat-1", "model": "gate", "choices": [
            {"message": {"reasoning": "thinking", "content": ""}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192,
                      "prompt_tokens_details": {"cached_tokens": 0}}}
        result = sampled.chat_completion(document)
        self.assertEqual((result["reasoning_chars"], result["content_chars"]), (8, 0))
        self.assertEqual(result["finish"], "length")
        with self.assertRaises(ValueError):
            sampled.chat_completion({**document, "choices": document["choices"] * 2})


if __name__ == "__main__":
    unittest.main()
