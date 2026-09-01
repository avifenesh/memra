#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("replay.py")
SPEC = importlib.util.spec_from_file_location("cachespec_replay", MODULE_PATH)
REPLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPLAY)


class ReplayTests(unittest.TestCase):
    def test_render_keeps_prior_assistant_plain_and_opens_live_reasoning(self):
        prompt = REPLAY.render(
            [("user", "change it"), ("assistant", "<html>ok</html>")], "system"
        )
        self.assertIn("<|im_start|>assistant\n<html>ok</html><|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

    def test_strip_reasoning_matches_step_generation_prompt_shape(self):
        kept, rewritten = REPLAY.strip_reasoning(
            "private chain\n</think>\n<html>answer</html>"
        )
        self.assertTrue(rewritten)
        self.assertEqual(kept, "<html>answer</html>")

    def test_metrics_receipt_separates_counters_from_gauges(self):
        before = {"completed": 7, "prefix_cache_evictions": 2, "spec_pool_entries": 1}
        after = {"completed": 8, "prefix_cache_evictions": 5, "spec_pool_entries": 2}
        receipt = REPLAY.metrics_receipt(before, after)
        self.assertEqual(receipt["delta"]["completed"], 1)
        self.assertEqual(receipt["delta"]["prefix_cache_evictions"], 3)
        self.assertEqual(receipt["after"]["spec_pool_entries"], 2)

    def test_hardening_metrics_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "server lacks cachespec metrics"):
            REPLAY.validate_metrics({})
        REPLAY.validate_metrics({name: 0 for name in REPLAY.REQUIRE_HARDENING_METRICS})


if __name__ == "__main__":
    unittest.main()
