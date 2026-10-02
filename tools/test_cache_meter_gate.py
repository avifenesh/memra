"""Usage contracts for the cache meter's OpenAI and native response shapes."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cache_meter_gate', Path(__file__).with_name('cache-meter-gate.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ResponseUsage(unittest.TestCase):
    def test_native_token_count_is_available_to_the_gap_check(self):
        got = gate.response_usage({'prompt_tokens': 272, 'cached_tokens': 256, 'n_tokens': 7})
        self.assertEqual(got, {'prompt_tokens': 272, 'completion_tokens': 7,
                              'prompt_tokens_details': {'cached_tokens': 256}})

    def test_openai_usage_is_preserved_including_extensions(self):
        usage = {'prompt_tokens': 272, 'completion_tokens': 7,
                 'prompt_tokens_details': {'cached_tokens': 256}, 'spec': {'drafted': 3}}
        self.assertIs(gate.response_usage({'usage': usage}), usage)

    def test_missing_native_count_is_not_fabricated(self):
        with self.assertRaises(KeyError):
            gate.response_usage({'prompt_tokens': 272, 'cached_tokens': 256})


if __name__ == '__main__':
    unittest.main()
