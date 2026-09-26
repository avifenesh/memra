"""Retry explicit throttling without repeating unknown provider failures."""

import io
import json
import unittest
from unittest import mock
import urllib.error

import judge_bedrock


class Response(io.BytesIO):
    def __init__(self):
        super().__init__(json.dumps({
            "output": {"message": {"content": []}},
        }).encode())
        self.headers = {"x-amzn-requestid": "test-request"}


class BedrockRetryTest(unittest.TestCase):
    def test_429_is_retried_and_500_is_not(self):
        config = {
            "model_id": "global.anthropic.claude-sonnet-5",
            "region": "us-east-1",
            "max_output_tokens": 4096,
        }
        throttled = urllib.error.HTTPError(
            "https://example.invalid", 429, "throttled",
            {"Retry-After": "1"}, io.BytesIO(),
        )
        with mock.patch(
            "judge_bedrock.urllib.request.urlopen",
            side_effect=[throttled, Response()],
        ) as request, mock.patch(
            "judge_bedrock.time.sleep",
        ) as sleep:
            answer = judge_bedrock.call(
                "synthetic prompt", config, "synthetic token",
            )
            self.assertEqual(answer["request_id"], "test-request")
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(1)
        failed = urllib.error.HTTPError(
            "https://example.invalid", 500, "unknown",
            {}, io.BytesIO(),
        )
        with mock.patch(
            "judge_bedrock.urllib.request.urlopen",
            side_effect=failed,
        ) as request:
            with self.assertRaises(urllib.error.HTTPError):
                judge_bedrock.call(
                    "synthetic prompt", config,
                    "synthetic token",
                )
            self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
