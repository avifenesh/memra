"""CPU refusal checks for the serving collector; these do not qualify a model."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "collector", Path(__file__).with_name("collect-serving-qualification.py"))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def response():
    return {"id": "completion-1", "model": "gate", "choices": [
        {"text": "alpha beta", "finish_reason": "length"}], "usage": {
            "prompt_tokens": 106, "completion_tokens": 2, "total_tokens": 108,
            "prompt_tokens_details": {"cached_tokens": 0}}}


def stream():
    full = response()
    first = copy.deepcopy(full)
    first.pop("usage")
    first["choices"][0]["finish_reason"] = None
    full["choices"][0]["text"] = ""
    return "\n\n".join("data: " + json.dumps(d) for d in (first, full)) + "\n\ndata: [DONE]\n\n"


class RefusalChecks(unittest.TestCase):
    def test_stub_and_error_are_rejected(self):
        for document in ({"choices": [{"text": "ok"}]}, {"error": "failed"}, {}):
            with self.assertRaises(ValueError):
                collector.completion(document, 64)

    def test_accounting_requires_numbers_and_total(self):
        for key, bad in (("completion_tokens", 0), ("completion_tokens", 65),
                         ("prompt_tokens", True), ("total_tokens", 109)):
            doc = response()
            doc["usage"][key] = bad
            with self.assertRaises(ValueError):
                collector.completion(doc, 64)

    def test_stream_has_same_bytes_and_usage(self):
        a = collector.completion(response(), 64)
        b = collector.stream_completion(stream(), 64)
        self.assertEqual(a["text_sha256"], b["text_sha256"])
        self.assertEqual(a["completion_tokens"], b["completion_tokens"])

    def test_stream_requires_done_finish_usage_and_stable_id(self):
        raw = stream()
        for bad in (raw.replace("data: [DONE]", ""), raw + "data: [DONE]\n",
                    raw.replace('"finish_reason": "length"', '"finish_reason": null'),
                    raw.replace('"usage":', '"omitted_usage":'),
                    raw.replace('"id": "completion-1"', '"id": "other"', 1)):
            with self.assertRaises(ValueError):
                collector.stream_completion(bad, 64)

    def test_cache_requires_real_grid_hit_and_output_identity(self):
        cold = collector.completion(response(), 64)
        warm = {**cold, "cached_tokens": 64}
        collector.cache_check(cold, warm)
        for bad in ({**warm, "cached_tokens": 0}, {**warm, "cached_tokens": 106},
                    {**warm, "text_sha256": "different"}, {**warm, "completion_tokens": 3}):
            with self.assertRaises(ValueError):
                collector.cache_check(cold, bad)

    def test_red_arm_must_actually_fail(self):
        with self.assertRaisesRegex(ValueError, "incorrectly accepted"):
            collector.rejected(lambda: collector.completion(response(), 64))

    def test_parent_grid_override_does_not_change_child_grid_assertions(self):
        document = response()
        document["usage"].update(prompt_tokens=241, total_tokens=243)
        cold = collector.completion(document, 64)
        warm = {**cold, "cached_tokens": 224}
        with mock.patch.dict(collector.os.environ, {"MEMRA_GDN_CHUNK": "64"}):
            self.assertEqual(collector.server_capture_len(241), 224)
            collector.cache_check(cold, warm)

    def test_transport_failure_preserves_partial_stream_and_request(self):
        first = b'data: {"choices":[{"text":"partial"}]}\n'
        connection = mock.Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.readline.side_effect = [first, OSError("connection reset")]
        with tempfile.TemporaryDirectory() as tmp:
            server = object.__new__(collector.Server)
            server.args = SimpleNamespace(port=18120)
            server.out = Path(tmp)
            server.key = "synthetic-test-key"
            with mock.patch.object(collector.http.client, "HTTPConnection", return_value=connection):
                with self.assertRaisesRegex(OSError, "connection reset"):
                    server.request("broken", {"stream": True, "max_tokens": 64})
            self.assertEqual((server.out / "broken.sse").read_bytes(), first)
            receipt = json.loads((server.out / "broken.request.json").read_text())
            self.assertEqual(receipt["error"], "connection reset")
            self.assertNotIn(server.key, json.dumps(receipt))
            connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
