"""Exercise real cache battery entrypoints against protocol failure controls."""
import contextlib
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import cache_qualification as cq

ROOT = Path(__file__).resolve().parents[1]
BATTERIES = [ROOT / "research/glm5-prefix-latent-20260830/battery.py",
             ROOT / "research/glm5-prefix-latent2-20260901/battery2.py"]


def usage():
    return {"prompt_tokens": 200, "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 0}, "spec": {"drafted": 4, "accepted": 3}}


class Service(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode = mode
        self.requests = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        if self.server.mode == "401" or self.headers.get("Authorization") != "Bearer synthetic-test-key":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"unauthorized"}}')
            return
        self.send_response(200)
        self.end_headers()
        if self.server.mode == "error200":
            self.wfile.write(b'{"error":{"message":"engine failed"}}')
        elif body.get("stream"):
            if self.server.mode == "native":
                frames = [{"text": "answer"}, {"stop_reason": "Eos", "prompt_tokens": 200,
                                                "cached_tokens": 0, "n_tokens": 4}]
            else:
                frames = [{"choices": [{"delta": {"reasoning_content": "private thinking"}}]}]
                if self.server.mode != "truncated":
                    frames.append({"choices": [{"delta": {"content": "answer"}}]})
                frames.append({"choices": [{"delta": {}, "finish_reason": "length" if self.server.mode == "truncated" else "stop"}], "usage": usage()})
                if self.server.mode == "stream_error":
                    frames.append({"error": {"message": "settlement failed"}})
            for frame in frames:
                self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
            if self.server.mode != "native":
                self.wfile.write(b"data: [DONE]\n\n")
        elif self.server.mode == "native":
            self.wfile.write(json.dumps({"text": "answer", "stop_reason": "MaxNew", "prompt_tokens": 200,
                                        "cached_tokens": 0, "n_tokens": 4}).encode())
        else:
            self.wfile.write(json.dumps({"choices": [{"text": "answer", "finish_reason": "length"}], "usage": usage()}).encode())


class PoolTests(unittest.TestCase):
    def test_missing_wrong_shape_short_and_empty_pools_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            with self.assertRaisesRegex(cq.QualificationError, "default; set PROMPTS_JSON"):
                cq.load_prompt_pool(path, explicit=False)
            for content in [b"{broken", b'{"A":"wrong format"}', b'{"decode":[]}',
                            json.dumps({"decode": [{"text": "x"}] * 7}).encode(),
                            json.dumps({"decode": [{"text": " "}] * 8}).encode()]:
                path.write_bytes(content)
                with self.assertRaisesRegex(cq.QualificationError, "prompt pool"):
                    cq.load_prompt_pool(path)

    def test_identity_is_the_exact_loaded_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            raw = json.dumps({"decode": [{"text": str(i)} for i in range(8)]}).encode()
            path.write_bytes(raw)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                pool, metadata = cq.load_prompt_pool(path)
            self.assertEqual(metadata["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(metadata["n"], len(pool))
            self.assertIn(str(path), output.getvalue())


class VerdictTests(unittest.TestCase):
    def test_budget_limited_reasoning_is_not_answer_identity_or_continuation(self):
        row = cq.classify("", "reasoning", "length", (20, 0, 128))
        self.assertEqual(row["verdict"], "mechanics_only")
        self.assertTrue(row["mechanics_observed"])
        self.assertFalse(row["completed_answer"])
        self.assertIsNone(row["out_sha16"])
        self.assertFalse(cq.same_identity([row, row]))
        messages = []
        with self.assertRaises(cq.QualificationError):
            cq.append_answer(messages, row)
        self.assertEqual(messages, [])

    def test_raw_tape_does_not_claim_completed_answer(self):
        row = cq.classify("bounded tape", "", "length", (20, 0, 64), raw_tape=True)
        self.assertTrue(cq.same_identity([row, row]))
        self.assertFalse(row["completed_answer"])
        self.assertEqual(row["identity_kind"], "raw_tape")
        chat = cq.classify("bounded tape", "", "length", (20, 0, 64))
        self.assertFalse(chat["identity_eligible"])
        self.assertFalse(cq.same_identity([row, chat]))

    def test_partial_error_invalid_accounting_and_empty_success_never_hash(self):
        for row in [cq.classify("answer", "", "stop", (20, 0, 4), "HTTP 401", True),
                    cq.classify("", "", "stop", (20, 0, 4), raw_tape=True),
                    cq.classify("answer", "", None, (20, 0, 4)),
                    cq.classify("answer", "", "stop", (20, 21, 4)),
                    cq.classify("answer", "", "stop", (20, 0, None))]:
            self.assertFalse(row["identity_eligible"])
            self.assertIsNone(row["out_sha16"])
            self.assertFalse(cq.same_identity([row, row]))

    def test_native_and_openai_json_have_the_same_observed_bytes(self):
        native = {"text": "answer", "stop_reason": "Eos", "prompt_tokens": 200, "cached_tokens": 0, "n_tokens": 4}
        openai = {"choices": [{"text": "answer", "finish_reason": "stop"}], "usage": usage()}
        rows = [cq.classify(*cq.response_fields(d)[:4]) for d in [native, openai]]
        self.assertTrue(cq.same_identity(rows))
        for document in [{"text": "answer", "error": {"code": "deadline_exceeded"}}, {"choices": [], "error": {}}]:
            with self.assertRaises(cq.QualificationError):
                cq.response_fields(document)


class StreamTests(unittest.TestCase):
    def collect(self, raw, body=None):
        response = io.BytesIO(raw)
        response.status = 200
        with tempfile.TemporaryDirectory() as directory, patch.object(cq.urllib.request, "urlopen", return_value=response):
            row = cq.completion("http://unused", body or {"messages": [], "stream": True}, directory, "case")
            recorded = (Path(directory) / ("case.sse" if (body or {}).get("stream", True) else "case.json")).read_bytes()
        self.assertEqual(recorded, raw)
        return row

    def frame(self, document):
        return b"data: " + json.dumps(document).encode() + b"\n\n"

    def test_openai_requires_done_even_after_stop_and_usage(self):
        raw = self.frame({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}], "usage": usage()})
        row = self.collect(raw)
        self.assertFalse(row["completed_answer"])
        self.assertIn("missing [DONE]", row["error"])
        self.assertTrue(self.collect(raw + b"data: [DONE]\n\n")["completed_answer"])

    def test_native_requires_its_own_terminal_frame(self):
        raw = self.frame({"text": "answer"})
        self.assertFalse(self.collect(raw)["identity_eligible"])
        raw += self.frame({"stop_reason": "Eos", "prompt_tokens": 20, "cached_tokens": 4, "n_tokens": 4})
        row = self.collect(raw)
        self.assertTrue(row["completed_answer"])
        self.assertEqual(row["cached_tokens"], 4)

    def test_usage_only_frame_preserves_known_fields(self):
        raw = self.frame({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 20, "completion_tokens": 4}})
        raw += self.frame({"choices": [], "usage": {"prompt_tokens_details": {"cached_tokens": 8}}})
        row = self.collect(raw + b"data: [DONE]\n\n")
        self.assertTrue(row["completed_answer"])
        self.assertEqual((row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]), (20, 8, 4))

    def test_malformed_or_error_stream_retains_the_rest_of_raw_bytes(self):
        for prefix in [b"data: {broken}\n\n", self.frame({"error": {"code": "engine_error"}})]:
            raw = prefix + self.frame({"choices": [{"delta": {"content": "must remain in raw receipt"}}]}) + b"data: [DONE]\n\n"
            self.assertFalse(self.collect(raw)["identity_eligible"])
        raw = self.frame({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}], "usage": usage()})
        self.assertFalse(self.collect(raw + b"data: {unfinished")["identity_eligible"])
        self.assertFalse(self.collect(raw + b"data: [DONE]\n\n" + self.frame({"error": "late"}))["identity_eligible"])

    def test_empty_choices_json_is_not_a_success(self):
        raw = json.dumps({"choices": [], "usage": usage()}).encode()
        self.assertFalse(self.collect(raw, {"stream": False})["completed_answer"])

    def test_raw_tape_cannot_be_armed_for_chat_or_sampled_requests(self):
        for body in [{"messages": [], "stream": False, "temperature": 0, "max_tokens": 4},
                     {"prompt": "x", "stream": False, "max_tokens": 4},
                     {"prompt": "x", "stream": True, "temperature": 0, "max_tokens": 4}]:
            with self.assertRaises(cq.QualificationError), patch.object(cq.urllib.request, "urlopen") as request:
                cq.completion("http://unused", body, "/unused", "case", raw_tape=True)
            request.assert_not_called()

    def test_partial_transport_body_is_retained_and_fails(self):
        class Broken(io.BytesIO):
            status = 200
            def read(self, *args):
                raise http.client.IncompleteRead(b'{"text":"partial', 20)
        with tempfile.TemporaryDirectory() as directory, patch.object(cq.urllib.request, "urlopen", return_value=Broken()):
            row = cq.completion("http://unused", {"stream": False}, directory, "case")
            self.assertFalse(row["identity_eligible"])
            self.assertEqual((Path(directory) / "case.json").read_bytes(), b'{"text":"partial')


class BatteryTests(unittest.TestCase):
    def run_battery(self, script, mode="openai", bad_pool=False):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pool = directory / "prompts.json"
            pool.write_text(json.dumps({"decode": [{"text": "A real test prompt. " * 20}] * 8} if not bad_pool else {"A": "wrong pool"}))
            key = directory / "key"
            key.write_text("synthetic-test-key")
            service = Service(mode)
            thread = threading.Thread(target=service.serve_forever, daemon=True)
            thread.start()
            env = {**os.environ, "EP": f"http://127.0.0.1:{service.server_port}",
                   "PROMPTS_JSON": str(pool), "CACHE_BATTERY_KEY_FILE": str(key), "MODEL": "test"}
            try:
                result = subprocess.run([sys.executable, str(script), str(directory / "out"), "off", "2"],
                                        env=env, capture_output=True, text=True, timeout=20)
            finally:
                service.shutdown()
                service.server_close()
                thread.join()
            receipt = directory / "out/battery.json"
            return result, json.loads(receipt.read_text()) if receipt.exists() else None, service.requests

    def test_both_real_batteries_accept_native_and_openai_with_auth(self):
        for script in BATTERIES:
            for shape in ["native", "openai"]:
                with self.subTest(script=script.name, shape=shape):
                    result, receipt, requests = self.run_battery(script, shape)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                    self.assertTrue(receipt["c1_byte_identity"])
                    self.assertEqual(receipt["verdict"], "PASS")
                    chats = [r for r in requests if "messages" in r]
                    for request in chats:
                        self.assertFalse({"temperature", "top_p", "top_k", "seed", "reasoning_effort"} & request.keys())
                        self.assertTrue(all(m["content"] == "answer" for m in request["messages"] if m["role"] == "assistant"))

    def test_original_false_greens_fail_through_both_entrypoints(self):
        for script in BATTERIES:
            for mode in ["401", "error200", "stream_error", "truncated"]:
                with self.subTest(script=script.name, mode=mode):
                    result, receipt, requests = self.run_battery(script, mode)
                    self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
                    self.assertEqual(receipt["verdict"], "FAIL")
                    self.assertFalse(receipt["completed_answer_pass"])
                    if mode in ("401", "error200"):
                        self.assertFalse(receipt["c1_byte_identity"])
                        self.assertNotIn("ONE sha", result.stdout)
                        self.assertTrue(all(r["out_sha16"] is None for r in receipt["raw"]))
                    chats = [r for r in requests if "messages" in r]
                    self.assertTrue(all(m["role"] != "assistant" for r in chats for m in r["messages"]))

    def test_bad_pool_stops_before_any_http_request(self):
        for script in BATTERIES:
            result, receipt, requests = self.run_battery(script, bad_pool=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("REFUSE: prompt pool", result.stderr)
            self.assertIsNone(receipt)
            self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
