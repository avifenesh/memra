"""Wire-level release controls; no GPU, server process, or model artifacts needed."""

import json
import unittest

from serving_release import (ServingGateError, account_attempts, json_object, sse_events,
                             validate_chat_response, validate_chat_stream,
                             validate_native_response, validate_native_stream)


def chunk(delta=None, finish=None, model="gate"):
    return {"model": model, "choices": [{"index": 0, "delta": delta or {},
                                        "finish_reason": finish}]}


def stream(*events):
    return "".join("data: " + (e if isinstance(e, str) else json.dumps(e))
                   + "\n\n" for e in events).encode()


class ChatWireTests(unittest.TestCase):
    def valid(self, raw, **options):
        return validate_chat_stream(raw, model="gate", **options)

    def refuses(self, raw, **options):
        with self.assertRaises(ServingGateError):
            self.valid(raw, **options)

    def test_text_and_reasoning_are_both_counted_as_output(self):
        result = self.valid(stream(chunk({"role": "assistant"}),
                                   chunk({"reasoning": "Think."}),
                                   chunk({"content": "Answer."}, "stop"), "[DONE]"))
        self.assertEqual((result["reasoning"], result["content"]), ("Think.", "Answer."))
        result = self.valid(stream(chunk({"reasoning_content": "Reasoning only"}, "length"), "[DONE]"))
        self.assertEqual(result["content"], "")

    def test_terminal_usage_and_zero_output_are_explicit(self):
        usage = {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}
        raw = stream(chunk(finish="stop"), {"model": "gate", "choices": [], "usage": usage}, "[DONE]")
        self.refuses(raw)
        result = self.valid(raw, require_output=False, require_usage=True)
        self.assertEqual(result["usage"], usage)

    def test_200_and_done_cannot_hide_missing_finish_or_error(self):
        self.refuses(stream(chunk({"content": "partial"}), "[DONE]"))
        self.refuses(stream("[DONE]"))
        self.refuses(stream(chunk({"content": "partial"}), {"error": "worker died"}, "[DONE]"))
        self.refuses(stream(chunk({"content": "partial"}, "stop")))

    def test_truncated_frames_invalid_json_and_utf8_refuse(self):
        self.refuses(stream(chunk({"content": "ok"}, "stop"), "[DONE]")[:-1])
        self.refuses(b'data: {"model":\n\n')
        self.refuses(b'data: \xff\n\n')
        self.refuses(b'not sse\n\n')

    def test_other_models_choices_and_late_output_refuse(self):
        self.refuses(stream(chunk({"content": "wrong"}, "stop", "other"), "[DONE]"))
        extra = chunk({"content": "x"}, "stop"); extra["choices"][0]["index"] = 1
        self.refuses(stream(extra, "[DONE]"))
        self.refuses(stream(chunk({"content": "ok"}, "stop"), chunk({"content": "late"}), "[DONE]"))
        self.refuses(stream(chunk({"content": "ok"}, "stop"), "[DONE]", "[DONE]"))

    def test_usage_must_be_terminal_complete_and_consistent(self):
        terminal = chunk({"content": "ok"}, "stop")
        usage = {"model": "gate", "choices": [],
                 "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}
        self.valid(stream(terminal, usage, "[DONE]"), require_usage=True)
        self.refuses(stream(usage, terminal, "[DONE]"))
        self.refuses(stream(terminal, "[DONE]"), require_usage=True)
        self.refuses(stream(terminal, usage, usage, "[DONE]"))
        for invalid in (True, -1, 3.5, "1"):
            altered = json.loads(json.dumps(usage)); altered["usage"]["completion_tokens"] = invalid
            self.refuses(stream(terminal, altered, "[DONE]"))
        usage["usage"]["total_tokens"] = 99
        self.refuses(stream(terminal, usage, "[DONE]"))

    def test_memra_finish_usage_and_requested_copy_must_match(self):
        terminal = chunk({"content": "ok"}, "stop")
        terminal["usage"] = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
                             "prompt_tokens_details": {"cached_tokens": 2}, "elapsed_s": 0.25}
        copy = {"model": "gate", "choices": [], "usage": dict(terminal["usage"])}
        self.valid(stream(terminal, copy, "[DONE]"), require_usage=True)
        self.valid(stream(terminal, "[DONE]"), require_usage=True)
        self.refuses(stream(terminal, copy, copy, "[DONE]"))
        for field, changed in (("completion_tokens", 2), ("elapsed_s", 0.5)):
            altered = json.loads(json.dumps(copy)); altered["usage"][field] = changed
            self.refuses(stream(terminal, altered, "[DONE]"))
        terminal["usage"]["prompt_tokens_details"]["cached_tokens"] = 5
        self.refuses(stream(terminal, "[DONE]"))

    def test_crlf_multiline_comments_and_unicode_line_separator(self):
        raw = (': heartbeat\r\ndata: {"model":"gate",\r\n'
               'data: "choices":[{"index":0,"delta":{"content":"a\u2028b"},"finish_reason":"stop"}]}\r\n\r\n'
               'data: [DONE]\r\n\r\n').encode()
        self.assertEqual(self.valid(raw)["content"], "a\u2028b")
        self.assertEqual(sse_events('event: done\ndata: {}\n\n'), [("done", "{}")])
        self.refuses(b'event: error\ndata: {}\n\n')


def native_done(**changes):
    return dict(stop_reason="MaxNew", n_tokens=2, prompt_tokens=4,
                cached_tokens=0, elapsed_s=0.25, **changes)


def native_stream(*events):
    return "".join(f"event: {kind}\ndata: {json.dumps(value)}\n\n"
                   for kind, value in events).encode()


class NativeWireTests(unittest.TestCase):
    def test_native_sse_has_named_done_and_coalesced_ids(self):
        raw = native_stream(("message", {"model": "gate", "id": 10, "text": "two tokens"}),
                            ("done", native_done()))
        result = validate_native_stream(raw, model="gate")
        self.assertEqual(result["emitted_ids"], [10])
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        with self.assertRaises(ServingGateError):
            validate_chat_stream(raw, model="gate")
        for invalid in (raw.split(b"event: done")[0], raw + b"data: [DONE]\n\n",
                        raw + native_stream(("done", native_done())), raw[:-1]):
            with self.subTest(raw=invalid), self.assertRaises(ServingGateError):
                validate_native_stream(invalid, model="gate")

    def test_native_empty_token_text_is_allowed_but_empty_output_is_not(self):
        raw = native_stream(("message", {"model": "gate", "id": 10, "text": ""}),
                            ("message", {"model": "gate", "id": 11, "text": "ok"}),
                            ("done", native_done()))
        validate_native_stream(raw, model="gate")
        with self.assertRaises(ServingGateError):
            validate_native_stream(native_stream(("done", native_done())), model="gate")

    def test_native_blocking_snapshot_matches_count(self):
        value = dict(native_done(), model="gate", text="ok", tokens=[10, 11])
        result = validate_native_response(json.dumps(value), model="gate")
        self.assertEqual(result["tokens"], [10, 11])
        for changed in ([10], [10, True], [10, -1], [10, 2**32]):
            with self.subTest(tokens=changed), self.assertRaises(ServingGateError):
                validate_native_response(json.dumps(dict(value, tokens=changed)), model="gate")

    def test_native_invalid_terminal_counts_reason_and_errors_refuse(self):
        value = dict(native_done(), model="gate", text="ok", tokens=[10, 11])
        for key, changed in (("stop_reason", "Deadline"), ("stop_reason", "unknown"),
                             ("cached_tokens", 5), ("n_tokens", True),
                             ("elapsed_s", -1), ("elapsed_s", float("nan")),
                             ("error", {"code": "deadline_exceeded", "message": "deadline"})):
            with self.subTest(key=key), self.assertRaises(ServingGateError):
                validate_native_response(json.dumps(dict(value, **{key: changed})), model="gate")
        raw = native_stream(("error", {"error": {"code": "worker_unavailable", "message": "gone"}}))
        with self.assertRaises(ServingGateError) as error:
            validate_native_stream(raw, model="gate")
        self.assertEqual(error.exception.code, "typed_error")


class BlockingAndAccountingTests(unittest.TestCase):
    def chat(self):
        return {"model": "gate", "choices": [{"index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}

    def test_blocking_chat_requires_output_terminal_and_usage(self):
        value = self.chat()
        validate_chat_response(json.dumps(value), model="gate")
        value["choices"][0]["message"] = {"role": "assistant", "reasoning": "think"}
        validate_chat_response(json.dumps(value), model="gate")
        for key, changed in (("finish_reason", None), ("index", True),
                             ("message", {"role": "assistant", "content": ""}),
                             ("message", {"role": "assistant", "tool_calls": [1]})):
            altered = self.chat(); altered["choices"][0][key] = changed
            with self.subTest(key=key), self.assertRaises(ServingGateError):
                validate_chat_response(json.dumps(altered), model="gate")
        for key in ("model", "usage"):
            altered = self.chat(); del altered[key]
            with self.assertRaises(ServingGateError):
                validate_chat_response(json.dumps(altered), model="gate")

    def test_json_duplicate_members_and_non_json_numbers_refuse(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '[1]', b'{"x":"\xff"}'):
            with self.subTest(raw=raw), self.assertRaises(ServingGateError):
                json_object(raw)

    def test_every_client_stays_in_denominator(self):
        ids = ["ok", "truncated", "overloaded", "timeout", "cancelled", "malformed", "stream_error"]
        schedule = [dict(id=i, model="gate", wire="chat_sse" if i in ("truncated", "stream_error")
                         else "chat_json") for i in ids]
        observations = [dict(id=i, started_ns=10, finished_ns=20, status=200,
                             body=json.dumps(self.chat()).encode()) for i in ids]
        observations[1]["body"] = stream(chunk({"content": "partial"}))
        observations[2].update(status=503, body=json.dumps({"error": {
            "code": "overloaded", "message": "busy"}}).encode())
        for index, kind in ((3, "timeout"), (4, "cancelled")):
            observations[index].update(transport_error={"kind": kind, "message": kind})
        observations[5]["body"] = b"<html>error</html>"
        observations[6]["body"] = stream({"error": {"code": "overloaded", "message": "worker gone"}}, "[DONE]")
        result = account_attempts(schedule, observations)
        self.assertEqual(result["attempted"], 7)
        self.assertEqual(sum(result["counts"].values()), 7)
        self.assertEqual(result["counts"], {"clean_success": 1, "client_cancelled": 1,
                         "invalid_response": 1, "refused": 1, "transport_error": 1,
                         "truncated_200": 1, "typed_error": 1})
        self.assertEqual(result["clean_latency_ns"], [10])

    def test_omitted_duplicate_or_unknown_clients_refuse(self):
        schedule = [dict(id="one", model="gate", wire="chat_json")]
        observations = [dict(id="one", started_ns=10, finished_ns=20, status=200,
                             body=json.dumps(self.chat()).encode())]
        for wrong in ([], observations * 2, [dict(observations[0], id="other")]):
            with self.assertRaises(ServingGateError):
                account_attempts(schedule, wrong)
        with self.assertRaises(ServingGateError):
            account_attempts(schedule * 2, observations)
        for changes in ({"status": None}, {"started_ns": 30}, {"finished_ns": True},
                        {"transport_error": {"kind": "success", "message": "ok"}}):
            with self.subTest(changes=changes), self.assertRaises(ServingGateError):
                account_attempts(schedule, [dict(observations[0], **changes)])

    def test_numeric_limits_and_deep_json_do_not_drop_healthy_peers(self):
        good = json.dumps(self.chat()).encode()
        native = dict(native_done(), model="gate", text="ok", tokens=[10, 11], elapsed_s=10**400)
        bad_cases = [(b'{"oversized":' + b"1" * 5000 + b"}", "chat_json", 200),
                     (json.dumps(native).encode(), "native_json", 200),
                     (b'{"deep":' + b"[" * 3000 + b"0" + b"]" * 3000 + b"}", "chat_json", 200),
                     (b'{"oversized":' + b"1" * 5000 + b"}", "chat_json", 503)]
        for raw, wire, status in bad_cases:
            with self.subTest(wire=wire, status=status, body_size=len(raw)):
                schedule = [dict(id=i, model="gate", wire=wire if i == "bad" else "chat_json")
                            for i in ("first", "bad", "last")]
                observations = [dict(id=i, started_ns=10, finished_ns=20,
                                     status=status if i == "bad" else 200,
                                     body=raw if i == "bad" else good)
                                for i in ("first", "bad", "last")]
                result = account_attempts(schedule, observations)
                self.assertEqual(result["attempted"], 3)
                self.assertEqual(len(result["requests"]), 3)
                self.assertEqual(result["counts"], {"clean_success": 2,
                                 "invalid_response" if status == 200 else "invalid_http_error": 1})
                self.assertEqual(result["clean_latency_ns"], [10, 10])


if __name__ == "__main__":
    unittest.main()
