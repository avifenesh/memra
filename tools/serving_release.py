"""Derive serving-gate outcomes from recorded wire data, never summary statuses."""

import json
import sys
from collections import Counter


class ServingGateError(ValueError):
    def __init__(self, message, code="invalid_response"):
        super().__init__(message)
        self.code = code


def require(condition, message, code="invalid_response"):
    if not condition:
        raise ServingGateError(message, code)


def json_object(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON member: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ServingGateError(f"non-JSON number: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except ServingGateError:
        raise
    except (ValueError, RecursionError) as error:
        # JSONDecodeError/UnicodeDecodeError are ValueErrors too. Python's integer
        # digit limit and nesting limit must become one malformed-response row,
        # rather than aborting accounting for all of its healthy peers.
        raise ServingGateError("invalid JSON payload") from error
    require(isinstance(value, dict), "payload must be an object")
    return value


def sse_events(raw):
    """Parse complete Memra SSE frames, retaining their explicit event type."""
    try:
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw.removeprefix("\ufeff")
    except UnicodeDecodeError as error:
        raise ServingGateError("stream is not valid UTF-8") from error
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    require(not text or text.endswith("\n"), "stream ends inside a line", "truncated_200")
    events, data, kind = [], [], "message"
    # The trailing split element is a delimiter artifact, not another blank wire line.
    for line in text.split("\n")[:-1]:
        if not line:
            if data:
                events.append((kind, "\n".join(data)))
            data, kind = [], "message"
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        require(separator and field in {"data", "event", "id", "retry"},
                "unexpected SSE field or non-SSE body")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data.append(value)
        elif field == "event":
            kind = value or "message"
    require(not data, "stream ends inside an event", "truncated_200")
    return events


def usage_counts(value):
    require(isinstance(value, dict), "usage must be an object")
    counts = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = value.get(key)
        require(type(count) is int and count >= 0, f"invalid usage count: {key}")
        counts[key] = count
    require(counts["total_tokens"] == counts["prompt_tokens"] + counts["completion_tokens"],
            "usage total does not equal prompt plus completion")
    if "prompt_tokens_details" in value:
        details = value["prompt_tokens_details"]
        require(isinstance(details, dict), "prompt details must be an object")
        cached = details.get("cached_tokens")
        require(type(cached) is int and 0 <= cached <= counts["prompt_tokens"],
                "invalid cached prompt count")
        counts["cached_tokens"] = cached
    return counts


def reject_wire_error(value):
    if "error" in value:
        error = value["error"]
        require(isinstance(error, dict) and isinstance(error.get("code"), str)
                and error["code"] and isinstance(error.get("message"), str),
                "server emitted a malformed error")
        raise ServingGateError(f"server error {error['code']}: {error['message']}", "typed_error")


def validate_chat_stream(raw, *, model, require_output=True, require_usage=False):
    """Validate the one-choice text/reasoning chat contract used by release cells.

    A 200 response, a role chunk or a DONE sentinel alone is not completion. Native
    SSE and tool-call cells have distinct contracts and must not use this validator.
    """
    content, reasoning, finished, done, usage, frames = [], [], None, False, None, 0
    raw_usage, usage_only_seen = None, False
    for kind, data in sse_events(raw):
        require(not done, "data follows DONE")
        require(kind == "message", f"unexpected chat event type: {kind}")
        if data == "[DONE]":
            require(finished is not None, "DONE without a terminal finish_reason")
            done = True
            continue
        value = json_object(data)
        reject_wire_error(value)
        require(value.get("model") == model, "stream model differs from requested model")
        choices = value.get("choices")
        require(isinstance(choices, list), "stream choices must be an array")
        frames += 1
        if choices:
            require(finished is None, "choice data follows terminal finish_reason")
            require(len(choices) == 1 and isinstance(choices[0], dict)
                    and type(choices[0].get("index")) is int and choices[0]["index"] == 0,
                    "release chat cell requires exactly choice index zero")
            choice = choices[0]
            delta = choice.get("delta")
            require(isinstance(delta, dict), "chat delta must be an object")
            require(not delta.get("tool_calls"), "tool output requires a tool-call cell")
            for key, output in (("content", content), ("reasoning", reasoning),
                                ("reasoning_content", reasoning)):
                piece = delta.get(key)
                require(piece is None or isinstance(piece, str), f"invalid delta {key}")
                if piece:
                    output.append(piece)
            reason = choice.get("finish_reason")
            if reason is not None:
                require(reason in ("stop", "length"), "unexpected text finish_reason")
                finished = reason
        else:
            require(finished is not None and value.get("usage") is not None,
                    "empty choices without final usage")
            require(not usage_only_seen, "duplicate usage-only chunk")
            usage_only_seen = True
        if value.get("usage") is not None:
            require(finished is not None, "usage arrived before completion")
            # Memra always emits usage on the finish chunk, then repeats it in
            # choices=[] when include_usage was requested. Only that identical
            # copy is permitted; two different accounting records cannot pass.
            require(raw_usage is None or (not choices and value["usage"] == raw_usage),
                    "conflicting terminal usage")
            raw_usage = value["usage"]
            usage = usage_counts(value["usage"])
    require(done and finished is not None, "stream is truncated or missing DONE", "truncated_200")
    require(not require_output or content or reasoning, "no emitted text or reasoning")
    require(not require_usage or usage is not None, "requested terminal usage is absent")
    require(not require_output or usage is None or usage["completion_tokens"] > 0,
            "output conflicts with zero completion count")
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "finish_reason": finished, "usage": usage, "frames": frames}


def validate_chat_response(raw, *, model, require_output=True):
    value = json_object(raw)
    reject_wire_error(value)
    require(value.get("model") == model, "response model differs from requested model")
    choices = value.get("choices")
    require(isinstance(choices, list) and len(choices) == 1
            and isinstance(choices[0], dict), "release chat cell requires one choice")
    choice = choices[0]
    require(type(choice.get("index")) is int and choice["index"] == 0,
            "release chat cell requires choice index zero")
    require(choice.get("finish_reason") in ("stop", "length"), "missing or invalid finish_reason")
    message = choice.get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant",
            "chat response must contain an assistant message")
    require(not message.get("tool_calls"), "tool output requires a tool-call cell")
    content, reasoning = message.get("content"), message.get("reasoning", message.get("reasoning_content"))
    require(content is None or isinstance(content, str), "invalid response content")
    require(reasoning is None or isinstance(reasoning, str), "invalid response reasoning")
    require(not require_output or content or reasoning, "no emitted text or reasoning")
    usage = usage_counts(value.get("usage"))
    require(not require_output or usage["completion_tokens"] > 0,
            "output conflicts with zero completion count")
    return {"content": content or "", "reasoning": reasoning or "",
            "finish_reason": choice["finish_reason"], "usage": usage}


def native_terminal(value):
    reject_wire_error(value)
    require(value.get("stop_reason") in ("Eos", "Callback", "MaxNew", "ContextFull"),
            "missing or invalid native stop_reason")
    for key in ("n_tokens", "prompt_tokens", "cached_tokens"):
        require(type(value.get(key)) is int and value[key] >= 0, f"invalid native {key}")
    require(value["cached_tokens"] <= value["prompt_tokens"], "cached count exceeds prompt")
    elapsed = value.get("elapsed_s")
    # Comparing an integer against this bound cannot overflow a float conversion.
    # The comparisons also reject NaN and infinities when elapsed is a float.
    require(type(elapsed) in (int, float) and 0 <= elapsed <= sys.float_info.max,
            "invalid native elapsed time")
    return {"prompt_tokens": value["prompt_tokens"], "completion_tokens": value["n_tokens"],
            "total_tokens": value["prompt_tokens"] + value["n_tokens"],
            "cached_tokens": value["cached_tokens"]}


def validate_native_stream(raw, *, model, require_output=True):
    content, ids, terminal = [], [], None
    for kind, data in sse_events(raw):
        require(terminal is None, "data follows native done")
        value = json_object(data)
        reject_wire_error(value)
        require(kind in ("message", "done"), "native stream emitted an unknown event")
        if kind == "done":
            terminal = value
            continue
        require(value.get("model") == model, "native stream model differs from requested model")
        require(type(value.get("id")) is int and 0 <= value["id"] <= 0xffffffff,
                "invalid emitted native token id")
        require(isinstance(value.get("text"), str), "invalid native token text")
        ids.append(value["id"])
        content.append(value["text"])
    require(terminal is not None, "native stream lacks done event", "truncated_200")
    usage = native_terminal(terminal)
    require(not require_output or (any(content) and usage["completion_tokens"] > 0),
            "native stream emitted no output")
    # A speculative token event can contain coalesced text. Emitted ids are not
    # the full token array; only the blocking native API exposes that snapshot.
    return {"content": "".join(content), "emitted_ids": ids,
            "stop_reason": terminal["stop_reason"], "usage": usage}


def validate_native_response(raw, *, model, require_output=True):
    value = json_object(raw)
    require(value.get("model") == model, "native response model differs from requested model")
    usage = native_terminal(value)
    require(isinstance(value.get("text"), str), "invalid native response text")
    tokens = value.get("tokens")
    require(isinstance(tokens, list) and all(type(t) is int and 0 <= t <= 0xffffffff for t in tokens),
            "invalid native response token array")
    require(len(tokens) == usage["completion_tokens"], "native token array/count mismatch")
    require(not require_output or (value["text"] and tokens), "native response emitted no output")
    return {"content": value["text"], "tokens": tokens,
            "stop_reason": value["stop_reason"], "usage": usage}


WIRE_VALIDATORS = {"chat_sse": validate_chat_stream, "chat_json": validate_chat_response,
                   "native_sse": validate_native_stream, "native_json": validate_native_response}


def account_attempts(schedule, observations):
    """Classify every scheduled client attempt from its raw response.

    This is accounting, not a scenario pass. Fault/lifecycle predicates must still
    prove recovery and server-side cancellation. Missing observations are errors,
    so a collector cannot publish latency for just its surviving clients.
    """
    require(isinstance(schedule, list) and schedule, "empty request schedule")
    require(isinstance(observations, list), "observations must be an array")
    planned, observed = {}, {}
    for item in schedule:
        require(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"],
                "invalid scheduled request id")
        require(item["id"] not in planned, "duplicate scheduled request id")
        require(item.get("wire") in WIRE_VALIDATORS and isinstance(item.get("model"), str)
                and item["model"], "invalid scheduled wire/model")
        planned[item["id"]] = item
    for item in observations:
        require(isinstance(item, dict) and isinstance(item.get("id"), str), "invalid observation id")
        require(item["id"] not in observed, "duplicate observed request id")
        observed[item["id"]] = item
    require(planned.keys() == observed.keys(), "missing or unknown request observations")
    results = []
    for request_id, request in planned.items():
        observation = observed[request_id]
        start, end = observation.get("started_ns"), observation.get("finished_ns")
        require(type(start) is int and type(end) is int and 0 <= start <= end,
                "invalid attempt monotonic timestamps")
        result = {"id": request_id, "wall_ns": end - start}
        transport = observation.get("transport_error")
        status = observation.get("status")
        require(status is None or (type(status) is int and 100 <= status <= 599), "invalid HTTP status")
        if transport is not None:
            require(isinstance(transport, dict) and transport.get("kind") in
                    ("timeout", "disconnect", "cancelled", "connect", "read")
                    and isinstance(transport.get("message"), str), "unclassified transport error")
            # A cancelled socket alone does not prove the worker released state.
            result.update(outcome="client_cancelled" if transport["kind"] == "cancelled"
                          else "transport_error", detail=transport["kind"])
        else:
            require(status is not None, "attempt has neither HTTP response nor transport error")
            raw = observation.get("body")
            require(isinstance(raw, bytes), "response body must be preserved raw bytes")
            try:
                if status != 200:
                    value = json_object(raw)
                    error = value.get("error")
                    require(isinstance(error, dict) and isinstance(error.get("code"), str)
                            and error["code"] and isinstance(error.get("message"), str),
                            "HTTP failure lacks a typed error")
                    result.update(outcome="refused" if status in (429, 503) else "typed_error",
                                  status=status, error_code=error["code"])
                else:
                    options = {"model": request["model"]}
                    if request["wire"] == "chat_sse":
                        options["require_usage"] = True
                    parsed = WIRE_VALIDATORS[request["wire"]](raw, **options)
                    result.update(outcome="clean_success", response=parsed)
            except ServingGateError as error:
                result.update(outcome=error.code if status == 200 else "invalid_http_error",
                              detail=str(error))
        results.append(result)
    counts = Counter(result["outcome"] for result in results)
    return {"attempted": len(schedule), "counts": dict(sorted(counts.items())),
            "requests": results,
            "clean_latency_ns": [r["wall_ns"] for r in results if r["outcome"] == "clean_success"]}
