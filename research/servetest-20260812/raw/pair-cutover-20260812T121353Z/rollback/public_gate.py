#!/usr/bin/env python3
"""Gate the public Q27 endpoint and retain content-free protocol receipts."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


PROMPT_TOKENS = 4_860
PROMPT_SHA256 = "eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb"


class GateFailure(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def fixed_prompt_ids(count: int = PROMPT_TOKENS) -> list[int]:
    prompt = [5_000 + ((position + 105 + 1_008 * 131) % 1_024) for position in range(count)]
    if count == PROMPT_TOKENS and canonical_sha256(prompt) != PROMPT_SHA256:
        raise GateFailure("frozen sellgate prompt identity does not match its locked SHA-256")
    return prompt


def cached_tokens(usage: dict[str, Any]) -> int:
    return int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def validate_usage(
    usage: Any,
    *,
    expected_prompt: int | None = None,
    expected_cached: int | None = None,
    expected_completion: int | None = None,
) -> tuple[bool, str]:
    if not isinstance(usage, dict):
        return False, "usage is absent or not an object"
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    cached = cached_tokens(usage)
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return False, "prompt/completion usage is not integral"
    if min(prompt, completion, cached) < 0 or cached > prompt:
        return False, "usage violates non-negative cached<=prompt"
    if total != prompt + completion:
        return False, f"total_tokens={total!r}, expected {prompt + completion}"
    for label, actual, expected in (
        ("prompt", prompt, expected_prompt),
        ("cached", cached, expected_cached),
        ("completion", completion, expected_completion),
    ):
        if expected is not None and actual != expected:
            return False, f"{label}_tokens={actual}, expected {expected}"
    return True, "ok"


@dataclasses.dataclass
class HttpResult:
    status: int | None
    headers: dict[str, str]
    body: Any
    elapsed_ms: float
    request_id: str | None
    error: str | None = None
    first_content_ms: float | None = None
    done: bool | None = None

    def receipt(self) -> dict[str, Any]:
        body = self.body if isinstance(self.body, dict) else {}
        return {
            "http_status": self.status,
            "request_id": self.request_id,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "first_content_ms": (
                round(self.first_content_ms, 3)
                if self.first_content_ms is not None
                else None
            ),
            "done": self.done,
            "usage": body.get("usage"),
            "error": self.error,
            "rate_limit": {
                key: self.headers.get(key)
                for key in (
                    "x-ratelimit-limit",
                    "x-ratelimit-remaining",
                    "x-ratelimit-reset",
                    "retry-after",
                    "retry-after-ms",
                )
                if key in self.headers
            },
        }


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context()

    def _headers(self, authenticated: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "memra-cx-servetest/1",
        }
        if authenticated and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_json(self, path: str, *, authenticated: bool = False) -> HttpResult:
        request = urllib.request.Request(
            self.base_url + path,
            headers=self._headers(authenticated),
        )
        return self._json_request(request)

    def post_json(self, path: str, payload: dict[str, Any]) -> HttpResult:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=self._headers(),
            method="POST",
        )
        return self._json_request(request)

    def post_sse(self, path: str, payload: dict[str, Any]) -> HttpResult:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=self._headers(),
            method="POST",
        )
        started = time.monotonic()
        status: int | None = None
        response_headers: dict[str, str] = {}
        request_ids: set[str] = set()
        usage: dict[str, Any] = {}
        visible: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        first_content_ms: float | None = None
        done = False
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self.ssl_context
            ) as response:
                status = response.status
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                for raw_line in response:
                    line = raw_line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise GateFailure(json.dumps(event["error"], sort_keys=True))
                    if event.get("id"):
                        request_ids.add(str(event["id"]))
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = choice.get("text") or delta.get("content") or ""
                        if piece:
                            if first_content_ms is None:
                                first_content_ms = (time.monotonic() - started) * 1_000
                            visible.append(str(piece))
                        for tool_delta in delta.get("tool_calls") or []:
                            index = int(tool_delta.get("index") or 0)
                            call = tool_calls.setdefault(
                                index,
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": None, "arguments": ""},
                                },
                            )
                            if tool_delta.get("id") is not None:
                                call["id"] = tool_delta["id"]
                            function = tool_delta.get("function") or {}
                            if function.get("name") is not None:
                                call["function"]["name"] = function["name"]
                            if function.get("arguments") is not None:
                                call["function"]["arguments"] += str(function["arguments"])
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
        except urllib.error.HTTPError as exc:
            return self._http_error(exc, started)
        except Exception as exc:  # Preserve the exact synthetic-probe error in the receipt.
            return HttpResult(
                status,
                response_headers,
                {},
                (time.monotonic() - started) * 1_000,
                response_headers.get("x-request-id"),
                f"{type(exc).__name__}: {exc}",
                first_content_ms,
                done,
            )
        header_id = response_headers.get("x-request-id")
        request_id = next(iter(request_ids), header_id)
        error = None
        if len(request_ids) > 1:
            error = f"stream changed request id: {sorted(request_ids)}"
        elif header_id and request_ids and header_id not in request_ids:
            error = f"header request id {header_id!r} differs from stream {request_id!r}"
        text = "".join(visible)
        return HttpResult(
            status,
            response_headers,
            {
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "finish_reason": finish_reason,
                "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
                "usage": usage,
            },
            (time.monotonic() - started) * 1_000,
            request_id,
            error,
            first_content_ms,
            done,
        )

    def _json_request(self, request: urllib.request.Request) -> HttpResult:
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self.ssl_context
            ) as response:
                raw = response.read()
                body = json.loads(raw) if raw else {}
                headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                body_id = str(body["id"]) if isinstance(body, dict) and body.get("id") else None
                header_id = headers.get("x-request-id")
                error = None
                if body_id and header_id and body_id != header_id:
                    error = f"header request id {header_id!r} differs from body {body_id!r}"
                return HttpResult(
                    response.status,
                    headers,
                    body,
                    (time.monotonic() - started) * 1_000,
                    body_id or header_id,
                    error,
                )
        except urllib.error.HTTPError as exc:
            return self._http_error(exc, started)
        except Exception as exc:
            return HttpResult(
                None,
                {},
                {},
                (time.monotonic() - started) * 1_000,
                None,
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError, started: float) -> HttpResult:
        raw = exc.read()
        try:
            body: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {"raw": raw.decode(errors="replace")[:1_000]}
        headers = {key.lower(): value for key, value in exc.headers.items()}
        return HttpResult(
            exc.code,
            headers,
            body,
            (time.monotonic() - started) * 1_000,
            headers.get("x-request-id"),
        )


class ReceiptWriter:
    def __init__(self, path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")
        self.rows: list[dict[str, Any]] = []

    def write(self, check: str, ok: bool, detail: str, **fields: Any) -> None:
        row = {
            "seq": len(self.rows) + 1,
            "ts": utc_now(),
            "check": check,
            "ok": ok,
            "detail": detail,
            **fields,
        }
        self.file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.file.flush()
        self.rows.append(row)

    def close(self) -> None:
        self.file.close()


def chat_payload(model: str, text: str, stream: bool, max_tokens: int = 24) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 34_071,
        "reasoning": {"enabled": False},
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def completion_payload(
    model: str,
    prompt: list[int],
    salt: str,
    *,
    stream: bool,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_ids": prompt,
        "max_ctx": len(prompt) + max_tokens + 8,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 3_407,
        "cache_salt": salt,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def rate_headers_ok(result: HttpResult) -> bool:
    limit = result.headers.get("x-ratelimit-limit")
    remaining = result.headers.get("x-ratelimit-remaining")
    return limit == "4" and remaining is not None and remaining.isdigit() and 0 <= int(remaining) <= 3


def usage_from(result: HttpResult) -> dict[str, Any]:
    return result.body.get("usage") if isinstance(result.body, dict) else {}


def message_text(result: HttpResult, stream: bool) -> str:
    if not isinstance(result.body, dict):
        return ""
    if stream:
        return str(result.body.get("text") or "")
    choices = result.body.get("choices") or []
    message = choices[0].get("message") if choices else {}
    return str((message or {}).get("content") or "")


def record_result(
    writer: ReceiptWriter,
    check: str,
    result: HttpResult,
    ok: bool,
    detail: str,
    **fields: Any,
) -> None:
    writer.write(check, ok, detail, **result.receipt(), **fields)
    if not ok:
        raise GateFailure(f"{check}: {detail}")


def run_plain(client: Client, writer: ReceiptWriter, model: str, stream: bool) -> HttpResult:
    payload = chat_payload(model, "Reply with exactly GATEWAY_OK and nothing else.", stream)
    result = (
        client.post_sse("/v1/chat/completions", payload)
        if stream
        else client.post_json("/v1/chat/completions", payload)
    )
    text = message_text(result, stream)
    usage_ok, usage_detail = validate_usage(usage_from(result))
    ok = (
        result.status == 200
        and result.error is None
        and result.request_id is not None
        and text.strip() == "GATEWAY_OK"
        and usage_ok
        and rate_headers_ok(result)
        and (not stream or result.done is True)
    )
    detail = "ok" if ok else f"plain protocol/output mismatch; usage={usage_detail}"
    record_result(
        writer,
        "plain_stream" if stream else "plain_nonstream",
        result,
        ok,
        detail,
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return result


def run_tool(client: Client, writer: ReceiptWriter, model: str, stream: bool) -> HttpResult:
    payload = chat_payload(
        model,
        'Call echo_probe exactly once with value "gateway-tools-ok". Do not answer in prose.',
        stream,
        max_tokens=128,
    )
    payload["tools"] = [{
        "type": "function",
        "function": {
            "name": "echo_probe",
            "description": "Return the supplied synthetic probe value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    payload["tool_choice"] = "auto"
    result = (
        client.post_sse("/v1/chat/completions", payload)
        if stream
        else client.post_json("/v1/chat/completions", payload)
    )
    body = result.body if isinstance(result.body, dict) else {}
    choices = body.get("choices") or []
    message = choices[0].get("message") if choices else {}
    calls = (
        body.get("tool_calls") or []
        if stream
        else (message or {}).get("tool_calls") or []
    )
    finish_reason = body.get("finish_reason") if stream else (
        choices[0].get("finish_reason") if choices else None
    )
    arguments: Any = None
    if len(calls) == 1 and calls[0].get("function", {}).get("name") == "echo_probe":
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            arguments = None
    usage_ok, usage_detail = validate_usage(usage_from(result))
    ok = (
        result.status == 200
        and result.error is None
        and result.request_id is not None
        and finish_reason == "tool_calls"
        and arguments == {"value": "gateway-tools-ok"}
        and usage_ok
        and rate_headers_ok(result)
        and (not stream or result.done is True)
    )
    detail = "ok" if ok else f"tool call mismatch; usage={usage_detail}"
    record_result(writer, f"tool_{'stream' if stream else 'nonstream'}", result, ok, detail,
                  tool_arguments=arguments)
    return result


def run_structured(client: Client, writer: ReceiptWriter, model: str, stream: bool) -> HttpResult:
    payload = chat_payload(
        model,
        'Return a JSON object whose probe field is exactly "gateway-structured-ok" and ok is true.',
        stream,
        max_tokens=64,
    )
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "gateway_probe",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "probe": {"type": "string", "const": "gateway-structured-ok"},
                    "ok": {"type": "boolean", "const": True},
                },
                "required": ["probe", "ok"],
                "additionalProperties": False,
            },
        },
    }
    result = (
        client.post_sse("/v1/chat/completions", payload)
        if stream
        else client.post_json("/v1/chat/completions", payload)
    )
    text = message_text(result, stream)
    try:
        parsed: Any = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    usage_ok, usage_detail = validate_usage(usage_from(result))
    ok = (
        result.status == 200
        and result.error is None
        and result.request_id is not None
        and parsed == {"probe": "gateway-structured-ok", "ok": True}
        and usage_ok
        and rate_headers_ok(result)
        and (not stream or result.done is True)
    )
    detail = "ok" if ok else f"structured output mismatch; usage={usage_detail}"
    record_result(
        writer,
        f"structured_{'stream' if stream else 'nonstream'}",
        result,
        ok,
        detail,
        parsed=parsed,
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    return result


def run_cache(client: Client, writer: ReceiptWriter, model: str) -> list[HttpResult]:
    # Use the frozen sellgate prompt: it is qualified to reach the 60-token cap,
    # unlike arbitrary short token-id prefixes that can legitimately emit EOS.
    prompt = fixed_prompt_ids()
    salt = f"servetest-cache-{time.time_ns()}"
    results: list[HttpResult] = []
    golden: str | None = None
    for index in range(3):
        result = client.post_json(
            "/v1/completions",
            completion_payload(model, prompt, salt, stream=False, max_tokens=60),
        )
        body = result.body if isinstance(result.body, dict) else {}
        choices = body.get("choices") or []
        text = str(choices[0].get("text") or "") if choices else ""
        output_hash = hashlib.sha256(text.encode()).hexdigest()
        if golden is None:
            golden = output_hash
        expected_cached = 0 if index == 0 else len(prompt)
        usage_ok, usage_detail = validate_usage(
            usage_from(result),
            expected_prompt=len(prompt),
            expected_cached=expected_cached,
            expected_completion=60,
        )
        ok = (
            result.status == 200
            and result.error is None
            and result.request_id is not None
            and bool(text)
            and output_hash == golden
            and usage_ok
            and rate_headers_ok(result)
        )
        detail = "ok" if ok else f"cache/output/accounting mismatch; usage={usage_detail}"
        record_result(writer, "cache_exact", result, ok, detail, index=index,
                      expected_cached_tokens=expected_cached, output_sha256=output_hash)
        results.append(result)
    return results


def validate_429(result: HttpResult) -> tuple[bool, str]:
    body = result.body if isinstance(result.body, dict) else {}
    error = body.get("error") or {}
    retry = result.headers.get("retry-after")
    retry_ms = result.headers.get("retry-after-ms")
    ok = (
        result.status == 429
        and error.get("type") == "rate_limit_error"
        and error.get("code") == "rate_limit_exceeded"
        and retry is not None
        and retry_ms is not None
        and retry.isdigit()
        and retry_ms.isdigit()
        and int(retry_ms) == int(retry) * 1_000
        and result.headers.get("x-ratelimit-limit") == "4"
        and result.headers.get("x-ratelimit-remaining") == "0"
        and result.request_id is not None
    )
    return ok, "ok" if ok else "429 body or retry/rate-limit headers are not exact"


def run_overload(client: Client, writer: ReceiptWriter, model: str) -> list[HttpResult]:
    barrier = threading.Barrier(8)
    prompt = fixed_prompt_ids()

    def one(index: int) -> HttpResult:
        payload = completion_payload(
            model,
            prompt,
            f"servetest-overload-{time.time_ns()}-{index}",
            stream=index % 2 == 0,
            max_tokens=60,
        )
        barrier.wait(timeout=30)
        return (
            client.post_sse("/v1/completions", payload)
            if index % 2 == 0
            else client.post_json("/v1/completions", payload)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(one, index) for index in range(8)]
        results = [future.result(timeout=client.timeout + 30) for future in futures]
    statuses = Counter(result.status for result in results)
    overall_ok = statuses == Counter({200: 4, 429: 4})
    for index, result in enumerate(results):
        if result.status == 429:
            ok, detail = validate_429(result)
        elif result.status == 200:
            usage_ok, usage_detail = validate_usage(
                usage_from(result),
                expected_prompt=PROMPT_TOKENS,
                expected_completion=60,
            )
            ok = (
                result.error is None
                and result.request_id is not None
                and usage_ok
                and rate_headers_ok(result)
                and (index % 2 != 0 or result.done is True)
            )
            detail = "ok" if ok else f"accepted overload request failed; usage={usage_detail}"
        else:
            ok, detail = False, f"unexpected overload status {result.status}"
        ok = ok and overall_ok
        if not overall_ok:
            detail = f"status distribution {dict(statuses)}, expected 4x200 + 4x429; {detail}"
        record_result(writer, "overload", result, ok, detail, index=index,
                      statuses={str(key): value for key, value in statuses.items()})
    return results


def metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        field: int(after.get(field) or 0) - int(before.get(field) or 0)
        for field in ("admitted", "completed", "tokens_out", "prompt_tokens_in", "cached_tokens_in")
    }


def expected_metrics(results: list[HttpResult]) -> dict[str, int]:
    successful = [result for result in results if result.status == 200]
    usages = [usage_from(result) for result in successful]
    return {
        "admitted": len(successful),
        "completed": len(successful),
        "tokens_out": sum(int(usage.get("completion_tokens") or 0) for usage in usages),
        "prompt_tokens_in": sum(int(usage.get("prompt_tokens") or 0) for usage in usages),
        "cached_tokens_in": sum(cached_tokens(usage) for usage in usages),
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(root: pathlib.Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "MANIFEST.sha256"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    manifest = "\n".join(rows) + "\n"
    (root / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    return hashlib.sha256(manifest.encode()).hexdigest()


def run(args: argparse.Namespace) -> int:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    api_key = pathlib.Path(args.api_key_file).read_text(encoding="utf-8").strip()
    metrics_token = pathlib.Path(args.metrics_token_file).read_text(encoding="utf-8").strip()
    client = Client(args.base_url, api_key, args.timeout)
    metrics_client = Client(args.base_url, metrics_token, args.timeout)
    writer = ReceiptWriter(out / "requests.jsonl")
    started = time.monotonic()
    started_utc = utc_now()
    results: list[HttpResult] = []
    verdict = "FAIL"
    failure: str | None = None
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    expected: dict[str, int] = {}
    actual: dict[str, int] = {}
    try:
        catalog = client.get_json("/v1/models")
        ids = [item.get("id") for item in (catalog.body.get("data") or [])] if isinstance(catalog.body, dict) else []
        catalog_ok = catalog.status == 200 and args.model in ids
        record_result(writer, "catalog", catalog, catalog_ok,
                      "ok" if catalog_ok else f"model absent from catalog: {ids}")

        unauthorized = Client(args.base_url, "", args.timeout).post_json(
            "/v1/chat/completions", chat_payload(args.model, "auth probe", False, 1)
        )
        unauthorized_body = unauthorized.body if isinstance(unauthorized.body, dict) else {}
        auth_error = unauthorized_body.get("error") or {}
        auth_ok = unauthorized.status == 401 and auth_error.get("type") == "authentication_error"
        record_result(writer, "auth_required", unauthorized, auth_ok,
                      "ok" if auth_ok else "missing bearer was not a clean 401")

        before_result = metrics_client.get_json("/metrics", authenticated=True)
        if before_result.status != 200 or not isinstance(before_result.body, dict):
            raise GateFailure(f"operator metrics baseline failed: {before_result.status}")
        before = before_result.body
        write_json(out / "metrics-before.json", before)

        for stream in (False, True):
            results.append(run_plain(client, writer, args.model, stream))
        for stream in (False, True):
            results.append(run_tool(client, writer, args.model, stream))
        for stream in (False, True):
            results.append(run_structured(client, writer, args.model, stream))
        results.extend(run_cache(client, writer, args.model))
        results.extend(run_overload(client, writer, args.model))

        expected = expected_metrics(results)
        deadline = time.monotonic() + 10
        while True:
            after_result = metrics_client.get_json("/metrics", authenticated=True)
            if after_result.status != 200 or not isinstance(after_result.body, dict):
                raise GateFailure(f"operator metrics final scrape failed: {after_result.status}")
            after = after_result.body
            actual = metrics_delta(before, after)
            if actual["completed"] >= expected["completed"] or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        write_json(out / "metrics-after.json", after)
        metrics_ok = actual == expected
        writer.write(
            "usage_reconciliation",
            metrics_ok,
            "ok" if metrics_ok else "client usage does not equal engine metric deltas",
            expected=expected,
            actual=actual,
        )
        if not metrics_ok:
            raise GateFailure(f"usage reconciliation mismatch: expected={expected}, actual={actual}")

        tenant_rows = after.get("tenants") or {}
        tenant = tenant_rows.get(f"t:{args.tenant}") or tenant_rows.get(args.tenant)
        tenant_ok = isinstance(tenant, dict)
        writer.write(
            "tenant_meter",
            tenant_ok,
            "ok" if tenant_ok else f"tenant {args.tenant!r} absent from operator metrics",
            tenant=tenant,
        )
        if not tenant_ok:
            raise GateFailure("tenant metering row is absent")
        verdict = "PASS"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        writer.close()
        summary = {
            "schema": "memra.cx-servetest.public-gate.v1",
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "duration_s": round(time.monotonic() - started, 3),
            "base_url": args.base_url,
            "model": args.model,
            "tenant": args.tenant,
            "checks": len(writer.rows),
            "failed_checks": sum(not row["ok"] for row in writer.rows),
            "expected_metrics_delta": expected,
            "actual_metrics_delta": actual,
            "verdict": verdict,
            "failure": failure,
        }
        write_json(out / "summary.json", summary)
        manifest_hash = write_manifest(out)
        print(json.dumps({**summary, "manifest_sha256": manifest_hash}, sort_keys=True))
    return 0 if verdict == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tenant", default="servetest")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--metrics-token-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
