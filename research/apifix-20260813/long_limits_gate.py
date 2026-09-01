#!/usr/bin/env python3
"""Content-free receipts for field-length prompt, output, and KV admission gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any


FIELD_TOP = 262_144
LONG_GENERATION = 131_072
LONG_GENERATION_PROMPT_REPETITIONS = 256


class GateFailure(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclasses.dataclass
class HttpResult:
    status: int | None
    headers: dict[str, str]
    body: Any
    elapsed_s: float
    error: str | None = None

    @property
    def request_id(self) -> str | None:
        return self.headers.get("x-request-id")


class Client:
    def __init__(self, base_url: str, bearer: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.bearer = bearer
        self.timeout = timeout

    def request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> HttpResult:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        headers = {"Accept": "application/json", "User-Agent": "memra-cx-apifix/1"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.bearer:
            headers["Authorization"] = f"Bearer {self.bearer}"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                body = json.loads(raw) if raw else None
                return HttpResult(
                    response.status,
                    {key.lower(): value for key, value in response.headers.items()},
                    body,
                    time.monotonic() - started,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                body: Any = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw.decode(errors="replace")
            return HttpResult(
                exc.code,
                {key.lower(): value for key, value in exc.headers.items()},
                body,
                time.monotonic() - started,
            )
        except Exception as exc:
            return HttpResult(
                None,
                {},
                None,
                time.monotonic() - started,
                f"{type(exc).__name__}: {exc}",
            )

    def post_sse(self, path: str, payload: dict[str, Any]) -> HttpResult:
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.bearer}",
            "Content-Type": "application/json",
            "User-Agent": "memra-cx-apifix/1",
        }
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        output_hash = hashlib.sha256()
        output_bytes = 0
        token_events = 0
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        done = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
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
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
                            continue
                        piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                        encoded = str(piece).encode()
                        output_hash.update(encoded)
                        output_bytes += len(encoded)
                        token_events += 1
                return HttpResult(
                    response.status,
                    response_headers,
                    {
                        "done": done,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "token_events": token_events,
                        "output_bytes": output_bytes,
                        "output_sha256": output_hash.hexdigest(),
                    },
                    time.monotonic() - started,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                body: Any = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = raw.decode(errors="replace")
            return HttpResult(
                exc.code,
                {key.lower(): value for key, value in exc.headers.items()},
                body,
                time.monotonic() - started,
            )
        except Exception as exc:
            return HttpResult(
                None,
                {},
                {
                    "done": done,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "token_events": token_events,
                    "output_bytes": output_bytes,
                    "output_sha256": output_hash.hexdigest(),
                },
                time.monotonic() - started,
                f"{type(exc).__name__}: {exc}",
            )


class Receipts:
    def __init__(self, out: pathlib.Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.path = out / "requests.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.rows: list[dict[str, Any]] = []

    def record(self, check: str, ok: bool, **fields: Any) -> None:
        row = {"check": check, "ok": ok, "utc": utc_now(), **fields}
        self.file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.file.flush()
        self.rows.append(row)
        if not ok:
            raise GateFailure(f"{check}: {fields.get('detail', 'failed')}")

    def close(self) -> None:
        self.file.close()


def result_fields(result: HttpResult) -> dict[str, Any]:
    body = result.body if isinstance(result.body, dict) else {}
    error = body.get("error") if isinstance(body, dict) else None
    return {
        "http_status": result.status,
        "request_id": result.request_id,
        "elapsed_s": round(result.elapsed_s, 3),
        "client_error": result.error,
        "api_error": error,
    }


def typed_values(items: Any) -> dict[str, int]:
    return {
        str(item["type"]): int(item["value"])
        for item in (items or [])
        if isinstance(item, dict) and "type" in item and "value" in item
    }


def typed_prices(items: Any) -> dict[str, str]:
    return {
        str(item["type"]): str(item["cost_usd"])
        for item in (items or [])
        if isinstance(item, dict) and "type" in item and "cost_usd" in item
    }


def check_metadata(client: Client, receipts: Receipts, model: str) -> None:
    result = client.request_json("GET", "/models?schema=openrouter")
    data = result.body.get("data") if isinstance(result.body, dict) else None
    entries = {entry.get("id"): entry for entry in (data or []) if isinstance(entry, dict)}
    entry = entries.get(model) or {}
    input_side = (entry.get("input_modalities") or [{}])[0]
    output_side = (entry.get("output_modalities") or [{}])[0]
    supported = input_side.get("supported_inputs") or {}
    input_capacity = typed_values(input_side.get("capacity"))
    output_capacity = typed_values(output_side.get("capacity"))
    request_capacity = typed_values(entry.get("capacity"))
    input_prices = typed_prices(input_side.get("pricing"))
    output_prices = typed_prices(output_side.get("pricing"))
    ok = (
        result.status == 200
        and set(entries) == {model}
        and supported.get("max_context_length", {}).get("value") == FIELD_TOP
        and supported.get("max_prompt_length", {}).get("value") == FIELD_TOP
        and output_side.get("max_length", {}).get("value") == FIELD_TOP
        and output_side.get("supported_parameters", {}).get("max_tokens", {}).get("max")
        == FIELD_TOP
        and input_capacity == {"prompt": 700_000, "cached_prompt": 630_000}
        and output_capacity == {"completion": 8_600, "concurrency": 4}
        and request_capacity == {"request": 140}
        and input_prices
        == {"prompt": "0.0000000931", "cached_prompt": "0.0000000652"}
        and output_prices == {"completion": "0.0000009025"}
    )
    receipts.record(
        "metadata",
        ok,
        detail="ok" if ok else "active roster, limits, capacity, or prices differ",
        active_models=sorted(entries),
        max_context=supported.get("max_context_length", {}).get("value"),
        max_prompt=supported.get("max_prompt_length", {}).get("value"),
        max_output=output_side.get("max_length", {}).get("value"),
        input_capacity=input_capacity,
        output_capacity=output_capacity,
        request_capacity=request_capacity,
        input_prices=input_prices,
        output_prices=output_prices,
        **result_fields(result),
    )


def check_limits(client: Client, receipts: Receipts, model: str) -> None:
    output = client.request_json(
        "POST",
        "/v1/completions",
        {"model": model, "prompt_ids": [5_000], "max_tokens": FIELD_TOP + 1},
    )
    output_error = output.body.get("error") if isinstance(output.body, dict) else {}
    output_ok = (
        output.status == 400
        and output_error.get("param") == "max_tokens"
        and output_error.get("type") == "invalid_request_error"
        and output_error.get("code") is None
    )
    receipts.record(
        "output_over_limit",
        output_ok,
        detail="ok" if output_ok else "262145 output tokens did not fail as a clean 400",
        **result_fields(output),
    )

    prompt = [5_000 + (index % 1_024) for index in range(FIELD_TOP + 1)]
    too_long = client.request_json(
        "POST",
        "/v1/completions",
        {
            "model": model,
            "prompt_ids": prompt,
            "max_tokens": 1,
            "max_ctx": FIELD_TOP,
        },
    )
    prompt_error = too_long.body.get("error") if isinstance(too_long.body, dict) else {}
    prompt_ok = (
        too_long.status == 400
        and prompt_error.get("type") == "invalid_request_error"
        and prompt_error.get("code") == "context_length_exceeded"
    )
    receipts.record(
        "prompt_over_limit",
        prompt_ok,
        detail="ok" if prompt_ok else "262145 prompt tokens did not fail as a clean 400",
        prompt_tokens=len(prompt),
        **result_fields(too_long),
    )


def check_long_prompt(client: Client, receipts: Receipts, model: str) -> None:
    # On the pinned Q35 tokenizer, each leading-space "a" is one token. Using real UTF-8 text
    # exercises HTTP parsing and the production tokenizer instead of relying only on memra's
    # exact-token validation extension. Usage is the authoritative post-tokenization count.
    prompt = " a" * (FIELD_TOP - 1)
    result = client.request_json(
        "POST",
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "max_ctx": FIELD_TOP,
            "temperature": 0,
            "seed": 34_071,
            "stream": False,
            "cache_salt": f"apifix-long-prompt-{time.time_ns()}",
        },
    )
    usage = result.body.get("usage") if isinstance(result.body, dict) else {}
    ok = (
        result.status == 200
        and result.error is None
        and usage.get("prompt_tokens") == FIELD_TOP - 1
        and usage.get("completion_tokens") == 1
        and usage.get("total_tokens") == FIELD_TOP
    )
    receipts.record(
        "long_prompt",
        ok,
        detail="ok" if ok else "text prompt did not tokenize to 262143 and complete end to end",
        expected_prompt_tokens=FIELD_TOP - 1,
        prompt_repetitions=FIELD_TOP - 1,
        prompt_bytes=len(prompt.encode()),
        prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        usage=usage,
        **result_fields(result),
    )


def check_long_generation(
    client: Client,
    receipts: Receipts,
    model: str,
    generation_tokens: int,
) -> None:
    # This deliberately uses the ordinary unconstrained completion path. A repeated, leading-
    # space "a" prompt is a stable greedy continuation for the pinned Q35 artifact and reaches
    # the requested length without an EOS override. That makes this an engine/context test, not
    # a constrained-decoder stress test. The 2026-08-13 negative control that tried to force a
    # billion-character JSON string is retained separately: llguidance reached its 250k lexer-
    # state guard after 13,085 tokens, before the memra generation/context limit was involved.
    prompt = " a" * LONG_GENERATION_PROMPT_REPETITIONS
    result = client.post_sse(
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": generation_tokens,
            "max_ctx": FIELD_TOP,
            "temperature": 0,
            "seed": 34_071,
            "stream": True,
            "cache_salt": f"apifix-long-generation-{time.time_ns()}",
        },
    )
    body = result.body if isinstance(result.body, dict) else {}
    usage = body.get("usage") or {}
    ok = (
        result.status == 200
        and result.error is None
        and body.get("done") is True
        and body.get("finish_reason") == "length"
        and result.token_events == generation_tokens
        and usage.get("completion_tokens") == generation_tokens
        and usage.get("total_tokens")
        == usage.get("prompt_tokens", -generation_tokens) + generation_tokens
    )
    receipts.record(
        "long_generation",
        ok,
        detail=(
            "ok"
            if ok
            else f"{generation_tokens}-token unconstrained generation did not finish by length"
        ),
        requested_generation_tokens=generation_tokens,
        prompt_repetitions=LONG_GENERATION_PROMPT_REPETITIONS,
        prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        usage=usage,
        finish_reason=body.get("finish_reason"),
        done=body.get("done"),
        token_events=body.get("token_events"),
        output_bytes=body.get("output_bytes"),
        output_sha256=body.get("output_sha256"),
        **result_fields(result),
    )


def check_admission(
    client: Client,
    metrics: Client,
    receipts: Receipts,
    model: str,
    concurrency: int,
) -> None:
    before = metrics.request_json("GET", "/metrics")
    if before.status != 200 or not isinstance(before.body, dict):
        raise GateFailure(f"operator metrics baseline failed: {before.status}, {before.error}")
    barrier = threading.Barrier(concurrency)
    prompt = [5_000 + (index % 1_024) for index in range(4_860)]

    def one(index: int) -> HttpResult:
        barrier.wait(timeout=30)
        return client.request_json(
            "POST",
            "/v1/completions",
            {
                "model": model,
                "prompt_ids": prompt,
                "max_tokens": 64,
                "max_ctx": FIELD_TOP,
                "temperature": 0,
                "seed": 3_407,
                "stream": False,
                "cache_salt": f"apifix-kv-{time.time_ns()}-{index}",
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(one, range(concurrency)))
    after = metrics.request_json("GET", "/metrics")
    health = client.request_json("GET", "/livez")
    if after.status != 200 or not isinstance(after.body, dict):
        raise GateFailure(f"operator metrics final scrape failed: {after.status}, {after.error}")
    statuses = Counter(result.status for result in results)
    before_defers = int(before.body.get("admission_vram_defers") or 0)
    after_defers = int(after.body.get("admission_vram_defers") or 0)
    defer_delta = after_defers - before_defers
    clean = True
    for result in results:
        if result.status == 200:
            usage = result.body.get("usage") if isinstance(result.body, dict) else {}
            clean = clean and usage.get("prompt_tokens") == len(prompt)
            clean = clean and usage.get("completion_tokens") == 64
        elif result.status == 429:
            error = result.body.get("error") if isinstance(result.body, dict) else {}
            clean = clean and error.get("type") == "rate_limit_error"
            clean = clean and error.get("code") == "rate_limit_exceeded"
        else:
            clean = False
    ok = (
        clean
        and statuses.get(200, 0) + statuses.get(429, 0) == concurrency
        and statuses.get(200, 0) >= 1
        and (defer_delta > 0 or statuses.get(429, 0) > 0)
        and health.status == 200
    )
    receipts.record(
        "long_context_admission",
        ok,
        detail="ok" if ok else "burst did not queue/429 cleanly under measured KV pressure",
        concurrency=concurrency,
        request_max_ctx=FIELD_TOP,
        statuses={str(key): value for key, value in sorted(statuses.items(), key=lambda item: str(item[0]))},
        admission_vram_defers_delta=defer_delta,
        server_live_status=health.status,
        requests=[result_fields(result) for result in results],
    )


def write_manifest(out: pathlib.Path) -> str:
    rows = []
    for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "MANIFEST.sha256"):
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    text = "\n".join(rows) + "\n"
    (out / "MANIFEST.sha256").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--metrics-token-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checks", default="metadata,limits,long-prompt,long-generation,admission")
    parser.add_argument("--admission-concurrency", type=int, default=8)
    parser.add_argument("--long-generation-tokens", type=int, default=LONG_GENERATION)
    parser.add_argument("--timeout", type=float, default=3_600)
    args = parser.parse_args()

    selected = [item.strip() for item in args.checks.split(",") if item.strip()]
    known = {"metadata", "limits", "long-prompt", "long-generation", "admission"}
    unknown = set(selected) - known
    if unknown:
        parser.error(f"unknown checks: {sorted(unknown)}")

    api_key = pathlib.Path(args.api_key_file).read_text(encoding="utf-8").strip()
    metrics_token = pathlib.Path(args.metrics_token_file).read_text(encoding="utf-8").strip()
    out = pathlib.Path(args.out)
    receipts = Receipts(out)
    client = Client(args.base_url, api_key, args.timeout)
    metrics = Client(args.base_url, metrics_token, args.timeout)
    started = time.monotonic()
    started_utc = utc_now()
    failure: str | None = None
    verdict = "FAIL"
    try:
        for check in selected:
            if check == "metadata":
                check_metadata(client, receipts, args.model)
            elif check == "limits":
                check_limits(client, receipts, args.model)
            elif check == "long-prompt":
                check_long_prompt(client, receipts, args.model)
            elif check == "long-generation":
                check_long_generation(
                    client,
                    receipts,
                    args.model,
                    args.long_generation_tokens,
                )
            elif check == "admission":
                check_admission(
                    client,
                    metrics,
                    receipts,
                    args.model,
                    args.admission_concurrency,
                )
        verdict = "PASS"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        receipts.close()
        summary = {
            "schema": "memra.cx-apifix.long-limits.v1",
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "duration_s": round(time.monotonic() - started, 3),
            "base_url": args.base_url,
            "model": args.model,
            "selected_checks": selected,
            "checks": len(receipts.rows),
            "failed_checks": sum(not row["ok"] for row in receipts.rows),
            "verdict": verdict,
            "failure": failure,
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = write_manifest(out)
        print(json.dumps({**summary, "manifest_sha256": manifest}, sort_keys=True))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
