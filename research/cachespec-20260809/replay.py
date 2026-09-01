#!/usr/bin/env python3
"""Agent-shaped cache/spec replay with per-request latency and metrics receipts.

Record mode drives one growing conversation, strips the model's reasoning before the
next turn (the pi history-rewrite shape), then launches four branches concurrently and
finally returns to the main conversation. Replay mode issues the exact recorded prompts
against another server arm.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

CUMULATIVE_METRICS = (
    "admitted",
    "completed",
    "tokens_out",
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_hit_tokens",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
    "continuation_pool_hits",
    "continuation_pool_evictions",
    "spec_pool_hits",
    "spec_pool_misses",
    "spec_pool_affinity_rewinds",
    "spec_pool_evictions",
)

GAUGE_METRICS = (
    "prefix_cache_entries",
    "prefix_cache_bytes",
    "active_sessions",
    "queued_requests",
    "continuation_pool_entries",
    "spec_pool_entries",
    "cuda_driver_free_bytes",
    "cuda_pool_reserved_bytes",
    "cuda_pool_used_bytes",
    "cuda_pool_cached_bytes",
)

REQUIRE_HARDENING_METRICS = (
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
    "spec_pool_hits",
    "spec_pool_misses",
    "spec_pool_evictions",
    "active_sessions",
    "queued_requests",
    "spec_pool_entries",
    "cuda_pool_cached_bytes",
)

SYSTEM_BASE = """You are editing a production single-page operations dashboard. Return a
complete, self-contained HTML document on every turn. Preserve working behavior from the
previous document, make the requested change, use semantic HTML and accessible labels, and
include CSS and JavaScript inline. Think briefly, then output the document without commentary.

The dashboard models an inference service. It has a request timeline, cache hit-rate chart,
admission queue, GPU-memory gauge, incident annotations, and a compact mobile layout. The
following operating notes are source material and must remain represented in the UI:
"""

NOTE = (
    "Request {i} records its arrival time, time to first token, generated-token rate, "
    "cache source, queue delay, and completion state. The timeline must make rising latency "
    "visually distinguishable from slower decoding. Cache counters are cumulative while "
    "pool occupancy and active sessions are gauges. Evidence rows retain units and exact "
    "timestamps. Operators can filter by session and compare speculative and plain arms.\n"
)

TURN_REQUESTS = (
    "Create the initial dashboard with all required panels and realistic placeholder rows.",
    "Add a sticky request inspector that explains TTFT separately from decode time.",
    "Add an accessible cache-tier legend for continuation, speculative, and prefix reuse.",
    "Add a compact admission-pressure strip with defer, park, active, and queued values.",
    "Add a responsive incident timeline and emphasize the first counter correlated with a slope.",
    "Add keyboard navigation and visible focus treatment throughout the dashboard.",
    "Add a comparison drawer for spec-default versus spec-off measurements.",
    "Add a pool-residency section with driver-free, reserved, used, and cached byte gauges.",
    "Add an eviction-thrash warning state and a short operator remediation checklist.",
    "Add a print stylesheet that preserves the request receipt as a legible table.",
    "Add a small sparkline beside every request without introducing external dependencies.",
    "Finish the dashboard with a concise evidence footer and no marketing language.",
)

BURST_REQUESTS = (
    "Branch A: add a modal showing the raw metrics delta for one request.",
    "Branch B: add a dense four-column view for concurrent request comparison.",
    "Branch C: add an alert when cached input unexpectedly becomes zero.",
    "Branch D: add an annotation for a pool-cap eviction during a burst.",
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def render(history: list[tuple[str, str]], system: str) -> str:
    parts = [f"{IM_START}system\n{system}{IM_END}\n"]
    for role, content in history:
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    # Step-3.7's generation prompt opens the reasoning section. A later turn stores only
    # text after </think>, reproducing the client's rewritten-history seam.
    parts.append(f"{IM_START}assistant\n<think>\n")
    return "".join(parts)


def strip_reasoning(text: str) -> tuple[str, bool]:
    close = text.find("</think>")
    if close >= 0:
        return text[close + len("</think>") :].lstrip(), True
    open_at = text.find("<think>")
    if open_at >= 0:
        close = text.find("</think>", open_at + len("<think>"))
        if close >= 0:
            return (
                text[:open_at] + text[close + len("</think>") :]
            ).lstrip(), True
    return text, False


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "memra-cachespec-replay/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_json(url: str, api_key: str | None, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=request_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def scrape_metrics(args) -> dict:
    return fetch_json(args.base.rstrip("/") + "/metrics", args.api_key, args.timeout)


def validate_metrics(metrics: dict) -> None:
    missing = [name for name in REQUIRE_HARDENING_METRICS if name not in metrics]
    if missing:
        raise RuntimeError("server lacks cachespec metrics: " + ", ".join(missing))


def number(metrics: dict, key: str) -> int | float:
    value = metrics.get(key, 0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return 0


def metrics_receipt(before: dict, after: dict) -> dict:
    return {
        "delta": {
            key: number(after, key) - number(before, key)
            for key in CUMULATIVE_METRICS
        },
        "after": {key: number(after, key) for key in GAUGE_METRICS},
    }


def wait_for_completed(args, before: dict, count: int) -> dict:
    target = int(number(before, "completed")) + count
    deadline = time.monotonic() + 30.0
    last = before
    while time.monotonic() < deadline:
        last = scrape_metrics(args)
        if int(number(last, "completed")) >= target:
            return last
        time.sleep(0.05)
    raise RuntimeError(
        f"metrics did not publish completed={target}; last={number(last, 'completed')}"
    )


def read_stream(response, started_ns: int) -> dict:
    pieces: list[str] = []
    first_ns = None
    last_text_ns = None
    usage: dict = {}
    finish_reason = None
    request_id = None
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        event = json.loads(body)
        if event.get("error"):
            raise RuntimeError(json.dumps(event["error"], sort_keys=True))
        request_id = event.get("id") or request_id
        event_usage = event.get("usage")
        if isinstance(event_usage, dict) and event_usage:
            usage = event_usage
        for choice in event.get("choices") or []:
            piece = choice.get("text") or ""
            if piece:
                now_ns = time.perf_counter_ns()
                first_ns = first_ns or now_ns
                last_text_ns = now_ns
                pieces.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    ended_ns = time.perf_counter_ns()
    if first_ns is None:
        raise RuntimeError("stream completed without a visible text chunk")
    completion_tokens = int(usage.get("completion_tokens") or 0)
    decode_s = max(0.0, ((last_text_ns or ended_ns) - first_ns) / 1e9)
    decode_tps = None
    if completion_tokens > 1 and decode_s > 0:
        decode_tps = (completion_tokens - 1) / decode_s
    return {
        "request_id": request_id,
        "text": "".join(pieces),
        "usage": usage,
        "finish_reason": finish_reason,
        "ttft_s": (first_ns - started_ns) / 1e9,
        "wall_s": (ended_ns - started_ns) / 1e9,
        "decode_s_after_first": decode_s,
        "decode_tok_s_after_first": decode_tps,
    }


def stream_request(args, prompt: str, session_id: str, request_index: int) -> dict:
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": args.cache_salt,
        "session_id": session_id,
    }
    req = urllib.request.Request(
        args.base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=request_headers(args.api_key),
    )
    started_at = utc_now()
    started_ns = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            result = read_stream(response, started_ns)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:4000]
        raise RuntimeError(
            f"HTTP {error.code}: {detail}; request_index={request_index} "
            f"prompt_chars={len(prompt)} session_id={session_id!r}"
        ) from error
    result["started_at"] = started_at
    result["request_index"] = request_index
    result["session_id"] = session_id
    result["prompt_chars"] = len(prompt)
    result["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    return result


def usage_fields(result: dict) -> dict:
    usage = result.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", usage.get("cached_tokens")),
        "completion_tokens": usage.get("completion_tokens"),
    }


def write_response(raw_dir: pathlib.Path, label: str, result: dict) -> None:
    payload = {
        "request_id": result.get("request_id"),
        "finish_reason": result.get("finish_reason"),
        "usage": result.get("usage"),
        "text": result.get("text"),
    }
    (raw_dir / f"{label}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


def row_from_result(result: dict, phase: str, phase_index: int, metrics: dict) -> dict:
    text = result["text"]
    return {
        "type": "request",
        "phase": phase,
        "phase_index": phase_index,
        "request_index": result["request_index"],
        "session_id": result["session_id"],
        "started_at": result["started_at"],
        "prompt_chars": result["prompt_chars"],
        "prompt_sha256": result["prompt_sha256"],
        **usage_fields(result),
        "ttft_s": round(result["ttft_s"], 6),
        "decode_s_after_first": round(result["decode_s_after_first"], 6),
        "decode_tok_s_after_first": (
            None
            if result["decode_tok_s_after_first"] is None
            else round(result["decode_tok_s_after_first"], 6)
        ),
        "wall_s": round(result["wall_s"], 6),
        "finish_reason": result["finish_reason"],
        "text_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "metrics": metrics,
    }


def append_row(path: pathlib.Path, row: dict) -> None:
    with path.open("a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    compact = {k: v for k, v in row.items() if k not in {"metrics"}}
    print(json.dumps(compact, sort_keys=True), flush=True)


def issue_sequential(args, prompt: str, session_id: str, request_index: int) -> tuple[dict, dict]:
    before = scrape_metrics(args)
    result = stream_request(args, prompt, session_id, request_index)
    after = wait_for_completed(args, before, 1)
    return result, metrics_receipt(before, after)


def build_system(base_notes: int) -> str:
    return SYSTEM_BASE + "".join(NOTE.format(i=i + 1) for i in range(base_notes))


def make_recorded_workload(args, out_path: pathlib.Path, raw_dir: pathlib.Path) -> dict:
    system = build_system(args.base_notes)
    history: list[tuple[str, str]] = []
    requests = []
    next_index = 0
    for turn in range(args.sequential):
        user = TURN_REQUESTS[turn % len(TURN_REQUESTS)]
        history.append(("user", f"Iteration {turn + 1}. {user}"))
        prompt = render(history, system)
        result, receipt = issue_sequential(
            args, prompt, args.session_id, next_index
        )
        write_response(raw_dir, f"sequential-{turn:03d}", result)
        kept, rewrote = strip_reasoning(result["text"])
        row = row_from_result(result, "sequential", turn, receipt)
        row["history_rewritten"] = rewrote
        row["kept_text_sha256"] = hashlib.sha256(kept.encode()).hexdigest()
        append_row(out_path, row)
        requests.append(
            {
                "phase": "sequential",
                "phase_index": turn,
                "request_index": next_index,
                "session_id": args.session_id,
                "prompt": prompt,
            }
        )
        history.append(("assistant", kept))
        next_index += 1

    burst_requests = []
    for branch in range(args.concurrency):
        user = BURST_REQUESTS[branch % len(BURST_REQUESTS)]
        branch_history = history + [("user", user)]
        burst_requests.append(
            {
                "phase": "burst",
                "phase_index": branch,
                "request_index": next_index + branch,
                "session_id": f"{args.session_id}-burst-{branch}",
                "prompt": render(branch_history, system),
            }
        )
    run_burst(args, burst_requests, out_path, raw_dir)
    requests.extend(burst_requests)
    next_index += len(burst_requests)

    history.append(("user", "Return to the main branch. Audit the page for regressions and emit the complete final HTML."))
    prompt = render(history, system)
    result, receipt = issue_sequential(args, prompt, args.session_id, next_index)
    write_response(raw_dir, "postburst-000", result)
    row = row_from_result(result, "postburst", 0, receipt)
    append_row(out_path, row)
    requests.append(
        {
            "phase": "postburst",
            "phase_index": 0,
            "request_index": next_index,
            "session_id": args.session_id,
            "prompt": prompt,
        }
    )
    return {
        "schema": 1,
        "created_at": utc_now(),
        "model": args.model,
        "max_tokens": args.max_tokens,
        "cache_salt": args.cache_salt,
        "sequential": args.sequential,
        "concurrency": args.concurrency,
        "requests": requests,
    }


def sample_metrics(args, stop: threading.Event, sample_path: pathlib.Path, t0: float) -> None:
    while not stop.is_set():
        try:
            metrics = scrape_metrics(args)
            row = {
                "at_s": round(time.monotonic() - t0, 6),
                "wall_at": utc_now(),
                **{key: number(metrics, key) for key in CUMULATIVE_METRICS + GAUGE_METRICS},
            }
            with sample_path.open("a") as output:
                output.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception as error:  # retained in the raw sampler stream
            with sample_path.open("a") as output:
                output.write(json.dumps({"at_s": time.monotonic() - t0, "error": str(error)}) + "\n")
        stop.wait(0.05)


def run_burst(args, requests: list[dict], out_path: pathlib.Path, raw_dir: pathlib.Path) -> None:
    before = scrape_metrics(args)
    ready = threading.Barrier(len(requests) + 1)

    def one(item: dict) -> dict:
        ready.wait()
        return stream_request(
            args, item["prompt"], item["session_id"], item["request_index"]
        )

    stop = threading.Event()
    t0 = time.monotonic()
    sample_path = raw_dir / "burst-metrics-samples.jsonl"
    sampler = threading.Thread(
        target=sample_metrics, args=(args, stop, sample_path, t0), daemon=True
    )
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [pool.submit(one, item) for item in requests]
            ready.wait()
            results = [future.result(timeout=args.timeout + 30) for future in futures]
    finally:
        stop.set()
        sampler.join(timeout=5)
    after = wait_for_completed(args, before, len(requests))
    receipt = metrics_receipt(before, after)
    for item, result in zip(requests, results):
        label = f"burst-{item['phase_index']:03d}"
        write_response(raw_dir, label, result)
        row = row_from_result(
            result, "burst", item["phase_index"],
            {"scope": "shared-c4", **receipt},
        )
        append_row(out_path, row)
    append_row(
        out_path,
        {
            "type": "burst_summary",
            "phase": "burst",
            "concurrency": len(requests),
            "metrics": receipt,
        },
    )


def replay_workload(args, workload: dict, out_path: pathlib.Path, raw_dir: pathlib.Path) -> None:
    sequential = [r for r in workload["requests"] if r["phase"] == "sequential"]
    burst = [r for r in workload["requests"] if r["phase"] == "burst"]
    postburst = [r for r in workload["requests"] if r["phase"] == "postburst"]
    for item in sequential:
        result, receipt = issue_sequential(
            args, item["prompt"], item["session_id"], item["request_index"]
        )
        write_response(raw_dir, f"sequential-{item['phase_index']:03d}", result)
        row = row_from_result(result, "sequential", item["phase_index"], receipt)
        append_row(out_path, row)
    run_burst(args, burst, out_path, raw_dir)
    for item in postburst:
        result, receipt = issue_sequential(
            args, item["prompt"], item["session_id"], item["request_index"]
        )
        write_response(raw_dir, f"postburst-{item['phase_index']:03d}", result)
        row = row_from_result(result, "postburst", item["phase_index"], receipt)
        append_row(out_path, row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--api-key")
    auth.add_argument("--api-key-file", type=pathlib.Path)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--workload", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sequential", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--base-notes", type=int, default=80)
    parser.add_argument("--session-id", default="cachespec-main")
    parser.add_argument("--cache-salt", default="cachespec-20260809")
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if args.api_key_file is not None:
        args.api_key = args.api_key_file.read_text().strip()
        if not args.api_key:
            parser.error(f"API key file is empty: {args.api_key_file}")
    return args


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    initial = scrape_metrics(args)
    validate_metrics(initial)
    (args.raw_dir / "metrics-initial.json").write_text(
        json.dumps(initial, indent=2, sort_keys=True) + "\n"
    )
    if args.mode == "record":
        workload = make_recorded_workload(args, args.out, args.raw_dir)
        args.workload.write_text(
            json.dumps(workload, indent=2, ensure_ascii=False) + "\n"
        )
    else:
        workload = json.loads(args.workload.read_text())
        if workload.get("max_tokens") != args.max_tokens:
            raise RuntimeError(
                f"workload max_tokens={workload.get('max_tokens')} != CLI {args.max_tokens}"
            )
        replay_workload(args, workload, args.out, args.raw_dir)
    final = scrape_metrics(args)
    (args.raw_dir / "metrics-final.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    print(f"REPLAY_DONE rows={args.out} workload={args.workload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
