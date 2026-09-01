#!/usr/bin/env python3
"""Different-prefix concurrent-prefill receipt with a live decode background."""

import argparse
import concurrent.futures
import datetime
import json
import math
import statistics
import threading
import time
import urllib.request


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def request_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def prompt_ids(serial, request_index, n_tokens):
    seed = 2000 + serial * 17 + request_index
    return [seed] + [3000 + ((seed * 31 + i * 17) % 20000)
                     for i in range(n_tokens - 1)]


def stream_completion(base, body, api_key, timeout, on_visible=None, stop=None):
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=request_headers(api_key),
    )
    request_id = None
    usage = {}
    visible = 0
    first_visible = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if stop is not None and stop.is_set():
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(json.dumps(event["error"], sort_keys=True))
            request_id = event.get("id") or request_id
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    now = time.monotonic()
                    visible += 1
                    first_visible = first_visible or now
                    if on_visible is not None:
                        on_visible(now)
    return request_id, usage, visible, first_visible


class BackgroundLoad:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.ready = [threading.Event() for _ in range(args.background)]
        self.events = []
        self.errors = []
        self.restarts = [0] * args.background
        self.lock = threading.Lock()
        self.pool = None
        self.futures = []

    def _record(self, worker, timestamp):
        with self.lock:
            self.events.append((worker, timestamp))
        self.ready[worker].set()

    def _worker(self, worker):
        prompt = prompt_ids(9000, worker, 128)
        while not self.stop.is_set():
            body = {
                "model": self.args.model,
                "prompt_ids": prompt,
                "max_ctx": 16384,
                "max_tokens": 8192,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache_salt": f"{self.args.label}-background-{worker}",
            }
            try:
                stream_completion(
                    self.args.base,
                    body,
                    self.args.api_key,
                    self.args.timeout,
                    on_visible=lambda ts, w=worker: self._record(w, ts),
                    stop=self.stop,
                )
                self.restarts[worker] += 1
            except Exception as error:
                if self.stop.is_set():
                    break
                with self.lock:
                    self.errors.append(f"worker {worker}: {error}")
                time.sleep(0.1)

    def start(self):
        if not self.ready:
            return
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.ready))
        self.futures = [
            self.pool.submit(self._worker, worker)
            for worker in range(len(self.ready))
        ]
        deadline = time.monotonic() + self.args.timeout
        for worker, ready in enumerate(self.ready):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not ready.wait(remaining):
                raise RuntimeError(f"background worker {worker} did not emit a token")

    def summary(self, started, ended):
        with self.lock:
            events = [(worker, ts) for worker, ts in self.events
                      if started <= ts <= ended]
            errors = list(self.errors)
        duration = max(ended - started, 1e-9)
        per_worker = {}
        for worker, timestamp in events:
            per_worker.setdefault(worker, []).append(timestamp)
        gaps = []
        for timestamps in per_worker.values():
            timestamps.sort()
            gaps.extend((b - a) * 1000.0 for a, b in zip(timestamps, timestamps[1:]))
        return {
            "background_concurrency": len(self.ready),
            "background_visible_tokens": len(events),
            "background_visible_tps": len(events) / duration,
            "background_itl_p95_ms": percentile(gaps, 95) if gaps else None,
            "background_restarts": sum(self.restarts),
            "background_errors": errors,
        }

    def close(self):
        self.stop.set()
        if self.pool is None:
            return
        for future in self.futures:
            try:
                future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                pass
        self.pool.shutdown(wait=False, cancel_futures=True)


def one_prime(args, serial, request_index, ready, go):
    prompt = prompt_ids(serial, request_index, args.prompt_tokens)
    body = {
        "model": args.model,
        "prompt_ids": prompt,
        "max_ctx": args.prompt_tokens + 64,
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": f"{args.label}-prime-{serial}-{request_index}-{time.time_ns()}",
    }
    ready.wait()
    go.wait()
    started = time.monotonic()
    request_id, usage, visible, first_visible = stream_completion(
        args.base, body, args.api_key, args.timeout)
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError("prime request completed without visible output")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if usage.get("prompt_tokens") != args.prompt_tokens or cached != 0:
        raise RuntimeError(
            f"unexpected prime usage prompt={usage.get('prompt_tokens')} cached={cached}")
    return {
        "id": request_id,
        "request": request_index,
        "client_ttft_s": first_visible - started,
        "client_wall_s": ended - started,
        "first_visible_at": first_visible,
        "visible_chunks": visible,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
    }


def burst(args, background, concurrency, repeat, serial):
    ready = threading.Barrier(concurrency + 1)
    go = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_prime, args, serial, request, ready, go)
            for request in range(concurrency)
        ]
        ready.wait()
        started_utc = utc_now()
        started = time.monotonic()
        go.set()
        rows = [future.result() for future in futures]
    ended = max(row["first_visible_at"] for row in rows)
    ended_utc = utc_now()
    wall = ended - started
    ttfts = [row["client_ttft_s"] for row in rows]
    background_summary = background.summary(started, ended)
    if background_summary["background_errors"]:
        raise RuntimeError(
            f"background decode errors: {background_summary['background_errors']}")
    summary = {
        "kind": "summary",
        "label": args.label,
        "concurrency": concurrency,
        "repeat": repeat,
        "n_requests": len(rows),
        "prompt_tokens_each": args.prompt_tokens,
        "aggregate_prompt_tokens": concurrency * args.prompt_tokens,
        "burst_start_utc": started_utc,
        "burst_end_utc": ended_utc,
        "wall_to_last_first_token_s": wall,
        "aggregate_prefill_tps": concurrency * args.prompt_tokens / wall,
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
        **background_summary,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--background", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    cells = [int(value) for value in args.cells.split(",")]
    if any(value < 1 for value in cells) or args.repeats < 1:
        parser.error("cells and repeats must be positive")

    output = open(args.out, "a", encoding="utf-8")
    background = BackgroundLoad(args)
    try:
        warm_args = argparse.Namespace(**vars(args))
        warm_args.label = f"{args.label}-warmup"
        warm_background = BackgroundLoad(warm_args)
        rows, summary = burst(warm_args, warm_background, 1, 0, 1)
        summary = {**summary, "kind": "warmup"}
        output.write(json.dumps(summary, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(summary, sort_keys=True), flush=True)

        background.start()
        serial = 10
        for concurrency in cells:
            for repeat in range(1, args.repeats + 1):
                rows, summary = burst(args, background, concurrency, repeat, serial)
                for row in rows:
                    output.write(json.dumps({
                        "kind": "request",
                        "label": args.label,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        **{key: value for key, value in row.items()
                           if key != "first_visible_at"},
                    }, sort_keys=True) + "\n")
                output.write(json.dumps(summary, sort_keys=True) + "\n")
                output.flush()
                print(json.dumps(summary, sort_keys=True), flush=True)
                serial += 1
    finally:
        background.close()
        output.close()


if __name__ == "__main__":
    main()
