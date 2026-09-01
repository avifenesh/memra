#!/usr/bin/env python3
"""Drive the live pi-shaped growing conversation against an isolated test server.

The empty stop string deliberately ends generation after the first decoded token while keeping
the request's 32,768-token allowance (and therefore its request-owned KV charge) unchanged.
This makes the gate about parked-cache growth, not a multi-hour generation soak.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM = """You are editing a production operations dashboard. Return the complete document on
every turn. Preserve prior behavior, use accessible semantic HTML, and keep exact evidence values.
The following numbered operating notes are source material and must remain represented:
"""
NOTE = (
    "Operating note {i}: each request records arrival, time to first token, decode rate, cache "
    "source, queue delay, and completion state. Counters are cumulative; occupancy and active "
    "sessions are gauges. Evidence retains units and exact timestamps. Operators can filter by "
    "session and compare speculative and plain paths.\n"
)
SMALL_TURNS = (
    "Create the initial dashboard with a request timeline, cache gauges, admission queue, and "
    "incident annotations. Return only the document.",
    "Add a sticky request inspector that distinguishes prompt processing from decode time, plus "
    "keyboard navigation and visible focus treatment. Return the full revised document.",
    "Add an accessible cache-tier legend and a compact admission-pressure strip with defer, park, "
    "active, and queued values. Preserve every earlier requirement.",
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def render(system: str, history: list[tuple[str, str]]) -> str:
    parts = [f"{IM_START}system\n{system}{IM_END}\n"]
    for role, content in history:
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n<think>\n")
    return "".join(parts)


def headers(api_key: str | None) -> dict[str, str]:
    value = {"Content-Type": "application/json", "User-Agent": "memra-affroom-pi/1"}
    if api_key:
        value["Authorization"] = f"Bearer {api_key}"
    return value


def fetch_json(url: str, api_key: str | None, timeout: float) -> dict:
    request = urllib.request.Request(url, headers=headers(api_key))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return value


def metrics(args: argparse.Namespace) -> dict:
    return fetch_json(args.base.rstrip("/") + "/metrics", args.api_key, args.timeout)


def wait_completed(args: argparse.Namespace, before: dict) -> dict:
    target = int(before.get("completed", 0)) + 1
    deadline = time.monotonic() + 30
    last = before
    while time.monotonic() < deadline:
        last = metrics(args)
        if int(last.get("completed", 0)) >= target:
            return last
        time.sleep(0.05)
    raise RuntimeError(f"metrics never published completed={target}; last={last.get('completed')}")


def issue(args: argparse.Namespace, prompt: str, turn: int, raw_dir: pathlib.Path) -> dict:
    before = metrics(args)
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stop": "",
        "cache_salt": args.cache_salt,
        "session_id": args.session_id,
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers(args.api_key),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:4000]
        raise RuntimeError(f"turn {turn}: HTTP {error.code}: {detail}") from error
    elapsed = time.perf_counter() - started
    after = wait_completed(args, before)
    (raw_dir / f"turn-{turn:02d}-response.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    usage = result.get("usage") or {}
    choice = (result.get("choices") or [{}])[0]
    text = choice.get("text") or result.get("text") or ""
    prompt_tokens = int(usage.get("prompt_tokens") or result.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or result.get("n_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0)
    return {
        "turn": turn,
        "at": utc_now(),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "request_need": prompt_tokens + args.max_tokens + 64,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": choice.get("finish_reason") or result.get("stop_reason"),
        "wall_s": round(elapsed, 6),
        "plain_affinity_rewinds_before": int(before.get("plain_affinity_rewinds", 0)),
        "plain_affinity_rewinds_after": int(after.get("plain_affinity_rewinds", 0)),
        "plain_affinity_rewinds_delta": int(after.get("plain_affinity_rewinds", 0))
        - int(before.get("plain_affinity_rewinds", 0)),
        "continuation_pool_hits_delta": int(after.get("continuation_pool_hits", 0))
        - int(before.get("continuation_pool_hits", 0)),
        "cache_hit_token_ratio_after": after.get("cache_hit_token_ratio"),
        "cuda_driver_free_bytes_after": after.get("cuda_driver_free_bytes"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default="step35")
    parser.add_argument("--api-key")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--session-id", default="affroom-owner-pi-shape")
    parser.add_argument("--cache-salt", default="affroom-20260809-pod")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--base-notes", type=int, default=206)
    parser.add_argument("--paste-notes", type=int, default=129)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--window-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / "responses"
    raw_dir.mkdir(exist_ok=True)
    initial = metrics(args)
    system = SYSTEM + "".join(NOTE.format(i=i + 1) for i in range(args.base_notes))
    history: list[tuple[str, str]] = []
    rows: list[dict] = []

    for turn, user in enumerate(SMALL_TURNS, start=1):
        history.append(("user", user))
        prompt = render(system, history)
        row = issue(args, prompt, turn, raw_dir)
        rows.append(row)
        history.append(("assistant", row["text"]))
        print(json.dumps({key: value for key, value in row.items() if key != "text"}, sort_keys=True), flush=True)

    paste = "Large pasted incident appendix follows. Preserve it verbatim in the evidence view.\n" + "".join(
        NOTE.format(i=10_000 + i) for i in range(args.paste_notes)
    )
    history.append(("user", paste))
    row = issue(args, render(system, history), 4, raw_dir)
    rows.append(row)
    print(json.dumps({key: value for key, value in row.items() if key != "text"}, sort_keys=True), flush=True)

    failures: list[str] = []
    tokens = [row["prompt_tokens"] for row in rows]
    deltas = [tokens[i] - tokens[i - 1] for i in range(1, len(tokens))]
    if not 12_000 <= tokens[0] <= 13_500:
        failures.append(f"first prompt {tokens[0]} not in owner-shape 12.0k..13.5k window")
    for index, delta in enumerate(deltas[:2], start=2):
        if not 1 <= delta <= 500:
            failures.append(f"turn {index} growth {delta} not in small-turn 1..500 window")
    if not 7_000 <= deltas[2] <= 9_000:
        failures.append(f"large-paste growth {deltas[2]} not in 7k..9k window")
    if not 19_500 <= tokens[-1] <= 22_000:
        failures.append(f"final prompt {tokens[-1]} not in owner-shape 19.5k..22k window")
    for row in rows[1:]:
        if row["plain_affinity_rewinds_delta"] != 1:
            failures.append(
                f"turn {row['turn']} rewind delta={row['plain_affinity_rewinds_delta']} (want 1)"
            )
    if any(row["completion_tokens"] != 1 for row in rows):
        failures.append("empty-stop control did not stop every request after one token")
    final = metrics(args)
    summary = {
        "schema": 1,
        "created_at": utc_now(),
        "base": args.base,
        "model": args.model,
        "session_id": args.session_id,
        "max_tokens": args.max_tokens,
        "context_cap": 262144,
        "spec_k": 0,
        "pp_stages": 2,
        "pp_devices": [0, 1],
        "window_clean": args.window_clean,
        "empty_stop_control": "one generated token; request-owned max_tokens charge unchanged",
        "prompt_token_deltas": deltas,
        "plain_affinity_rewinds_delta": int(final.get("plain_affinity_rewinds", 0))
        - int(initial.get("plain_affinity_rewinds", 0)),
        "rows": rows,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "metrics-initial.json").write_text(json.dumps(initial, indent=2, sort_keys=True) + "\n")
    (args.out / "metrics-final.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
