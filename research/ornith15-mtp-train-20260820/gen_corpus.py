#!/usr/bin/env python3
"""Own-gen corpus driver for the Ornith-1.5-35B-A3B MTP-head training lane.

Drives a memra-server (spec-off, the NVFP4-MTP artifact we serve) over the
stratified 4K prompt pack (`build-prompt-pack.py` output: prompts.promptpack
\\0-separated + prompts.tsv metadata) plus the 44 real agentic .txt prompts.
Sampling = the vendor serving recipe (T=0.6, top_p=0.95, top_k=20) so the
corpus is on-policy for the distribution the head must accept at serve time;
`nothink` rows go through `reasoning_effort:"none"`, `think` rows take the
model default. One JSONL record per completion, append-only, resumable by id.
Seeds derive from the row id (stable across resumes).
"""
import argparse
import json
import pathlib
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor


def load_pack(pack_dir: pathlib.Path):
    raw = (pack_dir / "prompts.promptpack").read_bytes().split(b"\0")
    prompts = [p.decode("utf-8") for p in raw if p]
    lines = (pack_dir / "prompts.tsv").read_text(encoding="utf-8").splitlines()
    hdr = lines[0].split("\t")
    rows = [dict(zip(hdr, line.split("\t"))) for line in lines[1:]]
    assert len(prompts) == len(rows), (len(prompts), len(rows))
    for row, prompt in zip(rows, prompts):
        row["prompt"] = prompt
    return rows


def load_agentic(agentic_dir: pathlib.Path):
    return [
        {
            "id": f"agentic-{f.stem}",
            "split": "train",
            "mode": "think",
            "category": "agentic",
            "prompt": f.read_text(encoding="utf-8").strip(),
        }
        for f in sorted(agentic_dir.glob("*.txt"))
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", required=True, type=pathlib.Path)
    ap.add_argument("--agentic-dir", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--url", default="http://127.0.0.1:8094/v1/chat/completions")
    ap.add_argument("--model", default="ornith15")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=20260820)
    args = ap.parse_args()

    rows = load_pack(args.pack_dir)
    if args.agentic_dir and args.agentic_dir.is_dir():
        rows += load_agentic(args.agentic_dir)

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if str(r["id"]) not in done]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} todo", flush=True)

    lock = threading.Lock()
    out = open(args.out, "a", encoding="utf-8")
    stats = {"ok": 0, "err": 0, "toks": 0, "t0": time.time()}

    def work(row):
        seed = (args.seed_base + zlib.crc32(str(row["id"]).encode())) & 0xFFFFFFFF
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "max_tokens": 1024 if row["mode"] == "think" else 512,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "seed": seed,
            "include_reasoning": True,
        }
        if row["mode"] == "nothink":
            body["reasoning_effort"] = "none"
        req = urllib.request.Request(
            args.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    j = json.loads(resp.read())
                msg = j["choices"][0]["message"]
                rec = {
                    "id": str(row["id"]),
                    "split": row["split"],
                    "mode": row["mode"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "reasoning": msg.get("reasoning"),
                    "content": msg.get("content"),
                    "finish_reason": j["choices"][0].get("finish_reason"),
                    "usage": j.get("usage"),
                    "seed": seed,
                }
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    stats["ok"] += 1
                    stats["toks"] += (j.get("usage") or {}).get("completion_tokens", 0)
                    n = stats["ok"] + stats["err"]
                    if n % 50 == 0:
                        dt = time.time() - stats["t0"]
                        print(
                            f"[{n}/{len(todo)}] ok={stats['ok']} err={stats['err']} "
                            f"gen_toks={stats['toks']} avg {stats['toks']/dt:.0f} tok/s",
                            flush=True,
                        )
                return
            except Exception as e:  # noqa: BLE001 — driver must survive transient 5xx
                if attempt == 2:
                    with lock:
                        stats["err"] += 1
                        print(f"ERR id={row['id']}: {e}", flush=True)
                else:
                    time.sleep(5 * (attempt + 1))

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(work, todo))
    print(f"DONE ok={stats['ok']} err={stats['err']} toks={stats['toks']}", flush=True)


if __name__ == "__main__":
    main()
