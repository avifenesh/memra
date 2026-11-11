#!/usr/bin/env python3
"""One served cell, run INSIDE a gpulock lease: boot memra-server, run the requests, stop it.

usage: serve_cell.py <server binary> <out dir> <mode> [ENV=V ...]
  mode = pair    one /v1/completions board-2048 request, temperature 0, 128 tokens
         itl     session A streams 384 tokens from a short prompt; when A has 48 tokens, session B
                 sends the board-2048 request; A's inter-token latency is reported for the
                 window B is in flight, and for the same-length window before B was sent

Writes server.log, response*.json, memory.csv (nvidia-smi memory.used every 250 ms), result.json.
"""
import http.client
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import threading
import time

server, out, mode = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
out.mkdir(parents=True, exist_ok=True)
REPO = pathlib.Path.home() / ".cache/memra-1225/wt"
MODEL = os.environ.get("CELL_MODEL", "<models>/gemma-4-26B_q4_0-it.gguf")
PORT = 18225
prompt = (REPO / "research/e2e/prompts/board-2048.txt").read_text()

env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
env.update({"MEMRA_MODELS": f"g26={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}"})
# measurement override: the staged program prefills board-2048 for minutes; same for every cell
env["MEMRA_TIMEOUT_MS_MAX"] = "3600000"
env.update(extra)
# MEMRA_GPU_LOCK is inherited from the lease on purpose (the server must not flock it itself).
if "MEMRA_GPU_LOCK" in os.environ:
    env["MEMRA_GPU_LOCK"] = os.environ["MEMRA_GPU_LOCK"]

mem_stop = threading.Event()


def sample_memory():
    with open(out / "memory.csv", "w") as f:
        f.write("t_s,used_mib\n")
        t0 = time.time()
        while not mem_stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True)
            f.write(f"{time.time() - t0:.3f},{r.stdout.strip()}\n")
            f.flush()
            mem_stop.wait(0.25)


def post(body, stream=False, timeout=3600, path="/v1/completions"):
    body = dict(body, timeout_ms=3_600_000)  # the staged program outlives the 90 s default
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return c, c.getresponse()


def stream_completion(body, path="/v1/completions"):
    """POST with stream=true; returns (text, events, ttft_s, e2e_s, last_event)."""
    t0 = time.time()
    c, r = post(dict(body, stream=True), path=path)
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:500]!r}")
    buf, text, events, ttft, last = b"", [], 0, None, None
    while True:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if not ev.startswith(b"data: ") or ev[6:].strip() == b"[DONE]":
                continue
            obj = json.loads(ev[6:])
            last = obj
            if "error" in obj:
                raise SystemExit(f"stream error: {obj}")
            # memra's native stream: one {"id": token, "text": piece} event per token
            pieces = [ch.get("text") or (ch.get("delta") or {}).get("content")
                      for ch in obj.get("choices", [])] if "choices" in obj \
                else ([obj["text"]] if "text" in obj and "id" in obj else [])
            for piece in pieces:
                if piece is None:
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
                events += 1
    return "".join(text), events, ttft, time.time() - t0, last


log = open(out / "server.log", "w")
boot0 = time.time()
proc = subprocess.Popen([server], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                        start_new_session=True)
mem = threading.Thread(target=sample_memory, daemon=True)
mem.start()
result = {"mode": mode, "env": extra, "server": server}
try:
    deadline = time.time() + 600
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode} during boot")
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
            c.request("GET", "/v1/models")
            if c.getresponse().status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            raise SystemExit("boot timeout")
        time.sleep(1)
    result["boot_s"] = round(time.time() - boot0, 1)
    result["mem_ready_s"] = round(time.time() - boot0, 3)

    if mode == "pair":
        import hashlib
        text, events, ttft, e2e, last = stream_completion(
            {"model": "g26", "prompt": prompt, "max_tokens": 128, "temperature": 0})
        (out / "response.txt").write_text(text)
        (out / "last_event.json").write_text(json.dumps(last, indent=1))
        result.update({"text_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
                       "events": events, "ttft_s": round(ttft, 3) if ttft else None,
                       "e2e_s": round(e2e, 3), "usage": (last or {}).get("usage")})
        # the same prompt through the chat template (the -it model's served form), 128 tokens
        chat, events, ttft, e2e, last = stream_completion(
            {"model": "g26", "max_tokens": 128, "temperature": 0,
             "messages": [{"role": "user", "content": prompt + "\n\nSummarize the table above in three sentences."}]},
            path="/v1/chat/completions")
        (out / "chat.txt").write_text(chat)
        result.update({"chat_sha16": hashlib.sha256(chat.encode()).hexdigest()[:16],
                       "chat_events": events, "chat_ttft_s": round(ttft, 3) if ttft else None,
                       "chat_e2e_s": round(e2e, 3)})
    elif mode == "decode":
        # decode rate: a short prompt, 256 greedy tokens; rate over the tokens after the first
        import hashlib
        text, events, ttft, e2e, last = stream_completion(
            {"model": "g26", "prompt": "Write a long story about a lighthouse keeper.",
             "max_tokens": 256, "temperature": 0, "ignore_eos": True})
        (out / "response.txt").write_text(text)
        (out / "last_event.json").write_text(json.dumps(last, indent=1))
        usage = (last or {}).get("usage") or {}
        n = usage.get("completion_tokens") or events
        result.update({"text_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
                       "events": events, "completion_tokens": n,
                       "ttft_s": round(ttft, 3) if ttft else None, "e2e_s": round(e2e, 3),
                       "decode_tok_per_s": round((n - 1) / (e2e - ttft), 2) if ttft and n > 1 else None})
    elif mode == "itl":
        arrivals = []
        b_times = {}

        def session_b():
            b_times["sent"] = time.time()
            _, r = post({"model": "g26", "prompt": prompt, "max_tokens": 16, "temperature": 0})
            body = json.loads(r.read())
            b_times["done"] = time.time()
            (out / "response_b.json").write_text(json.dumps(body, indent=1))

        tb = None
        c, r = post({"model": "g26", "prompt": "Write a long story about a lighthouse keeper.",
                     "max_tokens": 384, "temperature": 0, "stream": True})
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if ev.startswith(b"data: ") and ev[6:].strip() != b"[DONE]":
                    arrivals.append(time.time())
                    if len(arrivals) == 48 and tb is None:
                        tb = threading.Thread(target=session_b)
                        tb.start()
        if tb:
            tb.join()
        gaps = [(arrivals[i], arrivals[i] - arrivals[i - 1]) for i in range(1, len(arrivals))]
        s, d = b_times["sent"], b_times["done"]
        during = [g * 1000 for t, g in gaps if s < t <= d + 0.0]
        span = d - s
        before = [g * 1000 for t, g in gaps if s - span < t <= s]

        def pct(v, q):
            if not v:
                return None
            v = sorted(v)
            return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2)

        result.update({
            "a_events": len(arrivals),
            "b_s": round(span, 3),
            "itl_during_b": {"n": len(during), "p50_ms": pct(during, 0.5),
                             "p99_ms": pct(during, 0.99), "max_ms": round(max(during), 2) if during else None},
            "itl_before_b": {"n": len(before), "p50_ms": pct(before, 0.5),
                             "p99_ms": pct(before, 0.99)},
        })
        (out / "arrivals.json").write_text(json.dumps({"a": arrivals, "b": b_times}))
finally:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL)
    mem_stop.set()
    mem.join(timeout=5)
    rows = [l.split(",") for l in (out / "memory.csv").read_text().splitlines()[1:] if "," in l]
    used = [int(u) for _, u in rows if u.strip().isdigit()]
    result["peak_mem_mib"] = max(used) if used else None
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
