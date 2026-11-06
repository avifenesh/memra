#!/usr/bin/env python3
"""n2_stream_check.py: memra#1076's GB10 live check. One streaming chat completion with n = 2 on a
long prompt (the gb10-prefill harness document, sized by its size cache, fresh nonce so nothing
hits the prefix cache), temperature 0, thinking off, max_tokens 40.

  n2_stream_check.py OUT_DIR --lines 4678 --nonce TEXT [--n 2]

Writes OUT_DIR/events.jsonl (one row per SSE line with its arrival time since send, the raw line
cut to 400 bytes) and OUT_DIR/result.json. PASS when: HTTP 200, every choice index streamed
content and ended with finish_reason stop or length, no error event, a [DONE] terminal, and the
first token came after the 60 s SSE prefill commit (so the check reached the committed path).
BENCH_BASE (default http://127.0.0.1:8090), BENCH_MODEL (default q38). stdlib only."""
import argparse, hashlib, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_long_memra import doc, KW  # noqa: E402

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8090").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "q38")
COMMIT_S = 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--lines", type=int, default=4678)
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc(a.lines, a.nonce)}],
            "max_tokens": a.max_tokens, "temperature": 0.0, "presence_penalty": 0.0, "n": a.n,
            "chat_template_kwargs": KW, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    res = {"n": a.n, "lines": a.lines, "nonce": a.nonce, "max_tokens": a.max_tokens, "http": None,
           "headers_s": None, "first_token_s": {}, "finish": {}, "text_sha256": {}, "error_events": [],
           "done": False, "usage": None, "exception": None}
    texts = {}
    t0 = time.time()
    with open(out / "events.jsonl", "w") as ev:
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                res["http"] = r.status
                res["headers_s"] = round(time.time() - t0, 3)
                for raw in r:
                    t = round(time.time() - t0, 3)
                    line = raw.decode(errors="replace").rstrip("\r\n")
                    if not line:
                        continue
                    ev.write(json.dumps({"t": t, "line": line[:400]}) + "\n")
                    ev.flush()
                    if line == "data: [DONE]":
                        res["done"] = True
                        continue
                    if not line.startswith("data:"):
                        continue
                    d = json.loads(line[5:])
                    if "error" in d:
                        res["error_events"].append({"t": t, "error": d["error"]})
                        continue
                    res["usage"] = d.get("usage") or res["usage"]
                    for ch in d.get("choices", []):
                        i = str(ch.get("index"))
                        if ch.get("finish_reason"):
                            res["finish"][i] = ch["finish_reason"]
                        dl = ch.get("delta", {})
                        piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                        if piece:
                            res["first_token_s"].setdefault(i, t)
                            texts.setdefault(i, []).append(piece)
        except urllib.error.HTTPError as e:
            res["http"] = e.code
            res["headers_s"] = round(time.time() - t0, 3)
            res["exception"] = e.read().decode(errors="replace")[:2000]
        except Exception as e:  # a failed request is a result
            res["exception"] = repr(e)
    res["total_s"] = round(time.time() - t0, 3)
    for i, parts in texts.items():
        s = "".join(parts)
        res["text_sha256"][i] = hashlib.sha256(s.encode()).hexdigest()
        res.setdefault("text", {})[i] = s
    idx = [str(i) for i in range(a.n)]
    first = min(res["first_token_s"].values()) if res["first_token_s"] else None
    res["checks"] = {
        "http_200": res["http"] == 200,
        "every_choice_streamed": all(i in res["first_token_s"] for i in idx),
        "every_choice_finished": all(res["finish"].get(i) in ("stop", "length") for i in idx),
        "no_error_event": not res["error_events"] and res["exception"] is None,
        "done_terminal": res["done"],
        "first_token_after_commit": first is not None and first > COMMIT_S,
    }
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("verdict", "http", "headers_s", "first_token_s", "finish",
                                          "error_events", "done", "total_s", "exception", "checks")}))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
