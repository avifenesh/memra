#!/usr/bin/env python3
"""glm5 TP-2 serving gate (memra #14, lane/glm5-tp-serve-wiring-20260902).

Runs against an ALREADY LISTENING memra-server that was booted with
`MEMRA_GLM5_TP=all@<d0>,<d1>` and exits NON-ZERO the moment a request errors. It exists
because the first TP-2 box run (2026-09-02) failed every request with
`batch step: KDA layer is glm5-TP-sharded ...` and the box script still exited 0: a
functional failure must be the gate's own verdict, never something a reader has to spot
in a log.

Items (each one PASS / FAIL / SKIPPED, one JSONL receipt row per item):

  I1  readiness         GET /readyz answers 200
  I2  identity          GET /v1/models lists the pinned model id
  I3  greedy tape       128 greedy tokens on --prompt, reasoning_effort --effort, streamed;
                        sha16 of the full text (reasoning + content, bench.py assembly)
                        must equal --expect-sha16 (the PP-2 tape on the same artifact)
  I4  concurrent tapes  the same greedy request twice, concurrently: both tapes must equal
                        the expected sha16 (the per-session eager path is load-independent
                        and never enters a batch step)
  I5  vendor-default    one request with NO sampling params (the shape customers send),
                        512 tokens: must finish with completion_tokens > 0 and no loop
  I6  surfaces          /v1/completions and /v1/messages round trips answer 200 with text
  I7  tools             a tool-call chat request answers 200 with tool_calls or content
  I8  long prompt       --long-prompt (a 256k-class file), 64 greedy tokens: prompt_tokens
                        must reach --long-min-tokens and the request must complete
  I9  boot log          --boot-log carries the TP-2 admission, preflight and EAGER-ONLY
                        (sharded) markers, and the bytes appended during this gate carry
                        no `[engine-error]` / `batch step:` line

Verdict: FAIL (exit 1) if any item fails; PARTIAL (exit 3) if an item was SKIPPED because
its input was not supplied (a skipped item is never a PASS); PASS (exit 0) otherwise. Usage
errors exit 2.

Example (box):

  python3 tools/glm5-tp2-serve-gate.py --base http://127.0.0.1:18400 \\
      --model zai/glm-5.3-flash --prompt /root/prompts/digits.txt \\
      --expect-sha16 9437b599f6b9d2a9 --long-prompt /root/prompts/256k.txt \\
      --boot-log /root/lane/boot-tp2.log --out /root/lane/tp2-gate.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:18400")
    ap.add_argument("--model", required=True, help="pinned model id the server must list")
    ap.add_argument("--prompt", required=True, help="prompt file for the greedy tape (the digits prompt)")
    ap.add_argument("--expect-sha16", required=True, help="16-hex sha256 prefix of the PP-2 greedy tape")
    ap.add_argument("--effort", default="low", help="reasoning_effort for the tapes (default low)")
    ap.add_argument("--max-tokens", type=int, default=128, help="greedy tape length (default 128)")
    ap.add_argument("--sampled-max-tokens", type=int, default=512)
    ap.add_argument("--long-prompt", default=None, help="256k-class prompt file (I8); SKIPPED -> PARTIAL if absent")
    ap.add_argument("--long-min-tokens", type=int, default=200000, help="I8 minimum prompt_tokens (default 200000)")
    ap.add_argument("--long-max-tokens", type=int, default=64)
    ap.add_argument("--boot-log", default=None, help="server log to grep (I9); SKIPPED -> PARTIAL if absent")
    ap.add_argument("--api-key", default=os.environ.get("MEMRA_API_KEY", ""))
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None, help="JSONL receipts (default ./tp2-gate-<utc>.jsonl)")
    return ap.parse_args()


A = None
RECEIPTS = []
LOG_START_BYTES = 0


def log(msg):
    print(f"[tp2-gate] {msg}", flush=True)


def headers():
    h = {"content-type": "application/json"}
    if A.api_key:
        h["authorization"] = "Bearer " + A.api_key
    return h


def http_get(path):
    req = urllib.request.Request(A.base.rstrip("/") + path, headers=headers())
    with urllib.request.urlopen(req, timeout=A.timeout) as r:
        return r.status, r.read()


def http_post(path, body):
    req = urllib.request.Request(
        A.base.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers()
    )
    with urllib.request.urlopen(req, timeout=A.timeout) as r:
        return r.status, r.read()


def chat_stream(prompt, max_tokens, greedy, effort):
    """One streamed chat completion, assembled exactly like the floor bench (bench.py):
    text = concat of (reasoning_content | reasoning) + content per delta; sha16 over it."""
    body = {
        "model": A.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": uuid.uuid4().hex,
    }
    if greedy:
        body["temperature"] = 0
    if effort and effort != "none":
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        A.base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers()
    )
    t0 = time.time()
    ttft = None
    usage = None
    finish = None
    fp = None
    parts = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=A.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    j = json.loads(p)
                except Exception:
                    continue
                if j.get("error"):
                    err = f"stream error: {json.dumps(j['error'])[:300]}"
                    break
                fp = j.get("system_fingerprint") or fp
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        parts.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001 - the gate must report, never crash
        err = repr(e)
    wall = time.time() - t0
    text = "".join(parts)
    tail = text[-2000:]
    loop = False
    for i in range(0, max(0, len(tail) - 40), 20):
        if tail.count(tail[i : i + 40]) >= 6:
            loop = True
            break
    u = usage or {}
    return {
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish": finish,
        "fp": fp,
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "wall_s": round(wall, 3),
        "loop": loop,
        "sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
        "head": text[:120],
        "err": err,
    }


def tape_ok(r, expect_sha=None):
    """The request-level bar every tape item shares: no error, real tokens, a finish reason,
    no loop; plus the byte-identity bar when an expected sha is given."""
    if r["err"]:
        return False, r["err"]
    if not r["completion_tokens"]:
        return False, f"completion_tokens={r['completion_tokens']!r} (usage block missing or zero)"
    if not r["finish"]:
        return False, "no finish_reason on the stream"
    if r["loop"]:
        return False, "loop screen tripped (repeating 40-char window in the tail)"
    if expect_sha is not None and r["sha16"] != expect_sha:
        return False, f"sha16 {r['sha16']} != expected {expect_sha} (head={r['head'][:60]!r})"
    return True, (
        f"sha16={r['sha16']} prompt_tokens={r['prompt_tokens']} completion_tokens={r['completion_tokens']} "
        f"finish={r['finish']} ttft={r['ttft_s']}s wall={r['wall_s']}s"
    )


def item(name, fn):
    """Run one gate item; any exception is a FAIL with the exception text, never a crash."""
    t0 = time.time()
    try:
        status, detail, receipt = fn()
    except Exception as e:  # noqa: BLE001 - the gate must report, never crash
        status, detail, receipt = "FAIL", f"exception: {e!r}", None
    row = {
        "item": name,
        "status": status,
        "detail": detail,
        "elapsed_s": round(time.time() - t0, 3),
        "receipt": receipt,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RECEIPTS.append(row)
    log(f"{name} {status} {detail}")
    return status


def i1_ready():
    status, body = http_get("/readyz")
    text = body.decode("utf-8", "replace")
    if status != 200:
        return "FAIL", f"/readyz HTTP {status}: {text[:200]}", {"status": status}
    try:
        j = json.loads(text)
        st = j.get("status")
        if st is not None and st != "ready":
            return "FAIL", f"/readyz status={st!r}", j
        return "PASS", f"/readyz 200 status={st!r}", j
    except Exception:
        return "PASS", "/readyz 200 (non-JSON body)", {"body": text[:200]}


def i2_identity():
    status, body = http_get("/v1/models")
    j = json.loads(body.decode("utf-8", "replace"))
    ids = [m.get("id") for m in (j.get("data") or [])]
    if A.model not in ids:
        return "FAIL", f"/v1/models lists {ids}, pinned {A.model!r} absent", j
    return "PASS", f"/v1/models lists {A.model!r}", {"ids": ids}


def i3_greedy_tape():
    prompt = open(A.prompt, encoding="utf-8").read().strip()
    r = chat_stream(prompt, A.max_tokens, greedy=True, effort=A.effort)
    ok, why = tape_ok(r, A.expect_sha16)
    return ("PASS" if ok else "FAIL"), why, r


def i4_concurrent_tapes():
    prompt = open(A.prompt, encoding="utf-8").read().strip()
    out = [None, None]

    def run(k):
        out[k] = chat_stream(prompt, A.max_tokens, greedy=True, effort=A.effort)

    th = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    whys = []
    all_ok = True
    for k, r in enumerate(out):
        ok, why = tape_ok(r, A.expect_sha16)
        all_ok = all_ok and ok
        whys.append(f"[{k}] {why}")
    return ("PASS" if all_ok else "FAIL"), "; ".join(whys), {"reps": out}


def i5_vendor_default():
    prompt = open(A.prompt, encoding="utf-8").read().strip()
    r = chat_stream(prompt, A.sampled_max_tokens, greedy=False, effort=A.effort)
    ok, why = tape_ok(r)
    return ("PASS" if ok else "FAIL"), f"(no sampling params) {why}", r


def i6_surfaces():
    receipts = {}
    status, body = http_post("/v1/completions", {"model": A.model, "prompt": "Say hi.", "max_tokens": 8})
    j = json.loads(body.decode("utf-8", "replace"))
    text = ((j.get("choices") or [{}])[0]).get("text")
    receipts["completions"] = {"status": status, "text": text}
    if status != 200 or not isinstance(text, str):
        return "FAIL", f"/v1/completions HTTP {status} text={text!r}", receipts
    status, body = http_post(
        "/v1/messages",
        {"model": A.model, "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]},
    )
    j = json.loads(body.decode("utf-8", "replace"))
    content = j.get("content")
    receipts["messages"] = {"status": status, "content": content}
    if status != 200 or not isinstance(content, list) or not content:
        return "FAIL", f"/v1/messages HTTP {status} content={content!r}", receipts
    return "PASS", "/v1/completions 200 + /v1/messages 200 with text", receipts


def i7_tools():
    body = {
        "model": A.model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    status, raw = http_post("/v1/chat/completions", body)
    j = json.loads(raw.decode("utf-8", "replace"))
    msg = ((j.get("choices") or [{}])[0]).get("message") or {}
    calls = msg.get("tool_calls")
    content = msg.get("content")
    receipt = {"status": status, "tool_calls": calls, "content": content}
    if status != 200 or not (calls or content):
        return "FAIL", f"tools chat HTTP {status} tool_calls={calls!r} content={content!r}", receipt
    kind = "tool_calls" if calls else "content"
    return "PASS", f"tools chat 200 with {kind}", receipt


def i8_long_prompt():
    if not A.long_prompt:
        return "SKIPPED", "--long-prompt not supplied (256k item not exercised)", None
    prompt = open(A.long_prompt, encoding="utf-8", errors="replace").read()
    r = chat_stream(prompt, A.long_max_tokens, greedy=True, effort=A.effort)
    ok, why = tape_ok(r)
    if ok and (r["prompt_tokens"] or 0) < A.long_min_tokens:
        ok, why = False, f"prompt_tokens={r['prompt_tokens']} < --long-min-tokens {A.long_min_tokens}"
    return ("PASS" if ok else "FAIL"), why, r


def i9_boot_log():
    if not A.boot_log:
        return "SKIPPED", "--boot-log not supplied (route markers not checked)", None
    data = open(A.boot_log, "rb").read()
    whole = data.decode("utf-8", "replace")
    appended = data[LOG_START_BYTES:].decode("utf-8", "replace")
    must = [
        "MEMRA_GLM5_TP admitted for serving: ranks=2",
        "[glm5-tp-preflight] armed ranks=2",
        "EAGER-ONLY serving (glm5-TP-sharded trunk",
    ]
    missing = [m for m in must if m not in whole]
    bad = [ln for ln in appended.splitlines() if "[engine-error]" in ln or "batch step:" in ln]
    receipt = {"missing_markers": missing, "error_lines": bad[:10], "appended_bytes": len(data) - LOG_START_BYTES}
    if missing or bad:
        return (
            "FAIL",
            f"missing markers={missing} error lines during gate={len(bad)} first={bad[:1]!r}",
            receipt,
        )
    return "PASS", f"all {len(must)} route markers present, 0 engine-error lines during the gate", receipt


def main():
    global A, LOG_START_BYTES
    A = parse_args()
    for path, label in ((A.prompt, "--prompt"), (A.long_prompt, "--long-prompt"), (A.boot_log, "--boot-log")):
        if path and not os.path.exists(path):
            log(f"usage: {label} {path} does not exist")
            return 2
    if len(A.expect_sha16) != 16:
        log(f"usage: --expect-sha16 must be 16 hex chars, got {A.expect_sha16!r}")
        return 2
    if A.boot_log:
        LOG_START_BYTES = os.path.getsize(A.boot_log)
    out = A.out or time.strftime("tp2-gate-%Y%m%dT%H%M%SZ.jsonl", time.gmtime())
    log(f"invocation: {' '.join(sys.argv)}")
    log(f"base={A.base} model={A.model} expect_sha16={A.expect_sha16} effort={A.effort} out={out}")

    items = [
        ("I1-readiness", i1_ready),
        ("I2-identity", i2_identity),
        ("I3-greedy-tape", i3_greedy_tape),
        ("I4-concurrent-tapes", i4_concurrent_tapes),
        ("I5-vendor-default", i5_vendor_default),
        ("I6-surfaces", i6_surfaces),
        ("I7-tools", i7_tools),
        ("I8-long-prompt", i8_long_prompt),
        ("I9-boot-log", i9_boot_log),
    ]
    statuses = [item(name, fn) for name, fn in items]
    failed = [r["item"] for r in RECEIPTS if r["status"] == "FAIL"]
    skipped = [r["item"] for r in RECEIPTS if r["status"] == "SKIPPED"]
    if failed:
        verdict, rc = "FAIL", 1
    elif skipped:
        verdict, rc = "PARTIAL", 3
    else:
        verdict, rc = "PASS", 0
    summary = {
        "verdict": verdict,
        "passed": statuses.count("PASS"),
        "failed": failed,
        "skipped": skipped,
        "items": len(items),
        "invocation": sys.argv,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out, "a", encoding="utf-8") as f:
        for r in RECEIPTS:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    log(f"VERDICT {verdict} passed={summary['passed']}/{len(items)} failed={failed} skipped={skipped} receipts={out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
