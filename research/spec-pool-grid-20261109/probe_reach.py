#!/usr/bin/env python3
"""Which multi-turn shapes reach the MTP spec pool's exact and text tiers (memra#1093 item 1)?

One boot of memra-server on the MTP route (no DFlash drafter: the MTP spec program serves),
everything else at the device default except the cell config the earlier pool cells used
(context 8,192, two sessions). Each shape is a short multi-turn conversation sent in order; the
script slices the server log per request and counts the pool lines:

  spec-reuse: N committed tokens resumed        the exact or text tier resumed a parked end
  spec-affinity: rewound to N                   the affinity tier rewound to the turn checkpoint
  [prefix-cache] spec restore / hit             the prefix cache claimed the turn

usage: probe_reach.py BINARY OUT [SHAPE...]   (no SHAPE = all)
Writes OUT/server.log, OUT/requests.jsonl (one row per request with its log lines) and
OUT/summary.json. Exit 0 always; the summary says what fired.
"""
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

BIN = sys.argv[1]
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
for kv in os.environ.get("PROBE_ENV", "").split():
    k, v = kv.split("=", 1)
    ENV[k] = v

PATTERNS = {
    "spec_pool": r"spec-reuse: (\d+) committed tokens resumed",
    "spec_pool_text": r"spec-reuse: \d+ committed tokens resumed \[text-prefix\]",
    "spec_affinity": r"spec-affinity: rewound to (\d+) of (\d+)",
    "spec_affinity_declined": r"spec-affinity: declined \(([^;]*)",
    "spec_restore": r"spec restore: (\d+) of (\d+) prompt tokens",
    "prefix_hit": r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens",
    "spec_refused": r"spec-reuse REFUSED",
    "grid_decline": r"grid-rewind: \w+ declined",
    "pool_evict": r"spec pool evicted",
}


def boot(log):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(2)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read().decode())
    obj["_client_elapsed_s"] = round(time.perf_counter() - t0, 4)
    return obj


SHORT_USERS = ["List three prime numbers.", "Now three more, larger ones.", "Which of all six is the largest?"]
MID_USERS = [
    "Explain in two short paragraphs why a recurrent state cache cannot be rewound for free.",
    "Give one concrete example of a test that would catch a wrong rewind.",
    "Summarize both answers in one sentence.",
]


def chat_shape(users, max_tokens, think, pass_reasoning=False, user_id=None, system=None):
    def run(log_tail):
        msgs = [{"role": "system", "content": system}] if system else []
        rows = []
        for i, u in enumerate(users):
            msgs.append({"role": "user", "content": u})
            body = {"model": "q", "messages": msgs, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": think}}
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if user_id:
                body["user"] = user_id
            obj, lines = log_tail(lambda: post("/v1/chat/completions", body))
            m = obj["choices"][0]["message"]
            a = {"role": "assistant", "content": m.get("content") or ""}
            if pass_reasoning and m.get("reasoning_content"):
                a["reasoning_content"] = m["reasoning_content"]
            msgs.append(a)
            rows.append({"turn": i + 1, "finish": obj["choices"][0].get("finish_reason"),
                         "usage": obj.get("usage"), "lines": lines})
        return rows
    return run


def raw_template_shape(users, max_tokens, think_block=True):
    """A client that renders the Qwen chat template itself and replays the stream verbatim."""
    gen = "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if think_block else "")

    def run(log_tail):
        prompt = ""
        rows = []
        for i, u in enumerate(users):
            prompt += f"<|im_start|>user\n{u}<|im_end|>\n" + gen
            body = {"model": "q", "prompt": prompt, "temperature": 0}
            if max_tokens is not None:
                body["max_tokens"] = max_tokens[i] if isinstance(max_tokens, list) else max_tokens
            obj, lines = log_tail(lambda: post("/v1/completions", body))
            text = obj["choices"][0].get("text") or ""
            prompt += text + "<|im_end|>\n"
            rows.append({"turn": i + 1, "finish": obj["choices"][0].get("finish_reason"),
                         "usage": obj.get("usage"), "lines": lines})
        return rows
    return run


def raw_text_shape(seed_text, follows, max_tokens):
    def run(log_tail):
        prompt = seed_text
        rows = []
        for i in range(len(follows) + 1):
            body = {"model": "q", "prompt": prompt, "temperature": 0, "max_tokens": max_tokens}
            obj, lines = log_tail(lambda: post("/v1/completions", body))
            text = obj["choices"][0].get("text") or ""
            if i < len(follows):
                prompt += text + follows[i]
            rows.append({"turn": i + 1, "finish": obj["choices"][0].get("finish_reason"),
                         "usage": obj.get("usage"), "lines": lines})
        return rows
    return run


LONG_SYSTEM = " ".join(
    f"Note {i}: a parked session keeps its prefix entry, and a later turn that extends the same "
    f"prompt restores that entry and primes only the new suffix." for i in range(1, 31))

SHAPES = {
    "chat-short-nothink-256": chat_shape(SHORT_USERS, 256, False),
    "chat-short-nothink-open": chat_shape(SHORT_USERS, None, False),
    "chat-short-think-open-reasoning": chat_shape(SHORT_USERS, 1024, True, pass_reasoning=True),
    "chat-short-nothink-user": chat_shape(SHORT_USERS, 256, False, user_id="probe-user-1"),
    "chat-mid-nothink-open": chat_shape(MID_USERS, None, False),
    "raw-tmpl-short-open": raw_template_shape(SHORT_USERS, None),
    "raw-tmpl-short-256": raw_template_shape(SHORT_USERS, 256),
    "raw-tmpl-short-desc": raw_template_shape(SHORT_USERS, [512, 256, 128]),
    "raw-tmpl-mid-open": raw_template_shape(MID_USERS, None),
    "raw-text-short-64": raw_text_shape("The three laws of thermodynamics are", [" In short,", " Finally,"], 64),
    "chat-long-nothink-open": chat_shape(SHORT_USERS, None, False, system=LONG_SYSTEM),
    "raw-tmpl-long-open": raw_template_shape(
        [LONG_SYSTEM + "\n\n" + SHORT_USERS[0]] + SHORT_USERS[1:], None),
}


def main():
    names = sys.argv[3:] or list(SHAPES)
    log = OUT / "server.log"
    p = boot(log)
    rows = []
    try:
        for name in names:
            def log_tail(fn):
                before = log.stat().st_size
                obj = fn()
                time.sleep(0.3)  # let the retire/park lines flush
                with open(log, errors="replace") as f:
                    f.seek(before)
                    seg = f.read()
                keep = [ln for ln in seg.splitlines()
                        if re.search(r"spec-reuse|spec-affinity|spec restore|prefix-cache\] (hit|insert|spec)|"
                                     r"grid-rewind|spec pool|kv-reuse|seed REFUSED", ln)]
                return obj, keep
            for r in SHAPES[name](log_tail):
                r["shape"] = name
                rows.append(r)
                with open(OUT / "requests.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(name, r["turn"], r["finish"], (r["usage"] or {}).get("prompt_tokens"),
                      [ln[-140:] for ln in r["lines"]], flush=True)
    finally:
        stop(p)
    summary = {}
    for r in rows:
        s = summary.setdefault(r["shape"], {k: 0 for k in PATTERNS})
        for ln in r["lines"]:
            for k, rx in PATTERNS.items():
                if re.search(rx, ln):
                    s[k] += 1
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


main()
