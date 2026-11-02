#!/usr/bin/env python3
"""Served-stream battery for the batch-spec round split (pure refactor gate).

usage: serve_battery.py BIN OUTDIR [PORT]

Arms (each a fresh memra-server boot, q9 + owntrim draft, MEMRA_CTX=4096):
  main      sequential requests: greedy/sampled short, greedy/sampled long (multi-burst,
            pending carry), sampled+penalties+top_p, greedy json_schema (constrained spec),
            then a concurrent greedy pair (admission yield / burst interleave).
  demote_a  MEMRA_SPEC_DEMOTE_AT=40: one sampled + one greedy request; demotion must fire.
  demote_b  same as demote_a (determinism).
Writes OUTDIR/<arm>-server.log, OUTDIR/<arm>-<req>.json and OUTDIR/summary.json
(sha256 of every response's reasoning\\0content, finish reasons, usage, demote lines).
"""
import hashlib, json, os, re, subprocess, sys, threading, time, urllib.request

BIN = sys.argv[1]
OUT = sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8343
BASE = f"http://127.0.0.1:{PORT}"
Q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
DRAFT = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
os.makedirs(OUT, exist_ok=True)

SHORT = ("Write a short story about a lighthouse keeper who finds a message in a bottle. "
         "Use vivid detail.")
NOTES = " ".join(
    f"Note {i}: in {1800 + 3 * i} the harbor council recorded {(i * 17) % 97 + 5} ships, "
    f"{(i * 29) % 61} storms and a grain price of {(i * 13) % 41 + 10} shillings; the keeper "
    f"of the north light wrote that the {['east', 'west', 'south', 'north'][i % 4]} channel "
    f"was {['silting', 'clear', 'dredged', 'flooded', 'frozen'][i % 5]}."
    for i in range(60))
LONG = NOTES + "\n\nSummarize the notes above, then list three trends with the years that show them."
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
                         "ships": {"type": "array", "items": {"type": "integer"}}},
          "required": ["name", "year", "ships"]}


def body(prompt, max_tokens, temperature, seed=None, **extra):
    b = {"model": "q9", "messages": [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens, "temperature": temperature, "stream": False,
         "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        b["seed"] = seed
    b.update(extra)
    return b


MAIN = [
    ("g_short", body(SHORT, 200, 0.0)),
    ("s_short", body(SHORT, 200, 0.8, 1234)),
    ("g_long", body(LONG, 320, 0.0)),
    ("s_long", body(LONG, 320, 0.8, 77)),
    ("s_pen", body(SHORT, 200, 0.8, 4321, top_p=0.9, frequency_penalty=0.3, presence_penalty=0.2)),
    ("g_json", body("Give a JSON record for a lighthouse keeper: name, year, and ships seen "
                    "on five days.", 160, 0.0,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "rec", "schema": SCHEMA}})),
]
PAIR = [
    ("g_c2a", body("Explain how a lighthouse lens focuses light, step by step.", 200, 0.0)),
    ("g_c2b", body(LONG, 200, 0.0)),
]
DEMOTE = [
    ("s_demote", body(SHORT, 160, 0.8, 1234)),
    ("g_demote", body(SHORT, 160, 0.0)),
]


def post(b):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def record(arm, name, r, summary):
    with open(os.path.join(OUT, f"{arm}-{name}.json"), "w") as f:
        json.dump(r, f, indent=1)
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning") or msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    summary[f"{arm}/{name}"] = {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "finish": r["choices"][0].get("finish_reason"),
        "completion_tokens": r.get("usage", {}).get("completion_tokens"),
    }


def boot(arm, extra_env):
    log = os.path.join(OUT, f"{arm}-server.log")
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": f"q9={Q9}+{DRAFT}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
                "MEMRA_CTX": "4096"})
    env.update(extra_env)
    lf = open(log, "w")
    p = subprocess.Popen([BIN], env=env, stdout=lf, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
            return p, log
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"{arm}: server exited rc={p.returncode}; see {log}")
            time.sleep(1)
    p.kill()
    raise SystemExit(f"{arm}: server never came up; see {log}")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


summary = {"bin": BIN}
p, log = boot("main", {})
try:
    for name, b in MAIN:
        record("main", name, post(b), summary)
    res = {}
    ths = [threading.Thread(target=lambda n=n, b=b: res.__setitem__(n, post(b))) for n, b in PAIR]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for n, _ in PAIR:
        record("main", n, res[n], summary)
finally:
    stop(p)

for arm in ("demote_a", "demote_b"):
    p, log = boot(arm, {"MEMRA_SPEC_DEMOTE_AT": "40"})
    try:
        for name, b in DEMOTE:
            record(arm, name, post(b), summary)
    finally:
        stop(p)
    lines = [l.strip() for l in open(log) if "[spec-gate] demoted" in l]
    summary[f"{arm}/demote_lines"] = [re.sub(r"^.*?\[spec-gate\]", "[spec-gate]", l) for l in lines]

det = all(summary[f"demote_a/{n}"]["sha256"] == summary[f"demote_b/{n}"]["sha256"] for n, _ in DEMOTE)
summary["demote_fired"] = len(summary["demote_a/demote_lines"]) >= 1
summary["demote_deterministic"] = det and summary["demote_a/demote_lines"] == summary["demote_b/demote_lines"]
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
print(json.dumps(summary, indent=1, sort_keys=True))
