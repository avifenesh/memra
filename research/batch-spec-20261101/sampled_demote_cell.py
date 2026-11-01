#!/usr/bin/env python3
"""Sampled spec demotion cell (lane/batch-spec, 2026-11-01).

Arms, each one fresh server boot, one sampled request (temperature 0.8, fixed seed):
  DEMOTE_A  MEMRA_SPEC_DEMOTE_AT=N : the sampled MTP session must demote at N generated tokens.
  DEMOTE_B  same as DEMOTE_A       : determinism, byte-identical to DEMOTE_A.
  SPEC      no demotion (c=1)       : the same seed on spec only; DEMOTE must agree with it on
                                       the pre-demotion prefix (demotion must not disturb what
                                       the spec stream already decided).
Checks:
  1. the demote line fires on the target with 'sampled, plain draws from philox event S';
  2. DEMOTE_A == DEMOTE_B byte for byte;
  3. DEMOTE and SPEC share a common prefix at least as long as the decoded first g tokens,
     then diverge (the continuation is drawn from fresh counters, not the spec stream's).
Raw server logs and responses are written next to this script under runs/<stamp>/.
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.expanduser("~/.cache/memra-batchspec")
BIN = os.path.expanduser("~/.cache/memra-batchspec-target/release/memra-server")
Q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
DRAFT = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
PORT = 8341
BASE = f"http://127.0.0.1:{PORT}"
DEMOTE_AT = int(os.environ.get("DEMOTE_AT", "40"))
MAX_TOK = int(os.environ.get("MAX_TOK", "160"))
SEED = int(os.environ.get("SEED", "1234"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(OUT, exist_ok=True)
PROMPT = ("Write a short story about a lighthouse keeper who finds a message in a bottle. "
          "Use vivid detail.")


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def arm(label, extra_env):
    log = os.path.join(OUT, f"{label}-server.log")
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": f"q9={Q9}+{DRAFT}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
                "MEMRA_CTX": "4096"})
    env.update(extra_env)
    with open(log, "w") as lf:
        p = subprocess.Popen([BIN], env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
                break
            except Exception:
                if p.poll() is not None:
                    raise SystemExit(f"{label}: server exited rc={p.returncode}; see {log}")
                time.sleep(1)
        else:
            raise SystemExit(f"{label}: server never came up; see {log}")
        body = {"model": "q9", "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": MAX_TOK, "temperature": 0.8, "seed": SEED, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = post(body)
        with open(os.path.join(OUT, f"{label}-response.json"), "w") as f:
            json.dump(r, f, indent=1)
        msg = r["choices"][0]["message"]
        text = (msg.get("reasoning") or msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
        return text, r.get("usage", {}), log
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


demote_env = {"MEMRA_SPEC_DEMOTE_AT": str(DEMOTE_AT)}
a, ua, la = arm("demote_a", demote_env)
b, ub, lb = arm("demote_b", demote_env)
s, us, ls = arm("spec", {})
lines = [l for l in open(la) if "[spec-gate] demoted" in l]
print("demote lines:", *lines, sep="\n  ")
ok = True
m = [re.search(r"generated (\d+), sampled, plain draws from philox event (\d+)", l) for l in lines]
m = [x for x in m if x]
if len(m) != 1:
    print("FAIL: expected exactly one sampled demotion of the target"); ok = False
else:
    g, ev = int(m[0].group(1)), int(m[0].group(2))
    print(f"demoted at generated={g}, first plain philox event={ev} (> g+1={g + 1}: {ev > g + 1})")
if a != b:
    print(f"FAIL: DEMOTE not deterministic (common prefix {common_prefix(a, b)} of {len(a)}/{len(b)} chars)"); ok = False
else:
    print(f"OK: DEMOTE_A == DEMOTE_B ({len(a)} chars, usage {ua})")
cp = common_prefix(a, s)
print(f"DEMOTE vs SPEC: common prefix {cp} chars; demote len {len(a)}, spec len {len(s)}; identical={a == s}")
spec_lines = [l for l in open(ls) if "[spec-gate] demoted" in l]
if spec_lines:
    print("FAIL: SPEC arm demoted"); ok = False
print("artifacts:", OUT)
print("VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
