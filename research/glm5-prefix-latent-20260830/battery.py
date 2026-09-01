#!/usr/bin/env python3
"""Box battery for MEMRA_PREFIX_LATENT (lane/glm5-prefix-latent). Design + flip condition:
DESIGN.md in this directory.

Two arms, each its own server boot, run this script once per arm:

  ARM on   server booted with MEMRA_PREFIX_LATENT=1 and MEMRA_PREFIX_CACHE_MB=4096.
           PASS = restored-vs-cold BYTE IDENTITY (one greedy sha per raw prompt across the
           cold rep and every restored rep) WITH the cache demonstrably engaged
           (cached_tokens == prompt_tokens on every restored rep), multiturn cache
           engagement from turn 2 on, and hit-vs-cold TTFT receipts at three depths.
  ARM off  server booted WITHOUT the flag (any cache budget). PASS = the guard holds:
           cached_tokens == 0 everywhere (glm5 captures refuse), one greedy sha per raw
           prompt (no restore can occur), and the refusal line present in the server log
           (grep '[prefix-cache] snapshot failed (latent' — checked outside this script).

The raw byte cell is the parent lane's own discriminator (rebaseline-cacheoff.py R3,
prompts p5/p7 of the banked agent pool); the arm design (rep 0 cold, reps 1..N restored,
same boot, byte-identical bodies, rep index the only variable) is latentprobe.py's.
Run latentprobe.py from research/prefix-restore-toolcall-20260828/ in the same window for
the unguessable-tool round trip; this script covers the raw oracle, the owner-law 8-turn
multiturn twin, and TTFT-at-depth.

PRECONDITIONS ARE ASSERTED, NOT ASSUMED: any row violating its arm's cache-engagement
expectation is named and the script exits 2. A battery that silently measures the wrong
regime is worse than no battery (parent lane lesson).

usage: battery.py <outdir> <on|off> [raw_reps]
env:   EP (default http://127.0.0.1:18400), MODEL (default zai/glm-5.3-flash),
       PROMPTS_JSON (default /home/ubuntu/prompts.json)
"""

import hashlib
import json
import os
import sys
import time
import urllib.request

OUT = sys.argv[1]
ARM = sys.argv[2]
assert ARM in ("on", "off"), "arm must be 'on' or 'off'"
RAW_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
EP = os.environ.get("EP", "http://127.0.0.1:18400")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")
POOL = json.load(open(os.environ.get("PROMPTS_JSON", "/home/ubuntu/prompts.json")))["decode"]
os.makedirs(OUT, exist_ok=True)
VIOLATIONS = []
RESULTS = {"arm": ARM, "ep": EP, "model": MODEL, "raw": [], "multiturn": [], "depth": []}


def loopiness(s, w=48):
    """The parent lane's own repeat-window score, kept identical so the numbers compare."""
    if len(s) < 4 * w:
        return 0.0
    tail = s[-2000:]
    seen, best = {}, 0
    for i in range(0, len(tail) - w):
        k = tail[i : i + w]
        seen[k] = seen.get(k, 0) + 1
        best = max(best, seen[k])
    return round(best * w / len(tail), 3)


def expect_cached(row, cold, where):
    """The arm's cache-engagement contract, asserted per row."""
    c = row.get("cached_tokens") or 0
    if ARM == "off" and c > 0:
        VIOLATIONS.append(f"{where}: cached_tokens={c} but the OFF arm must never restore")
    if ARM == "on" and not cold and c != (row.get("prompt_tokens") or -1):
        VIOLATIONS.append(
            f"{where}: cached_tokens={c} != prompt_tokens={row.get('prompt_tokens')} — "
            "the restored rep did not take a whole-entry hit (eviction? budget? guard?)"
        )
    if ARM == "on" and cold and c > 0:
        VIOLATIONS.append(f"{where}: cold rep reports cached_tokens={c}; the cell is contaminated")


def raw_completion(prompt, name, max_tokens=64):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        EP + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            r = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 - every failure is a named receipt row
        r = {"__error__": f"{type(e).__name__}: {e}"}
    ch = (r.get("choices") or [{}])[0]
    txt = ch.get("text", "") or ""
    u = r.get("usage") or {}
    row = {
        "name": name,
        "wall_s": round(time.time() - t0, 3),
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish": ch.get("finish_reason"),
        "out_sha16": hashlib.sha256(txt.encode()).hexdigest()[:16],
        "error": r.get("__error__"),
    }
    open(f"{OUT}/{name}.json", "w").write(json.dumps(r, indent=1))
    return row


def chat(messages, name, max_tokens=512):
    """Vendor-default SAMPLED (no temperature/top_p/top_k/seed), reasoning_effort pinned,
    streaming so TTFT = wall time to the FIRST reasoning/content delta."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        EP + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"content-type": "application/json"},
    )
    think, out, fr, usage, sse, ttft = [], [], None, None, [], None
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for rl in r:
                line = rl.decode()
                sse.append(line)
                s = line.strip()
                if not s.startswith("data:"):
                    continue
                pay = s[5:].strip()
                if pay == "[DONE]":
                    break
                o = json.loads(pay)
                if o.get("usage"):
                    usage = o["usage"]
                c = (o.get("choices") or [{}])[0]
                if c.get("finish_reason"):
                    fr = c["finish_reason"]
                d = c.get("delta") or {}
                tok = d.get("reasoning") or d.get("reasoning_content") or d.get("content")
                if tok and ttft is None:
                    ttft = round(time.time() - t0, 3)
                if d.get("reasoning") or d.get("reasoning_content"):
                    think.append(d.get("reasoning") or d.get("reasoning_content"))
                if d.get("content"):
                    out.append(d["content"])
    except Exception as e:  # noqa: BLE001
        sse.append(f"{type(e).__name__}: {e}")
    open(f"{OUT}/{name}.sse", "w").write("".join(sse))
    text, thought = "".join(out), "".join(think)
    u = usage or {}
    return {
        "name": name,
        "ttft_s": ttft,
        "wall_s": round(time.time() - t0, 3),
        "finish": fr,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "loop_content": loopiness(text),
        "loop_reasoning": loopiness(thought),
        "content": text,
    }


BAR = "#" * 78

# C1  RAW BYTE ORACLE, p5 and p7: rep 0 cold, reps 1..N-1 restored (arm on) or refused
#     (arm off). THE acceptance bar: ONE sha per prompt across every rep.
print(BAR)
print(f"# C1 raw /v1/completions greedy, p5/p7, rep0 cold + {RAW_REPS - 1} repeats, arm={ARM}")
print(BAR)
c1_pass = True
for idx in (5, 7):
    shas = []
    for rep in range(RAW_REPS):
        row = raw_completion(POOL[idx]["text"], f"c1-p{idx}-rep{rep}")
        row["idx"], row["rep"] = idx, rep
        expect_cached(row, cold=(rep == 0), where=f"C1 p{idx} rep{rep}")
        RESULTS["raw"].append(row)
        shas.append(row["out_sha16"])
        print(
            f"  p{idx} rep{rep}: sha={row['out_sha16']} cached={row['cached_tokens']} "
            f"prompt={row['prompt_tokens']} finish={row['finish']} err={row['error']}"
        )
    if len(set(shas)) != 1:
        c1_pass = False
        VIOLATIONS.append(f"C1 p{idx}: BYTE DIVERGENCE across reps: {shas}")
    print(f"  == p{idx}: {'ONE sha' if len(set(shas)) == 1 else 'DIVERGED'} ({shas[0]})")
print()

# C2  THE OWNER-LAW MULTITURN TWIN: 8 turns, larger prompt, full history resent per turn
#     (the agent shape), vendor-default sampled. Per-turn TTFT + cache engagement.
print(BAR)
print(f"# C2 8-turn multiturn twin, larger prompt, sampled, arm={ARM}")
print(BAR)
seed_context = "\n\n".join(p["text"] for p in POOL[:4])
messages = [
    {
        "role": "user",
        "content": "Context documents for this session, refer back to them as needed:\n\n"
        + seed_context,
    }
]
for turn in range(8):
    follow = POOL[turn % len(POOL)]["text"]
    if turn > 0:
        messages.append({"role": "user", "content": f"Next task (turn {turn + 1}): {follow}"})
    row = chat(messages, f"c2-turn{turn}")
    row["turn"] = turn
    # Engagement contract: from turn 2 on, the previous turn's whole-entry seed must hit.
    if ARM == "on" and turn >= 1 and (row.get("cached_tokens") or 0) == 0:
        VIOLATIONS.append(f"C2 turn{turn}: cached_tokens=0 — no warm-turn engagement")
    if ARM == "off" and (row.get("cached_tokens") or 0) > 0:
        VIOLATIONS.append(f"C2 turn{turn}: cached_tokens>0 on the OFF arm")
    looped = row["loop_content"] > 0.5 or row["loop_reasoning"] > 0.5
    row["degenerate_excluded"] = looped
    RESULTS["multiturn"].append({k: v for k, v in row.items() if k != "content"})
    messages.append({"role": "assistant", "content": row["content"] or "(empty)"})
    print(
        f"  turn{turn}: ttft={row['ttft_s']}s cached={row['cached_tokens']}/"
        f"{row['prompt_tokens']} completion={row['completion_tokens']} "
        f"loop_c={row['loop_content']} loop_r={row['loop_reasoning']}"
        + ("  [LOOPED — excluded from aggregates, reported here only]" if looped else "")
    )
print()

# C3  TTFT AT DEPTH: real pool text concatenated to three depths; request A seeds (cold),
#     request B repeats the exact bytes (hit on the ON arm). TTFT receipts for the flip.
print(BAR)
print(f"# C3 TTFT at depth (cold vs repeat), sampled shape, arm={ARM}")
print(BAR)
texts = [p["text"] for p in POOL]
for target_chars in (8_000, 16_000, 32_000):
    doc, i = [], 0
    while sum(len(t) for t in doc) < target_chars:
        doc.append(texts[i % len(texts)])
        i += 1
    prompt = [
        {
            "role": "user",
            "content": "Reference material:\n\n"
            + "\n\n".join(doc)
            + "\n\nSummarize the single most important risk in one sentence.",
        }
    ]
    cold = chat(prompt, f"c3-{target_chars}-cold", max_tokens=64)
    expect_cached(cold, cold=True, where=f"C3 {target_chars} cold")
    hit = chat(prompt, f"c3-{target_chars}-hit", max_tokens=64)
    expect_cached(hit, cold=False, where=f"C3 {target_chars} hit")
    for row, kind in ((cold, "cold"), (hit, "hit")):
        row["depth_chars"], row["kind"] = target_chars, kind
        RESULTS["depth"].append({k: v for k, v in row.items() if k != "content"})
    print(
        f"  ~{target_chars} chars ({cold['prompt_tokens']} tok): cold ttft={cold['ttft_s']}s"
        f" -> repeat ttft={hit['ttft_s']}s cached={hit['cached_tokens']}"
    )
print()

RESULTS["violations"] = VIOLATIONS
RESULTS["c1_byte_identity"] = c1_pass
open(f"{OUT}/battery.json", "w").write(json.dumps(RESULTS, indent=1))
print(BAR)
if VIOLATIONS:
    print(f"# BATTERY arm={ARM}: FAIL — {len(VIOLATIONS)} violation(s):")
    for v in VIOLATIONS:
        print(f"#   {v}")
    print(BAR)
    sys.exit(2)
print(f"# BATTERY arm={ARM}: PASS (C1 byte identity, C2 engagement, C3 receipts banked)")
print(BAR)
