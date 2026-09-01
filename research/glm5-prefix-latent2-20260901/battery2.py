#!/usr/bin/env python3
"""Slot-B qualification battery for the glm5 cache fix (lane/glm5-prefix-latent2).
Extends the parent lane's battery.py (research/glm5-prefix-latent-20260830/) for the
SPEC serving shape: the box boots MEMRA_GLM5_SPEC=1 + MEMRA_GLM5_DFLASH + the ship
recipe (3-card PP3 SPLITS=15,30) on EVERY arm; only the cache flags differ.

Arms (one server boot each, run this script once per arm):

  on    MEMRA_PREFIX_LATENT=1 MEMRA_HYPER_SUFFIX_PRIME=1 MEMRA_GLM5_SPEC_PREFIX=1
        MEMRA_PREFIX_CACHE_MB=4096. PASS = C1 byte identity w/ engagement, C1b
        restored-suffix CONTINUATION byte identity, C2 warm-turn engagement from turn 2,
        C3 hit TTFT receipts.
  off   cache flags unset (any budget). PASS = cached_tokens == 0 everywhere, one sha
        per raw prompt, refusal lines present in the server log (grepped outside).
  bust  same flags as `on` but every request rotates cache_salt — the cache-bust
        control: cached_tokens == 0 everywhere proves C2/C3's `on`-arm gains are the
        cache, not the boot.

Server-log greps (RUNBOOK-SLOTB.md) complete each arm: [suffix-prime] ENGAGED /
DECLINED / TOKENWISE, [glm5-spec] route=... restored=, [glm5-acc] (spec engagement per
the never-serve-greedy law), [prefix-cache] inserts (why=glm5-boundary), and the OFF
arm's snapshot-refusal lines.

PRECONDITIONS ARE ASSERTED, NOT ASSUMED (parent lane law): any row violating its arm's
cache-engagement expectation is named and the script exits 2.

usage: battery2.py <outdir> <on|off|bust> [raw_reps]
env:   EP (default http://127.0.0.1:18400), MODEL (default zai/glm-5.3-flash),
       PROMPTS_JSON (default /root/prompts.json)
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid

OUT = sys.argv[1]
ARM = sys.argv[2]
assert ARM in ("on", "off", "bust"), "arm must be 'on', 'off' or 'bust'"
RAW_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
EP = os.environ.get("EP", "http://127.0.0.1:18400")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")
POOL = json.load(open(os.environ.get("PROMPTS_JSON", "/root/prompts.json")))["decode"]
os.makedirs(OUT, exist_ok=True)
VIOLATIONS = []
RESULTS = {"arm": ARM, "ep": EP, "model": MODEL, "raw": [], "rawext": [], "multiturn": [], "depth": []}
BUST = ARM == "bust"


def salt():
    """Rotating per-request cache namespace on the bust arm; stable default otherwise."""
    return {"cache_salt": f"bust-{uuid.uuid4().hex[:12]}"} if BUST else {}


def loopiness(s, w=48):
    if len(s) < 4 * w:
        return 0.0
    tail = s[-2000:]
    seen, best = {}, 0
    for i in range(0, len(tail) - w):
        k = tail[i : i + w]
        seen[k] = seen.get(k, 0) + 1
        best = max(best, seen[k])
    return round(best * w / len(tail), 3)


def expect_cached(row, cold, where, full_cover=True):
    c = row.get("cached_tokens") or 0
    if ARM in ("off", "bust") and c > 0:
        VIOLATIONS.append(f"{where}: cached_tokens={c} but the {ARM} arm must never restore")
    if ARM == "on" and not cold:
        if full_cover and c != (row.get("prompt_tokens") or -1):
            VIOLATIONS.append(
                f"{where}: cached_tokens={c} != prompt_tokens={row.get('prompt_tokens')} — "
                "the restored rep did not take a whole-entry hit (eviction? budget? guard?)"
            )
        if not full_cover and c == 0:
            VIOLATIONS.append(f"{where}: cached_tokens=0 — the strict-prefix hit did not engage")
    if ARM == "on" and cold and c > 0:
        VIOLATIONS.append(f"{where}: cold rep reports cached_tokens={c}; the cell is contaminated")


def raw_completion(prompt, name, max_tokens=64, extra=None):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
        **salt(),
        **(extra or {}),
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
        "spec": (u.get("spec") or {}),
        "error": r.get("__error__"),
    }
    open(f"{OUT}/{name}.json", "w").write(json.dumps(r, indent=1))
    return row


def chat(messages, name, max_tokens=512):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
        **salt(),
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
        "spec": (u.get("spec") or {}),
        "loop_content": loopiness(text),
        "loop_reasoning": loopiness(thought),
        "content": text,
    }


BAR = "#" * 78

# C1  RAW BYTE ORACLE (identical-repeat / full-cover shape) — the parent lane's bar.
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
            f"prompt={row['prompt_tokens']} spec={row['spec']} finish={row['finish']} "
            f"err={row['error']}"
        )
    if len(set(shas)) != 1:
        c1_pass = False
        VIOLATIONS.append(f"C1 p{idx}: BYTE DIVERGENCE across reps: {shas}")
    print(f"  == p{idx}: {'ONE sha' if len(set(shas)) == 1 else 'DIVERGED'} ({shas[0]})")
print()

# C1b RESTORED-SUFFIX CONTINUATION BYTE ORACLE (THIS lane's bar — the strict-prefix shape
#     the parent lane never byte-gated): within ONE boot, (1) serve P2 = prefix+suffix
#     COLD and record its bytes, (2) serve P1 = prefix (seeds the entry at its boundary),
#     (3) serve P2 again — now a strict-prefix hit whose suffix primes through the
#     continuation program (plain route) or re-arms the spec session (spec route).
#     PASS = step (3)'s bytes == step (1)'s, with engagement (cached>0, < prompt).
#     Order matters: P2-cold must run BEFORE P1 seeds, or the "cold" row is a hit.
print(BAR)
print(f"# C1b restored-suffix continuation byte oracle, arm={ARM}")
print(BAR)
c1b_pass = True
for idx, jdx in ((5, 6), (7, 3)):
    prefix = POOL[idx]["text"]
    suffix = "\n\nFollow-up (answer directly, no preamble): " + POOL[jdx]["text"][:600]
    p2 = prefix + suffix
    # The cold reference runs in its OWN cache namespace: served in the default one it
    # would SEED a P2 entry and turn the later "hit" into a FULL-COVER resume, never
    # exercising the suffix prime (desk-check catch, 2026-09-01). cache_salt partitions
    # lookup visibility only — the numeric program is identical, so the bytes compare.
    cold2 = raw_completion(
        p2, f"c1b-p{idx}-p2cold", extra={"cache_salt": f"c1b-coldref-{uuid.uuid4().hex[:8]}"}
    )
    expect_cached(cold2, cold=True, where=f"C1b p{idx} p2cold")
    seed1 = raw_completion(prefix, f"c1b-p{idx}-p1seed")
    expect_cached(seed1, cold=True, where=f"C1b p{idx} p1seed")
    hit2 = raw_completion(p2, f"c1b-p{idx}-p2hit")
    expect_cached(hit2, cold=False, where=f"C1b p{idx} p2hit", full_cover=False)
    if ARM == "on":
        c = hit2.get("cached_tokens") or 0
        pt = hit2.get("prompt_tokens") or 0
        if not (0 < c < pt):
            VIOLATIONS.append(
                f"C1b p{idx}: cached={c} of {pt} — expected a STRICT-prefix hit "
                "(0 < cached < prompt); a full-cover or zero row does not exercise the "
                "suffix prime"
            )
    for r in (cold2, seed1, hit2):
        RESULTS["rawext"].append(r)
    same = hit2["out_sha16"] == cold2["out_sha16"]
    if not same:
        c1b_pass = False
        VIOLATIONS.append(
            f"C1b p{idx}: CONTINUATION BYTES DIVERGED: cold={cold2['out_sha16']} "
            f"restored={hit2['out_sha16']}"
        )
    print(
        f"  p{idx}: cold sha={cold2['out_sha16']} -> restored sha={hit2['out_sha16']} "
        f"cached={hit2['cached_tokens']}/{hit2['prompt_tokens']} "
        f"wall {cold2['wall_s']}s -> {hit2['wall_s']}s  {'OK' if same else 'DIVERGED'}"
    )
print()

# C2  THE OWNER-LAW MULTITURN TWIN: 8 turns, larger prompt, full history resent per turn,
#     vendor-default sampled, reasoning_effort pinned. Per-turn TTFT + engagement.
#     Turn messages are SAVED per turn so the C4 entry-digest compare (runbook) can
#     replay the exact rendered prompt on a fresh boot.
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
    open(f"{OUT}/c2-messages-turn{turn}.json", "w").write(json.dumps(messages, indent=1))
    row = chat(messages, f"c2-turn{turn}")
    row["turn"] = turn
    if ARM == "on" and turn >= 1 and (row.get("cached_tokens") or 0) == 0:
        VIOLATIONS.append(f"C2 turn{turn}: cached_tokens=0 — no warm-turn engagement")
    if ARM in ("off", "bust") and (row.get("cached_tokens") or 0) > 0:
        VIOLATIONS.append(f"C2 turn{turn}: cached_tokens>0 on the {ARM} arm")
    looped = row["loop_content"] > 0.5 or row["loop_reasoning"] > 0.5
    row["degenerate_excluded"] = looped
    RESULTS["multiturn"].append({k: v for k, v in row.items() if k != "content"})
    messages.append({"role": "assistant", "content": row["content"] or "(empty)"})
    print(
        f"  turn{turn}: ttft={row['ttft_s']}s cached={row['cached_tokens']}/"
        f"{row['prompt_tokens']} completion={row['completion_tokens']} spec={row['spec']} "
        f"loop_c={row['loop_content']} loop_r={row['loop_reasoning']}"
        + ("  [LOOPED — excluded from aggregates, reported here only]" if looped else "")
    )
print()

# C3  TTFT AT DEPTH (full-cover repeat shape): request A seeds (cold), request B repeats
#     the exact bytes. On the ON arm the repeat is a whole-entry hit (plain boundary
#     resume — near-instant, the parent lane's C3-green class).
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
RESULTS["c1b_continuation_byte_identity"] = c1b_pass
open(f"{OUT}/battery.json", "w").write(json.dumps(RESULTS, indent=1))
print(BAR)
if VIOLATIONS:
    print(f"# BATTERY arm={ARM}: FAIL — {len(VIOLATIONS)} violation(s):")
    for v in VIOLATIONS:
        print(f"#   {v}")
    print(BAR)
    sys.exit(2)
print(
    f"# BATTERY arm={ARM}: PASS (C1 byte identity, C1b continuation byte identity, "
    "C2 engagement, C3 receipts banked)"
)
print(BAR)
