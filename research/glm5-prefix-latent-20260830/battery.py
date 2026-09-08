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
       CACHE_BATTERY_KEY_FILE (optional bearer key file),
       CACHE_BATTERY_MAX_TOKENS (default 2048 for sampled completed answers),
       PROMPTS_JSON (default /home/ubuntu/prompts.json)
"""

import json
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cache_qualification import (QualificationError, append_answer, completion,
                                 load_prompt_pool, same_identity)

OUT = sys.argv[1]
ARM = sys.argv[2]
assert ARM in ("on", "off"), "arm must be 'on' or 'off'"
RAW_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
EP = os.environ.get("EP", "http://127.0.0.1:18400")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")
try:
    POOL, POOL_META = load_prompt_pool(os.environ.get("PROMPTS_JSON", "/home/ubuntu/prompts.json"),
                                      explicit="PROMPTS_JSON" in os.environ)
except QualificationError as error:
    sys.exit(str(error))
if RAW_REPS < 2:
    sys.exit("REFUSE: byte identity requires at least two repetitions")
CHAT_MAX_TOKENS = int(os.environ.get("CACHE_BATTERY_MAX_TOKENS", "2048"))

os.makedirs(OUT, exist_ok=True)
VIOLATIONS = []
RESULTS = {"arm": ARM, "ep": EP, "model": MODEL, "pool": POOL_META, "raw": [], "multiturn": [], "depth": []}


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


def raw_completion(prompt, name, max_tokens=64, extra=None):
    # Bounded greedy tapes are byte instruments, not completed-answer claims.
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0,
            **(extra or {})}
    row = completion(EP, body, OUT, name, raw_tape=True)
    if not row["identity_eligible"]:
        VIOLATIONS.append(f"{name}: invalid byte-oracle row: {row['error']}")
    return row


def chat(messages, name, max_tokens=None):
    # No sampling or reasoning overrides: use the model's vendor defaults.
    body = {"model": MODEL, "messages": messages,
            "max_tokens": CHAT_MAX_TOKENS if max_tokens is None else max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
    }
    row = completion(EP, body, OUT, name)
    row["loop_content"] = loopiness(row["content"])
    row["loop_reasoning"] = loopiness(row["reasoning"])
    if not row["completed_answer"]:
        VIOLATIONS.append(f"{name}: {row['verdict']}: {row['error']}")
    return row


BAR = "#" * 78

# C1  RAW BYTE ORACLE, p5 and p7: rep 0 cold, reps 1..N-1 restored (arm on) or refused
#     (arm off). THE acceptance bar: ONE sha per prompt across every rep.
print(BAR)
print(f"# C1 raw /v1/completions greedy, p5/p7, rep0 cold + {RAW_REPS - 1} repeats, arm={ARM}")
print(BAR)
c1_pass = True
for idx in (5, 7):
    shas, identity_rows = [], []
    for rep in range(RAW_REPS):
        row = raw_completion(POOL[idx]["text"], f"c1-p{idx}-rep{rep}")
        row["idx"], row["rep"] = idx, rep
        expect_cached(row, cold=(rep == 0), where=f"C1 p{idx} rep{rep}")
        RESULTS["raw"].append(row)
        shas.append(row["out_sha16"])
        identity_rows.append(row)
        print(
            f"  p{idx} rep{rep}: sha={row['out_sha16']} cached={row['cached_tokens']} "
            f"prompt={row['prompt_tokens']} finish={row['finish']} err={row['error']}"
        )
    if not same_identity(identity_rows):
        c1_pass = False
        VIOLATIONS.append(f"C1 p{idx}: BYTE DIVERGENCE across reps: {shas}")
    print(f"  == p{idx}: {'ONE sha' if same_identity(identity_rows) else 'INVALID OR DIVERGED'} ({shas[0]})")
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
    if not row["completed_answer"] or looped:
        VIOLATIONS.append(f"C2 turn{turn}: stopped continuation after invalid/incomplete/looped answer")
        break
    append_answer(messages, row)
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
    cold = chat(prompt, f"c3-{target_chars}-cold", max_tokens=CHAT_MAX_TOKENS)
    expect_cached(cold, cold=True, where=f"C3 {target_chars} cold")
    hit = chat(prompt, f"c3-{target_chars}-hit", max_tokens=CHAT_MAX_TOKENS)
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
RESULTS["completed_answer_pass"] = not VIOLATIONS
RESULTS["verdict"] = "FAIL" if VIOLATIONS else "PASS"
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
