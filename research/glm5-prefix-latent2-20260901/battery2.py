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
       CACHE_BATTERY_KEY_FILE (optional bearer key file),
       CACHE_BATTERY_MAX_TOKENS (default 2048 for sampled completed answers),
       PROMPTS_JSON (default /root/prompts.json)
"""

import json
import os
import sys
import uuid

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cache_qualification import (QualificationError, append_answer, completion,
                                 load_prompt_pool, same_identity)

OUT = sys.argv[1]
ARM = sys.argv[2]
assert ARM in ("on", "off", "bust"), "arm must be 'on', 'off' or 'bust'"
RAW_REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
EP = os.environ.get("EP", "http://127.0.0.1:18400")
MODEL = os.environ.get("MODEL", "zai/glm-5.3-flash")
try:
    POOL, POOL_META = load_prompt_pool(os.environ.get("PROMPTS_JSON", "/root/prompts.json"),
                                      explicit="PROMPTS_JSON" in os.environ)
except QualificationError as error:
    sys.exit(str(error))
if RAW_REPS < 2:
    sys.exit("REFUSE: byte identity requires at least two repetitions")
CHAT_MAX_TOKENS = int(os.environ.get("CACHE_BATTERY_MAX_TOKENS", "2048"))

os.makedirs(OUT, exist_ok=True)
VIOLATIONS = []
RESULTS = {"arm": ARM, "ep": EP, "model": MODEL, "pool": POOL_META, "raw": [], "rawext": [], "multiturn": [], "depth": []}
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
    # Bounded greedy tapes are byte instruments, not completed-answer claims.
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0,
            **salt(),
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
            **salt(),
    }
    row = completion(EP, body, OUT, name)
    row["loop_content"] = loopiness(row["content"])
    row["loop_reasoning"] = loopiness(row["reasoning"])
    if not row["completed_answer"]:
        VIOLATIONS.append(f"{name}: {row['verdict']}: {row['error']}")
    return row


BAR = "#" * 78

# C1  RAW BYTE ORACLE (identical-repeat / full-cover shape) — the parent lane's bar.
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
            f"prompt={row['prompt_tokens']} spec={row['spec']} finish={row['finish']} "
            f"err={row['error']}"
        )
    if not same_identity(identity_rows):
        c1_pass = False
        VIOLATIONS.append(f"C1 p{idx}: BYTE DIVERGENCE across reps: {shas}")
    print(f"  == p{idx}: {'ONE sha' if same_identity(identity_rows) else 'INVALID OR DIVERGED'} ({shas[0]})")
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
    same = seed1["identity_eligible"] and same_identity([cold2, hit2])
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
#     vendor-default sampled, no sampling overrides. Per-turn TTFT + engagement.
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
    if not row["completed_answer"] or looped:
        VIOLATIONS.append(f"C2 turn{turn}: stopped continuation after invalid/incomplete/looped answer")
        break
    append_answer(messages, row)
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
