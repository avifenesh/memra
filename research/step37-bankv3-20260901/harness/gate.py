#!/usr/bin/env python3
"""Milestone-4 gates (b) and (c): end-to-end GREEDY byte identity, and the short-prompt
output-content oracle.

Greedy here is the INSTRUMENT, never the product (owner law, 2026-08-21). It is used for
exactly one reason: argmax is byte-deterministic, so "the arms produce the same bytes" is a
decidable question. The priced rows are sampled and live in price.py.

Why this gate is shaped the way it is — every clause below is a lesson from the incident this
lane exists to close (research/step37-bankv3-20260901/DIAGNOSIS.md):

  * PREFILL-HEAVY SHAPES ARE MANDATORY. Both defects the slot-major layout was blamed for
    lived in a PREFILL kernel, and no gate the door ever had ran one. A decode-only tape is
    the exact blind spot that shipped fluent wrong text. So the corpus mixes short prompts
    with multi-thousand-token ones, and the long ones are the reason this file exists.

  * SHORT PROMPTS ARE MANDATORY, AND THEY ARE THE MARGIN INSTRUMENT. The corruption was
    equally present at 613 tokens and merely INVISIBLE there: a wrong per-16 scale on correct
    codes is a bounded perturbation, so the argmax flips only where the top-1 margin is
    narrow. Every gate prompt in the original qualification was >= 613 tokens, which is why a
    length-indexed ladder could not find it. The 25-token arithmetic prompt that DID catch it
    is the first entry in the short corpus and it stays there.

  * CONTENT IS CHECKED, NOT JUST EQUALITY. step37 is a thinking model whose bytes arrive in
    `reasoning_content`; a reader that watches only `content` sees two empty strings and
    reports a PASS. So a completion that is empty, or that does not contain the expected
    answer on the arithmetic probes, FAILS loudly rather than comparing equal. This is the
    "decisive probe" rule: assert on fields before consuming them.

  * SPEC IS OFF for these arms (launch.sh gate-* modes). With spec on, the tape depends on
    draft/verify scheduling and the gate would be measuring the scheduler.

Usage: gate.py <arm-tag> <out.json>
Exit 0 = every prompt produced non-empty, expected content. The CALLER compares two arms'
output files for byte identity (compare.py) — this script's job is one arm's tape.
"""
import hashlib, json, sys, time, urllib.request

ARM, OUT = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:18640/v1/chat/completions"
MODEL = "stepfun/step-3.7-flash"
D = "/home/ubuntu/bankv3/lane"

# SHORT, MARGIN-SENSITIVE PROBES. The first is verbatim the prompt whose first generated token
# was `Ass` with the door on and `Got` with it off, at 25 prompt tokens. `expect` is the
# can't-hallucinate check: a corrupted routed-expert FFN produced fluent text that never
# reached the number.
SHORT = [
    {"p": "What is 17*23? Reply with the number only.", "expect": "391"},
    {"p": "What is 31*29? Reply with the number only.", "expect": "899"},
    {"p": "What is 144/12? Reply with the number only.", "expect": "12"},
    {"p": "Name the capital of Japan. One word.", "expect": "Tokyo"},
    {"p": "What is 2^10? Reply with the number only.", "expect": "1024"},
]


def greedy(prompt, maxtok):
    # GREEDY, explicitly: temperature 0 and top_p 1 override the registry's vendor sampling
    # defaults. That is correct HERE and only here — this request shape is the oracle, not the
    # product, and the file's docstring says so.
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    text, first, usage, fp, finish = [], None, None, None, None
    r = urllib.request.urlopen(
        urllib.request.Request(
            URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        ),
        timeout=1800,
    )
    while True:
        line = r.readline()
        if not line:
            break
        s = line.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        p = s[5:].strip()
        if p == "[DONE]":
            continue
        try:
            j = json.loads(p)
        except Exception:
            continue
        fp = j.get("system_fingerprint") or fp
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
            if piece:
                text.append(piece)
                if first is None:
                    first = time.perf_counter() - t0
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    r.close()
    u = usage or {}
    full = "".join(text)
    return {
        "prompt_chars": len(prompt),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish_reason": finish,
        "ttft_s": round(first, 4) if first is not None else None,
        "wall_s": round(time.perf_counter() - t0, 4),
        "out": full,
        "out_sha256": hashlib.sha256(full.encode()).hexdigest(),
        "first_32": full[:32],
        "fingerprint": fp,
    }


receipt = dict(
    l.strip().split("=", 1)
    for l in open("%s/receipts/boot-%s.receipt" % (D, ARM))
    if "=" in l and not l.startswith(" ")
)
res = {
    "arm": ARM,
    "bin_md5": receipt.get("bin_md5"),
    "boot_nonce": receipt.get("boot_nonce"),
    "built_from": receipt.get("built_from"),
    "cells": {},
}
ok = True

# ── (c) the short-prompt output-content oracle ─────────────────────────────────────────────
for i, probe in enumerate(SHORT):
    row = greedy(probe["p"], 256)
    nonempty = len(row["out"].strip()) > 0
    hit = probe["expect"].lower() in row["out"].lower()
    row["content_nonempty"] = nonempty
    row["expect"] = probe["expect"]
    row["expect_found"] = hit
    cell_ok = nonempty and hit
    ok &= cell_ok
    res["cells"]["short%d" % i] = row
    print(
        "[short%d] pt=%s ct=%s nonempty=%s expect=%r found=%s first32=%r sha=%s => %s"
        % (
            i,
            row["prompt_tokens"],
            row["completion_tokens"],
            nonempty,
            probe["expect"],
            hit,
            row["first_32"],
            row["out_sha256"][:16],
            "OK" if cell_ok else "FAIL",
        )
    )

# ── (b) end-to-end greedy tape, INCLUDING prefill-heavy shapes ─────────────────────────────
# The long corpora are the point: they drive the grouped-GEMM prime, which is where both
# defects lived and where no v2 gate had ever looked.
# agentic8: 8 real agentic turns (317-1523 chars), the decode-shaped corpus every step37
# receipt uses. prefill-ladder: the SAME real conversation replayed at 5 growing prefill
# depths (~250 / 682 / 1566 / 3182 / 4729 prompt tokens), which is what drives the grouped-GEMM
# prime through nkb>1 and multiple CSR group-size regimes. The 30k pair is deliberately out:
# the incident's blind spot was "no prefill gate at all", not "not deep enough", and a greedy
# 30k prime per arm is card-minutes for a shape whose smaller siblings hit the same kernel.
for corpus, maxtok in (("agentic8.json", 256), ("prefill-ladder.json", 96)):
    prompts = json.load(open("%s/harness/%s" % (D, corpus)))
    for i, p in enumerate(prompts):
        row = greedy(p, maxtok)
        nonempty = len(row["out"].strip()) > 0
        row["content_nonempty"] = nonempty
        ok &= nonempty
        # keep the tape lean in the banked json; the sha is the gate, the text is evidence
        res["cells"]["%s#%d" % (corpus, i)] = row
        print(
            "[%s#%d] pt=%s ct=%s nonempty=%s ttft=%s first32=%r sha=%s"
            % (
                corpus,
                i,
                row["prompt_tokens"],
                row["completion_tokens"],
                nonempty,
                row["ttft_s"],
                row["first_32"],
                row["out_sha256"][:16],
            )
        )
        if not nonempty:
            print("[%s#%d] FAIL: empty completion — a comparison of two empties is not a PASS" % (corpus, i))

# One tape hash over every cell, in a fixed key order, so two arms compare in one number.
tape = hashlib.sha256(
    "\n".join("%s=%s" % (k, res["cells"][k]["out_sha256"]) for k in sorted(res["cells"])).encode()
).hexdigest()
res["tape_sha256"] = tape
res["all_content_ok"] = ok
json.dump(res, open(OUT, "w"), indent=1)
print("TAPE arm=%s cells=%d tape_sha256=%s content_ok=%s" % (ARM, len(res["cells"]), tape, ok))
sys.exit(0 if ok else 1)
