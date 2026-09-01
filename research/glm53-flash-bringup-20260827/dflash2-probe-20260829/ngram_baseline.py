#!/usr/bin/env python3
"""Free-drafter floor: prompt-lookup (longest-suffix-match copy) drafting over the same
teacher-forced serving-binary greedy paths, same cycle dynamics as the DFlash2 run
(7 drafts per cycle, advance accepted+1). This is the induction-head regime that
dominates agent traffic; it costs no GPU and no training, so it floors what any
trained drafter must beat to justify itself."""
import json
import re

from tokenizers import Tokenizer

ART = "/root/models/glm53-nvfp4"
K = 7
MIN_MATCH = 2
MAX_MATCH = 16

tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
prompts = {p["name"]: p for p in json.load(open("/root/dfp2/scoring_prompts.json"))}
rollouts = json.load(open("/root/dfp2/rollouts.json"))


def tool_spans(text):
    spans = []
    for m in re.finditer(r"<tool_call>", text):
        end = text.find("</tool_call>", m.start())
        spans.append((m.start(), len(text) if end == -1 else end + len("</tool_call>")))
    for m in re.finditer(r"```", text):
        end = text.find("```", m.end())
        if end != -1:
            spans.append((m.start(), end + 3))
    for m in re.finditer(r'\{"', text):
        depth, i = 0, m.start()
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.start(), min(i + 1, len(text))))
    return spans


def classify(offset, spans):
    return "tool" if any(a <= offset < b for a, b in spans) else "prose"


def rfind_aligned(hay, needle, unit=4):
    """Last aligned occurrence of needle in hay (both bytes, unit-aligned ids)."""
    i = hay.rfind(needle)
    while i != -1 and i % unit:
        i = hay.rfind(needle, 0, i + len(needle) - 1)
    return i


def draft_ngram(buf, S, i):
    """Longest-suffix-match copy draft at position i (context S[:i], bytes in buf[:4*i])."""
    hay = buf[: 4 * (i - 1)]  # candidate occurrences must END before i-1 -> start earlier
    best = None
    lo, hi = MIN_MATCH, min(MAX_MATCH, i)
    # longest suffix of S[:i] that occurs earlier; scan down from hi for simplicity
    for L in range(hi, lo - 1, -1):
        needle = buf[4 * (i - L): 4 * i]
        j = rfind_aligned(hay, needle)
        if j != -1:
            best = j // 4 + L  # position right after the earlier occurrence
            break
    if best is None:
        return []
    return S[best: min(best + K, i)]  # copy forward, never past known context


all_cycles = []
for r in rollouts:
    name = r["name"]
    p = prompts[name]
    S = p["ids"] + r["cont_ids"]
    P = p["n_ids"]
    N = len(S)
    buf = b"".join(x.to_bytes(4, "little") for x in S)

    enc = tok.encode(r["text"], add_special_tokens=False)
    offsets = enc.offsets
    spans = tool_spans(r["text"])

    start = P
    cycles = []
    while N - start >= K + 1:
        draft = draft_ngram(buf, S, start + 1)
        truth = S[start + 1: start + 1 + K]
        L_acc = 0
        for d, t in zip(draft, truth):
            if d != t:
                break
            L_acc += 1
        produced = L_acc + 1
        j = start + 1 - P
        cls = classify(offsets[j][0], spans) if 0 <= j < len(offsets) else "prose"
        hits = [int(d == t) for d, t in zip(draft, truth)] + [0] * (K - len(draft))
        cycles.append({
            "name": name, "start": start, "class": cls, "accepted": L_acc,
            "produced": produced, "hits": hits, "draft_len": len(draft),
        })
        start += produced

    mean_acc = sum(c["accepted"] for c in cycles) / max(len(cycles), 1)
    cov = sum(1 for c in cycles if c["draft_len"]) / max(len(cycles), 1)
    print(f"{name}: cycles={len(cycles)} mean_accepted={mean_acc:.2f} draft_coverage={cov:.2f}")
    all_cycles.extend(cycles)

json.dump(all_cycles, open("/root/dfp2/cycles_ngram.json", "w"))
print(f"total cycles: {len(all_cycles)}")
