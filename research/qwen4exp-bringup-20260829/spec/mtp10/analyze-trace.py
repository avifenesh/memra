#!/usr/bin/env python3
"""mtp10 trace analysis: accept decay, fork content classes, margins, carrier drift.

Reads spec-trace TSVs (qwen4exp_real_gate --spec-trace) and reports, per file:
  - accept-length vs generation position (32-token buckets) + zero-accept share
  - the token CLASS at fork rows (what the target wanted vs what the draft proposed)
  - fork margins (draft top1-top2, target top1-top2), draft rank of the target token
  - carrier drift (rel L2 of the draft seed vs the trunk's true wide) by chain step
    and split by round outcome
  - think-block boundary: mean accept before vs after </think> per prompt

Usage: analyze-trace.py <artifact_dir> <spec-trace.tsv> [more.tsv ...]
"""
import sys
from collections import defaultdict

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])


def tclass(tid):
    s = tok.decode([int(tid)])
    st = s.strip()
    if "</think>" in s or "<think>" in s:
        return "think_tag"
    if st == "":
        return "ws_newline" if "\n" in s else "ws_space"
    if all(c.isdigit() or c in ".,-+" for c in st):
        return "digit"
    if st and all(not c.isalnum() for c in st):
        return "punct"
    if s.startswith(" "):
        return "word_start"
    return "word_cont"


for path in sys.argv[2:]:
    rows = []
    for line in open(path):
        if line.startswith("#") or line.startswith("prompt\t") or not line.strip():
            continue
        rows.append(line.rstrip("\n").split("\t"))
    print(f"\n==== {path} ({len(rows)} rounds) ====")

    buckets = defaultdict(lambda: [0, 0, 0])
    for f in rows:
        b = int(f[2]) // 32 * 32
        a = int(f[5])
        buckets[b][0] += 1
        buckets[b][1] += a
        buckets[b][2] += 1 if a == 0 else 0
    print("gen_pos_bucket  rounds  mean_accept  zero_share")
    for b in sorted(buckets):
        r, s, z = buckets[b]
        print(f"  {b:4d}-{b + 31:4d}   {r:4d}    {s / r:5.2f}       {z / r:.2f}")

    cls = defaultdict(int)
    clsd = defaultdict(int)
    ent_fork, ent_n = 0.0, 0
    dmargin, tmargin, dranks = [], [], []
    for f in rows:
        k, a = int(f[4]), int(f[5])
        if a < k:
            tgt = f[7].split(",")[a]
            drf = f[6].split(",")[a]
            cls[tclass(tgt)] += 1
            clsd[tclass(drf)] += 1
            dmargin.append(float(f[8]) - float(f[9]))
            tmargin.append(float(f[12]) - float(f[13]))
            dranks.append(int(f[11]))
            ent_fork += float(f[15])
            ent_n += 1
    n = max(sum(cls.values()), 1)
    print(f"fork rows: {n}; TARGET token class at fork:")
    for c, v in sorted(cls.items(), key=lambda x: -x[1]):
        print(f"  {c:11s} {v:4d} ({v / n:.2f})")
    print("DRAFT (wrong) token class at fork:")
    for c, v in sorted(clsd.items(), key=lambda x: -x[1]):
        print(f"  {c:11s} {v:4d} ({v / n:.2f})")

    def med(v):
        return sorted(v)[len(v) // 2] if v else float("nan")

    print(
        f"fork medians: draft_margin={med(dmargin):.3f} target_margin={med(tmargin):.3f} "
        f"draft_rank_of_target={med(dranks)} target_entropy={ent_fork / max(ent_n, 1):.3f}"
    )
    rk = sorted(dranks)
    if rk:
        def q(p):
            return rk[min(len(rk) - 1, int(p * len(rk)))]

        print(
            f"draft_rank_of_target quartiles: p25={q(0.25)} p50={q(0.5)} p75={q(0.75)} "
            f"p90={q(0.9)} max={rk[-1]}"
        )
        in32k = sum(1 for r in rk if r < 32768) / len(rk)
        in5k = sum(1 for r in rk if r < 5538) / len(rk)
        print(f"rank<32768 share: {in32k:.3f}; rank<5538 share: {in5k:.3f}")

    by_step = defaultdict(list)
    first_by_accept = defaultdict(list)
    for f in rows:
        a = int(f[5])
        if len(f) > 16 and f[16]:
            drifts = [float(x) for x in f[16].split(",") if x]
            for j, d in enumerate(drifts):
                by_step[j].append(d)
            if drifts:
                first_by_accept[0 if a == 0 else 1].append(drifts[0])
    print(
        "carrier rel_l2 by chain step j:",
        {j: f"{sum(v) / len(v):.4f}" for j, v in sorted(by_step.items())},
    )
    for kk, v in sorted(first_by_accept.items()):
        tag = "==0" if kk == 0 else ">0"
        print(f"  first-seed drift when accept{tag}: mean {sum(v) / len(v):.4f} (n={len(v)})")

    per_prompt = defaultdict(list)
    for f in rows:
        per_prompt[f[0]].append(f)
    close_id = None
    ids = tok.encode("</think>", add_special_tokens=False)
    if len(ids) == 1:
        close_id = ids[0]
    for p, rs in sorted(per_prompt.items()):
        stream = []
        for f in rs:
            a = int(f[5])
            stream.extend(int(x) for x in f[7].split(",")[: a + 1])
        pos = stream.index(close_id) if (close_id is not None and close_id in stream) else -1
        pre = [int(f[5]) for f in rs if pos >= 0 and int(f[2]) < pos]
        post = [int(f[5]) for f in rs if pos >= 0 and int(f[2]) >= pos]
        pm = sum(pre) / len(pre) if pre else float("nan")
        qm = sum(post) / len(post) if post else float("nan")
        za = sum(1 for f in rs if int(f[5]) == 0) / len(rs)
        print(
            f"prompt {p}: rounds={len(rs)} zero_share={za:.2f} think_close_at={pos} "
            f"mean_accept pre={pm:.2f} post={qm:.2f}"
        )
