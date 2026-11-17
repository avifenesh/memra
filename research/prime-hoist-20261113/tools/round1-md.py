#!/usr/bin/env python3
"""memra#1321 review round 1: the record's tables for the fold's RTX 5090 cells, as markdown, from
the row directories under sweep/ (b-first.sh, b-fresh.sh, b-sweep.sh --pre-words, b-sweep.sh).
Arms: dmain2 = main, ds2h2 = the PR head before the fold, ds2i = the fold.
usage: round1-md.py [SWEEP_DIR]   (default ~/.cache/memra-1258/sweep)"""
import json
import re
import sys
from pathlib import Path

B = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/memra-1258/sweep"
ARMS = [("dmain2", "main"), ("ds2h2", "head before the fold"), ("ds2i", "the fold")]
SNAP = re.compile(r"\[diag1255\] (dspark-prime-[a-z-]+) id=(\S+?)(?: rows=\d+)?: driver_free=(\d+)MB pool_res=(\d+)MB "
                  r"pool_used=(\d+)MB")
PASS = re.compile(r"\[diag1255\] gate-pass id=(\S+) prompt=(\d+) path=(\w+)")


def rows_of(d: Path, long_min: int = 1000) -> list[dict]:
    """One dict per placed row of a boot directory: the row, plus the pool around its two calls."""
    if not (d / "rows.jsonl").exists():
        return []
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    log = (d / "serve.log").read_text(errors="replace").splitlines()
    ids = [m.group(1) for l in log if (m := PASS.search(l)) and int(m.group(2)) > long_min]
    placed = [i for i in ids]
    out = []
    for k, row in enumerate(rows):
        # the placed primes are the gate passes that carry the row's prompt length, in order
        want = [m.group(1) for l in log if (m := PASS.search(l)) and int(m.group(2)) == row["prompt_tokens"]]
        rid = want[0] if want else (placed[k + 1] if len(placed) > k + 1 else None)
        snaps = [(m.group(1), int(m.group(4)), int(m.group(5))) for l in log if (m := SNAP.search(l)) and m.group(2) == rid]
        after = [s for s in snaps if s[0] == "dspark-prime-chunk-after"]
        before = [s for s in snaps if s[0] == "dspark-prime-chunk-before"]
        row = dict(row)
        row["a1"] = after[0][1:] if after else None
        row["b2"] = before[1][1:] if len(before) > 1 else None
        row["dir"] = d.name
        out.append(row)
    return out


def span(vals: list[int]) -> str:
    vals = sorted(set(vals))
    if not vals:
        return "-"
    return f"{vals[0]:,}" if len(vals) == 1 else f"{vals[0]:,}-{vals[-1]:,}"


def first_row(margin: int) -> None:
    print(f"\n#### Prompt 8,173 as the first placed row, {margin} MiB\n")
    print("| arm | fresh boots | out of memory | margin seen | pool used after call 1 | pool growth in the gap |")
    print("|---|---|---|---|---|---|")
    for arm, name in ARMS:
        rs = [r for d in sorted(B.glob(f"cap-{arm}-first-8k-m{margin}/s401-r*")) for r in rows_of(d)]
        if not rs:
            continue
        fails = sum(1 for r in rs if r["oom_lines"])
        used = [r["a1"][1] for r in rs if r["a1"]]
        growth = [r["b2"][0] - r["a1"][0] for r in rs if r["a1"] and r["b2"]]
        print(f"| {name} (`{arm}`) | {len(rs)} | {fails} | {span([r['margin_seen_mib'] for r in rs])} | "
              f"{span(used)} MB | {span(growth)} MB |")


def fresh(margin: int) -> None:
    print(f"\n#### The seven 8.1k prompts, one row per fresh boot, {margin} MiB\n")
    arms = [a for a in ARMS if list(B.glob(f"cap-{a[0]}-fresh-8k-m{margin}/s*")) or list(B.glob(f"cap-{a[0]}-first-8k-m{margin}/s401-r1"))]
    print("| seed | prompt | " + " | ".join(f"{name}" for _, name in arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    for seed in range(401, 408):
        cells, prompt = [], None
        for arm, _ in arms:
            d = B / f"cap-{arm}-fresh-8k-m{margin}/s{seed}"
            if seed == 401 and not d.exists():
                d = B / f"cap-{arm}-first-8k-m{margin}/s401-r1"
            rs = rows_of(d)
            if not rs:
                cells.append("not run")
                continue
            r = rs[0]
            prompt = r["prompt_tokens"]
            cells.append(("OUT OF MEMORY" if r["oom_lines"] else "clean") + f", seen {r['margin_seen_mib']}")
        print(f"| {seed} | {prompt if prompt else '-'} | " + " | ".join(cells) + " |")


def parked_short() -> None:
    print("\n#### A short prime placed after an unplaced 5.7k prime\n")
    print("| arm | rows | out of memory | margins seen | long prime before the row | in front of the short prime's admission |")
    print("|---|---|---|---|---|---|")
    for arm, name in ARMS:
        rs = [r for d in sorted(B.glob(f"cap-{arm}-parked-short-*")) for r in rows_of(d, 100000)]
        if not rs:
            continue
        fails = sum(1 for r in rs if r["oom_lines"])
        pre = [((r.get("pre") or {}).get("prime_ws_lines") or []) for r in rs]
        parked = sum(1 for p in pre if not any("not parked" in l for l in p))
        released = sum(1 for r in rs if any("parked prime workspace released" in l for l in r.get("admit_trim", [])))
        print(f"| {name} (`{arm}`) | {len(rs)} | {fails} | {', '.join(str(r['margin_seen_mib']) for r in rs)} | "
              f"{'no workspace' if arm == 'dmain2' else f'parked in {parked} of {len(rs)}'} | "
              f"{'-' if arm == 'dmain2' else f'workspace released in {released} of {len(rs)}'} |")


def sweeps() -> None:
    print("\n#### Margin sweeps at ~5.7k tokens (m0 to m150, one boot per row group)\n")
    print("| arm | rows | out of memory | rows with an out-of-memory line |")
    print("|---|---|---|---|")
    for arm, name in ARMS:
        rs = [r for d in sorted(B.glob(f"cap-{arm}-warm-5.7k-*")) for r in rows_of(d)]
        if not rs:
            continue
        bad = [f"prompt {r['prompt_tokens']} at {r['margin_seen_mib']} MiB ({r['dir'][-1]})" for r in rs if r["oom_lines"]]
        print(f"| {name} (`{arm}`) | {len(rs)} | {len(bad)} | {'; '.join(bad) or 'none'} |")


first_row(100)
first_row(150)
fresh(100)
parked_short()
sweeps()
