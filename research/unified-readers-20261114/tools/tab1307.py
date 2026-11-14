#!/usr/bin/env python3
"""tab1307.py RUN_DIR...: memra#1307's tables from one or more campaigns' receipts
(campaign1307.sh layout: RUN_DIR/{probe,probe2,mtp,dflash,dflash4,dflashshort}).

- probe, probe2: per state and size, the median and range of each reading's delta (MiB) against
  the allocation;
- mtp: per cell, MemFree over the cell, every draft-state bracket's deltas on the three readings
  (raw, MemAvailable, free plus reclaimable), the high-water lines the arm printed (recorded,
  parked delta, pool peak) and the charge the cell ended with, and the rows;
- dflash, dflash4, dflashshort: per cell, the pooled forwards the lockstep planned with two or more
  live rounds: how many members they took, how many the memory cap stopped, the headroom the width
  compared against what the members taken needed beyond one serial round, and the rows;
- texts: each request's text sha256 across the cells of each part. A row with no text is counted
  as such and never as an identical text."""
import json
import re
import statistics
import sys
from pathlib import Path

runs = [Path(a) for a in sys.argv[1:]]
NUM = r"(-?\d+)"


def rows(cell):
    p = cell / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def memfree(cell):
    p = cell / "memfree.log"
    vals = []
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.search(r"MemFree: (\d+)", line)
            if m:
                vals.append(int(m.group(1)))
    return f"{min(vals):,} to {max(vals):,}" if vals else "-"


def cells(part):
    out = []
    for run in runs:
        d = run / part
        if d.exists():
            out.extend(sorted(p for p in d.iterdir() if p.is_dir()))
    return out


def ok_rows(rs):
    return f"{sum(r['status'] == 200 and not r.get('error') for r in rs)}/{len(rs)}"


def trace(cell):
    p = cell / "trace.txt"
    return p.read_text() if p.exists() else ""


def ooms(tr):
    return len(re.findall(r"out of memory|OUT_OF_MEMORY", tr))


def probe(part):
    data = []
    for run in runs:
        p = run / part / "probe.jsonl"
        if p.exists():
            data.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    if not data:
        return
    print(f"## {part}\n")
    print("| state | size MiB | n | raw | MemAvailable | free + reclaimable | MemFree | Cached | MemFree before, MiB |")
    print("|---|---:|---:|---|---|---|---|---|---|")
    for state in ("quiet", "full", "live"):
        for size in sorted({d["size_mib"] for d in data}):
            ds = [d for d in data if d["label"] == state and d["size_mib"] == size]
            if not ds:
                continue

            def cell(v):
                return f"{statistics.median(v):+.1f} ({min(v):+.1f} to {max(v):+.1f})"

            def col(k):
                return cell([d["delta_mib"][k] for d in ds])

            fpr = [sum(d["delta_mib"].get(k, 0.0) for k in ("MemFree", "Buffers", "Cached", "SReclaimable"))
                   for d in ds]
            mf = [d["before_mib"]["MemFree"] for d in ds]
            print(f"| {state} | {size} | {len(ds)} | {col('raw')} | {col('MemAvailable')} | {cell(fpr)} | "
                  f"{col('MemFree')} | {col('Cached')} | {min(mf):,.0f} to {max(mf):,.0f} |")
    print("\nDeltas in MiB, median (min to max); a reading that tracks the allocation reads its size.\n")


def mtp():
    cs = cells("mtp")
    if not cs:
        return
    print("## mtp: the draft-state bracket\n")
    print("| cell | state | MemFree during the cell, MB | brackets: raw / MemAvailable / free + reclaimable delta, MiB | "
          "high-water lines (recorded / parked / pool peak, MB) | charge at the end, MB | rows ok | OOM lines |")
    print("|---|---|---|---|---|---:|---|---:|")
    for cell in cs:
        tr = trace(cell)
        br = re.findall(r"draft-bracket .*raw_delta_kib=" + NUM + r" avail_delta_kib=" + NUM
                        + r" fpr_delta_kib=" + NUM, tr)
        hw = re.findall(r"draft-session state high-water: (\d+)MB \(max of parked delta (\d+)MB and "
                        r"capture-time pool peak (\d+)MB", tr)
        st = re.search(r"STATE=(\w+)", (cell.parent / "campaign.log").read_text()
                       .split(f" {cell.name} start ", 1)[-1])
        print(f"| {cell.name} | {st.group(1) if st else '-'} | {memfree(cell)} | "
              f"{', '.join('/'.join(f'{int(x) / 1024:.1f}' for x in b) for b in br) or '-'} | "
              f"{', '.join('/'.join(h) for h in hw) or '-'} | {max((int(h[0]) for h in hw), default=0)} | "
              f"{ok_rows(rows(cell))} | {ooms(tr)} |")
    print()


def dflash(part):
    cs = cells(part)
    if not cs:
        return
    print(f"## {part}: the pooled verify width\n")
    print("| cell | state | MemFree during the cell, MB | forwards with 2+ live rounds | members taken (count x members) | "
          "stopped by the width | headroom MB (min to max) | raw free MB | MemAvailable MB | "
          "extra the members taken needed, MB (max) | need the width refused, MB | column KiB | cap lines | rows ok | OOM lines |")
    print("|---|---|---|---:|---|---:|---|---|---|---:|---|---:|---:|---|---:|")
    for cell in cs:
        tr = trace(cell)
        fw = []
        for m in re.finditer(r"dspark-pool lead=\d+ live=(\d+) members=(\d+) rows=\d+ headroom_mb=(\d+) "
                             r"raw_free_mb=(\d+) mem_available_mb=(\d+) cached_mb=(\d+) capped_extra_mb=(\S+)"
                             r"(?: extra_kib=(\d+) col_kib=(\d+))?", tr):
            live, mem, head, raw, av, _c, cap, extra, colk = m.groups()
            fw.append({"live": int(live), "mem": int(mem), "head": int(head), "raw": int(raw), "av": int(av),
                       "cap": None if cap == "None" else int(re.sub(r"\D", "", cap)),
                       "extra": int(extra) if extra else None, "col": int(colk) if colk else None})
        multi = [f for f in fw if f["live"] >= 2]
        st = re.search(r"STATE=(\w+)", (cell.parent / "campaign.log").read_text()
                       .split(f" {cell.name} start ", 1)[-1])

        def rng(k):
            v = [f[k] for f in multi]
            return f"{min(v):,} to {max(v):,}" if v else "-"
        dist = {}
        for f in multi:
            dist[f["mem"]] = dist.get(f["mem"], 0) + 1
        caps = [f["cap"] for f in multi if f["cap"] is not None]
        extra = [f["extra"] for f in multi if f["extra"] is not None]
        col = next((f["col"] for f in fw if f["col"] is not None), None)
        print(f"| {cell.name} | {st.group(1) if st else '-'} | {memfree(cell)} | {len(multi)} | "
              f"{', '.join(f'{n} x {k}' for k, n in sorted(dist.items())) or '-'} | {len(caps)} | {rng('head')} | "
              f"{rng('raw')} | {rng('av')} | {max(extra) // 1024 if extra else '-'} | "
              f"{f'{min(caps):,} to {max(caps):,}' if caps else '-'} | {f'{col:,}' if col else '-'} | "
              f"{len(re.findall(r'capped by VRAM headroom', tr))} | {ok_rows(rows(cell))} | {ooms(tr)} |")
    print()


def identity():
    for part in ("mtp", "dflash", "dflash4", "dflash4low", "dflashshort"):
        by = {}
        for cell in cells(part):
            for r in rows(cell):
                by.setdefault(r["req"], {})[cell.name] = r
        if not by:
            continue
        print(f"## {part}: texts\n")
        for req, cs in sorted(by.items()):
            empty = [c for c, r in cs.items() if not r.get("text_chars")]
            shas = {r["sha256"] for c, r in cs.items() if c not in empty}
            one = next(iter(cs.values()))
            toks = {r["completion_tokens"] for r in cs.values()}
            fins = {r["finish"] for r in cs.values()}
            chars = {r.get("text_chars") for r in cs.values()}
            print(f"- {req}: {len(cs)} cells, {len(shas)} distinct text(s)"
                  + (f", {len(empty)} WITHOUT TEXT ({', '.join(empty)})" if empty else "")
                  + f": sha256 {', '.join(sorted(s[:12] for s in shas))}; completion tokens {sorted(toks)}, "
                  f"finish {sorted(map(str, fins))}, characters {sorted(map(str, chars))}, prompt tokens {one['prompt_tokens']}")
        print()


probe("probe")
probe("probe2")
mtp()
for part in ("dflash", "dflash4", "dflash4low", "dflashshort"):
    dflash(part)
identity()
