#!/usr/bin/env python3
"""openout_tab.py CAMPAIGN_DIR (memra#1221 part 2 review round 1, should-fix 7): the served
open-output cells (openout_cell.sh). Per cell (NN-B main, NN-T the review head) and per open-output
request (open-a, then open-b), d_tab.py's readings of the first burst boundary (stall, publisher),
then what the head's cache did around them: the compactions (`[prefix-loan] compact:` lines, one
per plane: count, bytes held and copied, scheduler-thread time summed) inside each request's
window, the last `[prefix-loan] reconcile` line before the request ends (charged against real,
adopted), the `[specpub]` publication's time, and the pool's used bytes."""
import json
import pathlib
import re
import subprocess
import sys

d = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).resolve().parent
comp = re.compile(r"(\d+\.\d+) \[prefix-loan\] compact: plane \d+ held ([\d.]+)MB, its readers copied ([\d.]+)MB \(([\d.]+)MB read\) us=(\d+)")
fail = re.compile(r"(\d+\.\d+) \[prefix-loan\] compact: plane \d+ .*stopped after")
rec = re.compile(r"(\d+\.\d+) \[prefix-loan\] reconcile why=(\S+) books=(\d+) lent_live=(\d+) adopted=([\d.]+)MB "
                 r"charged=([\d.]+)MB real=([\d.]+)MB adoptions=(\d+) compactions=(\d+) pool_used=([\d.]+)MB")
spub = re.compile(r"(\d+\.\d+) \[specpub\] (\w+) publication why=spec-boundary pos=(\d+) lent=([\d.]+)MB us=(\d+) pool_reserved=(\d+) pool_used=(\d+)")
budget = re.compile(r"\[prefix-cache\] on: budget (\d+)MB")
for req in ("open-a", "open-b"):
    out = subprocess.run([sys.executable, str(here / "d_tab.py"), str(d), req], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if re.match(r"\d\d-[A-Z]: ", line):
            print(line[:400])
print()
for c in sorted(p for p in d.iterdir() if p.is_dir() and re.match(r"\d\d-[A-Z]$", p.name)):
    log = (c / "server.log").read_text(errors="replace")
    rows = [json.loads(l) for l in (c / "ab.jsonl").read_text().splitlines() if l.strip()]
    env = (c / "guard.log").read_text(errors="replace") if (c / "guard.log").exists() else ""
    knobs = " ".join(sorted(set(re.findall(r"MEMRA_(?:REUSE_POOL|PREFIX_CACHE_MB)=\S+", env))))
    b = budget.search(log)
    print(f"{c.name} [{knobs}] budget {b.group(1) if b else '?'} MB")
    for r in rows:
        t0, t1 = r["send_unix"], r["send_unix"] + r["total_s"]
        cs = [m for m in comp.finditer(log) if t0 <= float(m.group(1)) <= t1]
        fs = [m for m in fail.finditer(log) if t0 <= float(m.group(1)) <= t1]
        rs = [m for m in rec.finditer(log) if float(m.group(1)) <= t1]
        sp = [m for m in spub.finditer(log) if t0 <= float(m.group(1)) <= t1]
        last = rs[-1] if rs else None
        print(f"  {r['req']}: prompt {r['prompt_tokens']} tok {r['completion_tokens']} {r['finish']} TTFT {r['ttft_s']:.3f} s "
              f"total {r['total_s']:.3f} s text {r['text_sha256'][:12]}"
              + (f"; publication {sp[0].group(2)} lent {sp[0].group(4)} MB us {sp[0].group(5)} pool_used {int(sp[0].group(7)) / 2**20:.0f} MiB" if sp else "")
              + (f"; compacted {len(cs)} planes: held {sum(float(m.group(2)) for m in cs):.0f} MB, copied "
                 f"{sum(float(m.group(3)) for m in cs):.0f} MB, {sum(int(m.group(5)) for m in cs) / 1000:.1f} ms" if cs else "")
              + (f"; {len(fs)} compactions stopped" if fs else "")
              + (f"; last reconcile ({last.group(2)}): books {last.group(3)} adopted {last.group(5)} MB charged {last.group(6)} MB "
                 f"real {last.group(7)} MB pool_used {last.group(10)} MB" if last else ""))
    breaks = log.count("[prefix-loan] INVARIANT BROKEN")
    print(f"  invariant breaks {breaks}")
