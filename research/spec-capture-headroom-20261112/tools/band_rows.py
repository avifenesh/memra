#!/usr/bin/env python3
"""band_rows.py CAMPAIGN_DIR: memra#1221 review S3, one row per band cell. The capture point is the
guard's `[spec] WARN` line when it refused, else the end of the cold request's last trunk prime
chunk (the guard runs right after it). MemAvailable at the capture is the holder's last reading
before it; the minimum and the holder's give-backs (a 256 MB chunk released because the server's
use pushed MemAvailable under the target) are counted from the capture to the request's end."""
import json
import pathlib
import re
import sys

d = pathlib.Path(sys.argv[1])
hl = re.compile(r"^(\d+\.\d+) holder MemAvailable (\d+) MB held (\d+) MB")
for c in sorted(p for p in d.iterdir() if p.is_dir()):
    srv = (c / "server.log").read_text(errors="replace").splitlines()
    warn = [l for l in srv if "[spec] WARN" in l and "capture" in l]
    trunk = [l for l in srv if "[prime-chunk] phase=mtp-trunk" in l]
    trims = [l for l in srv if "pre-capture pool trim" in l]
    oom = [l for l in srv if re.search(r"out of memory|OUT_OF_MEMORY", l)]
    t_cap = float((warn[-1] if warn else trunk[-1]).split()[0])
    rows = [json.loads(l) for l in (c / "ab.jsonl").read_text().splitlines() if l.startswith("{")]
    r = rows[-1]
    t_end = r["send_unix"] + r["total_s"]
    h = [(float(m[1]), int(m[2]), int(m[3])) for m in map(hl.match, (c / "holder.log").read_text().splitlines()) if m]
    before = [x for x in h if x[0] <= t_cap]
    win = [x for x in h if t_cap <= x[0] <= t_end]
    gb = sum(1 for a, b in zip(win, win[1:]) if b[2] < a[2])
    out = {
        "cell": c.name,
        "avail_at_capture_mb": before[-1][1] if before else None,
        "min_avail_to_end_mb": min(x[1] for x in win) if win else None,
        "holder_givebacks_after_capture": gb,
        "held_mb_at_capture": before[-1][2] if before else None,
        "guard": warn[-1].split("(", 1)[1][:160] if warn else "no refusal",
        "trims": len(trims),
        "oom_lines": len(oom),
        "kernel_oom": (c / "kernel-oom-count.txt").read_text().strip(),
        "error": r["error"],
        "text": r["text_sha256"][:12],
        "ttft_s": r["ttft_s"],
    }
    print(json.dumps(out))
