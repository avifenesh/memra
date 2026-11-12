#!/usr/bin/env python3
"""memra#1233: where a boot's time went, per boot of the frozen mixed hit/cold cell.

Reads the cell's summary rows, the server's `[tick]` trace (MEMRA_TICK_TRACE=1) and, when the boot
logged it, the 1 Hz telemetry (SM clock, power). One row per boot: the hit-role requests that
restored nothing (each is a cold prime of 4,860 tokens), the prime ticks of at least 512 tokens
(cold chunks) and their summed time, the short prime ticks (hit suffixes), the summed decode time,
cold TTFT p50, output tok/s, and the mean SM clock and power over the boot.

usage: boot-accounting.py RUN_DIR [RUN_DIR...] [--json]
"""
import json
import statistics
import sys
from pathlib import Path


def opened(path):
    """Open a banked text file, gzipped or not (`path` or `path.gz`)."""
    import gzip
    from pathlib import Path
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", errors="replace")
    return open(path, errors="replace")



def tick_sums(log: Path) -> dict:
    out = {"ticks": 0, "cold_ticks": 0, "cold_s": 0.0, "cold_tokens": 0, "short_ticks": 0, "short_s": 0.0, "decode_s": 0.0}
    for line in opened(log):
        if not line.startswith("[tick]"):
            continue
        d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        toks = int(d["prefill_single_tokens"]) + int(d["prefill_batch_tokens"])
        pf = float(d["prefill_ms"]) / 1000
        out["ticks"] += 1
        out["decode_s"] += float(d["decode_ms"]) / 1000
        if toks >= 512:
            out["cold_ticks"] += 1
            out["cold_s"] += pf
            out["cold_tokens"] += toks
        elif toks > 0:
            out["short_ticks"] += 1
            out["short_s"] += pf
    return out


def telemetry(csv: Path) -> dict:
    if not csv.exists() and not Path(str(csv) + ".gz").exists():
        return {}
    clk, pw = [], []
    for line in opened(csv):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            clk.append(float(parts[1].split()[0]))
            pw.append(float(parts[2].split()[0]))
        except ValueError:
            continue
    if not clk:
        return {}
    return {"sm_clock_mhz_mean": round(statistics.fmean(clk), 1), "power_w_mean": round(statistics.fmean(pw), 1), "telemetry_samples": len(clk)}


rows = []
for d in [Path(a) for a in sys.argv[1:] if not a.startswith("--")]:
    for line in (d / "summary.log").read_text().splitlines():
        if not line.startswith("{"):
            continue
        s = json.loads(line)
        label = s["label"]
        hc = s["hit_cached_tokens"]
        t = tick_sums(d / f"server-{label}.log")
        rows.append({
            "window": d.name, "label": label, "arm": label.rsplit("-", 1)[0],
            "missed": hc.get("0", 0), "cold_p50_ms": s["cold_ttft_p50_ms"], "tok_s": s["output_tok_s_median"],
            **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in t.items()},
            "cold_ms_per_1k_tokens": round(1e6 * t["cold_s"] / t["cold_tokens"], 1) if t["cold_tokens"] else None,
            **telemetry(d / f"telemetry-{label}.csv"),
        })
if "--json" in sys.argv:
    for r in rows:
        print(json.dumps(r, sort_keys=True))
    sys.exit(0)
print("| window | boot | hit-role requests not restored | cold-chunk ticks / tokens / s | ms per 1,000 cold tokens | short prime ticks / s | decode s | cold TTFT p50 ms | output tok/s | SM clock MHz / power W, mean |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    tel = f"{r['sm_clock_mhz_mean']:.0f} / {r['power_w_mean']:.0f}" if "sm_clock_mhz_mean" in r else "-"
    print(f"| {r['window']} | {r['label']} | {r['missed']} | {r['cold_ticks']} / {r['cold_tokens']:,} / {r['cold_s']:.1f} | {r['cold_ms_per_1k_tokens']:.1f} | "
          f"{r['short_ticks']} / {r['short_s']:.1f} | {r['decode_s']:.1f} | {r['cold_p50_ms']:,.0f} | {r['tok_s']:.1f} | {tel} |")
