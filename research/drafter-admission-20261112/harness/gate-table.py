#!/usr/bin/env python3
"""memra#1173: one table per gate run (summary.json): per step, what the gate decided and the
memory the admission left. usage: gate-table.py RUN_DIR..."""
import json
import sys
from pathlib import Path

MB = 1e6


def n(v):
    return "-" if v is None else f"{v / MB:,.0f}"


for d in sys.argv[1:]:
    s = json.loads((Path(d) / "summary.json").read_text())
    g = s["gpu"]
    full = s["full_admission"]
    print(f"\n### {Path(d).name}\n\n{s['route']} route, binary `{Path(s['binary']).name}`, {g['name']}, enforced power limit "
          f"{g['enforced_power_limit_w']} W, prompt {full['prompt_tokens']:,} tokens, draft state "
          f"{s['draft_plane_bytes'] / MB:,.0f} MB, reserve {s['reserve_estimate_bytes'] / MB:,.0f} MB\n")
    print("| step | kind | request | ballast MB | effective free before MB | receipt | available / with / without MB | speculated | status, tokens | request's peak pool use MB | min effective free MB | min driver free MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [full] + s["steps"] + ([s["replay"]] if s.get("replay") else []):
        rc = r.get("receipt")
        receipt = ("eager-arm admit" if r["eager_lines"] else f"drafter refused ({rc['arm']})" if rc
                   else "full" if r["status"] == 200 and r["speculated"] else "plain" if r["status"] == 200
                   else "capacity reject")
        nums = f"{rc['available_mb']:,} / {rc['with_mb']:,} / {rc['without_mb']:,}" if rc else "-"
        if r["kind"] == "replay":
            # The replay boot samples no memory; its row is status and route only.
            costs = ", then ".join(f"{p} {mb:,} MB" for p, mb in r["cost_lines"])
            print(f"| replay | replay | {'sampled' if r['sampled'] else 'greedy'} | - | - | priced {costs} | - | "
                  f"{'yes' if r['speculated'] else 'no'} | {r['status']}, {r['completion_tokens']} | - | - | - |")
            continue
        print(f"| {r['step']} | {r['kind']} | {'sampled' if r['sampled'] else 'greedy'} | {n(r['ballast_bytes'])} | "
              f"{n(r['effective_free_before'])} | {receipt} | {nums} | {'yes' if r['speculated'] else 'no'} | "
              f"{r['status']}, {r['completion_tokens']} | {n(r['peak_pool_used'] - r['pool_used_before'])} | "
              f"{n(r['min_effective_free'])} | {n(r['min_driver_free'])} |")
    print()
    if s.get("replay"):
        print(f"Replay boot: cost lines {s['replay']['cost_lines']}, speculated {s['replay']['speculated']}.\n")
    for k, v in s["verdicts"].items():
        print(f"- {k}: {v['detail']}" if k.startswith("INFO") else f"- {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['detail']})")
