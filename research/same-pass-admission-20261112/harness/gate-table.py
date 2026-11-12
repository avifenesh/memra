#!/usr/bin/env python3
"""memra#1255: one block per tools/same-pass-admission-gate.py run (summary.json + gate.log).
usage: gate-table.py GATES_DIR"""
import json
import sys
from pathlib import Path

MB = 1_000_000
for d in sorted(Path(sys.argv[1]).iterdir()):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    gl = (d / "gate.log").read_text().splitlines()
    rc = next((l for l in reversed(gl) if l.startswith("gate ")), "")
    print(f"### `{d.name}`\n")
    print(f"- binary `{s['bin'].rsplit('/', 1)[-1]}` (sha in `gate.log`), card {s['gpu']['name']}, "
          f"enforced power limit {s['gpu']['enforced_power_limit_w']} W at the start; wrapper `{rc}`")
    if "learn" in s:
        a, b = s["learn"]["A"], s["learn"]["B"]
        print(f"- learn alone: A {a['prompt_tokens']} prompt tokens, charge {a['charge'] / MB:.0f} MB; "
              f"B {b['prompt_tokens']} prompt tokens, charge {b['charge'] / MB:.0f} MB; "
              f"transient floor {s['floor'] / MB:.0f} MB")
    if "target_free" in s:
        busy = s.get("busy_free", s.get("idle_free", 0))
        print(f"- placement: effective free beside the keeper {busy / MB:.0f} MB, ballast "
              f"{s.get('ballast', 0) / MB:.0f} MB, before the pair {s.get('free_before_pair', 0) / MB:.0f} MB; "
              f"one fits at {s['one_fits_at'] / MB:.0f} MB, two at {s['two_fit_at'] / MB:.0f} MB, "
              f"target {s['target_free'] / MB:.0f} MB")
    if "shape" in s:
        sh = s["shape"]
        print(f"- the pair's admissions (`[spec-k]` lines after the fire, line index): {sh.get('pair_admit_lines', sh.get('worker_admit_lines'))}; "
              f"their routes: {sh.get('pair_routes', 'n/a')}; next tick line: {sh.get('next_tick_line', sh.get('first_prime_line'))}; "
              f"settle-wait lines: {sh.get('settle_waits')} "
              f"(first at {sh.get('settle_lines', [None])[:1]}); `out of memory` lines: {sh.get('oom_lines')}; "
              f"panic/respawn lines: {sh.get('bad_lines')}; server alive: {s.get('alive')}")
        for n, r in sorted(s.get("pair", {}).items()):
            err = r.get("error")
            msg = ""
            if err:
                e = err.get("error", err) if isinstance(err, dict) else err
                msg = f" ({e.get('code') or e.get('type') or ''}: {str(e.get('message', ''))[:90]})" if isinstance(e, dict) else ""
            print(f"- pair {n}: HTTP {r['status']}, completion tokens {r.get('completion_tokens')}{msg}")
        print(f"- same greedy text as the learn pass: {s.get('same_text')}; keeper: {s.get('keeper')}")
    if "verdicts" in s:
        print("- verdicts: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in s["verdicts"].items()))
    print()
