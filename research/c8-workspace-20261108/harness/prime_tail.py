#!/usr/bin/env python3
"""The agentic prime pass's first-token tail per run: failures, p50, the slowest first token and
its prompt, and the turns past 85 s and past 90 s (the non-streaming first-token ceiling).

  python3 prime_tail.py PHASE_DIR [--md]
"""
import json
import re
import sys
from pathlib import Path

phase = Path(sys.argv[1])
out = []
for run in sorted(p for p in phase.iterdir() if p.is_dir() and re.fullmatch(r'\d\d-[A-Z]', p.name)):
    if not (run / 'rows.json').exists():
        continue
    rows = [r for r in json.loads((run / 'rows.json').read_text())['rows'] if r['phase'] == 'prime']
    ok = [r for r in rows if r.get('status') == 200 and r.get('ttft_s') is not None]
    ttft = sorted(r['ttft_s'] for r in ok)
    slow = max(ok, key=lambda r: r['ttft_s']) if ok else None
    log = (run / 'serve.log').read_text(errors='replace') if (run / 'serve.log').exists() else ''
    out.append({
        'run': run.name,
        'requests': len(rows),
        'failed': sum(1 for r in rows if r.get('status') != 200),
        'p50_s': ttft[len(ttft) // 2] if ttft else None,
        'max_s': slow['ttft_s'] if slow else None,
        'max_prompt': slow.get('prompt_tokens') if slow else None,
        'over_85': sum(1 for x in ttft if x > 85.0),
        'over_90': sum(1 for x in ttft if x > 90.0),
        'edf_lines': log.count('earliest deadline first'),
        'oom': len(re.findall(r'out of memory|CUDA_ERROR_OUT_OF_MEMORY', log, re.I)),
    })
if '--md' in sys.argv:
    print('| run | failed | first token p50 s | slowest s (prompt tokens) | over 85 s | over 90 s | deadline-order switches | OOM lines |')
    print('|---|---|---|---|---|---|---|---|')
    for r in out:
        print(f"| {r['run']} | {r['failed']} of {r['requests']} | {r['p50_s']:.1f} | {r['max_s']:.1f} ({r['max_prompt']:,}) | "
              f"{r['over_85']} | {r['over_90']} | {r['edf_lines']} | {r['oom']} |")
else:
    print(json.dumps(out, indent=1))
