#!/usr/bin/env python3
"""Per grow-replay arm: the lowest driver-free sample (24,463 MiB total less the largest
`memory.used` in its 250 ms telemetry), the OOM lines in its server log, the step-OOM parks, and
the pool counters `grow_ab.sh` wrote to arm.txt.

  python3 grow_floor.py GROW_DIR [--md]
"""
import json
import re
import sys
from pathlib import Path

TOTAL_MIB = 24463
d = Path(sys.argv[1])
out = []
for arm in sorted(p for p in d.iterdir() if p.is_dir() and (p / 'arm.txt').exists()):
    used = []
    tel = arm / 'gpu-250ms.csv'
    if tel.exists():
        for line in tel.read_text(errors='replace').splitlines():
            m = re.search(r'(\d+) MiB\s*$', line)
            if m:
                used.append(int(m.group(1)))
    log = (arm / 'serve.log').read_text(errors='replace') if (arm / 'serve.log').exists() else ''
    counters = dict(re.findall(r'^([a-z-]+)=(\d+)$', (arm / 'arm.txt').read_text(), re.M))
    out.append({
        'arm': arm.name,
        'free_floor_mib': TOTAL_MIB - max(used) if used else None,
        'log_free_floor_mib': min([int(x) / 2**20 for x in re.findall(r'driver_free_bytes[= ](\d+)', log)]
                                  + [float(x) for x in re.findall(r'dev0 free (\d+)MB', log)], default=None),
        'oom_lines': len(re.findall(r'out of memory', log, re.I)),
        'step_oom_parks': log.count('step OOM parked session'),
        'resumes': int(counters.get('dspark-rewound-resumes', 0)),
        'grows': int(counters.get('dspark-grows', 0)),
        'misses': int(counters.get('dspark-miss', 0)),
        'seat_declines': int(counters.get('seat-declines', 0)),
    })
if '--md' in sys.argv:
    print('| arm | lowest driver free, 250 ms telemetry MiB | lowest driver free the server logged MiB | OOM lines | step-OOM parks | rewound resumes | grows | pool misses | seat declines |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in out:
        lf = '-' if r['log_free_floor_mib'] is None else f"{r['log_free_floor_mib']:.0f}"
        print(f"| {r['arm']} | {r['free_floor_mib']} | {lf} | {r['oom_lines']} | {r['step_oom_parks']} | {r['resumes']} | "
              f"{r['grows']} | {r['misses']} | {r['seat_declines']} |")
else:
    print(json.dumps(out, indent=1))
