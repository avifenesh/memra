"""Markdown tables for RESULTS.md from score.py's output.

  python3 tables.py PHASE_DIR [PHASE_DIR ...]
Agentic phases: one row per run (measured pass, then prime pass), then the per-arm median (N stated).
Mixed phases: one row per run, then the median. Waits count an error at the time it arrived.
"""
import json
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent


def s(x, div=1000, nd=1):
    return '-' if x is None else f'{x / div:.{nd}f}'


def score(d):
    out = subprocess.run([sys.executable, str(H / 'score.py'), d], capture_output=True, text=True, check=True).stdout
    runs, meds = [], []
    for line in out.splitlines():
        if line.startswith('MEDIAN'):
            meds.append(json.loads(line.split(' ', 3)[-1]))
        elif line.strip():
            runs.append(json.loads(line))
    return runs, meds


for d in sys.argv[1:]:
    runs, meds = score(d)
    print(f'### {Path(d).name}\n')
    if runs and 'measure' in runs[0]:
        print('| run | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB | defers | deadline-order ticks | reserve evictions | OOM |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in runs:
            for part in ('measure', 'prime'):
                m = r[part]
                extra = (f"{r['defer']} | {r['hol']} | {r['headroom']} | {r['oom']}" if part == 'measure' else ' | | | ')
                print(f"| {r['run']} | {part} | {m['ok']}/{m['n']} | {m['e408']} | {s(m['ttft_p50'])} | {s(m['ttft_p99'])} | "
                      f"{s(m['wait_p50_all'])} | {s(m['wait_p99_all'])} | {m['tokens'] if part == 'measure' else '-'} | "
                      f"{m['tok_s'] if part == 'measure' else '-'} | {m['free_min']} | {extra} |")
        print()
        print('| arm | N | pass | served | 408 | first token p50 s | p99 s | wait p50 s (all) | wait p99 s (all) | tokens | tok/s | free floor MiB |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for m in meds:
            for part in ('measure', 'prime'):
                x = m[part]
                print(f"| {m['arm']} | {m['N']} | {part} | {x['ok']:g}/{x['n']:g} | {x['e408']:g} | {s(x.get('ttft_p50'))} | {s(x.get('ttft_p99'))} | "
                      f"{s(x['wait_p50_all'])} | {s(x['wait_p99_all'])} | {x['tokens']:g} | {x['tok_s']:.2f} | {x['free_min']:g} |")
    else:
        print('| run | batch tok/s | batch first token p50 ms | repeat hits | repeat cached | repeat first token median ms | free floor MiB | OOM |')
        print('|---|---|---|---|---|---|---|---|')
        for r in runs:
            m = r['mixed']
            print(f"| {r['run']} | {m['tok_s']} | {m['ttft_p50']} | {m['repeat_hits']} | {m['repeat_cached']} | {m['repeat_ttft_med']} | {m['free_min']} | {r['oom']} |")
        print()
        print('| arm | N | batch tok/s | batch first token p50 ms | repeat hits | repeat first token median ms | free floor MiB |')
        print('|---|---|---|---|---|---|---|')
        for m in meds:
            x = m['mixed']
            print(f"| {m['arm']} | {m['N']} | {x['tok_s']:.1f} | {x['ttft_p50']:g} | {x['repeat_hits']:g} | {x['repeat_ttft_med']:g} | {x['free_min']:g} |")
    print()
