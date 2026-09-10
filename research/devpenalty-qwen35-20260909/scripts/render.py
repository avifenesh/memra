#!/usr/bin/env python3
"""Reduce the devpenalty qualification cell: OFF vs ON, medians over 3 boots x N reps."""
import json, statistics as st, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/dp/out')
ARMS = ('off', 'on')
BOOTS = (1, 2, 3)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def load(arm, name, conc):
    rows, blocks = [], []
    for b in BOOTS:
        p = OUT / f'{arm}-b{b}-{name}-c{conc}.json'
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        for r in d['rows']:
            blocks.append(r)
            rows.extend(r['tasks'])
    return rows, blocks


def agg_overlap(blocks):
    """Aggregate decode inside the window every worker of a rep is still decoding.

    Measured, not assumed: per-stream tok/s times concurrency would overstate a staggered
    block. Workers of one rep start together, so their decode windows share an origin.
    """
    per_rep = []
    for r in blocks:
        t1 = [t['turn1'] for t in r['tasks']]
        start, end = max(x['ttft_s'] for x in t1), min(x['total_s'] for x in t1)
        if end > start:
            per_rep.append(sum(x['decode_tok_s'] for x in t1 if x['decode_tok_s']))
    return st.median(per_rep) if per_rep else None


def cell(arm, name, conc):
    rows, blocks = load(arm, name, conc)
    if not rows:
        return None
    t1 = [r['turn1'] for r in rows]
    t2 = [r['turn2'] for r in rows]
    spec1 = [x.get('spec') for x in t1 if x.get('spec')]
    kpos = [s for s in spec1 if (s.get('rounds') or 0) > 0]
    return dict(
        arm=arm, cell=name, c=conc, n=len(rows), boots=len({b['rep'] for b in blocks}),
        p1=med([x['prompt_tokens'] for x in t1]),
        ttft1=med([x['ttft_s'] for x in t1]), ttft1_p95=pct([x['ttft_s'] for x in t1], 0.95),
        prefill=med([x['prefill_tok_s'] for x in t1]),
        dec1=med([x['decode_tok_s'] for x in t1]),
        dec2=med([x['decode_tok_s'] for x in t2]),
        agg=agg_overlap(blocks),
        comp1=med([x['completion_tokens'] for x in t1]),
        cached2=med([x['cached_tokens'] for x in t2]),
        e2e=med([r['e2e_s'] for r in rows]),
        wall=med([b['block_wall_s'] for b in blocks]),
        vram_peak=max(b['vram_peak_mib'] for b in blocks),
        spec_n=len(spec1), spec_kpos=len(kpos), req_n=len(t1),
        spec_rounds=med([s['rounds'] for s in spec1]) if spec1 else None,
        spec_acc=med([s['acceptance_rate'] for s in spec1]) if spec1 else None)


def pptg(arm):
    rows = []
    for b in BOOTS:
        p = OUT / f'{arm}-b{b}-pptg.json'
        if p.is_file():
            rows += json.loads(p.read_text())['rows']
    if not rows:
        return None
    return dict(arm=arm, n=len(rows), prompt=med([r['prompt_tokens'] for r in rows]),
                pp=med([r['prefill_tok_s'] for r in rows]),
                tg=med([r['decode_tok_s'] for r in rows]),
                comp=med([r['completion_tokens'] for r in rows]))


def greedy():
    seen = {}
    for arm in ARMS:
        for b in BOOTS:
            p = OUT / f'{arm}-b{b}-greedy.json'
            if not p.is_file():
                continue
            d = json.loads(p.read_text())
            for name, c in d['cases'].items():
                seen.setdefault(name, []).append((f'{arm}-b{b}', c['text_sha256'],
                                                  c['completion_tokens']))
    return seen


R, P = {}, {}
for arm in ARMS:
    for name in ('vendor', 'pp0'):
        for c in (1, 4, 8):
            x = cell(arm, name, c)
            if x:
                R[(arm, name, c)] = x
    x = pptg(arm)
    if x:
        P[arm] = x

f = lambda v, n=1: '-' if v is None else f'{v:,.{n}f}'


def delta(a, b):
    if a in (None, 0) or b is None:
        return '-'
    return f'{(b / a - 1) * 100:+.1f}%'


lines = []
lines.append('## Page-task, vendor non-thinking shape (client sends no sampling field)\n')
lines.append('| c | per-stream decode tok/s OFF | ON | delta | aggregate decode tok/s OFF | ON | '
             'delta | TTFT t1 p50 s OFF | ON | e2e s OFF | ON |')
lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for c in (1, 4, 8):
    a, b = R.get(('off', 'vendor', c)), R.get(('on', 'vendor', c))
    if not a or not b:
        continue
    lines.append(f"| {c} | {f(a['dec1'])} | {f(b['dec1'])} | {delta(a['dec1'], b['dec1'])} | "
                 f"{f(a['agg'])} | {f(b['agg'])} | {delta(a['agg'], b['agg'])} | "
                 f"{f(a['ttft1'],3)} | {f(b['ttft1'],3)} | {f(a['e2e'],2)} | {f(b['e2e'],2)} |")

lines.append('\n## presence_penalty 0.0 control (penalty-free rows never enter the door)\n')
lines.append('| c | per-stream decode tok/s OFF | ON | delta | aggregate decode tok/s OFF | ON | delta |')
lines.append('|---:|---:|---:|---:|---:|---:|---:|')
for c in (1, 4, 8):
    a, b = R.get(('off', 'pp0', c)), R.get(('on', 'pp0', c))
    if not a or not b:
        continue
    lines.append(f"| {c} | {f(a['dec1'])} | {f(b['dec1'])} | {delta(a['dec1'], b['dec1'])} | "
                 f"{f(a['agg'])} | {f(b['agg'])} | {delta(a['agg'], b['agg'])} |")

lines.append('\n## How much of the collapse the door removes\n')
lines.append('| c | vendor OFF agg | vendor ON agg | pp0 ceiling agg | ON as % of ceiling | remaining gap tok/s |')
lines.append('|---:|---:|---:|---:|---:|---:|')
for c in (1, 4, 8):
    a, b, ceil = R.get(('off', 'vendor', c)), R.get(('on', 'vendor', c)), R.get(('off', 'pp0', c))
    if not (a and b and ceil):
        continue
    share = f"{b['agg'] / ceil['agg'] * 100:.1f}%" if ceil['agg'] else '-'
    lines.append(f"| {c} | {f(a['agg'])} | {f(b['agg'])} | {f(ceil['agg'])} | {share} | "
                 f"{f(ceil['agg'] - b['agg'])} |")

lines.append('\n## pp512/tg128 twin, vendor shape, c=1\n')
lines.append('| arm | prompt tok | pp tok/s | tg tok/s | completion tok | n |')
lines.append('|---|---:|---:|---:|---:|---:|')
for arm in ARMS:
    x = P.get(arm)
    if x:
        lines.append(f"| {arm} | {x['prompt']:,.0f} | {f(x['pp'],0)} | {f(x['tg'])} | "
                     f"{x['comp']:,.0f} | {x['n']} |")
if P.get('off') and P.get('on'):
    lines.append(f"\npp delta {delta(P['off']['pp'], P['on']['pp'])}, "
                 f"tg delta {delta(P['off']['tg'], P['on']['tg'])}")

lines.append('\n## Spec engagement (turn 1)\n')
lines.append('| arm | cell | c | requests | usage.spec present | K>0 | median rounds | acceptance |')
lines.append('|---|---|---:|---:|---:|---:|---:|---:|')
for arm in ARMS:
    for name in ('vendor', 'pp0'):
        for c in (1, 4, 8):
            x = R.get((arm, name, c))
            if x:
                lines.append(f"| {arm} | {name} | {c} | {x['req_n']} | {x['spec_n']} | "
                             f"{x['spec_kpos']} | {f(x['spec_rounds'],0)} | {f(x['spec_acc'],3)} |")

lines.append('\n## Greedy exactness instrument\n')
g = greedy()
verdict_lines = []
for name, obs in g.items():
    shas = {s for _, s, _ in obs}
    ok = len(shas) == 1
    verdict_lines.append(f'- `{name}`: {len(obs)} boots, '
                         f'{"BYTE-IDENTICAL" if ok else "DIVERGED"} '
                         f'({", ".join(sorted(shas))[:80]}{"..." if len(shas) > 1 else ""}), '
                         f'{obs[0][2]} tokens')
lines += verdict_lines

text = '\n'.join(lines)
print(text)
(OUT.parent / 'TABLE.md').write_text(text + '\n')
(OUT.parent / 'AGG.json').write_text(json.dumps(
    {'cells': {f'{k[0]}|{k[1]}|c{k[2]}': v for k, v in R.items()},
     'pptg': P, 'greedy': g}, indent=1) + '\n')
