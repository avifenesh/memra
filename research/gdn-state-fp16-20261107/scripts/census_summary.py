#!/usr/bin/env python3
"""Summarize a GDN state census (memra-server MEMRA_GDN_STATE_CENSUS) against fp16 storage.

usage: census_summary.py CENSUS_DIR [CENSUS_DIR ...] > summary.md
Reads <dir>/census.jsonl (one row per sample: t_s, id, phase, rows, n_prompt, generated, layers),
<dir>/columns.json and <dir>/agg.json (per layer and head extremes over every sample).
"""
import json
import math
import sys
from pathlib import Path

F16_MAX = 65504.0


def load(d):
    d = Path(d)
    cols = json.loads((d / 'columns.json').read_text())['layer']
    ix = {c: i for i, c in enumerate(cols)}
    rows = [json.loads(l) for l in (d / 'census.jsonl').read_text().splitlines() if l.strip()]
    agg = json.loads((d / 'agg.json').read_text())
    return ix, rows, agg


def fmt(x):
    return f'{x:.3e}' if isinstance(x, float) else str(x)


def main():
    for d in sys.argv[1:]:
        ix, rows, agg = load(d)
        name = Path(d).name
        sessions = {r[1] for r in rows}
        phases = {}
        for r in rows:
            phases.setdefault(r[2], []).append(r)
        max_rows = max(r[3] for r in rows) if rows else 0
        print(f'## census `{name}`: {len(rows)} samples, {len(sessions)} sessions, '
              f'deepest sample {max_rows} rows; ' +
              ', '.join(f'{p} {len(v)}' for p, v in sorted(phases.items())))
        print()
        layers = agg['layers']
        elems = sum(l['elems'] for l in layers)
        tot = {k: sum(l[k] for l in layers) for k in ('zero', 'nonfinite', 'ovf', 'ftz', 'sub')}
        gmax = max((max(l['head_max_abs']), l['layer'], l['head_max_abs'].index(max(l['head_max_abs'])))
                   for l in layers)
        print(f'- element samples: {elems:,} ({agg["samples"]} samples x {len(layers)} layers x '
              f'{len(layers[0]["head_max_abs"])} heads x 16,384)')
        print(f'- largest |x|: {gmax[0]:.4g} (layer {gmax[1]}, head {gmax[2]}); fp16 headroom '
              f'{F16_MAX / gmax[0]:.0f}x; elements that would overflow fp16: {tot["ovf"]}; '
              f'non-finite in fp32: {tot["nonfinite"]}')
        print(f'- exact zeros: {tot["zero"] / elems:.3e} of elements; in fp16 subnormal range '
              f'(0 < |x| < 2^-14): {tot["sub"] / elems:.3e}; would flush to zero: '
              f'{tot["ftz"] / elems:.3e}')
        esub = max((max(l['head_max_esub']), l['layer']) for l in layers)
        print(f'- worst head energy share of subnormal-range elements, any sample: {esub[0]:.3e} '
              f'(L2 share {math.sqrt(esub[0]):.3e}, layer {esub[1]})')
        r16 = max((max(l['head_max_r16']), l['layer']) for l in layers)
        rb16 = max((max(l['head_max_rb16']), l['layer']) for l in layers)
        ratios = [b / a for l in layers for a, b in zip(l['head_max_r16'], l['head_max_rb16']) if a > 0]
        print(f'- worst head relative L2 error of one rounding, any sample: fp16 {r16[0]:.3e} '
              f'(layer {r16[1]}), bf16 {rb16[0]:.3e} (layer {rb16[1]}); per head bf16/fp16 ratio '
              f'min {min(ratios):.2f}, median {sorted(ratios)[len(ratios) // 2]:.2f}')
        rms = min((min(l['head_min_rms']), l['layer']) for l in layers)
        print(f'- smallest head RMS in any sample: {rms[0]:.3e} (layer {rms[1]})')
        print()
        print('| phase | samples | max abs | max sub frac (layer) | max esub share (layer) | '
              'max r16 layer | max r16 worst head | max rb16 layer | max rb16 worst head |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for p, rs in sorted(phases.items()):
            def mx(c):
                return max(L[ix[c]] for r in rs for L in r[6])
            print(f'| {p} | {len(rs)} | {mx("max_abs"):.4g} | {mx("sub_frac"):.3e} | '
                  f'{mx("esub_share"):.3e} | {mx("r16_layer"):.3e} | {mx("r16_worst_head"):.3e} | '
                  f'{mx("rb16_layer"):.3e} | {mx("rb16_worst_head"):.3e} |')
        print()
        print('Per layer (extremes over every sample):')
        print()
        print('| layer | max abs | max-abs head | sub frac | ftz frac | max head esub | '
              'max head r16 | max head rb16 | min head rms |')
        print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for l in layers:
            m = max(l['head_max_abs'])
            print(f'| {l["layer"]} | {m:.4g} | {l["head_max_abs"].index(m)} | '
                  f'{l["sub"] / l["elems"]:.2e} | {l["ftz"] / l["elems"]:.2e} | '
                  f'{max(l["head_max_esub"]):.2e} | {max(l["head_max_r16"]):.2e} | '
                  f'{max(l["head_max_rb16"]):.2e} | {min(l["head_min_rms"]):.2e} |')
        print()
        hist = [sum(l['hist'][i] for l in layers) for i in range(len(layers[0]['hist']))]
        nz = sum(hist)
        cum = 0
        print('log2 |x| distribution, all layers (bucket = floor(log2|x|); exponent <= -40 and '
              '>= 17 clamp):')
        print()
        print('| exponent | share | cumulative |')
        print('|---:|---:|---:|')
        for i, c in enumerate(hist):
            cum += c
            if c:
                print(f'| {i - 40} | {c / nz:.3e} | {cum / nz:.3e} |')
        print()


if __name__ == '__main__':
    main()
