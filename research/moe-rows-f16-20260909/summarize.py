#!/usr/bin/env python3
"""Derive the numeric no-go from raw oracle rows; no timing inference."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'


def main():
    summary = []
    per_layer = []
    for t, phase in [(2, 'gate-replay'), (4, 'gate')]:
        base = RAW / f't{t}'
        with (base / 'oracle.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        assert all(int(r['t']) == t for r in rows)
        for layer in sorted({int(r['layer']) for r in rows}):
            selected = [r for r in rows if int(r['layer']) == layer]
            chain = next(r for r in selected if r['tensor_or_arm'] == 'full_chain')
            argmax = [r for r in selected if r['tensor_or_arm'].startswith('argmax_row_')]
            per_layer.append(dict(t=t, layer=layer, current_mean=chain['baseline_mean'], f16_mean=chain['mean_or_us'], current_max=chain['baseline_max'], f16_max=chain['max'], band=chain['status'], argmax_passed=sum(r['status'] == 'PASS' for r in argmax), argmax_checked=len(argmax), argmax='PASS' if all(r['status'] == 'PASS' for r in argmax) else 'FAIL'))
        failed = [r for r in rows if r['status'] != 'PASS']
        summary.append(dict(t=t, layers_checked=len({r['layer'] for r in rows}), rows=len(rows), failures=failed, exit=int((base / f'{phase}.exit').read_text()), binary_sha256=(base / f'{phase}.binary.sha256').read_text().split()[0]))
    assert summary[0]['rows'] == 168 and summary[0]['exit'] == 0 and not summary[0]['failures']
    assert (RAW / 't2/gate.pass').read_text() == '42 layers passed\n'
    failure = summary[1]['failures']
    assert len(failure) == 1 and failure[0]['layer'] == '20' and failure[0]['tensor_or_arm'] == 'argmax_row_3'
    assert failure[0]['baseline_mean'] == '4' and failure[0]['mean_or_us'] == '1819'
    assert summary[1]['exit'] == 1
    assert summary[0]['binary_sha256'] == summary[1]['binary_sha256']
    assert not list(RAW.glob('t*/timings.tsv'))
    with (ROOT / 'per-layer-oracle.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_layer[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(per_layer)
    result = dict(verdict='NEGATIVE', reason='numeric no-go', source='bc89f4d22', binary_sha256=summary[0]['binary_sha256'], shapes=summary, weighted_saving_ms={str(t): None for t in [2, 4, 7]}, timing='REFUSED after oracle failure', real_t7='NOT RUN', threshold_ms=.5)
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
