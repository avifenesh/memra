"""Describe first versus later requests from already-audited selected receipts."""
import argparse
import csv
import json
import math
from pathlib import Path

from analyze import analyze


def diagnose(ledger, receipts, family):
    audit = analyze(ledger, receipts, family)
    groups = json.loads(ledger.read_text())
    arms = ['fixed', 'native', 'measured', 'learned']
    totals = {segment: {arm: {'tokens': 0, 'seconds': 0.0, 'requests': 0} for arm in arms}
              for segment in ['initial_requests', 'later_requests']}
    for group in groups:
        for record in group['records']:
            path = receipts / group['receipt_dir'] / f"{record['seed']}-{record['arm']}" / 'turns.tsv'
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    initial = int(row['turn']) in ([1] if family == 'qwen' else [1, 9])
                    dest = totals['initial_requests' if initial else 'later_requests'][record['arm']]
                    seconds = float(row['elapsed_s'])
                    assert math.isfinite(seconds) and seconds > 0
                    dest['tokens'] += int(row['output_tokens'])
                    dest['seconds'] += seconds
                    dest['requests'] += 1
    result = {'family': family, 'segments': {}}
    for segment, rows in totals.items():
        rates = {arm: row['tokens'] / row['seconds'] for arm, row in rows.items()}
        result['segments'][segment] = {
            'totals': rows, 'pooled_e2e_tok_s': rates,
            'learned_vs_fixed_percent': 100 * (rates['learned'] / rates['fixed'] - 1),
            'learned_vs_native_percent': 100 * (rates['learned'] / rates['native'] - 1),
        }
    for arm in arms:
        records = [r for g in groups for r in g['records'] if r['arm'] == arm]
        assert sum(totals[s][arm]['tokens'] for s in totals) == sum(r['tokens'] for r in records)
        assert math.isclose(sum(totals[s][arm]['seconds'] for s in totals),
                            sum(r['elapsed_s'] for r in records), abs_tol=1e-6)
        first = 8 if family == 'qwen' else 16
        assert totals['initial_requests'][arm]['requests'] == first
        assert totals['later_requests'][arm]['requests'] == first * 7
    result['audited_turns'] = audit['audited_turns']
    result['note'] = 'Request-position segments are descriptive; this is not a pure initialization-cost ablation.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['qwen', 'gemma'], required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.ledger, args.receipts, args.family)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: row[k] for k in ['learned_vs_fixed_percent', 'learned_vs_native_percent']}
                      for name, row in result['segments'].items()}, indent=2))
