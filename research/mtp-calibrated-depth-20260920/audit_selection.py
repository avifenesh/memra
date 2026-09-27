"""Independently verify the calibration inputs and frozen choice used in evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze import table
from controller import select_depth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, family, partial=False):
    selection = json.loads((root / f'{family}-selection.json').read_text())
    maximum, turns, historical = (7, 8, 3) if family == 'qwen' else (5, 16, 5)
    assert selection['family'] == family
    assert selection['workloads_lock_sha256'] == sha(root / 'workloads.lock.json')
    workloads = json.loads((root / 'workloads.lock.json').read_text())
    assert [group['cycle'] for group in selection['calibration']] == [0, 1]
    identity = None
    groups = []
    for group in selection['calibration']:
        folder = root / group['receipt_dir']
        assert sha(folder / 'runs.json') == selection['calibration_runs_sha256'][folder.name]
        records = json.loads((folder / 'runs.json').read_text())
        assert records == group['records']
        ident = json.loads((folder / 'identity.json').read_text())
        assert ident['workload_sha256'] == workloads[f'calibration-{ "a" if group["cycle"] == 0 else "b" }']['sha256']
        assert sha(folder / 'runner.py') == ident['runner_sha256']
        stable = {key: ident[key] for key in ['gpu', 'binary_sha256', 'source_commit', 'artifacts', 'runner_sha256', 'settings', 'max_new', 'ctx']}
        if identity is None:
            identity = stable
        assert stable == identity
        assert ident['vocabulary_head'] == 'full-in-every-arm'
        for record in records:
            depth = int(record['arm'][1:])
            assert ident['fixed_depths'][record['arm']] == depth
            run = folder / f'{record["seed"]}-{record["arm"]}'
            rows = table(run / 'turns.tsv')
            assert len(rows) == turns == record['turns']
            assert math.isfinite(record['elapsed_s']) and record['elapsed_s'] >= 60
            assert abs(sum(float(row['elapsed_s']) for row in rows) - record['elapsed_s']) < 1e-6
            assert sum(int(row['output_tokens']) for row in rows) == record['tokens']
            for row in rows:
                n = row['turn']
                assert row['resumed'] == 'false'
                assert len((run / f'turn-{n}.output.ids').read_text().splitlines()) == int(row['output_tokens'])
                assert len((run / f'turn-{n}.prompt.ids').read_text().splitlines()) == int(row['prompt_tokens'])
            eligible = [row for row in table(run / 'rounds.tsv') if row['eligible_for_learning'] == 'true']
            assert eligible and all(int(row['draft_depth']) == depth for row in eligible)
        groups.append(records)
    winner, rates = select_depth(groups, maximum)
    assert winner == selection['selected_k']
    assert {int(k): v for k, v in selection['pooled_calibration_tok_s'].items()} == rates
    ledger = root / f'{family}-selected-sets.json'
    evaluations = json.loads(ledger.read_text()) if ledger.exists() else []
    assert [g['cycle'] for g in evaluations] == list(range(len(evaluations)))
    sampled_variation = {}
    for group in evaluations:
        folder = root / group['receipt_dir']
        ident = json.loads((folder / 'identity.json').read_text())
        assert {key: ident[key] for key in identity} == identity
        assert ident['fixed_depths'] == {'calibrated': winner}
        assert ident['workload_sha256'] == workloads[f'heldout-{ "a" if group["cycle"] % 2 == 0 else "b" }']['sha256']
        sampled_variation.setdefault(ident['workload_sha256'], set()).add(
            sha(folder / f'{group["seed"]}-native' / 'turn-1.output.ids'))
        if winner == historical:
            for n in range(1, turns + 1):
                for kind in ['prompt', 'output']:
                    filename = f'turn-{n}.{kind}.ids'
                    assert (folder / f'{group["seed"]}-fixed' / filename).read_bytes() == (folder / f'{group["seed"]}-calibrated' / filename).read_bytes()
    if not partial:
        assert len(evaluations) == 10
        assert all(len(values) > 1 for values in sampled_variation.values()), "No observed sampling variation across held-out seeds"
    return {'family': family, 'selected_k': winner, 'calibration_runs': sum(map(len, groups)),
            'audited_evaluation_sets': len(evaluations),
            'native_first_reply_variants_by_workload': {key: len(values) for key, values in sampled_variation.items()}, 'verdict': 'PASS' if len(evaluations) == 10 else 'INCOMPLETE'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--family', choices=['qwen', 'gemma'], required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.family, args.partial), indent=2))
