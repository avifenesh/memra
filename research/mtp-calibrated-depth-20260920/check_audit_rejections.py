"""Exercise audit failures against immutable real calibration receipts."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

from audit_selection import audit


def rejection_checks(root, family):
    baseline = audit(root, family, partial=True)
    selection_name = f'{family}-selection.json'
    original = json.loads((root / selection_name).read_text())
    cases = []
    for name in ('winner', 'calibration_hash', 'fixed_trace'):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for source in root.iterdir():
                (temp / source.name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
            selected = copy.deepcopy(original)
            path = temp / selection_name
            path.unlink()
            if name == 'winner':
                selected['selected_k'] = 1 if selected['selected_k'] != 1 else 2
            elif name == 'calibration_hash':
                key = next(iter(selected['calibration_runs_sha256']))
                selected['calibration_runs_sha256'][key] = '0' * 64
            else:
                group = selected['calibration'][0]
                relative = Path(group['receipt_dir']) / f'{group["seed"]}-k1' / 'rounds.tsv'
                # Replace only temporary links; source receipt bytes stay untouched.
                for parent in (relative.parent.parent, relative.parent):
                    link = temp / parent
                    source = link.resolve()
                    link.unlink()
                    link.mkdir()
                    for child in source.iterdir():
                        (link / child.name).symlink_to(child, target_is_directory=child.is_dir())
                target = temp / relative
                lines = target.read_text().splitlines()
                columns = lines[0].split('\t')
                depth = columns.index('draft_depth')
                eligible = columns.index('eligible_for_learning')
                for index in range(1, len(lines)):
                    row = lines[index].split('\t')
                    if row[eligible] == 'true':
                        row[depth] = '2'
                        lines[index] = '\t'.join(row)
                        break
                else:
                    raise AssertionError('No eligible fixed-depth round in fixture')
                target.unlink()
                target.write_text('\n'.join(lines) + '\n')
            path.write_text(json.dumps(selected))
            try:
                audit(temp, family, partial=True)
            except AssertionError:
                cases.append(name)
            else:
                raise AssertionError(f'Audit accepted corrupted {name}')
    assert len(cases) == 3
    return {'status': 'PASS', 'family': family, 'rejected_corruptions': cases,
            'original_selected_k': baseline['selected_k']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--family', choices=['qwen', 'gemma'], required=True)
    args = parser.parse_args()
    print(json.dumps(rejection_checks(args.root, args.family), indent=2))
