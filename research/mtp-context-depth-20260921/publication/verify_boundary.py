"""Verify the complete expanded-archive review without rewriting sealed evidence."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ARCHIVES = {'qwen-records.tar.gz', 'gemma-records.tar.gz',
            'common-records.tar.gz', 'runtime-source.tar.gz'}


def verify(scan, review):
    for key in ('runtime_archive_sha256', 'scanner_sha256', 'policy_sha256'):
        if scan[key] != review[key]:
            raise ValueError('The reviewed runtime or policy identity changed')
    inventory = {r['file']: r for r in review['archive_inventory']}
    actual = {r['file']: r for r in scan['archives']}
    if (len(inventory) != 4 or len(actual) != 4 or set(inventory) != ARCHIVES
            or set(actual) != ARCHIVES or len(scan['archives']) != 4
            or len(review['archive_inventory']) != 4):
        raise ValueError('The complete four-archive review is required')
    source = {(r['file'], r['sha256']): r for r in review['source_rule_pins']}
    records = {(r['archive'], r['file'], r['sha256']): r for r in review['record_rule_pins']}
    if len(source) != len(review['source_rule_pins']) or len(records) != len(review['record_rule_pins']):
        raise ValueError('Duplicate review pin')
    used_source, used_records = set(), set()
    reports = []
    for name in sorted(ARCHIVES):
        row, expected = actual[name], inventory[name]
        if (row['sha256'] != expected['sha256'] or row['files'] != expected['files']
                or sorted(row['compressed_rules']) != sorted(expected['compressed_rules'])):
            raise ValueError('Archive identity, member coverage or outer rules changed')
        for match in row['expanded_matches']:
            if name == 'runtime-source.tar.gz':
                key = match['file'], match['sha256']
                pin = source.get(key)
                used_source.add(key)
            else:
                key = name, match['file'], match['sha256']
                pin = records.get(key)
                used_records.add(key)
            if not pin or set(pin['rules']) != set(match['rules']):
                raise ValueError('Expanded match has no exact reviewed file and rule decision')
        reports.append({'file': name, 'sha256': row['sha256'], 'files': row['files'],
                        'compressed_rules': sorted(row['compressed_rules']),
                        'expanded_matches_reviewed': len(row['expanded_matches'])})
    if used_source != set(source) or used_records != set(records):
        raise ValueError('A reviewed source or record pin was not exercised')
    return {'runtime_archive_sha256': review['runtime_archive_sha256'],
            'scanner_sha256': review['scanner_sha256'], 'policy_sha256': review['policy_sha256'],
            'archives': reports, 'unresolved_matches': 0}


def store(path, result, check):
    if check:
        if json.loads(path.read_text()) != result:
            raise ValueError('Sealed boundary verification differs from this reproduction')
    else:
        path.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='context-boundary-check-') as directory:
        output = Path(directory)/'scan.json'
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('scan_archives.py')),
                                 '--receipts', str(args.receipts), '--prefix', args.prefix,
                                 '--output', str(output)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        checked = verify(json.loads(output.read_text()), json.loads(args.review.read_text()))
    store(args.receipts/'boundary-verification.json', checked, args.check)
    print(json.dumps(checked))


if __name__ == '__main__':
    main()
