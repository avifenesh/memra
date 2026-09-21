"""Check every expanded record and document compressed-byte false positives."""
import hashlib
import json
import re
import tarfile
import tomllib
from pathlib import Path

REVIEWED_RECORD = (
    'gemma-records.tar.gz',
    'gemma-mtp-followup-calibration-0/20263000-k2/turn-7.answer.txt',
    '8ebaff29d51fb516486a97dbff12133f910b0e148a0334923bb591c9646d2c5c',
    'provider_name_aws',
)


def verify():
    lane = Path(__file__).resolve().parent
    policy = tomllib.loads((lane.parents[1] / 'tools/public-boundary-policy.toml').read_text())
    rules = {name: re.compile(pattern) for name, pattern in policy['secret_patterns'].items()}
    reports = []
    for path in sorted((lane / 'receipts').glob('*-records.tar.gz')):
        data = path.read_bytes()
        text = data.decode('utf-8', errors='ignore')
        compressed = {name: len(pattern.findall(text)) for name, pattern in rules.items() if pattern.search(text)}
        files = 0
        expanded_hits = []
        reviewed_hits = []
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                assert member.isfile()
                files += 1
                content = archive.extractfile(member).read()
                text = content.decode('utf-8', errors='ignore')
                for name, pattern in rules.items():
                    if pattern.search(text):
                        hit = {'file': member.name, 'rule': name}
                        if (path.name, member.name, hashlib.sha256(content).hexdigest(), name) == REVIEWED_RECORD:
                            reviewed_hits.append({**hit, 'reason': 'Generated hypothetical instance-type example; no deployment or account identity.'})
                        else:
                            expanded_hits.append(hit)
        reports.append({'file': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                        'compressed_hits': compressed, 'expanded_files_checked': files,
                        'expanded_matches': expanded_hits, 'reviewed_record_matches': reviewed_hits})
    return reports


if __name__ == '__main__':
    reports = verify()
    print(json.dumps(reports, indent=2))
    assert all(not row['expanded_matches'] for row in reports)
