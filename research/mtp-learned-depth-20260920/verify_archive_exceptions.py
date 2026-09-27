"""Reproduce the rule-scoped false positives in immutable gzip bytes."""
import hashlib
import json
import re
import tarfile
import tomllib
from pathlib import Path


def verify():
    here = Path(__file__).resolve().parent
    policy = tomllib.loads((here.parents[1] / 'tools/public-boundary-policy.toml').read_text())
    records = json.loads((here / 'receipts/archive-boundary-exceptions.json').read_text())
    result = []
    for row in records:
        assert Path(row['file']).name == row['file']
        path = here / 'receipts/archives' / row['file']
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        pattern = re.compile(policy['secret_patterns'][row['rule']])
        compressed_hits = len(pattern.findall(data.decode('utf-8', errors='ignore')))
        assert compressed_hits > 0
        files = 0
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    files += 1
                    content = archive.extractfile(member).read().decode('utf-8', errors='ignore')
                    assert pattern.search(content) is None, member.name
        result.append({'file': row['file'], 'sha256': row['sha256'], 'rule': row['rule'],
                       'compressed_hits': compressed_hits, 'expanded_files_checked': files,
                       'expanded_matches': 0})
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
