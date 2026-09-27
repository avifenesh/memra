"""Verify and unpack immutable MTP receipts into a new directory (Python 3.12+)."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


def unpack(manifest_path, archives, destination):
    rows = json.loads(manifest_path.read_text())['archives']
    if not rows or len({r['file'] for r in rows}) != len(rows):
        raise ValueError('manifest must name nonempty, unique archives')
    if destination.exists():
        raise ValueError('destination must not already exist')
    if not hasattr(tarfile, 'data_filter'):
        raise RuntimeError('use Python 3.12+ for safe tar extraction')
    # Retain the validated snapshots so extraction cannot reopen changed files.
    payloads = []
    for row in rows:
        name = row['file']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*\.tar\.gz', name):
            raise ValueError('invalid archive filename')
        path = archives / name
        if path.is_symlink():
            raise ValueError('archive must be a regular file, not a symlink')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'archive hash/size mismatch: {name}')
        root = name.removesuffix('.tar.gz')
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if (not parts or parts[0] != root or '..' in parts
                        or PurePosixPath(member.name).is_absolute()
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'unsafe archive member: {member.name}')
        payloads.append(data)
    destination.mkdir(parents=True)
    for data in payloads:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('receipts-expanded'))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    count = unpack(here / 'receipts/manifest.json', here / 'receipts/archives', args.destination)
    print(f'Verified and unpacked {count} archives into {args.destination}')
