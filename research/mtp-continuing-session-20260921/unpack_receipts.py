"""Verify frozen archives and extract regular files into a new directory."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unpack(receipts, destination):
    manifest = json.loads((receipts / 'manifest.json').read_text())
    if destination.exists():
        raise ValueError('destination must be new')
    payloads, seen = [], set()
    for row in manifest['archives']:
        name = row['file']
        if not re.fullmatch(r'(qwen|gemma|common)-records\.tar\.gz', name):
            raise ValueError('invalid archive filename')
        path = receipts / name
        if path.is_symlink():
            raise ValueError('archive must not be a symlink')
        data = path.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('archive hash/size mismatch')
        checks = receipts / name.replace('.tar.gz', '.sha256')
        if digest(checks.read_bytes()) != row['file_manifest_sha256']:
            raise ValueError('file manifest mismatch')
        expected = dict(line.split('  ', 1)[::-1] for line in checks.read_text().splitlines())
        family = name.split('-')[0]
        members = set()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if (not member.isfile() or not parts or '..' in parts
                        or PurePosixPath(member.name).is_absolute() or '\\' in member.name
                        or member.name in seen):
                    raise ValueError('unsafe or duplicate archive member')
                if family != 'common' and (len(parts) < 2 or parts[0] not in ('experiment', 'qualification')
                        or not parts[1].startswith(family + '-')):
                    raise ValueError('member belongs to a different study family')
                if digest(archive.extractfile(member).read()) != expected.get(member.name):
                    raise ValueError('member hash mismatch')
                seen.add(member.name)
                members.add(member.name)
        if members != set(expected):
            raise ValueError('missing archive members')
        payloads.append(data)
    if not payloads:
        raise ValueError('manifest has no archives')
    if manifest.get('shared'):
        raise ValueError('shared files must be inside the common archive')
    destination.mkdir(parents=True)
    for data in payloads:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    return len(payloads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipts', type=Path, default=Path(__file__).resolve().parent / 'receipts')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    count = unpack(args.receipts, args.destination)
    print(f'Verified and unpacked {count} archives')
