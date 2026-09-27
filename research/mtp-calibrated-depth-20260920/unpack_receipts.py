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


def unpack(lane, destination):
    manifest = json.loads((lane / 'receipts/manifest.json').read_text())
    if destination.exists():
        raise ValueError('destination must be new')
    payloads, shared, seen = [], [], set()
    for row in manifest['archives']:
        name = row['file']
        if not re.fullmatch(r'(qwen|gemma)-records\.tar\.gz', name):
            raise ValueError('invalid archive filename')
        path = lane / 'receipts' / name
        if path.is_symlink():
            raise ValueError('archive must not be a symlink')
        data = path.read_bytes()
        if len(data) != row['bytes'] or digest(data) != row['sha256']:
            raise ValueError('archive hash/size mismatch')
        checks = lane / 'receipts' / name.replace('.tar.gz', '.sha256')
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
                        or not parts[0].startswith(family + '-') or member.name in seen):
                    raise ValueError('unsafe or duplicate archive member')
                if digest(archive.extractfile(member).read()) != expected.get(member.name):
                    raise ValueError('member hash mismatch')
                seen.add(member.name)
                members.add(member.name)
        if members != set(expected):
            raise ValueError('missing archive members')
        payloads.append(data)
    if not payloads:
        raise ValueError('manifest has no archives')
    for row in manifest['shared']:
        name = row['file']
        if Path(name).name != name or name in seen:
            raise ValueError('unsafe shared filename')
        data = (lane / name).read_bytes()
        if digest(data) != row['sha256']:
            raise ValueError('shared file hash mismatch')
        seen.add(name)
        shared.append((name, data))
    destination.mkdir(parents=True)
    for data in payloads:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    for name, data in shared:
        (destination / name).write_bytes(data)
    return len(payloads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    count = unpack(Path(__file__).resolve().parent, args.destination)
    print(f'Verified and unpacked {count} archives')
