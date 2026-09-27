"""Verify a closed three-archive inventory before extracting any regular file."""
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

NAMES = {'qwen-records.tar.gz', 'gemma-records.tar.gz', 'common-records.tar.gz'}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def unpack(receipts, destination, expected_manifest=None):
    receipts, destination = Path(receipts), Path(destination)
    manifest_path = receipts/'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if expected_manifest and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest:
        raise ValueError('Manifest differs from the reviewed checksum')
    manifest = json.loads(manifest_bytes)
    rows = manifest['archives']
    if len(rows) != 3 or {r['file'] for r in rows} != NAMES:
        raise ValueError('The complete three-archive inventory is required')
    if destination.exists():
        raise ValueError('Destination must be new')
    seen = set()
    payloads = []
    for row in rows:
        path = receipts/row['file']
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024**2:
            raise ValueError('Archive is not a bounded regular file')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Archive hash or size mismatch')
        checks = receipts/row['file'].replace('.tar.gz', '.sha256')
        if checks.is_symlink():
            raise ValueError('Member manifest must not be a symlink')
        check_bytes = checks.read_bytes()
        if hashlib.sha256(check_bytes).hexdigest() != row['file_manifest_sha256']:
            raise ValueError('Member manifest changed')
        expected = {}
        for line in check_bytes.decode().splitlines():
            sha, name = line.split('  ', 1)
            if name in expected or not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError('Duplicate or invalid member declaration')
            expected[name] = sha
        if len(expected) != row['files']:
            raise ValueError('Member count changed')
        actual = set()
        total_bytes = 0
        family = row['file'].split('-')[0]
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or not parts or PurePosixPath(member.name).is_absolute() or '..' in parts or '\\' in member.name or member.name in seen or PurePosixPath(member.name).as_posix() != member.name:
                    raise ValueError('Unsafe or duplicate archive member')
                if family != 'common' and (len(parts) < 2 or parts[0] != 'experiment' or not parts[1].startswith(family+'-')):
                    raise ValueError('Archive member belongs to another family')
                total_bytes += member.size
                if total_bytes > 4 * 1024**3:
                    raise ValueError('Expanded archive exceeds the study bound')
                with archive.extractfile(member) as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                if expected.get(member.name) != sha:
                    raise ValueError('Member hash mismatch')
                actual.add(member.name); seen.add(member.name)
        if actual != set(expected):
            raise ValueError('Missing archive members')
        payloads.append(data)
    destination.mkdir(parents=True)
    for data in payloads:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(destination, filter='data')
    return manifest
