"""Exact data reads of the pinned support-record census, without running a model."""

import hashlib
from pathlib import PurePosixPath
import tomllib


RECORDS = 'docs/support-records.toml'
READERS = {
    'tools/check-support-states.py': '27b4a68fb545cfd434caab8764f42becd021a52d7d6a36fb9fba466d396cf8ec',
    'tools/test_check_support_states.py': '366556ebf11d27173032f1fe9cd71ef59d18b3c1f2314f920f214a63e5482050',
}
GATES = {'Config', 'TokenizerTemplate', 'TensorCensus', 'TinyParity',
         'CheckpointParity', 'RewriteParity', 'Serve'}


class InputContractError(ValueError):
    pass


def canonical_path(value):
    if not isinstance(value, str) or not value or any(c in value for c in '\n\r\t\0\\'):
        raise InputContractError('unmodelled support evidence path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value or value == '.':
        raise InputContractError('noncanonical support evidence path: ' + value)
    return value


def resolve(tree):
    """Required evidence plus both exact potential family sidecars in this tree.

    A tree with none of this optional test fixture's census inputs has no data
    reader. Partial or changed readers are unknown and must expand validation.
    Presence does not turn optional alternatives into required execution inputs.
    """
    paths = set(tree.paths('tools', 'docs'))
    expected = set(READERS) | {RECORDS}
    if not (expected & paths):
        return {'required': [], 'optional': []}
    if not expected <= paths:
        raise InputContractError('support data reader inputs are incomplete')
    for name, digest in READERS.items():
        if hashlib.sha256(tree.read(name).encode()).hexdigest() != digest:
            raise InputContractError('unmodelled support data reader: ' + name)
    try:
        metadata = tomllib.loads(tree.read(RECORDS))
    except (tomllib.TOMLDecodeError, UnicodeError) as error:
        raise InputContractError('unmodelled support record metadata') from error
    records = metadata.get('record')
    if not isinstance(records, list) or not records:
        raise InputContractError('support metadata needs nonempty record entries')
    required, optional = set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise InputContractError('unmodelled support record entry')
        gates, evidence = record.get('gates'), record.get('evidence')
        if not isinstance(gates, dict) or not isinstance(evidence, dict):
            raise InputContractError('unmodelled support gate/evidence mapping')
        if set(gates) - GATES or set(evidence) - GATES:
            raise InputContractError('unmodelled support gate name')
        for gate, status in gates.items():
            if status not in ('passed', 'pending'):
                raise InputContractError('unmodelled support gate status')
            sources = evidence.get(gate, [])
            if not isinstance(sources, list) or any(not isinstance(p, str) for p in sources):
                raise InputContractError('unmodelled support evidence list')
            if status == 'pending':
                if sources:
                    raise InputContractError('pending support gate cites evidence')
                continue
            if not sources:
                raise InputContractError('passed support gate has no evidence')
            for source in sources:
                if source == 'ci:verify-tiny' and gate in ('Config', 'TinyParity'):
                    continue
                if source.startswith('ci:'):
                    raise InputContractError('unmodelled support CI evidence')
                name = canonical_path(source)
                required.add(name)
                parent = PurePosixPath(name).parent
                optional.update(str(parent / n) for n in ('artifact.lock', 'tiny-gate.tsv'))
    checked = set()
    for name in expected | required | optional:
        path = PurePosixPath(name)
        checked.update(str(parent) for parent in (path, *path.parents) if str(parent) != '.')
    links = tree.symlinks_exact(checked)
    if links:
        raise InputContractError('support data reader path contains a symlink: ' + sorted(links)[0])
    return {'required': sorted(required), 'optional': sorted(optional)}
