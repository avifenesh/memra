"""Exact data reads of the pinned support-record census, without running a model."""

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
import tomllib


RECORDS = 'docs/support-records.toml'
READERS = {
    'tools/check-support-states.py': '27b4a68fb545cfd434caab8764f42becd021a52d7d6a36fb9fba466d396cf8ec',
    'tools/test_check_support_states.py': '366556ebf11d27173032f1fe9cd71ef59d18b3c1f2314f920f214a63e5482050',
}
GATES = {'Config', 'TokenizerTemplate', 'TensorCensus', 'TinyParity',
         'CheckpointParity', 'RewriteParity', 'Serve'}
PACK_ROOT = 'crates/memra-gguf/src/model_packs'
CLI_SOURCE = 'crates/memra-cli/src/lib.rs'
ROOT_DOCS = {'README.md', 'STATUS.md', 'AGENTS.md'}
COPY_ROOTS = (PACK_ROOT, 'docs')


class InputContractError(ValueError):
    pass


class UnmodelledReader(InputContractError):
    pass


class DirectoryTree:
    """Execution admission needs these exact files, not Git metadata."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def paths(self, *prefixes):
        found = []
        for name in (*READERS, RECORDS):
            path = self.root / name
            # A broken link or non-directory parent remains an unsafe input, even if
            # exists() would hide all of its children.
            if (path.exists() or path.is_symlink() or
                    any(parent.is_symlink() or (parent.exists() and not parent.is_dir())
                        for parent in path.parents
                        if parent.is_relative_to(self.root))):
                found.append(name)
        return found

    def _open_regular(self, name):
        """Anchor each component; never read a FIFO or follow a replaced link."""
        parts = PurePosixPath(canonical_path(name)).parts
        parent = None
        leaf = None
        try:
            parent = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
                os.close(parent)
                parent = child
            info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise InputContractError('support data reader input contains a symlink or '
                                         'is not a regular file: ' + name)
            # NONBLOCK protects the open itself if a regular file is replaced by
            # a FIFO after stat. fstat checks the object actually opened.
            leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                           dir_fd=parent)
            if not stat.S_ISREG(os.fstat(leaf).st_mode):
                raise InputContractError('support data reader input is not a regular file: ' + name)
            result, leaf = leaf, None
            return result
        except OSError as error:
            raise InputContractError('support data reader input is missing, nonregular or '
                                     'contains a symlink: ' + name) from error
        finally:
            if leaf is not None:
                os.close(leaf)
            if parent is not None:
                os.close(parent)

    def validate_inputs(self, names):
        for name in sorted(names):
            os.close(self._open_regular(name))

    def read(self, name):
        with os.fdopen(self._open_regular(name), 'r') as source:
            return source.read()

    def read_bytes(self, name):
        with os.fdopen(self._open_regular(name), 'rb') as source:
            return source.read()

    def symlinks_exact(self, paths):
        return {name: os.readlink(self.root / name) for name in paths
                if (self.root / name).is_symlink()}


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
    if isinstance(tree, DirectoryTree):
        tree.validate_inputs(expected)
    unknown_readers = [name for name, digest in READERS.items()
                       if hashlib.sha256(tree.read_bytes(name)).hexdigest() != digest]
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
    if unknown_readers:
        raise UnmodelledReader('unmodelled support data reader: ' + unknown_readers[0])
    return {'required': sorted(required), 'optional': sorted(optional)}


def reads_source_doc(name):
    """Potential content paths of the pinned checker, including new/deleted files."""
    if name in ROOT_DOCS or name in (CLI_SOURCE, PACK_ROOT + '/mod.rs'):
        return True
    if name.startswith(PACK_ROOT + '/') or name.startswith('docs/'):
        canonical_path(name)
    if name.startswith(PACK_ROOT + '/'):
        relative = PurePosixPath(name).relative_to(PACK_ROOT).parts
        return len(relative) == 2 and relative[-1] == 'mod.rs'
    return name.startswith('docs/') and name.endswith('.md') and not name.startswith('docs/archive/')


def source_docs(tree):
    """Content inventory plus fixture-copy type safety for the two pinned readers.

    Excluded regular content remains excluded. Copying an ambiguous file type can
    fail the CPU test even when the checker does not read that file's content.
    """
    expected = set(READERS) | {RECORDS}
    if not (expected & set(tree.paths('tools', 'docs'))):
        return {'active': False, 'inputs': []}
    roots = set(COPY_ROOTS) | ROOT_DOCS | {CLI_SOURCE}
    ancestors = {str(parent) for name in roots for parent in PurePosixPath(name).parents
                 if str(parent) != '.'}
    exact = tree.input_modes(*sorted(roots | ancestors), recursive=False)
    directories = set(COPY_ROOTS) | ancestors
    bad = [name for name, mode in exact.items()
           if (mode != '040000' if name in directories else mode not in ('100644', '100755'))]
    if bad:
        raise InputContractError('ambiguous support fixture copy input type: ' + sorted(set(bad))[0])
    copied = tree.input_modes(*COPY_ROOTS)
    bad.extend(name for name, mode in copied.items() if mode not in ('100644', '100755'))
    if bad:
        raise InputContractError('ambiguous support fixture copy input type: ' + sorted(set(bad))[0])
    ignored = [name for name in tree.ignored_paths(*COPY_ROOTS, *sorted(ROOT_DOCS), CLI_SOURCE)
               if reads_source_doc(name)]
    if ignored:
        raise InputContractError('ignored support content input prevents scoped planning: ' + sorted(ignored)[0])
    resolve(tree)
    paths = set(tree.paths(*COPY_ROOTS)) | ROOT_DOCS | {CLI_SOURCE, PACK_ROOT + '/mod.rs'}
    return {'active': True, 'inputs': sorted(name for name in paths if reads_source_doc(name))}
