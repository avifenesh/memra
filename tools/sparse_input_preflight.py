#!/usr/bin/env python3
"""Read-only materialization check for three pinned CPU input contracts.

This does not run a consumer or establish content/native qualification. Expected
inputs come from one immutable Git tree, never from the sparse working inventory.
"""

import argparse
import ast
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tomllib


READER_PINS = {
    'tools/hooks/pre-push':
        '7f6a52c5432a09aee2125e5dfe03c927a545f3e9b61b2dfca2f3decac49877db',
    'tools/update-perf-board.py':
        '4f1a20c47639bdcf1f53f622e3b13bf1336dc5a42d19f2977739cfdc3ac37c4e',
    'tools/support_record_inputs.py':
        'd3548a8fa3f765046e1864efd78c028d596151971a45908dad38230f021276d5',
    'tools/check-support-states.py':
        '27b4a68fb545cfd434caab8764f42becd021a52d7d6a36fb9fba466d396cf8ec',
    'tools/test_check_support_states.py':
        '366556ebf11d27173032f1fe9cd71ef59d18b3c1f2314f920f214a63e5482050',
    'tools/test_public_boundary.py':
        '8c56eb6bff531e71d80983294e2988bb9507173b37994ee138bca6d267004b0f',
    'tools/check-public-boundary.py':
        '33a35dc59df398a0dab24aba2fc8f73c1d29f35c49c688ffab25f108953ce4f1',
}

CONTRACTS = {
    'perf-board': ('tools/hooks/pre-push', 'tools/update-perf-board.py'),
    'support-records': ('tools/support_record_inputs.py',
                        'tools/check-support-states.py', 'tools/test_check_support_states.py'),
    'public-boundary-links': ('tools/test_public_boundary.py', 'tools/check-public-boundary.py'),
}
PACK_ROOT = 'crates/memra-gguf/src/model_packs'
CLI_SOURCE = 'crates/memra-cli/src/lib.rs'
RECORDS = 'docs/support-records.toml'
LEGACY = 'research/modelplan-onboarding-hy3-20260830/tiny'
GATES = {'Config', 'TokenizerTemplate', 'TensorCensus', 'TinyParity',
         'CheckpointParity', 'RewriteParity', 'Serve'}


class Refusal(ValueError):
    pass


def canonical(name):
    if (not isinstance(name, str) or not name or
            any(c in name for c in '\n\r\t\0\\') or
            PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
            PurePosixPath(name).as_posix() != name or name == '.'):
        raise Refusal('noncanonical Git/input path: ' + repr(name))
    return name


class GitTree:
    def __init__(self, root_fd, ref):
        self.root_fd = root_fd
        self.commit = self.git('rev-parse', '--verify', '--end-of-options', ref + '^{commit}').strip().decode()
        self.algorithm = self.git('rev-parse', '--show-object-format').strip().decode()
        if self.algorithm not in ('sha1', 'sha256'):
            raise Refusal('unknown Git object format')
        self.entries = {}
        self.copy_roots = set()
        self.direct_inputs = set()
        self.blob_cache = {}
        for row in self.git('ls-tree', '-r', '-t', '-z', self.commit).split(b'\0'):
            if row:
                meta, path = row.split(b'\t', 1)
                mode, kind, oid = meta.decode().split()
                name = canonical(path.decode())
                self.entries[name] = (mode, kind, oid)

    def git(self, *args):
        result = subprocess.run(['git', '--no-replace-objects', '-C',
                                 '/proc/self/fd/' + str(self.root_fd), *args],
                                pass_fds=(self.root_fd,), capture_output=True, timeout=30,
                                env=dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0'))
        if result.returncode:
            # Do not print config, remote, credential or arbitrary Git stderr.
            raise Refusal('Git reader failed: ' + args[0])
        return result.stdout

    def blob(self, name):
        entry = self.entries.get(canonical(name))
        if entry is None or entry[1] != 'blob':
            raise Refusal('required regular/blob input absent from pinned Git tree: ' + name)
        if entry[2] in self.blob_cache:
            return self.blob_cache[entry[2]]
        if int(self.git('cat-file', '-s', entry[2])) > 4 * 1024 * 1024:
            raise Refusal('modeled source/metadata/link exceeds read bound: ' + name)
        value = self.git('cat-file', 'blob', entry[2])
        self.blob_cache[entry[2]] = value
        return value

    def regular(self, name):
        entry = self.entries.get(canonical(name))
        if entry is None or entry[0] not in ('100644', '100755'):
            raise Refusal('required regular input missing or wrong Git type: ' + name)
        self.direct_inputs.add(name)
        return name

    def subtree(self, name):
        if self.entries.get(canonical(name), (None,))[0] != '040000':
            raise Refusal('required copy root missing or wrong Git type: ' + name)
        return {n for n in self.entries if n == name or n.startswith(name + '/')}


def validate_link_index(tree, links):
    """The pinned boundary assertion enumerates the index, not just HEAD."""
    indexed = {}
    for row in tree.git('ls-files', '--stage', '-z').split(b'\0'):
        if row:
            meta, path = row.split(b'\t', 1)
            mode, oid, stage = meta.decode().split()
            name = path.decode()
            if mode == '120000' or name in links:
                canonical(name)
                if stage != '0' or name in indexed:
                    raise Refusal('unmerged/ambiguous index link input: ' + name)
                indexed[name] = (mode, oid)
    expected = {n: (tree.entries[n][0], tree.entries[n][2]) for n in links}
    if indexed != expected:
        differing = sorted(n for n in indexed.keys() | expected.keys()
                           if indexed.get(n) != expected.get(n))
        raise Refusal('index link inventory differs from pinned Git tree: ' + differing[0])


def modeled_inputs(tree, checks):
    if not checks:
        raise Refusal('empty reader contract selection')
    required = set()
    for check in checks:
        if check not in CONTRACTS:
            raise Refusal('unknown reader contract: ' + check)
        for source in CONTRACTS[check]:
            tree.regular(source)
            if hashlib.sha256(tree.blob(source)).hexdigest() != READER_PINS[source]:
                raise Refusal('unknown reader source/version: ' + source)
            required.add(source)
        if check == 'perf-board':
            required.update(tree.regular(n) for n in (
                'research/tune-data/current-board.json', 'docs/MODELS.md', 'docs/PERFORMANCE.md'))
            source = ast.parse(tree.blob('tools/update-perf-board.py'))
            maps = [node.value for node in source.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'MODEL_CARD_PATHS'
                            for t in node.targets)]
            if len(maps) != 1:
                raise Refusal('unknown perf model-card read shape')
            cards = ast.literal_eval(maps[0])
            if not isinstance(cards, dict) or any(not isinstance(n, str) for n in cards.values()):
                raise Refusal('unknown perf model-card map')
            required.update(tree.regular('docs/models/' + canonical(n)) for n in cards.values())
        elif check == 'support-records':
            required.add(tree.regular(RECORDS))
            data = tomllib.loads(tree.blob(RECORDS).decode())
            records = data.get('record')
            if not isinstance(records, list) or not records:
                raise Refusal('unknown support record read shape')
            evidence_paths, optional = set(), set()
            for record in records:
                if not isinstance(record, dict):
                    raise Refusal('unknown support record entry')
                gates, evidence = record.get('gates'), record.get('evidence')
                if (not isinstance(gates, dict) or not isinstance(evidence, dict) or
                        set(gates) - GATES or set(evidence) - GATES or set(evidence) - set(gates)):
                    raise Refusal('unknown support gate/evidence mapping')
                for gate, status in gates.items():
                    sources = evidence.get(gate, [])
                    if (status not in ('passed', 'pending') or not isinstance(sources, list) or
                            any(not isinstance(p, str) for p in sources) or
                            (status == 'pending' and sources) or (status == 'passed' and not sources)):
                        raise Refusal('unknown support gate status/evidence')
                    for name in sources:
                        if name == 'ci:verify-tiny' and gate in ('Config', 'TinyParity'):
                            continue
                        if name.startswith('ci:'):
                            raise Refusal('unknown support CI evidence')
                        evidence_paths.add(tree.regular(name))
                        parent = PurePosixPath(name).parent
                        optional.update(str(parent / n) for n in ('artifact.lock', 'tiny-gate.tsv'))
            required.update(evidence_paths)
            # Optional siblings are required materialization only when tracked.
            required.update(tree.regular(n) for n in optional if n in tree.entries)
            roots = {str(PurePosixPath(n).parent) for n in evidence_paths} | {LEGACY, PACK_ROOT, 'docs'}
            if '.' in roots:
                raise Refusal('support fixture copies repository root')
            tree.copy_roots.update(roots)
            for root in roots:
                copied = tree.subtree(root)
                for n in copied:
                    if tree.entries[n][0] not in ('040000', '100644', '100755'):
                        raise Refusal('ambiguous support copy input Git type: ' + n)
                required.update(copied)
            required.update(tree.regular(n) for n in ('README.md', 'STATUS.md', 'AGENTS.md',
                                                       CLI_SOURCE, PACK_ROOT + '/mod.rs'))
        else:
            links = {n for n, entry in tree.entries.items() if entry[0] == '120000'}
            validate_link_index(tree, links)
            required.update(links)
            tree.direct_inputs.update(links)
    return required


def link_closure(tree, required):
    """Resolve components using Git blobs, including links in target parents."""
    closure = set(required)

    def resolve(parts, active, budget):
        prefix = []
        for offset, part in enumerate(parts):
            if part in ('.', ''):
                continue
            if part == '..':
                if not prefix:
                    raise Refusal('link escapes pinned Git tree')
                prefix.pop()
                continue
            prefix.append(part)
            name = '/'.join(prefix)
            entry = tree.entries.get(name)
            if entry is None:
                raise Refusal('link target absent from pinned Git tree: ' + name)
            closure.add(name)
            if entry[0] == '120000':
                if name in active or budget[0] == 0:
                    raise Refusal('link cycle/depth in pinned Git tree: ' + name)
                budget[0] -= 1
                target = tree.blob(name).decode()
                if (not target or target.startswith('/') or
                        any(c in target for c in '\0\n\r\t\\')):
                    raise Refusal('absolute/invalid link target in pinned Git tree: ' + name)
                # A link is active only while expanding its own target. The
                # remaining path may validly traverse it again after '..'.
                prefix = resolve(prefix[:-1] + target.split('/'), active + (name,), budget).split('/')
                if prefix == ['']:
                    prefix = []
                if offset < len(parts) - 1 and prefix and tree.entries['/'.join(prefix)][0] != '040000':
                    raise Refusal('link target parent is not a directory: ' + name)
                continue
            if offset < len(parts) - 1 and entry[0] != '040000':
                raise Refusal('link target parent is not a directory: ' + name)
        return '/'.join(prefix)

    for name in sorted(required):
        if tree.entries[name][0] == '120000':
            target = resolve(name.split('/'), (), [40])
            if not target:
                raise Refusal('link targets repository root: ' + name)
            if tree.entries[target][0] == '040000':
                closure.update(tree.subtree(target))
            else:
                tree.direct_inputs.add(target)
    # Directory target subtrees may include additional links. All links are
    # modeled when selecting the boundary contract; other contracts forbid them.
    for name in tuple(closure):
        closure.update(str(p) for p in PurePosixPath(name).parents if str(p) != '.')
    return closure


def inspect_path(root_fd, tree, name):
    """No-follow, descriptor-anchored ancestors and leaf, with nonblocking open."""
    parent = os.dup(root_fd)
    leaf = None
    try:
        parts = name.split('/')
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        mode, kind, oid = tree.entries[name]
        if mode == '120000':
            if not stat.S_ISLNK(before.st_mode):
                return 'type-mismatch'
            target = os.readlink(os.fsencode(parts[-1]), dir_fd=parent)
            if target != tree.blob(name):
                return 'link-target-mismatch'
        elif mode == '040000':
            leaf = os.open(parts[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        elif mode in ('100644', '100755'):
            if not stat.S_ISREG(before.st_mode):
                return 'type-mismatch'
            leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            actual = os.fstat(leaf)
            if not stat.S_ISREG(actual.st_mode):
                return 'type-mismatch'
            if bool(actual.st_mode & stat.S_IXUSR) != (mode == '100755'):
                return 'executable-mode-mismatch'
            digest = hashlib.new(tree.algorithm)
            digest.update(b'blob ' + str(actual.st_size).encode() + b'\0')
            while block := os.read(leaf, 1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != oid:
                return 'blob-mismatch'
            after = os.fstat(leaf)
            if (actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                return 'changed-during-read'
        else:
            return 'unsupported-Git-type'
        after = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if (before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            return 'changed-during-read'
        return None
    except OSError as error:
        return {errno.ENOENT: 'missing-materialization', errno.EACCES: 'unreadable',
                errno.EPERM: 'unreadable', errno.ENOTDIR: 'unsafe-parent-or-type',
                errno.ELOOP: 'unsafe-parent-or-link'}.get(error.errno, 'filesystem-reader-failed')
    finally:
        if leaf is not None:
            os.close(leaf)
        os.close(parent)


def open_root(root):
    """Anchor the root too, refusing symlinks in every absolute component."""
    if '..' in Path(root).parts:
        raise Refusal('root parent traversal is not canonical')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in Path(os.path.abspath(root)).parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def copy_extras(root_fd, tree):
    """Unknown copied inputs refuse without reading their content or links."""
    problems = []
    for root in sorted(tree.copy_roots):
        fd = os.dup(root_fd)
        try:
            for part in root.split('/'):
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child

            def walk(directory, prefix):
                with os.scandir(directory) as children:
                    for child in children:
                        name = prefix + '/' + child.name
                        if name not in tree.entries:
                            problems.append({'path': name, 'reason': 'unmodeled-copy-input'})
                        elif tree.entries[name][0] == '040000':
                            nested = os.open(child.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                             dir_fd=directory)
                            try:
                                walk(nested, name)
                            finally:
                                os.close(nested)
            walk(fd, root)
        except OSError:
            # Existing per-path checks identify the missing/unsafe component;
            # enumeration failure also refuses, even if per-path reads passed.
            problems.append({'path': root, 'reason': 'copy-enumeration-failed'})
        finally:
            os.close(fd)
    return problems


def sparse_suggestions(tree, problems):
    """Cone additions use leaf parents, never omitted ancestor directories."""
    parents = {str(PurePosixPath(p['path']).parent) for p in problems
               if p['reason'] == 'missing-materialization' and
               tree.entries[p['path']][0] != '040000'} - {'.'}
    minimal = []
    for parent in sorted(parents, key=lambda n: (len(PurePosixPath(n).parts), n)):
        if not any(parent == root or parent.startswith(root + '/') for root in minimal):
            minimal.append(parent)
    return sorted(minimal)


def preflight(root, ref, checks):
    root_fd = open_root(root)
    try:
        tree = GitTree(root_fd, ref)
        required = link_closure(tree, modeled_inputs(tree, checks))
        problems = copy_extras(root_fd, tree)
        for name in sorted(required):
            problem = inspect_path(root_fd, tree, name)
            if problem:
                problems.append({'path': name, 'reason': problem})
        # Required consumer leaves precede directories and recursively copied
        # bulk files, so bounded diagnostics still identify the direct omission.
        problems.sort(key=lambda p: (p['path'] not in tree.direct_inputs, p['path']))
        suggestions = sparse_suggestions(tree, problems)
        return {'ok': not problems, 'commit': tree.commit, 'checks': checks,
                'required_count': len(required), 'problem_count': len(problems),
                'problems': problems[:200], 'problems_truncated': len(problems) > 200,
                'suggested_sparse_paths': suggestions[:200],
                'suggested_sparse_path_count': len(suggestions),
                'suggestions_truncated': len(suggestions) > 200,
                'qualification': False}
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).absolute().parent.parent))
    parser.add_argument('--ref', default='HEAD', help='Pinned commit to compare materialization against')
    parser.add_argument('--check', action='append', help='perf-board, support-records, public-boundary-links; default all')
    args = parser.parse_args()
    try:
        result = preflight(args.root, args.ref, args.check or list(CONTRACTS))
    except (Refusal, OSError, ValueError, UnicodeError, subprocess.TimeoutExpired) as error:
        result = {'ok': False, 'refusal': str(error), 'qualification': False}
    print(json.dumps(result, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
