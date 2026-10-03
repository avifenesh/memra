#!/usr/bin/env python3
"""One real Git witness with original consumers and coherent preflight mutants.

Run with ordinary Python (assertions enabled). All fixture mutations stay in one
owned temporary repository. No checkout, model, credential or network actions
are performed outside that repository.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import tomllib

import sparse_input_preflight as preflight


REPO = Path(__file__).absolute().parent.parent
SCRIPT = Path(__file__).with_name('sparse_input_preflight.py')
PRODUCERS = (
    'tools/hooks/pre-push', 'tools/update-perf-board.py', 'tools/support_record_inputs.py',
    'tools/check-support-states.py', 'tools/test_check_support_states.py',
    'tools/test_public_boundary.py', 'tools/check-public-boundary.py',
)
BOARD = 'research/tune-data/current-board.json'
LEGACY = 'research/modelplan-onboarding-hy3-20260830/tiny'


def command(root, *args, success=True):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=60,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    if success:
        assert result.returncode == 0, (args, result.stdout, result.stderr)
    return result


def git(root, *args):
    return command(root, 'git', *args).stdout.strip()


def consumer(root, relative, *args, success=True):
    return command(root, sys.executable, str(root / relative), *args, success=success)


def populate(root):
    """Independent inventory from the original copy fixture and board consumer."""
    metadata = tomllib.loads(git(REPO, 'show', 'HEAD:docs/support-records.toml'))
    cited = {p for r in metadata['record'] for ps in r.get('evidence', {}).values()
             for p in ps if not p.startswith('ci:')}
    roots = {'docs', 'crates/memra-gguf/src/model_packs', LEGACY}
    roots.update(str(PurePosixPath(n).parent) for n in cited)
    names = sorted(roots | set(PRODUCERS) | {BOARD, 'README.md', 'STATUS.md', 'AGENTS.md',
                                           'crates/memra-cli/src/lib.rs'})
    rows = subprocess.check_output(['git', '-C', str(REPO), '--literal-pathspecs',
                                    'ls-tree', '-r', '-z', 'HEAD', '--', *names]).split(b'\0')
    inputs = []
    for row in rows:
        if row:
            meta, name = row.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            assert mode in ('100644', '100755'), (name, mode)
            inputs.append((name.decode(), mode, oid))
    batch = subprocess.run(['git', '-C', str(REPO), 'cat-file', '--batch'],
                           input=''.join(oid + '\n' for _, _, oid in inputs).encode(),
                           capture_output=True, check=True, timeout=60).stdout
    cursor = 0
    for name, mode, oid in inputs:
        end = batch.index(b'\n', cursor)
        header = batch[cursor:end].decode().split()
        assert header[:2] == [oid, 'blob'], header
        size = int(header[2])
        data = batch[end + 1:end + 1 + size]
        cursor = end + 2 + size
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if mode == '100755' else 0o644)
    assert cursor == len(batch)
    # Small explicit link graph, distinct from existing receipt content.
    (root / 'links').mkdir()
    (root / 'links/gates').symlink_to('../' + LEGACY + '/gates.txt')
    (root / 'links/directory').symlink_to('../' + LEGACY)
    (root / 'links/via-parent').symlink_to('directory/gates.txt')
    return cited


def assert_problem(root, path, reason, module=preflight, checks=None):
    report = module.preflight(root, 'HEAD', checks or list(preflight.CONTRACTS))
    assert not report['ok'], report
    assert {'path': path, 'reason': reason} in report['problems'], report
    return report


def assert_green(root, module=preflight, checks=None):
    try:
        report = module.preflight(root, 'HEAD', checks or list(preflight.CONTRACTS))
    except ValueError as error:
        raise AssertionError('valid materialization was refused') from error
    assert report['ok'], report


def assert_refusal(root, needle, module=preflight, checks=None, ref='HEAD'):
    try:
        module.preflight(root, ref, checks or list(preflight.CONTRACTS))
    except preflight.Refusal as error:
        assert needle in str(error), str(error)
        return
    except ValueError as error:
        # Mutant modules carry their own concrete Refusal type.
        assert type(error).__name__ == 'Refusal' and needle in str(error), error
        return
    raise AssertionError('expected specific preflight refusal: ' + needle)


@contextlib.contextmanager
def mutant(scratch, name, old, new):
    source = SCRIPT.read_text()
    assert source.count(old) == 1, ('mutant anchor', name, old)
    path = scratch / (name + '.py')
    path.write_text(source.replace(old, new))
    spec = importlib.util.spec_from_file_location('mutant_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        path.unlink()


def killed(witness, module):
    try:
        witness(module)
    except AssertionError:
        return
    raise AssertionError('coherent preflight mutant survived witness')


def main():
    if not __debug__:
        raise SystemExit('integration witness requires Python assertions')
    original_source = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    events = []
    consumer_witnesses = []
    with tempfile.TemporaryDirectory(prefix='memra-sparse-preflight-') as temp:
        scratch = Path(temp)
        root = scratch / 'repo'
        root.mkdir()
        git(root, 'init', '-q')
        git(root, 'config', 'user.name', 'Sparse Input Test')
        git(root, 'config', 'user.email', 'sparse@example.invalid')
        git(root, 'config', 'gc.auto', '0')
        git(root, 'config', 'maintenance.auto', 'false')
        cited = populate(root)
        git(root, 'add', '.')
        git(root, 'commit', '-q', '-m', 'Exact original input closure')
        pinned = git(root, 'rev-parse', 'HEAD')
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        cli = command(root, sys.executable, str(SCRIPT), '--root', str(root), '--ref', pinned)
        assert json.loads(cli.stdout)['ok'], cli
        consumer(root, 'tools/update-perf-board.py', '--check')
        consumer(root, 'tools/check-support-states.py', '--root', str(root))
        consumer(root, 'tools/test_public_boundary.py',
                 'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo')
        events.append('original-consumers-and-clean-preflight-pass')

        # Real sparse omission, not a malformed board or a synthetic reader.
        git(root, 'sparse-checkout', 'init', '--cone')
        git(root, 'sparse-checkout', 'set', 'tools', 'docs', 'crates', 'links')
        failed = consumer(root, 'tools/update-perf-board.py', '--check', success=False)
        assert failed.returncode != 0 and 'current-board.json' in failed.stderr, failed
        consumer_witnesses.append({'phase': 'board-missing', 'rc': failed.returncode,
                                   'stdout': failed.stdout, 'stderr': failed.stderr})
        report = assert_problem(root, BOARD, 'missing-materialization')
        cli = command(root, sys.executable, str(SCRIPT), '--root', str(root), '--ref', pinned,
                      '--check', 'perf-board', success=False)
        payload = json.loads(cli.stdout)
        assert cli.returncode == 1 and {'path': BOARD, 'reason': 'missing-materialization'} in payload['problems'], cli
        consumer_witnesses.append({'phase': 'board-preflight', 'diagnostic':
                                   next(p for p in report['problems'] if p['path'] == BOARD)})
        assert BOARD in report['suggested_sparse_paths']
        with mutant(scratch, 'ignore_board', "'research/tune-data/current-board.json', 'docs/MODELS.md'",
                    "'docs/MODELS.md'") as module:
            killed(lambda m: assert_problem(root, BOARD, 'missing-materialization', m), module)
        events.append('original-missing-board-to-specific-preflight-refusal')
        git(root, 'sparse-checkout', 'add', 'research/tune-data')
        result = consumer(root, 'tools/update-perf-board.py', '--check')
        consumer_witnesses.append({'phase': 'board-materialized', 'rc': result.returncode,
                                   'stdout': result.stdout, 'stderr': result.stderr})

        receipt = sorted(cited)[0]
        assert_problem(root, receipt, 'missing-materialization')
        failed = consumer(root, 'tools/check-support-states.py', '--root', str(root), success=False)
        assert failed.returncode != 0, failed
        consumer_witnesses.append({'phase': 'receipts-missing', 'rc': failed.returncode,
                                   'stdout': failed.stdout, 'stderr': failed.stderr})
        failed = consumer(root, 'tools/test_public_boundary.py',
                          'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo', success=False)
        assert failed.returncode != 0 and 'dangling' in failed.stderr, failed
        consumer_witnesses.append({'phase': 'link-target-missing', 'rc': failed.returncode,
                                   'stdout': failed.stdout, 'stderr': failed.stderr})
        git(root, 'sparse-checkout', 'add', *sorted({str(PurePosixPath(p).parent) for p in cited} | {LEGACY}))
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        result = consumer(root, 'tools/check-support-states.py', '--root', str(root))
        consumer_witnesses.append({'phase': 'receipts-materialized', 'rc': result.returncode,
                                   'stdout': result.stdout, 'stderr': result.stderr})
        result = consumer(root, 'tools/test_public_boundary.py',
                          'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo')
        consumer_witnesses.append({'phase': 'link-target-materialized', 'rc': result.returncode,
                                   'stdout': result.stdout, 'stderr': result.stderr})
        assert git(root, 'diff', '--exit-code') == ''
        assert git(root, 'rev-parse', 'HEAD') == pinned
        events.append('exact-Git-materialization-to-original-consumers-pass')

        # Every mutant runs against the same valid real repository and the same
        # assertion witness as the real linter, never against a broken setup.
        board = root / BOARD
        original = board.read_bytes()
        board.write_bytes(original + b' ')
        assert_problem(root, BOARD, 'blob-mismatch')
        with mutant(scratch, 'ignore_hash', 'if digest.hexdigest() != oid:', 'if False:') as module:
            killed(lambda m: assert_problem(root, BOARD, 'blob-mismatch', m), module)
        board.write_bytes(original)
        board.chmod(0o755)
        assert_problem(root, BOARD, 'executable-mode-mismatch')
        with mutant(scratch, 'ignore_mode', "if (actual.st_mode & 0o111) != (0o111 if mode == '100755' else 0):",
                    'if False:') as module:
            killed(lambda m: assert_problem(root, BOARD, 'executable-mode-mismatch', m), module)
        board.chmod(0o644)
        hook = root / 'tools/hooks/pre-push'
        hook.chmod(0o644)
        assert_problem(root, 'tools/hooks/pre-push', 'executable-mode-mismatch')
        hook.chmod(0o755)
        board.unlink()
        os.mkfifo(board)
        assert_problem(root, BOARD, 'type-mismatch')
        # Avoid an unsafe FIFO open in the mutant: it keeps NONBLOCK and fstat,
        # and its deliberate fault is misreporting type errors as success.
        with mutant(scratch, 'ignore_type', "if problem:\n                problems.append",
                    "if problem and problem != 'type-mismatch':\n                problems.append") as module:
            killed(lambda m: assert_problem(root, BOARD, 'type-mismatch', m), module)
        board.unlink()
        board.write_bytes(original)
        board.chmod(0)
        assert_problem(root, BOARD, 'unreadable')
        with mutant(scratch, 'ignore_unreadable', "if problem:\n                problems.append",
                    "if problem and problem != 'unreadable':\n                problems.append") as module:
            killed(lambda m: assert_problem(root, BOARD, 'unreadable', m), module)
        board.chmod(0o644)
        events.append('regular-byte-mode-executable-type-unreadable-mutants-refused')

        link = root / 'links/gates'
        link.unlink()
        link.symlink_to('../' + BOARD)
        assert_problem(root, 'links/gates', 'link-target-mismatch')
        with mutant(scratch, 'ignore_link', 'if target != tree.blob(name):', 'if False:') as module:
            killed(lambda m: assert_problem(root, 'links/gates', 'link-target-mismatch', m), module)
        link.unlink()
        link.symlink_to('../' + LEGACY + '/gates.txt')
        # An on-disk parent link is never followed, even to another owned dir.
        parent = root / 'research/tune-data'
        hidden = scratch / 'hidden-board'
        parent.rename(hidden)
        parent.symlink_to(hidden, target_is_directory=True)
        assert_problem(root, BOARD, 'unsafe-parent-or-type')
        with mutant(scratch, 'follow_parent',
                    'child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)',
                    'child = os.open(part, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent)') as module:
            killed(lambda m: assert_problem(root, BOARD, 'unsafe-parent-or-type', m), module)
        parent.unlink()
        hidden.rename(parent)
        alias = scratch / 'root-alias'
        alias.symlink_to(root, target_is_directory=True)
        try:
            preflight.preflight(alias, pinned, list(preflight.CONTRACTS))
        except OSError:
            pass
        else:
            raise AssertionError('root symlink was followed')
        alias.unlink()

        # Pinned Git link failures cannot be made safe by disk materialization.
        for name, target, needle in (
                ('links/cycle', 'cycle', 'link cycle/depth'),
                ('links/outside', '../../outside', 'link escapes'),
                ('links/absolute', '/absent-owned-test-target', 'absolute/invalid'),
                ('links/unknown', '../not-tracked', 'target absent'),
                ('links/file-parent', 'gates/child', 'not a directory')):
            path = root / name
            path.symlink_to(target)
            git(root, 'add', name)
            git(root, 'commit', '-q', '-m', 'Plant invalid pinned link')
            assert_refusal(root, needle)
            with mutant(scratch, 'ignore_link_closure',
                        "required = link_closure(tree, modeled_inputs(tree, checks))",
                        "required = modeled_inputs(tree, checks)") as module:
                killed(lambda m: assert_refusal(root, needle, m), module)
            git(root, 'reset', '-q', '--hard', pinned)
        events.append('pinned-link-parent-cycle-outside-absolute-unknown-mutants-refused')

        required = root / receipt
        data = required.read_bytes()
        required.unlink()
        assert_problem(root, receipt, 'missing-materialization')
        # Drop one required leaf after full copy enumeration, so the fault cannot
        # hide behind the broader receipt-root transport inventory.
        with mutant(scratch, 'ignore_required', 'return required\n',
                    'return required - {' + repr(receipt) + '}\n') as module:
            killed(lambda m: assert_problem(root, receipt, 'missing-materialization', m), module)
        required.write_bytes(data)
        git(root, 'reset', '-q', '--hard', pinned)
        # Optional sidecars missing from Git are not mandatory; a tracked one is.
        optional_parent = root / 'research/owned-optional'
        optional_parent.mkdir()
        git(root, 'sparse-checkout', 'add', 'research/owned-optional')
        (optional_parent / 'gates.txt').write_text('Config=passed\n')
        metadata = root / 'docs/support-records.toml'
        original_metadata = metadata.read_bytes()
        metadata.write_text('[[record]]\n[record.gates]\nConfig="passed"\n'
                            '[record.evidence]\nConfig=["research/owned-optional/gates.txt"]\n')
        git(root, 'add', 'docs/support-records.toml', 'research/owned-optional')
        git(root, 'commit', '-q', '-m', 'Optional sidecars genuinely absent')
        assert preflight.preflight(root, 'HEAD', ['support-records'])['ok']
        with mutant(scratch, 'require_optional', 'for n in optional if n in tree.entries',
                    'for n in optional') as module:
            killed(lambda m: assert_green(root, m, ['support-records']), module)
        sidecar = optional_parent / 'artifact.lock'
        sidecar.write_text('family=owned\n')
        git(root, 'add', 'research/owned-optional/artifact.lock')
        git(root, 'commit', '-q', '-m', 'Tracked optional sidecar')
        sidecar.unlink()
        assert_problem(root, 'research/owned-optional/artifact.lock', 'missing-materialization',
                       checks=['support-records'])
        git(root, 'reset', '-q', '--hard', pinned)
        assert not optional_parent.exists()
        assert metadata.read_bytes() == original_metadata

        reader = root / 'tools/update-perf-board.py'
        reader.write_bytes(reader.read_bytes() + b'\n# unknown read shape\n')
        git(root, 'add', 'tools/update-perf-board.py')
        git(root, 'commit', '-q', '-m', 'Unknown reader source')
        assert_refusal(root, 'unknown reader source/version', checks=['perf-board'])
        with mutant(scratch, 'ignore_reader',
                    'if hashlib.sha256(tree.blob(source)).hexdigest() != READER_PINS[source]:',
                    'if False:') as module:
            killed(lambda m: assert_refusal(root, 'unknown reader source/version', m, ['perf-board']), module)
        git(root, 'reset', '-q', '--hard', pinned)
        unknown = root / 'docs/owned-unknown-copy-input'
        unknown.write_text('unmodeled copy content\n')
        assert_problem(root, 'docs/owned-unknown-copy-input', 'unmodeled-copy-input',
                       checks=['support-records'])
        with mutant(scratch, 'ignore_copy_extras', 'problems = copy_extras(root_fd, tree)',
                    'problems = []') as module:
            killed(lambda m: assert_problem(root, 'docs/owned-unknown-copy-input',
                                            'unmodeled-copy-input', m, ['support-records']), module)
        unknown.unlink()
        assert_refusal(root, 'unknown reader contract', checks=['unknown'])
        assert_refusal(root, 'Git reader failed', checks=['perf-board'], ref='invalid-owned-integration-ref')
        cli = command(root, sys.executable, str(SCRIPT), '--root', str(root), '--check', 'unknown', success=False)
        assert cli.returncode == 1 and 'unknown reader contract' in json.loads(cli.stdout)['refusal'], cli
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        assert git(root, 'diff', '--exit-code') == ''
        events.append('required-optional-reader-and-Git-refusal-controls-pass')
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == original_source
    print(json.dumps({'ok': True, 'integration_events': events,
                      'consumer_witnesses': consumer_witnesses,
                      'source_sha256': original_source, 'scratch_retired': True}, indent=2))


if __name__ == '__main__':
    main()
