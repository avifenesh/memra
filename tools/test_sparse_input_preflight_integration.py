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
    (root / 'links/actual').mkdir()
    (root / 'links/actual/data.txt').write_text('owned directory link target\n')
    (root / 'links/alias').symlink_to('actual')
    (root / 'links/reused-parent').symlink_to('alias/../alias/data.txt')
    (root / 'links/trailing-separators').symlink_to('actual//')
    (root / 'links/hop-limit-good').symlink_to('alias/../' * 39 + 'actual/data.txt')
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


def assert_cone_suggestions(root, module=preflight):
    report = module.preflight(root, 'HEAD', list(preflight.CONTRACTS))
    assert 'research/tune-data' in report['suggested_sparse_paths'], report
    assert 'research' not in report['suggested_sparse_paths'], report
    for addition in report['suggested_sparse_paths']:
        assert git(root, 'cat-file', '-t', 'HEAD:' + addition) == 'tree', addition
        assert not any(addition.startswith(other + '/') for other in report['suggested_sparse_paths']
                       if other != addition), report


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
        consumer(root, 'tools/test_public_boundary.py',
                 'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo')
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        with mutant(scratch, 'false_cycle_on_reuse',
                    "prefix = resolve(prefix[:-1] + target.split('/'), active + (name,), budget).split('/')",
                    "return resolve(prefix[:-1] + target.split('/') + parts[offset + 1:], active + (name,), budget)") as module:
            killed(lambda m: assert_green(root, m), module)
        with mutant(scratch, 'reject_valid_separators', "if part in ('.', ''):",
                    "if part == '.':") as module:
            killed(lambda m: assert_green(root, m), module)
        with mutant(scratch, 'reject_40th_hop', "target = resolve(name.split('/'), (), [40])",
                    "target = resolve(name.split('/'), (), [39])") as module:
            killed(lambda m: assert_green(root, m), module)
        cli = command(root, sys.executable, str(SCRIPT), '--root', str(root), '--ref', pinned)
        assert json.loads(cli.stdout)['ok'], cli
        consumer(root, 'tools/update-perf-board.py', '--check')
        consumer(root, 'tools/check-support-states.py', '--root', str(root))
        consumer(root, 'tools/test_public_boundary.py',
                 'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo')
        events.append('original-consumers-and-clean-preflight-pass')

        # Real sparse omission, not a malformed board or a synthetic reader.
        git(root, 'sparse-checkout', 'init', '--cone')
        git(root, 'sparse-checkout', 'set', 'tools')
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
        assert report['problem_count'] > 200 and report['problems_truncated'], report
        receipt = sorted(cited)[0]
        assert {'path': receipt, 'reason': 'missing-materialization'} in report['problems'], report
        assert_cone_suggestions(root)
        with mutant(scratch, 'leaf_suggestions', 'return sorted(minimal)',
                    "return sorted({p['path'] for p in problems if p['reason'] == 'missing-materialization'})") as module:
            killed(lambda m: assert_cone_suggestions(root, m), module)
        with mutant(scratch, 'redundant_suggestions', 'return sorted(minimal)',
                    'return sorted(parents)') as module:
            killed(lambda m: assert_cone_suggestions(root, m), module)
        with mutant(scratch, 'starve_direct_inputs',
                    "problems.sort(key=lambda p: (p['path'] not in tree.direct_inputs, p['path']))",
                    "problems.sort(key=lambda p: p['path'])") as module:
            killed(lambda m: assert_problem(root, BOARD, 'missing-materialization', m), module)
        with mutant(scratch, 'ignore_board', "'research/tune-data/current-board.json', 'docs/MODELS.md'",
                    "'docs/MODELS.md'") as module:
            killed(lambda m: assert_problem(root, BOARD, 'missing-materialization', m), module)
        events.append('original-missing-board-to-specific-preflight-refusal')
        git(root, 'sparse-checkout', 'add', *report['suggested_sparse_paths'])
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        result = consumer(root, 'tools/update-perf-board.py', '--check')
        consumer_witnesses.append({'phase': 'board-cone-suggestions-materialized', 'rc': result.returncode,
                                   'stdout': result.stdout, 'stderr': result.stderr})
        events.append('tools-only-cone-suggestions-materialize-complete-closure')

        # Isolate the original receipt/link failure after restoring source/docs.
        git(root, 'sparse-checkout', 'set', 'tools', 'docs', 'crates', 'links', 'research/tune-data')
        report = assert_problem(root, receipt, 'missing-materialization')
        failed = consumer(root, 'tools/check-support-states.py', '--root', str(root), success=False)
        assert failed.returncode != 0 and receipt in failed.stderr and 'evidence' in failed.stderr, failed
        consumer_witnesses.append({'phase': 'receipts-missing', 'rc': failed.returncode,
                                   'stdout': failed.stdout, 'stderr': failed.stderr})
        # Keep tracked link sources present while their target roots stay sparse.
        git(root, 'sparse-checkout', 'add', 'links')
        failed = consumer(root, 'tools/test_public_boundary.py',
                          'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo', success=False)
        assert failed.returncode != 0 and 'dangling' in failed.stderr, failed
        consumer_witnesses.append({'phase': 'link-target-missing', 'rc': failed.returncode,
                                   'stdout': failed.stdout, 'stderr': failed.stderr})
        # Use the actual cone-safe suggestions, not hand-selected fixture paths.
        git(root, 'sparse-checkout', 'add', *report['suggested_sparse_paths'])
        assert preflight.preflight(root, pinned, list(preflight.CONTRACTS))['ok']
        result = consumer(root, 'tools/update-perf-board.py', '--check')
        consumer_witnesses.append({'phase': 'board-materialized', 'rc': result.returncode,
                                   'stdout': result.stdout, 'stderr': result.stderr})
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
        # The original assertion enumerates ls-files, so index-only links are
        # unknown inputs until they are included in the pinned commit.
        indexed_link = root / 'links/index-only'
        indexed_link.symlink_to('../owned-absent-index-target')
        git(root, 'add', 'links/index-only')
        failed = consumer(root, 'tools/test_public_boundary.py',
                          'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo', success=False)
        assert failed.returncode != 0 and 'index-only' in failed.stderr, failed
        assert_refusal(root, 'index link inventory differs')
        with mutant(scratch, 'ignore_index_link_inventory', '            validate_link_index(tree, links)',
                    '            pass') as module:
            killed(lambda m: assert_refusal(root, 'index link inventory differs', m), module)
        git(root, 'reset', '-q', '--hard', pinned)
        assert not indexed_link.is_symlink()
        events.append('actual-index-only-link-consumer-failure-preflight-refusal')

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
        with mutant(scratch, 'ignore_mode', "if bool(actual.st_mode & stat.S_IXUSR) != (mode == '100755'):",
                    'if False:') as module:
            killed(lambda m: assert_problem(root, BOARD, 'executable-mode-mismatch', m), module)
        board.chmod(0o644)
        hook = root / 'tools/hooks/pre-push'
        for permissions in (0o700, 0o744):
            hook.chmod(permissions)
            assert git(root, 'diff', '--exit-code') == ''
            assert_green(root)
            with mutant(scratch, 'require_all_execute_bits',
                        "if bool(actual.st_mode & stat.S_IXUSR) != (mode == '100755'):",
                        "if (actual.st_mode & 0o111) != (0o111 if mode == '100755' else 0):") as module:
                killed(lambda m: assert_green(root, m), module)
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
        events.append('regular-byte-Git-owner-execute-mode-type-unreadable-mutants-refused')

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
        # abspath must not erase an unsafe ancestor before no-follow inspection.
        noncanonical = alias / '..' / root.name
        assert_refusal(noncanonical, 'root parent traversal')
        with mutant(scratch, 'normalize_away_root_ancestor', "if '..' in Path(root).parts:",
                    'if False:') as module:
            killed(lambda m: assert_refusal(noncanonical, 'root parent traversal', m), module)
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
        excessive = root / 'links/hop-limit-bad'
        excessive.symlink_to('alias/../' * 40 + 'actual/data.txt')
        git(root, 'add', 'links/hop-limit-bad')
        git(root, 'commit', '-q', '-m', 'Pinned path with 41 link expansions')
        failed = consumer(root, 'tools/test_public_boundary.py',
                          'UnstatablePathTests.test_no_tracked_symlink_escapes_the_repo', success=False)
        assert failed.returncode != 0 and 'hop-limit-bad' in failed.stderr, failed
        assert_refusal(root, 'link cycle/depth')
        with mutant(scratch, 'ignore_total_hop_limit', "if name in active or budget[0] == 0:",
                    'if name in active:') as module:
            killed(lambda m: assert_refusal(root, 'link cycle/depth', m), module)
        git(root, 'reset', '-q', '--hard', pinned)
        events.append('finite-link-reuse-and-real-40-41-hop-boundary-pass')

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
