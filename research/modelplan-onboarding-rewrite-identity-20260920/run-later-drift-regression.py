#!/usr/bin/env python3
"""Run the same CPU re-entry cases against frozen old entry logic and current logic.

The old snapshot body is extracted from cbafa2eb. A signature-only adapter adds an
ignored validator argument, exposing the old missing-boundary behavior to the same
tests. The validator and fixture code are current in BOTH arms. Library tests inject
real file metadata into the production inventory comparator; they are not /proc or
CUDA tests. Each old expected-refusal case must fail, and every current case must pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).with_name('later-drift-20260920')
BEFORE = 'cbafa2ebba8823e75b65d9f976649b64a2af12a6'
SNAPSHOT = Path('crates/memra-engine/src/plan_backend/execution_snapshot.rs')
RUNTIME = Path('crates/memra-engine/src/plan_backend/runtime_identity.rs')
TESTS = Path('crates/memra-engine/src/plan_backend/execution_snapshot/reentry_tests.rs')
CASES = ('retained_qualified_scope_must_refuse_later_environment_drift',
         'retained_qualified_scope_must_refuse_later_library_inventory_drift')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = subprocess.check_output(['git', 'show', f'{BEFORE}:{SNAPSHOT}'], cwd=ROOT)
    old = frozen.decode().split('\n#[cfg(test)]\nmod tests {', 1)[0]
    old = old.replace("        _program: &'a T,\n", "        _program: &'a T,\n        _unvalidated_boundary: impl FnOnce() -> Result<(), String>,\n", 1)
    assert '_unvalidated_boundary' in old
    old += f'\n#[cfg(test)]\n#[path = "{ROOT / TESTS}"]\nmod reentry_tests;\n'
    evidence = {
        'frozen_snapshot_ref': BEFORE, 'frozen_snapshot_sha256': sha(frozen),
        'adapter': 'ignored validator argument only; old entry body unchanged; old in-file tests omitted',
        'adapted_snapshot_sha256': sha(old.encode()),
        'current_files': {str(path): sha((ROOT / path).read_bytes()) for path in (SNAPSHOT, RUNTIME, TESTS)},
        'limits': 'CPU synthetic qualification; portable file-inventory injection, not Linux mappings or CUDA',
        'cases': [],
    }
    with tempfile.TemporaryDirectory(prefix='rewrite-reentry-comparison-', dir=ROOT / 'target') as temporary:
        project = Path(temporary)
        (project / 'src').mkdir()
        (project / 'Cargo.toml').write_text(f'''[package]
name = "memra-reentry-comparison"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
memra-gguf = {{ path = "{ROOT / 'crates/memra-gguf'}" }}
sha2 = "0.10"
libc = "0.2"
''')
        (project / 'src/old_snapshot.rs').write_text(old)
        for arm, snapshot, expected in [('before', project / 'src/old_snapshot.rs', 101), ('after', ROOT / SNAPSHOT, 0)]:
            (project / 'src/lib.rs').write_text(f'''#![allow(dead_code)]
#[path = "{ROOT / RUNTIME}"]
mod runtime_identity;
#[path = "{snapshot}"]
mod execution_snapshot;
''')
            for case in CASES:
                command = ['cargo', 'test', '--offline', '--manifest-path', str(project / 'Cargo.toml'),
                           f'execution_snapshot::reentry_tests::{case}', '--', '--exact', '--test-threads=1', '--nocapture']
                log = out / f'{arm}-{case}.log'
                with log.open('x') as stream:
                    result = subprocess.run(command, cwd=project, stdout=stream, stderr=subprocess.STDOUT,
                                            env={**os.environ, 'CARGO_TARGET_DIR': str(project / 'target')})
                contents = log.read_bytes()
                if result.returncode != expected or b'test result: ' not in contents:
                    raise RuntimeError(f'{arm}/{case}: unexpected result {result.returncode}; see {log}')
                failure = (b'retained qualified token path admitted after environment drift'
                           if 'environment' in case else b'library drift admitted retained output')
                if arm == 'before' and failure not in contents:
                    raise RuntimeError(f'{case}: missing exact expected-refusal assertion')
                if arm == 'after' and (b'1 passed; 0 failed; 0 ignored;' not in contents or b'REENTRY_' not in contents):
                    raise RuntimeError(f'{case}: missing non-vacuous passing regression')
                evidence['cases'].append({'arm': arm, 'case': case, 'returncode': result.returncode,
                                          'expected_returncode': expected, 'log_sha256': sha(contents)})
                print(f'{arm}: {case}: exit={result.returncode} (expected {expected})', flush=True)
    if evidence['current_files'] != {str(path): sha((ROOT / path).read_bytes()) for path in (SNAPSHOT, RUNTIME, TESTS)}:
        raise RuntimeError('source changed during the before/after regression')
    with (out / 'comparison.json').open('x') as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
