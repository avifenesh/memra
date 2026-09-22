#!/usr/bin/env python3
"""Host CPU checks of actual error redaction, channel, and scoped diagnostic witness.

Extracts production Event/channel/EngineError code and includes the exact test-only
diagnostics module. Only the unrelated SpecUsage payload is a stand-in. This does
not execute a native model or prove a worker caller's GPU qualification. Removing
the producer hook in an isolated copy must fail the paired-cause witness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'crates/memra-server/src/worker.rs'
FIXTURE = ROOT / 'crates/memra-server/src/worker/rewrite_native_tests.rs'
DIAGNOSTICS = FIXTURE.parent / 'rewrite_native_tests/diagnostics.rs'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def function(source, marker):
    start = source.index(marker)
    opening = source.index('{', start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise RuntimeError('unterminated function')


def run(out):
    out = out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise RuntimeError('CPU evidence must be outside source')
    out.mkdir(parents=True, exist_ok=False)
    source, fixture, diagnostics = WORKER.read_text(), FIXTURE.read_text(), DIAGNOSTICS.read_text()
    start = source.index('#[derive(Debug, Clone)]\npub enum Event {')
    end = source.index('/// Per-request spec-decode acceptance summary', start)
    production = source[start:end] + '\n' + '\n'.join(function(source, marker) for marker in (
        'pub(crate) fn engine_client_message(', 'fn is_cuda_oom(',
    ))
    helpers = '\n'.join(function(fixture, marker) for marker in ('fn require(', 'fn events('))
    hook = 'rewrite_native_tests::diagnostics::record_engine_error(class, &message);'
    if production.count(hook) != 1:
        raise RuntimeError('producer hook not unique')
    results = {}
    for mode in ('actual', 'no-hook-negative'):
        crate = out / mode
        (crate / 'src').mkdir(parents=True)
        diag_path = crate / 'src/diagnostics.rs'
        diag_path.write_text(diagnostics)
        code = ('#![allow(dead_code)]\npub mod worker {\nuse std::sync::Arc;\n'
                '#[derive(Debug, Clone)] pub struct SpecUsage;\n' + production +
                '\n#[cfg(test)] mod rewrite_native_tests {\nuse super::*;\n'
                'type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;\n' + helpers +
                f'\n#[path = {json.dumps(str(diag_path))}] pub(super) mod diagnostics;\n' + '}}\n')
        if mode == 'no-hook-negative':
            code = code.replace(hook, 'let _ = (class, &message);')
        (crate / 'src/lib.rs').write_text(code)
        (crate / 'Cargo.toml').write_text(
            '[package]\nname = "worker-diagnostic-control"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[dependencies]\ntokio = { version = "=1.52.3", features = ["sync"] }\n')
        command = ['cargo', 'test', '--offline', '--manifest-path', str(crate / 'Cargo.toml'),
                   '--lib', '--', '--nocapture']
        result = subprocess.run(command, env={**os.environ, 'CARGO_TARGET_DIR': str(crate / 'target')},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (out / f'{mode}.log').write_bytes(result.stdout)
        text = result.stdout.decode(errors='replace')
        expected = (result.returncode == 0 and re.search(
            r'^test result: ok\. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;', text, re.M)) if mode == 'actual' else (
                result.returncode != 0 and re.search(r'^test result: FAILED\. 3 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out;', text, re.M)
                and 'actual_producer_preserves_public_redaction_and_exact_cause' in text)
        if not expected:
            raise RuntimeError(f'{mode} did not give the expected non-vacuous result; see {out}')
        results[mode] = {'returncode': result.returncode, 'command': command,
                         'log_sha256': sha(result.stdout), 'extracted_program_sha256': sha(code.encode())}
    originals = {WORKER: source, FIXTURE: fixture, DIAGNOSTICS: diagnostics}
    if any(path.read_text() != text for path, text in originals.items()):
        raise RuntimeError('source changed during controls')
    report = {'status': 'passed', 'scope': 'CPU diagnostic/redaction/channel only; native unqualified',
              'source_sha256': {str(path.relative_to(ROOT)): sha(text.encode()) for path, text in originals.items()},
              'results': results}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
