#!/usr/bin/env python3
"""Host CPU controls for the actual worker usage publisher and bounded event channel.

Extracts the production Event/channel/publisher and its two actual Rust tests.
Only the unrelated SpecUsage/EngineError payload types are inert stand-ins. This
does not construct a model/session or qualify native admission or worker callers.
The negative control removes the publisher send from an isolated copy; the exact
delivery test must then fail. No repository source is modified by this runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'crates/memra-server/src/worker.rs'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_function(source, name):
    start = source.index(f'    #[test]\n    fn {name}(')
    opening = source.index('{', start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == '{') - (source[end] == '}')
        if depth == 0:
            return source[start:end + 1]
    raise RuntimeError('unterminated test function')


def run(out):
    out = out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise RuntimeError('CPU evidence must be outside the checkout')
    out.mkdir(parents=True, exist_ok=False)
    source = SOURCE.read_text()
    start = source.index('#[derive(Debug, Clone)]\npub enum Event {')
    end = source.index('/// THE ERROR TAXONOMY', start)
    channel = source[start:end]
    tests = '\n'.join(test_function(source, name) for name in (
        'admitted_prompt_usage_publishes_actual_counts_once',
        'admitted_prompt_usage_preserves_closed_receiver_behavior',
    ))
    prefix = ('use std::sync::Arc;\n'
              '#[derive(Debug, Clone)] pub struct SpecUsage;\n'
              '#[derive(Debug, Clone)] pub struct EngineError { pub message: String }\n')
    code = prefix + channel + '\n#[cfg(test)] mod tests {\n' + tests + '\n}\n'
    send = 'let _ = tx.send(Event::PromptUsage { n_prompt, n_cached });'
    if code.count(send) != 1:
        raise RuntimeError('publisher send not unique')
    results = {}
    for mode, program in (
        ('actual', code),
        ('no-send-negative', code.replace(send, 'let _ = (tx, n_prompt, n_cached);')),
    ):
        crate = out / mode
        (crate / 'src').mkdir(parents=True)
        (crate / 'src/lib.rs').write_text(program)
        (crate / 'Cargo.toml').write_text(
            '[package]\nname = "worker-publisher-control"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[dependencies]\ntokio = { version = "=1.52.3", features = ["sync"] }\n')
        command = ['cargo', 'test', '--offline', '--manifest-path', str(crate / 'Cargo.toml'),
                   '--lib', '--', '--nocapture']
        env = {**os.environ, 'CARGO_TARGET_DIR': str(crate / 'target')}
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (out / f'{mode}.log').write_bytes(result.stdout)
        text = result.stdout.decode(errors='replace')
        expected = (result.returncode == 0 and
                    re.search(r'^test result: ok\. 2 passed; 0 failed;', text, re.M)) if mode == 'actual' else (
                        result.returncode != 0 and
                        re.search(r'^test result: FAILED\. 1 passed; 1 failed;', text, re.M) and
                        'tests::admitted_prompt_usage_publishes_actual_counts_once' in text)
        if not expected:
            raise RuntimeError(f'{mode} did not produce the required non-vacuous outcome; see {out}')
        results[mode] = {'returncode': result.returncode, 'command': command,
                         'log_sha256': digest(result.stdout), 'extracted_program_sha256': digest(program.encode())}
    report = {'status': 'passed', 'scope': 'CPU publisher/channel only; native callers unqualified',
              'worker_source_sha256': digest(source.encode()), 'results': results}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
