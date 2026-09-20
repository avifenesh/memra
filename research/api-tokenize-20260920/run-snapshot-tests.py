#!/usr/bin/env python3
"""CPU-only tokenizer replacement reproduction and production snapshot tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--red', action='store_true', help='reproduce the old independent-open contract failure')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
source = root / 'crates/memra-server/src/worker/tokenizers.rs'
fixture = source.with_suffix('') / 'fixture.rs'
target = root / 'target/tokenizer-snapshot-host'
target.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='snapshot-src-', dir=root / 'target') as temporary:
    project = Path(temporary)
    (project / 'Cargo.toml').write_text(f'''[package]
name = "memra-tokenizer-snapshot-host"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
memra-tokenizer = {{ path = "{root / 'crates/memra-tokenizer'}" }}
[lib]
path = "lib.rs"
''')
    if args.red:
        code = f'''#[path = "{fixture}"] mod fixture;
#[test]
fn independent_openings_must_have_one_prompt_interpretation() {{
    let fixture = fixture::Fixture::new();
    let http = memra_tokenizer::Tokenizer::from_hf_dir(&fixture.0).unwrap();
    fixture.replace_config(false);
    let worker = memra_tokenizer::Tokenizer::from_hf_dir(&fixture.0).unwrap();
    assert_eq!(http.encode("hello", true), worker.encode("hello", true),
               "endpoint and worker token ids must match");
}}
'''
    else:
        code = f'#![allow(dead_code)]\n#[path = "{source}"] mod tokenizers;\n'
    (project / 'lib.rs').write_text(code)
    (out / 'harness.rs').write_text(code)
    command = ['cargo', 'test', '--offline', '--manifest-path', str(project / 'Cargo.toml'), '--lib', '--', '--nocapture']
    with (out / 'tests.log').open('wb') as log:
        result = subprocess.run(command, cwd=root,
                                env={**os.environ, 'CARGO_TARGET_DIR': str(target)},
                                stdout=log, stderr=subprocess.STDOUT)
    text = (out / 'tests.log').read_text()
    passed = (result.returncode != 0 and 'endpoint and worker token ids must match' in text
              and '0 passed; 1 failed;' in text) if args.red else (
                  result.returncode == 0 and '3 passed; 0 failed; 0 ignored;' in text)
    files = [source, source.with_suffix('') / 'tests.rs', fixture]
    metadata = {'mode': 'independent-open-red' if args.red else 'production-snapshot-green',
                'command': command, 'test_returncode': result.returncode,
                'expected_outcome_observed': passed,
                'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'log_sha256': hashlib.sha256((out / 'tests.log').read_bytes()).hexdigest(),
                'scope': 'CPU tokenizer semantics and snapshot sharing only; no CUDA boot or model qualification'}
    (out / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))
    raise SystemExit(0 if passed else 1)
