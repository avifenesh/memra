#!/usr/bin/env python3
"""Run the actual CPU identity module without linking the CUDA engine; no GPU qualification."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
(ROOT / 'target').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='rewrite-identity-cpu-', dir=ROOT / 'target') as scratch:
    project = Path(scratch)
    (project / 'src').mkdir()
    (project / 'Cargo.toml').write_text(f'''[package]
name = "memra-rewrite-identity-host-tests"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
memra-gguf = {{ path = "{ROOT / 'crates/memra-gguf'}" }}
sha2 = "0.10"
libc = "0.2"
''')
    (project / 'src/lib.rs').write_text(f'''#![allow(dead_code)]
#[path = "{ROOT / 'crates/memra-engine/src/plan_backend/runtime_identity.rs'}"]
mod runtime_identity;
''')
    result = subprocess.run(['cargo', 'test', '--manifest-path', str(project / 'Cargo.toml')],
                            cwd=project, env={**os.environ, 'CARGO_TARGET_DIR': str(project / 'target')})
raise SystemExit(result.returncode)
