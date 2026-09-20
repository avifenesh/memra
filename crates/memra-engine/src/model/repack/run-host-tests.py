#!/usr/bin/env python3
"""Compile/test the production native repack loader against memra-gguf, without CUDA.

Run: python3 crates/memra-engine/src/model/repack/run-host-tests.py
Uses the selected Rust toolchain (MSRV 1.97); RUSTUP_TOOLCHAIN can select an installed version.
"""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[5]
(ROOT / "target").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="native-repack-host-", dir=ROOT / "target") as scratch:
    project = Path(scratch)
    (project / "src").mkdir()
    (project / "Cargo.toml").write_text(f'''[package]
name = "memra-native-repack-host-tests"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
memra-gguf = {{ path = "{ROOT / 'crates/memra-gguf'}" }}
memmap2 = "0.9"
libc = "0.2"
''')
    (project / "src/lib.rs").write_text(f'''#![allow(dead_code)]
#[path = "{ROOT / 'crates/memra-engine/src/model/repack.rs'}"]
mod repack;
''')
    result = subprocess.run(
        ["cargo", "test", "--offline", "--manifest-path", str(project / "Cargo.toml")],
        cwd=project,
        env={**os.environ, "CARGO_TARGET_DIR": str(project / "target")},
    )
raise SystemExit(result.returncode)
