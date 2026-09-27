#!/usr/bin/env python3
"""Compile the actual C header/consumer against the production Rust reader; no CUDA."""
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/memra-gguf/tests/ffi/retained_reader.c"
HEADER = ROOT / "crates/memra-gguf/include/memra_scoped_disk_v1.h"
TARGET = ROOT / "target"
TARGET.mkdir(exist_ok=True)
for path in (SOURCE, HEADER, ROOT / "crates/memra-gguf/src/bound_disk/ffi.rs"):
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(ROOT), flush=True)
with tempfile.TemporaryDirectory(prefix="scoped-disk-ffi-", dir=TARGET) as directory:
    project = Path(directory)
    (project / "src").mkdir()
    shutil.copy2(ROOT / "Cargo.lock", project / "Cargo.lock")
    (project / "Cargo.toml").write_text(f'''[package]
name="memra-scoped-disk-ffi-control"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
memra-gguf={{path="{ROOT / 'crates/memra-gguf'}"}}
''')
    (project / "build.rs").write_text(f'''
use std::process::Command;
fn main() {{
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let object = out.join("reader.o");
    assert!(Command::new("cc").args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-c", "{SOURCE}", "-I", "{HEADER.parent}", "-o"]).arg(&object).status().unwrap().success());
    assert!(Command::new("ar").arg("crs").arg(out.join("libreader.a")).arg(object).status().unwrap().success());
    println!("cargo:rustc-link-search=native={{}}", out.display());
    println!("cargo:rustc-link-lib=static=reader");
}}
''')
    (project / "src/main.rs").write_text('''
use memra_gguf::{GgufFile, bound_source::BoundTensorSource, source::GgufSource,
    tensor_contract::TensorId, bound_disk::{BoundReadMode, ffi::ScopedDiskReaderV1}};
use std::ffi::c_void;
unsafe extern "C" {
    fn memra_ffi_test_capture(reader: *const ScopedDiskReaderV1) -> *mut c_void;
    fn memra_ffi_test_read_and_release(reader: *mut c_void, out: *mut u8, len: usize) -> i32;
}
fn main() {
    let root = std::path::PathBuf::from(std::env::args_os().nth(1).unwrap());
    let path = root.join("model.gguf");
    memra_gguf::micro_gguf::write_glm_dsa_micro(&path, 541).unwrap();
    let (foreign, expected) = {
        let file=GgufFile::open(&path).unwrap();
        let source=GgufSource(&file);
        let bound=BoundTensorSource::compile(&source).unwrap();
        let view=bound.disk(&TensorId::TokenEmbedding).unwrap().unwrap();
        let expected=view.bytes().to_vec();
        let handle=view.reader(BoundReadMode::Buffered).unwrap().into_ffi_handle();
        let foreign=unsafe { memra_ffi_test_capture(&handle.as_abi()) };
        assert!(!foreign.is_null());
        (foreign,expected)
    };
    // C is now the only reader owner: file, source, bundle, view and Rust handle are gone.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path,b"replacement file").unwrap();
    let mut actual=vec![0;expected.len()];
    assert_eq!(unsafe { memra_ffi_test_read_and_release(foreign,actual.as_mut_ptr(),actual.len()) },0);
    assert_eq!(actual,expected);
    println!("C ABI PASS: retained bytes after all Rust owners dropped and path replaced; range/overflow refusals; balanced release");
}
''')
    result = subprocess.run(
        ["cargo", "run", "--offline", "--manifest-path", str(project / "Cargo.toml"), "--", str(project)],
        env={**os.environ, "CARGO_TARGET_DIR": str(TARGET / "scoped-disk-ffi-build")},
    )
raise SystemExit(result.returncode)
