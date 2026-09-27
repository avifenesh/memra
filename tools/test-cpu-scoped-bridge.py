#!/usr/bin/env python3
"""Build actual Rust CPU dispatch + actual host-storage methods without linking CUDA.

Only unexercised CUDA-pinned storage types and router construction are doubles. The ordinary
Rust CPU tests run without a companion; --companion also runs the ignored actual numerical
bridge gate. This is component evidence, not GPU/model-loader qualification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--companion", type=Path)
parser.add_argument("--check-target")
parser.add_argument("--legacy-companion", type=Path)
parser.add_argument("--target-dir", type=Path)
parser.add_argument("--test-case", choices=["parity", "aligned", "mirror", "prefetch", "mirrored", "mirror-bytes", "mirror-source-generation", "mirror-target-generation", "mirror-same-fs", "mirror-cached-generation", "mirror-counter", "mirror-counter-error"], default="parity")
args = parser.parse_args()
TARGET = (args.target_dir or ROOT / "target").resolve()
TARGET.mkdir(exist_ok=True)
model = (ROOT / "crates/memra-engine/src/model.rs").read_text()
spill = (ROOT / "crates/memra-engine/src/spill_pread.rs").read_text()
engine = (ROOT / "crates/memra-engine/src/lib.rs").read_text()

def span(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]

host = span(model, "pub enum HostBuf {", "/// One layer's stacked")
layout_at = model.index("pub struct ExpertLayout {")
layout_start = model.rfind("#[derive(", 0, layout_at)
layout_end = model.index("\n}", layout_at) + 2
layout = model[layout_start:layout_end]
stores = span(model, "pub struct HostExps {", "impl HostExps {")
methods = span(model, "    #[inline]\n    pub fn macro_scale", "\n#[cfg(test)]\nmod tests")
read_at = spill.index("pub(crate) enum DiskReadSource<'a> {")
read_source = spill[read_at:spill.index("\n}", read_at) + 2]
constants = "\n".join(re.findall(r"^pub const QT_\w+: i32 = \d+;[^\n]*", engine, re.M))
assert len(constants.splitlines()) >= 14
sources = ["crates/memra-engine/src/cpu_experts.rs", "crates/memra-engine/src/model.rs",
           "crates/memra-engine/src/spill_pread.rs", "crates/memra-engine/src/lib.rs",
           "crates/memra-gguf/src/bound_disk/ffi.rs", "Cargo.lock"]
print(json.dumps({"source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  "extracted_host_code_sha256": hashlib.sha256((host+layout+stores+methods+read_source+constants).encode()).hexdigest(),
                  "cuda_executed": False, "test_doubles": ["unused CUDA-pinned type", "unused hybrid router constructor"]}, indent=2), flush=True)
with tempfile.TemporaryDirectory(prefix="cpu-scoped-bridge-", dir=TARGET) as directory:
    project = Path(directory)
    (project / "src").mkdir()
    shutil.copy2(ROOT / "Cargo.lock", project / "Cargo.lock")
    (project / "Cargo.toml").write_text(f'''[package]
name="memra-cpu-scoped-bridge-control"
version="0.0.0"
edition="2024"
[workspace]
[dependencies]
memra-gguf={{path="{ROOT / 'crates/memra-gguf'}"}}
libc="0.2"
memmap2="0.9"
''')
    (project / "src/model.rs").write_text('''use crate::{cudarc, spill_pread::DiskReadSource};
use memra_gguf::bound_disk::{BoundDiskView, ExpertDiskView};
''' + host + layout + stores + "impl HostExps {\n" + methods)
    (project / "src/spill_pread.rs").write_text('''use std::{fs::File, sync::Arc};
use memra_gguf::bound_disk::BoundDiskView;
''' + read_source)
    (project / "src/lib.rs").write_text('''#![allow(dead_code)]
mod model;
mod spill_pread;
mod cudarc { pub mod driver { pub struct PinnedHostSlice<T>(std::marker::PhantomData<T>); } }
mod hybrid {
    pub struct MoeWeights {
        pub gate_exps: crate::model::HostExps,
        pub up_exps: crate::model::HostExps,
        pub down_exps: crate::model::HostExps,
        pub active_experts: Option<Vec<bool>>,
    }
    pub struct HybridModel;
    impl HybridModel {
        #[allow(clippy::too_many_arguments)]
        pub fn moe_route_sigmoid_host_public(_: &[f32], _: usize, _: usize, _: usize,
            _: Option<&[f32]>, _: f32, _: bool, _: Option<&[bool]>) -> Result<(Vec<u32>, Vec<f32>), String> {
            panic!("router is outside the scoped CPU bridge control")
        }
    }
}
''' + constants + f'\n#[path="{ROOT / "crates/memra-engine/src/cpu_experts.rs"}"]\nmod cpu_experts;\n')
    env = {**os.environ, "CARGO_TARGET_DIR": str(TARGET / "cpu-scoped-bridge-build"), "CARGO_BUILD_JOBS": "4"}
    base = ["cargo", "check" if args.check_target else "test", "--offline", "--manifest-path", str(project / "Cargo.toml"), "--lib"]
    if args.check_target:
        result = subprocess.run([*base, "--tests", "--target", args.check_target], env=env)
    else:
        if args.companion and args.legacy_companion:
            raise SystemExit("choose one companion kind per process")
        companion = args.companion or args.legacy_companion
        if companion:
            env["MEMRA_CPU_EXPERT_LIB"] = str(companion.resolve(strict=True))
        result = subprocess.run([*base, "--", "--test-threads=1"], env=env)
        if result.returncode == 0 and companion:
            cases = {"parity": "scoped_bridge_native_token_rows_and_opened_inode_parity",
                     "aligned": "scoped_bridge_aligned_direct_token_rows_parity",
                     "mirror": "scoped_bridge_aligned_mirror_request_refuses",
                     "prefetch": "scoped_bridge_prefetch_submission_retains_reads_after_argument_drop",
                     "mirrored": "scoped_mirror_token_rows_and_detached_prefetch_match_legacy",
                     "mirror-bytes": "scoped_mirror_wrong_bytes_refuse",
                     "mirror-source-generation": "scoped_mirror_wrong_source_generation_refuses",
                     "mirror-target-generation": "scoped_mirror_wrong_target_generation_refuses",
                     "mirror-same-fs": "scoped_mirror_same_filesystem_refuses",
                     "mirror-cached-generation": "scoped_mirror_cached_generation_change_refuses",
                     "mirror-counter": "scoped_mirror_inflight_waits_for_final_half_and_enforces_cap",
                     "mirror-counter-error": "scoped_mirror_inflight_error_half_balances"}
            test = "scoped_bridge_refuses_legacy_companion_without_fallback" if args.legacy_companion else cases[args.test_case]
            result = subprocess.run([*base, "--", "--ignored", "--exact",
                "cpu_experts::tests::" + test, "--nocapture", "--test-threads=1"], env=env)
raise SystemExit(result.returncode)
