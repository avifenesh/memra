//! Bounded native NVFP4 disk-cache lifecycle gate through PUBLIC HostExps callers.
//!
//! Compile on CPU (also from macOS): DOCS_RS=1 MEMRA_MMQ_ARCHIVE_HASH=host-check-placeholder
//! cargo check --target x86_64-unknown-linux-gnu -p memra-engine --test native_repack_gpu --locked
//! (placeholder CUDA modules; compilation is not native evidence).
//!
//! After final-delta review, build without DOCS_RS before acquiring a device:
//! cargo test -p memra-engine --test native_repack_gpu --release --locked --no-run
//! Run that reported test executable ($TEST_BIN) inside the parent's single-card wrapper:
//! memra-gpu-run --gpus GPU-<assigned-full-uuid> --receipt /absolute/fresh/lease -- \
//!   env MEMRA_ARTIFACT_LOCK='' MEMRA_ST_REPACK_DISK=1 MEMRA_ST_PINNED=0 \
//!   REWRITE_REPACK_OUT=/absolute/fresh/evidence "$TEST_BIN" \
//!   native_nvfp4_stacked_cache_lifecycle --exact --ignored --nocapture --test-threads=1
//! Repeat with native_nvfp4_per_expert_cache_lifecycle. Require 1 passed, 0 ignored per run.
//! MEMRA_REWRITE_BUNDLE may replace MEMRA_ARTIFACT_LOCK: presence selects the loader's strict
//! policy; this fixture intentionally performs no full-model admission or lock qualification.
//! No in-process environment mutation. The existing Python lease verifier checks the wrapper's
//! ancestor-owned per-card FLOCK, MEMRA_GPU_LEASE_FILE and CUDA_VISIBLE_DEVICES before Engine
//! creation and after all checks. Python3 and the built checkout must remain available.
//!
//! One immutable generated safetensors artifact per layout, 2 experts x 512 x 128, 73,728
//! repacked bytes. Every observation stages both mmap slices and retained-fd positioned reads
//! through Engine::stage_expert, checks D2H bytes, and runs qmatvec_view + scale_inplace.
//! The arithmetic gate is fixed-input NVFP4/f32 dequant matvec, not a W4A4 rewrite, model-quality,
//! serving, or whole-model qualification; these fixtures never promote a support state.

#[cfg(unix)]
#[path = "support/native_repack_fixture.rs"]
mod native_repack_fixture;

#[cfg(unix)]
use native_repack_fixture::*;

#[test]
#[ignore = "requires assigned CUDA device"]
fn native_nvfp4_stacked_cache_lifecycle() {
    #[cfg(unix)]
    lifecycle(Layout::Stacked);
    #[cfg(not(unix))]
    panic!("native cache lifecycle requires Unix private file backing and CUDA");
}

#[test]
#[ignore = "requires assigned CUDA device"]
fn native_nvfp4_per_expert_cache_lifecycle() {
    #[cfg(unix)]
    lifecycle(Layout::PerExpert);
    #[cfg(not(unix))]
    panic!("native cache lifecycle requires Unix private file backing and CUDA");
}

#[cfg(unix)]
fn observe(
    engine: &memra_engine::Engine,
    bank: &memra_engine::model::HostExps,
    phase: &str,
    evidence: &mut Evidence,
) {
    use memra_engine::model::HostBuf;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, MetadataExt};

    assert_eq!(
        (bank.n_expert, bank.out_f, bank.in_f),
        (EXPERTS, ROWS, COLS)
    );
    assert_eq!(
        (bank.qtype, bank.row_bytes, bank.expert_stride),
        (memra_engine::QT_NVFP4, ROW_BYTES, STRIDE)
    );
    assert!(bank.layouts.is_none() && bank.tiers.is_none() && bank.fp8_blk.is_none());
    assert_eq!(bank.macros.as_deref(), Some(MACROS.as_slice()));
    let HostBuf::Mmap { file, off, len, .. } = &bank.bytes else {
        panic!("public loader did not select native disk-backed HostExps");
    };
    let metadata = file.metadata().unwrap();
    assert!(metadata.is_file());
    assert_eq!((metadata.nlink(), metadata.mode() & 0o777), (0, 0o400));
    assert_eq!(
        (*off, *len, metadata.len()),
        (0, EXPERTS * STRIDE, (EXPERTS * STRIDE) as u64)
    );
    // SAFETY: querying flags of the retained, live File owned by HostExps.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(flags & libc::O_ACCMODE, libc::O_RDONLY);
    let descriptor_flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) };
    assert!(descriptor_flags >= 0 && descriptor_flags & libc::FD_CLOEXEC != 0);
    assert!(file.write_at(&[0xee], 0).is_err());
    assert!(file.set_len(0).is_err());
    let expected = expected_bytes();
    assert_eq!(bank.bytes.as_bytes(), expected);
    let mut positioned = vec![0; *len];
    file.read_exact_at(&mut positioned, *off as u64).unwrap();
    assert_eq!(positioned, expected);
    evidence.save(
        &format!("{phase}-backing.txt"),
        format!(
            "dev={} ino={} nlink={} mode={:o} length={} read_only=true cloexec=true\n",
            metadata.dev(),
            metadata.ino(),
            metadata.nlink(),
            metadata.mode() & 0o777,
            metadata.len()
        )
        .as_bytes(),
    );

    let x = engine.htod(&input()).expect("fixed input H2D");
    for (origin, bytes) in [
        ("mmap", bank.bytes.as_bytes()),
        ("pread", positioned.as_slice()),
    ] {
        let mut readback = Vec::new();
        // Scratch is rebuilt from the live host backing on EVERY observation, including after
        // truncation. An old GPU copy cannot conceal a stale/mutated host mapping.
        let mut scratch = engine.alloc_u8(STRIDE).expect("native staging allocation");
        for expert in 0..EXPERTS {
            let layout = bank.expert_layout(expert);
            assert_eq!(
                (layout.offset, layout.len, layout.qtype, layout.row_bytes),
                (expert * STRIDE, STRIDE, memra_engine::QT_NVFP4, ROW_BYTES)
            );
            let host = if origin == "mmap" {
                bank.expert_bytes(expert)
            } else {
                &bytes[layout.offset..layout.offset + layout.len]
            };
            engine
                .stage_expert(host, &mut scratch, 0)
                .expect("actual expert staging H2D");
            let got = engine.dtoh_u8(&scratch).expect("staged expert D2H");
            assert_eq!(
                got,
                expected[expert * STRIDE..(expert + 1) * STRIDE],
                "{phase}/{origin} expert {expert} GPU bytes"
            );
            readback.extend(got);
            let mut y = engine
                .qmatvec_view(
                    &scratch,
                    0..layout.len,
                    &x.as_view(),
                    1,
                    bank.in_f,
                    bank.out_f,
                    layout.qtype,
                    layout.row_bytes,
                )
                .expect("native NVFP4 f32-dequant matvec");
            for scaled in [false, true] {
                if scaled {
                    engine
                        .scale_inplace(&mut y, bank.macro_scale(expert), ROWS)
                        .expect("native macro-scale");
                }
                let output = engine.dtoh(&y).expect("matvec D2H");
                let output_bytes = f32_bytes(&output);
                evidence.save(
                    &format!(
                        "{phase}-{origin}-{expert}-{}.f32le",
                        if scaled { "scaled" } else { "raw" }
                    ),
                    &output_bytes,
                );
                assert_eq!(
                    output_bytes,
                    f32_bytes(&expected_output(expert, scaled)),
                    "{phase}/{origin} expert {expert} scaled={scaled} output bits"
                );
            }
        }
        evidence.save(&format!("{phase}-{origin}.nvfp4"), &readback);
    }
}

#[cfg(unix)]
fn lifecycle(layout: Layout) {
    use memra_engine::model::HostBuf;
    use memra_gguf::source::{SafetensorsSource, TensorSource};
    use std::fs::OpenOptions;
    use std::os::unix::fs::{FileExt, MetadataExt};
    use std::sync::Mutex;

    static GPU: Mutex<()> = Mutex::new(());
    let _process_lock = GPU.lock().expect("another native fixture failed");
    let lease = native_lease();
    let mut evidence = Evidence::new(layout);
    evidence.save("lease.json", &lease);
    let engine = memra_engine::Engine::new(0)
        .expect("assigned CUDA device and real native modules required; no skip");
    let source = SafetensorsSource::open(&evidence.artifact).unwrap();
    let identity = source.artifact_sha256().unwrap();
    let cache = layout.cache(&source, &evidence.artifact);
    assert!(
        !cache.exists(),
        "cold load must start without a named cache"
    );
    let expected = expected_bytes();
    let cold = layout.load(&engine, &source);
    observe(&engine, &cold, "cold", &mut evidence);
    assert_eq!(std::fs::read(&cache).unwrap(), expected);
    let cached_inode = std::fs::metadata(&cache).unwrap().ino();
    let HostBuf::Mmap { file, .. } = &cold.bytes else {
        unreachable!()
    };
    assert_ne!(file.metadata().unwrap().ino(), cached_inode);
    let hit = layout.load(&engine, &source);
    assert_eq!(
        std::fs::metadata(&cache).unwrap().ino(),
        cached_inode,
        "verified cache hit must preserve the published inode"
    );
    observe(&engine, &hit, "verified-hit", &mut evidence);
    assert_eq!(std::fs::read(&cache).unwrap(), expected);

    // Poison beyond the comparison's first 64 KiB, keeping the legacy size gate green.
    assert!(expected.len() > 64 * 1024);
    let old_writer = OpenOptions::new().write(true).open(&cache).unwrap();
    old_writer
        .write_all_at(&[0xff], (expected.len() - 1) as u64)
        .unwrap();
    old_writer.sync_all().unwrap();
    assert_eq!(
        std::fs::metadata(&cache).unwrap().len(),
        expected.len() as u64
    );
    let poisoned = std::fs::read(&cache).unwrap();
    assert_ne!(poisoned, expected);
    evidence.save("poisoned-same-size.nvfp4", &poisoned);
    let reopened = SafetensorsSource::open(&evidence.artifact).unwrap();
    assert_eq!(reopened.artifact_sha256().unwrap(), identity);
    // Current strict-loader contract repairs a regular poisoned cache from canonical source.
    // A successful return carrying poisoned bytes can never be accepted as a cache hit.
    let repaired = layout.load(&engine, &reopened);
    assert_eq!(std::fs::read(&cache).unwrap(), expected);
    assert_ne!(std::fs::metadata(&cache).unwrap().ino(), cached_inode);
    observe(&engine, &repaired, "regenerated", &mut evidence);
    old_writer.set_len(0).unwrap();
    assert_eq!(
        std::fs::read(&cache).unwrap(),
        expected,
        "retired cache writer must not mutate the replacement"
    );

    let writer = OpenOptions::new().write(true).open(&cache).unwrap();
    writer.write_all_at(&vec![0xee; expected.len()], 0).unwrap();
    writer.sync_all().unwrap();
    assert_eq!(std::fs::read(&cache).unwrap(), vec![0xee; expected.len()]);
    for (name, bank) in [("cold", &cold), ("hit", &hit), ("repaired", &repaired)] {
        observe(&engine, bank, &format!("mutated-{name}"), &mut evidence);
    }
    writer.set_len(0).unwrap();
    writer.sync_all().unwrap();
    assert_eq!(std::fs::metadata(&cache).unwrap().len(), 0);
    drop(source);
    drop(reopened);
    for (name, bank) in [("cold", &cold), ("hit", &hit), ("repaired", &repaired)] {
        observe(
            &engine,
            bank,
            &format!("truncated-source-dropped-{name}"),
            &mut evidence,
        );
    }
    let final_source = SafetensorsSource::open(&evidence.artifact).unwrap();
    assert_eq!(final_source.artifact_sha256().unwrap(), identity);
    let final_bank = layout.load(&engine, &final_source);
    observe(&engine, &final_bank, "truncated-reload", &mut evidence);
    assert_eq!(std::fs::read(&cache).unwrap(), expected);
    // A completed strict load leaves only the named cache, no private/publication temporaries.
    let entries: Vec<_> = std::fs::read_dir(cache.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(entries, vec![cache]);
    assert_eq!(
        native_lease(),
        lease,
        "native lease changed during the fixture"
    );
    let binary_hash = file_sha256(&std::env::current_exe().unwrap());
    evidence.finish(&format!(
        "test={}\nresult=pass\nsource_format=native NVFP4 safetensors\nartifact_sha256={identity}\nbinary_sha256={binary_hash}\nshape=2x512x128\nrepacked_bytes={}\nstrict_identity=true\ncache_hit=verified inode preserved\npoisoned_cache=canonical regeneration\nretained_backing=unlinked read-only mmap and positioned File reads\ntransfer=Engine::stage_expert H2D and Engine::dtoh_u8 D2H\nmath=Engine::qmatvec_view NVFP4 f32 dequant plus scale_inplace; exact f32 bits\nmodel_qualification=not performed; no support promotion\n",
        layout.name(), expected.len()
    ));
}
