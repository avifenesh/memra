//! Real safetensors -> production native disk loader regressions, with no CUDA dependency.
use super::*;
use memra_gguf::source::SafetensorsSource;
use std::ffi::CString;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt, symlink};
use std::path::PathBuf;

const N: usize = 2;
const OUT: usize = 512;
const IN: usize = 128;
const STACKED_NAME: &str = "blk.1.ffn_gate_exps.weight";

struct Fixture {
    root: PathBuf,
    stacked: bool,
}

impl Fixture {
    fn new(stacked: bool) -> Self {
        let template = std::env::temp_dir().join("memra-native-repack-XXXXXX");
        let mut name = CString::new(template.as_os_str().as_bytes())
            .unwrap()
            .into_bytes_with_nul();
        assert!(!unsafe { libc::mkdtemp(name.as_mut_ptr().cast()) }.is_null());
        let root = PathBuf::from(std::ffi::OsStr::from_bytes(&name[..name.len() - 1]));
        let fixture = Self { root, stacked };
        let config = if stacked {
            r#"{
              "model_type":"step3p5","num_hidden_layers":2,"hidden_size":128,
              "intermediate_size":512,"num_attention_heads":1,"num_attention_groups":1,
              "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
              "moe_num_experts":2,"moe_top_k":1,"moe_intermediate_size":512,
              "share_expert_dim":128,"moe_layers_enum":"1",
              "moe_router_activation":"sigmoid",
              "layer_types":["full_attention","sliding_attention"],
              "rope_theta":[5000000,10000],"partial_rotary_factors":[0.5,1],
              "sliding_window":512,
              "attention_other_setting":{"num_attention_heads":1,"num_attention_groups":1},
              "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0]
            }"#
        } else {
            r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":128,
              "num_attention_heads":2,"intermediate_size":512,"vocab_size":64,
              "max_position_embeddings":2048,"num_key_value_heads":2,"head_dim":64,
              "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":512}"#
        };
        std::fs::write(fixture.root.join("config.json"), config).unwrap();
        fixture.write_source(0);
        ensure_repack_cache_dir(&fixture.root.join(".memra-repack")).unwrap();
        fixture
    }

    fn write_source(&self, revision: u8) {
        let mut header = Vec::new();
        let mut data = Vec::new();
        let mut tensor = |name: &str, dtype: &str, shape: &[usize], bytes: &[u8]| {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{start},{}]}}",
                data.len()
            ));
        };
        if self.stacked {
            let codes: Vec<_> = (0..N)
                .flat_map(|e| std::iter::repeat_n(code(e, revision), OUT * IN / 2))
                .collect();
            let scales: Vec<_> = (0..N)
                .flat_map(|e| std::iter::repeat_n(scale(e), OUT * IN / 16))
                .collect();
            tensor(
                "model.layers.1.moe.gate_proj.weight",
                "U8",
                &[N, OUT, IN / 2],
                &codes,
            );
            tensor(
                "model.layers.1.moe.gate_proj.weight_scale",
                "F8_E4M3",
                &[N, OUT, IN / 16],
                &scales,
            );
            tensor(
                "model.layers.1.moe.gate_proj.weight_scale_2",
                "F32",
                &[N],
                &[0.25f32.to_le_bytes(), 0.5f32.to_le_bytes()].concat(),
            );
        } else {
            for e in 0..N {
                let stem = format!("model.layers.0.mlp.experts.{e}.gate_proj");
                tensor(
                    &format!("{stem}.weight"),
                    "U8",
                    &[OUT, IN / 2],
                    &vec![code(e, revision); OUT * IN / 2],
                );
                tensor(
                    &format!("{stem}.weight_scale"),
                    "F8_E4M3",
                    &[OUT, IN / 16],
                    &vec![scale(e); OUT * IN / 16],
                );
                tensor(
                    &format!("{stem}.weight_scale_2"),
                    "F32",
                    &[1],
                    &((e + 1) as f32 * 0.25).to_le_bytes(),
                );
            }
        }
        let header = format!("{{{}}}", header.join(","));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend(data);
        let replacement = self.root.join("replacement.safetensors");
        std::fs::write(&replacement, bytes).unwrap();
        std::fs::rename(replacement, self.root.join("model.safetensors")).unwrap();
    }

    fn source(&self) -> SafetensorsSource {
        SafetensorsSource::open(&self.root).unwrap()
    }

    fn cache(&self, src: &dyn TensorSource) -> PathBuf {
        let name = if self.stacked {
            format!(
                "{}-stacked-{N}x{OUT}x{IN}{}.nvfp4",
                STACKED_NAME.replace(['.', '/'], "-"),
                src.nvfp4_cache_tag()
            )
        } else {
            format!("blk0-gate-{N}x{OUT}x{IN}{}.nvfp4", src.nvfp4_cache_tag())
        };
        self.root.join(".memra-repack").join(name)
    }

    fn load(&self, src: &dyn TensorSource) -> io::Result<MappedRepack> {
        if self.stacked {
            let bank = src.find_nvfp4_stacked_native(STACKED_NAME).unwrap();
            assert_eq!(bank.macros, vec![0.25, 0.5]);
            load_stacked(&self.cache(src), &bank)
        } else {
            let first = src
                .find_nvfp4_native("blk.0.ffn_gate_exps.0.weight")
                .unwrap();
            load_experts(&self.cache(src), src, 0, "gate", N, first.out_f, first.in_f)
        }
    }

    fn no_temps(&self) {
        for entry in std::fs::read_dir(self.root.join(".memra-repack")).unwrap() {
            assert!(
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with('.')
            );
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

fn code(expert: usize, revision: u8) -> u8 {
    0x22 + expert as u8 * 0x22 + revision * 0x11
}

fn scale(expert: usize) -> u8 {
    0x38 + expert as u8 * 8
}

// Uniform equal nibbles make the expected block bytes explicit, independent of the repacker.
fn expected(revision: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    for expert in 0..N {
        for _ in 0..OUT * IN / 64 {
            bytes.extend([scale(expert); 4]);
            bytes.extend([code(expert, revision); 32]);
        }
    }
    bytes
}

fn assert_private(loaded: &MappedRepack, bytes: &[u8]) {
    assert_eq!(&loaded.map[..], bytes);
    let mut read = vec![0; bytes.len()];
    loaded.file.read_exact_at(&mut read, 0).unwrap();
    assert_eq!(
        read, bytes,
        "positioned reads must retain the same canonical inode"
    );
    let metadata = loaded.file.metadata().unwrap();
    assert_eq!(metadata.nlink(), 0);
    assert_eq!(metadata.mode() & 0o777, 0o400);
    let flags = unsafe { libc::fcntl(loaded.file.as_raw_fd(), libc::F_GETFL) };
    assert_eq!(flags & libc::O_ACCMODE, libc::O_RDONLY);
    assert_ne!(
        unsafe { libc::fcntl(loaded.file.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
        0
    );
    assert!(loaded.file.write_at(b"x", 0).is_err());
    assert!(loaded.file.set_len(1).is_err());
}

// Env is immutable within each test process; never set_var beside concurrent source/tests.
fn subprocess(flag: Option<&str>, test: impl FnOnce()) {
    let thread = std::thread::current();
    let name = thread.name().unwrap();
    if std::env::var("REPACK_TEST_CHILD").as_deref() == Ok(name) {
        assert_eq!(strict_identity_requested(), flag.is_some());
        test();
        return;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", name, "--nocapture"])
        .env("REPACK_TEST_CHILD", name)
        .env_remove("MEMRA_ARTIFACT_LOCK")
        .env_remove("MEMRA_REWRITE_BUNDLE");
    if let Some(flag) = flag {
        command.env(flag, ""); // presence, not validity/Unicode or a nonempty path, selects strict
    }
    assert!(command.status().unwrap().success());
}

fn lifecycle(stacked: bool) {
    let fixture = Fixture::new(stacked);
    let src = fixture.source();
    let identity = src.artifact_sha256().unwrap();
    let cache = fixture.cache(&src);
    let bytes = expected(0);
    let cold = fixture.load(&src).unwrap();
    assert_private(&cold, &bytes);
    assert_eq!(std::fs::read(&cache).unwrap(), bytes);
    let cached_inode = std::fs::metadata(&cache).unwrap().ino();
    assert_ne!(cached_inode, cold.file.metadata().unwrap().ino());
    let hit = fixture.load(&src).unwrap();
    assert_private(&hit, &bytes);
    assert_eq!(
        std::fs::metadata(&cache).unwrap().ino(),
        cached_inode,
        "verified hit must not rewrite the persistent cache"
    );

    let writer = std::fs::OpenOptions::new()
        .write(true)
        .open(&cache)
        .unwrap();
    // Corrupt past the first comparison chunk, preserving the accepted legacy length.
    writer
        .write_all_at(&[0xff], (bytes.len() - 1) as u64)
        .unwrap();
    assert!(repack_cache_is_fresh(&cache, bytes.len()));
    assert_private(&cold, &bytes);
    assert_private(&hit, &bytes);
    let reloaded_src = fixture.source();
    assert_eq!(reloaded_src.artifact_sha256().unwrap(), identity);
    let repaired = fixture.load(&reloaded_src).unwrap();
    assert_private(&repaired, &bytes);
    assert_eq!(std::fs::read(&cache).unwrap(), bytes);
    assert_ne!(std::fs::metadata(&cache).unwrap().ino(), cached_inode);

    // A writer retained across cache replacement and a new cache writer cannot affect live data.
    writer.set_len(0).unwrap();
    let named_writer = std::fs::OpenOptions::new()
        .write(true)
        .open(&cache)
        .unwrap();
    named_writer
        .write_all_at(&vec![0xee; bytes.len()], 0)
        .unwrap();
    assert_private(&repaired, &bytes);
    named_writer.set_len(0).unwrap();
    assert_private(&repaired, &bytes);
    let truncated = fixture.load(&src).unwrap();
    assert_private(&truncated, &bytes);
    std::fs::remove_file(&cache).unwrap();
    let missing = fixture.load(&src).unwrap();
    assert_private(&missing, &bytes);

    // Path replacement changes a freshly opened source; it must not change the existing source.
    fixture.write_source(1);
    let new_source = fixture.source();
    assert_ne!(new_source.artifact_sha256().unwrap(), identity);
    assert_eq!(src.artifact_sha256().unwrap(), identity);
    let new = fixture.load(&new_source).unwrap();
    assert_private(&new, &expected(1));
    let old = fixture.load(&src).unwrap();
    assert_private(&old, &bytes);
    assert_eq!(std::fs::read(&cache).unwrap(), bytes);
    drop(src);
    drop(reloaded_src);
    drop(new_source);
    assert_private(&cold, &bytes);
    assert_private(&new, &expected(1));
    assert_private(&old, &bytes);
    fixture.no_temps();
}

#[test]
fn stacked_artifact_lock_lifecycle() {
    subprocess(Some("MEMRA_ARTIFACT_LOCK"), || lifecycle(true));
}

#[test]
fn stacked_rewrite_bundle_lifecycle() {
    subprocess(Some("MEMRA_REWRITE_BUNDLE"), || lifecycle(true));
}

#[test]
fn per_expert_artifact_lock_lifecycle() {
    subprocess(Some("MEMRA_ARTIFACT_LOCK"), || lifecycle(false));
}

#[test]
fn per_expert_rewrite_bundle_lifecycle() {
    subprocess(Some("MEMRA_REWRITE_BUNDLE"), || lifecycle(false));
}

#[test]
fn legacy_stacked_and_per_expert_keep_existing_cache_behavior() {
    subprocess(None, || {
        for stacked in [true, false] {
            let fixture = Fixture::new(stacked);
            let source = fixture.source();
            let cache = fixture.cache(&source);
            let cold = fixture.load(&source).unwrap();
            assert_eq!(&cold.map[..], expected(0));
            assert_eq!(cold.file.metadata().unwrap().nlink(), 1);
            drop(cold);
            let unverified = vec![0xa5; expected(0).len()];
            std::fs::write(&cache, &unverified).unwrap();
            let hit = fixture.load(&source).unwrap();
            assert_eq!(&hit.map[..], unverified);
            assert_eq!(
                hit.file.metadata().unwrap().ino(),
                std::fs::metadata(&cache).unwrap().ino()
            );
            fixture.no_temps();
        }
    });
}

#[test]
fn strict_both_loaders_reject_links_directories_and_fifos() {
    subprocess(Some("MEMRA_REWRITE_BUNDLE"), || {
        for stacked in [true, false] {
            let fixture = Fixture::new(stacked);
            let src = fixture.source();
            let cache = fixture.cache(&src);
            let outside = fixture.root.join("outside");
            let bytes = expected(0);
            std::fs::write(&outside, &bytes).unwrap();
            symlink(&outside, &cache).unwrap();
            assert!(fixture.load(&src).is_err());
            std::fs::remove_file(&cache).unwrap();
            std::fs::hard_link(&outside, &cache).unwrap();
            assert!(fixture.load(&src).is_err());
            std::fs::remove_file(&cache).unwrap();
            std::fs::create_dir(&cache).unwrap();
            assert!(fixture.load(&src).is_err());
            std::fs::remove_dir(&cache).unwrap();
            let name = CString::new(cache.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            assert!(fixture.load(&src).is_err());
            std::fs::remove_file(&cache).unwrap();
            assert_eq!(std::fs::read(&outside).unwrap(), bytes);
            fixture.no_temps();
            std::fs::remove_dir(cache.parent().unwrap()).unwrap();
            symlink(&fixture.root, cache.parent().unwrap()).unwrap();
            assert!(fixture.load(&src).is_err());
        }
    });
}

#[test]
fn strict_failure_does_not_publish_partial_bytes_or_leave_temps() {
    let fixture = Fixture::new(true);
    let src = fixture.source();
    let cache = fixture.cache(&src);
    std::fs::write(&cache, b"previous").unwrap();
    assert!(
        strict_file(&cache, 4, |out| {
            assert_eq!(out.get_ref().metadata()?.nlink(), 0);
            out.write_all(b"bad")?;
            Err(io::Error::other("injected repack failure"))
        })
        .is_err()
    );
    assert!(strict_file(&cache, 4, |out| out.write_all(b"bad")).is_err());
    assert_eq!(std::fs::read(&cache).unwrap(), b"previous");
    fixture.no_temps();
}

#[test]
fn strict_publication_uses_held_directory_even_if_cache_path_is_replaced() {
    let fixture = Fixture::new(true);
    let src = fixture.source();
    let cache = fixture.cache(&src);
    let moved = fixture.root.join("moved-cache");
    let backing = strict_file(&cache, 4, |out| {
        std::fs::rename(cache.parent().unwrap(), &moved)?;
        std::fs::create_dir(cache.parent().unwrap())?;
        std::fs::write(&cache, b"keep")?;
        out.write_all(b"real")
    })
    .unwrap();
    assert_eq!(std::fs::read(&cache).unwrap(), b"keep");
    assert_eq!(
        std::fs::read(moved.join(cache.file_name().unwrap())).unwrap(),
        b"real"
    );
    assert_eq!(backing.metadata().unwrap().nlink(), 0);
    let mut bytes = [0; 4];
    backing.read_exact_at(&mut bytes, 0).unwrap();
    assert_eq!(&bytes, b"real");
    assert_eq!(std::fs::read_dir(moved).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn repack_cache_refuses_symlinked_directory_and_file() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!("memra-repack-links-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let target_dir = root.join("target-dir");
    std::fs::create_dir(&target_dir).unwrap();
    let cache_dir = root.join(".memra-repack");
    symlink(&target_dir, &cache_dir).unwrap();
    let error = ensure_repack_cache_dir(&cache_dir).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

    std::fs::remove_file(&cache_dir).unwrap();
    std::fs::create_dir(&cache_dir).unwrap();
    let target = root.join("outside.bin");
    std::fs::write(&target, b"keep").unwrap();
    let cache_file = cache_dir.join("artifact.nvfp4");
    symlink(&target, &cache_file).unwrap();
    assert!(!repack_cache_is_fresh(&cache_file, 4));
    let error = open_repack_cache(&cache_file, true).unwrap_err();
    assert_ne!(error.kind(), std::io::ErrorKind::NotFound);
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");

    let hardlink = cache_dir.join("hardlink.nvfp4");
    std::fs::hard_link(&target, &hardlink).unwrap();
    let error = write_repack_cache(&hardlink, |out| {
        use std::io::Write;
        out.write_all(b"replacement")
    })
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
    std::fs::remove_dir_all(root).ok();
}
