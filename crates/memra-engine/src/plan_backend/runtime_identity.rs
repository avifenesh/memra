//! Host-only identity and admission rules, shared by the loader and CPU regressions.
use memra_gguf::execution_manifest::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ffi::OsString;

pub(crate) fn identity_requested() -> bool {
    std::env::var_os("MEMRA_ARTIFACT_LOCK").is_some()
        || std::env::var_os("MEMRA_REWRITE_BUNDLE").is_some()
}

// Location controls do not select a numerical program. Everything else is included,
// even diagnostic and unused flags: false rejection is preferable to false qualification.
fn numeric_env_key(key: &str) -> bool {
    (key.starts_with("MEMRA_")
        && !matches!(
            key,
            "MEMRA_ARTIFACT_LOCK"
                | "MEMRA_REWRITE_BUNDLE"
                | "MEMRA_REWRITE_RECEIPT"
                | "MEMRA_GPU_LEASE_FILE"
        ))
        || key.starts_with("CUDA_")
        || key.starts_with("NVIDIA_")
        || key.starts_with("CUBLAS_")
        || key.starts_with("CUBLASLT_")
        || matches!(key, "LD_PRELOAD" | "LD_LIBRARY_PATH")
}

pub(super) fn numeric_environment(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> BTreeMap<OsString, OsString> {
    vars.into_iter()
        .filter(|(key, _)| numeric_env_key(&key.to_string_lossy()))
        .collect()
}

/// The captured environment must describe a policy the engine latched at its first read.
/// A later write cannot change the latch, so an identity built from the new value would lie.
pub(super) fn check_latched_env(
    key: &str,
    latched: Option<bool>,
    env_on: bool,
) -> Result<(), String> {
    match latched {
        Some(on) if on != env_on => Err(format!(
            "{key} latched {} at its first read but the environment now selects {}; the \
             captured identity would not describe the running program. Set {key} for the whole \
             process before start",
            if on { "on" } else { "off" },
            if env_on { "on" } else { "off" },
        )),
        _ => Ok(()),
    }
}

pub(crate) fn refuse_external_rewrite_artifacts() -> Result<(), String> {
    for key in [
        "MEMRA_MTP_DRAFT",
        "MEMRA_DRAFT",
        "MEMRA_SPEC_DFLASH",
        "MEMRA_DSPARK_DRAFT",
        "MEMRA_GLM5_DFLASH",
        "MEMRA_FRSPEC_TRIM",
    ] {
        if std::env::var_os(key).is_some_and(|value| !value.is_empty()) {
            return Err(format!(
                "rewrite identity with {key} is unsupported until composite artifact identity is implemented"
            ));
        }
    }
    Ok(())
}

/// Hash the executable inode actually running, including when its pathname was replaced.
/// On Linux current_exe() names a path that can already refer to a different build.
pub fn running_implementation_sha256() -> Result<String, Box<dyn std::error::Error>> {
    #[cfg(target_os = "linux")]
    let path = std::path::PathBuf::from("/proc/self/exe");
    #[cfg(not(target_os = "linux"))]
    let path = std::env::current_exe()?;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

pub(super) fn hash_parts(parts: impl IntoIterator<Item = impl AsRef<[u8]>>) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        let bytes = part.as_ref();
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

pub(super) fn numeric_program_sha256(
    load_mtp: bool,
    interpretation: &str,
    hardware: &str,
    model_state: &str,
    environment: &BTreeMap<OsString, OsString>,
) -> String {
    let mut parts = vec![
        b"memra-runtime-numeric-v1".to_vec(),
        vec![u8::from(load_mtp)],
        interpretation.as_bytes().to_vec(),
        hardware.as_bytes().to_vec(),
        model_state.as_bytes().to_vec(),
    ];
    for (key, value) in environment {
        parts.push(key.as_encoded_bytes().to_vec());
        parts.push(value.as_encoded_bytes().to_vec());
    }
    hash_parts(parts)
}

/// Captured after loader defaults and capacity-dependent mirror/placement decisions.
/// This is private loader state, never deserialized from a qualification bundle.
pub(crate) struct RewriteLoadState {
    pub(super) mutation_generation: u64,
    pub(crate) pipeline: bool,
    pub(super) model_sha256: String,
    pub(super) environment: BTreeMap<OsString, OsString>,
    pub(super) libraries: Option<LoadedLibraries>,
}

/// Content identity plus inode metadata of all file-backed executable mappings. The
/// files must remain immutable while loaded, just like checkpoint mappings. A replaced
/// pathname is rejected even if it contains identical bytes: it is not the mapped inode.
#[derive(Debug)]
pub(super) struct LoadedLibraries {
    pub(super) sha256: String,
    inventory: BTreeMap<String, LibraryStamp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LibraryStamp {
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

// Parse the five fixed maps columns without losing spaces inside a pathname.
fn mapped_library(line: &str) -> Result<Option<(u64, u64, u64, &str)>, String> {
    let mut rest = line;
    let mut fields = Vec::with_capacity(5);
    for _ in 0..5 {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        fields.push(&rest[..end]);
        rest = rest[end..].trim_start();
    }
    if !fields[1].contains('x') || fields[4] == "0" {
        return Ok(None);
    }
    if !rest.starts_with('/') || rest.ends_with(" (deleted)") || rest.contains('\\') {
        return Err("rewrite library identity cannot verify an anonymous, deleted, or escaped executable mapping".into());
    }
    let (major, minor) = fields[3].split_once(':').ok_or("malformed maps device")?;
    Ok(Some((
        u64::from_str_radix(major, 16).map_err(|e| e.to_string())?,
        u64::from_str_radix(minor, 16).map_err(|e| e.to_string())?,
        fields[4].parse::<u64>().map_err(|e| e.to_string())?,
        rest,
    )))
}

#[cfg(any(target_os = "linux", all(test, unix)))]
fn library_stamp(metadata: &std::fs::Metadata) -> LibraryStamp {
    use std::os::unix::fs::MetadataExt;
    LibraryStamp {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.len(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    }
}

fn library_inventory() -> Result<BTreeMap<String, LibraryStamp>, String> {
    #[cfg(target_os = "linux")]
    {
        let maps = std::fs::read_to_string("/proc/self/maps").map_err(|e| e.to_string())?;
        let mut inventory = BTreeMap::new();
        for line in maps.lines() {
            let Some((major, minor, inode, path)) = mapped_library(line)? else {
                continue;
            };
            let stamp = library_stamp(
                &std::fs::metadata(path).map_err(|e| format!("mapped library {path}: {e}"))?,
            );
            if u64::from(libc::major(stamp.device)) != major
                || u64::from(libc::minor(stamp.device)) != minor
                || stamp.inode != inode
            {
                return Err(format!(
                    "mapped library {path} was replaced; inode/device differs from /proc/self/maps"
                ));
            }
            inventory.insert(path.to_string(), stamp);
        }
        if inventory.is_empty() {
            return Err("no verifiable executable library mappings".into());
        }
        Ok(inventory)
    }
    #[cfg(not(target_os = "linux"))]
    Err("trusted loaded-library identity is currently supported only on Linux".into())
}

impl LoadedLibraries {
    pub(super) fn capture() -> Result<Self, String> {
        let inventory = library_inventory()?;
        let mut digests = Vec::new();
        for (path, expected) in &inventory {
            let mut file =
                std::fs::File::open(path).map_err(|e| format!("mapped library {path}: {e}"))?;
            #[cfg(target_os = "linux")]
            if library_stamp(&file.metadata().map_err(|e| e.to_string())?) != *expected {
                return Err(format!("mapped library {path} changed before hashing"));
            }
            let mut hash = Sha256::new();
            std::io::copy(&mut file, &mut hash).map_err(|e| e.to_string())?;
            #[cfg(target_os = "linux")]
            if library_stamp(&file.metadata().map_err(|e| e.to_string())?) != *expected {
                return Err(format!("mapped library {path} changed while hashing"));
            }
            #[cfg(not(target_os = "linux"))]
            let _ = expected;
            digests.push(format!("{:x}", hash.finalize()));
        }
        // Content, not installation paths or ASLR addresses, is the portable identity.
        digests.sort();
        if library_inventory()? != inventory {
            return Err("loaded libraries changed while capturing identity".into());
        }
        Ok(Self {
            sha256: hash_parts(digests),
            inventory,
        })
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        self.validate_inventory(library_inventory()?)
    }

    fn validate_inventory(&self, current: BTreeMap<String, LibraryStamp>) -> Result<(), String> {
        if current != self.inventory {
            let added: Vec<_> = current
                .keys()
                .filter(|path| !self.inventory.contains_key(*path))
                .collect();
            let removed: Vec<_> = self
                .inventory
                .keys()
                .filter(|path| !current.contains_key(*path))
                .collect();
            let changed: Vec<_> = current
                .iter()
                .filter_map(|(path, stamp)| {
                    self.inventory
                        .get(path)
                        .filter(|old| *old != stamp)
                        .map(|_| path)
                })
                .collect();
            return Err(format!(
                "loaded executable mappings changed: added={added:?} removed={removed:?} modified={changed:?}"
            ));
        }
        Ok(())
    }
}

/// Portable injected inventory for re-entry tests. It uses the production file stamps
/// and comparison; the Linux /proc reader and real executable mappings have separate gates.
#[cfg(all(test, unix))]
pub(super) fn file_inventory_for_reentry_test(
    path: &std::path::Path,
) -> impl Fn() -> Result<(), String> + use<> {
    let path = path.to_path_buf();
    let read = move || -> Result<BTreeMap<String, LibraryStamp>, String> {
        let stamp = library_stamp(&std::fs::metadata(&path).map_err(|error| error.to_string())?);
        Ok(BTreeMap::from([(path.display().to_string(), stamp)]))
    };
    let expected = LoadedLibraries {
        sha256: String::new(),
        inventory: read().unwrap(),
    };
    move || expected.validate_inventory(read()?)
}

/// Untrusted source implementations must explicitly support identity of their opened bytes.
pub(crate) fn source_artifact_identity(
    source: &dyn memra_gguf::source::TensorSource,
    requested: bool,
) -> Result<Option<String>, String> {
    if requested {
        source
            .artifact_sha256()
            .map(Some)
            .map_err(|error| format!("rewrite artifact identity: {error}"))
    } else {
        Ok(None)
    }
}

pub(crate) fn checked_rewrite_identity(
    identity: Option<&RewriteIdentity>,
    state_matches: bool,
) -> Result<&RewriteIdentity, String> {
    let identity = identity.ok_or(
        "rewrite identity was not captured at load; set MEMRA_ARTIFACT_LOCK or MEMRA_REWRITE_BUNDLE before loading",
    )?;
    identity.validate()?;
    if !state_matches {
        return Err("rewrite identity is stale: plan, loaded program, or environment changed since load; external draft attachment requires a composite identity".into());
    }
    Ok(identity)
}

#[cfg(test)]
pub(crate) fn rewrite_admission_allows(
    admission: &RewriteAdmission,
    identity_current: bool,
    strict_requested: bool,
    surface: RewriteSurface,
) -> bool {
    match admission {
        RewriteAdmission::LegacyUnbundled => !strict_requested,
        RewriteAdmission::StrictPending => false,
        RewriteAdmission::Qualified(qualifications) => {
            identity_current && qualifications.allows(surface)
        }
    }
}

pub(crate) fn install_rewrite_admission(
    admission: &mut RewriteAdmission,
    bundle: &std::path::Path,
    plan: &memra_gguf::model_plan::ModelPlan,
    identity: Result<&RewriteIdentity, String>,
) -> Result<(), String> {
    // Reset first, even when reinstalling on a previously qualified model.
    *admission = RewriteAdmission::StrictPending;
    let qualifications = RewriteQualifications::load(bundle, plan, identity?)?;
    // Some direct eager paths have no optional-rewrite check. Never return a strict
    // model to any caller without a qualified native eager baseline.
    if !qualifications.allows(RewriteSurface::DecodeEager) {
        return Err("rewrite qualification requires a qualified decode-eager baseline".into());
    }
    *admission = RewriteAdmission::Qualified(qualifications);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
        numeric_environment(pairs.iter().map(|(k, v)| ((*k).into(), (*v).into())))
    }

    #[test]
    fn identity_capture_refuses_an_environment_that_disagrees_with_a_latch() {
        // Not yet latched, or latched to the same value: capture proceeds as before.
        for env_on in [false, true] {
            check_latched_env("MEMRA_BF16_MMV", None, env_on).unwrap();
            check_latched_env("MEMRA_BF16_MMV", Some(env_on), env_on).unwrap();
        }
        let error = check_latched_env("MEMRA_BF16_MMV", Some(false), true).unwrap_err();
        assert!(error.contains("MEMRA_BF16_MMV latched off"), "{error}");
        assert!(error.contains("now selects on"), "{error}");
        assert!(error.contains("for the whole process"), "{error}");
        let error = check_latched_env("MEMRA_BF16_MMV", Some(true), false).unwrap_err();
        assert!(error.contains("latched on"), "{error}");
    }

    #[test]
    fn numerical_identity_ignores_receipt_locations_but_binds_runtime_settings() {
        let base = vars(&[("MEMRA_PP_F16", "0"), ("CUDA_VISIBLE_DEVICES", "0")]);
        let moved = vars(&[
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("MEMRA_PP_F16", "0"),
            ("MEMRA_ARTIFACT_LOCK", "/different/lock"),
            ("MEMRA_REWRITE_BUNDLE", "/bundle"),
            ("MEMRA_REWRITE_RECEIPT", "/out"),
            ("MEMRA_GPU_LEASE_FILE", "/new-wrapper-receipt/lease.json"),
            ("PATH", "/different"),
        ]);
        assert_eq!(base, moved);
        let hash = |mtp, source, hardware, model, env: &BTreeMap<OsString, OsString>| {
            numeric_program_sha256(mtp, source, hardware, model, env)
        };
        let expected = hash(true, "source", "gpu/driver/placement", "mirrors", &base);
        assert_eq!(
            expected,
            hash(true, "source", "gpu/driver/placement", "mirrors", &moved)
        );
        for changed in [
            hash(false, "source", "gpu/driver/placement", "mirrors", &base),
            hash(
                true,
                "other-source",
                "gpu/driver/placement",
                "mirrors",
                &base,
            ),
            hash(
                true,
                "source",
                "other-gpu/driver/placement",
                "mirrors",
                &base,
            ),
            hash(
                true,
                "source",
                "gpu/driver/placement",
                "other-mirrors",
                &base,
            ),
            hash(
                true,
                "source",
                "gpu/driver/placement",
                "mirrors",
                &vars(&[("MEMRA_PP_F16", "1")]),
            ),
        ] {
            assert_ne!(expected, changed);
        }
    }

    #[test]
    fn numerical_snapshot_frames_values_and_keeps_unknown_memra_settings() {
        assert_ne!(hash_parts(["ab", "c"]), hash_parts(["a", "bc"]));
        assert!(numeric_env_key("MEMRA_FUTURE_NUMERIC_PROGRAM"));
        assert!(numeric_env_key("CUBLAS_WORKSPACE_CONFIG"));
        // cuBLASLt reads its own prefix (for example its heuristics cache capacity).
        assert!(numeric_env_key("CUBLASLT_HEURISTICS_CACHE_CAPACITY"));
        assert!(numeric_env_key("CUBLASLT_LOG_LEVEL"));
        assert_eq!(
            vars(&[
                ("CUBLASLT_HEURISTICS_CACHE_CAPACITY", "0"),
                ("PATH", "/bin")
            ])
            .len(),
            1
        );
        assert_eq!(vars(&[("MEMRA_FAST", "0"), ("MEMRA_FAST", "1")]).len(), 1);
    }

    #[test]
    fn implementation_identity_hashes_running_executable() {
        let identity = running_implementation_sha256().unwrap();
        assert_eq!(identity.len(), 64);
        #[cfg(target_os = "linux")]
        assert_eq!(
            identity,
            format!(
                "{:x}",
                Sha256::digest(std::fs::read("/proc/self/exe").unwrap())
            )
        );
    }

    fn identity() -> RewriteIdentity {
        RewriteIdentity {
            artifact_sha256: "a".repeat(64),
            implementation_sha256: "b".repeat(64),
            numeric_program_sha256: "c".repeat(64),
        }
    }

    fn plan() -> memra_gguf::model_plan::ModelPlan {
        memra_gguf::model_packs::by_alias("qwen3")
            .unwrap()
            .compile_tiny_plan()
            .unwrap()
    }

    #[test]
    fn accessor_rejects_absent_invalid_or_stale_loaded_identity() {
        let identity = identity();
        assert!(
            checked_rewrite_identity(None, true)
                .unwrap_err()
                .contains("not captured")
        );
        assert!(
            checked_rewrite_identity(Some(&identity), false)
                .unwrap_err()
                .contains("stale")
        );
        assert_eq!(
            checked_rewrite_identity(Some(&identity), true).unwrap(),
            &identity
        );
        let mut invalid = identity.clone();
        invalid.numeric_program_sha256.clear();
        assert!(checked_rewrite_identity(Some(&invalid), true).is_err());
    }

    struct UnsupportedSource;
    impl memra_gguf::source::TensorSource for UnsupportedSource {
        fn config(&self) -> memra_gguf::config::ModelConfig {
            panic!("identity refusal must precede model loading")
        }
        fn find(&self, _: &str) -> Option<memra_gguf::source::TensorView<'_>> {
            None
        }
    }

    #[test]
    fn unsupported_source_is_legacy_only_and_refuses_before_tensor_loading() {
        assert_eq!(
            source_artifact_identity(&UnsupportedSource, false).unwrap(),
            None
        );
        assert!(
            source_artifact_identity(&UnsupportedSource, true)
                .unwrap_err()
                .contains("trusted loaded-artifact identity")
        );
    }

    struct Bundle(std::path::PathBuf);
    impl Bundle {
        fn new(surfaces: &[RewriteSurface]) -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "memra-runtime-identity-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(root.join("rewrite-receipts")).unwrap();
            let lock = b"diagnostic artifact lock, never a source of identity";
            std::fs::write(root.join("artifact.lock"), lock).unwrap();
            let mut index = "rewrite\tplan_sha256\treceipt_sha256\tstatus\n".to_string();
            for surface in surfaces {
                let rewrite = execution_rewrites(&plan())
                    .into_iter()
                    .find(|r| r.surface == *surface)
                    .unwrap();
                let receipt = rewrite
                    .verify_tokens(&identity().implementation_sha256, &[1, 2], &[1, 2])
                    .unwrap()
                    .bind_runtime_identity(&identity())
                    .unwrap()
                    .bind_artifact_lock(lock)
                    .to_tsv();
                std::fs::write(
                    root.join("rewrite-receipts")
                        .join(format!("{}.tsv", rewrite.id)),
                    &receipt,
                )
                .unwrap();
                index.push_str(&format!(
                    "{}\t{}\t{:x}\tpassed\n",
                    rewrite.id,
                    rewrite.plan_sha256,
                    Sha256::digest(receipt.as_bytes())
                ));
            }
            std::fs::write(root.join("rewrite-receipts.tsv"), index).unwrap();
            Self(root)
        }
    }
    impl Drop for Bundle {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn strict_install_requires_eager_and_failed_reinstall_revokes_all_permission() {
        let plan = plan();
        let identity = identity();
        let eager = Bundle::new(&[RewriteSurface::DecodeEager]);
        let batch_only = Bundle::new(&[RewriteSurface::DecodeBatch]);
        let mut admission = RewriteAdmission::LegacyUnbundled;
        assert!(!admission.is_qualified());
        install_rewrite_admission(&mut admission, &eager.0, &plan, Ok(&identity)).unwrap();
        assert!(admission.is_qualified());
        assert!(admission.allows(RewriteSurface::DecodeEager));
        assert!(!admission.allows(RewriteSurface::DecodeBatch));
        let error = install_rewrite_admission(&mut admission, &batch_only.0, &plan, Ok(&identity))
            .unwrap_err();
        assert!(error.contains("decode-eager baseline"), "{error}");
        assert_eq!(admission, RewriteAdmission::StrictPending);
        assert!(!admission.allows(RewriteSurface::DecodeEager));
        assert!(!admission.allows(RewriteSurface::DecodeBatch));
        install_rewrite_admission(&mut admission, &eager.0, &plan, Ok(&identity)).unwrap();
        assert!(
            install_rewrite_admission(
                &mut admission,
                &eager.0,
                &plan,
                Err("identity missing".into())
            )
            .is_err()
        );
        assert_eq!(admission, RewriteAdmission::StrictPending);
    }

    #[test]
    fn mapped_library_parser_preserves_paths_and_refuses_deleted_code() {
        assert_eq!(
            mapped_library("0000-1000 r-xp 00000000 08:0a 42 /a path/libcuda.so").unwrap(),
            Some((8, 10, 42, "/a path/libcuda.so"))
        );
        assert!(mapped_library("0000-1000 r-xp 00000000 08:0a 42 /libcuda.so (deleted)").is_err());
        assert_eq!(
            mapped_library("0000-1000 r--p 00000000 08:0a 42 /weights").unwrap(),
            None
        );
        assert_eq!(
            mapped_library("0000-1000 r-xp 00000000 00:00 0 [vdso]").unwrap(),
            None
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn loaded_libraries_bind_actual_running_mappings() {
        let captured = LoadedLibraries::capture().unwrap();
        assert_eq!(captured.sha256.len(), 64);
        assert!(captured.validate().is_ok());
        let mut stale = captured;
        stale.inventory.values_mut().next().unwrap().inode += 1;
        assert!(
            stale
                .validate()
                .unwrap_err()
                .contains("loaded executable mappings changed")
        );
    }

    #[test]
    fn missing_bundle_never_preserves_legacy_permission() {
        let bundle = Bundle::new(&[RewriteSurface::DecodeEager]);
        let mut admission = RewriteAdmission::LegacyUnbundled;
        assert!(
            install_rewrite_admission(
                &mut admission,
                &bundle.0.join("missing"),
                &plan(),
                Ok(&identity())
            )
            .is_err()
        );
        assert_eq!(admission, RewriteAdmission::StrictPending);
        assert!(!admission.allows(RewriteSurface::DecodeEager));
    }
    #[test]
    fn public_output_admission_refuses_unqualified_graph_spec_and_stale_eager() {
        let bundle = Bundle::new(&[RewriteSurface::DecodeEager]);
        let mut admission = RewriteAdmission::LegacyUnbundled;
        install_rewrite_admission(&mut admission, &bundle.0, &plan(), Ok(&identity())).unwrap();
        assert!(rewrite_admission_allows(
            &admission,
            true,
            true,
            RewriteSurface::DecodeEager
        ));
        for surface in [
            RewriteSurface::DecodeGraph,
            RewriteSurface::MtpSpec,
            RewriteSurface::Glm5Spec,
        ] {
            assert!(!rewrite_admission_allows(&admission, true, true, surface));
        }
        for surface in [
            RewriteSurface::DecodeEager,
            RewriteSurface::DecodeGraph,
            RewriteSurface::MtpSpec,
            RewriteSurface::Glm5Spec,
        ] {
            assert!(!rewrite_admission_allows(&admission, false, true, surface));
            assert!(!rewrite_admission_allows(
                &RewriteAdmission::StrictPending,
                true,
                true,
                surface
            ));
        }
        assert!(!rewrite_admission_allows(
            &RewriteAdmission::LegacyUnbundled,
            false,
            true,
            RewriteSurface::DecodeEager
        ));
        assert!(rewrite_admission_allows(
            &RewriteAdmission::LegacyUnbundled,
            false,
            false,
            RewriteSurface::DecodeEager
        ));
    }
    #[test]
    fn public_output_boundaries_check_admission_before_work() {
        let decode = include_str!("../decode.rs");
        let forward = include_str!("../hybrid_forward.rs");
        let spec = include_str!("../spec.rs");
        let glm = include_str!("../glm_spec.rs");
        for (source, name, surface) in [
            (decode, "decode_step_h", "DecodeEager"),
            (decode, "decode_step_aux_inner", "DecodeEager"),
            (decode, "decode_step_dc", "DecodeEager"),
            (decode, "decode_step_dc_cap_masked", "DecodeGraph"),
            (decode, "graph_session_new", "DecodeGraph"),
            (decode, "gemma4_e4b_graph_exec_loop", "DecodeGraph"),
            (decode, "graph_session_from_cache_masked", "DecodeGraph"),
            (decode, "generate", "DecodeEager"),
            (forward, "forward", "ForwardFreshKv"),
            (forward, "forward_last", "ForwardFreshKv"),
            (forward, "prime_cache_overlaid", "DecodeEager"),
            (forward, "gemma4_generate_graph", "DecodeGraph"),
            (
                spec,
                "generate_spec_session_constrained_prime_split",
                "MtpSpec",
            ),
            (glm, "glm5_spec_session_new", "Glm5Spec"),
            (glm, "glm5_spec_session_burst_inner", "Glm5Spec"),
            (glm, "glm5_verify_rows_graphed", "Glm5Spec"),
        ] {
            let start = source.find(&format!("fn {name}(")).unwrap();
            let body = source[start..].split_once(" {\n").unwrap().1.trim_start();
            let compact = body
                .chars()
                .filter(|c| !c.is_whitespace() && *c != ',')
                .collect::<String>();
            let compact = compact
                .strip_prefix("let_rewrite_execution=self.protect_rewrite_execution()?;")
                .or_else(|| {
                    compact.strip_prefix("let_rewrite_scope=self.protect_rewrite_execution()?;")
                })
                .unwrap_or(&compact);
            assert!(compact.starts_with(&format!("self.require_rewrite(memra_gguf::execution_manifest::RewriteSurface::{surface})?;")), "{name} must check {surface} before work");
        }
    }
}
