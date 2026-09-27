use super::*;
use crate::hybrid::root_trim::{CompleteTargetTrimProof, RootTrimLoad};
use crate::{Slice, make, sha};
use memra_gguf::{
    GgufFile,
    bound_source::PreparedModelSource,
    execution_manifest::{
        RewriteAdmission, RewriteIdentity, RewriteQualifications, RewriteSurface,
        execution_rewrites,
    },
    source::{GgufSource, TensorSource},
};
use std::{
    ffi::OsString,
    os::unix::fs::FileExt,
    path::Path,
    sync::{Mutex, MutexGuard},
};
static ENV: Mutex<()> = Mutex::new(());
struct Env {
    values: Vec<(&'static str, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}
impl Env {
    fn new(path: &Path) -> Self {
        let guard = ENV.lock().unwrap_or_else(|p| p.into_inner());
        let keys = [
            "MEMRA_MTP_DRAFT",
            "MEMRA_DRAFT",
            "MEMRA_SPEC_DFLASH",
            "MEMRA_DSPARK_DRAFT",
            "MEMRA_GLM5_DFLASH",
            "MEMRA_FRSPEC_TRIM",
            "MEMRA_MTP_SKIP",
            "MEMRA_MTP_HEADS",
            "MEMRA_FULL_PREC",
            "MEMRA_FRSPEC_TRIM_NVFP4",
            "MEMRA_ARTIFACT_LOCK",
            "MEMRA_REWRITE_BUNDLE",
            "MEMRA_ROOT_TEST_NUMERIC",
        ];
        let values = keys.into_iter().map(|k| (k, std::env::var_os(k))).collect();
        // Test process runs this group serially; no background env readers are started.
        for k in keys {
            unsafe { std::env::remove_var(k) }
        }
        unsafe {
            std::env::set_var("MEMRA_FRSPEC_TRIM", path);
            std::env::set_var("MEMRA_ARTIFACT_LOCK", "cpu-protocol-fixture");
        }
        Self {
            values,
            _lock: guard,
        }
    }
    fn set(&self, key: &str, value: Option<&str>) {
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
impl Drop for Env {
    fn drop(&mut self) {
        for (k, v) in &self.values {
            unsafe {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }
}
fn with_source<T>(f: &crate::fixture::Fixture, run: impl FnOnce(&dyn TensorSource) -> T) -> T {
    let g = GgufFile::open(f.dir.join("chain.gguf")).unwrap();
    PreparedModelSource::text(&GgufSource(&g))
        .unwrap()
        .with_runtime(run)
        .unwrap()
}
fn old_tensor() -> GpuTensor {
    GpuTensor::Float {
        data: Slice {
            data: vec![0.0],
            device: 2,
            allocation: 0,
        },
        ne: vec![1, 1],
    }
}
fn head(index: u32) -> MtpHead {
    MtpHead {
        external_source_identity: None,
        embedded_block_index: Some(index),
        pending_trim_slot: None,
        geom: None,
        shared_head_head: Some(old_tensor()),
        d2t: None,
        d2t_from_target_head: false,
    }
}
fn model(e: &Engine, source: &dyn TensorSource, heads: Vec<MtpHead>, sha16: String) -> HybridModel {
    let (cfg, plan) = memra_gguf::model_packs::compile_for_source(source).unwrap();
    let mut heads = heads.into_iter();
    let generation = Arc::default();
    e.events.borrow_mut().push("constructed".into());
    HybridModel {
        program: crate::plan_backend::TrackedProgram::new(
            HybridProgram {
                cfg,
                plan,
                mtp: heads.next(),
                mtp_extra: heads.collect(),
                frspec_src_sha16: Some(sha16),
                glm5_dflash: None,
                dflash_trim: None,
            },
            Arc::clone(&generation),
        ),
        rewrite_generation: generation,
    }
}
fn staged<'a>(
    e: &Engine,
    source: &'a dyn TensorSource,
    count: usize,
) -> (RootTrimLoad<'a>, Vec<MtpHead>, String, usize) {
    let mut ranks = crate::trim_ranks::RankInput::default();
    let mut pending = RootTrimLoad::begin(source, true, true, &mut ranks)
        .unwrap()
        .unwrap();
    let (cfg, _) = memra_gguf::model_packs::compile_for_source(source).unwrap();
    let trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
    let mut heads = (0..count)
        .map(|i| head((trunk + i) as u32))
        .collect::<Vec<_>>();
    pending.note_loaded_heads(heads.len()).unwrap();
    let mut kept = 0;
    for (slot, h) in heads.iter_mut().enumerate() {
        let Some(info) = pending.stage(e, slot, h).unwrap() else {
            break;
        };
        assert!(!info.name.is_empty());
        assert_eq!(info.dtype, GgmlType::NVFP4);
        assert_eq!(info.sizes, None);
        h.d2t = Some(ranks.captured().unwrap().artifact().ids().to_vec());
        h.d2t_from_target_head = info.from_model_output;
        kept += 1;
    }
    heads.truncate(kept);
    (
        pending,
        heads,
        ranks.captured().unwrap().sha16().into(),
        trunk,
    )
}
fn complete(
    e: &Engine,
    source: &dyn TensorSource,
    count: usize,
) -> (HybridModel, CompleteTargetTrimProof, String) {
    let (pending, heads, sha16, trunk) = staged(e, source, count);
    let mut m = model(e, source, heads, sha16);
    let artifact = source.artifact_sha256().unwrap();
    let proof = pending.complete(e, &mut m, &artifact, trunk).unwrap();
    (m, proof, artifact)
}
fn allocation(t: &GpuTensor) -> usize {
    match t {
        GpuTensor::Quant { bytes, .. } => bytes.allocation,
        GpuTensor::Float { data, .. } => data.allocation,
        GpuTensor::FloatBf16 { data, .. } => data.allocation,
    }
}
fn bundle(
    path: &Path,
    plan: &memra_gguf::model_plan::ModelPlan,
    id: &RewriteIdentity,
    surfaces: &[RewriteSurface],
) {
    std::fs::create_dir_all(path.join("rewrite-receipts")).unwrap();
    let lock = b"cpu-gate-fixture; not native evidence\n";
    std::fs::write(path.join("artifact.lock"), lock).unwrap();
    let mut index = String::from("rewrite\tplan_sha256\treceipt_sha256\tstatus\n");
    for surface in surfaces {
        let rewrite = execution_rewrites(plan)
            .into_iter()
            .find(|r| r.surface == *surface)
            .unwrap();
        assert!(rewrite.eligible());
        let receipt = rewrite
            .verify_tokens(&id.implementation_sha256, &[1, 2, 3], &[1, 2, 3])
            .unwrap()
            .bind_artifact_lock(lock)
            .bind_runtime_identity(id)
            .unwrap()
            .to_tsv();
        std::fs::write(
            path.join("rewrite-receipts")
                .join(format!("{}.tsv", rewrite.id)),
            &receipt,
        )
        .unwrap();
        index.push_str(&format!(
            "{}\t{}\t{}\tpassed\n",
            rewrite.id,
            rewrite.plan_sha256,
            sha(receipt.as_bytes())
        ));
    }
    std::fs::write(path.join("rewrite-receipts.tsv"), index).unwrap();
}

#[test]
fn typed_preflight_refuses_raw_external_stub_and_missing_rank_inputs_before_uploads() {
    let f = make(GgmlType::NVFP4, None);
    let env = Env::new(&f.dir.join("ranks.txt"));
    let g = GgufFile::open(f.dir.join("chain.gguf")).unwrap();
    let mut ranks = crate::trim_ranks::RankInput::default();
    assert!(RootTrimLoad::begin(&GgufSource(&g), true, true, &mut ranks).is_err());
    with_source(&f, |source| {
        assert!(
            RootTrimLoad::begin(source, true, false, &mut ranks)
                .unwrap()
                .is_none()
        );
        for key in [
            "MEMRA_MTP_DRAFT",
            "MEMRA_DRAFT",
            "MEMRA_SPEC_DFLASH",
            "MEMRA_DSPARK_DRAFT",
            "MEMRA_GLM5_DFLASH",
        ] {
            env.set(key, Some("0"));
            assert!(
                RootTrimLoad::begin(source, true, true, &mut ranks).is_err(),
                "{key}"
            );
            env.set(key, None);
        }
        for (k, v) in [
            ("MEMRA_MTP_SKIP", "1"),
            ("MEMRA_MTP_SKIP", "invalid"),
            ("MEMRA_FULL_PREC", "1"),
        ] {
            env.set(k, Some(v));
            assert!(RootTrimLoad::begin(source, true, true, &mut ranks).is_err());
            env.set(k, None);
        }
        assert!(RootTrimLoad::begin(source, false, true, &mut ranks).is_err());
        env.set("MEMRA_FRSPEC_TRIM", Some("/missing/541/ranks.txt"));
        assert!(RootTrimLoad::begin(source, true, true, &mut ranks).is_err());
    });
}

#[test]
fn actual_private_uploads_move_into_all_constructed_slots_only_after_barrier() {
    let f = make(GgmlType::NVFP4, None);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let e = Engine::default();
        let (pending, heads, sha16, trunk) = staged(&e, source, 3);
        assert!(
            heads
                .iter()
                .all(|h| h.shared_head_head.is_none() && h.pending_trim_slot.is_some())
        );
        let mut m = model(&e, source, heads, sha16);
        let artifact = source.artifact_sha256().unwrap();
        let proof = pending.complete(&e, &mut m, &artifact, trunk).unwrap();
        assert_eq!(
            *e.events.borrow(),
            ["upload", "upload", "upload", "constructed", "barrier"]
        );
        for (i, h) in m.mtp.iter().chain(&m.mtp_extra).enumerate() {
            assert_eq!(allocation(h.shared_head_head.as_ref().unwrap()), i + 1);
            assert_eq!(
                h.shared_head_head.as_ref().unwrap().observed_bytes(),
                e.calls.borrow()[i]
            );
            assert_eq!(h.d2t, Some(vec![3, 1]));
            assert!(!h.d2t_from_target_head);
            assert!(h.pending_trim_slot.is_none());
        }
        proof.validate(&m, source, &artifact).unwrap();
        assert_ne!(proof.artifact_sha256(), artifact);
        println!("complete-loaded-artifact={}", proof.artifact_sha256());
    });
}

#[test]
fn partial_failed_reordered_or_duplicate_staging_never_completes() {
    let f = make(GgmlType::NVFP4, None);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let mut ranks = crate::trim_ranks::RankInput::default();
        let mut p = RootTrimLoad::begin(source, true, true, &mut ranks)
            .unwrap()
            .unwrap();
        assert!(p.note_loaded_heads(2).is_err());
        p.note_loaded_heads(3).unwrap();
        assert!(p.note_loaded_heads(3).is_err());
        let e = Engine::default();
        assert!(p.stage(&e, 1, &mut head(2)).is_err());
        assert!(p.stage(&e, 0, &mut head(2)).is_err());
        assert!(e.calls.borrow().is_empty());
        p.stage(&e, 0, &mut head(1)).unwrap();
        assert!(p.stage(&e, 0, &mut head(1)).is_err());
        let artifact = source.artifact_sha256().unwrap();
        let mut m = model(
            &e,
            source,
            vec![head(1), head(2), head(3)],
            ranks.captured().unwrap().sha16().into(),
        );
        assert!(p.complete(&e, &mut m, &artifact, 1).is_err());
        assert!(!e.events.borrow().iter().any(|e| e == "barrier"));
        let mut ranks = crate::trim_ranks::RankInput::default();
        let mut p = RootTrimLoad::begin(source, true, true, &mut ranks)
            .unwrap()
            .unwrap();
        p.note_loaded_heads(3).unwrap();
        let e = Engine {
            fail_on: Some(1),
            ..Engine::default()
        };
        assert!(p.stage(&e, 0, &mut head(1)).is_err());
        assert!(p.stage(&e, 0, &mut head(1)).is_err());
    });
}

#[test]
fn complete_refuses_extra_missing_reordered_substituted_mapping_and_sentinel_slots() {
    let f = make(GgmlType::NVFP4, None);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        for case in [
            "missing",
            "extra",
            "reordered",
            "tensor",
            "token",
            "map",
            "origin",
            "sentinel",
            "external",
            "stub",
        ] {
            let e = Engine::default();
            let (p, heads, sha16, trunk) = staged(&e, source, 3);
            let mut m = model(&e, source, heads, sha16);
            match case {
                "missing" => {
                    m.mtp_extra.pop();
                }
                "extra" => m.mtp_extra.push(head(4)),
                "reordered" => m.mtp_extra.swap(0, 1),
                "tensor" => m.mtp.as_mut().unwrap().shared_head_head = Some(old_tensor()),
                "token" => m.mtp.as_mut().unwrap().pending_trim_slot = None,
                "map" => m.mtp.as_mut().unwrap().d2t = Some(vec![1, 3]),
                "origin" => m.mtp.as_mut().unwrap().d2t_from_target_head = true,
                "sentinel" => m.frspec_src_sha16 = None,
                "external" => m.glm5_dflash = Some(()),
                "stub" => m.dflash_trim = Some(()),
                _ => unreachable!(),
            }
            assert!(
                p.complete(&e, &mut m, &source.artifact_sha256().unwrap(), trunk)
                    .is_err(),
                "{case}"
            );
            assert!(!e.events.borrow().iter().any(|e| e == "barrier"));
        }
    });
}

#[test]
fn failed_barrier_and_changed_source_cannot_install_or_seal() {
    let f = make(GgmlType::NVFP4, None);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let e = Engine {
            barrier_fail: true,
            ..Engine::default()
        };
        let (p, heads, sha16, trunk) = staged(&e, source, 3);
        let mut m = model(&e, source, heads, sha16);
        assert!(
            p.complete(&e, &mut m, &source.artifact_sha256().unwrap(), trunk)
                .is_err()
        );
        assert!(m.mtp.as_ref().unwrap().shared_head_head.is_none());
    });
    with_source(&f, |source| {
        let e = Engine::default();
        let (p, heads, sha16, trunk) = staged(&e, source, 3);
        let mut m = model(&e, source, heads, sha16);
        let artifact = source.artifact_sha256().unwrap();
        let path = f.dir.join("chain.gguf");
        let g = GgufFile::open(&path).unwrap();
        let t = g.find("output.weight").unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.write_all_at(&[g.tensor_data(t)[0] ^ 1], g.data_start + t.offset)
            .unwrap();
        file.sync_all().unwrap();
        assert!(p.complete(&e, &mut m, &artifact, trunk).is_err());
        assert!(m.mtp.as_ref().unwrap().shared_head_head.is_none());
    });
}

#[test]
fn proof_is_specific_to_model_generation_and_mutation_revokes_real_snapshot() {
    let f = make(GgmlType::NVFP4, None);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let (mut m, proof, artifact) = complete(&Engine::default(), source, 3);
        let (other, _, _) = complete(&Engine::default(), source, 3);
        assert!(proof.validate(&other, source, &artifact).is_err());
        let snapshot = crate::plan_backend::RewriteExecutionSnapshot::validated(
            &m.rewrite_generation,
            &RewriteAdmission::StrictPending,
            false,
            || Ok(()),
        )
        .unwrap();
        assert!(snapshot.current(&m.rewrite_generation));
        proof.validate(&m, source, &artifact).unwrap();
        m.mtp.as_mut().unwrap().d2t = Some(vec![3, 1]);
        assert!(proof.validate(&m, source, &artifact).is_err());
        assert!(!snapshot.current(&m.rewrite_generation));
    });
}

#[test]
fn explicit_caps_and_known_missing_private_heads_keep_existing_truncation_policy() {
    let f = make(GgmlType::NVFP4, None);
    let env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        env.set("MEMRA_MTP_HEADS", Some("1"));
        let (m, p, a) = complete(&Engine::default(), source, 1);
        assert!(m.mtp_extra.is_empty());
        p.validate(&m, source, &a).unwrap();
        env.set("MEMRA_MTP_HEADS", Some("2"));
        let mut r = crate::trim_ranks::RankInput::default();
        assert!(RootTrimLoad::begin(source, true, true, &mut r).is_err());
        env.set("MEMRA_MTP_HEADS", None);
    });
    // The fixture variant omits all private rows of the first extra head; the real root keeps
    // only the first head after observing that missing private projection.
    crate::remove_second_private_head(&f);
    with_source(&f, |source| {
        let (m, p, a) = complete(&Engine::default(), source, 3);
        assert!(m.mtp_extra.is_empty());
        p.validate(&m, source, &a).unwrap();
    });
}

#[test]
fn real_capture_and_qualification_require_exact_pair_binary_numeric_and_explicit_spec() {
    let f = make(GgmlType::NVFP4, None);
    let env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let (m, p, a) = complete(&Engine::default(), source, 3);
        let (id, state) =
            crate::plan_backend::capture_target_trim_rewrite_identity(&m, source, a.clone(), p)
                .unwrap();
        state.validate(&m).unwrap();
        assert_ne!(id.artifact_sha256, a);
        let own_exe = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        assert_eq!(id.implementation_sha256, sha(&own_exe));
        let path = f.dir.join("qualified");
        bundle(
            &path,
            &m.plan,
            &id,
            &[RewriteSurface::DecodeEager, RewriteSurface::DecodeGraph],
        );
        let mut admission = RewriteAdmission::StrictPending;
        crate::plan_backend::install_rewrite_admission(&mut admission, &path, &m.plan, Ok(&id))
            .unwrap();
        assert!(admission.allows(RewriteSurface::DecodeEager));
        assert!(admission.allows(RewriteSurface::DecodeGraph));
        assert!(!admission.allows(RewriteSurface::MtpSpec));
        bundle(
            &path,
            &m.plan,
            &id,
            &[RewriteSurface::DecodeEager, RewriteSurface::MtpSpec],
        );
        let q = RewriteQualifications::load(&path, &m.plan, &id).unwrap();
        assert!(q.allows(RewriteSurface::MtpSpec));
        for field in ["target", "binary", "numeric"] {
            let mut other = id.clone();
            match field {
                "target" => other.artifact_sha256 = a.clone(),
                "binary" => other.implementation_sha256 = "b".repeat(64),
                "numeric" => other.numeric_program_sha256 = "c".repeat(64),
                _ => unreachable!(),
            };
            assert!(RewriteQualifications::load(&path, &m.plan, &other).is_err());
        }
        env.set("MEMRA_ROOT_TEST_NUMERIC", Some("changed"));
        assert!(state.validate(&m).is_err());
        env.set("MEMRA_ROOT_TEST_NUMERIC", None);
        println!("cpu-provider-paired-runtime={id:?}");
    });
}

#[test]
fn target_only_and_different_rank_pairs_never_reuse_the_qualified_identity() {
    let f = make(GgmlType::NVFP4, None);
    let env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let (m, p, a) = complete(&Engine::default(), source, 3);
        let (id, _) =
            crate::plan_backend::capture_target_trim_rewrite_identity(&m, source, a, p).unwrap();
        let path = f.dir.join("pair");
        bundle(
            &path,
            &m.plan,
            &id,
            &[RewriteSurface::DecodeEager, RewriteSurface::MtpSpec],
        );
        let rp = f.dir.join("other.txt");
        std::fs::write(&rp, "1\n3\n").unwrap();
        env.set("MEMRA_FRSPEC_TRIM", Some(rp.to_str().unwrap()));
        let (other, p, a) = complete(&Engine::default(), source, 3);
        let (other_id, _) =
            crate::plan_backend::capture_target_trim_rewrite_identity(&other, source, a, p)
                .unwrap();
        assert_ne!(id.artifact_sha256, other_id.artifact_sha256);
        assert!(RewriteQualifications::load(&path, &other.plan, &other_id).is_err());
    });
}

#[test]
fn eager_eligible_model_without_embedded_spec_program_refuses_typed_exception() {
    let f = make(GgmlType::NVFP4, None);
    crate::remove_embedded_program(&f);
    let _env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let (_, plan) = memra_gguf::model_packs::compile_for_source(source).unwrap();
        let rewrites = execution_rewrites(&plan);
        assert!(
            rewrites
                .iter()
                .find(|r| r.surface == RewriteSurface::DecodeEager)
                .unwrap()
                .eligible()
        );
        assert!(
            !rewrites
                .iter()
                .find(|r| r.surface == RewriteSurface::MtpSpec)
                .unwrap()
                .eligible()
        );
        let mut ranks = crate::trim_ranks::RankInput::default();
        assert!(RootTrimLoad::begin(source, true, true, &mut ranks).is_err());
        assert!(ranks.captured().is_none());
    });
}

#[test]
fn changed_preparation_options_refuse_before_the_model_barrier() {
    let f = make(GgmlType::NVFP4, None);
    let env = Env::new(&f.dir.join("ranks.txt"));
    with_source(&f, |source| {
        let e = Engine::default();
        let (pending, heads, sha16, trunk) = staged(&e, source, 3);
        let mut m = model(&e, source, heads, sha16);
        env.set("MEMRA_MTP_HEADS", Some("1"));
        assert!(
            pending
                .complete(&e, &mut m, &source.artifact_sha256().unwrap(), trunk)
                .is_err()
        );
        assert!(!e.events.borrow().iter().any(|e| e == "barrier"));
        assert!(m.mtp.as_ref().unwrap().shared_head_head.is_none());
    });
}
