#![cfg(test)]
use memra_gguf::{
    GgmlType, GgufFile, MetaValue,
    bound_source::{
        BoundArtifactIdentity, PreparedModelSource,
        draft::{
            DraftGeometry, DraftVocabulary, PreparedExternalDraftSource,
            composite::CompositeExternalDraftInput,
        },
        draft_pair::{DraftProposalIdentity, PreparedDraftTarget},
        head_trim::{HeadChoice, TrimPolicy},
        ranks::RankArtifact,
    },
    config::ModelConfig,
    micro_gguf::{GgufWriter, MetaW},
    source::{GgufSource, TensorSource},
    tensor_contract::{MtpTensor, TensorId},
};
use std::{cell::Cell, collections::BTreeMap, os::unix::fs::FileExt, path::Path};
#[path = "../../../draft-repair-20260921/consumer-harness/src/fixture.rs"]
mod draft_fixture;
#[path = "../../../../../crates/memra-engine/src/head_trim.rs"]
mod head_trim;
#[path = "../../../../../crates/memra-engine/src/model/nvfp4_scale.rs"]
mod nvfp4_scale;
#[path = "../../../head-trim-20260921/consumer-harness/src/fixture.rs"]
mod target_fixture;
#[path = "../../../../../crates/memra-engine/src/trim_ranks.rs"]
mod trim_ranks;

#[derive(Default)]
struct Engine {
    fail: bool,
}
impl Engine {
    fn htod_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bytes.to_vec())
    }
    fn htod(&self, bytes: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(bytes.to_vec())
    }
}
#[derive(Debug)]
struct MtpHead {
    external: BoundArtifactIdentity,
    bytes: Vec<u8>,
    scale: f32,
    map: Option<Vec<u32>>,
}
impl MtpHead {
    // The native allocation body is a recorder. The two entry methods below are compiled
    // verbatim from hybrid.rs; preparation, identity, bound reads and scale consumer are real.
    fn load_prepared_draft(
        e: &Engine,
        source: &dyn TensorSource,
        prepared: &PreparedExternalDraftSource<'_>,
        cfg: &ModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if e.fail {
            return Err("injected native allocation failure".into());
        }
        assert_eq!(cfg.n_embd, prepared.config().n_embd);
        Ok(Self {
            external: prepared.artifact_identity()?,
            bytes: source
                .try_find(prepared.head_name())?
                .unwrap()
                .bytes
                .into_owned(),
            scale: nvfp4_scale::load(source, prepared.head_name())?,
            map: prepared.token_map().map(<[u32]>::to_vec),
        })
    }
}
include!(concat!(env!("OUT_DIR"), "/paired-entry.rs"));
mod model {
    use super::*;
    #[derive(Debug)]
    pub enum GpuTensor {
        FloatBf16 {
            data: Vec<u8>,
            ne: Vec<u64>,
        },
        Float {
            data: Vec<f32>,
            ne: Vec<u64>,
        },
        Quant {
            bytes: Vec<u8>,
            dtype: GgmlType,
            ne: Vec<u64>,
            scale: f32,
        },
    }
    impl GpuTensor {
        pub fn from_quant_bytes(
            _: &Engine,
            bytes: &[u8],
            dtype: GgmlType,
            n: u64,
            m: u64,
            scale: f32,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self::Quant {
                bytes: bytes.to_vec(),
                dtype,
                ne: vec![n, m],
                scale,
            })
        }
    }
}
#[derive(Clone)]
struct Row {
    dtype: GgmlType,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}
fn rows(g: &GgufFile) -> BTreeMap<String, Row> {
    g.tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                Row {
                    dtype: t.ggml_type,
                    shape: t.ne.clone(),
                    bytes: g.tensor_data(t).to_vec(),
                },
            )
        })
        .collect()
}
fn write(g: &GgufFile, path: &Path, rows: &BTreeMap<String, Row>, vocab: Option<u32>) {
    let mut w = GgufWriter::new();
    for (key, value) in &g.metadata {
        let value = match value {
            MetaValue::U32(v) => MetaW::U32(if key == "qwen3.vocab_size" {
                vocab.unwrap_or(*v)
            } else {
                *v
            }),
            MetaValue::F32(v) => MetaW::F32(*v),
            MetaValue::Bool(v) => MetaW::Bool(*v),
            MetaValue::String(v) if v == "qwen3" => MetaW::Str("qwen3"),
            _ => panic!("unexpected fixture metadata {key}"),
        };
        w.kv(key, value);
    }
    for (name, r) in rows {
        w.tensor_raw(name, &r.shape, r.dtype, r.bytes.clone());
    }
    w.write(path).unwrap();
}
fn target16(width: u32) -> target_fixture::Fixture {
    let f = target_fixture::make(GgmlType::F32, width, false, None, GgmlType::F32);
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let mut data = rows(&g);
    for (name, r) in &mut data {
        if name == "token_embd.weight"
            || name == "output.weight"
            || name.ends_with("shared_head_head.weight")
        {
            r.shape[1] = 16;
            r.bytes = r.bytes.repeat(4);
        }
    }
    write(&g, &f.dir.join("target.gguf"), &data, Some(16));
    f
}
fn mutate(g: &GgufFile, path: &Path, name: &str) {
    let t = g.find(name).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    let b = [g.tensor_data(t)[0] ^ 1];
    file.write_all_at(&b, g.data_start + t.offset).unwrap();
    file.sync_all().unwrap();
}
fn check_upload(trim: &memra_gguf::bound_source::head_trim::PreparedHeadTrim) {
    let (gpu, sizes) = head_trim::load(&Engine::default(), trim).unwrap();
    assert_eq!(sizes, trim.requant_sizes());
    match gpu {
        model::GpuTensor::FloatBf16 { data, ne } => {
            assert_eq!(data, trim.bytes());
            assert_eq!(ne, trim.shape());
        }
        model::GpuTensor::Float { data, ne } => {
            assert_eq!(
                data.into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
                trim.bytes()
            );
            assert_eq!(ne, trim.shape());
        }
        model::GpuTensor::Quant {
            bytes,
            dtype,
            ne,
            scale,
        } => {
            assert_eq!(bytes, trim.bytes());
            assert_eq!(dtype, trim.dtype());
            assert_eq!(ne, trim.shape());
            assert_eq!(scale.to_bits(), trim.macro_scale().to_bits());
        }
    }
}

#[test]
fn actual_paired_engine_entry_keeps_target_and_external_sources_separate() {
    let t = target16(64);
    let d = draft_fixture::make(64, false, Some(GgmlType::F32), false);
    let tg = GgufFile::open(t.dir.join("target.gguf")).unwrap();
    let dg = GgufFile::open(d.0.join("draft.gguf")).unwrap();
    assert!(MtpHead::load_paired_draft(&Engine::default(), &GgufSource(&tg), &dg).is_err());
    let (loaded, id) = PreparedModelSource::text(&GgufSource(&tg))
        .unwrap()
        .with_runtime(|source| MtpHead::load_paired_draft(&Engine::default(), source, &dg))
        .unwrap()
        .unwrap();
    assert_eq!(
        id.target().opened_source_sha256,
        GgufSource(&tg).artifact_sha256().unwrap()
    );
    let DraftProposalIdentity::External(external) = id.proposal() else {
        panic!("external role")
    };
    assert_eq!(external.artifact(), &loaded.external);
    assert_eq!(
        external.artifact().opened_source_sha256,
        GgufSource(&dg).artifact_sha256().unwrap()
    );
    assert_ne!(id.target(), external.artifact());
    assert!(external.components().is_none());
    assert_eq!(
        external.head_role(),
        &TensorId::Mtp {
            depth: 0,
            tensor: MtpTensor::OutputProjection
        }
    );
    assert_eq!(external.head_name(), "blk.2.nextn.shared_head_head.weight");
    assert_eq!(
        external.norm_name(),
        Some("blk.2.nextn.shared_head_norm.weight")
    );
    assert_eq!(external.declared_blocks().len(), 1);
    assert_eq!(external.selected_block().geometry, DraftGeometry::Natural);
    assert_eq!(loaded.bytes, vec![0x22; 16 * 36]);
    assert_eq!(loaded.scale, 2.0);
    assert_eq!(loaded.map, None);
    assert!(matches!(
        external.vocabulary(),
        DraftVocabulary::Full { target_size: 16 }
    ));
    println!("paired-natural={id:?}");
}

#[test]
fn composite_pair_binds_shadowed_components_and_uses_actual_engine_entry() {
    let t = target16(64);
    let d = draft_fixture::make(64, false, Some(GgmlType::F32), false);
    let tg = GgufFile::open(t.dir.join("target.gguf")).unwrap();
    let base = d.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let mut top = rows(&g);
    top.retain(|name, _| name.starts_with("blk.2.nextn.shared_head_head."));
    top.get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .bytes
        .fill(0x33);
    top.get_mut("blk.2.nextn.shared_head_head.scale")
        .unwrap()
        .bytes = 3.0f32.to_le_bytes().to_vec();
    let upper = d.0.join("upper.gguf");
    write(&g, &upper, &top, None);
    let changed = d.0.join("changed.gguf");
    let mut lower = rows(&g);
    lower
        .get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .bytes
        .fill(0x44);
    write(&g, &changed, &lower, None);
    let a = CompositeExternalDraftInput::open(&[&upper, &base]).unwrap();
    let b = CompositeExternalDraftInput::open(&[&upper, &changed]).unwrap();
    PreparedModelSource::text(&GgufSource(&tg))
        .unwrap()
        .with_runtime(|source| {
            let (la, ia) =
                MtpHead::load_paired_composite_draft(&Engine::default(), source, &a).unwrap();
            let (lb, ib) =
                MtpHead::load_paired_composite_draft(&Engine::default(), source, &b).unwrap();
            assert_eq!(la.bytes, lb.bytes);
            assert_eq!(la.bytes, vec![0x33; 16 * 36]);
            assert_eq!(la.scale, 3.0);
            assert_eq!(ia.target(), ib.target());
            assert_ne!(ia.source_pair_sha256(), ib.source_pair_sha256());
            let (DraftProposalIdentity::External(ea), DraftProposalIdentity::External(eb)) =
                (ia.proposal(), ib.proposal())
            else {
                panic!("external role")
            };
            assert_eq!(
                ea.artifact().semantic_scope_sha256,
                eb.artifact().semantic_scope_sha256
            );
            assert_ne!(
                ea.artifact().opened_source_sha256,
                eb.artifact().opened_source_sha256
            );
            assert_eq!(ea.components().unwrap(), &a.source_identity().unwrap());
            assert_eq!(ea.components().unwrap().components().len(), 2);
            println!(
                "paired-composite-a={} paired-composite-b={}",
                ia.source_pair_sha256(),
                ib.source_pair_sha256()
            );
        })
        .unwrap();
}

#[test]
fn paired_student_and_trimmed_external_programs_keep_exact_mapping() {
    for (student, file_head) in [(false, false), (false, true), (true, false), (true, true)] {
        let width = if student { 128 } else { 64 };
        let t = target16(width);
        let d = draft_fixture::make(width, student, Some(GgmlType::F32), false);
        let tg = GgufFile::open(t.dir.join("target.gguf")).unwrap();
        let g = GgufFile::open(d.0.join("draft.gguf")).unwrap();
        let mut data = rows(&g);
        let h = data.get_mut("blk.2.nextn.shared_head_head.weight").unwrap();
        h.shape[1] = 3;
        h.bytes.truncate(h.bytes.len() / 16 * 3);
        if file_head {
            let head = data.remove("blk.2.nextn.shared_head_head.weight").unwrap();
            data.insert("output.weight".into(), head);
            if let Some(scale) = data.remove("blk.2.nextn.shared_head_head.scale") {
                data.insert("output.scale".into(), scale);
            }
        }
        data.insert(
            "d2t".into(),
            Row {
                dtype: GgmlType::I32,
                shape: vec![3],
                bytes: [9i32, 2, 11]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            },
        );
        let path = d.0.join("trim.gguf");
        write(&g, &path, &data, None);
        let trimmed = GgufFile::open(&path).unwrap();
        PreparedModelSource::text(&GgufSource(&tg))
            .unwrap()
            .with_runtime(|source| {
                let (loaded, id) =
                    MtpHead::load_paired_draft(&Engine::default(), source, &trimmed).unwrap();
                assert_eq!(loaded.map, Some(vec![9, 2, 11]));
                let DraftProposalIdentity::External(ext) = id.proposal() else {
                    panic!("external role")
                };
                assert_eq!(
                    ext.head_role(),
                    &if file_head {
                        TensorId::OutputProjection
                    } else {
                        TensorId::Mtp {
                            depth: 0,
                            tensor: MtpTensor::OutputProjection,
                        }
                    }
                );
                assert_eq!(
                    matches!(
                        ext.selected_block().geometry,
                        DraftGeometry::Student {
                            hidden_size: 64,
                            ..
                        }
                    ),
                    student
                );
                assert_eq!(
                    ext.vocabulary(),
                    &DraftVocabulary::Trimmed {
                        target_size: 16,
                        token_ids: vec![9, 2, 11]
                    }
                );
                let prepared = PreparedDraftTarget::bind(source).unwrap();
                let (_, checked) = prepared
                    .with_external(&trimmed, |p, s| {
                        if student {
                            assert_eq!(
                                nvfp4_scale::load(s, "blk.2.nextn.out_up.weight").unwrap(),
                                2.0
                            );
                        }
                        assert_eq!(p.token_map(), Some(&[9, 2, 11][..]));
                        assert!(PreparedDraftTarget::bind(s).is_err());
                    })
                    .unwrap();
                assert_eq!(id, checked);
            })
            .unwrap();
    }
}

#[test]
fn target_trim_pair_retains_roles_macros_ranks_and_actual_host_upload_inputs() {
    for dtype in [GgmlType::NVFP4, GgmlType::BF16, GgmlType::F32] {
        let f = target_fixture::make(dtype, 64, false, None, GgmlType::F32);
        let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
        let mut rank_input = trim_ranks::RankInput::default();
        let captured = rank_input
            .capture(f.dir.join("ranks.txt").to_str().unwrap())
            .unwrap();
        assert_eq!(
            captured.sha16(),
            &captured.artifact().identity().source_sha256()[..16]
        );
        assert!(captured.path().ends_with("ranks.txt"));
        let ranks = head_trim::extra_head_ranks(&rank_input, 1, Some(&[3, 1]))
            .unwrap()
            .unwrap();
        PreparedModelSource::text(&GgufSource(&g))
            .unwrap()
            .with_runtime(|source| {
                let target = PreparedDraftTarget::bind(source).unwrap();
                for (choice, owner, scale) in [
                    (HeadChoice::ModelOutput, 0, 2.0),
                    (HeadChoice::FirstMtpOrModel, 1, 3.0),
                    (HeadChoice::MtpBlock { index: 2 }, 2, 5.0),
                ] {
                    let pair = target
                        .prepare_head_trim(ranks, choice, TrimPolicy::Preserve)
                        .unwrap()
                        .unwrap();
                    let trim = pair.materialization();
                    let DraftProposalIdentity::TargetHeadTrim(receipt) = pair.identity().proposal()
                    else {
                        panic!("trim role")
                    };
                    assert_eq!(receipt.as_ref(), trim.identity());
                    assert_eq!(pair.identity().target(), target.identity());
                    assert_eq!(receipt.rank(), ranks.identity());
                    assert_eq!(trim.from_model_output(), owner == 0);
                    assert_eq!(trim.ids(), &[3, 1]);
                    let expected = &f.heads[trim.runtime_name()];
                    let row = expected.len() / 4;
                    assert_eq!(
                        trim.bytes(),
                        trim_ranks::gather_rows(expected, row, &[3, 1])
                    );
                    assert_eq!(
                        trim.macro_scale(),
                        if dtype == GgmlType::NVFP4 { scale } else { 1.0 }
                    );
                    check_upload(trim);
                    if owner == 0 {
                        println!("paired-trim={:?}", pair.identity());
                    }
                }
                let converted = target
                    .prepare_head_trim(
                        ranks,
                        HeadChoice::ModelOutput,
                        TrimPolicy::Nvfp4ForEligibleBf16,
                    )
                    .unwrap()
                    .unwrap();
                check_upload(converted.materialization());
                assert!(
                    target
                        .prepare_head_trim(
                            ranks,
                            HeadChoice::MtpBlock { index: 99 },
                            TrimPolicy::Preserve
                        )
                        .is_err()
                );
            })
            .unwrap();
    }
}

#[test]
fn whole_target_and_rank_encoding_remain_distinct_from_selected_materialization() {
    let f = target_fixture::make(GgmlType::F32, 64, false, None, GgmlType::F32);
    let path = f.dir.join("model.gguf");
    let g = GgufFile::open(&path).unwrap();
    let text = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    let mut rank_inputs = vec![text.clone()];
    for dtype in [GgmlType::I32, GgmlType::I64] {
        let p = f.dir.join(format!("ranks-{dtype:?}.gguf"));
        let mut w = GgufWriter::new();
        let b = if dtype == GgmlType::I32 {
            [3i32, 1].into_iter().flat_map(i32::to_le_bytes).collect()
        } else {
            [3i64, 1].into_iter().flat_map(i64::to_le_bytes).collect()
        };
        w.tensor_raw("d2t", &[2], dtype, b);
        w.write(&p).unwrap();
        rank_inputs.push(RankArtifact::open(&p).unwrap());
    }
    let mut data = rows(&g);
    data.get_mut("blk.0.attn_norm.weight").unwrap().bytes[0] ^= 1;
    let changed = f.dir.join("changed.gguf");
    write(&g, &changed, &data, None);
    let h = GgufFile::open(&changed).unwrap();
    let get = |file: &GgufFile, ranks: &RankArtifact| {
        PreparedModelSource::text(&GgufSource(file))
            .unwrap()
            .with_runtime(|s| {
                PreparedDraftTarget::bind(s).unwrap().prepare_head_trim(
                    ranks,
                    HeadChoice::ModelOutput,
                    TrimPolicy::Preserve,
                )
            })
            .unwrap()
            .unwrap()
            .unwrap()
    };
    let a = get(&g, &text);
    let b = get(&h, &text);
    assert_eq!(
        a.materialization().identity(),
        b.materialization().identity()
    );
    assert_ne!(
        a.identity().source_pair_sha256(),
        b.identity().source_pair_sha256()
    );
    for ranks in &rank_inputs[1..] {
        let p = get(&g, ranks);
        assert_eq!(p.materialization().bytes(), a.materialization().bytes());
        assert_eq!(
            ranks.identity().order_sha256(),
            text.identity().order_sha256()
        );
        assert_ne!(
            ranks.identity().source_sha256(),
            text.identity().source_sha256()
        );
        assert_ne!(
            p.identity().source_pair_sha256(),
            a.identity().source_pair_sha256()
        );
    }
    std::fs::write(f.dir.join("ranks.txt"), "1\n3\n").unwrap();
    assert_eq!(get(&g, &text).identity(), a.identity());
    let changed_ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    assert_ne!(get(&g, &changed_ranks).identity(), a.identity());
}

#[test]
fn drift_checks_refuse_changed_opened_sources_and_path_replacement_keeps_ownership() {
    for during in [
        "before-target",
        "during-target",
        "during-draft",
        "replace-paths",
    ] {
        let t = target16(64);
        let d = draft_fixture::make(64, false, Some(GgmlType::F32), false);
        let tp = t.dir.join("target.gguf");
        let dp = d.0.join("draft.gguf");
        let tg = GgufFile::open(&tp).unwrap();
        let dg = GgufFile::open(&dp).unwrap();
        PreparedModelSource::text(&GgufSource(&tg))
            .unwrap()
            .with_runtime(|source| {
                let target = PreparedDraftTarget::bind(source).unwrap();
                let called = Cell::new(false);
                if during == "before-target" {
                    mutate(&tg, &tp, "output.weight");
                }
                if during == "replace-paths" {
                    std::fs::rename(&tp, t.dir.join("opened-target.gguf")).unwrap();
                    std::fs::write(&tp, b"replacement target").unwrap();
                    std::fs::rename(&dp, d.0.join("opened-draft.gguf")).unwrap();
                    std::fs::write(&dp, b"replacement draft").unwrap();
                }
                let result = target.with_external(&dg, |_, _| {
                    called.set(true);
                    match during {
                        "during-target" => mutate(&tg, &tp, "output.weight"),
                        "during-draft" => mutate(&dg, &dp, "blk.2.nextn.shared_head_head.weight"),
                        _ => {}
                    }
                });
                if during == "replace-paths" {
                    assert!(result.is_ok());
                } else {
                    assert!(result.unwrap_err().contains("changed"));
                }
                assert_eq!(called.get(), during != "before-target");
            })
            .unwrap();
    }
}

#[test]
fn paired_load_propagates_consumer_failure_and_target_contract_refusal() {
    let t = target16(64);
    let d = draft_fixture::make(64, false, Some(GgmlType::F32), false);
    let tg = GgufFile::open(t.dir.join("target.gguf")).unwrap();
    let dg = GgufFile::open(d.0.join("draft.gguf")).unwrap();
    PreparedModelSource::text(&GgufSource(&tg))
        .unwrap()
        .with_runtime(|source| {
            assert!(
                MtpHead::load_paired_draft(&Engine { fail: true }, source, &dg)
                    .unwrap_err()
                    .to_string()
                    .contains("allocation failure")
            );
            let bad = draft_fixture::make(128, true, Some(GgmlType::F32), false);
            let g = GgufFile::open(bad.0.join("draft.gguf")).unwrap();
            let called = Cell::new(false);
            assert!(
                PreparedDraftTarget::bind(source)
                    .unwrap()
                    .with_external(&g, |_, _| called.set(true))
                    .is_err()
            );
            assert!(!called.get());
        })
        .unwrap();
}
