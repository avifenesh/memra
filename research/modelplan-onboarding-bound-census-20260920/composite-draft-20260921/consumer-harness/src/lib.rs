#![cfg(test)]
use memra_gguf::{
    GgmlType, GgufFile, MetaValue,
    bound_source::{
        PreparedModelSource,
        draft::{PreparedExternalDraftSource, composite::CompositeExternalDraftInput},
        head_trim::{HeadChoice, PreparedHeadTrim, TrimPolicy},
        ranks::RankArtifact,
    },
    micro_gguf::{GgufWriter, MetaW},
    source::{GgufSource, TensorSource},
};
use std::{cell::Cell, collections::BTreeMap, path::Path};
#[path = "../../../draft-repair-20260921/consumer-harness/src/fixture.rs"]
mod fixture;
#[path = "../../../../../crates/memra-engine/src/model/nvfp4_scale.rs"]
mod nvfp4_scale;

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
fn writer(g: &GgufFile, rows: &BTreeMap<String, Row>, heads: Option<u32>) -> GgufWriter {
    let mut writer = GgufWriter::new();
    for (key, value) in &g.metadata {
        let value = match value {
            MetaValue::U32(v) => MetaW::U32(if key == "qwen3.attention.head_count" {
                heads.unwrap_or(*v)
            } else {
                *v
            }),
            MetaValue::F32(v) => MetaW::F32(*v),
            MetaValue::Bool(v) => MetaW::Bool(*v),
            MetaValue::String(v) if v == "qwen3" => MetaW::Str("qwen3"),
            _ => panic!("fixture metadata is not an expected existing GGUF value: {key}"),
        };
        writer.kv(key, value);
    }
    for (name, row) in rows {
        writer.tensor_raw(name, &row.shape, row.dtype, row.bytes.clone());
    }
    writer
}
fn write(g: &GgufFile, path: &Path, rows: &BTreeMap<String, Row>, heads: Option<u32>) {
    writer(g, rows, heads).write(path).unwrap();
}
fn f32row(shape: Vec<u64>, value: f32) -> Row {
    Row {
        bytes: value
            .to_le_bytes()
            .repeat(shape.iter().product::<u64>() as usize),
        dtype: GgmlType::F32,
        shape,
    }
}
fn head_rows(value: u8, scale: Option<f32>) -> BTreeMap<String, Row> {
    let mut rows = BTreeMap::from([(
        "blk.2.nextn.shared_head_head.weight".into(),
        Row {
            dtype: GgmlType::NVFP4,
            shape: vec![64, 16],
            bytes: vec![value; 16 * 36],
        },
    )]);
    if let Some(scale) = scale {
        rows.insert(
            "blk.2.nextn.shared_head_head.scale".into(),
            f32row(vec![1], scale),
        );
    }
    rows
}
fn input_error(
    input: &CompositeExternalDraftInput,
    target: &memra_gguf::config::ModelConfig,
) -> String {
    let called = Cell::new(false);
    let result = input.with_runtime(target, |_, _| called.set(true));
    assert!(!called.get());
    result.unwrap_err()
}

#[test]
fn selected_composite_draft_uses_bound_components_and_actual_scale_consumer() {
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    write(&g, &upper, &head_rows(0x33, Some(3.0)), None);
    let input = CompositeExternalDraftInput::open(&[&upper, &base]).unwrap();
    let whole = input.source_identity().unwrap();
    assert_eq!(whole.components().len(), 2);
    assert_eq!(
        whole.components()[0],
        GgufSource(&GgufFile::open(&upper).unwrap())
            .artifact_sha256()
            .unwrap()
    );
    assert_eq!(input.physical_inventory().components.len(), 2);
    assert!(
        input.physical_inventory().components[1]
            .census
            .tensors
            .iter()
            .any(|r| r.entry.name == "blk.2.nextn.shared_head_head.weight")
    );
    let disk = input
        .with_runtime(&target, |prepared, source| {
            let head = prepared.head_name();
            assert_eq!(head, "blk.2.nextn.shared_head_head.weight");
            let view = source.try_find(head).unwrap().unwrap();
            assert_eq!(view.ne, [64, 16]);
            assert_eq!(view.bytes.as_ref(), vec![0x33; 16 * 36]);
            assert!(source.try_find_nvfp4_native(head).unwrap().is_none());
            assert_eq!(nvfp4_scale::load(source, head).unwrap(), 3.0);
            let lower = source.try_find("blk.2.attn_q.weight").unwrap().unwrap();
            assert_eq!(
                lower.bytes,
                GgufSource(&g).find("blk.2.attn_q.weight").unwrap().bytes
            );
            let identity = prepared.artifact_identity().unwrap();
            assert_eq!(identity.opened_source_sha256, whole.opened_source_sha256());
            assert_eq!(source.artifact_sha256().unwrap(), identity.artifact_sha256);
            assert!(PreparedModelSource::text(source).is_err());
            let rp = f.0.join("ranks.txt");
            std::fs::write(&rp, "3\n1\n").unwrap();
            let ranks = RankArtifact::open(&rp).unwrap();
            assert!(
                PreparedHeadTrim::prepare(
                    source,
                    &ranks,
                    HeadChoice::ModelOutput,
                    TrimPolicy::Preserve
                )
                .is_err()
            );
            assert!(source.try_find("blk.99.attn_q.weight").is_err());
            assert!(source.try_find("blk.2.unknown.weight").is_err());
            source.try_find_gguf_disk(head).unwrap().unwrap()
        })
        .unwrap();
    drop(input);
    assert_eq!(disk.bytes(), vec![0x33; 16 * 36]);
}

#[test]
fn whole_source_identity_binds_shadowed_data_order_and_opened_files() {
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    write(&g, &upper, &head_rows(0x33, Some(3.0)), None);
    let changed = f.0.join("changed.gguf");
    let mut changed_rows = rows(&g);
    changed_rows
        .get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .bytes
        .fill(0x44);
    write(&g, &changed, &changed_rows, None);
    let a = CompositeExternalDraftInput::open(&[&upper, &base]).unwrap();
    let b = CompositeExternalDraftInput::open(&[&upper, &changed]).unwrap();
    let aid = a.source_identity().unwrap();
    let bid = b.source_identity().unwrap();
    assert_ne!(aid, bid);
    let value = |input: &CompositeExternalDraftInput| {
        input
            .with_runtime(&target, |p, s| {
                (
                    p.artifact_identity().unwrap(),
                    s.try_find(p.head_name())
                        .unwrap()
                        .unwrap()
                        .bytes
                        .into_owned(),
                )
            })
            .unwrap()
    };
    let (ia, va) = value(&a);
    let (ib, vb) = value(&b);
    assert_eq!(va, vb);
    assert_eq!(ia.semantic_scope_sha256, ib.semantic_scope_sha256);
    assert_ne!(ia.artifact_sha256, ib.artifact_sha256);
    println!(
        "whole-shadowed-a={aid:?}\nwhole-shadowed-b={bid:?}\nbinding-a={ia:?}\nbinding-b={ib:?}"
    );
    let reversed = CompositeExternalDraftInput::open(&[&base, &upper]).unwrap();
    assert_ne!(reversed.source_identity().unwrap(), aid);
    std::fs::rename(&upper, f.0.join("opened-upper.gguf")).unwrap();
    write(&g, &upper, &head_rows(0x55, Some(5.0)), None);
    assert_eq!(a.source_identity().unwrap(), aid);
    assert_eq!(value(&a).1, va);
    let fresh = CompositeExternalDraftInput::open(&[&upper, &base]).unwrap();
    assert_ne!(fresh.source_identity().unwrap(), aid);
    assert_ne!(value(&fresh).1, va);
}

#[test]
fn replacement_weights_never_inherit_lower_macros() {
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    write(&g, &upper, &head_rows(0x33, None), None);
    let input = CompositeExternalDraftInput::open(&[&upper, &base]).unwrap();
    input
        .with_runtime(&target, |prepared, source| {
            assert_eq!(
                nvfp4_scale::load(source, prepared.head_name()).unwrap(),
                1.0
            );
            assert!(
                source
                    .try_find("blk.2.nextn.shared_head_head.scale")
                    .unwrap()
                    .is_none()
            );
        })
        .unwrap();
    write(
        &g,
        &upper,
        &BTreeMap::from([(
            "blk.2.nextn.shared_head_head.scale".into(),
            f32row(vec![1], 5.0),
        )]),
        None,
    );
    assert!(
        input_error(
            &CompositeExternalDraftInput::open(&[&upper, &base]).unwrap(),
            &target
        )
        .contains("component-owned weight")
    );
}

#[test]
fn shadowed_bad_metadata_and_incompatible_programs_refuse_before_callback() {
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    write(&g, &upper, &head_rows(0x33, Some(3.0)), None);
    for case in ["shape", "unknown", "missing", "config", "alias"] {
        let path = f.0.join("bad.gguf");
        let mut data = rows(&g);
        match case {
            "shape" => {
                data.insert(
                    "blk.2.nextn.shared_head_head.weight".into(),
                    f32row(vec![64, 15], 0.25),
                );
            }
            "unknown" => {
                data.insert("unexpected.weight".into(), f32row(vec![1], 0.25));
            }
            "missing" => {
                data.remove("blk.2.attn_q.weight");
            }
            "alias" => {
                let value = data.remove("blk.2.nextn.shared_head_head.weight").unwrap();
                data.insert("blk.2.nextn.shared_head.weight".into(), value);
                data.remove("blk.2.nextn.shared_head_head.scale");
            }
            _ => {}
        }
        write(&g, &path, &data, (case == "config").then_some(4));
        assert!(
            !input_error(
                &CompositeExternalDraftInput::open(&[&upper, &path]).unwrap(),
                &target
            )
            .is_empty(),
            "{case}"
        );
    }
}

#[test]
fn selected_head_owns_its_map_and_all_metadata_precedes_map_payload() {
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let lower = f.0.join("lower.gguf");
    let upper = f.0.join("upper.gguf");
    let mut lower_rows = rows(&g);
    lower_rows
        .get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .shape = vec![64, 3];
    lower_rows
        .get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .bytes
        .truncate(3 * 36);
    let map = |ids: &[i32]| Row {
        dtype: GgmlType::I32,
        shape: vec![ids.len() as u64],
        bytes: ids.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    lower_rows.insert("d2t".into(), map(&[11, 2, 9]));
    write(&g, &lower, &lower_rows, None);
    let mut top = head_rows(0x33, Some(3.0));
    top.get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .shape = vec![64, 3];
    top.get_mut("blk.2.nextn.shared_head_head.weight")
        .unwrap()
        .bytes
        .truncate(3 * 36);
    top.insert("d2t".into(), map(&[9, 2, 11]));
    write(&g, &upper, &top, None);
    CompositeExternalDraftInput::open(&[&upper, &lower])
        .unwrap()
        .with_runtime(&target, |p, _| {
            assert_eq!(p.token_map(), Some(&[9, 2, 11][..]))
        })
        .unwrap();
    top.remove("d2t");
    write(&g, &upper, &top, None);
    assert!(
        !input_error(
            &CompositeExternalDraftInput::open(&[&upper, &lower]).unwrap(),
            &target
        )
        .is_empty()
    );
    top.insert("d2t".into(), map(&[-1, 2, 9]));
    write(&g, &upper, &top, None);
    lower_rows.insert("unexpected.weight".into(), f32row(vec![1], 0.25));
    write(&g, &lower, &lower_rows, None);
    let error = input_error(
        &CompositeExternalDraftInput::open(&[&upper, &lower]).unwrap(),
        &target,
    );
    assert!(error.contains("extra checkpoint tensors"), "{error}");
    assert!(
        !error.contains("negative or exceeds"),
        "metadata must precede selected map reads"
    );
}

#[test]
fn student_composite_keeps_geometry_and_component_limit_refusals() {
    let f = fixture::make(128, true, Some(GgmlType::F32), false);
    let base = f.0.join("draft.gguf");
    let g = GgufFile::open(&base).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    let mut data = rows(&g);
    data.retain(|name, _| name.starts_with("blk.2.nextn.out_up"));
    data.get_mut("blk.2.nextn.out_up.scale").unwrap().bytes = 5.0f32.to_le_bytes().to_vec();
    write(&g, &upper, &data, None);
    CompositeExternalDraftInput::open(&[&upper, &base])
        .unwrap()
        .with_runtime(&target, |p, s| {
            assert!(matches!(
                p.blocks()[0].geometry,
                memra_gguf::bound_source::draft::DraftGeometry::Student {
                    hidden_size: 64,
                    ..
                }
            ));
            assert_eq!(
                nvfp4_scale::load(s, "blk.2.nextn.out_up.weight").unwrap(),
                5.0
            );
        })
        .unwrap();
    assert!(CompositeExternalDraftInput::open(&[&base]).is_err());
    assert!(CompositeExternalDraftInput::open(&vec![&base; 65]).is_err());
    // The ordinary reviewed entry still accepts its original one-source class.
    assert!(PreparedExternalDraftSource::compile(&GgufSource(&g), &target).is_ok());
}

#[test]
fn split_component_identity_covers_shadowed_shard_and_unused_bytes() {
    use std::io::Write;
    let f = fixture::make(64, false, Some(GgmlType::F32), false);
    let g = GgufFile::open(f.0.join("draft.gguf")).unwrap();
    let target = GgufSource(&g).config();
    let upper = f.0.join("upper.gguf");
    write(&g, &upper, &head_rows(0x33, Some(3.0)), None);
    let all = rows(&g);
    let parts: [BTreeMap<_, _>; 2] = std::array::from_fn(|i| {
        all.iter()
            .filter(|(name, _)| usize::from(name.starts_with("blk.2.nextn.shared_head_head.")) == i)
            .map(|(name, row)| (name.clone(), row.clone()))
            .collect()
    });
    let split = |prefix: &str| {
        std::array::from_fn::<_, 2, _>(|i| {
            let path = f.0.join(format!("{prefix}-{:05}-of-00002.gguf", i + 1));
            let mut w = writer(&g, &parts[i], None);
            w.kv("split.no", MetaW::U32(i as u32));
            w.kv("split.count", MetaW::U32(2));
            w.kv("split.tensors.count", MetaW::U32(all.len() as u32));
            w.write(&path).unwrap();
            path
        })
    };
    let a_paths = split("a");
    let b_paths = split("b");
    std::fs::OpenOptions::new()
        .append(true)
        .open(&b_paths[1])
        .unwrap()
        .write_all(b"unused artifact annex")
        .unwrap();
    let a = CompositeExternalDraftInput::open(&[&upper, &a_paths[0]]).unwrap();
    let same = CompositeExternalDraftInput::open(&[&upper, &a_paths[1]]).unwrap();
    let b = CompositeExternalDraftInput::open(&[&upper, &b_paths[0]]).unwrap();
    let ia = a.source_identity().unwrap();
    let ib = b.source_identity().unwrap();
    assert_eq!(ia, same.source_identity().unwrap());
    assert_ne!(ia.components()[1], ib.components()[1]);
    assert_eq!(
        std::fs::read(&a_paths[0]).unwrap(),
        std::fs::read(&b_paths[0]).unwrap()
    );
    let selected = |input: &CompositeExternalDraftInput| {
        input
            .with_runtime(&target, |p, s| {
                (
                    p.artifact_identity().unwrap(),
                    s.try_find(p.head_name())
                        .unwrap()
                        .unwrap()
                        .bytes
                        .into_owned(),
                    nvfp4_scale::load(s, p.head_name()).unwrap(),
                )
            })
            .unwrap()
    };
    let (pa, va, ma) = selected(&a);
    let (pb, vb, mb) = selected(&b);
    assert_eq!((va, ma), (vb, mb));
    assert_eq!(pa.semantic_scope_sha256, pb.semantic_scope_sha256);
    assert_ne!(pa.artifact_sha256, pb.artifact_sha256);
    println!(
        "whole-split-a={ia:?}\nwhole-split-b={ib:?}\nbinding-split-a={pa:?}\nbinding-split-b={pb:?}"
    );
}
