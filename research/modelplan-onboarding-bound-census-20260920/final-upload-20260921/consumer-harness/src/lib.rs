#![cfg(test)]
use memra_gguf::{
    GgmlType, GgufFile, MetaValue,
    bound_source::{
        PreparedModelSource,
        draft_pair::{PreparedDraftTarget, PreparedPairedHeadTrim},
        head_trim::{HeadChoice, TrimPolicy},
        ranks::RankArtifact,
    },
    micro_gguf::{GgufWriter, MetaW},
    source::GgufSource,
};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap};
#[path = "../../../../../crates/memra-engine/src/model/final_upload.rs"]
pub mod final_upload;
#[path = "../../../head-trim-20260921/consumer-harness/src/fixture.rs"]
mod fixture;
#[allow(dead_code)] // allow: only the unchanged load dispatch is exercised in this focused harness
#[path = "../../../../../crates/memra-engine/src/head_trim.rs"]
mod head_trim;
#[allow(dead_code)] // allow: dependency of the unchanged dispatch module; ranks are tested through the source API
#[path = "../../../../../crates/memra-engine/src/trim_ranks.rs"]
mod trim_ranks;
use model::{GpuTensor, quant_type_for_trim};
const QT_Q8_0: i32 = 0;
const QT_Q4_K: i32 = 1;
const QT_Q6_K: i32 = 2;
const QT_Q5_K: i32 = 3;
const QT_Q3_K: i32 = 4;
const QT_IQ4_XS: i32 = 5;
const QT_IQ3_S: i32 = 6;
const QT_NVFP4: i32 = 7;
const QT_Q4_0: i32 = 12;
#[derive(Debug)]
pub struct Slice<T> {
    data: Vec<T>,
    device: usize,
}
impl<T> Slice<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
#[derive(Debug)]
pub struct Context {
    device: usize,
}
impl Context {
    pub fn ordinal(&self) -> usize {
        self.device
    }
}
#[derive(Debug)]
pub struct Engine {
    context: Context,
    calls: RefCell<Vec<Vec<u8>>>,
    fail_on: Option<usize>,
    short: bool,
    wrong_device: bool,
}
impl Default for Engine {
    fn default() -> Self {
        Self {
            context: Context { device: 2 },
            calls: RefCell::new(vec![]),
            fail_on: None,
            short: false,
            wrong_device: false,
        }
    }
}
impl Engine {
    pub fn ctx(&self) -> &Context {
        &self.context
    }
    fn copy<T: Clone>(
        &self,
        values: &[T],
        bytes: Vec<u8>,
    ) -> Result<Slice<T>, Box<dyn std::error::Error>> {
        self.calls.borrow_mut().push(bytes);
        if self.fail_on == Some(self.calls.borrow().len()) {
            return Err("injected H2D failure".into());
        }
        let mut data = values.to_vec();
        if self.short {
            data.pop();
        }
        Ok(Slice {
            data,
            device: if self.wrong_device {
                9
            } else {
                self.context.device
            },
        })
    }
    pub fn htod_bytes(&self, bytes: &[u8]) -> Result<Slice<u8>, Box<dyn std::error::Error>> {
        self.copy(bytes, bytes.to_vec())
    }
    pub fn htod(&self, values: &[f32]) -> Result<Slice<f32>, Box<dyn std::error::Error>> {
        self.copy(
            values,
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        )
    }
}
pub mod model {
    use super::*;
    pub(crate) use crate::final_upload;
    #[derive(Debug)]
    pub enum GpuTensor {
        Quant {
            bytes: Slice<u8>,
            qtype: i32,
            row_bytes: usize,
            ne: Vec<u64>,
            scale: f32,
            rp: bool,
            fp8: Option<()>,
            rp4: Option<()>,
            blk: Option<()>,
            f16: Option<()>,
            a4: Option<()>,
        },
        Float {
            data: Slice<f32>,
            ne: Vec<u64>,
        },
        FloatBf16 {
            data: Slice<u8>,
            ne: Vec<u64>,
        },
    }
    impl GpuTensor {
        pub fn ordinal(&self) -> usize {
            match self {
                Self::Quant { bytes, .. } => bytes.device,
                Self::Float { data, .. } => data.device,
                Self::FloatBf16 { data, .. } => data.device,
            }
        }
        pub fn observed_bytes(&self) -> Vec<u8> {
            match self {
                Self::Quant { bytes, .. } => bytes.data.clone(),
                Self::FloatBf16 { data, .. } => data.data.clone(),
                Self::Float { data, .. } => {
                    data.data.iter().flat_map(|v| v.to_ne_bytes()).collect()
                }
            }
        }
    }
    include!(concat!(env!("OUT_DIR"), "/quant-upload.rs"));
    include!("baseline_quant_upload.rs");
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
#[derive(Clone)]
struct Row {
    shape: Vec<u64>,
    dtype: GgmlType,
    bytes: Vec<u8>,
}
fn make(dtype: GgmlType, first_scale: Option<f32>) -> fixture::Fixture {
    let f = fixture::make(dtype, 256, false, None, GgmlType::F32);
    assert!(!f.heads.is_empty());
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let mut rows = g
        .tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                Row {
                    shape: t.ne.clone(),
                    dtype: t.ggml_type,
                    bytes: g.tensor_data(t).to_vec(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let extra = rows
        .iter()
        .filter(|(name, _)| name.starts_with("blk.2."))
        .map(|(name, row)| (name.replacen("blk.2.", "blk.3.", 1), row.clone()))
        .collect::<Vec<_>>();
    rows.extend(extra);
    if dtype == GgmlType::NVFP4 {
        rows.get_mut("blk.3.nextn.shared_head_head.scale")
            .unwrap()
            .bytes = 7.0f32.to_le_bytes().to_vec();
        if let Some(scale) = first_scale {
            rows.get_mut("blk.1.nextn.shared_head_head.scale")
                .unwrap()
                .bytes = scale.to_le_bytes().to_vec();
        }
    }
    let mut w = GgufWriter::new();
    for (key, value) in &g.metadata {
        let value = match value {
            MetaValue::U32(v) => MetaW::U32(match key.as_str() {
                "qwen3.block_count" => 4,
                "qwen3.nextn_predict_layers" => 3,
                _ => *v,
            }),
            MetaValue::F32(v) => MetaW::F32(*v),
            MetaValue::Bool(v) => MetaW::Bool(*v),
            MetaValue::String(v) if v == "qwen3" => MetaW::Str("qwen3"),
            _ => panic!("unexpected fixture metadata"),
        };
        w.kv(key, value);
    }
    for (name, row) in rows {
        w.tensor_raw(&name, &row.shape, row.dtype, row.bytes);
    }
    w.write(&f.dir.join("chain.gguf")).unwrap();
    f
}
fn prepare(
    f: &fixture::Fixture,
    choices: &[HeadChoice],
    rank_path: Option<&std::path::Path>,
) -> Vec<PreparedPairedHeadTrim> {
    let g = GgufFile::open(f.dir.join("chain.gguf")).unwrap();
    let path = f.dir.join("ranks.txt");
    let ranks = RankArtifact::open(rank_path.unwrap_or(&path)).unwrap();
    PreparedModelSource::text(&GgufSource(&g))
        .unwrap()
        .with_runtime(|source| {
            let target = PreparedDraftTarget::bind(source).unwrap();
            assert_eq!(target.config().n_embd, 256);
            choices
                .iter()
                .map(|&choice| {
                    target
                        .prepare_head_trim(&ranks, choice, TrimPolicy::Preserve)
                        .unwrap()
                        .unwrap()
                })
                .collect()
        })
        .unwrap()
}
const CHOICES: [HeadChoice; 3] = [
    HeadChoice::FirstMtpOrModel,
    HeadChoice::MtpBlock { index: 2 },
    HeadChoice::MtpBlock { index: 3 },
];
fn expected_nvfp4(host: &[u8]) -> Vec<u8> {
    if !model::rp_enabled() {
        return host.to_vec();
    }
    let mut result = vec![];
    for block in host.chunks_exact(36) {
        result.extend_from_slice(&block[4..]);
    }
    for block in host.chunks_exact(36) {
        result.extend_from_slice(&block[..4]);
    }
    result
}

#[test]
fn all_ordered_heads_bind_actual_submitted_buffers_shapes_macros_and_layouts() {
    let f = make(GgmlType::NVFP4, None);
    let prepared = prepare(&f, &CHOICES, None);
    let e = Engine::default();
    let chain = final_upload::UploadedTrimChain::load(&e, &prepared).unwrap();
    assert_eq!(chain.heads().len(), 3);
    for (slot, head) in chain.heads().iter().enumerate() {
        let trim = prepared[slot].materialization();
        let expected = expected_nvfp4(trim.bytes());
        let upload = head.upload();
        assert_eq!(e.calls.borrow()[slot], expected);
        assert_eq!(head.tensor().observed_bytes(), expected);
        assert_eq!(upload.host_sha256(), trim.identity().output_sha256());
        assert_eq!(upload.submitted_sha256(), sha(&expected));
        assert_eq!(upload.byte_len(), expected.len());
        assert_eq!(upload.dtype(), GgmlType::NVFP4);
        assert_eq!(upload.shape(), [256, 2]);
        assert_eq!(upload.device(), 2);
        assert_eq!(upload.macro_bits(), [3.0f32, 5.0, 7.0][slot].to_bits());
        assert_eq!(head.source(), prepared[slot].identity());
        assert_eq!(
            matches!(
                upload.layout(),
                final_upload::UploadLayout::Nvfp4SplitA6 { .. }
            ),
            model::rp_enabled()
        );
        if model::rp_enabled() {
            assert_ne!(upload.host_sha256(), upload.submitted_sha256());
        } else {
            assert_eq!(upload.host_sha256(), upload.submitted_sha256());
        }
        println!(
            "slot={slot} upload={upload:?} source-submitted={}",
            head.sha256()
        );
    }
    println!("ordered-chain={}", chain.sha256());
}

#[test]
fn ordinary_quant_constructor_and_recorded_path_keep_every_supported_encoding() {
    let types = [
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
        GgmlType::Q5_K,
        GgmlType::Q3_K,
        GgmlType::IQ4_XS,
        GgmlType::IQ3_S,
        GgmlType::NVFP4,
        GgmlType::Q4_0,
    ];
    for (dtype, qtype) in types.into_iter().zip([0, 1, 2, 3, 4, 5, 6, 7, 12]) {
        let bytes = (0..dtype.checked_nbytes(&[256, 2]).unwrap())
            .map(|x| (x % 251) as u8)
            .collect::<Vec<_>>();
        let e = Engine::default();
        let baseline =
            GpuTensor::from_quant_bytes_baseline(&e, &bytes, dtype, 256, 2, 3.0).unwrap();
        let ordinary = GpuTensor::from_quant_bytes(&e, &bytes, dtype, 256, 2, 3.0).unwrap();
        let (recorded, id) =
            GpuTensor::from_quant_bytes_recorded(&e, &bytes, dtype, 256, 2, 3.0).unwrap();
        assert_eq!(e.calls.borrow()[0], e.calls.borrow()[1]);
        assert_eq!(e.calls.borrow()[1], e.calls.borrow()[2]);
        assert_eq!(baseline.observed_bytes(), ordinary.observed_bytes());
        assert_eq!(ordinary.observed_bytes(), recorded.observed_bytes());
        let GpuTensor::Quant {
            qtype: actual,
            scale,
            ..
        } = recorded
        else {
            panic!("quant constructor changed kind")
        };
        assert_eq!(actual, qtype);
        assert_eq!(scale, 3.0);
        assert_eq!(id.submitted_sha256(), sha(&e.calls.borrow()[2]));
        assert_eq!(id.host_sha256(), sha(&bytes));
    }
}

#[test]
fn float_paths_hash_the_same_typed_or_byte_buffer_used_by_h2d() {
    for dtype in [GgmlType::F32, GgmlType::BF16] {
        let f = make(dtype, None);
        let p = prepare(&f, &CHOICES[..1], None);
        let e = Engine::default();
        let head = final_upload::UploadedHeadTrim::load(&e, &p[0]).unwrap();
        assert_eq!(e.calls.borrow().len(), 1);
        assert_eq!(head.tensor().observed_bytes(), e.calls.borrow()[0]);
        assert_eq!(head.upload().submitted_sha256(), sha(&e.calls.borrow()[0]));
        assert_eq!(head.upload().macro_bits(), 1.0f32.to_bits());
        assert_eq!(
            head.upload().layout(),
            &if dtype == GgmlType::F32 {
                final_upload::UploadLayout::F32Native
            } else {
                final_upload::UploadLayout::Bf16LittleEndian
            }
        );
        let (legacy, _) = head_trim::load(&e, p[0].materialization()).unwrap();
        assert_eq!(legacy.observed_bytes(), head.tensor().observed_bytes());
    }
}

#[test]
fn malformed_byte_shape_dtype_contracts_refuse_before_backend_copy() {
    for (bytes, dtype, n, m) in [
        (vec![], GgmlType::NVFP4, 64, 0),
        (vec![0; 36], GgmlType::NVFP4, 32, 2),
        (vec![0; 35], GgmlType::NVFP4, 64, 1),
        (vec![0; 36], GgmlType::NVFP4, u64::MAX, 2),
        (vec![0; 2], GgmlType::F16, 1, 1),
    ] {
        let e = Engine::default();
        assert!(GpuTensor::from_quant_bytes_recorded(&e, &bytes, dtype, n, m, 1.0).is_err());
        assert!(e.calls.borrow().is_empty());
    }
}

#[test]
fn chain_checks_target_rank_and_slot_order_before_any_upload() {
    let f = make(GgmlType::NVFP4, None);
    let e = Engine::default();
    for choices in [
        vec![],
        vec![CHOICES[1], CHOICES[0]],
        vec![CHOICES[0], CHOICES[0]],
        vec![CHOICES[0], CHOICES[2]],
    ] {
        assert!(final_upload::UploadedTrimChain::load(&e, &prepare(&f, &choices, None)).is_err());
        assert!(e.calls.borrow().is_empty());
    }
    let other = make(GgmlType::NVFP4, Some(11.0));
    let mut mixed = prepare(&f, &CHOICES[..1], None);
    mixed.extend(prepare(&other, &CHOICES[1..2], None));
    assert!(final_upload::UploadedTrimChain::load(&e, &mixed).is_err());
    let path = f.dir.join("other-ranks.txt");
    std::fs::write(&path, "1\n3\n").unwrap();
    let mut mixed = prepare(&f, &CHOICES[..1], None);
    mixed.extend(prepare(&f, &CHOICES[1..2], Some(&path)));
    assert!(final_upload::UploadedTrimChain::load(&e, &mixed).is_err());
    assert!(e.calls.borrow().is_empty());
}

#[test]
fn failed_copy_or_wrong_returned_extent_device_produces_no_opaque_upload() {
    let f = make(GgmlType::NVFP4, None);
    let prepared = prepare(&f, &CHOICES, None);
    let e = Engine {
        fail_on: Some(2),
        ..Engine::default()
    };
    assert!(final_upload::UploadedTrimChain::load(&e, &prepared).is_err());
    assert_eq!(e.calls.borrow().len(), 2);
    for e in [
        Engine {
            short: true,
            ..Engine::default()
        },
        Engine {
            wrong_device: true,
            ..Engine::default()
        },
    ] {
        assert!(final_upload::UploadedHeadTrim::load(&e, &prepared[0]).is_err());
        assert_eq!(e.calls.borrow().len(), 1);
    }
}

#[test]
fn equal_submitted_bytes_with_different_macro_have_distinct_upload_identity() {
    let a = make(GgmlType::NVFP4, Some(3.0));
    let b = make(GgmlType::NVFP4, Some(11.0));
    let pa = prepare(&a, &CHOICES[..1], None);
    let pb = prepare(&b, &CHOICES[..1], None);
    let e = Engine::default();
    let a = final_upload::UploadedHeadTrim::load(&e, &pa[0]).unwrap();
    let b = final_upload::UploadedHeadTrim::load(&e, &pb[0]).unwrap();
    assert_eq!(a.upload().submitted_sha256(), b.upload().submitted_sha256());
    assert_ne!(a.upload().macro_bits(), b.upload().macro_bits());
    assert_ne!(a.upload().sha256(), b.upload().sha256());
    assert_ne!(a.sha256(), b.sha256());
}

#[test]
fn receipt_rejects_wrong_mode_shape_scale_extent_mirrors_and_submitted_bytes() {
    let host = (0..288).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    for case in [
        "qtype", "shape", "macro", "extent", "row", "layout", "mirror", "buffer",
    ] {
        let e = Engine::default();
        let (mut tensor, _) =
            GpuTensor::from_quant_bytes_recorded(&e, &host, GgmlType::NVFP4, 256, 2, 3.0).unwrap();
        let mut submitted = e.calls.borrow()[0].clone();
        let GpuTensor::Quant {
            qtype,
            ne,
            scale,
            bytes,
            row_bytes,
            rp,
            fp8,
            ..
        } = &mut tensor
        else {
            unreachable!()
        };
        match case {
            "qtype" => *qtype = QT_Q8_0,
            "shape" => ne[0] = 512,
            "macro" => *scale = 5.0,
            "extent" => {
                bytes.data.pop();
            }
            "row" => *row_bytes += 1,
            "layout" => *rp = !*rp,
            "mirror" => *fp8 = Some(()),
            "buffer" => submitted[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(
            final_upload::FinalUploadIdentity::after_upload(
                &tensor,
                &host,
                &submitted,
                GgmlType::NVFP4,
                [256, 2],
                3.0
            )
            .is_err(),
            "{case}"
        );
    }
}
