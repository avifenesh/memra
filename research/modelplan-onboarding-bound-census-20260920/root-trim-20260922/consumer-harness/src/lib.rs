#![cfg(test)]
use memra_gguf::{
    GgmlType, GgufFile, MetaValue,
    micro_gguf::{GgufWriter, MetaW},
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
    allocation: usize,
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
    barrier_fail: bool,
    events: RefCell<Vec<String>>,
    short: bool,
    wrong_device: bool,
}
impl Default for Engine {
    fn default() -> Self {
        Self {
            context: Context { device: 2 },
            calls: RefCell::new(vec![]),
            fail_on: None,
            barrier_fail: false,
            events: RefCell::new(Vec::new()),
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
        self.events.borrow_mut().push("upload".into());
        self.calls.borrow_mut().push(bytes);
        if self.fail_on == Some(self.calls.borrow().len()) {
            return Err("injected H2D failure".into());
        }
        let mut data = values.to_vec();
        if self.short {
            data.pop();
        }
        Ok(Slice {
            allocation: self.calls.borrow().len(),
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
    pub fn full_prec_enabled() -> bool {
        std::env::var("MEMRA_FULL_PREC").as_deref() == Ok("1")
    }
    include!(concat!(env!("OUT_DIR"), "/quant-upload.rs"));
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
mod hybrid;
mod plan_backend;
mod pp {
    pub fn sync_stages_after_load(
        e: &crate::Engine,
        _: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        e.events.borrow_mut().push("barrier".into());
        if e.barrier_fail {
            Err("injected visibility barrier failure".into())
        } else {
            Ok(())
        }
    }
}

fn remove_second_private_head(f: &fixture::Fixture) {
    let path = f.dir.join("chain.gguf");
    let g = GgufFile::open(&path).unwrap();
    let mut w = GgufWriter::new();
    for (key, value) in &g.metadata {
        let value = match value {
            MetaValue::U32(v) => MetaW::U32(*v),
            MetaValue::F32(v) => MetaW::F32(*v),
            MetaValue::Bool(v) => MetaW::Bool(*v),
            MetaValue::String(v) if v == "qwen3" => MetaW::Str("qwen3"),
            _ => panic!("unexpected fixture metadata"),
        };
        w.kv(key, value);
    }
    for t in &g.tensors {
        if !t.name.starts_with("blk.2.nextn.shared_head_head.") {
            w.tensor_raw(&t.name, &t.ne, t.ggml_type, g.tensor_data(t).to_vec());
        }
    }
    let next = f.dir.join("next.gguf");
    w.write(&next).unwrap();
    std::fs::rename(next, path).unwrap();
}

fn remove_embedded_program(f: &fixture::Fixture) {
    let path = f.dir.join("chain.gguf");
    let g = GgufFile::open(&path).unwrap();
    let mut w = GgufWriter::new();
    for (key, value) in &g.metadata {
        let value = match value {
            MetaValue::U32(v) => MetaW::U32(if key == "qwen3.nextn_predict_layers" {
                0
            } else {
                *v
            }),
            MetaValue::F32(v) => MetaW::F32(*v),
            MetaValue::Bool(v) => MetaW::Bool(*v),
            MetaValue::String(v) if v == "qwen3" => MetaW::Str("qwen3"),
            _ => panic!("unexpected fixture metadata"),
        };
        w.kv(key, value);
    }
    for t in &g.tensors {
        if !t.name.contains(".nextn.") {
            w.tensor_raw(&t.name, &t.ne, t.ggml_type, g.tensor_data(t).to_vec());
        }
    }
    let next = f.dir.join("plain.gguf");
    w.write(&next).unwrap();
    std::fs::rename(next, path).unwrap();
}
