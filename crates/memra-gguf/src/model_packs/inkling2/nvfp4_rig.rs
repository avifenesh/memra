//! The `nvfp4-rig` numeric class of Inkling 2 Mini: the weights this class converts to NVFP4, the
//! quantizer that converts them, and the overlay that stores the result next to the shipped
//! checkpoint.
//!
//! The class (coordinator decision 8, 2026-11-21, from the per-layer KL receipts of the lane's
//! sensitivity study): the declared activation-quant experts, FP8 e4m3 KV on every cache slot, and
//! NVFP4 weights for (a) the MXFP8 expert banks (the layers whose experts the checkpoint stores in
//! MXFP8), (b) the output projection (`unembed`) and (c) the six attention projections (q, k, v,
//! o, output gate, relative-bias projection) of every physical layer outside the loop core. The
//! loop core's attention stays BF16; the shipped program stays the named class `shipped`.
//!
//! The quantizer is the study's own (`klsens/klh.py` `nvfp4_weight_quant`), ported operation for
//! operation and proven byte-equal on its goldens: e2m1 codes (two per byte, the even element in
//! the low nibble), one e4m3 scale per 16 consecutive inputs of a row and one f32 tensor scale
//! `t = amax / (6 * 448)` (per expert for an expert bank). Each block picks its scale from 16
//! candidates `e4m3(min(block_amax * r / 6 / t, 448))`, `r` in [`NV_RATIOS`], by the smallest
//! sum of squared error of `code * fl32(s * t)` against the source; the first candidate wins a
//! tie. The source values are exact: BF16 widened, or MXFP8 `e4m3 * 2^(e - 127)` in f32.
//!
//! The overlay stores an expert bank in the checkpoint's own NVFP4 expert layout (`*_weight` U8
//! codes, `*_weight_scale` F8_E4M3, `*_weight_amax` F32 per expert and
//! `*_weight_outer_scale_denominator` F32 = 2688), so a minted bank is read by the same loaders
//! and kernels as a shipped NVFP4 bank: its factor `amax / denominator` is the quantizer's `t`
//! exactly (an f32 quotient computed in f64 and rounded once is the f32 quotient). A dense tensor
//! takes the modelopt spelling: `*.weight` U8 codes, `*.weight_scale` F8_E4M3 and
//! `*.weight_scale_2` F32 scalar = `t`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::config::Inkling2Config;

/// The overlay format id written into the manifest.
pub const OVERLAY_FORMAT: &str = "memra-inkling2-nvfp4-rig-overlay-v1";
/// The class name a plan, a manifest and a receipt use.
pub const CLASS_NAME: &str = "nvfp4-rig";
/// The manifest's file name inside an overlay directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// The overlay tensor file inside an overlay directory.
pub const OVERLAY_FILE: &str = "overlay.safetensors";
/// The quantizer's id: the method below, one version.
pub const QUANTIZER: &str = "nvfp4-mse16-v1";
/// The outer-scale denominator: the largest e2m1 code times the largest e4m3 scale.
pub const DENOMINATOR: f32 = 6.0 * 448.0;
/// Block-scale candidates, as ratios of the block maximum to the largest code (`klh.NV_RATIOS`).
pub const NV_RATIOS: [f64; 16] = [
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
];

// ---- the quantizer ----------------------------------------------------------------------------

/// e4m3 magnitude of byte `b` (0..=126; the bytes increase with the value).
fn e4m3_magnitude(b: u8) -> f64 {
    let e = (b >> 3) as i32;
    let m = (b & 7) as f64;
    if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

fn e4m3_table() -> &'static [f64; 127] {
    static T: std::sync::OnceLock<[f64; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|b| e4m3_magnitude(b as u8)))
}

/// The f32 value of a positive e4m3 scale byte.
pub fn e4m3_scale_value(b: u8) -> f32 {
    e4m3_table()[b as usize] as f32
}

/// The byte (0..=126) of the e4m3 value nearest `v` in [0, 448], ties to the even byte
/// (`klh.e4m3_rne_idx`: a left search in the 127 increasing values, clamped to 1..=126).
pub fn e4m3_rne_index(v: f32) -> u8 {
    let p = e4m3_table();
    let v = v as f64;
    let hi = p.partition_point(|&x| x < v).clamp(1, 126);
    let lo = hi - 1;
    let (da, db) = (v - p[lo], p[hi] - v);
    if db < da || (db == da && hi % 2 == 0) {
        hi as u8
    } else {
        lo as u8
    }
}

/// e2m1 round to nearest, ties to the even code, saturating at 6 (`klh.e2m1_rne`). The sign of
/// the input is kept, so a negative input that rounds to zero is -0.
pub fn e2m1_rne(x: f32) -> f32 {
    let a = x.abs();
    let m: f32 = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    };
    m.copysign(x)
}

/// The nibble of an e2m1 value (sign in bit 3).
pub fn e2m1_nibble(code: f32) -> u8 {
    let m = code.abs();
    let index = if m <= 2.0 {
        (m * 2.0) as u8
    } else if m == 3.0 {
        5
    } else if m == 4.0 {
        6
    } else {
        7
    };
    index | ((code.is_sign_negative() as u8) << 3)
}

/// The e2m1 value of a nibble.
pub fn e2m1_value(nibble: u8) -> f32 {
    const V: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = V[(nibble & 7) as usize];
    if nibble & 8 != 0 { -v } else { v }
}

/// The tensor scale of a tensor (or one expert) whose largest magnitude is `amax`:
/// `max(fl32(amax / 2688), 1e-30)`.
pub fn tensor_scale(amax: f32) -> f32 {
    (amax / DENOMINATOR).max(1e-30)
}

/// Largest magnitude of a slice.
pub fn amax(w: &[f32]) -> f32 {
    w.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// One block of 16 inputs: writes 8 code bytes and returns the scale byte.
pub fn quantize_block(w: &[f32], t: f32, codes: &mut [u8]) -> u8 {
    debug_assert_eq!(w.len(), 16);
    debug_assert_eq!(codes.len(), 8);
    let bmax = amax(w);
    let mut best: Option<(f64, u8, [f32; 16])> = None;
    for r in NV_RATIOS {
        let candidate = ((bmax * (r / 6.0) as f32) / t).min(448.0);
        let si = e4m3_rne_index(candidate);
        let st = e4m3_scale_value(si) * t;
        let div = st.max(1e-38);
        let mut code = [0.0f32; 16];
        let mut err = 0.0f64;
        for j in 0..16 {
            code[j] = e2m1_rne(w[j] / div);
            let d = (code[j] * st - w[j]) as f64;
            err += d * d;
        }
        if best.as_ref().is_none_or(|b| err < b.0) {
            best = Some((err, si, code));
        }
    }
    let (_, si, code) = best.expect("16 candidates");
    for (i, byte) in codes.iter_mut().enumerate() {
        *byte = e2m1_nibble(code[2 * i]) | (e2m1_nibble(code[2 * i + 1]) << 4);
    }
    si
}

/// Rows of `k` inputs with one tensor scale: `codes` gets `k / 2` bytes per row and `scales`
/// `k / 16`. Runs on `threads` threads; the bytes do not depend on the thread count.
pub fn quantize_rows(
    w: &[f32],
    k: usize,
    t: f32,
    codes: &mut [u8],
    scales: &mut [u8],
    threads: usize,
) {
    assert!(k.is_multiple_of(16) && w.len().is_multiple_of(k));
    let rows = w.len() / k;
    assert_eq!(codes.len(), rows * k / 2);
    assert_eq!(scales.len(), rows * k / 16);
    let per = rows.div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for ((wc, cc), sc) in w
            .chunks(per * k)
            .zip(codes.chunks_mut(per * k / 2))
            .zip(scales.chunks_mut(per * k / 16))
        {
            s.spawn(move || {
                for (b, (wb, cb)) in wc.chunks_exact(16).zip(cc.chunks_exact_mut(8)).enumerate() {
                    sc[b] = quantize_block(wb, t, cb);
                }
            });
        }
    });
}

/// Dequantized value of one element: `fl32(code * fl32(s * t))`.
pub fn dequant_rows(codes: &[u8], scales: &[u8], k: usize, t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(codes.len() * 2);
    for (row, row_scales) in codes.chunks_exact(k / 2).zip(scales.chunks_exact(k / 16)) {
        for (j, byte) in row.iter().enumerate() {
            let st = e4m3_scale_value(row_scales[j / 8]) * t;
            out.push(e2m1_value(byte & 15) * st);
            out.push(e2m1_value(byte >> 4) * st);
        }
    }
    out
}

/// f32 values of BF16 bytes (exact).
pub fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// f32 value of an e4m3 byte (signed; the NaN codes 0x7f/0xff are NaN).
pub fn e4m3_value(b: u8) -> f32 {
    if b & 0x7f == 0x7f {
        return f32::NAN;
    }
    let v = e4m3_magnitude(b & 0x7f) as f32;
    if b & 0x80 != 0 { -v } else { v }
}

/// `2^(e - 127)` as an f32 (the ue8m0 scale; 0xFF is NaN and refused by the caller).
fn ue8m0(e: u8) -> f32 {
    2f64.powi(e as i32 - 127) as f32
}

/// Exact f32 values of MXFP8 rows: `e4m3 * 2^(e - 127)` per block of 32 inputs.
pub fn mxfp8_values(values: &[u8], exponents: &[u8], k: usize) -> Result<Vec<f32>, String> {
    if !k.is_multiple_of(32)
        || !values.len().is_multiple_of(k)
        || exponents.len() * 32 != values.len()
    {
        return Err(format!(
            "MXFP8 rows of {k}: {} values, {} exponents",
            values.len(),
            exponents.len()
        ));
    }
    if let Some(at) = exponents.iter().position(|e| *e == 0xff) {
        return Err(format!("ue8m0 byte {at} is 0xFF, the NaN code"));
    }
    if let Some(at) = values.iter().position(|v| v & 0x7f == 0x7f) {
        return Err(format!("e4m3 byte {at} is a NaN code"));
    }
    Ok(values
        .chunks_exact(32)
        .zip(exponents)
        .flat_map(|(block, e)| {
            let s = ue8m0(*e);
            block.iter().map(move |v| e4m3_value(*v) * s)
        })
        .collect())
}

// ---- the class's tensor set --------------------------------------------------------------------

/// The attention projections a physical layer outside the loop core converts, as the checkpoint
/// names them under `model.llm.layers.N.attn.`: q, k, v, o, the output gate and the
/// relative-bias projection. The sink projection, the clip scales, the bias table and the convs
/// stay as stored.
pub const ATTENTION_PROJECTIONS: [&str; 6] =
    ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn", "wr_du"];

/// One tensor the class converts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RigTensor {
    /// A BF16 matrix `name` (`[rows, k]`), one tensor scale.
    Dense { name: String },
    /// An MXFP8 expert bank `name` (`[experts, rows, k]`, with `{name}_scale`), one scale per
    /// expert.
    Experts { name: String },
}

impl RigTensor {
    pub fn name(&self) -> &str {
        match self {
            RigTensor::Dense { name } | RigTensor::Experts { name } => name,
        }
    }
}

/// True for a physical layer whose attention the class converts: every layer outside the loop
/// core (coordinator decision 8: the core's attention is read three times per token and is the
/// most sensitive; it stays BF16).
pub fn converts_attention(ink: &Inkling2Config, physical: u32) -> bool {
    physical < ink.loop_core_start || physical >= ink.loop_core_end
}

/// True for a physical layer whose expert banks the class converts: the layers the checkpoint
/// declares in MXFP8.
pub fn converts_experts(ink: &Inkling2Config, physical: u32) -> Result<bool, String> {
    Ok(ink.expert_quant(physical)?.compute == crate::quant_groups::ComputeDtype::Mxfp8)
}

/// Every tensor the class converts, in a fixed order (layer, then projection), then the output
/// projection.
pub fn rig_tensors(ink: &Inkling2Config) -> Result<Vec<RigTensor>, String> {
    let mut out = Vec::new();
    for layer in 0..ink.num_hidden_layers {
        if converts_attention(ink, layer) {
            for p in ATTENTION_PROJECTIONS {
                out.push(RigTensor::Dense {
                    name: format!("model.llm.layers.{layer}.attn.{p}.weight"),
                });
            }
        }
        if converts_experts(ink, layer)? {
            for bank in ["w13", "w2"] {
                out.push(RigTensor::Experts {
                    name: format!("model.llm.layers.{layer}.mlp.experts.{bank}_weight"),
                });
            }
        }
    }
    out.push(RigTensor::Dense {
        name: "model.llm.unembed.weight".into(),
    });
    Ok(out)
}

/// The overlay tensors that replace `source` (a converted tensor), with their dtypes and shapes
/// from the source's logical shape: codes and block scales, then a bank's amax and denominator or
/// a dense tensor's `scale_2`.
pub fn overlay_planes(t: &RigTensor, logical: &[usize]) -> Vec<(String, &'static str, Vec<usize>)> {
    let name = t.name();
    let k = *logical.last().expect("a matrix");
    let lead = &logical[..logical.len() - 1];
    let with_last = |last: usize| {
        let mut s = lead.to_vec();
        s.push(last);
        s
    };
    let mut planes = vec![
        (name.to_string(), "U8", with_last(k / 2)),
        (format!("{name}_scale"), "F8_E4M3", with_last(k / 16)),
    ];
    match t {
        RigTensor::Dense { .. } => planes.push((format!("{name}_scale_2"), "F32", vec![])),
        RigTensor::Experts { .. } => {
            planes.push((format!("{name}_amax"), "F32", vec![logical[0]]));
            planes.push((format!("{name}_outer_scale_denominator"), "F32", vec![1]));
        }
    }
    planes
}

/// The plane suffixes of a converted tensor's kind, after the codes.
pub fn plane_suffixes(t: &RigTensor) -> &'static [&'static str] {
    match t {
        RigTensor::Dense { .. } => &["_scale", "_scale_2"],
        RigTensor::Experts { .. } => &["_scale", "_amax", "_outer_scale_denominator"],
    }
}

// ---- hashing, the overlay file and the manifest -------------------------------------------------

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Hashes of many byte slices on up to `threads` threads, in input order.
pub fn sha256_many(parts: &[&[u8]], threads: usize) -> Vec<String> {
    let mut out = vec![String::new(); parts.len()];
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<&mut String>> =
        out.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1).min(parts.len().max(1)) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= parts.len() {
                        break;
                    }
                    let h = sha256_hex(parts[i]);
                    **slots[i].lock().expect("hash slot") = h;
                }
            });
        }
    });
    drop(slots);
    out
}

/// sha256 of a file, streamed.
pub fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut n = 0u64;
    loop {
        let r = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if r == 0 {
            break;
        }
        h.update(&buf[..r]);
        n += r as u64;
    }
    Ok((hex(&h.finalize()), n))
}

/// sha256 of a safetensors file's 8-byte length and JSON header: it binds every name, dtype,
/// shape and offset of the file.
pub fn header_sha256(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let n = u64::from_le_bytes(len) as usize;
    let mut header = vec![0u8; n];
    f.read_exact(&mut header)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    h.update(len);
    h.update(&header);
    Ok(hex(&h.finalize()))
}

/// Tensors of an image by name: `(dtype, shape, bytes)`.
pub type ImageTensors<'a> = BTreeMap<String, (String, Vec<usize>, &'a [u8])>;

/// The tensors of a safetensors image: `name -> (dtype, shape, bytes)`.
pub fn image_tensors(image: &[u8]) -> Result<ImageTensors<'_>, String> {
    let n = image
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")) as usize)
        .ok_or("a safetensors image starts with its header length")?;
    let header = image
        .get(8..8 + n)
        .and_then(|b| std::str::from_utf8(b).ok())
        .ok_or("a safetensors header is UTF-8 inside the image")?;
    let data = &image[8 + n..];
    crate::safetensors::parse_header_json_checked(header)?
        .into_iter()
        .map(|(name, info)| {
            let bytes = data
                .get(info.data_offsets[0]..info.data_offsets[1])
                .ok_or_else(|| format!("{name} lies outside the image"))?;
            Ok((
                name,
                (
                    info.dtype,
                    info.shape.iter().map(|v| *v as usize).collect(),
                    bytes,
                ),
            ))
        })
        .collect()
}

/// A whole minted overlay for a small checkpoint, as a safetensors image (the tiny gates).
pub fn mint_overlay_image(get: &TensorGet<'_>, ink: &Inkling2Config) -> Result<Vec<u8>, String> {
    let mut tensors = BTreeMap::new();
    for t in rig_tensors(ink)? {
        for (name, dtype, shape, bytes) in mint_tensor_with(get, &t, 1)?.0 {
            tensors.insert(name, (dtype, shape, bytes));
        }
    }
    let (header, _) = safetensors_header(&tensors, &BTreeMap::new())?;
    let mut image = header;
    for (_, _, bytes) in tensors.values() {
        image.extend_from_slice(bytes);
    }
    Ok(image)
}

/// The 8-byte length and JSON header of a safetensors file of `tensors` (name order, data 8-byte
/// aligned), and the data size.
fn safetensors_header(
    tensors: &BTreeMap<String, (&'static str, Vec<usize>, Vec<u8>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<(Vec<u8>, usize), String> {
    let mut header = serde_json::Map::new();
    if !metadata.is_empty() {
        header.insert(
            "__metadata__".into(),
            serde_json::Value::Object(
                metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect(),
            ),
        );
    }
    let mut offset = 0usize;
    for (name, (dtype, shape, bytes)) in tensors {
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len()]}),
        );
        offset += bytes.len();
    }
    let mut text =
        serde_json::to_string(&serde_json::Value::Object(header)).map_err(|e| e.to_string())?;
    while !(8 + text.len()).is_multiple_of(8) {
        text.push(' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    Ok((out, offset))
}

/// Writes a safetensors file of `(name, dtype, shape, bytes)` in name order, 8-byte aligned
/// data, and fsyncs it.
pub fn write_safetensors(
    path: &Path,
    tensors: &BTreeMap<String, (&'static str, Vec<usize>, Vec<u8>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<(), String> {
    let (header, _) = safetensors_header(tensors, metadata)?;
    let tmp = path.with_extension("partial");
    let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.write_all(&header).map_err(|e| e.to_string())?;
    for (_, _, bytes) in tensors.values() {
        f.write_all(bytes).map_err(|e| e.to_string())?;
    }
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// One converted tensor in the manifest: the source tensors' hashes (the bound input) and the
/// overlay tensors' hashes (the minted output).
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestTensor {
    pub name: String,
    pub kind: &'static str,
    pub source: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    /// Relative squared error of the dequantized tensor against the source, as a receipt.
    pub rel_mse: f64,
}

/// The overlay manifest: what the overlay was minted from, how, and what it holds.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub source_artifact: String,
    pub source_file: String,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub source_header_sha256: String,
    pub quant_config_sha256: String,
    pub overlay_sha256: String,
    pub overlay_bytes: u64,
    pub tool: String,
    pub tensors: Vec<ManifestTensor>,
}

impl Manifest {
    pub fn to_json(&self) -> String {
        let tensors: Vec<serde_json::Value> = self
            .tensors
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name, "kind": t.kind, "source": t.source, "outputs": t.outputs,
                    "rel_mse": format!("{:.6e}", t.rel_mse),
                })
            })
            .collect();
        let v = serde_json::json!({
            "format": OVERLAY_FORMAT,
            "class": CLASS_NAME,
            "quantizer": {
                "id": QUANTIZER,
                "block": 16,
                "ratios": NV_RATIOS,
                "tensor_scale": "max(fl32(amax / 2688), 1e-30), per expert for an expert bank",
                "port_of": "klsens/klh.py nvfp4_weight_quant",
            },
            "source": {
                "artifact": self.source_artifact,
                "file": self.source_file,
                "sha256": self.source_sha256,
                "bytes": self.source_bytes,
                "header_sha256": self.source_header_sha256,
                "hf_quant_config_sha256": self.quant_config_sha256,
            },
            "overlay": {"file": OVERLAY_FILE, "sha256": self.overlay_sha256, "bytes": self.overlay_bytes},
            "tool": self.tool,
            "tensors": tensors,
        });
        serde_json::to_string_pretty(&v).expect("manifest json") + "\n"
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("manifest: {e}"))?;
        let s = |p: &str| -> Result<String, String> {
            v.pointer(p)
                .and_then(|x| x.as_str())
                .map(str::to_string)
                .ok_or_else(|| format!("manifest: {p} is not a string"))
        };
        let u = |p: &str| -> Result<u64, String> {
            v.pointer(p)
                .and_then(|x| x.as_u64())
                .ok_or_else(|| format!("manifest: {p} is not an integer"))
        };
        if s("/format")? != OVERLAY_FORMAT {
            return Err(format!("manifest: format is not {OVERLAY_FORMAT}"));
        }
        if s("/class")? != CLASS_NAME {
            return Err(format!("manifest: class is not {CLASS_NAME}"));
        }
        if s("/quantizer/id")? != QUANTIZER {
            return Err(format!("manifest: quantizer is not {QUANTIZER}"));
        }
        let map = |x: &serde_json::Value, what: &str| -> Result<BTreeMap<String, String>, String> {
            x.as_object()
                .ok_or_else(|| format!("manifest: {what} is not an object"))?
                .iter()
                .map(|(k, h)| {
                    h.as_str()
                        .map(|h| (k.clone(), h.to_string()))
                        .ok_or_else(|| format!("manifest: {what}.{k} is not a hash"))
                })
                .collect()
        };
        let tensors = v
            .pointer("/tensors")
            .and_then(|x| x.as_array())
            .ok_or("manifest: no tensor list")?
            .iter()
            .map(|t| {
                let name = t["name"]
                    .as_str()
                    .ok_or("manifest: tensor without a name")?
                    .to_string();
                let kind = match t["kind"].as_str() {
                    Some("dense") => "dense",
                    Some("experts") => "experts",
                    _ => return Err(format!("manifest: {name} has no kind")),
                };
                Ok(ManifestTensor {
                    source: map(&t["source"], &format!("{name}.source"))?,
                    outputs: map(&t["outputs"], &format!("{name}.outputs"))?,
                    rel_mse: t["rel_mse"]
                        .as_str()
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(f64::NAN),
                    name,
                    kind,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            source_artifact: s("/source/artifact")?,
            source_file: s("/source/file")?,
            source_sha256: s("/source/sha256")?,
            source_bytes: u("/source/bytes")?,
            source_header_sha256: s("/source/header_sha256")?,
            quant_config_sha256: s("/source/hf_quant_config_sha256")?,
            overlay_sha256: s("/overlay/sha256")?,
            overlay_bytes: u("/overlay/bytes")?,
            tool: s("/tool")?,
            tensors,
        })
    }

    /// The manifest covers exactly the class's tensor set, each with the planes it must have.
    pub fn check_tensor_set(&self, want: &[RigTensor]) -> Result<(), String> {
        let have: Vec<&str> = self.tensors.iter().map(|t| t.name.as_str()).collect();
        let need: Vec<&str> = want.iter().map(|t| t.name()).collect();
        if have != need {
            let missing: Vec<_> = need.iter().filter(|n| !have.contains(n)).collect();
            let extra: Vec<_> = have.iter().filter(|n| !need.contains(n)).collect();
            return Err(format!(
                "the overlay manifest's tensor list is not the {CLASS_NAME} set: missing {missing:?}, extra {extra:?}, or out of order"
            ));
        }
        for (t, w) in self.tensors.iter().zip(want) {
            let kind = match w {
                RigTensor::Dense { .. } => "dense",
                RigTensor::Experts { .. } => "experts",
            };
            if t.kind != kind {
                return Err(format!(
                    "{}: manifest kind {} is not {kind}",
                    t.name, t.kind
                ));
            }
            let planes: std::collections::BTreeSet<String> = std::iter::once(t.name.clone())
                .chain(plane_suffixes(w).iter().map(|s| format!("{}{s}", t.name)))
                .collect();
            if t.outputs
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                != planes
            {
                return Err(format!(
                    "{}: manifest outputs are not its NVFP4 planes {planes:?}",
                    t.name
                ));
            }
        }
        Ok(())
    }
}

/// Source and overlay tensor bytes a binding check hashes: `(name, bytes, want hash)`.
pub type HashCheck<'a> = (String, &'a [u8], String);

/// Hashes every pair in parallel and refuses the first mismatch.
pub fn check_hashes(checks: &[HashCheck<'_>], threads: usize) -> Result<(), String> {
    let parts: Vec<&[u8]> = checks.iter().map(|c| c.1).collect();
    let got = sha256_many(&parts, threads);
    for ((name, _, want), got) in checks.iter().zip(got) {
        if *want != got {
            return Err(format!(
                "{name}: sha256 {got} is not the manifest's {want}: the overlay is not bound to these bytes"
            ));
        }
    }
    Ok(())
}

/// An overlay bound to a base checkpoint: the manifest and the opened overlay file.
pub struct BoundOverlay {
    pub manifest: Manifest,
    pub overlay: crate::safetensors::StShard,
    pub dir: std::path::PathBuf,
    /// Every overlay tensor name, mapped to the converted tensor it belongs to.
    pub planes: BTreeMap<String, String>,
}

/// A tensor source: `(dtype, shape, bytes)` of a name, or `None`.
pub type TensorGet<'a> = dyn Fn(&str) -> Option<(String, Vec<usize>, &'a [u8])> + Sync + 'a;

/// The getter of a safetensors model.
pub fn model_getter<'a>(
    base: &'a crate::safetensors::StModel,
) -> impl Fn(&str) -> Option<(String, Vec<usize>, &'a [u8])> + Sync + 'a {
    move |name| {
        base.raw(name).map(|(i, b)| {
            (
                i.dtype.clone(),
                i.shape.iter().map(|v| *v as usize).collect(),
                b,
            )
        })
    }
}

/// The logical shape of a converted tensor in the base checkpoint, checked against its kind.
pub fn source_shape(get: &TensorGet<'_>, t: &RigTensor) -> Result<Vec<usize>, String> {
    let (dtype_have, shape, _) =
        get(t.name()).ok_or_else(|| format!("the base checkpoint has no tensor {}", t.name()))?;
    let (dtype, dims) = match t {
        RigTensor::Dense { .. } => ("BF16", 2),
        RigTensor::Experts { .. } => ("F8_E4M3", 3),
    };
    if dtype_have != dtype || shape.len() != dims || !shape[dims - 1].is_multiple_of(32) {
        return Err(format!(
            "{}: {dtype_have} {shape:?} in the base checkpoint; the {CLASS_NAME} class converts a {dtype} tensor of {dims} dims",
            t.name(),
        ));
    }
    Ok(shape)
}

/// The source planes of a converted tensor (the tensor, and an MXFP8 bank's exponents).
pub fn source_planes(t: &RigTensor) -> Vec<String> {
    match t {
        RigTensor::Dense { name } => vec![name.clone()],
        RigTensor::Experts { name } => vec![name.clone(), format!("{name}_scale")],
    }
}

/// Binds an overlay directory to a base checkpoint, or refuses. Checks, in order: the manifest's
/// format, class and quantizer; its tensor list against the class's set for this config; the base
/// file's size and header hash; the quantization config's hash; the overlay's names, dtypes and
/// shapes; and the sha256 of every source tensor the overlay replaces and of every overlay tensor
/// (in parallel on `threads` threads). The full-file hashes are `ink2-mint verify`'s.
pub fn bind_overlay(
    ckpt_dir: &Path,
    overlay_dir: &Path,
    base: &crate::safetensors::StModel,
    ink: &Inkling2Config,
    quant_config_sha256: &str,
    threads: usize,
) -> Result<BoundOverlay, String> {
    let at = |e: String| format!("{CLASS_NAME} overlay {}: {e}", overlay_dir.display());
    let text = std::fs::read_to_string(overlay_dir.join(MANIFEST_FILE)).map_err(|e| {
        at(format!(
            "{MANIFEST_FILE}: {e} (mint it with ink2-mint mint)"
        ))
    })?;
    let manifest = Manifest::parse(&text).map_err(at)?;
    let want = rig_tensors(ink).map_err(at)?;
    manifest.check_tensor_set(&want).map_err(at)?;
    let base_file = ckpt_dir.join(&manifest.source_file);
    let bytes = std::fs::metadata(&base_file)
        .map_err(|e| at(format!("{}: {e}", base_file.display())))?
        .len();
    if bytes != manifest.source_bytes {
        return Err(at(format!(
            "the base file is {bytes} bytes, the manifest binds {}",
            manifest.source_bytes
        )));
    }
    let header = header_sha256(&base_file).map_err(at)?;
    if header != manifest.source_header_sha256 {
        return Err(at(format!(
            "the base file's header sha256 {header} is not the manifest's {}",
            manifest.source_header_sha256
        )));
    }
    if quant_config_sha256 != manifest.quant_config_sha256 {
        return Err(at(format!(
            "hf_quant_config.json sha256 {quant_config_sha256} is not the manifest's {}",
            manifest.quant_config_sha256
        )));
    }
    let path = overlay_dir.join(OVERLAY_FILE);
    let overlay_bytes = std::fs::metadata(&path)
        .map_err(|e| at(format!("{OVERLAY_FILE}: {e}")))?
        .len();
    if overlay_bytes != manifest.overlay_bytes {
        return Err(at(format!(
            "{OVERLAY_FILE} is {overlay_bytes} bytes, the manifest binds {}",
            manifest.overlay_bytes
        )));
    }
    let overlay =
        crate::safetensors::StShard::open(&path).map_err(|e| at(format!("{OVERLAY_FILE}: {e}")))?;
    let mut planes = BTreeMap::new();
    let mut checks: Vec<HashCheck<'_>> = Vec::new();
    let get = model_getter(base);
    for (t, m) in want.iter().zip(&manifest.tensors) {
        let shape = source_shape(&get, t).map_err(at)?;
        for (plane, dtype, pshape) in overlay_planes(t, &shape) {
            let (info, data) = overlay
                .raw(&plane)
                .ok_or_else(|| at(format!("{OVERLAY_FILE} has no tensor {plane}")))?;
            let got: Vec<usize> = info.shape.iter().map(|v| *v as usize).collect();
            if info.dtype != dtype || got != pshape {
                return Err(at(format!(
                    "{plane}: {} {got:?} in the overlay, the class stores {dtype} {pshape:?}",
                    info.dtype
                )));
            }
            checks.push((plane.clone(), data, m.outputs[&plane].clone()));
            planes.insert(plane, t.name().to_string());
        }
        for plane in source_planes(t) {
            let (_, data) = base
                .raw(&plane)
                .ok_or_else(|| at(format!("the base checkpoint has no tensor {plane}")))?;
            let want = m
                .source
                .get(&plane)
                .ok_or_else(|| at(format!("the manifest binds no source hash for {plane}")))?;
            checks.push((format!("base {plane}"), data, want.clone()));
        }
    }
    if overlay.len() != planes.len() {
        return Err(at(format!(
            "{OVERLAY_FILE} holds {} tensors, the class's set is {}",
            overlay.len(),
            planes.len()
        )));
    }
    check_hashes(&checks, threads).map_err(at)?;
    drop(checks);
    Ok(BoundOverlay {
        manifest,
        overlay,
        dir: overlay_dir.to_path_buf(),
        planes,
    })
}

/// Mints the class's overlay of the checkpoint in `ckpt_dir` (one model shard, `hf_quant_config.json`)
/// into `out_dir`: `overlay.safetensors` and `manifest.json`, both made read-only. Refuses an
/// `out_dir` that already holds either (an overlay is immutable). `log` gets one line a tensor.
pub fn mint_overlay_dir(
    ckpt_dir: &Path,
    out_dir: &Path,
    artifact: &str,
    threads: usize,
    log: &mut dyn FnMut(String),
) -> Result<Manifest, String> {
    let read = |n: &str| std::fs::read_to_string(ckpt_dir.join(n)).map_err(|e| format!("{n}: {e}"));
    let mut config = crate::config::ModelConfig::from_hf(&crate::config::HfConfig::try_parse(
        &read("config.json")?,
    )?);
    let quant = read("hf_quant_config.json")?;
    config.attach_quant_groups(&quant)?;
    let ink = config.inkling2.as_ref().ok_or("not an inkling2 config")?;
    let mut files: Vec<String> = std::fs::read_dir(ckpt_dir)
        .map_err(|e| format!("{}: {e}", ckpt_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        .collect();
    files.sort();
    let [file] = files.as_slice() else {
        return Err(format!(
            "the mint reads one model shard; {} has {files:?}",
            ckpt_dir.display()
        ));
    };
    let base = crate::safetensors::StModel::open(ckpt_dir)
        .map_err(|e| format!("open {}: {e}", ckpt_dir.display()))?;
    if out_dir.join(MANIFEST_FILE).exists() || out_dir.join(OVERLAY_FILE).exists() {
        return Err(format!(
            "{} already holds an overlay; an overlay is immutable",
            out_dir.display()
        ));
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let set = rig_tensors(ink)?;
    let mut tensors = BTreeMap::new();
    let mut entries = Vec::new();
    for t in &set {
        let t0 = std::time::Instant::now();
        let (planes, source, rel) = mint_tensor(&base, t, threads)?;
        let mut outputs = BTreeMap::new();
        for (name, dtype, shape, bytes) in planes {
            outputs.insert(name.clone(), sha256_hex(&bytes));
            tensors.insert(name, (dtype, shape, bytes));
        }
        log(format!(
            "{} rel_mse {rel:.4e} in {:.1}s",
            t.name(),
            t0.elapsed().as_secs_f64()
        ));
        entries.push(ManifestTensor {
            name: t.name().to_string(),
            kind: match t {
                RigTensor::Dense { .. } => "dense",
                RigTensor::Experts { .. } => "experts",
            },
            source,
            outputs,
            rel_mse: rel,
        });
    }
    let path = out_dir.join(OVERLAY_FILE);
    let meta = BTreeMap::from([
        ("format".to_string(), OVERLAY_FORMAT.to_string()),
        ("class".to_string(), CLASS_NAME.to_string()),
        ("source".to_string(), artifact.to_string()),
    ]);
    write_safetensors(&path, &tensors, &meta)?;
    drop(tensors);
    let (overlay_sha256, overlay_bytes) = sha256_file(&path)?;
    let base_path = ckpt_dir.join(file);
    let (source_sha256, source_bytes) = sha256_file(&base_path)?;
    let manifest = Manifest {
        source_artifact: artifact.to_string(),
        source_file: file.clone(),
        source_sha256,
        source_bytes,
        source_header_sha256: header_sha256(&base_path)?,
        quant_config_sha256: sha256_hex(quant.as_bytes()),
        overlay_sha256,
        overlay_bytes,
        tool: format!("ink2-mint memra-gguf {}", env!("CARGO_PKG_VERSION")),
        tensors: entries,
    };
    let mpath = out_dir.join(MANIFEST_FILE);
    std::fs::write(&mpath, manifest.to_json()).map_err(|e| format!("{}: {e}", mpath.display()))?;
    for p in [&path, &mpath] {
        let mut perm = std::fs::metadata(p)
            .map_err(|e| e.to_string())?
            .permissions();
        perm.set_readonly(true);
        std::fs::set_permissions(p, perm).map_err(|e| e.to_string())?;
    }
    Ok(manifest)
}

/// The default overlay directory of a checkpoint directory: its sibling `<name>.nvfp4-rig`.
pub fn default_overlay_dir(ckpt_dir: &Path) -> std::path::PathBuf {
    let name = ckpt_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "checkpoint".into());
    ckpt_dir.with_file_name(format!("{name}.{CLASS_NAME}"))
}

/// Converts one tensor of the base checkpoint: its overlay planes, the source planes' hashes and
/// the relative squared error. Deterministic: the bytes do not depend on `threads`.
pub type MintedPlanes = Vec<(String, &'static str, Vec<usize>, Vec<u8>)>;

pub fn mint_tensor(
    base: &crate::safetensors::StModel,
    t: &RigTensor,
    threads: usize,
) -> Result<(MintedPlanes, BTreeMap<String, String>, f64), String> {
    mint_tensor_with(&model_getter(base), t, threads)
}

/// [`mint_tensor`] over any tensor source.
pub fn mint_tensor_with<'a>(
    get: &TensorGet<'a>,
    t: &RigTensor,
    threads: usize,
) -> Result<(MintedPlanes, BTreeMap<String, String>, f64), String> {
    let shape = source_shape(get, t)?;
    let k = *shape.last().expect("matrix");
    let mut source = BTreeMap::new();
    for plane in source_planes(t) {
        let (_, _, data) = get(&plane).ok_or_else(|| format!("no tensor {plane}"))?;
        source.insert(plane, sha256_hex(data));
    }
    type Values<'v> = Box<dyn Fn(usize) -> Result<Vec<f32>, String> + 'v>;
    let (groups, values): (usize, Values<'a>) = match t {
        RigTensor::Dense { name } => {
            let (_, _, data) = get(name).ok_or_else(|| format!("no tensor {name}"))?;
            (1, Box::new(move |_| Ok(bf16_values(data))))
        }
        RigTensor::Experts { name } => {
            let (_, _, data) = get(name).ok_or_else(|| format!("no tensor {name}"))?;
            let scale = format!("{name}_scale");
            let (sdtype, _, exps) = get(&scale).ok_or_else(|| format!("no tensor {scale}"))?;
            let per = shape[1] * k;
            if sdtype != "U8" || exps.len() * 32 != data.len() {
                return Err(format!(
                    "{scale}: {sdtype} with {} bytes for {} values",
                    exps.len(),
                    data.len()
                ));
            }
            (
                shape[0],
                Box::new(move |e| {
                    mxfp8_values(
                        &data[e * per..(e + 1) * per],
                        &exps[e * per / 32..(e + 1) * per / 32],
                        k,
                    )
                }),
            )
        }
    };
    let total: usize = shape.iter().product();
    let per = total / groups;
    let mut codes = vec![0u8; total / 2];
    let mut scales = vec![0u8; total / 16];
    let mut amaxes = Vec::with_capacity(groups);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for g in 0..groups {
        let w = values(g)?;
        let a = amax(&w);
        if a.is_nan() || a / DENOMINATOR < 1e-30 {
            return Err(format!(
                "{}: group {g} has amax {a}; the class mints tensors whose scale amax / 2688 is a normal positive number",
                t.name()
            ));
        }
        let tscale = tensor_scale(a);
        let (c, s) = (
            &mut codes[g * per / 2..(g + 1) * per / 2],
            &mut scales[g * per / 16..(g + 1) * per / 16],
        );
        quantize_rows(&w, k, tscale, c, s, threads);
        let q = dequant_rows(c, s, k, tscale);
        for (a, b) in w.iter().zip(&q) {
            let d = (*b - *a) as f64;
            num += d * d;
            den += (*a as f64) * (*a as f64);
        }
        amaxes.push(a);
    }
    let planes = overlay_planes(t, &shape);
    let data = match t {
        RigTensor::Dense { .. } => vec![
            codes,
            scales,
            tensor_scale(amaxes[0]).to_le_bytes().to_vec(),
        ],
        RigTensor::Experts { .. } => vec![
            codes,
            scales,
            amaxes.iter().flat_map(|a| a.to_le_bytes()).collect(),
            DENOMINATOR.to_le_bytes().to_vec(),
        ],
    };
    let out = planes
        .into_iter()
        .zip(data)
        .map(|((n, d, s), b)| (n, d, s, b))
        .collect();
    Ok((out, source, num / den.max(f64::MIN_POSITIVE)))
}

#[cfg(test)]
mod tests;
