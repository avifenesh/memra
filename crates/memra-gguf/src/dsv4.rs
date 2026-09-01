//! DeepSeek-V4-Flash safetensors census + CPU reference quant decode (loader lane,
//! research/dsv4-flash-loader-20260818).
//!
//! Three quant recipes coexist in the one NVFP4 artifact — measured from bytes on the real
//! checkpoint (2026-08-18), not inferred from the V3/GLM ancestry:
//!
//! ── trunk routed experts: modelopt NVFP4 (hf_quant_config: quant_algo NVFP4, group_size 16) ──
//!   <stem>.weight          U8       [out, in/2]    two e2m1 codes/byte (elem 2i -> low nibble)
//!   <stem>.weight_scale    F8_E4M3  [out, in/16]   per-16 block scale, groups run along IN
//!   <stem>.weight_scale_2  F32      []             per-tensor macro scale (2^-12 measured)
//!   <stem>.input_scale     F32      []             W4A8 activation scale (unused for decode)
//!   dequant = e2m1(code) * e4m3(weight_scale[r, c/16]) * weight_scale_2
//!   Measured structure (cast provenance — the artifact is a LOSSLESS MXFP4->NVFP4 cast, see
//!   the artifact's cast_mxfp4_to_nvfp4.log): every effective scale e4m3*scale_2 is an exact
//!   power of two and adjacent 16-group scale pairs are IDENTICAL (32-group ancestry). The
//!   census gate hard-checks both on its samples: any scale-indexing bug shatters them.
//!
//! ── MTP routed experts (mtp.*): MXFP4 — excluded from the NVFP4 cast (hf_quant_config
//!    exclude_modules "mtp.*"), left in the original OCP-MX layout ──
//!   <stem>.weight  I8       [out, in/2]    same e2m1 nibble packing (dtype label differs)
//!   <stem>.scale   F8_E8M0  [out, in/32]   per-32 block scale, groups along IN (2^(b-127))
//!   dequant = e2m1(code) * 2^(scale[r, c/32] - 127)
//!
//! ── everything-else quantized (attn/shared-expert/indexer wq_b/mtp e_proj+h_proj): FP8 with
//!    128x128 block scales (config.json quantization_config: fmt e4m3, scale_fmt ue8m0,
//!    weight_block_size [128,128]) ──
//!   <stem>.weight  F8_E4M3  [out, in]
//!   <stem>.scale   F8_E8M0  [out/128, in/128]
//!   dequant = e4m3(byte) * 2^(scale[r/128, c/128] - 127)
//!
//! Group-axis note: every scale grid's shape factors UNIQUELY against its logical tensor
//! (e.g. NVFP4 (2048, 256) vs logical (2048, 4096): rows match 1:1, so 256 = 4096/16 groups
//! along IN is forced; MX (2048, 128) forces 32-along-IN; (8, 32) vs (1024, 4096) forces
//! 128x128) — plus the measured pair-sharing/power-of-two structure above pins the indexing.
//!
//! Norms/embeddings/head/router gate are BF16; hyper-connections, attn_sink, gate.bias,
//! compressor ape are F32; hash-router tid2eid is I64. The KV compressor has TWO shape classes
//! keyed on the layer's compress ratio (fine ratio-4 latent 2*head_dim vs coarse ratio-128
//! latent head_dim; ape = [ratio, latent]) — the banked header census had collapsed them, and
//! the first on-box gate run caught it with 60 named mismatches. The full expected tensor set
//! is DERIVED from config math in [`expected_census`]; loading REFUSES loudly (named tensors)
//! on any missing / extra / mis-shaped / mis-typed entry.

use std::collections::BTreeMap;

use crate::config::ModelConfig;
use crate::nvfp4_repack::fp8_e4m3_to_f32;
use crate::safetensors::StModel;

/// e2m1 (FP4) code -> value; sign bit at 0x8. Standard OCP FP4 table (the STANDARD codebook —
/// nvfp4_repack::KVALUES_MXFP4 is the GGUF DOUBLED table; do not mix the two conventions).
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// FP8 E8M0 (OCP MX scale) byte -> f32: 2^(b-127). 0xFF is the NaN code and is PROPAGATED —
/// the census gate must fail on it, never silently zero a scale (contrast: e4m3 NaN -> 0.0 is
/// the modelopt WEIGHT convention, wrong for a scale).
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        (b as f32 - 127.0).exp2()
    }
}

// ============================ expected tensor census ============================

/// One tensor's expected on-disk identity. `dtype` is the exact safetensors dtype string;
/// `shape` is row-major as stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub dtype: &'static str,
    pub shape: Vec<u64>,
}

impl TensorSpec {
    pub fn n_bytes(&self) -> u64 {
        let elems: u64 = self.shape.iter().product();
        let per = match self.dtype {
            "F32" => 4,
            "BF16" => 2,
            "I64" => 8,
            "U8" | "I8" | "F8_E4M3" | "F8_E8M0" => 1,
            other => panic!("dsv4 census: unhandled dtype {other}"),
        };
        elems * per
    }
}

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

/// Build the complete expected tensor map for a deepseek_v4 checkpoint from config math.
/// Panics (loud refusal) if the config is not a deepseek_v4 or lacks the MoE block.
pub fn expected_census(mc: &ModelConfig) -> BTreeMap<String, TensorSpec> {
    let d = mc
        .dsv4
        .as_ref()
        .expect("dsv4 census requires a deepseek_v4 ModelConfig");
    let moe = mc
        .moe
        .as_ref()
        .expect("deepseek_v4 is MoE; moe block missing");

    let h = mc.n_embd as u64; // 4096
    let v = mc.n_vocab as u64; // 129280
    let nh = mc.n_head as u64; // 64
    let hd = d.head_dim as u64; // 512
    let qlr = d.q_lora_rank as u64; // 1024
    let kv_out = d.num_key_value_heads as u64 * hd; // 512 (wkv/kv_norm width)
    let o_out = d.o_groups as u64 * d.o_lora_rank as u64; // 8192 (wo_a out / wo_b in)
    let ihd = d.index_head_dim as u64; // 128
    let inh = d.index_n_heads as u64; // 64
    let ne = moe.expert_count as u64; // 256
    let topk = moe.expert_used_count as u64; // 6
    let mff = moe.expert_ff_length as u64; // 2048
    let sff = d.n_shared_experts as u64 * mff; // 2048
    let hcm = d.hc_mult as u64; // 4
    // Layer-level hyper-connection widths. Single-artifact evidence, stated as formulas so a
    // sibling with another hc_mult re-verifies through the census gate: fn/base rows = hc_mult!
    // (24 == 4!, consistent with a permutation-basis parameterization of the Sinkhorn-normalized
    // stream mixer), scale = hc_mult - 1 (3); the HEAD-level hc uses hc_mult rows and scale 1.
    let hc_rows = factorial(hcm); // 24
    let hc_scale = hcm - 1; // 3
    let hc_w = hcm * h; // 16384 (fn width = hc_mult * hidden, both levels)
    let mut map: BTreeMap<String, TensorSpec> = BTreeMap::new();
    let t = |m: &mut BTreeMap<String, TensorSpec>,
             name: String,
             dtype: &'static str,
             shape: Vec<u64>| {
        let prev = m.insert(name.clone(), TensorSpec { dtype, shape });
        assert!(prev.is_none(), "dsv4 census: duplicate spec for {name}");
    };
    // FP8 linear + its 128x128 UE8M0 block-scale grid (config quantization_config
    // weight_block_size [128,128] — both dims must divide exactly; refuse otherwise).
    let fp8 = |m: &mut BTreeMap<String, TensorSpec>, stem: String, out: u64, inn: u64| {
        assert!(
            out.is_multiple_of(128) && inn.is_multiple_of(128),
            "dsv4 fp8 block quant needs 128-divisible dims: {stem} [{out}, {inn}]"
        );
        m.insert(
            format!("{stem}.weight"),
            TensorSpec {
                dtype: "F8_E4M3",
                shape: vec![out, inn],
            },
        );
        m.insert(
            format!("{stem}.scale"),
            TensorSpec {
                dtype: "F8_E8M0",
                shape: vec![out / 128, inn / 128],
            },
        );
    };
    // modelopt NVFP4 quad (trunk experts).
    let nvfp4 = |m: &mut BTreeMap<String, TensorSpec>, stem: String, out: u64, inn: u64| {
        assert!(
            inn.is_multiple_of(16),
            "dsv4 nvfp4 needs in % 16 == 0: {stem} [{out}, {inn}]"
        );
        m.insert(
            format!("{stem}.weight"),
            TensorSpec {
                dtype: "U8",
                shape: vec![out, inn / 2],
            },
        );
        m.insert(
            format!("{stem}.weight_scale"),
            TensorSpec {
                dtype: "F8_E4M3",
                shape: vec![out, inn / 16],
            },
        );
        m.insert(
            format!("{stem}.weight_scale_2"),
            TensorSpec {
                dtype: "F32",
                shape: vec![],
            },
        );
        m.insert(
            format!("{stem}.input_scale"),
            TensorSpec {
                dtype: "F32",
                shape: vec![],
            },
        );
    };
    // OCP MXFP4 pair (MTP experts — excluded from the NVFP4 cast, hf_quant_config "mtp.*").
    let mxfp4 = |m: &mut BTreeMap<String, TensorSpec>, stem: String, out: u64, inn: u64| {
        assert!(
            inn.is_multiple_of(32),
            "dsv4 mxfp4 needs in % 32 == 0: {stem} [{out}, {inn}]"
        );
        m.insert(
            format!("{stem}.weight"),
            TensorSpec {
                dtype: "I8",
                shape: vec![out, inn / 2],
            },
        );
        m.insert(
            format!("{stem}.scale"),
            TensorSpec {
                dtype: "F8_E8M0",
                shape: vec![out, inn / 32],
            },
        );
    };

    // ---- globals ----
    t(&mut map, "embed.weight".into(), "BF16", vec![v, h]);
    t(&mut map, "head.weight".into(), "BF16", vec![v, h]);
    t(&mut map, "norm.weight".into(), "BF16", vec![h]);
    t(&mut map, "hc_head_base".into(), "F32", vec![hcm]);
    t(&mut map, "hc_head_fn".into(), "F32", vec![hcm, hc_w]);
    t(&mut map, "hc_head_scale".into(), "F32", vec![1]);

    let n_trunk = mc.n_layer - mc.nextn_predict_layers;
    // One attention + hc + MoE block, shared by trunk layers and the MTP layer(s). `il` is the
    // GLOBAL layer index (trunk 0..n_trunk, then MTP), which drives the compressor/indexer/hash
    // splits through compress_ratios and num_hash_layers — never through hardcoded counts.
    let block = |m: &mut BTreeMap<String, TensorSpec>, p: &str, il: u32, mtp: bool| {
        // norms + sink
        t(m, format!("{p}attn_norm.weight"), "BF16", vec![h]);
        t(m, format!("{p}ffn_norm.weight"), "BF16", vec![h]);
        t(m, format!("{p}attn.q_norm.weight"), "BF16", vec![qlr]);
        t(m, format!("{p}attn.kv_norm.weight"), "BF16", vec![kv_out]);
        t(m, format!("{p}attn.attn_sink"), "F32", vec![nh]);
        // attention projections (FP8 + 128x128 UE8M0)
        fp8(m, format!("{p}attn.wq_a"), qlr, h);
        fp8(m, format!("{p}attn.wq_b"), nh * hd, qlr);
        fp8(m, format!("{p}attn.wkv"), kv_out, h);
        fp8(m, format!("{p}attn.wo_a"), o_out, h);
        fp8(m, format!("{p}attn.wo_b"), h, o_out);
        // per-layer KV compressor (compress_ratios[il] != 0). TWO measured shape classes, keyed
        // on the layer's ratio — the first gate run refused with 60 named mismatches and exposed
        // that the banked header census had collapsed them into one:
        //   fine (ratio 4, indexer-paired):   wkv/wgate [2*head_dim, hidden], ape [4, 2*head_dim]
        //   coarse (ratio 128):               wkv/wgate [head_dim, hidden],   ape [128, head_dim]
        // i.e. ape = [ratio, latent] (one positional row per position inside a compression
        // block) and latent = 2*head_dim on fine layers, head_dim on coarse; norm covers
        // head_dim on BOTH classes. The 2x-vs-1x latent rule is single-artifact evidence keyed
        // on indexer presence; a sibling with other ratio classes refuses at the census gate.
        if d.has_compressor(il) {
            let ratio = d.compress_ratio(il) as u64;
            let latent = if d.has_indexer(il) { 2 * hd } else { hd };
            t(
                m,
                format!("{p}attn.compressor.wkv.weight"),
                "BF16",
                vec![latent, h],
            );
            t(
                m,
                format!("{p}attn.compressor.wgate.weight"),
                "BF16",
                vec![latent, h],
            );
            t(
                m,
                format!("{p}attn.compressor.norm.weight"),
                "BF16",
                vec![hd],
            );
            t(
                m,
                format!("{p}attn.compressor.ape"),
                "F32",
                vec![ratio, latent],
            );
        }
        // DSA lightning indexer (compress_ratios[il] == 4): its private compressor is the fine
        // class on index_head_dim — latent 2*index_head_dim, ape [ratio, latent].
        if d.has_indexer(il) {
            let ratio = d.compress_ratio(il) as u64;
            t(
                m,
                format!("{p}attn.indexer.compressor.wkv.weight"),
                "BF16",
                vec![2 * ihd, h],
            );
            t(
                m,
                format!("{p}attn.indexer.compressor.wgate.weight"),
                "BF16",
                vec![2 * ihd, h],
            );
            t(
                m,
                format!("{p}attn.indexer.compressor.norm.weight"),
                "BF16",
                vec![ihd],
            );
            t(
                m,
                format!("{p}attn.indexer.compressor.ape"),
                "F32",
                vec![ratio, 2 * ihd],
            );
            t(
                m,
                format!("{p}attn.indexer.weights_proj.weight"),
                "BF16",
                vec![inh, h],
            );
            fp8(m, format!("{p}attn.indexer.wq_b"), inh * ihd, qlr);
        }
        // hyper-connections (per-layer pair; the MTP block also carries its own head-level set)
        t(m, format!("{p}hc_attn_base"), "F32", vec![hc_rows]);
        t(m, format!("{p}hc_attn_fn"), "F32", vec![hc_rows, hc_w]);
        t(m, format!("{p}hc_attn_scale"), "F32", vec![hc_scale]);
        t(m, format!("{p}hc_ffn_base"), "F32", vec![hc_rows]);
        t(m, format!("{p}hc_ffn_fn"), "F32", vec![hc_rows, hc_w]);
        t(m, format!("{p}hc_ffn_scale"), "F32", vec![hc_scale]);
        // router: every layer carries the gate matrix and the FULL expert bank; the leading
        // num_hash_layers route by token-id table (tid2eid) instead of score bias.
        t(m, format!("{p}ffn.gate.weight"), "BF16", vec![ne, h]);
        if d.is_hash_layer(il) {
            t(m, format!("{p}ffn.gate.tid2eid"), "I64", vec![v, topk]);
        } else {
            t(m, format!("{p}ffn.gate.bias"), "F32", vec![ne]);
        }
        // shared expert(s): FP8, fused width n_shared_experts * moe_intermediate
        fp8(m, format!("{p}ffn.shared_experts.w1"), sff, h);
        fp8(m, format!("{p}ffn.shared_experts.w2"), h, sff);
        fp8(m, format!("{p}ffn.shared_experts.w3"), sff, h);
        // routed experts: w1/w3 [moe_ff, hidden], w2 [hidden, moe_ff]
        for e in 0..ne {
            if mtp {
                mxfp4(m, format!("{p}ffn.experts.{e}.w1"), mff, h);
                mxfp4(m, format!("{p}ffn.experts.{e}.w2"), h, mff);
                mxfp4(m, format!("{p}ffn.experts.{e}.w3"), mff, h);
            } else {
                nvfp4(m, format!("{p}ffn.experts.{e}.w1"), mff, h);
                nvfp4(m, format!("{p}ffn.experts.{e}.w2"), h, mff);
                nvfp4(m, format!("{p}ffn.experts.{e}.w3"), mff, h);
            }
        }
    };

    for il in 0..n_trunk {
        block(&mut map, &format!("layers.{il}."), il, false);
    }
    for k in 0..mc.nextn_predict_layers {
        let p = format!("mtp.{k}.");
        block(&mut map, &p, n_trunk + k, true);
        // MTP-only extras: embedding/hidden fusion projections (FP8) + their norms + the
        // block's own output norm and head-level hyper-connections.
        fp8(&mut map, format!("{p}e_proj"), h, h);
        fp8(&mut map, format!("{p}h_proj"), h, h);
        t(&mut map, format!("{p}enorm.weight"), "BF16", vec![h]);
        t(&mut map, format!("{p}hnorm.weight"), "BF16", vec![h]);
        t(&mut map, format!("{p}norm.weight"), "BF16", vec![h]);
        t(&mut map, format!("{p}hc_head_base"), "F32", vec![hcm]);
        t(&mut map, format!("{p}hc_head_fn"), "F32", vec![hcm, hc_w]);
        t(&mut map, format!("{p}hc_head_scale"), "F32", vec![1]);
    }
    map
}

/// Census verification result. Empty vectors == the artifact matches the derived map exactly.
#[derive(Debug, Default)]
pub struct CensusReport {
    pub missing: Vec<String>,
    pub extra: Vec<String>,
    /// "name: expected dtype/shape/bytes vs found ..." — any header disagreement.
    pub mismatched: Vec<String>,
}

impl CensusReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.extra.is_empty() && self.mismatched.is_empty()
    }
}

/// Verify every expected tensor exists with the exact dtype, shape, AND byte length — and that
/// nothing unexpected exists. Header-only (no tensor bytes are touched).
pub fn verify_census(m: &StModel, expected: &BTreeMap<String, TensorSpec>) -> CensusReport {
    let mut r = CensusReport::default();
    for (name, spec) in expected {
        match m.raw(name) {
            None => r.missing.push(name.clone()),
            Some((info, bytes)) => {
                let dtype_ok = info.dtype == spec.dtype;
                let shape_ok = info.shape == spec.shape;
                let bytes_ok = bytes.len() as u64 == spec.n_bytes();
                if !(dtype_ok && shape_ok && bytes_ok) {
                    r.mismatched.push(format!(
                        "{name}: expected {}/{:?}/{}B, found {}/{:?}/{}B",
                        spec.dtype,
                        spec.shape,
                        spec.n_bytes(),
                        info.dtype,
                        info.shape,
                        bytes.len()
                    ));
                }
            }
        }
    }
    for name in m.names() {
        if !expected.contains_key(name) {
            r.extra.push(name.clone());
        }
    }
    r.extra.sort();
    r
}

/// Verify the artifact's own quant declaration (`hf_quant_config.json`) matches the recipe the
/// census encodes: NVFP4/group-16 declared for EVERY trunk layer's `ffn.experts` (and nothing
/// else), `mtp.*` excluded (those experts stay MXFP4). Returns the failures; empty == clean.
/// This reads the artifact's declaration rather than assuming the recipe split.
pub fn verify_quant_declaration(dir: &std::path::Path, mc: &ModelConfig) -> Vec<String> {
    let mut errs = Vec::new();
    let txt = match std::fs::read_to_string(dir.join("hf_quant_config.json")) {
        Ok(t) => t,
        Err(e) => return vec![format!("hf_quant_config.json unreadable: {e}")],
    };
    let top = crate::config::JsonObj::parse(&txt);
    let Some(q) = top.object("quantization") else {
        return vec!["hf_quant_config.json: no `quantization` object".into()];
    };
    if q.string("quant_algo").as_deref() != Some("MIXED_PRECISION") {
        errs.push(format!(
            "quant_algo != MIXED_PRECISION (found {:?})",
            q.string("quant_algo")
        ));
    }
    if q.u32("group_size") != Some(16) {
        errs.push(format!(
            "group_size != 16 (found {:?})",
            q.u32("group_size")
        ));
    }
    let n_trunk = mc.n_layer - mc.nextn_predict_layers;
    match q.object("quantized_layers") {
        None => errs.push("no `quantized_layers` object".into()),
        Some(ql) => {
            let mut declared: Vec<String> = Vec::new();
            for (k, _) in ql.fields() {
                declared.push(k.to_string());
                match ql.object(k) {
                    Some(e)
                        if e.string("quant_algo").as_deref() == Some("NVFP4")
                            && e.u32("group_size") == Some(16) => {}
                    _ => errs.push(format!("{k}: not NVFP4/group-16")),
                }
            }
            for il in 0..n_trunk {
                let want = format!("layers.{il}.ffn.experts");
                if !declared.contains(&want) {
                    errs.push(format!("missing declaration for {want}"));
                }
            }
            if declared.len() != n_trunk as usize {
                errs.push(format!(
                    "quantized_layers count {} != trunk layer count {n_trunk}",
                    declared.len()
                ));
            }
        }
    }
    // exclude_modules is a plain string array; the load-bearing member is "mtp.*" (the MTP
    // experts were left in the original MXFP4 layout — the census's I8+E8M0 branch).
    match q.raw("exclude_modules") {
        Some(raw) if raw.contains("\"mtp.*\"") => {}
        other => errs.push(format!("exclude_modules lacks \"mtp.*\": {other:?}")),
    }
    errs
}

// ============================ CPU reference decoders (correctness-first) ============================

/// Unpack the e2m1 code for element `c` of a modelopt-packed row (elem 2i -> low nibble).
#[inline]
fn fp4_code(row: &[u8], c: usize) -> usize {
    let byte = row[c / 2];
    (if c & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as usize
}

/// modelopt NVFP4 -> f32: `e2m1(code) * e4m3(weight_scale[r, c/16]) * weight_scale_2`.
/// `rows`/`cols` are the LOGICAL dims (weight bytes are rows*cols/2). The per-16 groups run
/// along the IN (col) axis — forced by the scale-grid shape and pinned by the measured
/// power-of-two/pair-sharing structure the census gate checks (module header).
pub fn dequant_nvfp4_expert(
    weight: &[u8],
    wscale: &[u8],
    scale_2: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        rows * cols / 2,
        "nvfp4 weight bytes != rows*cols/2"
    );
    assert_eq!(
        wscale.len(),
        rows * cols / 16,
        "nvfp4 scale bytes != rows*cols/16"
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let wrow = &weight[r * cols / 2..(r + 1) * cols / 2];
        let srow = &wscale[r * cols / 16..(r + 1) * cols / 16];
        let orow = &mut out[r * cols..(r + 1) * cols];
        for (c, o) in orow.iter_mut().enumerate() {
            *o = E2M1[fp4_code(wrow, c)] * fp8_e4m3_to_f32(srow[c / 16]) * scale_2;
        }
    }
    out
}

/// OCP MXFP4 -> f32: `e2m1(code) * 2^(scale[r, c/32] - 127)` (MTP experts). Same nibble packing
/// as NVFP4; per-32 E8M0 groups along the IN axis. An 0xFF scale byte propagates NaN.
pub fn dequant_mxfp4_expert(weight: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        rows * cols / 2,
        "mxfp4 weight bytes != rows*cols/2"
    );
    assert_eq!(
        scale.len(),
        rows * cols / 32,
        "mxfp4 scale bytes != rows*cols/32"
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let wrow = &weight[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scale[r * cols / 32..(r + 1) * cols / 32];
        let orow = &mut out[r * cols..(r + 1) * cols];
        for (c, o) in orow.iter_mut().enumerate() {
            *o = E2M1[fp4_code(wrow, c)] * e8m0_to_f32(srow[c / 32]);
        }
    }
    out
}

/// FP8 E4M3 + 128x128 E8M0 block scales -> f32 (attention/shared-expert/indexer/MTP-projection
/// linears): `e4m3(byte) * 2^(scale[r/128, c/128] - 127)`. Dims must be 128-divisible (the
/// census enforces it before any decode).
pub fn dequant_fp8_blk128(weight: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert!(
        rows.is_multiple_of(128) && cols.is_multiple_of(128),
        "fp8 blk128 needs 128-divisible dims"
    );
    assert_eq!(weight.len(), rows * cols, "fp8 weight bytes != rows*cols");
    assert_eq!(
        scale.len(),
        (rows / 128) * (cols / 128),
        "fp8 scale bytes != grid"
    );
    let scol = cols / 128;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let srow = &scale[(r / 128) * scol..(r / 128 + 1) * scol];
        let wrow = &weight[r * cols..(r + 1) * cols];
        let orow = &mut out[r * cols..(r + 1) * cols];
        for (c, o) in orow.iter_mut().enumerate() {
            *o = fp8_e4m3_to_f32(wrow[c]) * e8m0_to_f32(srow[c / 128]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    /// A shape-faithful miniature of the Flash config.json: all the deepseek_v4 keys with the
    /// REAL artifact's values (so the census math is exercised on true geometry) — parsed through
    /// the same HfConfig path the gate uses. `tag` keeps parallel tests off each other's files.
    fn flash_config(tag: &str) -> ModelConfig {
        let dir = std::env::temp_dir().join(format!("memra_dsv4_cfg_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.json");
        let mut ratios = vec![0u32, 0];
        for il in 2..43u32 {
            ratios.push(if il % 2 == 0 { 4 } else { 128 });
        }
        ratios.push(0); // MTP layer
        let ratios_json = ratios
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            &p,
            format!(
                r#"{{"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4",
                "num_hidden_layers":43,"hidden_size":4096,"num_attention_heads":64,
                "num_key_value_heads":1,"head_dim":512,"vocab_size":129280,
                "max_position_embeddings":1048576,"rms_norm_eps":1e-6,"rope_theta":10000,
                "n_routed_experts":256,"n_shared_experts":1,"num_experts_per_tok":6,
                "moe_intermediate_size":2048,"norm_topk_prob":true,"num_hash_layers":3,
                "num_nextn_predict_layers":1,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
                "routed_scaling_factor":1.5,"hc_eps":1e-6,"hc_mult":4,"hc_sinkhorn_iters":20,
                "q_lora_rank":1024,"qk_rope_head_dim":64,"o_lora_rank":1024,"o_groups":8,
                "index_n_heads":64,"index_head_dim":128,"index_topk":512,
                "compress_ratios":[{ratios_json}],"compress_rope_theta":160000,
                "sliding_window":128,"swiglu_limit":10.0,
                "rope_scaling":{{"type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
                "original_max_position_embeddings":65536}}}}"#
            ),
        )
        .unwrap();
        let mc = ModelConfig::from_config_json(&p).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        mc
    }

    #[test]
    fn config_parses_all_dsv4_fields() {
        let mc = flash_config("fields");
        assert_eq!(mc.arch, crate::config::Arch::DeepSeekV4);
        assert_eq!(
            mc.n_layer, 44,
            "n_layer includes the MTP layer (GGUF convention)"
        );
        assert_eq!(mc.nextn_predict_layers, 1);
        let d = mc.dsv4.as_ref().unwrap();
        assert_eq!(d.scoring_func, "sqrtsoftplus");
        assert_eq!(d.topk_method, "noaux_tc");
        assert_eq!(d.hc_mult, 4);
        assert_eq!(d.hc_sinkhorn_iters, 20);
        assert!((d.hc_eps - 1e-6).abs() < 1e-12);
        assert_eq!(d.index_topk, 512);
        assert_eq!(d.compress_ratios.len(), 44);
        assert_eq!(d.num_hash_layers, 3);
        assert!((d.rope_yarn_factor - 16.0).abs() < 1e-6);
        assert_eq!(d.rope_yarn_orig_ctx, 65536);
        assert!(d.norm_topk_prob);
        assert!((d.routed_scaling_factor - 1.5).abs() < 1e-6);
        // derived splits (never hardcoded): 41 compressor, 21 indexer, hash layers 0..2,
        // MTP layer (il 43) carries neither.
        let n_comp = (0..44).filter(|&il| d.has_compressor(il)).count();
        let n_idx = (0..44).filter(|&il| d.has_indexer(il)).count();
        assert_eq!((n_comp, n_idx), (41, 21));
        assert!(!d.has_compressor(43) && !d.has_indexer(43));
        assert!(d.is_hash_layer(2) && !d.is_hash_layer(3));
        let moe = mc.moe.as_ref().unwrap();
        assert_eq!(
            (
                moe.expert_count,
                moe.expert_used_count,
                moe.expert_ff_length
            ),
            (256, 6, 2048)
        );
    }

    #[test]
    fn census_matches_artifact_totals() {
        let mc = flash_config("totals");
        let census = expected_census(&mc);
        // The real artifact's index counts 135,235 tensors across 46 shards (measured
        // 2026-08-18). The census must reproduce that total from config math alone.
        assert_eq!(
            census.len(),
            135_235,
            "census total != artifact tensor count"
        );
        // spot shapes straight from the artifact header census
        let s = &census["layers.20.ffn.experts.7.w1.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("U8", &[2048u64, 2048][..]));
        let s = &census["layers.20.ffn.experts.7.w1.weight_scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E4M3", &[2048u64, 256][..])
        );
        let s = &census["mtp.0.ffn.experts.7.w2.scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E8M0", &[4096u64, 64][..])
        );
        let s = &census["layers.0.ffn.gate.tid2eid"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("I64", &[129280u64, 6][..]));
        let s = &census["layers.42.attn.wq_b.scale"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F8_E8M0", &[256u64, 8][..]));
        let s = &census["layers.2.attn.indexer.wq_b.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E4M3", &[8192u64, 1024][..])
        );
        let s = &census["layers.20.hc_attn_fn"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F32", &[24u64, 16384][..]));
        // the two measured compressor shape classes (fine ratio-4 vs coarse ratio-128 — the
        // split the first on-box gate run caught; ape = [ratio, latent])
        let s = &census["layers.2.attn.compressor.wkv.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[1024u64, 4096][..])
        );
        let s = &census["layers.2.attn.compressor.ape"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F32", &[4u64, 1024][..]));
        let s = &census["layers.3.attn.compressor.wkv.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("BF16", &[512u64, 4096][..]));
        let s = &census["layers.3.attn.compressor.ape"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F32", &[128u64, 512][..]));
        let s = &census["layers.3.attn.compressor.norm.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("BF16", &[512u64][..]));
        // absent-by-derivation: no indexer on an odd (ratio-128) layer, no bias on hash layers,
        // no compressor on layers 0/1 or the MTP block.
        assert!(!census.contains_key("layers.3.attn.indexer.wq_b.weight"));
        assert!(!census.contains_key("layers.0.ffn.gate.bias"));
        assert!(!census.contains_key("layers.1.attn.compressor.wkv.weight"));
        assert!(!census.contains_key("mtp.0.attn.compressor.wkv.weight"));
        assert!(census.contains_key("mtp.0.ffn.gate.bias"));
    }

    #[test]
    fn e8m0_known_values() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(119), 2f32.powi(-8));
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert!(e8m0_to_f32(0xFF).is_nan(), "E8M0 NaN code must PROPAGATE");
    }

    /// The dsv4 NVFP4 decode must agree with the qwen-lane reference (`dequant_modelopt_row`,
    /// proven against the GGUF kernel path) for non-negative scale bytes, up to weight_scale_2.
    /// (The lanes differ only on sign-bit scale bytes, which modelopt never emits and this
    /// artifact measures as absent.)
    #[test]
    fn nvfp4_matches_qwen_lane_reference() {
        let (rows, cols) = (3usize, 64usize);
        let mut weight = vec![0u8; rows * cols / 2];
        let mut wscale = vec![0u8; rows * cols / 16];
        for (i, b) in weight.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8;
        }
        for (i, b) in wscale.iter_mut().enumerate() {
            *b = (0x20 + ((i * 13 + 3) % 0x50)) as u8; // finite, non-negative e4m3 codes
        }
        let s2 = 2f32.powi(-12); // the artifact's measured weight_scale_2
        let got = dequant_nvfp4_expert(&weight, &wscale, s2, rows, cols);
        for r in 0..rows {
            let want = crate::nvfp4_repack::dequant_modelopt_row(
                &weight[r * cols / 2..(r + 1) * cols / 2],
                &wscale[r * cols / 16..(r + 1) * cols / 16],
                cols,
            );
            for c in 0..cols {
                let w = want[c] * s2;
                let g = got[r * cols + c];
                assert!(
                    (w - g).abs() <= w.abs() * 1e-6,
                    "r{r} c{c}: qwen-lane {w} != dsv4 {g}"
                );
            }
        }
    }

    #[test]
    fn mxfp4_synthetic_decode() {
        // one row, 64 cols = two 32-groups with scales 2^1 and 2^-2; codes cycle the table.
        let mut weight = vec![0u8; 32];
        for (i, b) in weight.iter_mut().enumerate() {
            *b = (((2 * i + 1) % 16) << 4) as u8 | ((2 * i) % 16) as u8;
        }
        let scale = vec![128u8, 125u8];
        let out = dequant_mxfp4_expert(&weight, &scale, 1, 64);
        for c in 0..64 {
            let want = E2M1[c % 16] * if c < 32 { 2.0 } else { 0.25 };
            assert_eq!(out[c], want, "col {c}");
        }
    }

    #[test]
    fn fp8_blk128_synthetic_decode() {
        // 128x256: two scale blocks (2^2, 2^-3); constant weight byte 0x38 (= 1.0).
        let (rows, cols) = (128usize, 256usize);
        let weight = vec![0x38u8; rows * cols];
        let scale = vec![129u8, 124u8];
        let out = dequant_fp8_blk128(&weight, &scale, rows, cols);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[127], 4.0); // col 127: still scale block 0
        assert_eq!(out[128], 0.125); // col 128 -> block 1
        assert_eq!(out[rows * cols - 1], 0.125);
        // NaN scale must poison, not zero
        let bad = dequant_fp8_blk128(&weight, &[129u8, 0xFF], rows, cols);
        assert!(bad[128].is_nan());
    }
}
