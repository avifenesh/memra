//! dsv4-mint: DeepSeek-V4-Flash-0731 MXFP4 -> NVFP4 lossless mint + 0731 census gates.
//! Mint lane (research/deepseek-flash-20260818/0731-PREP.md), CPU only, no GPU.
//!
//! This bin deliberately lives in ONE new file (auto-discovered bin target) and touches no
//! shared module: lane-7 (native GEMM) is running in this worktree concurrently. It reuses the
//! lane-1 bit-proven decoders from `memra_gguf::dsv4` read-only.
//!
//! ── the cast (house-measured recipe == nvidia's preview cast, 0731-PREP.md §4.1) ──
//! Source (0731 trunk routed experts, OCP MXFP4):
//!   <stem>.weight  I8       [out, in/2]   e2m1 nibble pairs (elem 2i -> low nibble)
//!   <stem>.scale   F8_E8M0  [out, in/32]  per-32 scale along IN, s = 2^(b-127); 0xFF = NaN
//! Output (modelopt NVFP4, no input_scale — see the §2 decision note below):
//!   <stem>.weight          U8       [out, in/2]   bytes copied VERBATIM (dtype label flips)
//!   <stem>.weight_scale    F8_E4M3  [out, in/16]  per-32 scale duplicated into two per-16
//!                                                 slots, encoded as e4m3(s / scale_2)
//!   <stem>.weight_scale_2  F32      []            2^(floor(log2(max_s)) - 8): every quotient
//!                                                 s/scale_2 is an exact power of two in
//!                                                 [2^-9, 2^8], all exactly representable in
//!                                                 e4m3 — REFUSE on anything else (0xFF scale,
//!                                                 >17-octave range). Zero rounding anywhere.
//! Everything else — FP8-blk linears, BF16, F32, I64, and the dspark (mtp.*) MXFP4 experts —
//! is copied byte-for-byte (mirrors nvidia's exclude_modules).
//!
//! input_scale decision (0731-PREP.md §2, measured on-box 2026-08-18): nvidia's preview
//! artifact ships 33,024 F32 input_scale scalars with 814 DISTINCT values (range ~1.7e-4..) —
//! CALIBRATED activation stats, not a constant. A lossless cast has no calibration story, so
//! the mint OMITS input_scale; engines running dynamic per-128 activation quant (the reference
//! kernel law, vLLM/SGLang NVFP4 paths) do not read it.
//!
//! ── the 0731 config trap (0731-PREP.md §1.1) ──
//! 0731 still ships the vestigial `num_nextn_predict_layers: 1`, but its drafter is 3 DSpark
//! blocks: n_drafter = len(compress_ratios) - num_hidden_layers = 46 - 43 = 3 (the repo's own
//! inference/config.json: n_mtp_layers 3). `load_0731_config` derives n_drafter from the
//! ratios and parses through the existing ModelConfig with the vestigial key PATCHED to the
//! derived value in a temp copy — config.json itself ships byte-identical to the source.
//!
//! Subcommands:
//!   dsv4_mint cast <src-dir> <shard-file> <out-dir> [--threads N]
//!       cast one shard, fsync, RE-READ the written file and verify: pass-through tensors
//!       byte-identical; every expert projection (a) weight bytes identical, (b) every
//!       effective scale e4m3*scale_2 exactly equal to the source 2^(b-127) with per-16 pairs
//!       identical, (c) FULL decode bit-identity dequant_nvfp4(out) == dequant_mxfp4(in) via
//!       the lane-1 decoders, every element of every block. First failure = exit 1.
//!   dsv4_mint census <model-dir> --recipe source|minted [--threads N]
//!       Gate A for 0731: full expected-census (trunk = lane-1 rows with experts in the
//!       recipe's container; dspark blocks mtp.0..2 per the prep §1.2 table; NO NextN heads),
//!       sample decodes with distribution bounds, tid2eid range check; minted mode adds the
//!       hf_quant_config declaration check + NVFP4 scale-structure samples; source mode adds a
//!       full E8M0 0xFF scan over every expert scale tensor.
//!   dsv4_mint index <out-dir>
//!       regenerate model.safetensors.index.json (weight_map + exact metadata.total_size)
//!       from the written shard headers.

use memra_gguf::config::ModelConfig;
use memra_gguf::dsv4::{
    TensorSpec, dequant_fp8_blk128, dequant_mxfp4_expert, dequant_nvfp4_expert, e8m0_to_f32,
    verify_census, verify_quant_declaration,
};
use memra_gguf::nvfp4_repack::fp8_e4m3_to_f32;
use memra_gguf::safetensors::{StModel, StShard};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// ============================ tiny flat-JSON field readers ============================
// config.json is flat for every key we need. These helpers refuse on ambiguity (0 or >1
// occurrences) instead of guessing.

fn find_key_value_start(txt: &str, key: &str) -> usize {
    let pat = format!("\"{key}\"");
    let mut hits = txt.match_indices(&pat).map(|(i, _)| i);
    let first = hits
        .next()
        .unwrap_or_else(|| panic!("config.json: key {key} not found"));
    assert!(
        hits.next().is_none(),
        "config.json: key {key} occurs more than once"
    );
    let after = first + pat.len();
    let rest = &txt[after..];
    let colon = rest
        .find(':')
        .unwrap_or_else(|| panic!("config.json: no ':' after {key}"));
    after + colon + 1
}

fn json_u64(txt: &str, key: &str) -> u64 {
    let start = find_key_value_start(txt, key);
    let s = txt[start..].trim_start();
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    s[..end]
        .parse()
        .unwrap_or_else(|e| panic!("config.json: {key} not a u64: {e}"))
}

fn json_u64_array(txt: &str, key: &str) -> Vec<u64> {
    let start = find_key_value_start(txt, key);
    let s = txt[start..].trim_start();
    assert!(s.starts_with('['), "config.json: {key} is not an array");
    let close = s.find(']').unwrap_or_else(|| panic!("{key}: no closing ]"));
    s[1..close]
        .split(',')
        .map(|t| {
            t.trim()
                .parse()
                .unwrap_or_else(|e| panic!("config.json: {key} entry not u64: {e}"))
        })
        .collect()
}

// ============================ 0731 config load (vestigial-key patch) ============================

#[derive(Debug, Clone)]
struct Dspark {
    n_drafter: u32,   // len(compress_ratios) - num_hidden_layers (3 on 0731)
    tap_n: u64,       // len(dspark_target_layer_ids) (3) — main_proj in = tap_n * hidden
    markov_rank: u64, // 256
    #[allow(dead_code)]
    block_size: u64, // 5
    noise_token_id: u64, // 128799 — checked against vocab
}

/// Parse the 0731 config through the existing ModelConfig with `num_nextn_predict_layers`
/// patched to the DERIVED drafter count (the key is vestigial and wrong on 0731 — module
/// header). The artifact's own config.json is never modified.
fn load_0731_config(dir: &Path) -> (ModelConfig, Dspark) {
    let cfg_path = dir.join("config.json");
    let txt = std::fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display()));
    let nhl = json_u64(&txt, "num_hidden_layers");
    let ratios = json_u64_array(&txt, "compress_ratios");
    assert!(
        ratios.len() as u64 > nhl,
        "compress_ratios len {} must exceed num_hidden_layers {nhl}",
        ratios.len()
    );
    let n_drafter = (ratios.len() as u64 - nhl) as u32;
    for (k, &r) in ratios[nhl as usize..].iter().enumerate() {
        assert_eq!(
            r, 0,
            "drafter compress_ratios entry {k} is {r}, expected 0 (window-only dspark block)"
        );
    }
    let dspark = Dspark {
        n_drafter,
        tap_n: json_u64_array(&txt, "dspark_target_layer_ids").len() as u64,
        markov_rank: json_u64(&txt, "dspark_markov_rank"),
        block_size: json_u64(&txt, "dspark_block_size"),
        noise_token_id: json_u64(&txt, "dspark_noise_token_id"),
    };
    assert_eq!(
        dspark.n_drafter as u64, dspark.tap_n,
        "n_drafter != len(dspark_target_layer_ids) — new geometry, refuse"
    );

    // Patch the vestigial key's VALUE only, in a temp copy, and parse through the normal path.
    let vstart = find_key_value_start(&txt, "num_nextn_predict_layers");
    let s = txt[vstart..].trim_start();
    let digits = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    assert!(digits > 0, "num_nextn_predict_layers has no numeric value");
    let ws = txt[vstart..].len() - s.len(); // leading whitespace kept as-is
    let patched = format!(
        "{}{}{}{}",
        &txt[..vstart],
        &txt[vstart..vstart + ws],
        n_drafter,
        &s[digits..]
    );
    static CFG_SEQ: AtomicUsize = AtomicUsize::new(0);
    let tmp = std::env::temp_dir().join(format!(
        "dsv4_mint_cfg_{}_{}",
        std::process::id(),
        CFG_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&tmp).expect("mk temp cfg dir");
    let tmp_cfg = tmp.join("config.json");
    std::fs::write(&tmp_cfg, &patched).expect("write patched config");
    let mc = ModelConfig::from_config_json(&tmp_cfg).expect("parse patched 0731 config");
    std::fs::remove_dir_all(&tmp).ok();

    assert_eq!(mc.nextn_predict_layers, n_drafter);
    assert_eq!(mc.n_layer as u64, ratios.len() as u64);
    assert!(
        dspark.noise_token_id < mc.n_vocab as u64,
        "dspark_noise_token_id {} outside vocab {}",
        dspark.noise_token_id,
        mc.n_vocab
    );
    (mc, dspark)
}

// ============================ 0731 expected census ============================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertRecipe {
    /// deepseek's own container: I8 e2m1 pairs + F8_E8M0 per-32 (source 0731, and dspark
    /// experts in BOTH source and mint).
    Mxfp4,
    /// house mint output: U8 + F8_E4M3 per-16 + F32 scale_2, NO input_scale (module header).
    Nvfp4NoInputScale,
}

/// Full expected tensor map for a 0731-shaped checkpoint. Trunk rows are the lane-1 census
/// rows verbatim (dsv4.rs `expected_census` — duplicated here rather than parameterizing the
/// shared module while lane-7 runs in this worktree); experts follow `trunk_recipe`; the
/// drafter is 3 DSpark blocks per the prep §1.2 table — NO NextN e_proj/h_proj/enorm/hnorm
/// anywhere (their presence fails the census as EXTRA).
fn expected_census_0731(
    mc: &ModelConfig,
    dspark: &Dspark,
    trunk_recipe: ExpertRecipe,
) -> BTreeMap<String, TensorSpec> {
    let d = mc.dsv4.as_ref().expect("dsv4 config");
    let moe = mc.moe.as_ref().expect("moe config");
    let h = mc.n_embd as u64;
    let v = mc.n_vocab as u64;
    let nh = mc.n_head as u64;
    let hd = d.head_dim as u64;
    let qlr = d.q_lora_rank as u64;
    let kv_out = d.num_key_value_heads as u64 * hd;
    let o_out = d.o_groups as u64 * d.o_lora_rank as u64;
    let ihd = d.index_head_dim as u64;
    let inh = d.index_n_heads as u64;
    let ne = moe.expert_count as u64;
    let topk = moe.expert_used_count as u64;
    let mff = moe.expert_ff_length as u64;
    let sff = d.n_shared_experts as u64 * mff;
    let hcm = d.hc_mult as u64;
    let hc_rows: u64 = (1..=hcm).product(); // hc_mult! (lane-1 formula)
    let hc_scale = hcm - 1;
    let hc_w = hcm * h;

    let mut map: BTreeMap<String, TensorSpec> = BTreeMap::new();
    fn t(m: &mut BTreeMap<String, TensorSpec>, name: String, dtype: &'static str, shape: Vec<u64>) {
        let prev = m.insert(name.clone(), TensorSpec { dtype, shape });
        assert!(prev.is_none(), "0731 census: duplicate spec for {name}");
    }
    fn fp8(m: &mut BTreeMap<String, TensorSpec>, stem: String, out: u64, inn: u64) {
        assert!(
            out.is_multiple_of(128) && inn.is_multiple_of(128),
            "fp8 blk128 dims: {stem}"
        );
        t(m, format!("{stem}.weight"), "F8_E4M3", vec![out, inn]);
        t(
            m,
            format!("{stem}.scale"),
            "F8_E8M0",
            vec![out / 128, inn / 128],
        );
    }
    fn expert(
        m: &mut BTreeMap<String, TensorSpec>,
        stem: String,
        out: u64,
        inn: u64,
        recipe: ExpertRecipe,
    ) {
        match recipe {
            ExpertRecipe::Mxfp4 => {
                assert!(inn.is_multiple_of(32));
                t(m, format!("{stem}.weight"), "I8", vec![out, inn / 2]);
                t(m, format!("{stem}.scale"), "F8_E8M0", vec![out, inn / 32]);
            }
            ExpertRecipe::Nvfp4NoInputScale => {
                assert!(inn.is_multiple_of(16));
                t(m, format!("{stem}.weight"), "U8", vec![out, inn / 2]);
                t(
                    m,
                    format!("{stem}.weight_scale"),
                    "F8_E4M3",
                    vec![out, inn / 16],
                );
                t(m, format!("{stem}.weight_scale_2"), "F32", vec![]);
            }
        }
    }

    // ---- globals (lane-1 rows) ----
    t(&mut map, "embed.weight".into(), "BF16", vec![v, h]);
    t(&mut map, "head.weight".into(), "BF16", vec![v, h]);
    t(&mut map, "norm.weight".into(), "BF16", vec![h]);
    t(&mut map, "hc_head_base".into(), "F32", vec![hcm]);
    t(&mut map, "hc_head_fn".into(), "F32", vec![hcm, hc_w]);
    t(&mut map, "hc_head_scale".into(), "F32", vec![1]);

    let n_trunk = mc.n_layer - mc.nextn_predict_layers;
    // Shared attention + hc + MoE block (lane-1 geometry; `il` is the GLOBAL layer index so
    // the compressor/indexer/hash splits derive from compress_ratios, never from counts).
    let block = |m: &mut BTreeMap<String, TensorSpec>, p: &str, il: u32, recipe: ExpertRecipe| {
        t(m, format!("{p}attn_norm.weight"), "BF16", vec![h]);
        t(m, format!("{p}ffn_norm.weight"), "BF16", vec![h]);
        t(m, format!("{p}attn.q_norm.weight"), "BF16", vec![qlr]);
        t(m, format!("{p}attn.kv_norm.weight"), "BF16", vec![kv_out]);
        t(m, format!("{p}attn.attn_sink"), "F32", vec![nh]);
        fp8(m, format!("{p}attn.wq_a"), qlr, h);
        fp8(m, format!("{p}attn.wq_b"), nh * hd, qlr);
        fp8(m, format!("{p}attn.wkv"), kv_out, h);
        fp8(m, format!("{p}attn.wo_a"), o_out, h);
        fp8(m, format!("{p}attn.wo_b"), h, o_out);
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
        t(m, format!("{p}hc_attn_base"), "F32", vec![hc_rows]);
        t(m, format!("{p}hc_attn_fn"), "F32", vec![hc_rows, hc_w]);
        t(m, format!("{p}hc_attn_scale"), "F32", vec![hc_scale]);
        t(m, format!("{p}hc_ffn_base"), "F32", vec![hc_rows]);
        t(m, format!("{p}hc_ffn_fn"), "F32", vec![hc_rows, hc_w]);
        t(m, format!("{p}hc_ffn_scale"), "F32", vec![hc_scale]);
        t(m, format!("{p}ffn.gate.weight"), "BF16", vec![ne, h]);
        if d.is_hash_layer(il) {
            t(m, format!("{p}ffn.gate.tid2eid"), "I64", vec![v, topk]);
        } else {
            t(m, format!("{p}ffn.gate.bias"), "F32", vec![ne]);
        }
        fp8(m, format!("{p}ffn.shared_experts.w1"), sff, h);
        fp8(m, format!("{p}ffn.shared_experts.w2"), h, sff);
        fp8(m, format!("{p}ffn.shared_experts.w3"), sff, h);
        for e in 0..ne {
            expert(m, format!("{p}ffn.experts.{e}.w1"), mff, h, recipe);
            expert(m, format!("{p}ffn.experts.{e}.w2"), h, mff, recipe);
            expert(m, format!("{p}ffn.experts.{e}.w3"), mff, h, recipe);
        }
    };

    for il in 0..n_trunk {
        block(&mut map, &format!("layers.{il}."), il, trunk_recipe);
    }
    // DSpark drafter blocks: window-only MoE blocks (ratio 0 -> no compressor/indexer, score
    // routing -> gate.bias), experts ALWAYS MXFP4 (excluded from the cast, prep §4.1).
    for k in 0..mc.nextn_predict_layers {
        let p = format!("mtp.{k}.");
        block(&mut map, &p, n_trunk + k, ExpertRecipe::Mxfp4);
    }
    // First block: the trunk tap projection (in = tap_n * hidden = 12288) + its norm.
    fp8(&mut map, "mtp.0.main_proj".into(), h, dspark.tap_n * h);
    t(&mut map, "mtp.0.main_norm.weight".into(), "BF16", vec![h]);
    // Last block: the drafter head — markov bigram factors, confidence head, own hc-head
    // collapse + output norm (prep §1.2 new-tensor table).
    let last = mc.nextn_predict_layers - 1;
    let r = dspark.markov_rank;
    t(
        &mut map,
        format!("mtp.{last}.markov_head.markov_w1.weight"),
        "BF16",
        vec![v, r],
    );
    t(
        &mut map,
        format!("mtp.{last}.markov_head.markov_w2.weight"),
        "BF16",
        vec![v, r],
    );
    t(
        &mut map,
        format!("mtp.{last}.confidence_head.proj.weight"),
        "BF16",
        vec![1, h + r],
    );
    t(
        &mut map,
        format!("mtp.{last}.hc_head_base"),
        "F32",
        vec![hcm],
    );
    t(
        &mut map,
        format!("mtp.{last}.hc_head_fn"),
        "F32",
        vec![hcm, hc_w],
    );
    t(
        &mut map,
        format!("mtp.{last}.hc_head_scale"),
        "F32",
        vec![1],
    );
    t(&mut map, format!("mtp.{last}.norm.weight"), "BF16", vec![h]);
    map
}

// ============================ the cast ============================

/// e4m3 code for an EXACT power of two 2^q. Representable pow2 range is [2^-9, 2^8]
/// (subnormal mantissas 1/2/4 at exp field 0; normals (q+7)<<3). Anything else refuses.
/// Every emitted code is roundtrip-asserted through the lane-1 e4m3 decoder.
fn e4m3_pow2_code(q: i32) -> Result<u8, String> {
    let code: u8 = match q {
        -6..=8 => ((q + 7) as u8) << 3,
        -7 => 0x04,
        -8 => 0x02,
        -9 => 0x01,
        _ => {
            return Err(format!(
                "2^{q} not exactly representable in e4m3 (pow2 range [2^-9, 2^8])"
            ));
        }
    };
    let back = fp8_e4m3_to_f32(code);
    assert_eq!(
        back,
        (q as f32).exp2(),
        "e4m3 pow2 roundtrip failed for q={q} code={code:#04x}"
    );
    Ok(code)
}

/// Cast one MXFP4 per-32 E8M0 scale tensor to the NVFP4 per-16 E4M3 + F32 scale_2 pair.
/// Flat duplication preserves row structure exactly (out row len == 2 * in row len).
/// Returns (per-16 e4m3 bytes, scale_2). REFUSES (Err) on 0xFF NaN codes and on any scale
/// whose quotient falls outside the exact-pow2 e4m3 range — never rounds.
fn cast_expert_scale(scale: &[u8], name: &str) -> Result<(Vec<u8>, f32), String> {
    let mut max_b = 0u8;
    for (i, &b) in scale.iter().enumerate() {
        if b == 0xFF {
            return Err(format!("{name}: E8M0 NaN code 0xFF at scale byte {i}"));
        }
        max_b = max_b.max(b);
    }
    // scale_2 = 2^(floor(log2(max_s)) - 8); max_s = 2^(max_b - 127) so m = max_b - 127 - 8.
    let m = max_b as i32 - 127 - 8;
    let mut out = Vec::with_capacity(scale.len() * 2);
    for (i, &b) in scale.iter().enumerate() {
        let q = (b as i32 - 127) - m;
        let code =
            e4m3_pow2_code(q).map_err(|e| format!("{name}: scale byte {i} (E8M0 {b}): {e}"))?;
        out.push(code);
        out.push(code);
    }
    Ok((out, (m as f32).exp2()))
}

/// Trunk routed-expert projection weights are the ONLY cast targets:
/// `layers.<L>.ffn.experts.<E>.w{1,2,3}` — everything else (incl. mtp.*) passes through.
fn is_trunk_expert_stem(stem: &str) -> bool {
    if !stem.starts_with("layers.") || !stem.contains(".ffn.experts.") {
        return false;
    }
    matches!(
        stem.rsplit('.').next(),
        Some("w1") | Some("w2") | Some("w3")
    )
}

enum Bytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl Bytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Borrowed(b) => b,
            Bytes::Owned(v) => v,
        }
    }
}

struct OutEntry<'a> {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: Bytes<'a>,
}

/// Write a safetensors file: 8-byte LE header length, JSON header (entry order = storage
/// order, `__metadata__ {"format":"pt"}`, header space-padded to an 8-byte multiple per the
/// HF writer convention), then the tensor bytes back to back. fsyncs before returning.
fn write_safetensors(path: &Path, entries: &[OutEntry]) -> std::io::Result<u64> {
    let mut header = String::with_capacity(entries.len() * 96);
    header.push_str("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0u64;
    for e in entries {
        let n = e.bytes.as_slice().len() as u64;
        let shape = e
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            ",\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            e.name,
            e.dtype,
            shape,
            off,
            off + n
        ));
        off += n;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for e in entries {
        w.write_all(e.bytes.as_slice())?;
    }
    w.flush()?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    Ok(8 + header.len() as u64 + off)
}

/// Names of a shard in STORAGE order (by data_offsets begin).
fn shard_names_in_order(sh: &StShard) -> Vec<String> {
    let mut v: Vec<(usize, String)> = sh
        .names()
        .map(|n| (sh.raw(n).unwrap().0.data_offsets[0], n.clone()))
        .collect();
    v.sort();
    v.into_iter().map(|(_, n)| n).collect()
}

fn par_run<T: Sync>(
    items: &[T],
    threads: usize,
    f: impl Fn(usize, &T) -> Result<String, String> + Sync,
) -> Result<Vec<String>, String> {
    let idx = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<Result<String, String>>>> =
        (0..items.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let i = idx.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let r = f(i, &items[i]);
                    *out[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    let mut lines = Vec::with_capacity(items.len());
    for m in out {
        {
            let l = m.into_inner().unwrap().expect("worker skipped an item")?;
            lines.push(l)
        }
    }
    Ok(lines)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// cast subcommand: cast + write + re-read + full verification of one shard.
fn cmd_cast(src_dir: &Path, shard: &str, out_dir: &Path, threads: usize) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let src_path = src_dir.join(shard);
    let sh = StShard::open(&src_path).map_err(|e| format!("open {shard}: {e}"))?;
    let order = shard_names_in_order(&sh);

    // ---- build output entries (cast scales serially — ~180 MB/shard; decode-verify is the
    // parallel part below) ----
    let mut entries: Vec<OutEntry> = Vec::with_capacity(order.len() + 512);
    let mut cast_stems: Vec<(String, usize, usize)> = Vec::new(); // (stem, rows, cols)
    for name in &order {
        let (info, bytes) = sh.raw(name).unwrap();
        let stem_scale = name.strip_suffix(".scale");
        let stem_weight = name.strip_suffix(".weight");
        if let Some(stem) = stem_weight.filter(|s| is_trunk_expert_stem(s)) {
            assert_eq!(info.dtype, "I8", "{name}: trunk expert weight not I8");
            let (rows, cols) = (info.shape[0] as usize, info.shape[1] as usize * 2);
            cast_stems.push((stem.to_string(), rows, cols));
            entries.push(OutEntry {
                name: name.clone(),
                dtype: "U8".into(), // dtype label flips; bytes verbatim
                shape: info.shape.clone(),
                bytes: Bytes::Borrowed(bytes),
            });
        } else if let Some(stem) = stem_scale.filter(|s| is_trunk_expert_stem(s)) {
            assert_eq!(info.dtype, "F8_E8M0", "{name}: trunk expert scale not E8M0");
            let (s16, s2) = cast_expert_scale(bytes, name)?;
            entries.push(OutEntry {
                name: format!("{stem}.weight_scale"),
                dtype: "F8_E4M3".into(),
                shape: vec![info.shape[0], info.shape[1] * 2],
                bytes: Bytes::Owned(s16),
            });
            entries.push(OutEntry {
                name: format!("{stem}.weight_scale_2"),
                dtype: "F32".into(),
                shape: vec![],
                bytes: Bytes::Owned(s2.to_le_bytes().to_vec()),
            });
        } else {
            entries.push(OutEntry {
                name: name.clone(),
                dtype: info.dtype.clone(),
                shape: info.shape.clone(),
                bytes: Bytes::Borrowed(bytes),
            });
        }
    }

    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir out: {e}"))?;
    let out_path = out_dir.join(shard);
    let out_bytes =
        write_safetensors(&out_path, &entries).map_err(|e| format!("write {shard}: {e}"))?;
    println!(
        "cast: {} tensors in -> {} out ({} projections cast), wrote {} bytes in {:.1}s",
        order.len(),
        entries.len(),
        cast_stems.len(),
        out_bytes,
        t0.elapsed().as_secs_f64()
    );
    drop(entries);

    // ---- verify what shipped: re-open the WRITTEN file ----
    let tv = std::time::Instant::now();
    let out_sh = StShard::open(&out_path).map_err(|e| format!("reopen {shard}: {e}"))?;

    // pass-through tensors byte-identical (and expert weights, which are also verbatim)
    let mut n_copy = 0usize;
    for name in &order {
        if name
            .strip_suffix(".scale")
            .is_some_and(is_trunk_expert_stem)
        {
            continue; // replaced by weight_scale/_2, verified below
        }
        let (ii, ib) = sh.raw(name).unwrap();
        let (oi, ob) = out_sh
            .raw(name)
            .ok_or_else(|| format!("{name}: missing from written shard"))?;
        let dtype_ok = if name
            .strip_suffix(".weight")
            .is_some_and(is_trunk_expert_stem)
        {
            oi.dtype == "U8"
        } else {
            oi.dtype == ii.dtype
        };
        if !dtype_ok || oi.shape != ii.shape {
            return Err(format!(
                "{name}: header changed ({}/{:?} -> {}/{:?})",
                ii.dtype, ii.shape, oi.dtype, oi.shape
            ));
        }
        if ib != ob {
            return Err(format!("{name}: bytes differ after write"));
        }
        n_copy += 1;
    }

    // full per-projection losslessness: scale structure + bit-identical decode (lane-1
    // decoders), every element of every block
    let lines = par_run(&cast_stems, threads, |_, (stem, rows, cols)| {
        let (_, w_in) = sh.raw(&format!("{stem}.weight")).unwrap();
        let (_, s_in) = sh.raw(&format!("{stem}.scale")).unwrap();
        let (_, w_out) = out_sh
            .raw(&format!("{stem}.weight"))
            .ok_or_else(|| format!("{stem}.weight missing from output"))?;
        let (_, s_out) = out_sh
            .raw(&format!("{stem}.weight_scale"))
            .ok_or_else(|| format!("{stem}.weight_scale missing from output"))?;
        let (_, s2b) = out_sh
            .raw(&format!("{stem}.weight_scale_2"))
            .ok_or_else(|| format!("{stem}.weight_scale_2 missing from output"))?;
        let s2 = f32::from_le_bytes(s2b.try_into().map_err(|_| "scale_2 not 4B")?);
        if w_in != w_out {
            return Err(format!("{stem}: weight bytes differ"));
        }
        if s_out.len() != s_in.len() * 2 {
            return Err(format!("{stem}: weight_scale length != 2x source"));
        }
        // structural: every effective scale exactly equals the source pow2; pairs identical
        for (j, &b) in s_in.iter().enumerate() {
            let (a, c) = (s_out[2 * j], s_out[2 * j + 1]);
            if a != c {
                return Err(format!("{stem}: per-16 pair {j} not identical"));
            }
            let eff = fp8_e4m3_to_f32(a) * s2;
            let src = e8m0_to_f32(b);
            if eff != src || !eff.is_finite() || eff <= 0.0 {
                return Err(format!(
                    "{stem}: effective scale group {j}: {eff:e} != source {src:e}"
                ));
            }
        }
        // the bit claim: full decode identity through the lane-1 bit-proven decoders
        let a = dequant_mxfp4_expert(w_in, s_in, *rows, *cols);
        let b = dequant_nvfp4_expert(w_out, s_out, s2, *rows, *cols);
        for i in 0..a.len() {
            if a[i].to_bits() != b[i].to_bits() {
                return Err(format!(
                    "{stem}: decode mismatch at element {i}: {} != {} — NOT lossless, STOP",
                    a[i], b[i]
                ));
            }
        }
        let mut hsh = Sha256::new();
        let mut buf = [0u8; 4 * 4096];
        for chunk in a.chunks(4096) {
            for (i, x) in chunk.iter().enumerate() {
                buf[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
            }
            hsh.update(&buf[..4 * chunk.len()]);
        }
        let blocks = (rows * cols / 32) as u64;
        Ok(format!(
            "{stem}: m={} blocks={blocks} lossless=100.0000% sha256(f32)={}",
            s2.log2() as i32,
            hex(&hsh.finalize())
        ))
    })?;
    for l in &lines {
        println!("{l}");
    }
    let blocks: u64 = cast_stems.iter().map(|(_, r, c)| (r * c / 32) as u64).sum();
    println!(
        "SHARD {shard}: PASS — {} projections cast+verified bit-identical ({} blocks lossless), \
         {} tensors byte-identical pass-through, verify {:.1}s, total {:.1}s",
        cast_stems.len(),
        blocks,
        n_copy,
        tv.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

// ============================ index regeneration ============================

fn cmd_index(out_dir: &Path) -> Result<(), String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(out_dir)
        .map_err(|e| format!("read_dir: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
        })
        .collect();
    files.sort();
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    let mut total: u64 = 0;
    for f in &files {
        let sh = StShard::open(f).map_err(|e| format!("open {}: {e}", f.display()))?;
        let fname = f.file_name().unwrap().to_str().unwrap().to_string();
        for n in sh.names() {
            let (info, bytes) = sh.raw(n).unwrap();
            assert_eq!(
                bytes.len() as u64,
                info.data_offsets[1] as u64 - info.data_offsets[0] as u64
            );
            total += bytes.len() as u64;
            let prev = weight_map.insert(n.clone(), fname.clone());
            assert!(prev.is_none(), "tensor {n} in two shards");
        }
    }
    let mut js = String::new();
    js.push_str("{\n  \"metadata\": {\n    \"total_size\": ");
    js.push_str(&total.to_string());
    js.push_str("\n  },\n  \"weight_map\": {\n");
    let n = weight_map.len();
    for (i, (k, v)) in weight_map.iter().enumerate() {
        js.push_str(&format!(
            "    \"{k}\": \"{v}\"{}\n",
            if i + 1 < n { "," } else { "" }
        ));
    }
    js.push_str("  }\n}\n");
    std::fs::write(out_dir.join("model.safetensors.index.json"), &js)
        .map_err(|e| format!("write index: {e}"))?;
    println!(
        "index: {} tensors across {} shards, total_size {total}",
        weight_map.len(),
        files.len()
    );
    Ok(())
}

// ============================ census gate (0731 source / minted) ============================

fn bf16_to_f32_vec(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn f32_vec(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Same sibling-set inference as dsv4-census (weight_scale_2 -> NVFP4; I8 -> MXFP4;
/// F8_E4M3 -> FP8 128x128).
fn decode_quant_stem(m: &StModel, stem: &str) -> Result<(Vec<f32>, usize, usize), String> {
    let (wi, wb) = m
        .raw(&format!("{stem}.weight"))
        .ok_or_else(|| format!("{stem}.weight: missing"))?;
    let rows = wi.shape[0] as usize;
    if m.raw(&format!("{stem}.weight_scale_2")).is_some() {
        let cols = wi.shape[1] as usize * 2;
        let (_, sb) = m
            .raw(&format!("{stem}.weight_scale"))
            .ok_or_else(|| format!("{stem}.weight_scale: missing"))?;
        let (_, s2b) = m.raw(&format!("{stem}.weight_scale_2")).unwrap();
        let s2 = f32::from_le_bytes(s2b.try_into().map_err(|_| "scale_2 not 4B")?);
        Ok((dequant_nvfp4_expert(wb, sb, s2, rows, cols), rows, cols))
    } else if wi.dtype == "I8" {
        let cols = wi.shape[1] as usize * 2;
        let (_, sb) = m
            .raw(&format!("{stem}.scale"))
            .ok_or_else(|| format!("{stem}.scale: missing"))?;
        Ok((dequant_mxfp4_expert(wb, sb, rows, cols), rows, cols))
    } else if wi.dtype == "F8_E4M3" {
        let cols = wi.shape[1] as usize;
        let (_, sb) = m
            .raw(&format!("{stem}.scale"))
            .ok_or_else(|| format!("{stem}.scale: missing"))?;
        Ok((dequant_fp8_blk128(wb, sb, rows, cols), rows, cols))
    } else {
        Err(format!("{stem}: no recognizable quant sibling set"))
    }
}

struct Stats {
    absmean: f64,
    zerofrac: f64,
    nan: usize,
    inf: usize,
    min: f32,
    max: f32,
}

fn stats(v: &[f32]) -> Stats {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut asum = 0f64;
    let (mut zeros, mut nan, mut inf) = (0usize, 0usize, 0usize);
    for &x in v {
        if x.is_nan() {
            nan += 1;
            continue;
        }
        if x.is_infinite() {
            inf += 1;
            continue;
        }
        min = min.min(x);
        max = max.max(x);
        asum += x.abs() as f64;
        if x == 0.0 {
            zeros += 1;
        }
    }
    let n = v.len().max(1) as f64;
    Stats {
        absmean: asum / n,
        zerofrac: zeros as f64 / n,
        nan,
        inf,
        min,
        max,
    }
}

fn cmd_census(dir: &Path, recipe: ExpertRecipe) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let (mc, dspark) = load_0731_config(dir);
    let d = mc.dsv4.as_ref().unwrap();
    let n_trunk = mc.n_layer - mc.nextn_predict_layers;
    println!(
        "config: {n_trunk} trunk + {} dspark blocks (derived from compress_ratios len {} - {} \
         hidden; vestigial num_nextn_predict_layers IGNORED), tap_n {}, markov rank {}, hidden \
         {}, {} experts, scoring {}",
        mc.nextn_predict_layers,
        d.compress_ratios.len(),
        n_trunk,
        dspark.tap_n,
        dspark.markov_rank,
        mc.n_embd,
        mc.moe.as_ref().unwrap().expert_count,
        d.scoring_func,
    );
    let mut failures: Vec<String> = Vec::new();

    // minted mode: the artifact must DECLARE the recipe split we minted
    if recipe == ExpertRecipe::Nvfp4NoInputScale {
        let decl = verify_quant_declaration(dir, &mc);
        if decl.is_empty() {
            println!(
                "hf_quant_config.json: NVFP4/group-16 declared on all {n_trunk} trunk expert \
                 banks; mtp.* excluded — matches census"
            );
        } else {
            for e in decl {
                failures.push(format!("quant declaration: {e}"));
            }
        }
    }

    let m = StModel::open(dir).map_err(|e| format!("open model: {e}"))?;
    println!("opened {} tensors across shards", m.n_tensors());
    let census = expected_census_0731(&mc, &dspark, recipe);
    let report = verify_census(&m, &census);
    println!(
        "census: {} expected | missing {} | extra {} | mismatched {}",
        census.len(),
        report.missing.len(),
        report.extra.len(),
        report.mismatched.len()
    );
    for (kind, list) in [
        ("MISSING", &report.missing),
        ("EXTRA", &report.extra),
        ("MISMATCH", &report.mismatched),
    ] {
        for e in list.iter().take(20) {
            failures.push(format!("census {kind}: {e}"));
        }
        if list.len() > 20 {
            failures.push(format!("census {kind}: ... and {} more", list.len() - 20));
        }
    }

    // ---- sample decode + distribution bounds (dsv4-census classes; dspark rows added) ----
    // (bounds: weight [1e-3,1] / norm [1e-2,10] / aux [1e-4,100] — measured lane-1 evidence)
    let check = |name: &str, lo: f64, hi: f64, v: &[f32], failures: &mut Vec<String>| {
        let s = stats(v);
        println!(
            "  {name}: n={} min={:.6} max={:.6} absmean={:.6e} zerofrac={:.4} nan={} inf={}",
            v.len(),
            s.min,
            s.max,
            s.absmean,
            s.zerofrac,
            s.nan,
            s.inf
        );
        if s.nan > 0 || s.inf > 0 {
            failures.push(format!("{name}: {} NaN / {} Inf decoded", s.nan, s.inf));
        }
        if s.zerofrac >= 1.0 {
            failures.push(format!("{name}: all zeros"));
        }
        if !(s.absmean >= lo && s.absmean <= hi) {
            failures.push(format!(
                "{name}: absmean {:.6e} outside bounds [{lo:e}, {hi:e}]",
                s.absmean
            ));
        }
    };
    const W: (f64, f64) = (1e-3, 1.0);
    const N: (f64, f64) = (1e-2, 10.0);
    const A: (f64, f64) = (1e-4, 100.0);

    println!("\n== quantized samples (decoded to F32) ==");
    let quant_samples: [(&str, (f64, f64)); 12] = [
        ("layers.20.ffn.experts.7.w1", W),
        ("layers.20.ffn.experts.7.w2", W),
        ("layers.20.ffn.experts.7.w3", W),
        ("layers.0.ffn.experts.0.w1", W),
        ("layers.42.ffn.experts.255.w3", W),
        ("layers.20.attn.wq_a", W),
        ("layers.20.attn.wo_b", W),
        ("layers.2.attn.indexer.wq_b", W),
        ("layers.20.ffn.shared_experts.w2", W),
        ("mtp.0.main_proj", W),
        ("mtp.0.ffn.experts.7.w1", W),
        ("mtp.2.ffn.experts.7.w2", W),
    ];
    for (stem, b) in quant_samples {
        match decode_quant_stem(&m, stem) {
            Ok((v, _, _)) => check(stem, b.0, b.1, &v, &mut failures),
            Err(e) => failures.push(e),
        }
    }

    println!("\n== BF16 / F32 samples ==");
    let plain_samples: [(&str, (f64, f64)); 16] = [
        ("embed.weight", W),
        ("head.weight", W),
        ("norm.weight", N),
        ("layers.20.attn_norm.weight", N),
        ("layers.2.attn.compressor.norm.weight", N),
        ("layers.20.ffn.gate.weight", W),
        ("layers.2.attn.compressor.ape", A),
        ("layers.3.attn.compressor.ape", A),
        ("layers.20.hc_attn_fn", A),
        ("hc_head_fn", A),
        ("layers.0.attn.attn_sink", A),
        ("layers.3.ffn.gate.bias", A),
        ("mtp.0.main_norm.weight", N),
        ("mtp.2.markov_head.markov_w1.weight", A),
        ("mtp.2.confidence_head.proj.weight", A),
        ("mtp.2.hc_head_fn", A),
    ];
    for (name, b) in plain_samples {
        match m.raw(name) {
            Some((info, raw)) => {
                let v = match info.dtype.as_str() {
                    "BF16" => bf16_to_f32_vec(raw),
                    "F32" => f32_vec(raw),
                    other => {
                        failures.push(format!("{name}: unexpected dtype {other}"));
                        continue;
                    }
                };
                check(name, b.0, b.1, &v, &mut failures);
            }
            None => failures.push(format!("{name}: missing (sample set)")),
        }
    }

    // ---- hash-router table range ----
    let ne = mc.moe.as_ref().unwrap().expert_count as i64;
    match m.raw("layers.0.ffn.gate.tid2eid") {
        Some((_, raw)) => {
            let bad = raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .filter(|&x| x < 0 || x >= ne)
                .count();
            println!("\ntid2eid layer0: {bad} entries out of [0, {ne})");
            if bad > 0 {
                failures.push(format!("tid2eid: {bad} entries outside [0, {ne})"));
            }
        }
        None => failures.push("layers.0.ffn.gate.tid2eid: missing".into()),
    }

    match recipe {
        // minted: NVFP4 scale structure on sampled trunk experts (pow2 + pair sharing)
        ExpertRecipe::Nvfp4NoInputScale => {
            println!();
            for stem in [
                "layers.20.ffn.experts.7.w1",
                "layers.20.ffn.experts.7.w2",
                "layers.20.ffn.experts.7.w3",
                "layers.0.ffn.experts.0.w1",
                "layers.42.ffn.experts.255.w3",
            ] {
                let Some((si, sb)) = m.raw(&format!("{stem}.weight_scale")) else {
                    failures.push(format!("{stem}.weight_scale: missing"));
                    continue;
                };
                let Some((_, s2b)) = m.raw(&format!("{stem}.weight_scale_2")) else {
                    failures.push(format!("{stem}.weight_scale_2: missing"));
                    continue;
                };
                let s2 = f32::from_le_bytes(s2b.try_into().unwrap());
                let gpr = *si.shape.last().unwrap() as usize;
                let (mut non_pow2, mut pair_mismatch) = (0usize, 0usize);
                for (i, &b) in sb.iter().enumerate() {
                    let eff = fp8_e4m3_to_f32(b) * s2;
                    if eff <= 0.0 || eff.log2().fract() != 0.0 {
                        non_pow2 += 1;
                    }
                    if i % gpr % 2 == 1 && sb[i - 1] != b {
                        pair_mismatch += 1;
                    }
                }
                println!(
                    "  {stem}: scale_2={s2:e}, {} scales, non-pow2 {non_pow2}, pair-mismatch \
                     {pair_mismatch}",
                    sb.len()
                );
                if non_pow2 > 0 || pair_mismatch > 0 {
                    failures.push(format!(
                        "{stem}: NVFP4 scale structure violated (non-pow2 {non_pow2}, \
                         pair-mismatch {pair_mismatch})"
                    ));
                }
            }
            // input_scale must be ABSENT everywhere (the §2 omit decision)
            let strays: Vec<&String> = m
                .names()
                .filter(|n| n.ends_with(".input_scale"))
                .take(5)
                .collect();
            if !strays.is_empty() {
                failures.push(format!("input_scale present (must be omitted): {strays:?}"));
            }
        }
        // source: full E8M0 NaN scan over EVERY expert scale tensor (trunk + dspark)
        ExpertRecipe::Mxfp4 => {
            let mut scanned = 0usize;
            let mut nan_hits = 0usize;
            for n in m.names() {
                if n.ends_with(".scale") && n.contains(".ffn.experts.") {
                    let (_, raw) = m.raw(n).unwrap();
                    scanned += 1;
                    let c = raw.iter().filter(|&&b| b == 0xFF).count();
                    if c > 0 {
                        nan_hits += c;
                        failures.push(format!("{n}: {c} E8M0 NaN (0xFF) scale bytes"));
                    }
                }
            }
            println!("\nE8M0 scan: {scanned} expert scale tensors, {nan_hits} NaN codes");
        }
    }

    println!("\nelapsed: {:.1}s", t0.elapsed().as_secs_f64());
    if failures.is_empty() {
        println!(
            "0731 CENSUS GATE ({}): PASS ({} tensors verified, 0 failures)",
            match recipe {
                ExpertRecipe::Mxfp4 => "source",
                ExpertRecipe::Nvfp4NoInputScale => "minted",
            },
            census.len()
        );
        Ok(())
    } else {
        println!("0731 CENSUS GATE: FAIL ({} failures)", failures.len());
        for f in &failures {
            println!("  FAIL: {f}");
        }
        Err("census gate failed".into())
    }
}

// ============================ main ============================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("INVOCATION: {}", args.join(" "));
    let threads = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let r = match args.get(1).map(String::as_str) {
        Some("cast") => cmd_cast(Path::new(&args[2]), &args[3], Path::new(&args[4]), threads),
        Some("census") => {
            let recipe = match args
                .iter()
                .position(|a| a == "--recipe")
                .and_then(|i| args.get(i + 1))
                .map(String::as_str)
            {
                Some("source") => ExpertRecipe::Mxfp4,
                Some("minted") => ExpertRecipe::Nvfp4NoInputScale,
                other => {
                    eprintln!("census needs --recipe source|minted (got {other:?})");
                    std::process::exit(2);
                }
            };
            cmd_census(Path::new(&args[2]), recipe)
        }
        Some("index") => cmd_index(Path::new(&args[2])),
        _ => {
            eprintln!(
                "usage: dsv4_mint cast <src-dir> <shard-file> <out-dir> [--threads N]\n\
                 \x20      dsv4_mint census <model-dir> --recipe source|minted\n\
                 \x20      dsv4_mint index <out-dir>"
            );
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

// ============================ tests ============================

#[cfg(test)]
mod tests {
    use super::*;

    /// 0731-shaped config.json: the REAL artifact's values incl. the vestigial
    /// num_nextn_predict_layers: 1 that load_0731_config must patch to 3.
    fn write_0731_config(dir: &Path) {
        let mut ratios = vec![0u32, 0];
        for il in 2..43u32 {
            ratios.push(if il % 2 == 0 { 4 } else { 128 });
        }
        ratios.extend([0, 0, 0]); // 3 dspark blocks
        assert_eq!(ratios.len(), 46);
        let rj = ratios
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["DeepseekV4ForCausalLM"],"model_type":"deepseek_v4",
                "num_hidden_layers":43,"hidden_size":4096,"num_attention_heads":64,
                "num_key_value_heads":1,"head_dim":512,"vocab_size":129280,
                "max_position_embeddings":1048576,"rms_norm_eps":1e-6,"rope_theta":10000,
                "n_routed_experts":256,"n_shared_experts":1,"num_experts_per_tok":6,
                "moe_intermediate_size":2048,"norm_topk_prob":true,"num_hash_layers":3,
                "num_nextn_predict_layers": 1,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc",
                "routed_scaling_factor":1.5,"hc_eps":1e-6,"hc_mult":4,"hc_sinkhorn_iters":20,
                "q_lora_rank":1024,"qk_rope_head_dim":64,"o_lora_rank":1024,"o_groups":8,
                "index_n_heads":64,"index_head_dim":128,"index_topk":512,
                "compress_ratios":[{rj}],"compress_rope_theta":160000,
                "sliding_window":128,"swiglu_limit":10.0,
                "dspark_block_size":5,"dspark_markov_rank":256,"dspark_noise_token_id":128799,
                "dspark_target_layer_ids":[40,41,42],
                "rope_scaling":{{"type":"yarn","factor":16,"beta_fast":32,"beta_slow":1,
                "original_max_position_embeddings":65536}}}}"#
            ),
        )
        .unwrap();
    }

    fn cfg(tag: &str) -> (ModelConfig, Dspark) {
        let dir = std::env::temp_dir().join(format!("dsv4_mint_test_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_0731_config(&dir);
        let out = load_0731_config(&dir);
        std::fs::remove_dir_all(&dir).ok();
        out
    }

    #[test]
    fn config_derives_three_dspark_blocks() {
        let (mc, dspark) = cfg("derive");
        assert_eq!(mc.n_layer, 46);
        assert_eq!(mc.nextn_predict_layers, 3, "derived, NOT the vestigial 1");
        assert_eq!(dspark.n_drafter, 3);
        assert_eq!(dspark.tap_n, 3);
        assert_eq!(dspark.markov_rank, 256);
        let d = mc.dsv4.as_ref().unwrap();
        // dspark blocks: window-only (no compressor/indexer), score-routed
        for il in 43..46 {
            assert!(!d.has_compressor(il) && !d.has_indexer(il));
            assert!(!d.is_hash_layer(il));
        }
    }

    #[test]
    fn census_totals_match_artifact() {
        let (mc, dspark) = cfg("totals");
        let src = expected_census_0731(&mc, &dspark, ExpertRecipe::Mxfp4);
        // measured truth: 0731 index counts 72,317 tensors (prep §1)
        assert_eq!(src.len(), 72_317, "source census != 0731 tensor count");
        // per-dspark-block counts from the prep §1.2 header census
        for (k, want) in [(0usize, 1568usize), (1, 1565), (2, 1572)] {
            let n = src
                .keys()
                .filter(|n| n.starts_with(&format!("mtp.{k}.")))
                .count();
            assert_eq!(n, want, "mtp.{k} tensor count");
        }
        // minted: each of 33,024 trunk projections gains one tensor (scale -> weight_scale
        // + weight_scale_2, no input_scale)
        let minted = expected_census_0731(&mc, &dspark, ExpertRecipe::Nvfp4NoInputScale);
        assert_eq!(minted.len(), 105_341, "minted census total");
        // spot rows from the prep §1.2 tables
        let s = &src["layers.20.ffn.experts.7.w1.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("I8", &[2048u64, 2048][..]));
        let s = &src["layers.20.ffn.experts.7.w1.scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E8M0", &[2048u64, 128][..])
        );
        let s = &minted["layers.20.ffn.experts.7.w1.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("U8", &[2048u64, 2048][..]));
        let s = &minted["layers.20.ffn.experts.7.w1.weight_scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E4M3", &[2048u64, 256][..])
        );
        let s = &src["mtp.0.main_proj.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E4M3", &[4096u64, 12288][..])
        );
        let s = &src["mtp.0.main_proj.scale"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("F8_E8M0", &[32u64, 96][..]));
        let s = &src["mtp.2.markov_head.markov_w1.weight"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("BF16", &[129280u64, 256][..])
        );
        let s = &src["mtp.2.confidence_head.proj.weight"];
        assert_eq!((s.dtype, s.shape.as_slice()), ("BF16", &[1u64, 4352][..]));
        let s = &src["mtp.2.ffn.experts.0.w2.scale"];
        assert_eq!(
            (s.dtype, s.shape.as_slice()),
            ("F8_E8M0", &[4096u64, 64][..])
        );
        // NextN heads are GONE — presence would fail as extra
        for gone in [
            "mtp.0.e_proj.weight",
            "mtp.0.h_proj.weight",
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
        ] {
            assert!(!src.contains_key(gone), "{gone} must not be expected");
        }
        // dspark blocks keep MXFP4 in the MINTED census too
        let s = &minted["mtp.1.ffn.experts.3.w1.weight"];
        assert_eq!(s.dtype, "I8");
        assert!(!minted.contains_key("mtp.1.ffn.experts.3.w1.weight_scale_2"));
        // hc-head extras only on the LAST dspark block
        assert!(src.contains_key("mtp.2.hc_head_fn") && !src.contains_key("mtp.1.hc_head_fn"));
        assert!(src.contains_key("mtp.0.main_norm.weight"));
        assert!(!src.contains_key("mtp.1.main_proj.weight"));
    }

    #[test]
    fn e4m3_pow2_codes_roundtrip_and_refuse() {
        for q in -9..=8 {
            let c = e4m3_pow2_code(q).unwrap();
            assert_eq!(fp8_e4m3_to_f32(c), (q as f32).exp2(), "q={q}");
        }
        assert!(e4m3_pow2_code(-10).is_err());
        assert!(e4m3_pow2_code(9).is_err());
    }

    #[test]
    fn scale_cast_is_lossless_and_refuses_nan() {
        // measured preview E8M0 range 119..122 — and a wider synthetic spread
        let scale: Vec<u8> = vec![119, 120, 121, 122, 105, 122, 119, 110];
        let (s16, s2) = cast_expert_scale(&scale, "t").unwrap();
        assert_eq!(s16.len(), 16);
        assert_eq!(s2, 2f32.powi(122 - 127 - 8));
        for (j, &b) in scale.iter().enumerate() {
            assert_eq!(s16[2 * j], s16[2 * j + 1]);
            assert_eq!(
                fp8_e4m3_to_f32(s16[2 * j]) * s2,
                e8m0_to_f32(b),
                "group {j}"
            );
        }
        assert!(
            cast_expert_scale(&[119, 0xFF], "t").is_err(),
            "NaN must refuse"
        );
        // >17-octave spread cannot be represented — refuse, never round
        assert!(cast_expert_scale(&[122, 104], "t").is_err());
        assert!(
            cast_expert_scale(&[122, 105], "t").is_ok(),
            "exactly 17 octaves fits"
        );
    }

    #[test]
    fn synthetic_cast_decodes_bit_identical() {
        // one full synthetic projection: 4 rows x 64 cols, pseudo-random codes + scales
        let (rows, cols) = (4usize, 64usize);
        let mut weight = vec![0u8; rows * cols / 2];
        for (i, b) in weight.iter_mut().enumerate() {
            *b = ((i * 89 + 17) & 0xFF) as u8;
        }
        let scale: Vec<u8> = (0..rows * cols / 32)
            .map(|i| 115 + ((i * 7) % 8) as u8)
            .collect();
        let (s16, s2) = cast_expert_scale(&scale, "syn").unwrap();
        let a = dequant_mxfp4_expert(&weight, &scale, rows, cols);
        let b = dequant_nvfp4_expert(&weight, &s16, s2, rows, cols);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "elem {i}");
        }
        // e2m1 negative-zero coverage: code 8 is -0.0 and must survive bit-exactly
        assert!(a.iter().any(|x| x.to_bits() == (-0.0f32).to_bits()));
    }

    #[test]
    fn trunk_stem_filter() {
        assert!(is_trunk_expert_stem("layers.0.ffn.experts.0.w1"));
        assert!(is_trunk_expert_stem("layers.42.ffn.experts.255.w3"));
        assert!(
            !is_trunk_expert_stem("mtp.0.ffn.experts.0.w1"),
            "dspark excluded"
        );
        assert!(!is_trunk_expert_stem("layers.0.ffn.shared_experts.w1"));
        assert!(!is_trunk_expert_stem("layers.0.attn.wq_a"));
    }

    #[test]
    fn safetensors_writer_roundtrips() {
        let dir = std::env::temp_dir().join(format!("dsv4_mint_wr_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let e = vec![
            OutEntry {
                name: "a.weight".into(),
                dtype: "U8".into(),
                shape: vec![2, 3],
                bytes: Bytes::Owned(vec![1, 2, 3, 4, 5, 6]),
            },
            OutEntry {
                name: "a.weight_scale_2".into(),
                dtype: "F32".into(),
                shape: vec![],
                bytes: Bytes::Owned(0.25f32.to_le_bytes().to_vec()),
            },
        ];
        let total = write_safetensors(&p, &e).unwrap();
        let sh = StShard::open(&p).unwrap();
        assert_eq!(sh.len(), 2);
        let (i, b) = sh.raw("a.weight").unwrap();
        assert_eq!((i.dtype.as_str(), b), ("U8", &[1u8, 2, 3, 4, 5, 6][..]));
        let (i, b) = sh.raw("a.weight_scale_2").unwrap();
        assert_eq!(i.dtype, "F32");
        assert_eq!(f32::from_le_bytes(b.try_into().unwrap()), 0.25);
        assert!(total > 10);
        // storage order preserved (a.weight first)
        assert_eq!(
            shard_names_in_order(&sh),
            vec!["a.weight", "a.weight_scale_2"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
