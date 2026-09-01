//! dsv4-census: the DeepSeek-V4-Flash loader-lane census gate (CPU only, no GPU).
//!
//! Proves the engine reads every byte of the NVFP4 artifact correctly, and nothing more:
//!   1. config.json parses into the typed DeepSeekV4 arch config (every forward-pass field).
//!   2. hf_quant_config.json's own declaration matches the census's recipe split
//!      (NVFP4/group-16 on every trunk layer's experts; mtp.* excluded -> MXFP4).
//!   3. Every one of the expected tensors (derived from config math — 135,235 on Flash) exists
//!      across the shards with the exact dtype, shape and byte length; zero extras.
//!   4. A deterministic sample set decodes to F32 with sane distributions.
//!
//! FAIL conditions (loud, named): any missing/extra/mismatched tensor; any NaN/Inf in a decoded
//! sample; an all-zero sample; absmean outside the per-class bounds; NVFP4 scale-structure
//! violation. Bounds are set from measured evidence on the real artifact (2026-08-18 probe,
//! research/dsv4-flash-loader-20260818/RECEIPTS.md):
//!   weight class (embed/head/gate.weight/compressor/quantized linears/experts):
//!       absmean in [1e-3, 1.0]     (measured 0.0186 .. 0.151)
//!   norm class (rmsnorm gains): absmean in [1e-2, 10.0]   (measured 0.057 .. 0.83)
//!   aux F32 class (hc_*, ape, attn_sink, gate.bias):
//!       absmean in [1e-4, 100.0]   (measured 0.0015 hc_head_fn .. 27.4 gate.bias)
//!
//! NVFP4 scale-structure check (cast provenance): this artifact is a LOSSLESS MXFP4->NVFP4
//! cast (its cast_mxfp4_to_nvfp4.log), so every effective scale (e4m3 * weight_scale_2) is an
//! exact power of two and adjacent 16-group scale pairs are identical. Any indexing bug in the
//! scale path shatters both, so the gate hard-checks them on its sampled experts. A future
//! artifact quantized natively to NVFP4 would legitimately relax this — revisit then.
//!
//! Usage:
//!   dsv4-census <model-dir>                    run the gate (exit 0 PASS / 1 FAIL)
//!   dsv4-census <model-dir> --dump <stem> <out.bin>
//!       decode one quantized tensor to raw little-endian F32 (row-major) for the
//!       Python-oracle cross-check (deliverable E) — compare sha256 of the two files.

use memra_gguf::config::ModelConfig;
use memra_gguf::dsv4::{
    dequant_fp8_blk128, dequant_mxfp4_expert, dequant_nvfp4_expert, expected_census, verify_census,
    verify_quant_declaration,
};
use memra_gguf::nvfp4_repack::fp8_e4m3_to_f32;
use memra_gguf::safetensors::StModel;
use std::io::Write;
use std::path::Path;

// ---- sample-stat bounds (measured evidence in the header) ----
const WEIGHT_ABSMEAN: (f32, f32) = (1e-3, 1.0);
const NORM_ABSMEAN: (f32, f32) = (1e-2, 10.0);
const AUX_ABSMEAN: (f32, f32) = (1e-4, 100.0);

#[derive(Clone, Copy)]
enum Class {
    Weight,
    Norm,
    Aux,
}

impl Class {
    fn bounds(self) -> (f32, f32) {
        match self {
            Class::Weight => WEIGHT_ABSMEAN,
            Class::Norm => NORM_ABSMEAN,
            Class::Aux => AUX_ABSMEAN,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Class::Weight => "weight",
            Class::Norm => "norm",
            Class::Aux => "aux",
        }
    }
}

struct Stats {
    min: f32,
    max: f32,
    mean: f64,
    absmean: f64,
    zerofrac: f64,
    nan: usize,
    inf: usize,
}

fn stats(v: &[f32]) -> Stats {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut sum, mut asum) = (0f64, 0f64);
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
        sum += x as f64;
        asum += x.abs() as f64;
        if x == 0.0 {
            zeros += 1;
        }
    }
    let n = v.len().max(1) as f64;
    Stats {
        min,
        max,
        mean: sum / n,
        absmean: asum / n,
        zerofrac: zeros as f64 / n,
        nan,
        inf,
    }
}

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

/// Decode a quantized stem by its sibling set (the same inference the Python oracle does):
/// `.weight_scale_2` present -> modelopt NVFP4; `.scale` + I8 weight -> MXFP4;
/// `.scale` + F8_E4M3 weight -> FP8 128x128. Returns (values, logical rows, logical cols).
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or_else(|| {
        eprintln!("usage: dsv4-census <model-dir> [--dump <stem> <out.bin>]");
        std::process::exit(2);
    }));

    let t0 = std::time::Instant::now();
    let mc = ModelConfig::from_config_json(&dir.join("config.json")).expect("parse config.json");
    assert!(
        mc.arch.is_dsv4(),
        "dsv4-census: model_type is not deepseek_v4 (arch {:?})",
        mc.arch
    );
    let d = mc.dsv4.as_ref().unwrap();
    let n_trunk = mc.n_layer - mc.nextn_predict_layers;
    println!(
        "config: {} trunk + {} MTP layers, hidden {}, {} experts ({} active, {} hash-routed \
         layers), scoring {}, hc_mult {} (sinkhorn {} iters), index_topk {}",
        n_trunk,
        mc.nextn_predict_layers,
        mc.n_embd,
        mc.moe.as_ref().unwrap().expert_count,
        mc.moe.as_ref().unwrap().expert_used_count,
        d.num_hash_layers,
        d.scoring_func,
        d.hc_mult,
        d.hc_sinkhorn_iters,
        d.index_topk,
    );
    let n_comp = (0..mc.n_layer).filter(|&il| d.has_compressor(il)).count();
    let n_idx = (0..mc.n_layer).filter(|&il| d.has_indexer(il)).count();
    println!(
        "derived splits: {n_comp} compressor layers, {n_idx} indexer layers (from compress_ratios)"
    );

    let mut failures: Vec<String> = Vec::new();

    // ---- artifact's own quant declaration vs the census recipe split ----
    let decl = verify_quant_declaration(dir, &mc);
    if decl.is_empty() {
        println!(
            "hf_quant_config.json: NVFP4/group-16 declared on all {n_trunk} trunk expert banks; \
             mtp.* excluded (MXFP4) — matches census"
        );
    } else {
        for e in &decl {
            failures.push(format!("quant declaration: {e}"));
        }
    }

    // ---- full header census across every shard ----
    let m = StModel::open(dir).expect("open safetensors model");
    println!("opened {} tensors across shards", m.n_tensors());
    let census = expected_census(&mc);
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

    // ---- dump mode (oracle cross-check) runs after the header census, before sampling ----
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        let stem = &args[i + 1];
        let out = &args[i + 2];
        let (v, rows, cols) = decode_quant_stem(&m, stem).expect("dump decode");
        let mut f = std::fs::File::create(out).expect("create dump file");
        let mut buf = Vec::with_capacity(v.len() * 4);
        for x in &v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf).expect("write dump");
        println!(
            "dumped {stem} [{rows}, {cols}] -> {out} ({} bytes LE f32)",
            buf.len()
        );
        return;
    }

    // ---- deterministic sample decode + distribution gate ----
    // (name, class); quantized stems decode through their sibling recipes, BF16/F32 verbatim.
    let quant_samples: [(&str, Class); 12] = [
        // one FULL expert (all three projections) mid-net + two edge experts
        ("layers.20.ffn.experts.7.w1", Class::Weight),
        ("layers.20.ffn.experts.7.w2", Class::Weight),
        ("layers.20.ffn.experts.7.w3", Class::Weight),
        ("layers.0.ffn.experts.0.w1", Class::Weight),
        ("layers.42.ffn.experts.255.w3", Class::Weight),
        // attention linears, both matmul directions + the indexer's FP8 pair
        ("layers.20.attn.wq_a", Class::Weight),
        ("layers.20.attn.wo_b", Class::Weight),
        ("layers.2.attn.indexer.wq_b", Class::Weight),
        ("layers.20.ffn.shared_experts.w2", Class::Weight),
        // MTP: fusion projection + MXFP4 experts
        ("mtp.0.e_proj", Class::Weight),
        ("mtp.0.ffn.experts.7.w1", Class::Weight),
        ("mtp.0.ffn.experts.7.w2", Class::Weight),
    ];
    let plain_samples: [(&str, Class); 18] = [
        ("embed.weight", Class::Weight),
        ("head.weight", Class::Weight),
        ("norm.weight", Class::Norm),
        ("layers.20.attn_norm.weight", Class::Norm),
        ("layers.20.attn.q_norm.weight", Class::Norm),
        ("layers.2.attn.compressor.norm.weight", Class::Norm),
        ("mtp.0.enorm.weight", Class::Norm),
        ("layers.20.ffn.gate.weight", Class::Weight),
        // both compressor shape classes: fine (ratio 4) and coarse (ratio 128)
        ("layers.2.attn.compressor.wkv.weight", Class::Weight),
        ("layers.2.attn.compressor.ape", Class::Aux),
        ("layers.3.attn.compressor.wkv.weight", Class::Weight),
        ("layers.3.attn.compressor.ape", Class::Aux),
        ("layers.20.hc_attn_fn", Class::Aux),
        ("layers.20.hc_attn_base", Class::Aux),
        ("layers.20.hc_ffn_scale", Class::Aux),
        ("hc_head_fn", Class::Aux),
        ("layers.0.attn.attn_sink", Class::Aux),
        ("layers.3.ffn.gate.bias", Class::Aux),
    ];

    let check = |name: &str, class: Class, v: &[f32], failures: &mut Vec<String>| {
        let s = stats(v);
        println!(
            "  {name} [{}]: n={} min={:.6} max={:.6} mean={:.6e} absmean={:.6e} zerofrac={:.4} \
             nan={} inf={}",
            class.name(),
            v.len(),
            s.min,
            s.max,
            s.mean,
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
        let (lo, hi) = class.bounds();
        if !(s.absmean >= lo as f64 && s.absmean <= hi as f64) {
            failures.push(format!(
                "{name}: absmean {:.6e} outside {} bounds [{lo:e}, {hi:e}]",
                s.absmean,
                class.name()
            ));
        }
    };

    println!("\n== quantized samples (decoded to F32) ==");
    for (stem, class) in quant_samples {
        match decode_quant_stem(&m, stem) {
            Ok((v, _, _)) => check(stem, class, &v, &mut failures),
            Err(e) => failures.push(e),
        }
    }

    println!("\n== BF16 / F32 samples ==");
    for (name, class) in plain_samples {
        match m.raw(name) {
            Some((info, raw)) => {
                let v = match info.dtype.as_str() {
                    "BF16" => bf16_to_f32_vec(raw),
                    "F32" => f32_vec(raw),
                    other => {
                        failures.push(format!("{name}: unexpected dtype {other} in sample set"));
                        continue;
                    }
                };
                check(name, class, &v, &mut failures);
            }
            None => failures.push(format!("{name}: missing (sample set)")),
        }
    }

    // ---- hash-router table: every entry must be a valid expert id ----
    let ne = mc.moe.as_ref().unwrap().expert_count as i64;
    match m.raw("layers.0.ffn.gate.tid2eid") {
        Some((_, raw)) => {
            let mut bad = 0usize;
            let (mut lo, mut hi) = (i64::MAX, i64::MIN);
            for c in raw.chunks_exact(8) {
                let x = i64::from_le_bytes(c.try_into().unwrap());
                lo = lo.min(x);
                hi = hi.max(x);
                if x < 0 || x >= ne {
                    bad += 1;
                }
            }
            println!("\ntid2eid layer0: range [{lo}, {hi}], {bad} out of [0, {ne})");
            if bad > 0 {
                failures.push(format!("tid2eid: {bad} entries outside [0, {ne})"));
            }
        }
        None => failures.push("layers.0.ffn.gate.tid2eid: missing (sample set)".into()),
    }

    // ---- NVFP4 scale structure (cast provenance; see header) on the sampled trunk experts ----
    println!();
    for stem in [
        "layers.20.ffn.experts.7.w1",
        "layers.20.ffn.experts.7.w2",
        "layers.20.ffn.experts.7.w3",
        "layers.0.ffn.experts.0.w1",
        "layers.42.ffn.experts.255.w3",
    ] {
        let Some((si, sb)) = m.raw(&format!("{stem}.weight_scale")) else {
            failures.push(format!("{stem}.weight_scale: missing (structure check)"));
            continue;
        };
        let Some((_, s2b)) = m.raw(&format!("{stem}.weight_scale_2")) else {
            failures.push(format!("{stem}.weight_scale_2: missing (structure check)"));
            continue;
        };
        let s2 = f32::from_le_bytes(s2b.try_into().unwrap());
        let groups_per_row = *si.shape.last().unwrap() as usize;
        let mut non_pow2 = 0usize;
        let mut pair_mismatch = 0usize;
        let mut zero_scales = 0usize;
        for (i, &b) in sb.iter().enumerate() {
            let eff = fp8_e4m3_to_f32(b) * s2;
            if eff == 0.0 {
                zero_scales += 1;
            } else if eff.log2().fract() != 0.0 {
                non_pow2 += 1;
            }
            // pairs (2k, 2k+1) within a row share their 32-group ancestor's scale
            if i % groups_per_row % 2 == 1 && sb[i - 1] != b {
                pair_mismatch += 1;
            }
        }
        println!(
            "  {stem}: scale_2={s2:e}, {} scales, non-pow2 {}, pair-mismatch {}, zero {}",
            sb.len(),
            non_pow2,
            pair_mismatch,
            zero_scales
        );
        if non_pow2 > 0 || pair_mismatch > 0 {
            failures.push(format!(
                "{stem}: NVFP4 scale structure violated (non-pow2 {non_pow2}, pair-mismatch \
                 {pair_mismatch}) — scale indexing suspect"
            ));
        }
    }

    println!("\nelapsed: {:.1}s", t0.elapsed().as_secs_f64());
    if failures.is_empty() {
        println!(
            "CENSUS GATE: PASS ({} tensors verified, 0 failures)",
            census.len()
        );
    } else {
        println!("CENSUS GATE: FAIL ({} failures)", failures.len());
        for f in &failures {
            println!("  FAIL: {f}");
        }
        std::process::exit(1);
    }
}
