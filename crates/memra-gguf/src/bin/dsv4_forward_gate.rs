//! dsv4-forward: the DeepSeek-V4-Flash CPU paired-forward gate (lane 3, CPU only).
//!
//! Replays the lane-2 fixture inputs through the Rust f32 forward
//! (memra_gguf::dsv4_forward) on the REAL artifact's weights and compares every banked
//! array. ONE fixture variant per invocation — the variant (ref FP8-round vs the NVFP4
//! artifact's clamp-only kernel.py:88) is read from the fixture JSON and selects the
//! matching act-quant behavior; variants are never mixed in a gate.
//!
//! Usage:
//!   dsv4-forward <model-dir> <fixtures.json>       exit 0 PASS / 1 FAIL
//!
//! The npz next to the JSON is the compared truth; each array's payload sha256 is
//! verified against the JSON before use (a mismatch means a corrupted fixture, hard
//! FAIL before any forward math runs).
//!
//! ── Threshold doctrine (why these numbers) ─────────────────────────────────────────
//! The fixtures are torch-2.13-CPU f32 (blocked/vectorized reductions, FMA); this port
//! accumulates dots in f64 (see dsv4_forward.rs header), so the expected gap per
//! length-k dot is torch's own rounding ≈ sqrt(k)·2⁻²⁴ of the dot's absolute mass —
//! ~4e-6 relative at k = 4096 — plus a-few-ULP transcendental-library skew. Per layer
//! a value crosses ~6 such GEMMs plus renormalizations (which re-anchor scale), so
//! depth-L outputs carry roughly L·(few·1e-5) relative noise. Budgets, ×~10 margin:
//!   embed_out               BIT-EXACT (table lookup + bf16→f32 widening, no arithmetic)
//!   depth-1 arrays          1e-4 × array absmax
//!   depth-3/4 arrays        2.5e-4 × array absmax
//!   logits (depth 43 + MTP) 0.5 absolute — must sit far below the 5.1 max-abs contract
//!                           fork between the two kernel variants (mixing variants is
//!                           the failure this bound must catch) and far above smooth
//!                           rounding noise; top-k id agreement is the semantic gate:
//!                           top-1 EXACT, top-5 SET-EXACT, ≥18/20 top-20 overlap.
//! Quantized-kv arrays (compressor_kv nope dims per-64 FP8, indexer_kv per-32 FP4) are
//! products of grid codes and pow2 scales — exact when both sides round the same way.
//! A ~1e-6-relative upstream difference can flip one code at a rounding boundary; a
//! flip changes that ONE element by ≤ one grid step (≤ 2⁻³ rel for e4m3, ≤ 1/2 rel for
//! e2m1). The gate allows ≤ 2 such named flips per quantized array, each verified to be
//! within one grid step; anything more or bigger is a real divergence and FAILs.
//! index_score −inf masks (causality) must match EXACTLY; NaN anywhere is a hard FAIL;
//! shape mismatch is a hard FAIL.

use memra_gguf::dsv4_forward::{
    BlockCapture, BlockW, Dsv4Model, FixtureSpec, hc_expand, mtp_logits_last, read_npz,
    trunk_logits_last,
};
use std::collections::BTreeSet;
use std::path::Path;

const MAX_SEQ: usize = 4096; // rope table rows (values are per-position, not length-dependent)

struct ArrayCheck {
    name: String,
    shape: Vec<usize>,
    max_abs: f64,
    max_rel: f64,
    threshold: f64,
    n_over: usize,
    verdict: &'static str,
    note: String,
}

enum Policy {
    BitExact,
    MaxAbs {
        rel_of_absmax: f64,
        flip_budget: usize,
        flip_rel: f32,
    },
    Logits {
        abs: f64,
        top_ids: Vec<u32>,
    },
}

// ── REF-variant threshold correction (0731 oracle rerun, derivation banked BEFORE the
// rerun per the lane-4 gate-formula protocol) ─────────────────────────────────────────
// The ref (RefFp8Round) contract carries a DISCONTINUOUS noise class the clamp-only twin
// does not: ~1e-6-relative torch-vs-Rust reduction skew occasionally flips one e4m3 code
// in the window/compressor KV QAT, and 43 layers accumulate those flips (lane-3 receipt).
// Measured draws of the class on final logits: 0.394 (preview weights), 0.979 (0731
// weights) — the fixed 0.5 bound was a lucky fit to the first draw, not a law. The
// failure this bound exists to catch is CONTRACT MIXING, whose measured magnitude is the
// ref-vs-clamp fork of the same generator run (5.1 preview, 3.361 on 0731 — banked in
// the fixture JSON as contract_fork_final_logits_maxabs). Corrected rule:
//   thr(ref) = fork/3   (mixing fails by ≥3x; same-contract draws pass with margin),
//   thr(clamp-only) = 0.5 unchanged (its measured class is 4.6e-5 — four orders below),
//   id gates unchanged either way (top-1 exact, top-5 set-exact, ≥18/20 top-20): the
//   semantic acceptance is id-level, as lane 3 already ruled.
// index_score under ref: the array sits DOWNSTREAM of two fp4 quantizers (indexer q and
// kv); one flipped e2m1 lane in a 128-lane dot moves one score element by ~1e-3 relative
// (measured 1.04e-3; a one-lane bound is grid-step/2 x lane-mass ~ 4e-3). Grant the same
// flip allowance quantized-kv arrays already have: budget 2, per-element ≤1e-2 relative
// — 10x the observed flip, still 5x under the smallest fork-class element (4.7e-2 rel),
// and mixing floods dozens of elements over the base threshold so the budget catches it.
// Selection semantics stay separately guarded by the exact −inf causality-mask compare.
fn policy_for(name: &str, spec: &FixtureSpec) -> Policy {
    let is_ref = spec.variant == memra_gguf::dsv4_forward::ActQuantVariant::RefFp8Round;
    match name {
        "embed_out" => Policy::BitExact,
        "final_logits_last" => Policy::Logits {
            abs: match (is_ref, spec.contract_fork_final_logits_maxabs) {
                (true, Some(fork)) => fork / 3.0,
                _ => 0.5,
            },
            top_ids: spec.top20_ids.clone(),
        },
        "mtp_logits_last" => Policy::Logits {
            abs: 0.5,
            top_ids: spec
                .mtp_top20_ids
                .clone()
                .expect("fixture banks mtp_logits_last but json has no mtp_top20"),
        },
        n if n.contains("indexer_kv") => {
            // FP4 grid: one e2m1 step is up to 1/2 the element magnitude
            Policy::MaxAbs {
                rel_of_absmax: 2.5e-4,
                flip_budget: 2,
                flip_rel: 0.5,
            }
        }
        n if n.contains("compressor_kv") => {
            // FP8 grid on the nope dims: one e4m3 step is ≤ 2^-3 of the element
            Policy::MaxAbs {
                rel_of_absmax: 2.5e-4,
                flip_budget: 2,
                flip_rel: 0.15,
            }
        }
        // ref only: one-lane fp4 flip allowance (derivation in the module header above);
        // clamp-only keeps zero budget (its measured class is 4.8e-7 — smooth noise only)
        n if n.contains("index_score") && is_ref => Policy::MaxAbs {
            rel_of_absmax: 2.5e-4,
            flip_budget: 2,
            flip_rel: 0.01,
        },
        n if n.starts_with("layer0") => Policy::MaxAbs {
            rel_of_absmax: 1e-4,
            flip_budget: 0,
            flip_rel: 0.0,
        },
        _ => Policy::MaxAbs {
            rel_of_absmax: 2.5e-4,
            flip_budget: 0,
            flip_rel: 0.0,
        },
    }
}

fn top_ids(v: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.into_iter().take(k).map(|x| x as u32).collect()
}

fn check_array(
    name: &str,
    got: &[f32],
    got_shape: &[usize],
    ref_shape: &[usize],
    ref_vals: &[f32],
    policy: &Policy,
) -> ArrayCheck {
    let mut c = ArrayCheck {
        name: name.to_string(),
        shape: got_shape.to_vec(),
        max_abs: 0.0,
        max_rel: 0.0,
        threshold: 0.0,
        n_over: 0,
        verdict: "PASS",
        note: String::new(),
    };
    if got_shape != ref_shape {
        c.verdict = "FAIL";
        c.note = format!("shape mismatch: got {got_shape:?}, fixture {ref_shape:?}");
        return c;
    }
    if got.iter().any(|x| x.is_nan()) {
        c.verdict = "FAIL";
        c.note = "NaN in computed array".into();
        return c;
    }
    // -inf mask pattern (index_score): positions must agree exactly
    let mut inf_mismatch = 0usize;
    let mut absmax_ref = 0f32;
    for (&g, &r) in got.iter().zip(ref_vals) {
        if g.is_infinite() != r.is_infinite() || (g.is_infinite() && g != r) {
            inf_mismatch += 1;
        }
        if r.is_finite() {
            absmax_ref = absmax_ref.max(r.abs());
        }
    }
    if inf_mismatch > 0 {
        c.verdict = "FAIL";
        c.note = format!("{inf_mismatch} ±inf-mask position mismatches (causality)");
        return c;
    }
    // diffs over finite positions
    let mut diffs: Vec<(usize, f64)> = Vec::new();
    for (i, (&g, &r)) in got.iter().zip(ref_vals).enumerate() {
        if !r.is_finite() {
            continue;
        }
        let d = (g as f64 - r as f64).abs();
        if d > c.max_abs {
            c.max_abs = d;
        }
        let rel = d / (r.abs() as f64).max(1e-6);
        if rel > c.max_rel {
            c.max_rel = rel;
        }
        if d > 0.0 {
            diffs.push((i, d));
        }
    }
    match policy {
        Policy::BitExact => {
            c.threshold = 0.0;
            if c.max_abs > 0.0 {
                c.verdict = "FAIL";
                c.note = format!(
                    "{} non-identical elements (bit-exact required)",
                    diffs.len()
                );
            }
        }
        Policy::MaxAbs {
            rel_of_absmax,
            flip_budget,
            flip_rel,
        } => {
            c.threshold = rel_of_absmax * absmax_ref as f64;
            let over: Vec<&(usize, f64)> = diffs.iter().filter(|(_, d)| *d > c.threshold).collect();
            c.n_over = over.len();
            if !over.is_empty() {
                if over.len() <= *flip_budget
                    && over.iter().all(|(i, d)| {
                        let bound = *flip_rel as f64
                            * (got[*i].abs() as f64).max(ref_vals[*i].abs() as f64);
                        *d <= bound
                    })
                {
                    c.note = format!(
                        "{} quantizer boundary flip(s) within one grid step at {:?}",
                        over.len(),
                        over.iter().map(|(i, _)| *i).collect::<Vec<_>>()
                    );
                } else {
                    c.verdict = "FAIL";
                    c.note = format!(
                        "{} elements over threshold (first at flat index {})",
                        over.len(),
                        over.first().map(|(i, _)| *i).unwrap_or(0)
                    );
                }
            }
        }
        Policy::Logits { abs, top_ids: want } => {
            c.threshold = *abs;
            let got_top = top_ids(got, 20);
            let ref_top = top_ids(ref_vals, 20);
            // fixture JSON top-20 must agree with the npz-derived top-20 (self check)
            if want[..5] != ref_top[..5] {
                c.verdict = "FAIL";
                c.note = format!(
                    "fixture self-inconsistency: json top5 {:?} vs npz {:?}",
                    &want[..5],
                    &ref_top[..5]
                );
                return c;
            }
            let top1_ok = got_top[0] == ref_top[0];
            let s5g: BTreeSet<u32> = got_top[..5].iter().cloned().collect();
            let s5r: BTreeSet<u32> = ref_top[..5].iter().cloned().collect();
            let s20g: BTreeSet<u32> = got_top.iter().cloned().collect();
            let s20r: BTreeSet<u32> = ref_top.iter().cloned().collect();
            let overlap20 = s20g.intersection(&s20r).count();
            c.note = format!(
                "top1 {}=={} | top5 set {} | top20 overlap {}/20",
                got_top[0],
                ref_top[0],
                if s5g == s5r { "EXACT" } else { "MISMATCH" },
                overlap20
            );
            if !top1_ok || s5g != s5r || overlap20 < 18 || c.max_abs > *abs {
                c.verdict = "FAIL";
            }
        }
    }
    c
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: dsv4-forward <model-dir> <fixtures.json>");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    println!(
        "dsv4-forward gate | model {} | fixtures {} | variant {} ({:?})",
        dir.display(),
        args[2],
        spec.variant_tag,
        spec.variant
    );

    // ---- fixture integrity: npz payload sha256 must match the JSON, per array ----
    let npz = read_npz(&spec.npz_path);
    let mut failures: Vec<String> = Vec::new();
    for (name, (shape, sha)) in &spec.arrays {
        match npz.get(name) {
            None => failures.push(format!("fixture npz missing array {name}")),
            Some((nshape, _, nsha)) => {
                if nshape != shape {
                    failures.push(format!(
                        "fixture shape disagreement {name}: json {shape:?} npz {nshape:?}"
                    ));
                }
                if nsha != sha {
                    failures.push(format!(
                        "fixture sha256 mismatch {name}: json {sha} npz {nsha} — corrupted fixture"
                    ));
                }
            }
        }
    }
    for name in npz.keys() {
        if !spec.arrays.contains_key(name) {
            failures.push(format!("npz array {name} not documented in fixture json"));
        }
    }
    if !failures.is_empty() {
        for f in &failures {
            println!("  FAIL: {f}");
        }
        println!(
            "FORWARD GATE: FAIL ({} fixture-integrity failures)",
            failures.len()
        );
        std::process::exit(1);
    }
    println!(
        "fixture integrity: {} arrays, every payload sha256 matches the json",
        npz.len()
    );

    let model = Dsv4Model::open(dir);
    let d = model.cfg();
    let hc = d.hc_mult as usize;
    let hidden = model.mc.n_embd as usize;
    let n_trunk = model.mc.n_layer - model.mc.nextn_predict_layers;
    println!(
        "config: {} trunk + {} MTP layers, hidden {hidden}, hc_mult {hc}, {} hash layers",
        n_trunk, model.mc.nextn_predict_layers, d.num_hash_layers
    );

    // capture layers derived from the fixture array names, never hardcoded
    let mut cap32: BTreeSet<u32> = BTreeSet::new();
    let mut cap160: BTreeSet<u32> = BTreeSet::new();
    for name in spec.arrays.keys() {
        if let Some(rest) = name.strip_prefix("c160_layer") {
            if let Ok(n) = rest.split('_').next().unwrap_or("").parse::<u32>() {
                cap160.insert(n);
            }
        } else if let Some(rest) = name.strip_prefix("layer") {
            if let Ok(n) = rest.split('_').next().unwrap_or("").parse::<u32>() {
                cap32.insert(n);
            }
        }
    }
    println!("capture layers: 32-token {cap32:?}, 160-token {cap160:?}");

    let mut table: Vec<ArrayCheck> = Vec::new();
    let mut checked: BTreeSet<String> = BTreeSet::new();
    let mut check = |name: &str, got: &[f32], got_shape: &[usize], table: &mut Vec<ArrayCheck>| {
        let Some((ref_shape, ref_vals, _)) = npz.get(name) else {
            return; // not banked for this fixture set
        };
        let c = check_array(
            name,
            got,
            got_shape,
            ref_shape,
            ref_vals,
            &policy_for(name, &spec),
        );
        println!(
            "  [{}] {} {:?}: max-abs {:.3e} max-rel {:.3e} thr {:.3e} over {}{}",
            c.verdict,
            c.name,
            c.shape,
            c.max_abs,
            c.max_rel,
            c.threshold,
            c.n_over,
            if c.note.is_empty() {
                String::new()
            } else {
                format!(" | {}", c.note)
            }
        );
        checked.insert(name.to_string());
        table.push(c);
    };

    // ================= 32-token run: full trunk + head + MTP =================
    let ids = &spec.tokens_32;
    let s = ids.len();
    let e = model.embed_rows(ids);
    check("embed_out", &e, &[1, s, hidden], &mut table);
    let mut h = hc_expand(&e, s, hc, hidden);
    for lid in 0..n_trunk {
        let blk = BlockW::load(&model, &format!("layers.{lid}"), lid, MAX_SEQ);
        let mut cap = BlockCapture::default();
        let want = cap32.contains(&lid);
        h = blk.forward(&model, &h, s, ids, spec.variant, want.then_some(&mut cap));
        if want {
            check(
                &format!("layer{lid}_out"),
                &h,
                &[1, s, hc, hidden],
                &mut table,
            );
            if let Some(a) = &cap.attn_out {
                check(
                    &format!("layer{lid}_attn_out"),
                    a,
                    &[1, s, hidden],
                    &mut table,
                );
            }
            if let Some((kv, nb)) = &cap.attn.compressor_kv {
                check(
                    &format!("layer{lid}_compressor_kv"),
                    kv,
                    &[1, *nb, kv.len() / nb],
                    &mut table,
                );
            }
            if let Some((kv, nb)) = &cap.attn.indexer_kv {
                check(
                    &format!("layer{lid}_indexer_kv"),
                    kv,
                    &[1, *nb, kv.len() / nb],
                    &mut table,
                );
            }
            if let Some((sc, nb)) = &cap.attn.index_score {
                check(
                    &format!("layer{lid}_index_score"),
                    sc,
                    &[1, s, *nb],
                    &mut table,
                );
            }
        }
        println!("layer {lid} done  t={:.1}s", t0.elapsed().as_secs_f64());
    }
    let logits = trunk_logits_last(&model, &h, s);
    check("final_logits_last", &logits, &[logits.len()], &mut table);
    // MTP (NextN) is fixture-driven: 0731 trunk-only sets bank no mtp_logits_last (the
    // checkpoint has no NextN head — `mtp.*` is the DSpark drafter, a separate oracle).
    if spec.arrays.contains_key("mtp_logits_last") {
        let mlogits = mtp_logits_last(&model, &h, s, ids, spec.variant, MAX_SEQ);
        check("mtp_logits_last", &mlogits, &[mlogits.len()], &mut table);
        println!("trunk+mtp logits done t={:.1}s", t0.elapsed().as_secs_f64());
    } else {
        println!(
            "trunk logits done t={:.1}s (no mtp_logits_last in this fixture set — skipped by contract)",
            t0.elapsed().as_secs_f64()
        );
    }

    // ================= 160-token run: layers 0..=max(cap160) =================
    if let (Some(ids160), Some(&max_l)) = (&spec.tokens_160, cap160.iter().max()) {
        let s2 = ids160.len();
        let e2 = model.embed_rows(ids160);
        let mut h2 = hc_expand(&e2, s2, hc, hidden);
        for lid in 0..=max_l {
            let blk = BlockW::load(&model, &format!("layers.{lid}"), lid, MAX_SEQ);
            let mut cap = BlockCapture::default();
            let want = cap160.contains(&lid);
            h2 = blk.forward(
                &model,
                &h2,
                s2,
                ids160,
                spec.variant,
                want.then_some(&mut cap),
            );
            if want {
                // banked as LAST-position slices
                check(
                    &format!("c160_layer{lid}_out_last"),
                    &h2[(s2 - 1) * hc * hidden..],
                    &[1, hc, hidden],
                    &mut table,
                );
                if let Some(a) = &cap.attn_out {
                    check(
                        &format!("c160_layer{lid}_attn_out_last"),
                        &a[(s2 - 1) * hidden..],
                        &[1, hidden],
                        &mut table,
                    );
                }
                if let Some((kv, nb)) = &cap.attn.compressor_kv {
                    check(
                        &format!("c160_layer{lid}_compressor_kv"),
                        kv,
                        &[1, *nb, kv.len() / nb],
                        &mut table,
                    );
                }
            }
            println!("c160 layer {lid} done t={:.1}s", t0.elapsed().as_secs_f64());
        }
    }

    // every banked array must have been checked
    for name in spec.arrays.keys() {
        if !checked.contains(name) {
            failures.push(format!(
                "banked array {name} was never computed by the gate"
            ));
        }
    }
    for c in &table {
        if c.verdict == "FAIL" {
            failures.push(format!("{}: {}", c.name, c.note));
        }
    }

    println!("\n== gate table (variant {}) ==", spec.variant_tag);
    println!("| array | shape | max-abs | max-rel | threshold | verdict |");
    println!("|---|---|---|---|---|---|");
    for c in &table {
        println!(
            "| {} | {:?} | {:.3e} | {:.3e} | {:.3e} | {}{} |",
            c.name,
            c.shape,
            c.max_abs,
            c.max_rel,
            c.threshold,
            c.verdict,
            if c.note.is_empty() {
                String::new()
            } else {
                format!(" ({})", c.note)
            }
        );
    }
    println!("\nelapsed: {:.1}s", t0.elapsed().as_secs_f64());
    if failures.is_empty() {
        println!(
            "FORWARD GATE [{}]: PASS ({} arrays compared, 0 failures)",
            spec.variant_tag,
            table.len()
        );
    } else {
        println!(
            "FORWARD GATE [{}]: FAIL ({} failures)",
            spec.variant_tag,
            failures.len()
        );
        for f in &failures {
            println!("  FAIL: {f}");
        }
        std::process::exit(1);
    }
}
