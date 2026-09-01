//! dsv4-gpu-gate: DeepSeek-V4-Flash GPU output-sample gate (lane 4, gate (a) + the
//! expert-dequant sub-gate + the VRAM placement table of gate (c)).
//!
//! Replays the lane-2 fixture inputs through the 2-card GPU forward
//! (memra_engine::dsv4_gpu) and compares every TRUNK banked array (mtp_logits_last is
//! SKIPPED — the MTP GPU path is optional for this lane and a skip is never a PASS).
//! ONE fixture variant per invocation, read from the JSON.
//!
//! Usage: dsv4-gpu-gate <model-dir> <fixtures.json> [dev0,dev1]      exit 0 PASS
//!
//! ── Threshold doctrine (bf16-vs-f32, derived in RECEIPTS.md "Lane 4", not guessed;
//!    corrected after run 1 — the correction is banked in RECEIPTS.md too) ──
//! Weights are EXACT in bf16 (rung proofs at load are refusals). f32 islands run f32/f64.
//! The only rounding vs the CPU oracle is the bf16 cast of activation inputs at the
//! non-island GEMMs: unit roundoff u = 2⁻⁸. Each residual sub-block injects ~2 rounded
//! GEMM hops; hc comb is doubly-stochastic and rmsnorm re-anchors scale, so per-element
//! errors random-walk with σ ≈ u·√d·scale. The gate bounds the MAX-ABS over an
//! n-element array, so the extreme-value factor of n zero-mean draws applies:
//!     thr(array) = u · √d · √(2·ln n) · absmax(fixture)
//! (run-1 correction: the banked formula used an ad-hoc 2× where √(2 ln n) belongs —
//! 4.85 at n=129280; measured drift at every banked depth sits at 0.3-0.7× this bound).
//! Depths: layerN_out d=2(N+1); layerN_attn_out d=2N+1; compressor/indexer arrays d=2N
//! (their input's depth; the mechanisms themselves are f32 islands); final logits d=86.
//! Quantizer-grid arrays (run-1 correction #2): indexer_kv is on the e2m1 grid — bf16
//! upstream skew flips codes at boundaries (~u/grid-step ≈ 1/64 of elements expected).
//! A one-step flip from 0 to ±0.5·s has |diff| == max(|got|,|ref|), so the adjacency
//! bound is |diff| ≤ 1.0 × max(|got|,|ref|) (run 1 used 0.5 and rejected legitimate
//! 0↔0.5 flips), budget 5% of elements. index_score consumes the flipped kv/q: one
//! flipped element perturbs a head dot by ≤ 2·s_kv·6·s_q = 12·s_q·s_kv against a
//! 128-term dot — a few percent of the score scale; exceeders are budgeted 5% of
//! elements, each ≤ 0.05·absmax (run 1 modeled the score as continuous — wrong for the
//! FP4 path). Selection is unaffected at gate lengths (all completed blocks selected;
//! −inf causality mask compared EXACTLY). Under the clamp-only contract act_quant is an
//! IDENTITY (448·pow2ceil(amax/448) ≥ amax — the clamp never bites) — compressor_kv
//! gets NO flip budget on that variant. WS-threshold-lane correction (2026-08-20):
//! under the REF variant act_quant REALLY rounds compressor_kv to e4m3, so the
//! bf16-dequant arm (whose analog threshold does not subsume a one-step top-binade
//! flip) carries a derived flip policy — adjacency ≤ 2⁻³·max(|got|,|ref|), budget 5% —
//! keyed on (variant, arm), never on the card class: the flip channel was measured in
//! the same magnitude class on Server and Workstation silicon alike, and every banked
//! Server-class behavior (native arm, clamp-only variant) is byte-unchanged — pinned
//! by the policy-keying test below.
//! Mandatory id gates on logits rows: top-1 exact, top-5 set-exact, and (run-3
//! correction, fully derived — no fixed overlap floor) top-20 set changes confined to
//! the drift-noise band of the rank-20 boundary: every symmetric-difference id must
//! satisfy |ref_logit(id) − ref_logit(rank 20)| ≤ 3·√2·u·√d·|ref rank-20 logit|.
//! An id outside that band dropping out or climbing in is a REAL bug. Raw overlap and
//! the 20/21 boundary gaps are always printed; the greedy-continuation gate remains the
//! end-to-end semantic instrument.

use memra_engine::dsv4_gpu::{Dsv4Gpu, GpuCapture};
use memra_gguf::dsv4_forward::{
    ActQuantVariant, FixtureSpec, drift_coeff, expert_arm_native, quant_depth_of, read_npz,
};
use std::collections::BTreeSet;
use std::path::Path;

const U_BF16: f64 = 1.0 / 256.0; // 2^-8

/// Lane 7: per-array drift coefficient of the active numeric class. bf16 arm:
/// C = u_b·√d_b (the lane-4 doctrine unchanged). Native arm: C = √(d_b·u_b² + d_q·u_q²)
/// with d_q = 2 quantized hops per upstream MoE sub-block (RECEIPTS.md "Lane 7" table).
fn class_coeff(name: &str, d_b: f64, native: bool) -> f64 {
    let d_q = if native { quant_depth_of(name) } else { 0.0 };
    drift_coeff(d_b, d_q)
}

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
    Derived {
        coeff: f64,
        flip_budget_frac: f64,
        /// per-element adjacency bound: |diff| ≤ flip_rel · max(|got|,|ref|) (0 = none)
        flip_rel: f32,
        /// alternative flip bound: |diff| ≤ flip_abs_of_amax · absmax_ref (0 = none)
        flip_abs_of_amax: f32,
    },
    Logits {
        coeff: f64,
        top_ids: Vec<u32>,
        /// native class: top-1/top-5 governed by the band rule (a flip must be in-band)
        /// instead of the bf16 class's strict top-1-exact + top-5-set-exact
        native: bool,
    },
}

fn depth_of(name: &str) -> u32 {
    // layer index parsed from the name; depths per the doctrine above
    let core = name.strip_prefix("c160_").unwrap_or(name);
    let n: u32 = core
        .strip_prefix("layer")
        .and_then(|r| r.split('_').next())
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    if core.contains("_out") && !core.contains("attn_out") {
        2 * (n + 1)
    } else if core.contains("attn_out") {
        2 * n + 1
    } else {
        2 * n
    }
}

fn policy_for(name: &str, spec: &FixtureSpec) -> Option<Policy> {
    policy_for_arm(name, spec, expert_arm_native())
}

fn policy_for_arm(name: &str, spec: &FixtureSpec, native: bool) -> Option<Policy> {
    match name {
        "embed_out" => Some(Policy::BitExact),
        "final_logits_last" => Some(Policy::Logits {
            coeff: class_coeff(name, 86.0, native),
            top_ids: spec.top20_ids.clone(),
            native,
        }),
        // MTP head: 2 more sub-blocks past the trunk's 86 (+ the e/h_proj casts).
        // mtp_top20 is absent on 0731-lineage trunk-only fixture sets; this policy is
        // only reachable when the set banks mtp_logits_last, so a missing json entry
        // there is fixture corruption — refuse loudly.
        "mtp_logits_last" => Some(Policy::Logits {
            coeff: class_coeff(name, 88.0, native),
            top_ids: spec
                .mtp_top20_ids
                .clone()
                .expect("fixture banks mtp_logits_last but json has no mtp_top20"),
            native,
        }),
        // native budgets re-derived for the class (RECEIPTS.md "Lane 7"): upstream rel
        // drift ≈ √(4u_b²+4u_q²) ≈ 0.125 vs e2m1 min one-step gap ≈ 0.25 ⇒ expected
        // one-step flip fraction ≈ 0.5 (budget 0.75); index_score flip noise on a
        // 128-dot ≈ 0.1-0.2 of scale (bound 0.25·absmax, budget 0.5)
        n if n.contains("indexer_kv") => Some(Policy::Derived {
            coeff: class_coeff(n, depth_of(n) as f64, native),
            flip_budget_frac: if native { 0.75 } else { 0.05 },
            flip_rel: 1.0, // one e2m1 step from 0 has |diff| == max(|got|,|ref|)
            flip_abs_of_amax: 0.0,
        }),
        n if n.contains("index_score") => Some(Policy::Derived {
            coeff: class_coeff(n, depth_of(n) as f64, native),
            flip_budget_frac: if native { 0.5 } else { 0.05 },
            flip_rel: 0.0,
            // one-flip propagation estimate (bf16) / native flip-noise bound (doc header)
            flip_abs_of_amax: if native { 0.25 } else { 0.05 },
        }),
        // compressor_kv, WS-threshold-lane correction (2026-08-20; derived, not
        // loosened): under the REF variant the kv QAT sim REALLY rounds to e4m3
        // (RefFp8Round), so upstream bf16 drift flips isolated codes at RNE
        // boundaries — the flip channel RECEIPTS.md "0731 REF set" predicted
        // (measured: one-step flips, ≤ 2⁻³·max(|got|,|ref|) on normals; box7 WS
        // idx 3263 ref 2.25 → got 2.00 = adjacent codes, ratio 0.111; box4 Server
        // shows the same 2.5e-1 magnitude class — the channel is variant physics,
        // NOT card-class silicon). The NATIVE arm keeps ZERO budget: its class
        // threshold C(2N,2N)·√(2 ln n)·absmax ≈ 0.51·absmax subsumes one-step
        // flips (the banked lane-7 derivation, Server receipts byte-unchanged).
        // The bf16-dequant arm's analog threshold (≈ 0.032·absmax at layer 2)
        // does NOT subsume a top-binade flip (0.25 at |v|≈2.25), so ref+bf16 gets
        // the derived flip policy: adjacency bound 2⁻³ (one e4m3 step among
        // normals — near-zero multi-code moves are absolutely tiny and never
        // reach the threshold), budget 5% of n (measured exceeders 1/4096;
        // expected flip fraction u_b·√d/2⁻³ ≈ 6-8%, over-threshold subset ≪ 5% —
        // the house indexer_kv cushion). The CLAMP-ONLY variant keeps ZERO
        // budget on BOTH arms: act_quant is an identity there (448·pow2ceil(
        // amax/448) ≥ amax — the clamp never bites), so any exceeder is real.
        n if n.contains("compressor_kv") => {
            let ref_flip_channel = spec.variant == ActQuantVariant::RefFp8Round && !native;
            Some(Policy::Derived {
                coeff: class_coeff(n, depth_of(n) as f64, native),
                flip_budget_frac: if ref_flip_channel { 0.05 } else { 0.0 },
                flip_rel: if ref_flip_channel { 0.125 } else { 0.0 },
                flip_abs_of_amax: 0.0,
            })
        }
        n => Some(Policy::Derived {
            coeff: class_coeff(n, depth_of(n) as f64, native),
            flip_budget_frac: 0.0,
            flip_rel: 0.0,
            flip_abs_of_amax: 0.0,
        }),
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

#[allow(clippy::too_many_arguments)]
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
            if c.max_abs > 0.0 {
                c.verdict = "FAIL";
                c.note = format!(
                    "{} non-identical elements (bit-exact required)",
                    diffs.len()
                );
            }
        }
        Policy::Derived {
            coeff,
            flip_budget_frac,
            flip_rel,
            flip_abs_of_amax,
        } => {
            let ev = (2.0 * (got.len().max(2) as f64).ln()).sqrt(); // extreme-value factor
            c.threshold = coeff.max(U_BF16) * ev * absmax_ref as f64;
            let over: Vec<&(usize, f64)> = diffs.iter().filter(|(_, d)| *d > c.threshold).collect();
            c.n_over = over.len();
            if !over.is_empty() {
                let budget = (*flip_budget_frac * got.len() as f64).floor() as usize;
                let bound_ok = |i: usize, d: f64| -> bool {
                    if *flip_abs_of_amax > 0.0 {
                        d <= *flip_abs_of_amax as f64 * absmax_ref as f64
                    } else if *flip_rel > 0.0 {
                        d <= *flip_rel as f64 * (got[i].abs() as f64).max(ref_vals[i].abs() as f64)
                    } else {
                        false
                    }
                };
                if over.len() <= budget && over.iter().all(|(i, d)| bound_ok(*i, *d)) {
                    c.note = format!(
                        "{} quantizer-flip exceeder(s) within the documented bound (budget {})",
                        over.len(),
                        budget
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
        Policy::Logits {
            coeff,
            top_ids: want,
            native,
        } => {
            let ev = (2.0 * (got.len().max(2) as f64).ln()).sqrt();
            c.threshold = coeff * ev * absmax_ref as f64;
            let got_top = top_ids(got, 21);
            let ref_top = top_ids(ref_vals, 21);
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
            let s20g: BTreeSet<u32> = got_top[..20].iter().cloned().collect();
            let s20r: BTreeSet<u32> = ref_top[..20].iter().cloned().collect();
            let overlap20 = s20g.intersection(&s20r).count();
            // boundary diagnostics: rank-20/21 logit gap on both sides
            let ref20 = ref_vals[ref_top[19] as usize];
            let gap_ref = ref20 - ref_vals[ref_top[20] as usize];
            let gap_got = got[got_top[19] as usize] - got[got_top[20] as usize];
            // top-20 id rule (DERIVED, run-3 correction): every id in the symmetric
            // difference of the two top-20 SETS must sit inside the drift-noise band of
            // the rank-20 boundary. Per-logit drift σ(x) = u·√d·|x|; a rank swap
            // compares two logits (√2), band = 3·√2·σ(ref rank-20 logit). An id more
            // than `band` above the reference boundary must never drop out, and an id
            // more than `band` below it must never climb in — either is a REAL bug, not
            // noise. No fixed overlap floor: the rule scales with the measured cluster
            // structure (trunk run 1: 2 misses in a 0.27-gap boundary; MTP: 4 misses in
            // a 0.026/0.037/0.065 near-tie cluster — both are in-band physics).
            let band = 3.0 * std::f64::consts::SQRT_2 * coeff * (ref20.abs() as f64);
            let mut out_of_band: Vec<(u32, f32)> = Vec::new();
            for id in s20r.symmetric_difference(&s20g) {
                let dist = (ref_vals[*id as usize] - ref20).abs() as f64;
                if dist > band {
                    out_of_band.push((*id, ref_vals[*id as usize]));
                }
            }
            c.note = format!(
                "top1 {}=={} | top5 set {} | top20 overlap {}/20 | 20/21 gap ref {:.4} gpu {:.4} | boundary band ±{:.3}: {} set-diff id(s), {} out-of-band",
                got_top[0],
                ref_top[0],
                if s5g == s5r { "EXACT" } else { "MISMATCH" },
                overlap20,
                gap_ref,
                gap_got,
                band,
                s20r.symmetric_difference(&s20g).count(),
                out_of_band.len()
            );
            if !out_of_band.is_empty() {
                c.note.push_str(&format!(" | OUT-OF-BAND: {out_of_band:?}"));
            }
            // native class (lane 7): a top-1 flip must be an in-band near-tie on the
            // REF row; top-5 set changes confined to the rank-5 boundary band (the
            // bf16 class keeps lane-4's strict top-1-exact + top-5-set-exact).
            let top1_band_ok = top1_ok || {
                let m = (ref_vals[ref_top[0] as usize] as f64
                    - ref_vals[got_top[0] as usize] as f64)
                    .abs();
                m <= 3.0
                    * std::f64::consts::SQRT_2
                    * coeff
                    * (ref_vals[ref_top[0] as usize].abs() as f64)
            };
            let ref5_boundary = ref_vals[ref_top[4] as usize];
            let band5 = 3.0 * std::f64::consts::SQRT_2 * coeff * (ref5_boundary.abs() as f64);
            let top5_band_ok = s5r
                .symmetric_difference(&s5g)
                .all(|id| (ref_vals[*id as usize] - ref5_boundary).abs() as f64 <= band5);
            if !top1_band_ok {
                c.note.push_str(" | top1 OUT-OF-BAND");
            }
            if *native && s5g != s5r && !top5_band_ok {
                c.note.push_str(" | top5 OUT-OF-BAND");
            }
            let strict_ok = if *native {
                top1_band_ok && top5_band_ok
            } else {
                top1_ok && s5g == s5r
            };
            if !strict_ok || !out_of_band.is_empty() || c.max_abs > c.threshold {
                c.verdict = "FAIL";
            }
        }
    }
    c
}

fn vram_table(gpu: &Dsv4Gpu, tag: &str) {
    match gpu.vram_report() {
        Ok(rows) => {
            for (dev, free, total, resident) in rows {
                println!(
                    "[vram {tag}] dev{dev}: used {:.2} GiB / {:.2} GiB (free {:.2}), loader-resident {:.2} GiB",
                    (total - free) as f64 / 2f64.powi(30),
                    total as f64 / 2f64.powi(30),
                    free as f64 / 2f64.powi(30),
                    resident as f64 / 2f64.powi(30),
                );
            }
        }
        Err(err) => println!("[vram {tag}] report failed: {err}"),
    }
}

/// Expert-dequant sub-gate: GPU NVFP4/MXFP4 kernel output must be BIT-EXACT vs the
/// lane-1 host decoders (exactness proofs make any mismatch a kernel bug). `prefix` is
/// "layers.N" (trunk NVFP4) or "mtp.0" (MXFP4 — dispatched by the layer's detected kind).
fn expert_dequant_subgate(gpu: &Dsv4Gpu, samples: &[(&str, usize, &str)]) -> bool {
    use cudarc::driver::DevicePtr;
    use memra_engine::dsv4_gpu::ExpertKind;
    let mut ok = true;
    for &(prefix, ex, proj) in samples {
        let name = format!("{prefix}.ffn.experts.{ex}.{proj}");
        let (shape, host) = gpu.model.tensor_f32(&name);
        let (rows, cols) = (shape[0], shape[1]);
        let (stage, layer) = if prefix == "mtp.0" {
            let m = gpu.mtp.as_ref().expect("mtp loaded");
            (&gpu.stages[gpu.stages.len() - 1], &m.layer)
        } else {
            let il: u32 = prefix.strip_prefix("layers.").unwrap().parse().unwrap();
            let stage = &gpu.stages[gpu.layer_stage[il as usize]];
            (
                stage,
                stage.layers.iter().find(|l| l.il == il).expect("layer"),
            )
        };
        let pi = match proj {
            "w1" => 0usize,
            "w2" => 1,
            _ => 2,
        };
        stage.gpu.ctx.bind_to_thread().expect("bind ctx");
        let stream = stage.gpu.stream();
        let wbytes = rows * cols / 2;
        let sbytes = match layer.expert_kind {
            ExpertKind::Nvfp4 => rows * cols / 16,
            ExpertKind::Mxfp4 => rows * cols / 32,
        };
        let wp = (layer.experts_w.device_ptr(&stream).0 as usize + (ex * 3 + pi) * wbytes)
            as *const std::os::raw::c_void;
        let scp = (layer.experts_sc.device_ptr(&stream).0 as usize + (ex * 3 + pi) * sbytes)
            as *const std::os::raw::c_void;
        let dst = stage.deq[pi].device_ptr(&stream).0 as *mut std::os::raw::c_void;
        let sv = stream.cu_stream() as *mut std::os::raw::c_void;
        let rc = unsafe {
            match layer.expert_kind {
                ExpertKind::Nvfp4 => memra_engine::dsv4_ffi::memra_dsv4_nvfp4_deq_bf16(
                    wp,
                    scp,
                    layer.experts_s2[ex * 3 + pi],
                    rows as i32,
                    cols as i32,
                    dst,
                    sv,
                ),
                ExpertKind::Mxfp4 => memra_engine::dsv4_ffi::memra_dsv4_mxfp4_deq_bf16(
                    wp,
                    scp,
                    rows as i32,
                    cols as i32,
                    dst,
                    sv,
                ),
            }
        };
        assert_eq!(rc, 0, "dequant kernel rc {rc}");
        let mut raw = vec![0u8; rows * cols * 2];
        stream
            .memcpy_dtoh(&stage.deq[pi].slice(0..rows * cols * 2), &mut raw[..])
            .expect("dtoh");
        stream.synchronize().expect("sync");
        let mut mismatches = 0usize;
        for i in 0..rows * cols {
            let b = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
            let g = f32::from_bits((b as u32) << 16);
            if g.to_bits() != host[i].to_bits() {
                if mismatches == 0 {
                    println!("  [FAIL] {name}[{i}]: gpu {g} vs host {}", host[i]);
                }
                mismatches += 1;
            }
        }
        let verdict = if mismatches == 0 { "PASS" } else { "FAIL" };
        println!(
            "  [{verdict}] expert-dequant {name} [{rows}x{cols}]: {mismatches} mismatches (bit-exact required)"
        );
        ok &= mismatches == 0;
    }
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dsv4-gpu-gate <model-dir> <fixtures.json> [dev0,dev1]");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = FixtureSpec::load(Path::new(&args[2]));
    let devices: Vec<usize> = args
        .get(3)
        .map(|s| {
            s.split(',')
                .map(|x| x.parse().expect("device ordinal"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1]);
    println!(
        "dsv4-gpu-gate | model {} | fixtures {} | variant {} ({:?}) | devices {devices:?}",
        dir.display(),
        args[2],
        spec.variant_tag,
        spec.variant
    );

    // fixture integrity (sha256 per array, lane-3 rule)
    let npz = read_npz(&spec.npz_path);
    let mut failures: Vec<String> = Vec::new();
    for (name, (shape, sha)) in &spec.arrays {
        match npz.get(name) {
            None => failures.push(format!("fixture npz missing array {name}")),
            Some((nshape, _, nsha)) => {
                if nshape != shape || nsha != sha {
                    failures.push(format!("fixture integrity mismatch on {name}"));
                }
            }
        }
    }
    if !failures.is_empty() {
        for f in &failures {
            println!("  FAIL: {f}");
        }
        std::process::exit(1);
    }
    println!(
        "fixture integrity: {} arrays, payload sha256 all match",
        npz.len()
    );

    let max_seq = 4096.min(
        spec.tokens_160
            .as_ref()
            .map(|t| t.len())
            .unwrap_or(0)
            .max(512),
    );
    let gpu = Dsv4Gpu::load(dir, &devices, spec.variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {} (stage0 {} layers, stage1 {} layers), t={:.0}s",
        gpu.split_at,
        gpu.stages[0].layers.len(),
        gpu.stages[1].layers.len(),
        t0.elapsed().as_secs_f64()
    );
    vram_table(&gpu, "post-load");

    // ---- expert-dequant sub-gate (bit-exact, both quadrant layers + the lane-1 pins,
    //      both recipes: trunk NVFP4 and MTP MXFP4)
    let n_trunk = gpu.model.mc.n_layer - gpu.model.mc.nextn_predict_layers;
    let stage1_prefix = format!("layers.{}", gpu.split_at);
    let last_prefix = format!("layers.{}", n_trunk - 1);
    let mut samples: Vec<(&str, usize, &str)> = vec![
        ("layers.0", 0, "w1"),
        ("layers.2", 100, "w3"),
        ("layers.20", 7, "w1"), // lane-1 NVFP4 oracle pin tensor
        (&stage1_prefix, 31, "w2"),
        (&last_prefix, 255, "w2"),
    ];
    // MXFP4 samples only when a NextN block is resident (0731: mtp.* is the DSpark
    // drafter, not loaded on GPU — its dequant gate belongs to the drafter lane).
    if gpu.mtp.is_some() {
        samples.push(("mtp.0", 7, "w1")); // lane-1 MXFP4 oracle pin tensor
        samples.push(("mtp.0", 200, "w2"));
    } else {
        println!(
            "  (MXFP4 mtp.* samples skipped: no NextN block resident — DSpark drafter lane owns that gate)"
        );
    }
    println!("\n== expert-dequant sub-gate (GPU kernels vs lane-1 host decoders) ==");
    let deq_ok = expert_dequant_subgate(&gpu, &samples);

    // capture layers derived from fixture names
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

    let hc = gpu.model.cfg().hc_mult as usize;
    let hidden = gpu.model.mc.n_embd as usize;
    let mut table: Vec<ArrayCheck> = Vec::new();
    let mut checked: BTreeSet<String> = BTreeSet::new();
    let mut skipped: Vec<String> = Vec::new();
    // Calibration instrument (WS-threshold lane): MEMRA_DSV4_GATE_DUMP=<dir> writes
    // every compared GPU array verbatim as little-endian f32 (<dir>/<name>.f32) so the
    // per-element error distribution can be measured offline against the fixture npz.
    // Dump-only: verdicts, thresholds and the compare path are byte-unchanged.
    let dump_dir = std::env::var("MEMRA_DSV4_GATE_DUMP").ok();
    if let Some(d) = &dump_dir {
        std::fs::create_dir_all(d).expect("MEMRA_DSV4_GATE_DUMP dir");
    }
    let mut check = |name: &str, got: &[f32], got_shape: &[usize], table: &mut Vec<ArrayCheck>| {
        let Some((ref_shape, ref_vals, _)) = npz.get(name) else {
            return;
        };
        let Some(policy) = policy_for(name, &spec) else {
            return;
        };
        if let Some(dir) = &dump_dir {
            let mut bytes = Vec::with_capacity(got.len() * 4);
            for v in got {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let p = format!("{dir}/{name}.f32");
            std::fs::write(&p, &bytes).expect("gate dump write");
            println!("  [dump] {name} -> {p} ({} f32)", got.len());
        }
        let c = check_array(name, got, got_shape, ref_shape, ref_vals, &policy);
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

    // ================= 32-token run =================
    let ids = &spec.tokens_32;
    let s = ids.len();
    let mut cap = GpuCapture {
        want: cap32.clone(),
        ..Default::default()
    };
    let fwd = gpu
        .forward(ids, Some(&mut cap), None)
        .expect("forward 32")
        .expect("logits expected");
    let logits = fwd.logits;
    println!("32-token forward done t={:.0}s", t0.elapsed().as_secs_f64());
    vram_table(&gpu, "post-warmup");

    if let Some(e0) = &cap.embed_out {
        check("embed_out", e0, &[1, s, hidden], &mut table);
    }
    for lid in &cap32 {
        if let Some(h) = cap.layer_out.get(lid) {
            check(
                &format!("layer{lid}_out"),
                h,
                &[1, s, hc, hidden],
                &mut table,
            );
        }
        if let Some(a) = cap.attn_out.get(lid) {
            check(
                &format!("layer{lid}_attn_out"),
                a,
                &[1, s, hidden],
                &mut table,
            );
        }
        if let Some((kv, nb)) = cap.compressor_kv.get(lid) {
            check(
                &format!("layer{lid}_compressor_kv"),
                kv,
                &[1, *nb, kv.len() / nb],
                &mut table,
            );
        }
        if let Some((kv, nb)) = cap.indexer_kv.get(lid) {
            check(
                &format!("layer{lid}_indexer_kv"),
                kv,
                &[1, *nb, kv.len() / nb],
                &mut table,
            );
        }
        if let Some((sc, nb)) = cap.index_score.get(lid) {
            check(
                &format!("layer{lid}_index_score"),
                sc,
                &[1, s, *nb],
                &mut table,
            );
        }
    }
    check("final_logits_last", &logits, &[logits.len()], &mut table);
    if spec.arrays.contains_key("mtp_logits_last") {
        if gpu.mtp.is_some() {
            let ml = gpu.mtp_logits_last(&fwd.h_last, ids).expect("mtp forward");
            check("mtp_logits_last", &ml, &[ml.len()], &mut table);
            println!("mtp forward done t={:.0}s", t0.elapsed().as_secs_f64());
        } else {
            skipped.push(
                "mtp_logits_last (MTP block absent from artifact — SKIPPED, not PASS)".into(),
            );
        }
    }

    // ================= 160-token run (early exit after the deepest capture layer) ====
    if let (Some(ids160), Some(&max_l)) = (&spec.tokens_160, cap160.iter().max()) {
        let s2 = ids160.len();
        let mut cap2 = GpuCapture {
            want: cap160.clone(),
            ..Default::default()
        };
        let r = gpu
            .forward(ids160, Some(&mut cap2), Some(max_l))
            .expect("forward 160");
        assert!(r.is_none(), "early exit expected");
        println!(
            "160-token partial forward done t={:.0}s",
            t0.elapsed().as_secs_f64()
        );
        for lid in &cap160 {
            if let Some(h) = cap2.layer_out.get(lid) {
                check(
                    &format!("c160_layer{lid}_out_last"),
                    &h[(s2 - 1) * hc * hidden..],
                    &[1, hc, hidden],
                    &mut table,
                );
            }
            if let Some(a) = cap2.attn_out.get(lid) {
                check(
                    &format!("c160_layer{lid}_attn_out_last"),
                    &a[(s2 - 1) * hidden..],
                    &[1, hidden],
                    &mut table,
                );
            }
            if let Some((kv, nb)) = cap2.compressor_kv.get(lid) {
                check(
                    &format!("c160_layer{lid}_compressor_kv"),
                    kv,
                    &[1, *nb, kv.len() / nb],
                    &mut table,
                );
            }
        }
    }
    vram_table(&gpu, "post-gate");

    // coverage: every banked TRUNK array must have been checked
    for name in spec.arrays.keys() {
        if !checked.contains(name) && name != "mtp_logits_last" {
            failures.push(format!(
                "banked array {name} was never computed by the gate"
            ));
        }
    }
    if !deq_ok {
        failures.push("expert-dequant sub-gate failed".into());
    }
    for c in &table {
        if c.verdict == "FAIL" {
            failures.push(format!("{}: {}", c.name, c.note));
        }
    }

    println!(
        "\n== GPU gate table (variant {}, {} class) ==",
        spec.variant_tag,
        if expert_arm_native() {
            "NATIVE expert arm: C = sqrt(d_b*u_b^2 + d_q*u_q^2), u_q = 2^-4"
        } else {
            "bf16-dequant arm: u = 2^-8 doctrine"
        }
    );
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
    for skip in &skipped {
        println!("| SKIPPED: {skip} |");
    }
    println!(
        "\nelapsed: {:.1}s (informational, single-run, not a perf claim)",
        t0.elapsed().as_secs_f64()
    );
    if failures.is_empty() {
        println!(
            "GPU OUTPUT-SAMPLE GATE [{}]: PASS ({} arrays compared, {} skipped, 0 failures)",
            spec.variant_tag,
            table.len(),
            skipped.len()
        );
    } else {
        println!(
            "GPU OUTPUT-SAMPLE GATE [{}]: FAIL ({} failures)",
            spec.variant_tag,
            failures.len()
        );
        for f in &failures {
            println!("  FAIL: {f}");
        }
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    //! Policy-keying teeth (WS-threshold lane, 2026-08-20). The gate's thresholds are
    //! a PURE function of (array name, fixture variant, expert arm) — there is no
    //! card-class input, and these pins are the receipt that no banked class can be
    //! silently loosened or re-keyed: every coefficient and flip budget below is the
    //! banked lane-4/lane-7 value written as an exact constant, compared exactly.
    use super::*;
    use std::collections::BTreeMap;

    const U_B: f64 = 1.0 / 256.0;
    const U_Q: f64 = 1.0 / 16.0;

    fn spec_with(variant: ActQuantVariant) -> FixtureSpec {
        FixtureSpec {
            variant,
            variant_tag: match variant {
                ActQuantVariant::RefFp8Round => "ref".into(),
                ActQuantVariant::ClampOnly => "clamp-only".into(),
            },
            npz_path: std::path::PathBuf::new(),
            tokens_32: vec![],
            tokens_160: None,
            arrays: BTreeMap::new(),
            top20_ids: vec![0; 21],
            mtp_top20_ids: None,
            contract_fork_final_logits_maxabs: None,
        }
    }

    fn derived(p: Option<Policy>, ctx: &str) -> (f64, f64, f32, f32) {
        match p {
            Some(Policy::Derived {
                coeff,
                flip_budget_frac,
                flip_rel,
                flip_abs_of_amax,
            }) => (coeff, flip_budget_frac, flip_rel, flip_abs_of_amax),
            _ => panic!("MIS-KEYED: {ctx} must resolve to Policy::Derived"),
        }
    }

    /// Exact doctrine pins: d_b = lane-4 bf16 hop count, d_q = lane-7 quantized hop
    /// count (they differ on attn_out: d_b = 2N+1, d_q = 2N).
    fn coeff_pin(d_b: f64, d_q: f64, native: bool) -> f64 {
        if native {
            (d_b * U_B * U_B + d_q * U_Q * U_Q).sqrt()
        } else {
            (d_b * U_B * U_B).sqrt()
        }
    }

    #[test]
    fn compressor_kv_flip_policy_is_variant_and_arm_keyed_only() {
        for (name, d) in [
            ("layer2_compressor_kv", 4.0),
            ("c160_layer3_compressor_kv", 6.0),
        ] {
            let (d_b, d_q) = (d, d);
            // clamp-only variant: ZERO flip budget on BOTH arms — act_quant is an
            // identity there, any exceeder is a real bug.
            for native in [false, true] {
                let (c, b, fr, fa) = derived(
                    policy_for_arm(name, &spec_with(ActQuantVariant::ClampOnly), native),
                    name,
                );
                assert!(
                    b == 0.0 && fr == 0.0 && fa == 0.0,
                    "MIS-KEYED: clamp-only compressor_kv must keep ZERO flip budget \
                     (act_quant identity — any exceeder is real): {name} native={native}"
                );
                assert!(
                    c == coeff_pin(d_b, d_q, native),
                    "LOOSENED: {name} clamp-only coeff drifted from the banked doctrine \
                     (native={native})"
                );
            }
            // ref variant + NATIVE arm: ZERO budget — the lane-7 class threshold
            // subsumes the e4m3 flip channel; this is the banked Server-class receipt
            // behavior and must stay byte-identical.
            let (c, b, fr, fa) = derived(
                policy_for_arm(name, &spec_with(ActQuantVariant::RefFp8Round), true),
                name,
            );
            assert!(
                b == 0.0 && fr == 0.0 && fa == 0.0,
                "MIS-KEYED: native-arm compressor_kv must keep ZERO flip budget \
                 (lane-7 class threshold subsumes the e4m3 flip channel): {name}"
            );
            assert!(
                c == coeff_pin(d_b, d_q, true),
                "LOOSENED: {name} native coeff drifted from the banked lane-7 formula"
            );
            // ref variant + bf16 arm: the derived flip policy, EXACT values — more or
            // less budget, or a different adjacency bound, is a re-derivation that
            // must be argued in RECEIPTS, not slipped in.
            let (c, b, fr, fa) = derived(
                policy_for_arm(name, &spec_with(ActQuantVariant::RefFp8Round), false),
                name,
            );
            assert!(
                c == coeff_pin(d_b, d_q, false),
                "LOOSENED: {name} bf16 coeff drifted from the banked lane-4 doctrine"
            );
            assert!(
                b == 0.05 && fr == 0.125 && fa == 0.0,
                "MIS-DERIVED: ref+bf16 compressor_kv flip policy must be exactly \
                 budget 5%, adjacency 2^-3, no absmax bound: {name}"
            );
        }
    }

    #[test]
    fn analog_and_gridded_policies_pin_the_banked_classes() {
        // Analog arrays: never any flip budget, coefficients exactly the doctrine.
        for (name, d_b, d_q) in [
            ("layer0_out", 2.0, 2.0),
            ("layer0_attn_out", 1.0, 0.0),
            ("layer2_out", 6.0, 6.0),
            ("layer2_attn_out", 5.0, 4.0),
            ("layer3_out", 8.0, 8.0),
            ("layer3_attn_out", 7.0, 6.0),
            ("c160_layer3_out_last", 8.0, 8.0),
            ("c160_layer3_attn_out_last", 7.0, 6.0),
        ] {
            for variant in [ActQuantVariant::RefFp8Round, ActQuantVariant::ClampOnly] {
                for native in [false, true] {
                    let (c, b, fr, fa) =
                        derived(policy_for_arm(name, &spec_with(variant), native), name);
                    assert!(
                        b == 0.0 && fr == 0.0 && fa == 0.0,
                        "LOOSENED: analog array {name} must never carry a flip budget \
                         (variant={variant:?} native={native})"
                    );
                    assert!(
                        c == coeff_pin(d_b, d_q, native),
                        "LOOSENED: {name} coeff drifted from the banked doctrine \
                         (variant={variant:?} native={native})"
                    );
                }
            }
        }
        // Gridded arrays: budgets pinned to the banked lane-4/lane-7 numbers.
        for variant in [ActQuantVariant::RefFp8Round, ActQuantVariant::ClampOnly] {
            for native in [false, true] {
                let (_, b, fr, fa) = derived(
                    policy_for_arm("layer2_indexer_kv", &spec_with(variant), native),
                    "layer2_indexer_kv",
                );
                assert!(
                    b == if native { 0.75 } else { 0.05 } && fr == 1.0 && fa == 0.0,
                    "LOOSENED: indexer_kv flip policy drifted from the banked budgets \
                     (variant={variant:?} native={native})"
                );
                let (_, b, fr, fa) = derived(
                    policy_for_arm("layer2_index_score", &spec_with(variant), native),
                    "layer2_index_score",
                );
                assert!(
                    b == if native { 0.5 } else { 0.05 }
                        && fr == 0.0
                        && fa == if native { 0.25 } else { 0.05 },
                    "LOOSENED: index_score flip policy drifted from the banked budgets \
                     (variant={variant:?} native={native})"
                );
            }
        }
        // Logits: the native flag must equal the arm — a mis-keyed class here would
        // swap the strict bf16 top-1/top-5 rule for the native band rule.
        for native in [false, true] {
            match policy_for_arm(
                "final_logits_last",
                &spec_with(ActQuantVariant::RefFp8Round),
                native,
            ) {
                Some(Policy::Logits {
                    coeff,
                    native: got_native,
                    ..
                }) => {
                    assert!(
                        got_native == native,
                        "MIS-KEYED: logits policy native flag must equal the expert arm"
                    );
                    assert!(
                        coeff == coeff_pin(86.0, 86.0, native),
                        "LOOSENED: final_logits coeff drifted from the banked doctrine \
                         (native={native})"
                    );
                }
                _ => panic!("MIS-KEYED: final_logits_last must resolve to Policy::Logits"),
            }
        }
    }
}
