//! DFlash2 drafter parity gate (lane/dflash2-port-20260820): the memra loader +
//! conv-wrapped forward + windowed attention + candidate-selector walk on the z-lab
//! q38 DFlash2 export must reproduce the z-lab reference (`dflash/model.py`,
//! sha-pinned; tools/dflash2_oracle.py flat dumps) on fixed-seed synthetic inputs.
//!
//! Stages and bars (calibration inherited from dflash_parity/dspark_q38_parity: the
//! f32/bf16 GEMM rides TF32-class compute — rel-to-max 2e-3; integer decisions EXACT):
//!   ctx_features      rel < 2e-3
//!   final             rel < 2e-3   (conv-wrapped 5-layer block forward, ctx 8)
//!   final_win         rel < 2e-3   (REDUCED window: ctx 24 > window 16 — proves the
//!                                   per-query old-side mask arithmetic; the config
//!                                   window is override-restored around the call)
//!   conv prepare/out  rel < 2e-3   (layer-0 attention_conv module isolation)
//!   selector topk     idx EXACT (as descending-sorted pairs), vals rel < 2e-3
//!   selector path     EXACT (7/7)  — the chain walk over the real codebooks
//!
//! HARVEST: DFlash2 is mask-fill by construction (manifest harvest=dflash asserted;
//! anything else REFUSES — DSPARK-POSTMORTEM-20260820.md discipline).
//!
//! Run with MEMRA_DFLASH_PREC=bf16 (parity precision class).
use memra_engine::Engine;
use memra_engine::dflash::DflashDraft;

#[path = "../parity_geometry.rs"]
mod parity_geometry;
use parity_geometry::{Geometry, expect_len, join_bool, join_usize, manifest_path};

fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn read_u32(p: &str) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_gate(name: &str, got: &[f32], want: &[f32], bar: f32) -> bool {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let (mut md, mut mi) = (0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > md {
            md = d;
            mi = i;
        }
    }
    let mx = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let rel = md / mx.max(1e-20);
    let pass = rel < bar;
    println!(
        "{name}: maxdiff {md:.3e} (idx {mi}: got {} want {}), rel-to-max {rel:.3e} -> {}",
        got[mi],
        want[mi],
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = std::env::args()
        .nth(1)
        .expect("usage: dflash2_parity <export_dir> <dump_dir>");
    let cache = std::env::args().nth(2).expect("dump dir");
    if std::env::var("MEMRA_DFLASH_PREC").as_deref() != Ok("bf16") {
        panic!("parity gate requires MEMRA_DFLASH_PREC=bf16");
    }
    // TF32 must be OFF for this gate (instrument setting, like PREC=bf16; serving
    // keeps TF32). cudarc's f32 cuBLASLt rides CUBLAS_COMPUTE_32F_FAST_TF32, and the
    // DFlash2 dynamic conv composes MULTIPLICATIVELY around every sublayer — measured
    // on the box6 fixture 2026-08-20: TF32-on final rel 2.25e-2 vs TF32-off 1.0e-5,
    // with the torch bracket f32-vs-f64 1.2e-5 / bf16-compute 7.1e-2. At TF32-off the
    // 2e-3 bars carry ~200x headroom, so a semantic wiring bug cannot hide inside a
    // precision allowance.
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        panic!(
            "parity gate requires NVIDIA_TF32_OVERRIDE=0 (true-f32 GEMMs; TF32 error \
             amplifies ~10x through the DFlash2 conv composition and would need bars \
             wide enough to hide real bugs)"
        );
    }
    let e = Engine::new(0)?;
    let mut m = DflashDraft::load(&e, std::path::Path::new(&ckpt))?;
    let c = &m.cfg;
    println!(
        "loaded dflash2 draft: {} layers, hidden {}, block {}, taps {:?}, window {} \
         (is_causal {:?})",
        c.n_layer, c.hidden, c.block_size, c.target_layer_ids, c.sliding_window, c.is_causal
    );
    assert!(
        m.dflash2.is_some(),
        "export did not load as DFlash2 — census/family detection broken"
    );
    assert!(
        m.markov.is_none() && m.confidence.is_none(),
        "DFlash2 export must not carry markov/confidence heads"
    );
    let b = c.block_size;
    let h = c.hidden;
    let n_taps = c.target_layer_ids.len();
    let mut ok = true;

    // ---- stage 0: GEOMETRY GATE — config is the authority, the dump is the claimant ----
    let manifest = manifest_path(&cache, "dflash2");
    let regen = format!(
        "python tools/dflash2_oracle.py {ckpt} {cache}/dflash2-oracle.npz \
         <zlab dflash/model.py>   # writes {manifest}"
    );
    let geo = Geometry::load(&manifest, &regen).map_err(|e| -> Box<dyn std::error::Error> {
        eprintln!("{e}");
        "dflash2_parity: REFUSED (geometry manifest)".into()
    })?;
    let (d2_rank, d2_topk, d2_convk, d2_group, d2_vocab) = {
        let d2 = m.dflash2.as_ref().unwrap();
        (d2.rank, d2.top_k, d2.conv_k, d2.group_size, d2.vocab)
    };
    let geo_checks: Vec<Result<(), String>> = vec![
        geo.expect_str("dtype", "f32"),
        geo.expect_str("harvest", "dflash"),
        geo.expect_usize("hidden", h),
        geo.expect_usize("n_layer", c.n_layer),
        geo.expect_usize("block_size", b),
        geo.expect_usize("n_taps", n_taps),
        geo.expect_str("target_layer_ids", &join_usize(&c.target_layer_ids)),
        geo.expect_usize("head_dim", c.head_dim),
        geo.expect_usize("n_head", c.n_head),
        geo.expect_usize("n_head_kv", c.n_kv),
        geo.expect_f64_near("rope_theta", c.rope_theta as f64, 1e-3),
        geo.expect_usize("sliding_window", c.sliding_window),
        geo.expect_str(
            "is_causal",
            if c.is_causal == Some(true) { "1" } else { "0" },
        ),
        geo.expect_str("layer_sliding", &join_bool(&c.layer_sliding)),
        geo.expect_usize("vocab", d2_vocab),
        geo.expect_usize("selector_rank", d2_rank),
        geo.expect_usize("selector_top_k", d2_topk),
        geo.expect_usize("conv_kernel_size", d2_convk),
        geo.expect_usize("conv_group_size", d2_group),
        geo.expect_usize("ndr", b - 1),
    ];
    let mut geo_fails = 0usize;
    for r in &geo_checks {
        if let Err(msg) = r {
            eprintln!("{msg}");
            geo_fails += 1;
        }
    }
    if geo_fails > 0 {
        eprintln!(
            "== dflash2_parity: REFUSED — {geo_fails} geometry field(s) disagree with the \
             export. A value compare across two different programs is not a parity result. =="
        );
        std::process::exit(1);
    }
    let ctx = geo.need_usize("ctx")?;
    let win_ctx = geo.need_usize("win_ctx")?;
    let win_window = geo.need_usize("win_window")?;
    let nd = b - 1;
    println!(
        "geometry OK ({n_geo} fields vs {manifest}): ctx={ctx} hidden={h} n_taps={n_taps} \
         block={b} window={win} rank={d2_rank} top_k={d2_topk} conv k={d2_convk}/g{d2_group} \
         win-cell ctx={win_ctx} window={win_window} harvest=dflash (nd={nd}, first_row=1)",
        n_geo = geo_checks.len(),
        win = c.sliding_window
    );

    // ---- stage 1: ctx features ----
    let taps = read_f32(&format!("{cache}/dflash2-taps.f32"));
    let ref_ctxf = read_f32(&format!("{cache}/dflash2-ctx_features.f32"));
    for r in [
        expect_len(
            "dflash2-taps.f32",
            taps.len(),
            ctx * n_taps * h,
            "ctx*n_taps*hidden",
        ),
        expect_len(
            "dflash2-ctx_features.f32",
            ref_ctxf.len(),
            ctx * h,
            "ctx*hidden",
        ),
    ] {
        if let Err(msg) = r {
            eprintln!("{msg}");
            eprintln!("== dflash2_parity: REFUSED (dump length vs config) ==");
            std::process::exit(1);
        }
    }
    let taps_d = e.htod(&taps)?;
    let ctxf = m.ctx_features(&e, &taps_d, ctx)?;
    ok &= rel_gate("ctx_features", &e.dtoh(&ctxf)?, &ref_ctxf, 2e-3);

    // ---- stages 2-3: conv-wrapped block forward, final hidden ----
    let noise = read_f32(&format!("{cache}/dflash2-noise.f32"));
    if let Err(msg) = expect_len("dflash2-noise.f32", noise.len(), b * h, "block_size*hidden") {
        eprintln!("{msg}");
        eprintln!("== dflash2_parity: REFUSED (dump length vs config) ==");
        std::process::exit(1);
    }
    let noise_d = e.htod(&noise)?;
    let pos: Vec<i32> = (0..(ctx + b) as i32).collect();
    let fin = m.forward(&e, &taps_d, &noise_d, &pos, ctx)?;
    ok &= rel_gate(
        "final",
        &e.dtoh(&fin)?,
        &read_f32(&format!("{cache}/dflash2-final.f32")),
        2e-3,
    );

    // ---- stage 3w: REDUCED-window forward (ctx crosses the window; per-query
    // old-side masking must match the reference's symmetric mask) ----
    let taps_win = read_f32(&format!("{cache}/dflash2-taps_win.f32"));
    if let Err(msg) = expect_len(
        "dflash2-taps_win.f32",
        taps_win.len(),
        win_ctx * n_taps * h,
        "win_ctx*n_taps*hidden",
    ) {
        eprintln!("{msg}");
        eprintln!("== dflash2_parity: REFUSED (dump length vs config) ==");
        std::process::exit(1);
    }
    let taps_win_d = e.htod(&taps_win)?;
    let pos_win: Vec<i32> = (0..(win_ctx + b) as i32).collect();
    let saved_window = m.cfg.sliding_window;
    m.cfg.sliding_window = win_window; // gate-scoped override, mirrors the oracle's
    let fin_win = m.forward(&e, &taps_win_d, &noise_d, &pos_win, win_ctx)?;
    m.cfg.sliding_window = saved_window;
    ok &= rel_gate(
        "final_win (window mask)",
        &e.dtoh(&fin_win)?,
        &read_f32(&format!("{cache}/dflash2-final_win.f32")),
        2e-3,
    );

    // ---- stage 4: conv module isolation (layer-0 attention_conv) ----
    let conv_x = read_f32(&format!("{cache}/dflash2-conv_x.f32"));
    let conv_y = read_f32(&format!("{cache}/dflash2-conv_y.f32"));
    let cx_d = e.htod(&conv_x)?;
    let cy_d = e.htod(&conv_y)?;
    let (prep, dyn_) = {
        let d2 = m.dflash2.as_ref().unwrap();
        m.d2_conv_prepare(&e, &d2.attn_conv[0], &cx_d, b)?
    };
    ok &= rel_gate(
        "conv prepare",
        &e.dtoh(&prep)?,
        &read_f32(&format!("{cache}/dflash2-conv_prepare_out.f32")),
        2e-3,
    );
    let finod = {
        let d2 = m.dflash2.as_ref().unwrap();
        m.d2_conv_finish(&e, &d2.attn_conv[0], &cy_d, &dyn_, b)?
    };
    ok &= rel_gate(
        "conv finish",
        &e.dtoh(&finod)?,
        &read_f32(&format!("{cache}/dflash2-conv_finish_out.f32")),
        2e-3,
    );

    // ---- stage 5: selector — device top-k + host codebook walk ----
    let sel_logits = read_f32(&format!("{cache}/dflash2-sel_logits.f32"));
    let sel_hidden = read_f32(&format!("{cache}/dflash2-sel_hidden.f32"));
    if let Err(msg) = expect_len(
        "dflash2-sel_logits.f32",
        sel_logits.len(),
        nd * d2_vocab,
        "ndr*vocab",
    ) {
        eprintln!("{msg}");
        eprintln!("== dflash2_parity: REFUSED (dump length vs config) ==");
        std::process::exit(1);
    }
    let anchor = read_u32(&format!("{cache}/dflash2-anchor.u32"))[0];
    let dl_d = e.htod(&sel_logits)?;
    let (vals_d, idx_d) = e.topk_rows(&dl_d, nd, d2_vocab, d2_topk)?;
    let got_vals = e.dtoh(&vals_d)?;
    let got_idx = e.dtoh_u32(&idx_d)?;
    // reference top-k (sorted=False order) -> sort per row desc by value, idx asc on
    // ties, to match the kernel's canonical output order.
    let ref_unary = read_f32(&format!("{cache}/dflash2-sel_unary.f32"));
    let ref_cand = read_u32(&format!("{cache}/dflash2-sel_candidates.u32"));
    let mut topk_idx_ok = true;
    let mut ref_vals_sorted = vec![0f32; nd * d2_topk];
    let mut ref_idx_sorted = vec![0u32; nd * d2_topk];
    for p in 0..nd {
        let mut pairs: Vec<(f32, u32)> = (0..d2_topk)
            .map(|k| (ref_unary[p * d2_topk + k], ref_cand[p * d2_topk + k]))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        for k in 0..d2_topk {
            ref_vals_sorted[p * d2_topk + k] = pairs[k].0;
            ref_idx_sorted[p * d2_topk + k] = pairs[k].1;
            topk_idx_ok &= got_idx[p * d2_topk + k] == pairs[k].1;
        }
    }
    println!(
        "selector topk indices: {}",
        if topk_idx_ok { "PASS (EXACT)" } else { "FAIL" }
    );
    ok &= topk_idx_ok;
    ok &= rel_gate("selector topk values", &got_vals, &ref_vals_sorted, 2e-3);
    // the walk: memra's own topk output + the real codebooks + a device hproj GEMM
    let hid_d = e.htod(&sel_hidden)?;
    let hproj_d = {
        let d2 = m.dflash2.as_ref().unwrap();
        e.matmul(&d2.hidden_proj, &hid_d, nd)?
    };
    let hproj = e.dtoh(&hproj_d)?;
    let path = {
        let d2 = m.dflash2.as_ref().unwrap();
        d2.walk_greedy(&got_vals, &got_idx, &hproj, anchor, nd)
    };
    let want_path = read_u32(&format!("{cache}/dflash2-sel_path.u32"));
    let path_pass = path[..] == want_path[..];
    println!(
        "selector path: got {:?} want {:?} -> {}",
        path,
        &want_path[..],
        if path_pass { "PASS (EXACT)" } else { "FAIL" }
    );
    ok &= path_pass;

    println!(
        "== dflash2_parity: {} ==",
        if ok { "ALL PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
