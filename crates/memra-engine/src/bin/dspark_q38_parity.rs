//! DSpark Q38 drafter parity gate (lane/dspark-q38-recover): the memra loader +
//! forward + markov chain + confidence head on the arm-a HF export must reproduce
//! the SpecForge reference (tools/dspark_q38_oracle.py flat dumps) on fixed-seed
//! synthetic inputs.
//!
//! Stages and bars (calibration inherited from dflash_parity, 2026-07-13: the f32
//! GEMM rides TF32-class compute — rel-to-max 2e-3; integer decisions EXACT):
//!   ctx_features  rel < 2e-3
//!   final         rel < 2e-3
//!   markov tokens EXACT (nd/nd per the dump's harvest convention) — chained greedy
//!   markov logits rel < 2e-3 (bf16 w2 under MEMRA_DFLASH_PREC=bf16)
//!   confidence    rel < 2e-3 (host dot, reference-final inputs isolate the head)
//!
//! HARVEST CONVENTION (DSPARK-POSTMORTEM-20260820.md). The markov/confidence row count
//! and the row -> position mapping are properties of the checkpoint's TRAINING STRATEGY
//! (dflash mask-fill: b-1 rows from row 1; dspark shifted: b rows from row 0 — SpecForge
//! dflash_family_model.py:816). The oracle writes its convention into the geometry
//! manifest (`harvest=`); this gate derives the row set from it, so the reference is no
//! longer row-count-agnostic — the blindness that let the q38 misalignment ship. A dump
//! without the field is a pre-postmortem mask-fill dump and is REFUSED for dspark-class
//! exports (markov head present): regenerate with tools/dspark_q38_oracle.py.
//!
//! Run with MEMRA_DFLASH_PREC=bf16 (parity precision class).
//!
//! GEOMETRY FIRST (GATE-INTEGRITY-20260819 §5, fixed 2026-08-19). Stage 1 used to read its
//! context length out of the reference bytes — `let ctx = taps.len() / (n_taps * h)` — with no
//! remainder check and no assertion that the dump came from the same config as the export under
//! test. Every structural check in this file was a PRODUCT (`noise.len() == b*h`,
//! `base.len() == (b-1)*v`, `got.len() == want.len()`), and a product cannot distinguish a
//! refactorisation that multiplies out the same. The manifest the oracle now writes is asserted
//! against the loaded config first; see crates/memra-engine/src/parity_geometry.rs.
use memra_engine::Engine;
use memra_engine::dflash::{DflashDraft, DsparkHarvest};

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
        .expect("usage: dspark_q38_parity <export_dir> <dump_dir>");
    let cache = std::env::args().nth(2).expect("dump dir");
    if std::env::var("MEMRA_DFLASH_PREC").as_deref() != Ok("bf16") {
        panic!("parity gate requires MEMRA_DFLASH_PREC=bf16");
    }
    let e = Engine::new(0)?;
    let m = DflashDraft::load(&e, std::path::Path::new(&ckpt))?;
    let c = &m.cfg;
    println!(
        "loaded dspark draft: {} layers, hidden {}, block {}, taps {:?}, markov {}, confidence {}",
        c.n_layer,
        c.hidden,
        c.block_size,
        c.target_layer_ids,
        m.markov.is_some(),
        m.confidence.is_some()
    );
    let mk = m
        .markov
        .as_ref()
        .expect("arm-a export must carry the markov head");
    let ch = m
        .confidence
        .as_ref()
        .expect("arm-a export must carry the confidence head");
    let b = c.block_size;
    let h = c.hidden;
    let n_taps = c.target_layer_ids.len();
    let v = mk.vocab;
    let mut ok = true;

    // ---- stage 0: GEOMETRY GATE — config is the authority, the dump is the claimant ----
    //
    // Asserted before any value compare, and it includes the fields that never touch a byte
    // (`head_dim`, `rope_theta`, `sliding_window`, `layer_sliding`): a dump produced under a
    // different rotary width compares byte-identically and is still a different model.
    let manifest = manifest_path(&cache, "dspark");
    let regen = format!(
        "MEMRA_DFLASH_PREC=bf16 python tools/dspark_q38_oracle.py {ckpt} \
         {cache}/dspark-oracle.npz   # writes {manifest}"
    );
    let geo = Geometry::load(&manifest, &regen).map_err(|e| -> Box<dyn std::error::Error> {
        eprintln!("{e}");
        "dspark_q38_parity: REFUSED (geometry manifest)".into()
    })?;
    let geo_checks: Vec<Result<(), String>> = vec![
        geo.expect_str("dtype", "f32"),
        geo.expect_usize("hidden", h),
        geo.expect_usize("n_layer", c.n_layer),
        geo.expect_usize("block_size", b),
        geo.expect_usize("n_taps", n_taps),
        geo.expect_str("target_layer_ids", &join_usize(&c.target_layer_ids)),
        geo.expect_usize("head_dim", c.head_dim),
        geo.expect_usize("n_head", c.n_head),
        geo.expect_usize("n_head_kv", c.n_kv),
        geo.expect_f64_near("rope_theta", c.rope_theta as f64, 1e-6),
        geo.expect_usize_if_present("sliding_window", c.sliding_window)
            .map(|_| ()),
        // layer_sliding + sliding_window are the pure-config class; asserted when the
        // producer recorded them, and their ABSENCE is reported below rather than read as
        // agreement (an export may nest or omit `layer_types`).
        geo.expect_str_if_present("layer_sliding", &join_bool(&c.layer_sliding))
            .map(|_| ()),
        // The markov head's vocab is the other geometry the base_logits dump silently encodes:
        // `base.len() == (b-1) * v` is a product, so a wrong v with a compensating b compares.
        geo.expect_usize("markov_vocab", v),
        geo.expect_usize_if_present("markov_rank", mk.rank)
            .map(|_| ()),
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
            "== dspark_q38_parity: REFUSED — {geo_fails} geometry field(s) disagree with the \
             export. A value compare across two different programs is not a parity result. =="
        );
        std::process::exit(1);
    }
    // ctx comes from the MANIFEST, never from a byte count.
    let ctx = geo.need_usize("ctx")?;
    // Harvest convention comes from the MANIFEST too (the oracle records what it
    // computed with). Refuse-on-ambiguity: a dspark-class export gated against a
    // convention-less (pre-postmortem, mask-fill) dump is exactly the blindness that
    // shipped the q38 misalignment.
    let harvest = match geo.get("harvest") {
        Some(name) => DsparkHarvest::from_name(name).unwrap_or_else(|| {
            panic!(
                "geometry manifest harvest={name}: unknown convention (dflash|dspark) — \
                 refusing (DSPARK-POSTMORTEM-20260820.md)"
            )
        }),
        None => {
            eprintln!(
                "== dspark_q38_parity: REFUSED — the geometry manifest has no `harvest` \
                 field. Pre-postmortem dumps encode the DFlash mask-fill row set, which \
                 is the WRONG reference for a dspark-strategy export \
                 (DSPARK-POSTMORTEM-20260820.md); regenerate: {regen} =="
            );
            std::process::exit(1);
        }
    };
    let nd = harvest.n_drafts(b);
    let row0 = harvest.first_row();
    println!(
        "geometry OK ({} fields vs {manifest}): ctx={ctx} hidden={h} n_taps={n_taps} block={b} \
         head_dim={} rope_theta={} vocab={v} rank={} harvest={} (nd={nd}, first_row={row0})",
        geo_checks.len(),
        c.head_dim,
        c.rope_theta,
        mk.rank,
        harvest.name()
    );

    // ---- stage 1: ctx features ----
    let taps = read_f32(&format!("{cache}/dspark-taps.f32"));
    let ref_ctxf = read_f32(&format!("{cache}/dspark-ctx_features.f32"));
    // Lengths ASSERTED against config-predicted products, not divided into a dimension. The
    // ctx_features dump over-determines ctx, so a manifest that lies about it is caught too.
    for r in [
        expect_len(
            "dspark-taps.f32",
            taps.len(),
            ctx * n_taps * h,
            "ctx*n_taps*hidden",
        ),
        expect_len(
            "dspark-ctx_features.f32",
            ref_ctxf.len(),
            ctx * h,
            "ctx*hidden",
        ),
    ] {
        if let Err(msg) = r {
            eprintln!("{msg}");
            eprintln!("== dspark_q38_parity: REFUSED (dump length vs config) ==");
            std::process::exit(1);
        }
    }
    let taps_d = e.htod(&taps)?;
    let ctxf = m.ctx_features(&e, &taps_d, ctx)?;
    ok &= rel_gate("ctx_features", &e.dtoh(&ctxf)?, &ref_ctxf, 2e-3);

    // ---- stages 2-3: block forward, final hidden ----
    let noise = read_f32(&format!("{cache}/dspark-noise.f32"));
    if let Err(msg) = expect_len("dspark-noise.f32", noise.len(), b * h, "block_size*hidden") {
        eprintln!("{msg}");
        eprintln!("== dspark_q38_parity: REFUSED (dump length vs config) ==");
        std::process::exit(1);
    }
    let noise_d = e.htod(&noise)?;
    let pos: Vec<i32> = (0..(ctx + b) as i32).collect();
    let fin = m.forward(&e, &taps_d, &noise_d, &pos, ctx)?;
    ok &= rel_gate(
        "final",
        &e.dtoh(&fin)?,
        &read_f32(&format!("{cache}/dspark-final.f32")),
        2e-3,
    );

    // ---- stage 4: markov chained greedy over the synthetic base logits ----
    // Mirrors generate_spec_dflash's device chain: chain_d[0] = anchor; per step k the
    // bias row gathers from chain_d[k], adds onto logits row k, argmax writes k+1.
    let base = read_f32(&format!("{cache}/dspark-base_logits.f32"));
    if let Err(msg) = expect_len(
        "dspark-base_logits.f32",
        base.len(),
        nd * v,
        "n_drafts(harvest)*markov_vocab",
    ) {
        eprintln!("{msg}");
        eprintln!("== dspark_q38_parity: REFUSED (dump length vs config) ==");
        std::process::exit(1);
    }
    let anchor = read_u32(&format!("{cache}/dspark-anchor.u32"))[0];
    let mut dl = e.htod(&base)?;
    let mut chain_d = e.stream().alloc_zeros::<u32>(nd + 1)?;
    e.set_u32_one(&mut chain_d, anchor)?;
    for k in 0..nd {
        let mut f = e.uninit(mk.rank)?;
        e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
        let bias = e.matmul(&mk.w2, &f, 1)?;
        e.add_row_inplace(&mut dl, &bias, v, k * v)?;
        e.argmax_token_device_col(&dl, k, v, &mut chain_d, k + 1)?;
    }
    let chain = e.dtoh_u32(&chain_d)?;
    let want_tok = read_u32(&format!("{cache}/dspark-markov_tokens.u32"));
    let tok_pass = chain[1..] == want_tok[..];
    println!(
        "markov tokens: got {:?} want {:?} -> {}",
        &chain[1..],
        &want_tok[..],
        if tok_pass { "PASS (EXACT)" } else { "FAIL" }
    );
    ok &= tok_pass;
    ok &= rel_gate(
        "markov logits",
        &e.dtoh(&dl)?,
        &read_f32(&format!("{cache}/dspark-markov_logits.f32")),
        2e-3,
    );

    // ---- stage 5: confidence head (host dot; reference-final rows isolate the head
    // from forward drift; prev ids = [anchor, chain[..-1]] — the gate contract the
    // oracle uses; the reference SERVING loop never consumes this head) ----
    let ref_final = read_f32(&format!("{cache}/dspark-final.f32"));
    let want_conf = read_f32(&format!("{cache}/dspark-confidence.f32"));
    assert!(ch.with_markov, "arm-a confidence head is with_markov");
    // w1 rows on host for the tiny gate dot
    let mut got_conf = Vec::with_capacity(nd);
    let w1_all = {
        // gather via device (same primitive the chain uses), one row per prev id
        let mut rows = Vec::new();
        let mut prev_ids: Vec<u32> = vec![anchor];
        prev_ids.extend_from_slice(&want_tok[..nd - 1]);
        let mut id_d = e.stream().alloc_zeros::<u32>(1)?;
        for &id in &prev_ids {
            e.set_u32_one(&mut id_d, id)?;
            let mut f = e.uninit(mk.rank)?;
            e.gather_row_bf16(&mk.w1_bf16, &id_d, 0, &mut f, mk.rank)?;
            rows.push(e.dtoh(&f)?);
        }
        rows
    };
    for i in 0..nd {
        // hidden row = the drafter output row draft i+1 is harvested from
        // (dflash: rows 1..b-1; dspark: rows 0..b-1 — the shifted convention).
        let hrow = &ref_final[(row0 + i) * h..(row0 + i + 1) * h];
        let emb = &w1_all[i];
        let mut acc = ch.b;
        for (j, x) in hrow.iter().enumerate() {
            acc += ch.w[j] * x;
        }
        for (j, x) in emb.iter().enumerate() {
            acc += ch.w[h + j] * x;
        }
        got_conf.push(acc);
    }
    ok &= rel_gate("confidence", &got_conf, &want_conf, 2e-3);

    println!(
        "== dspark_q38_parity: {} ==",
        if ok { "ALL PASS" } else { "FAIL" }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
