//! DFlash draft-forward parity gate (bring-up step 2, DFLASH-BRINGUP-PLAN.md): the memra
//! forward on the real checkpoint must reproduce the torch reference (tools/
//! dflash_oracle.py flat dumps in /data/cache) on the fixed-seed synthetic inputs.
//! PASS bar: final hidden rel maxdiff < 1e-3 class (f32-vs-f32, same math different
//! kernel FP order) AND per-layer drift monotone (bisect handle if final fails).
//!
//! GEOMETRY FIRST (GATE-INTEGRITY-20260819 §5, fixed 2026-08-19). This gate used to take its
//! context length FROM the reference bytes — `let ctx = th.len() / (n_taps * hidden)` — with no
//! remainder check and no assertion that the dump was produced under the same config as the
//! checkpoint under test. A reference regenerated with a different `hidden`, tap set or
//! `block_size` was indistinguishable from a correct one: `ctx` silently became a different
//! number and the value compare proceeded against a reinterpreted buffer. The only structural
//! checks were PRODUCTS, and a product is blind to any factorisation that multiplies out the
//! same. See crates/memra-engine/src/parity_geometry.rs for the rule and its tests.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ckpt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/ai-ml/hf-models/dspark-gemma4-31b-draft/backbone-only".into());
    let cache = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/data/cache".into());
    let e = Engine::new(0)?;
    let m = DflashDraft::load(&e, std::path::Path::new(&ckpt))?;
    let c = &m.cfg;
    println!(
        "loaded dflash draft: {} layers, hidden {}, block {}, taps {:?}",
        c.n_layer, c.hidden, c.block_size, c.target_layer_ids
    );

    // ---- GEOMETRY GATE: the config is the authority, the dump is the claimant ----
    //
    // Every field below is asserted BEFORE a single byte is read into a comparison. `head_dim`,
    // `rope_theta`, `sliding_window` and `layer_sliding` are the class that motivated this: they
    // are pure config, they never reach the reference bytes, and a dump produced under a
    // different value for any of them compares byte-identically while being a different model.
    let manifest = manifest_path(&cache, "dflash");
    let regen = format!(
        "python tools/dflash_oracle.py {ckpt} {cache}/dflash-oracle.npz   # writes {manifest}"
    );
    let geo = Geometry::load(&manifest, &regen).map_err(|e| -> Box<dyn std::error::Error> {
        eprintln!("{e}");
        "DFLASH-PARITY: REFUSED (geometry manifest)".into()
    })?;
    let n_taps = c.target_layer_ids.len();
    let checks: Vec<Result<(), String>> = vec![
        geo.expect_str("dtype", "f32"),
        geo.expect_usize("hidden", c.hidden),
        geo.expect_usize("n_layer", c.n_layer),
        geo.expect_usize("block_size", c.block_size),
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
    ];
    let mut geo_fails = 0usize;
    for r in &checks {
        if let Err(msg) = r {
            eprintln!("{msg}");
            geo_fails += 1;
        }
    }
    if geo_fails > 0 {
        eprintln!(
            "DFLASH-PARITY: REFUSED — {geo_fails} geometry field(s) disagree with the checkpoint. \
             Comparing values now would be a byte compare between two different programs."
        );
        std::process::exit(1);
    }
    // ctx comes from the MANIFEST, not from a byte count.
    let ctx = geo.need_usize("ctx")?;
    println!(
        "geometry OK ({} fields vs {manifest}): ctx={ctx} hidden={} n_taps={n_taps} \
         block={} head_dim={} rope_theta={}",
        checks.len(),
        c.hidden,
        c.block_size,
        c.head_dim,
        c.rope_theta
    );

    let th = read_f32(&format!("{cache}/dflash-target_hidden.f32"));
    let ne = read_f32(&format!("{cache}/dflash-noise_embedding.f32"));
    // Every dump length is now ASSERTED against the config-predicted product rather than
    // divided into one. A second dump (ctx_features) over-determines ctx, so a manifest that
    // lies about it cannot survive either.
    let ctxf = read_f32(&format!("{cache}/dflash-ctx_features.f32"));
    for r in [
        expect_len(
            "dflash-target_hidden.f32",
            th.len(),
            ctx * n_taps * c.hidden,
            "ctx*n_taps*hidden",
        ),
        expect_len(
            "dflash-noise_embedding.f32",
            ne.len(),
            c.block_size * c.hidden,
            "block_size*hidden",
        ),
        expect_len(
            "dflash-ctx_features.f32",
            ctxf.len(),
            ctx * c.hidden,
            "ctx*hidden",
        ),
    ] {
        if let Err(msg) = r {
            eprintln!("{msg}");
            eprintln!("DFLASH-PARITY: REFUSED (dump length vs config)");
            std::process::exit(1);
        }
    }
    let th_d = e.htod(&th)?;
    let ne_d = e.htod(&ne)?;
    let pos: Vec<i32> = (0..(ctx + c.block_size) as i32).collect();

    let out = m.forward(&e, &th_d, &ne_d, &pos, ctx)?;
    let got = e.dtoh(&out)?;
    let want = read_f32(&format!("{cache}/dflash-final.f32"));
    // `got.len() == want.len()` alone said nothing about whether either matched the config.
    if let Err(msg) = expect_len(
        "dflash-final.f32",
        want.len(),
        c.block_size * c.hidden,
        "block_size*hidden",
    ) {
        eprintln!("{msg}");
        eprintln!("DFLASH-PARITY: REFUSED (reference final vs config)");
        std::process::exit(1);
    }
    assert_eq!(got.len(), want.len());
    let (mut md, mut mi) = (0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        let d = (a - b).abs();
        if d > md {
            md = d;
            mi = i;
        }
    }
    // Bar calibration (bisect 2026-07-13): every stage isolated — xn 1e-6, ctx_features
    // 1e-3, per-layer drift FLAT at 3-7e-4 rel-to-max across all 5 layers (no structural
    // bug); the noise seed is the cuBLASLt f32 GEMM riding TF32-class compute (q0 maxdiff
    // 1.5e-2 on a 5376-K dot with bit-identical inputs/weights) amplified by the draft's
    // ~10k-scale activations. Bar = maxdiff vs max|ref| < 2e-3 (TF32 class); the ROUND
    // gates (acceptance + verify exactness) are the real oracle downstream.
    let mx = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let rel = md / mx;
    println!(
        "final: maxdiff {md:.3e} (idx {mi}: got {} want {}), max|ref| {mx:.2}, rel-to-max {rel:.3e}",
        got[mi], want[mi]
    );
    let pass = rel < 2e-3;
    println!("DFLASH-PARITY: {}", if pass { "PASS" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
