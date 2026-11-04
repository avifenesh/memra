//! `MEMRA_TCOL_TWINS` kernel-seam bit-gate (the compile-time-T tcol twins, restored 2026-11-04).
//!
//! With the door armed for this whole process, `matvec_bf16_qkvg_tcol_into` and
//! `matvec_bf16_b4_tcol_into` launch the compile-time-T twins at t = 2, 4 and 8 and the
//! runtime-T kernels at every other width. Every column at every width 1..=8 must be
//! bit-identical to the t=1 program run on that column alone (`matvec_bf16_qkvg_into`,
//! `matvec_bf16_b4_into`), and the engagement counter must advance once per twin launch and
//! never at the other widths. Red arm: a one-column activation shift must bite.
//!
//! The door is read once per process, so this binary arms it before its first CUDA call and
//! holds one test. It proves the kernel seam only. The serving-shape gate the door owes before
//! a flip (batch width crossing 2, 4 and 8 mid-request) is named in its docs/FLAGS.md row.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test tcol_twins_gpu -- --ignored --test-threads=1

use memra_engine::Engine;

/// Deterministic non-trivial f32s (varied signs and magnitudes).
fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed)
                .rotate_left(17) as f64
                / u64::MAX as f64;
            spread * (x as f32 - 0.5)
        })
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn expected_twin_launches(t: usize) -> u64 {
    u64::from(matches!(t, 2 | 4 | 8))
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_tcol_twins_match_the_t1_program_per_column_bitwise() {
    // SAFETY: the only test in this binary, before any CUDA call or thread that reads the env.
    unsafe {
        std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
        std::env::set_var("MEMRA_TCOL_TWINS", "1");
    }
    let e = Engine::new(0).expect("CUDA engine on device 0");

    // Fused QKV + gate: four weights, outputs land yq[c * out_q + row] etc.
    let (in_f, out_q, out_kv, out_g) = (2048usize, 48usize, 16usize, 48usize);
    let wq_h = varied(out_q * in_f, 0x0A11, 2.0);
    let wk_h = varied(out_kv * in_f, 0x0B22, 2.0);
    let wv_h = varied(out_kv * in_f, 0x0C33, 2.0);
    let wg_h = varied(out_g * in_f, 0x0D44, 2.0);
    let wq = e.htod_bytes(&bf16_bytes(&wq_h)).expect("wq");
    let wk = e.htod_bytes(&bf16_bytes(&wk_h)).expect("wk");
    let wv = e.htod_bytes(&bf16_bytes(&wv_h)).expect("wv");
    let wg = e.htod_bytes(&bf16_bytes(&wg_h)).expect("wg");

    // Deferred o_proj: four canonical blocks, each `block_cols` wide; y_t[c * out_f + row].
    let (block_cols, out_f) = (512usize, 96usize);
    let wo_h: Vec<Vec<f32>> = (0..4)
        .map(|b| varied(out_f * block_cols, 0x0E55 + b as u64, 2.0))
        .collect();
    let wo: Vec<_> = wo_h
        .iter()
        .map(|w| e.htod_bytes(&bf16_bytes(w)).expect("wo block"))
        .collect();
    let wo4 = [&wo[0], &wo[1], &wo[2], &wo[3]];

    for t in 1..=8usize {
        // ---- qkvg ----
        let x_h = varied(t * in_f, 0x7C01 + t as u64, 2.0);
        let x = e.htod(&x_h).expect("x");
        let mut yq = e.uninit(t * out_q).expect("yq");
        let mut yk = e.uninit(t * out_kv).expect("yk");
        let mut yv = e.uninit(t * out_kv).expect("yv");
        let mut yg = e.uninit(t * out_g).expect("yg");
        let d0 = memra_engine::tcol_twin_dispatches();
        e.matvec_bf16_qkvg_tcol_into(
            &wq, &wk, &wv, &wg, &x, &mut yq, &mut yk, &mut yv, &mut yg, in_f, out_q, out_kv, out_g,
            t,
        )
        .expect("qkvg tcol launch");
        assert_eq!(
            memra_engine::tcol_twin_dispatches() - d0,
            expected_twin_launches(t),
            "qkvg t={t}: twin engagement"
        );
        let got = [
            e.dtoh(&yq).expect("yq back"),
            e.dtoh(&yk).expect("yk back"),
            e.dtoh(&yv).expect("yv back"),
            e.dtoh(&yg).expect("yg back"),
        ];
        let widths = [out_q, out_kv, out_kv, out_g];
        for c in 0..t {
            let xc = e.htod(&x_h[c * in_f..(c + 1) * in_f]).expect("column");
            let mut rq = e.uninit(out_q).expect("rq");
            let mut rk = e.uninit(out_kv).expect("rk");
            let mut rv = e.uninit(out_kv).expect("rv");
            let mut rg = e.uninit(out_g).expect("rg");
            e.matvec_bf16_qkvg_into(
                &wq, &wk, &wv, &wg, &xc, &mut rq, &mut rk, &mut rv, &mut rg, in_f, out_q, out_kv,
                out_g,
            )
            .expect("t=1 qkvg launch");
            let want = [
                e.dtoh(&rq).expect("rq back"),
                e.dtoh(&rk).expect("rk back"),
                e.dtoh(&rv).expect("rv back"),
                e.dtoh(&rg).expect("rg back"),
            ];
            for (part, ((g, w), n)) in got.iter().zip(&want).zip(widths).enumerate() {
                let diffs = bit_diffs(&g[c * n..(c + 1) * n], w);
                assert_eq!(diffs, 0, "qkvg t={t} column {c} output {part}: {diffs}/{n}");
            }
        }

        // ---- b4 o_proj ----
        let xo_h = varied(t * 4 * block_cols, 0x5E01 + t as u64, 2.0);
        let xo = e.htod(&xo_h).expect("xo");
        let mut yo = e.uninit(t * out_f).expect("yo");
        let d0 = memra_engine::tcol_twin_dispatches();
        e.matvec_bf16_b4_tcol_into(wo4, &xo, &mut yo, block_cols, out_f, t)
            .expect("b4 tcol launch");
        assert_eq!(
            memra_engine::tcol_twin_dispatches() - d0,
            expected_twin_launches(t),
            "b4 t={t}: twin engagement"
        );
        let got = e.dtoh(&yo).expect("yo back");
        for c in 0..t {
            let span = 4 * block_cols;
            let xc = e.htod(&xo_h[c * span..(c + 1) * span]).expect("column");
            let mut r = e.uninit(out_f).expect("r");
            e.matvec_bf16_b4_into(wo4, &xc, &mut r, block_cols, out_f)
                .expect("t=1 b4 launch");
            let want = e.dtoh(&r).expect("r back");
            let diffs = bit_diffs(&got[c * out_f..(c + 1) * out_f], &want);
            assert_eq!(diffs, 0, "b4 t={t} column {c}: {diffs}/{out_f}");
        }
        println!(
            "t={t} PASS ({}): qkvg and b4 bit-identical per column",
            if expected_twin_launches(t) == 1 {
                "twin"
            } else {
                "runtime-T"
            }
        );
    }

    // RED ARM: shifting the activation by one column must bite on a twin width, so the
    // compare above can see a column-addressing bug in the unrolled accumulators.
    let t = 4usize;
    let xo_h = varied((t + 1) * 4 * block_cols, 0x0DD5, 2.0);
    let x_ok = e.htod(&xo_h[..t * 4 * block_cols]).expect("x ok");
    let x_shift = e.htod(&xo_h[4 * block_cols..]).expect("x shifted");
    let mut y_ok = e.uninit(t * out_f).expect("y ok");
    let mut y_shift = e.uninit(t * out_f).expect("y shifted");
    e.matvec_bf16_b4_tcol_into(wo4, &x_ok, &mut y_ok, block_cols, out_f, t)
        .expect("ok launch");
    e.matvec_bf16_b4_tcol_into(wo4, &x_shift, &mut y_shift, block_cols, out_f, t)
        .expect("shifted launch");
    let diffs = bit_diffs(
        &e.dtoh(&y_ok).expect("ok back"),
        &e.dtoh(&y_shift).expect("shifted back"),
    );
    assert!(
        diffs > 0,
        "shifted-column red arm produced identical outputs: the gate cannot see column addressing"
    );
    println!("RED bites: {diffs} outputs differ with a one-column activation shift");
}
