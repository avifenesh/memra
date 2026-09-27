//! WP-B day 44 (`research/spill-b-20260919/DAY44.md` 1.5): the in-call grid capture.
//!
//! GPU-gated (`#[ignore]`, CI is compile-only). Run on the rig under the card's lock:
//!   flock /tmp/memra-5090.lock cargo test -p memra-engine --test grid_capture_gpu -- --ignored --test-threads=1
//! The model-level tests read `MEMRA_TEST_QWEN_GGUF` (a Qwen3.5/3.8 hybrid GGUF, e.g. the 9B).

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::grid_capture;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn pr(i: usize, seed: u64) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    ((x >> 40) as f32 / 16_777_216.0) - 0.5
}

/// The chunked scan's capture at `rows` equals the final state of the same scan over the first
/// `rows` rows alone, bitwise, on the mma pair and on the f32 pair.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn gpu_scan_capture_equals_the_prefix_scan_state() {
    let e = Engine::new(0).expect("CUDA device 0");
    let (s_v, h, t, c) = (128usize, 4usize, 160usize, 32usize);
    let scale = 1.0 / (s_v as f32).sqrt();
    for mma in ["1", "0"] {
        // SAFETY: single-threaded test body; the scan reads this env per call.
        unsafe { std::env::set_var("MEMRA_GDN_MMA", mma) };
        for rows in [32usize, 96, 128] {
            let q: Vec<f32> = (0..s_v * h * t).map(|i| pr(i, 1) * 0.1).collect();
            let k: Vec<f32> = (0..s_v * h * t).map(|i| pr(i, 2) * 0.1).collect();
            let v: Vec<f32> = (0..s_v * h * t).map(|i| pr(i, 3) * 0.1).collect();
            let g: Vec<f32> = (0..h * t).map(|i| -0.05 - pr(i, 4).abs() * 0.1).collect();
            let beta: Vec<f32> = (0..h * t).map(|i| 0.5 + pr(i, 5) * 0.2).collect();
            let st0: Vec<f32> = (0..s_v * s_v * h).map(|i| pr(i, 6) * 0.01).collect();
            let up = |x: &[f32]| e.htod(x).unwrap();
            let (qd, kd, vd, gd, bd, sid) = (up(&q), up(&k), up(&v), up(&g), up(&beta), up(&st0));
            let mut sod = e.zeros(s_v * s_v * h).unwrap();
            let mut od = e.zeros(s_v * h * t).unwrap();
            let mut slot = None;
            e.gdn_scan_chunked_capture(
                &qd,
                &kd,
                &vd,
                &gd,
                &bd,
                None,
                None,
                &sid,
                &mut sod,
                &mut od,
                h,
                t,
                scale,
                c,
                h,
                Some((rows, &mut slot)),
            )
            .unwrap();
            let captured = e.dtoh(slot.as_ref().expect("capture taken")).unwrap();
            // The same scan over the first `rows` rows (token-major inputs: a prefix slice).
            let (qp, kp, vp) = (
                up(&q[..s_v * h * rows]),
                up(&k[..s_v * h * rows]),
                up(&v[..s_v * h * rows]),
            );
            let (gp, bp) = (up(&g[..h * rows]), up(&beta[..h * rows]));
            let mut sp = e.zeros(s_v * s_v * h).unwrap();
            let mut op = e.zeros(s_v * h * rows).unwrap();
            e.gdn_scan_chunked(
                &qp, &kp, &vp, &gp, &bp, None, None, &sid, &mut sp, &mut op, h, rows, scale, c, h,
            )
            .unwrap();
            let split = e.dtoh(&sp).unwrap();
            let differ = captured
                .iter()
                .zip(&split)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differ, 0,
                "mma={mma} rows={rows}: {differ} state values differ"
            );
            // The full scan's own output is unchanged by the capture.
            let mut sod2 = e.zeros(s_v * s_v * h).unwrap();
            let mut od2 = e.zeros(s_v * h * t).unwrap();
            e.gdn_scan_chunked(
                &qd, &kd, &vd, &gd, &bd, None, None, &sid, &mut sod2, &mut od2, h, t, scale, c, h,
            )
            .unwrap();
            assert_eq!(
                e.dtoh(&sod).unwrap(),
                e.dtoh(&sod2).unwrap(),
                "mma={mma}: final state"
            );
            assert_eq!(
                e.dtoh(&od).unwrap(),
                e.dtoh(&od2).unwrap(),
                "mma={mma}: output"
            );
        }
    }
    unsafe { std::env::remove_var("MEMRA_GDN_MMA") };
}

/// The ring capture is rows `rows - 3 .. rows` of the token-major conv input, in the ring layout.
#[test]
#[ignore = "needs a CUDA device; run under flock /tmp/memra-5090.lock"]
fn gpu_ring_capture_is_the_rows_before_the_capture_point() {
    let e = Engine::new(0).expect("CUDA device 0");
    let (conv_dim, t, d_conv, rows) = (64usize, 96usize, 4usize, 64usize);
    let x: Vec<f32> = (0..t * conv_dim).map(|i| pr(i, 7)).collect();
    let xd = e.htod(&x).unwrap();
    let ring = e
        .ssm_conv_ring_capture(&xd.slice(0..t * conv_dim), conv_dim, rows, d_conv)
        .unwrap();
    let got = e.dtoh(&ring).unwrap();
    let pad = d_conv - 1;
    for c in 0..conv_dim {
        for j in 0..pad {
            let tt = rows - pad + j;
            assert_eq!(got[c * pad + j].to_bits(), x[tt * conv_dim + c].to_bits());
        }
    }
}

fn model() -> Option<(Engine, HybridModel)> {
    let path = std::env::var("MEMRA_TEST_QWEN_GGUF").ok()?;
    let e = Engine::new(0).expect("CUDA device 0");
    let g = GgufFile::open(&path).expect("gguf");
    let m = HybridModel::load_without_mtp(&e, &g).expect("hybrid model");
    Some((e, m))
}

fn prompt(n: usize, seed: u64) -> Vec<u32> {
    (0..n)
        .map(|i| 1000 + (pr(i, seed).abs() * 20_000.0) as u32)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Model level: a capture inside one prime call equals the split prime's state at the capture
/// point, and a resume from it primes the next prompt to the cold prime's logits bitwise.
#[test]
#[ignore = "needs a CUDA device and MEMRA_TEST_QWEN_GGUF; run under flock /tmp/memra-5090.lock"]
fn gpu_model_capture_equals_split_and_resumes_cold_exact() {
    let Some((e, m)) = model() else {
        eprintln!("MEMRA_TEST_QWEN_GGUF unset; skipped");
        return;
    };
    let ctx = 4096;
    let p1 = prompt(700, 11);
    let mut p2 = p1.clone();
    p2.extend(prompt(96, 13));
    let g = grid_capture::capture_point(0, p1.len(), 16, Engine::gdn_chunk_size()).unwrap();
    assert_eq!(g, 672);

    // A: one call over p1 carrying the capture.
    let mut ca = Cache::new(&e, &m.cfg, ctx).unwrap();
    let guard = grid_capture::arm(g, grid_capture::needed_layers(&ca));
    m.prime_cache(&e, &p1, &mut ca, 0).unwrap();
    let cap = grid_capture::take().expect("capture taken inside the call");
    drop(guard);
    let snap_a = cap.into_snapshot(&ca).unwrap();

    // B: the split prime, stopped at g.
    let mut cb = Cache::new(&e, &m.cfg, ctx).unwrap();
    m.prime_cache(&e, &p1[..g], &mut cb, p1.len() - g).unwrap();
    let snap_b = cb.snapshot(&e).unwrap();
    for il in 0..snap_a.ssm.len() {
        match (&snap_a.ssm[il], &snap_b.ssm[il]) {
            (Some(a), Some(b)) => {
                assert_eq!(
                    bits(&e.dtoh(a).unwrap()),
                    bits(&e.dtoh(b).unwrap()),
                    "ssm layer {il}"
                );
                let (ca_, cb_) = (
                    snap_a.conv[il].as_ref().unwrap(),
                    snap_b.conv[il].as_ref().unwrap(),
                );
                assert_eq!(
                    bits(&e.dtoh(ca_).unwrap()),
                    bits(&e.dtoh(cb_).unwrap()),
                    "conv layer {il}"
                );
            }
            (None, None) => {}
            _ => panic!("layer {il}: plane presence differs"),
        }
    }

    // C: restore the capture into A's cache and prime p2's tail in one call.
    memra_engine::pp::restore_cache_checkpoint(&e, &m, None, &mut ca, &snap_a).unwrap();
    assert_eq!(ca.pos, g);
    let (resumed, _, _) = m.prime_cache(&e, &p2[g..], &mut ca, 0).unwrap();

    // D: a cold prime of p2.
    let mut cd = Cache::new(&e, &m.cfg, ctx).unwrap();
    let (cold, _, _) = m.prime_cache(&e, &p2, &mut cd, 0).unwrap();
    assert_eq!(
        bits(&resumed),
        bits(&cold),
        "resume from the capture is cold-exact"
    );
}

/// A capture point on a call boundary is the live state (no in-call capture needed).
#[test]
#[ignore = "needs a CUDA device and MEMRA_TEST_QWEN_GGUF; run under flock /tmp/memra-5090.lock"]
fn gpu_model_boundary_capture_is_the_live_state() {
    let Some((e, m)) = model() else {
        eprintln!("MEMRA_TEST_QWEN_GGUF unset; skipped");
        return;
    };
    let p = prompt(640, 17);
    let mut ca = Cache::new(&e, &m.cfg, 4096).unwrap();
    let guard = grid_capture::arm(640, grid_capture::needed_layers(&ca));
    m.prime_cache(&e, &p, &mut ca, 0).unwrap();
    let cap = grid_capture::take().expect("boundary capture at the call end");
    drop(guard);
    let snap = cap.into_snapshot(&ca).unwrap();
    let live = ca.snapshot(&e).unwrap();
    for il in 0..snap.ssm.len() {
        if let (Some(a), Some(b)) = (&snap.ssm[il], &live.ssm[il]) {
            assert_eq!(
                bits(&e.dtoh(a).unwrap()),
                bits(&e.dtoh(b).unwrap()),
                "ssm layer {il}"
            );
        }
    }
    assert!(!grid_capture::armed(), "the guard disarmed");
}

/// The settle (DAY44 1.2 (c)): from the capture, a prime of the committed rows up to the last
/// grid point with 16 rows after it (its own call, the request end unknown), then the next
/// prompt's tail primes from there to the cold prime's logits bitwise.
#[test]
#[ignore = "needs a CUDA device and MEMRA_TEST_QWEN_GGUF; run under flock /tmp/memra-5090.lock"]
fn gpu_model_settle_then_resume_is_cold_exact() {
    let Some((e, m)) = model() else {
        eprintln!("MEMRA_TEST_QWEN_GGUF unset; skipped");
        return;
    };
    let ctx = 4096;
    let p1 = prompt(700, 21);
    let mut committed = p1.clone();
    committed.extend(prompt(40, 23)); // the reply's tokens
    let mut p2 = committed.clone();
    p2.extend(prompt(64, 29));
    let g = 672;
    let s = 704;
    let mut c = Cache::new(&e, &m.cfg, ctx).unwrap();
    let guard = grid_capture::arm(g, grid_capture::needed_layers(&c));
    m.prime_cache(&e, &p1, &mut c, 0).unwrap();
    let snap = grid_capture::take().unwrap().into_snapshot(&c).unwrap();
    drop(guard);
    // (the decoded reply rows would sit here; the settle discards them)
    memra_engine::pp::restore_cache_checkpoint(&e, &m, None, &mut c, &snap).unwrap();
    m.prime_cache(&e, &committed[g..s], &mut c, 0).unwrap();
    assert_eq!(c.pos, s);
    let (resumed, _, _) = m.prime_cache(&e, &p2[s..], &mut c, 0).unwrap();
    let mut cold_c = Cache::new(&e, &m.cfg, ctx).unwrap();
    let (cold, _, _) = m.prime_cache(&e, &p2, &mut cold_c, 0).unwrap();
    assert_eq!(
        bits(&resumed),
        bits(&cold),
        "settle then resume is cold-exact"
    );
}
