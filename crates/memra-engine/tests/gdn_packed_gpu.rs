//! lane/dspark-gdn-packed (2026-09-08): the packed T-row twins of the batched-class linear
//! verify's per-row state kernels must be BIT-IDENTICAL to T chained per-row launches: the conv
//! outputs, the scan outputs, the final recurrent states, and every post-row snapshot the verify
//! checkpoint reads. Run under flock on the lane box; it needs a CUDA device.
use cudarc::driver::DevicePtr;
use memra_engine::Engine;

fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * spread
        })
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
#[ignore = "needs a CUDA device, run under flock on the lane box"]
fn gpu_gdn_packed_matches_per_row_bitwise() {
    packed_matches_per_row(2, 4, false);
}

#[test]
#[ignore = "needs a CUDA device, run under flock on the lane box"]
fn gpu_gdn_packed_ornith_matches_per_row_bitwise() {
    packed_matches_per_row(16, 32, false);
}

#[test]
#[ignore = "needs a CUDA device, run under flock on the lane box"]
fn gpu_gdn_packed_graph_rebind_matches_per_row_bitwise() {
    packed_matches_per_row(2, 4, true);
    packed_matches_per_row(16, 32, true);
}

fn packed_matches_per_row(num_k: usize, num_v: usize, graph_rebind: bool) {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (d_state, d_conv) = (128usize, 4usize);
    let key_dim = d_state * num_k;
    let value_dim = d_state * num_v;
    let conv_dim = key_dim * 2 + value_dim;
    let pad = d_conv - 1;
    let eps = 1e-6f32;
    let scale = 1.0 / (d_state as f32).sqrt();
    let state_words = d_state * d_state * num_v;
    let w = e.htod(&varied(conv_dim * d_conv, 1, 0.5)).expect("w");
    let dt_bias = e.htod(&varied(num_v, 2, 0.5)).expect("dt");
    let a = e.htod(&varied(num_v, 3, 1.0)).expect("a");
    let mut cells = 0usize;
    for (ti, &t) in [2usize, 3, 4, 5, 6, 7, 8].iter().enumerate() {
        let conv0 = varied(conv_dim * pad, 10 + ti as u64, 1.0);
        let ssm0 = varied(state_words, 20 + ti as u64, 0.3);
        let qkv = e
            .htod(&varied(t * conv_dim, 30 + ti as u64, 1.0))
            .expect("qkv");
        let beta_raw = e
            .htod(&varied(t * num_v, 40 + ti as u64, 2.0))
            .expect("beta");
        let alpha = e
            .htod(&varied(t * num_v, 50 + ti as u64, 2.0))
            .expect("alpha");

        // ---- reference: the per-row program, T chained launches, snapshots by clone ----
        let conv_state = e.htod(&conv0).expect("conv0");
        let s0 = e.htod(&ssm0).expect("ssm0");
        let s1 = e.zeros(state_words).expect("alt");
        let mut ref_conv_out = Vec::new();
        let mut ref_o = Vec::new();
        let mut ref_snap_conv = Vec::new();
        let mut ref_snap_ssm = Vec::new();
        {
            let stream = e.stream();
            let (pc, _g0) = conv_state.device_ptr(&stream);
            let (p0, _g1) = s0.device_ptr(&stream);
            let (p1, _g2) = s1.device_ptr(&stream);
            let table = e.htod_u64(&[pc, p0, p1, pc, p1, p0]).expect("table");
            let mut conv_out = e.uninit(conv_dim).expect("co");
            let mut q_l2 = e.uninit(value_dim).expect("q");
            let mut k_l2 = e.uninit(value_dim).expect("k");
            let mut v_g = e.uninit(value_dim).expect("v");
            let mut beta_b = e.uninit(num_v).expect("b");
            let mut g_log = e.uninit(num_v).expect("g");
            for r in 0..t {
                let base = if r % 2 == 0 { 0 } else { 3 };
                let conv_view = table.slice(base..base + 1);
                let in_view = table.slice(base + 1..base + 2);
                let out_view = table.slice(base + 2..base + 3);
                e.ssm_conv1d_fused_decode_b_view(
                    &qkv.slice(r * conv_dim..(r + 1) * conv_dim),
                    &conv_view,
                    &w,
                    &mut conv_out,
                    conv_dim,
                    d_conv,
                    1,
                )
                .expect("conv row");
                e.gdn_prep_decode_b_view(
                    &conv_out,
                    &beta_raw.slice(r * num_v..(r + 1) * num_v),
                    &alpha.slice(r * num_v..(r + 1) * num_v),
                    &dt_bias,
                    &a,
                    &mut q_l2,
                    &mut k_l2,
                    &mut v_g,
                    &mut beta_b,
                    &mut g_log,
                    d_state,
                    num_v,
                    num_k,
                    key_dim,
                    eps,
                    conv_dim,
                    1,
                )
                .expect("prep row");
                let mut o_row = e.uninit(value_dim).expect("o");
                {
                    let mut ov = o_row.slice_mut(0..value_dim);
                    e.gdn_scan_s128_batched_view(
                        &q_l2, &k_l2, &v_g, &g_log, &beta_b, &in_view, &out_view, &mut ov, num_v,
                        1, scale,
                    )
                    .expect("scan row");
                }
                ref_conv_out.extend(e.dtoh(&conv_out).expect("co h"));
                ref_o.extend(e.dtoh(&o_row).expect("o h"));
                if r + 1 < t {
                    ref_snap_conv.extend(e.dtoh(&conv_state).expect("cs h"));
                    let cur = if r % 2 == 0 { &s1 } else { &s0 };
                    ref_snap_ssm.extend(e.dtoh(cur).expect("ss h"));
                }
            }
        }
        let ref_final_ssm = e.dtoh(if t % 2 == 1 { &s1 } else { &s0 }).expect("final");
        let ref_final_conv = e.dtoh(&conv_state).expect("final conv");

        // ---- packed: one launch per kernel over all T rows ----
        let conv_state_p = e.htod(&conv0).expect("conv0 p");
        let s0p = e.htod(&ssm0).expect("ssm0 p");
        let s1p = e.zeros(state_words).expect("alt p");
        let mut conv_outs = e.uninit(t * conv_dim).expect("cos");
        let mut snap_conv = e.zeros((t - 1) * conv_dim * pad).expect("snap c");
        let mut snap_ssm = e.zeros((t - 1) * state_words).expect("snap s");
        let mut q_l2 = e.uninit(t * value_dim).expect("q");
        let mut k_l2 = e.uninit(t * value_dim).expect("k");
        let mut v_g = e.uninit(t * value_dim).expect("v");
        let mut beta_b = e.uninit(t * num_v).expect("b");
        let mut g_log = e.uninit(t * num_v).expect("g");
        let mut o_all = e.uninit(t * value_dim).expect("o all");
        let mut table = e
            .htod_u64(&state_ptrs(&e, &conv_state_p, &s0p, &s1p))
            .expect("packed table");
        let graph = {
            let mut run = |e: &Engine| {
                e.ssm_conv1d_fused_decode_tloop(
                    &qkv,
                    &table.slice(0..3),
                    &w,
                    &mut conv_outs,
                    Some(&mut snap_conv),
                    conv_dim,
                    d_conv,
                    t,
                )
                .expect("conv tloop");
                e.gdn_prep_decode_b(
                    &conv_outs,
                    &beta_raw,
                    &alpha,
                    &dt_bias,
                    &a,
                    &mut q_l2,
                    &mut k_l2,
                    &mut v_g,
                    &mut beta_b,
                    &mut g_log,
                    d_state,
                    num_v,
                    num_k,
                    key_dim,
                    eps,
                    conv_dim,
                    t,
                )
                .expect("prep packed");
                e.gdn_scan_s128_tsnap(
                    &q_l2,
                    &k_l2,
                    &v_g,
                    &g_log,
                    &beta_b,
                    &table.slice(0..3),
                    &mut o_all,
                    num_v,
                    t,
                    scale,
                    Some(&mut snap_ssm),
                )
                .expect("scan tsnap");
                Ok(())
            };
            if graph_rebind {
                Some(e.capture_graph_retained(&mut run).expect("capture packed"))
            } else {
                run(&e).expect("packed eager");
                None
            }
        };
        for replay in 0..if graph_rebind { 2 } else { 1 } {
            // Keep the capture-time buffers alive but replace every cache allocation. On the
            // second replay also reverse canonical/alt addresses, as an odd verify round does.
            let rebound = graph.as_ref().map(|(graph, _keeper)| {
                let conv = e.htod(&conv0).expect("rebound conv");
                let mut first = e.zeros(state_words).expect("rebound first");
                let mut second = e.zeros(state_words).expect("rebound second");
                let (canonical, alt) = if replay == 0 {
                    e.htod_f32_into(&ssm0, &mut first).expect("canonical first");
                    (first, second)
                } else {
                    e.htod_f32_into(&ssm0, &mut second)
                        .expect("canonical second");
                    (second, first)
                };
                e.htod_u64_into(&state_ptrs(&e, &conv, &canonical, &alt), &mut table)
                    .expect("refresh table");
                graph.launch().expect("replay packed");
                (conv, canonical, alt)
            });
            let (conv_state_p, s0p, s1p) =
                rebound
                    .as_ref()
                    .map(|(c, s, a)| (c, s, a))
                    .unwrap_or((&conv_state_p, &s0p, &s1p));

            let got_conv_out = e.dtoh(&conv_outs).expect("cos h");
            let got_o = e.dtoh(&o_all).expect("o h");
            // odd t landed in the alt buffer, even t ran in place: the per-row parity
            let got_final_ssm = e.dtoh(if t % 2 == 1 { s1p } else { s0p }).expect("final p");
            let got_final_conv = e.dtoh(conv_state_p).expect("final conv p");
            let got_snap_conv = e.dtoh(&snap_conv).expect("snap c h");
            let got_snap_ssm = e.dtoh(&snap_ssm).expect("snap s h");
            for (what, r, g) in [
                ("conv_out", &ref_conv_out, &got_conv_out),
                ("scan o", &ref_o, &got_o),
                ("final ssm state", &ref_final_ssm, &got_final_ssm),
                ("final conv ring", &ref_final_conv, &got_final_conv),
                ("conv snapshots", &ref_snap_conv, &got_snap_conv),
                ("ssm snapshots", &ref_snap_ssm, &got_snap_ssm),
            ] {
                assert!(
                    r.iter().chain(g.iter()).all(|x| x.is_finite()),
                    "t={t}: {what} is not finite"
                );
                let d = bit_diffs(r, g);
                assert_eq!(d, 0, "t={t}: {what} differs in {d} of {} words", r.len());
                cells += 1;
            }
        }
    }
    eprintln!(
        "[gdn-packed] graph_rebind={graph_rebind} num_k={num_k} num_v={num_v}: {cells} cells bit-identical (T in 2..=8)"
    );
}

fn state_ptrs(
    e: &Engine,
    conv: &cudarc::driver::CudaSlice<f32>,
    canonical: &cudarc::driver::CudaSlice<f32>,
    alt: &cudarc::driver::CudaSlice<f32>,
) -> [u64; 3] {
    let stream = e.stream();
    let (c, _gc) = conv.device_ptr(&stream);
    let (s, _gs) = canonical.device_ptr(&stream);
    let (a, _ga) = alt.device_ptr(&stream);
    [c, s, a]
}
