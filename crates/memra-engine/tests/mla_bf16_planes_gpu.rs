//! memra#135 consumption gate: the BF16-resident MLA absorb planes against the f32 kernels on the
//! SAME values. `__bfloat162float` is exact (BF16 is a truncated f32 significand), the twins keep
//! the f32 kernels' accumulation order, and the activation side stays f32 — so widening a BF16
//! plane to f32 and running the shipped kernel must produce THE SAME BITS as running the twin on
//! the BF16 plane. That is an identity gate, not a tolerance, and its red arm is below.
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
    M.lock().unwrap_or_else(|p| p.into_inner())
}

/// Deterministic values that are EXACTLY representable in BF16 (top 16 bits of an f32), so the
/// gate compares the two kernels rather than a rounding difference the mint would own.
fn bf16_exact(n: usize, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut s = seed;
    let mut bits = Vec::with_capacity(n);
    let mut wide = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
        let hi = (u.to_bits() >> 16) as u16; // truncate to BF16
        bits.push(hi);
        wide.push(f32::from_bits((hi as u32) << 16));
    }
    (bits, wide)
}

fn lcg_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn bf16_planes_match_the_f32_kernels_bitwise() {
    let _g = gpu_guard();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // The served GLM-5.3-Flash MLA geometry, one query row.
    let (t_q, n_head, d_nope, kv_rank, d_v) = (1usize, 64usize, 256usize, 512usize, 256usize);

    // ---- absorb: q_lat[t*nh, kv_rank] = q_nope[t*nh, d_nope] . wk_b[nh, kv_rank, d_nope] ----
    let q_nope = e
        .htod(&lcg_f32(t_q * n_head * d_nope, 0xA85_0413))
        .expect("q_nope");
    let (wk_bits, wk_wide) = bf16_exact(n_head * kv_rank * d_nope, 0x0B16_0135);
    let wk_f32 = e.htod(&wk_wide).expect("wk_b f32");
    let wk_bf16 = e
        .htod_bytes(
            &wk_bits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("wk_b bf16");
    let mut want = e.htod(&vec![0.0f32; t_q * n_head * kv_rank]).expect("want");
    let mut got = e.htod(&vec![0.0f32; t_q * n_head * kv_rank]).expect("got");
    e.mla_absorb_q(&q_nope, &wk_f32, &mut want, t_q, n_head, d_nope, kv_rank)
        .expect("f32 absorb");
    let s = e.stream();
    let rc = unsafe {
        memra_engine::mla_ffi::memra_mla_absorb_q_bf16(
            q_nope.device_ptr(&s).0 as *const f32,
            wk_bf16.device_ptr(&s).0 as *const core::ffi::c_void,
            got.device_ptr_mut(&s).0 as *mut f32,
            t_q as i32,
            n_head as i32,
            d_nope as i32,
            kv_rank as i32,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "bf16 absorb launcher rc");
    e.stream().synchronize().expect("sync");
    let (a, b) = (e.dtoh(&want).unwrap(), e.dtoh(&got).unwrap());
    let nz = a.iter().filter(|v| **v != 0.0).count();
    assert!(
        nz > a.len() / 2,
        "vacuous: the f32 absorb produced {nz} nonzero of {}",
        a.len()
    );
    let diffs = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diffs,
        0,
        "absorb: bf16 twin differs from the f32 kernel in {diffs}/{} outputs",
        a.len()
    );

    // ---- decompress: out[t*nh, d_v] = o_lat[t*nh, kv_rank] . wv_b[nh, d_v, kv_rank] ----
    let o_lat = e
        .htod(&lcg_f32(t_q * n_head * kv_rank, 0xDEC0_0413))
        .expect("o_lat");
    let (wv_bits, wv_wide) = bf16_exact(n_head * d_v * kv_rank, 0x0B16_0136);
    let wv_f32 = e.htod(&wv_wide).expect("wv_b f32");
    let wv_bf16 = e
        .htod_bytes(
            &wv_bits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("wv_b bf16");
    let mut want_v = e.htod(&vec![0.0f32; t_q * n_head * d_v]).expect("want_v");
    let mut got_v = e.htod(&vec![0.0f32; t_q * n_head * d_v]).expect("got_v");
    // Compare against the WARP-PER-ROW f32 launcher at the SAME split: `mla_decompress_v` is a
    // dispatcher (coalesce door, split policy) and its other kernels sum in a different order, so
    // calling it here would compare orders instead of dtypes — which is exactly what the first
    // run of this gate did (15,508 of 16,384 "differences" that were the wrong reference).
    let split = 4i32;
    let rc = unsafe {
        memra_engine::mla_ffi::memra_mla_decompress_v_wp_f32(
            o_lat.device_ptr(&s).0 as *const f32,
            wv_f32.device_ptr(&s).0 as *const f32,
            want_v.device_ptr_mut(&s).0 as *mut f32,
            t_q as i32,
            n_head as i32,
            d_v as i32,
            kv_rank as i32,
            split,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "f32 wp decompress launcher rc");
    e.stream().synchronize().expect("sync");
    let rc = unsafe {
        memra_engine::mla_ffi::memra_mla_decompress_v_wp_bf16(
            o_lat.device_ptr(&s).0 as *const f32,
            wv_bf16.device_ptr(&s).0 as *const core::ffi::c_void,
            got_v.device_ptr_mut(&s).0 as *mut f32,
            t_q as i32,
            n_head as i32,
            d_v as i32,
            kv_rank as i32,
            split,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "bf16 decompress launcher rc");
    e.stream().synchronize().expect("sync");
    let (a, b) = (e.dtoh(&want_v).unwrap(), e.dtoh(&got_v).unwrap());
    let diffs = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diffs,
        0,
        "decompress: bf16 twin differs from the f32 kernel in {diffs}/{} outputs",
        a.len()
    );

    // ---- RED ARM: perturb one BF16 weight and the identity must break. ----
    let mut poisoned = wk_bits.clone();
    poisoned[7] ^= 1;
    let wk_bad = e
        .htod_bytes(
            &poisoned
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("poisoned");
    let mut got_bad = e
        .htod(&vec![0.0f32; t_q * n_head * kv_rank])
        .expect("got_bad");
    let rc = unsafe {
        memra_engine::mla_ffi::memra_mla_absorb_q_bf16(
            q_nope.device_ptr(&s).0 as *const f32,
            wk_bad.device_ptr(&s).0 as *const core::ffi::c_void,
            got_bad.device_ptr_mut(&s).0 as *mut f32,
            t_q as i32,
            n_head as i32,
            d_nope as i32,
            kv_rank as i32,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "red-arm launcher rc");
    e.stream().synchronize().expect("sync");
    let bad = e.dtoh(&got_bad).unwrap();
    let want_a = e.dtoh(&want).unwrap();
    let red = want_a
        .iter()
        .zip(&bad)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert!(
        red > 0,
        "red arm did not bite: a flipped BF16 weight bit changed nothing"
    );

    println!(
        "[mla-bf16-planes] absorb and decompress bit-identical to the f32 kernels at the served \
         geometry (nh={n_head} d_nope={d_nope} kv_rank={kv_rank} d_v={d_v}); red arm bites {red} outputs"
    );
}
