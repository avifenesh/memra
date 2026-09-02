//! b200-matvec-bench: shipped-vs-arm microbench for the sm_100a occupancy arms behind
//! `MEMRA_B200_MATVEC_ARM` (lane/b200-matvec-occupancy-20260902, docs/FLAGS.md).
//!
//! WHY. The B200 decode-kernel census (nsys, 2x B200 SXM, GLM-5.3-Flash NVFP4 W4A16, PP2,
//! plain decode) found `moe_gate_up_preclamp8_q8` at 20.0% of GPU time (54.6us avg, ~9x its
//! NVFP4 roofline estimate), `moe_down8_fma_q8` at 10.5%, `matvec_bf16_f32acc_x4_rows` at
//! 17.0% (24.1us avg, ~3x its bf16 roofline estimate), and the NVFP4 rp singles/fused pair
//! at 5.2%/~4% — a latency/occupancy signature on B200's 148-SM/8-TB/s shape from kernels
//! whose block=(32,1,1) or RPW=2 grid was sized for the RTX PRO 6000's 188-SM/1.8-TB/s one.
//! This bin does NOT run on this machine (no GPU here); it is the exact invocation the
//! session with box access runs under `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock` to produce the
//! A/B receipt this lane's PR is pending.
//!
//! WHAT IT MEASURES, per kernel family: the SHIPPED kernel vs its `MEMRA_B200_MATVEC_ARM`
//! twin, called DIRECTLY by kernel name via bench-only `_raw`/`_arm_raw` Engine methods (NOT
//! through the env-gated dispatch, whose `b200_matvec_arm_on()` door is a process-wide
//! `OnceLock` and so cannot flip mid-process) — an interleaved N=5-median timing per shape,
//! plus a byte-for-byte (`f32::to_bits`) output comparison. Every arm here is claimed
//! BIT-IDENTICAL per output (see the .cu kernel comments); a mismatch prints as a WARNING
//! with the max abs diff rather than a silent pass.
//!
//! SHAPES. GLM-5.3-Flash decode, shape-faithful: n_embd=4096, expert ff=1536, 8 active
//! experts, NVFP4 W4A16 expert bytes (interleaved 36B/64-elem `expert_dot_g` layout for the
//! MoE pair; split-plane rp layout for the qmatvec_nvfp4_mmvq_*_rp family); KDA mixer
//! projections [8192,4096] and [4096,8192] bf16. The NVFP4 rp-family shapes (mr2_rp/fused2_rp)
//! are representative attention/mixer square projections (4096x4096), not a specific pinned
//! GLM-5.3 tensor -- noted in research/b200-matvec-occupancy-20260902/LANE.md.
//!
//! usage: b200-matvec-bench [iters=5] [copies=3]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use memra_engine::{Engine, F32x8, QT_NVFP4, WPtr8};
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------------------
// deterministic byte synthesis
// ---------------------------------------------------------------------------------------

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 16) as u8
    }
}

/// e4m3 magnitude 0x7F (either sign) is the hardware NaN code; remap it to a benign
/// mid-range value (same convention as gemv_e4m3_bench's synth_e4m3).
fn safe_e4m3(b: u8) -> u8 {
    if (b & 0x7F) == 0x7F {
        (b & 0x80) | 0x30
    } else {
        b
    }
}

/// Interleaved NVFP4 expert layout consumed by `expert_dot_nvfp4_g` (qmatvec.cu): 36 bytes
/// per 64-element sub-block (sblk) = 4 e4m3 scale bytes (one per 16-elem quarter) + 32
/// nibble-packed quant bytes. `in_f` must be a multiple of 64. Returns (bytes, row_bytes).
fn synth_nvfp4_expert_row_bytes(out_f: usize, in_f: usize, seed: u32) -> (Vec<u8>, usize) {
    assert!(
        in_f.is_multiple_of(64),
        "expert nvfp4 layout needs in_f % 64 == 0"
    );
    let nsb64 = in_f / 64;
    let row_bytes = nsb64 * 36;
    let mut w = vec![0u8; out_f * row_bytes];
    let mut r = Lcg(seed);
    for chunk in w.chunks_exact_mut(36) {
        for d in &mut chunk[0..4] {
            *d = safe_e4m3(r.byte());
        }
        for q in &mut chunk[4..36] {
            *q = r.byte();
        }
    }
    (w, row_bytes)
}

/// Split-plane NVFP4 rp layout consumed by `qmatvec_nvfp4_mmvq_rp` / `_mr2_rp` /
/// `nvfp4_mmvq_fused_seg_rp` (qmatvec.cu): quant plane `[out_f, nsb64*32]` followed by scale
/// plane `[out_f, nsb64*4]`. `in_f` must be a multiple of 64.
fn synth_nvfp4_rp_bytes(out_f: usize, in_f: usize, seed: u32) -> Vec<u8> {
    assert!(
        in_f.is_multiple_of(64),
        "rp nvfp4 layout needs in_f % 64 == 0"
    );
    let nsb64 = in_f / 64;
    let qplane_len = out_f * nsb64 * 32;
    let splane_len = out_f * nsb64 * 4;
    let mut w = vec![0u8; qplane_len + splane_len];
    let mut r = Lcg(seed);
    for b in &mut w[..qplane_len] {
        *b = r.byte();
    }
    for b in &mut w[qplane_len..] {
        *b = safe_e4m3(r.byte());
    }
    w
}

/// bf16 weight rows `[out_f, in_f]`, row stride `in_f` u16 code units. Clears the LSB of the
/// high byte (exponent bit 1) so the 8-bit exponent can never reach 0xFF -> every value is
/// finite (no Inf/NaN), which is all `matvec_bf16_f32acc_x4_rows`'s reduction cares about.
fn synth_bf16(out_f: usize, in_f: usize, seed: u32) -> Vec<u8> {
    let mut w = vec![0u8; out_f * in_f * 2];
    let mut r = Lcg(seed);
    for pair in w.chunks_exact_mut(2) {
        let lo = r.byte();
        let hi = r.byte() & 0xFE;
        pair[0] = lo;
        pair[1] = hi;
    }
    w
}

fn synth_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| ((r.byte() as f32) - 128.0) * (1.0 / 64.0))
        .collect()
}

fn wptr8(bufs: &[CudaSlice<u8>], stream: &Arc<CudaStream>) -> WPtr8 {
    let mut arr = [0u64; 8];
    for (i, b) in bufs.iter().enumerate() {
        let (p, _g) = b.device_ptr(stream);
        arr[i] = p;
    }
    WPtr8(arr)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn gpu_temp() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,clocks.sm",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().replace('\n', " | "))
        .unwrap_or_else(|| "n/a".into())
}

/// Byte-for-byte f32 comparison. Returns (mismatch_count, max_abs_diff).
fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut mism = 0usize;
    let mut maxd = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            mism += 1;
            maxd = maxd.max((x - y).abs());
        }
    }
    (mism, maxd)
}

fn report(label: &str, shipped_us: f64, arm_us: f64, bytes: f64, mism: usize, maxd: f32) {
    let speedup = shipped_us / arm_us;
    let gbs_shipped = bytes / shipped_us / 1e3; // bytes / us == GB/s (1e-6/1e-9 cancel to 1e3)
    let gbs_arm = bytes / arm_us / 1e3;
    let bits = if mism == 0 {
        "bit-identical".to_string()
    } else {
        format!("MISMATCH n={mism} max_abs_diff={maxd:.3e}")
    };
    println!(
        "{label:<40} shipped={shipped_us:>9.2}us ({gbs_shipped:>7.1} GB/s)  arm={arm_us:>9.2}us ({gbs_arm:>7.1} GB/s)  speedup={speedup:>6.3}x  {bits}"
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let copies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    let e = Engine::new(0)?;
    let stream = e.stream();
    println!(
        "GPU: {}  iters={iters}  copies={copies}  temp_in: {}",
        e.ctx().name()?,
        gpu_temp()
    );
    println!(
        "b200-matvec-bench: shipped vs MEMRA_B200_MATVEC_ARM twins, called directly by kernel \
         name (env door bypassed -- OnceLock cannot flip mid-process). N={iters} interleaved, \
         median of per-iteration times."
    );

    // -----------------------------------------------------------------------------------
    // Family 1: MoE gate_up_preclamp8_q8 / gate_up_preclamp8_q8_w4 (20.0% of census GPU time)
    // GLM-5.3-Flash: n_embd=4096 in, n_ff_exp=1536 rows, 8 active experts, NVFP4 W4A16.
    // -----------------------------------------------------------------------------------
    {
        let n_embd = 4096usize;
        let n_ff = 1536usize;
        let n_used = 8usize;
        let limit = 7.0f32;
        let gs = F32x8([1.0; 8]);
        let us = F32x8([1.0; 8]);

        let x = synth_f32(n_embd, 0xC0FF_EE01);
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, n_embd)?;

        let (row_bytes_data, row_bytes) = synth_nvfp4_expert_row_bytes(n_ff, n_embd, 0x1111_2222);
        let mk_experts = |seed: u32| -> Result<Vec<CudaSlice<u8>>, Box<dyn std::error::Error>> {
            (0..n_used)
                .map(|j| {
                    // distinct address per copy/expert; content perturbed per-expert so a
                    // buggy kernel that reads the wrong slot's row shows up as a mismatch.
                    let mut d = row_bytes_data.clone();
                    if let Some(b) = d.first_mut() {
                        *b ^= (seed.wrapping_add(j as u32) & 0xFF) as u8;
                    }
                    e.htod_bytes(&d)
                })
                .collect()
        };
        let gate_copies: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_experts(c as u32 * 97))
            .collect::<Result<_, _>>()?;
        let up_copies: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_experts(c as u32 * 197 + 7))
            .collect::<Result<_, _>>()?;

        // warmup + bit-identity check on copy 0
        let g0 = wptr8(&gate_copies[0], &stream);
        let u0 = wptr8(&up_copies[0], &stream);
        let shipped_act = e.moe_gate_up_preclamp8_q8(
            g0, u0, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4, row_bytes,
            row_bytes,
        )?;
        let g0 = wptr8(&gate_copies[0], &stream);
        let u0 = wptr8(&up_copies[0], &stream);
        let arm_act = e.moe_gate_up_preclamp8_q8_w4(
            g0, u0, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4, row_bytes,
            row_bytes,
        )?;
        e.stream().synchronize()?;
        let h_shipped = e.dtoh(&shipped_act)?;
        let h_arm = e.dtoh(&arm_act)?;
        let (mism, maxd) = compare(&h_shipped, &h_arm);

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_arm = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let g = wptr8(&gate_copies[c], &stream);
            let u = wptr8(&up_copies[c], &stream);
            let t0 = Instant::now();
            let _ = e.moe_gate_up_preclamp8_q8(
                g, u, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4, row_bytes,
                row_bytes,
            )?;
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let g = wptr8(&gate_copies[c], &stream);
            let u = wptr8(&up_copies[c], &stream);
            let t1 = Instant::now();
            let _ = e.moe_gate_up_preclamp8_q8_w4(
                g, u, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4, row_bytes,
                row_bytes,
            )?;
            e.stream().synchronize()?;
            t_arm.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = (n_used * n_ff * row_bytes * 2) as f64; // gate + up
        report(
            "moe_gate_up_preclamp8_q8 (w4)",
            median(&mut t_ship),
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
    }

    // -----------------------------------------------------------------------------------
    // Family 2: MoE down8_fma_q8 / down8_fma_q8_w4 (10.5% of census GPU time)
    // in_f=n_ff_exp=1536, out_f=n_embd=4096, 8 active experts, NVFP4 W4A16.
    // -----------------------------------------------------------------------------------
    {
        let n_ff = 1536usize;
        let n_embd = 4096usize;
        let n_used = 8usize;
        let w = F32x8([0.125; 8]);

        let act = synth_f32(n_used * n_ff, 0xDEAD_BEEF);
        let act_d = e.htod(&act)?;
        let (aq2, ad2) = e.quantize_q8_1(&act_d, n_used, n_ff)?;

        let (row_bytes_data, row_bytes) = synth_nvfp4_expert_row_bytes(n_embd, n_ff, 0x3333_4444);
        let mk_experts = |seed: u32| -> Result<Vec<CudaSlice<u8>>, Box<dyn std::error::Error>> {
            (0..n_used)
                .map(|j| {
                    let mut d = row_bytes_data.clone();
                    if let Some(b) = d.first_mut() {
                        *b ^= (seed.wrapping_add(j as u32) & 0xFF) as u8;
                    }
                    e.htod_bytes(&d)
                })
                .collect()
        };
        let down_copies: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_experts(c as u32 * 53 + 3))
            .collect::<Result<_, _>>()?;

        let mut y_ship = e.zeros(n_embd)?;
        let d0 = wptr8(&down_copies[0], &stream);
        {
            let mut dst = y_ship.slice_mut(0..n_embd);
            e.moe_down8_fma_q8(
                d0, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
            )?;
        }
        let mut y_arm = e.zeros(n_embd)?;
        let d0 = wptr8(&down_copies[0], &stream);
        {
            let mut dst = y_arm.slice_mut(0..n_embd);
            e.moe_down8_fma_q8_w4(
                d0, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
            )?;
        }
        e.stream().synchronize()?;
        let h_ship = e.dtoh(&y_ship)?;
        let h_arm = e.dtoh(&y_arm)?;
        let (mism, maxd) = compare(&h_ship, &h_arm);

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_arm = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let mut y = e.zeros(n_embd)?;
            let d = wptr8(&down_copies[c], &stream);
            let t0 = Instant::now();
            {
                let mut dst = y.slice_mut(0..n_embd);
                e.moe_down8_fma_q8(
                    d, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
                )?;
            }
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let mut y = e.zeros(n_embd)?;
            let d = wptr8(&down_copies[c], &stream);
            let t1 = Instant::now();
            {
                let mut dst = y.slice_mut(0..n_embd);
                e.moe_down8_fma_q8_w4(
                    d, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
                )?;
            }
            e.stream().synchronize()?;
            t_arm.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = (n_used * n_embd * row_bytes) as f64;
        report(
            "moe_down8_fma_q8 (w4)",
            median(&mut t_ship),
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
    }

    // -----------------------------------------------------------------------------------
    // Family 3: matvec_bf16_f32acc_x4_rows / _pf (17.0% of census GPU time)
    // KDA mixer projections, both directions, bf16 W, f32 accumulate, t=1 decode.
    // -----------------------------------------------------------------------------------
    for (in_f, out_f, label) in [
        (4096usize, 8192usize, "kda up 4096->8192"),
        (8192, 4096, "kda down 8192->4096"),
    ] {
        let h_w = synth_bf16(out_f, in_f, 0x5555_6666);
        let x = synth_f32(in_f, 0x7777_8888);
        let xd = e.htod(&x)?;
        let wcopies: Vec<CudaSlice<u8>> = (0..copies)
            .map(|_| e.htod_bytes(&h_w))
            .collect::<Result<_, _>>()?;

        let mut y_ship = e.zeros(out_f)?;
        e.matvec_bf16_f32acc_x4_rows_arm_raw(&wcopies[0], &xd, &mut y_ship, in_f, out_f, 1, false)?;
        let mut y_arm = e.zeros(out_f)?;
        e.matvec_bf16_f32acc_x4_rows_arm_raw(&wcopies[0], &xd, &mut y_arm, in_f, out_f, 1, true)?;
        e.stream().synchronize()?;
        let h_ship = e.dtoh(&y_ship)?;
        let h_arm = e.dtoh(&y_arm)?;
        let (mism, maxd) = compare(&h_ship, &h_arm);

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_arm = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let mut y = e.zeros(out_f)?;
            let t0 = Instant::now();
            e.matvec_bf16_f32acc_x4_rows_arm_raw(&wcopies[c], &xd, &mut y, in_f, out_f, 1, false)?;
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let mut y = e.zeros(out_f)?;
            let t1 = Instant::now();
            e.matvec_bf16_f32acc_x4_rows_arm_raw(&wcopies[c], &xd, &mut y, in_f, out_f, 1, true)?;
            e.stream().synchronize()?;
            t_arm.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = (in_f * out_f * 2) as f64;
        report(
            &format!("matvec_bf16_f32acc_x4_rows (pf) {label}"),
            median(&mut t_ship),
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
    }

    // -----------------------------------------------------------------------------------
    // Family 4: qmatvec_nvfp4_mmvq_mr2_rp / grid-fill (mr1, reuses the shipped _rp kernel)
    // (5.2% of census GPU time). Representative square NVFP4 rp projection, m=1 decode.
    // -----------------------------------------------------------------------------------
    {
        let in_f = 4096usize;
        let out_f = 4096usize;
        let h_w = synth_nvfp4_rp_bytes(out_f, in_f, 0x9999_AAAA);
        let x = synth_f32(in_f, 0xBBBB_CCCC);
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        let wcopies: Vec<CudaSlice<u8>> = (0..copies)
            .map(|_| e.htod_bytes(&h_w))
            .collect::<Result<_, _>>()?;
        let row_bytes = (in_f / 64) * 36; // unused by the rp kernels (split-plane addressing) but kept for the record

        let y_ship = e.qmatvec_nvfp4_rp_arm_raw(
            &wcopies[0],
            &aq,
            &ad,
            1,
            in_f,
            out_f,
            row_bytes,
            1.0,
            false,
        )?;
        let y_arm = e.qmatvec_nvfp4_rp_arm_raw(
            &wcopies[0],
            &aq,
            &ad,
            1,
            in_f,
            out_f,
            row_bytes,
            1.0,
            true,
        )?;
        e.stream().synchronize()?;
        let h_ship = e.dtoh(&y_ship)?;
        let h_arm = e.dtoh(&y_arm)?;
        let (mism, maxd) = compare(&h_ship, &h_arm);

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_arm = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let t0 = Instant::now();
            let _ = e.qmatvec_nvfp4_rp_arm_raw(
                &wcopies[c],
                &aq,
                &ad,
                1,
                in_f,
                out_f,
                row_bytes,
                1.0,
                false,
            )?;
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let t1 = Instant::now();
            let _ = e.qmatvec_nvfp4_rp_arm_raw(
                &wcopies[c],
                &aq,
                &ad,
                1,
                in_f,
                out_f,
                row_bytes,
                1.0,
                true,
            )?;
            e.stream().synchronize()?;
            t_arm.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = h_w.len() as f64;
        report(
            "qmatvec_nvfp4_mmvq_mr2_rp (grid-fill mr1)",
            median(&mut t_ship),
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
    }

    // -----------------------------------------------------------------------------------
    // Family 5: qmatvec_nvfp4_mmvq_fused2_rp / _g2 (RPW=1 grid-fill twin). Representative
    // gate+up-style NVFP4 rp pair, m=1 decode.
    // -----------------------------------------------------------------------------------
    {
        let in_f = 4096usize;
        let out0 = 4096usize;
        let out1 = 4096usize;
        let h_w0 = synth_nvfp4_rp_bytes(out0, in_f, 0x1234_5678);
        let h_w1 = synth_nvfp4_rp_bytes(out1, in_f, 0x8765_4321);
        let x = synth_f32(in_f, 0xF0F0_0F0F);
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        let w0_copies: Vec<CudaSlice<u8>> = (0..copies)
            .map(|_| e.htod_bytes(&h_w0))
            .collect::<Result<_, _>>()?;
        let w1_copies: Vec<CudaSlice<u8>> = (0..copies)
            .map(|_| e.htod_bytes(&h_w1))
            .collect::<Result<_, _>>()?;

        let (y0s, y1s) = e.qmatvec_nvfp4_fused2_rp_arm_raw(
            &w0_copies[0],
            &w1_copies[0],
            &aq,
            &ad,
            in_f,
            out0,
            out1,
            1.0,
            1.0,
            false,
        )?;
        let (y0a, y1a) = e.qmatvec_nvfp4_fused2_rp_arm_raw(
            &w0_copies[0],
            &w1_copies[0],
            &aq,
            &ad,
            in_f,
            out0,
            out1,
            1.0,
            1.0,
            true,
        )?;
        e.stream().synchronize()?;
        let (h_y0s, h_y1s) = (e.dtoh(&y0s)?, e.dtoh(&y1s)?);
        let (h_y0a, h_y1a) = (e.dtoh(&y0a)?, e.dtoh(&y1a)?);
        let (mism0, maxd0) = compare(&h_y0s, &h_y0a);
        let (mism1, maxd1) = compare(&h_y1s, &h_y1a);
        let mism = mism0 + mism1;
        let maxd = maxd0.max(maxd1);

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_arm = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let t0 = Instant::now();
            let _ = e.qmatvec_nvfp4_fused2_rp_arm_raw(
                &w0_copies[c],
                &w1_copies[c],
                &aq,
                &ad,
                in_f,
                out0,
                out1,
                1.0,
                1.0,
                false,
            )?;
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let t1 = Instant::now();
            let _ = e.qmatvec_nvfp4_fused2_rp_arm_raw(
                &w0_copies[c],
                &w1_copies[c],
                &aq,
                &ad,
                in_f,
                out0,
                out1,
                1.0,
                1.0,
                true,
            )?;
            e.stream().synchronize()?;
            t_arm.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = (h_w0.len() + h_w1.len()) as f64;
        report(
            "qmatvec_nvfp4_mmvq_fused2_rp (g2)",
            median(&mut t_ship),
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
    }

    println!("temp_out: {}", gpu_temp());
    Ok(())
}
