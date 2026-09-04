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
use memra_engine::tp::nvfp4_matrix_v2_permute;
use memra_engine::{Engine, F32x8, QT_NVFP4, QT_NVFP4_V2, QT_Q8_0, WPtr8};
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

/// bf16 weight rows `[out_f, in_f]`, row stride `in_f` u16 code units.
///
/// FIXED 2026-09-02 after the first box run. The old generator randomised the whole 16-bit
/// pattern and only cleared ONE exponent bit, so weights ranged over the full bf16 exponent
/// (magnitudes to ~2^127). Every element was individually finite, but a 4096-term dot of them
/// OVERFLOWS f32 — which is invisible while both arms run the same program (inf == inf compares
/// bit-identical, and shipped-vs-`_pf` did) and becomes `max_abs_diff=inf` the moment an arm
/// sums in a different order. That is exactly what the LT-reference line reported. The compare
/// was never broken; the DATA was.
///
/// Now: sample the same [-2, 2) range `synth_f32` uses and truncate f32 -> bf16 (drop the low 16
/// mantissa bits, the bf16 layout by construction). With |w| < 2 and |x| < 2 a 4096-term dot is
/// bounded by 16384, so every arm's sum is finite and a cross-order diff is a real number.
fn synth_bf16(out_f: usize, in_f: usize, seed: u32) -> Vec<u8> {
    let mut w = vec![0u8; out_f * in_f * 2];
    let mut r = Lcg(seed);
    for pair in w.chunks_exact_mut(2) {
        let v = ((r.byte() as f32) - 128.0) * (1.0 / 64.0);
        let bits = (v.to_bits() >> 16) as u16;
        pair.copy_from_slice(&bits.to_le_bytes());
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
///
/// PANICS on a non-finite value in either arm. A NaN/Inf output makes every downstream verdict
/// meaningless — bit-identity becomes `inf == inf` and a cross-order diff becomes `inf` — and
/// the first box run spent a receipt discovering that the hard way (`max_abs_diff=inf` on the
/// LT-reference line, from a weight generator whose bf16 exponents reached ~2^127). A bench
/// that cannot produce finite numbers must say so loudly, not print `inf` and move on.
fn compare(a: &[f32], b: &[f32]) -> (usize, f32) {
    let mut mism = 0usize;
    let mut maxd = 0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "bench arm produced a non-finite value at index {i}: shipped={x} arm={y} — the \
             synthetic operands overflow f32, so no identity or diff verdict from this run \
             means anything (see synth_bf16's note)"
        );
        if x.to_bits() != y.to_bits() {
            mism += 1;
            maxd = maxd.max((x - y).abs());
        }
    }
    (mism, maxd)
}

/// MEMRA_BENCH_DUMP=<dir>: write an arm's raw f32 output for cross-binary bit comparison.
fn dump(name: &str, v: &[f32]) {
    if let Ok(dir) = std::env::var("MEMRA_BENCH_DUMP") {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(format!("{dir}/{name}.f32"), bytes);
    }
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

/// One extra arm against the shipped baseline, on its own line.
fn report_arm(
    label: &str,
    arm: &str,
    base_us: f64,
    arm_us: f64,
    bytes: f64,
    mism: usize,
    maxd: f32,
) {
    let gbs = bytes / arm_us / 1e3;
    let bits = if mism == 0 {
        "bit-identical".to_string()
    } else {
        format!("MISMATCH n={mism} max_abs_diff={maxd:.3e}")
    };
    println!(
        "{:<40} {arm}={arm_us:>9.2}us ({gbs:>7.1} GB/s)  vs shipped={:>6.3}x  {bits}",
        format!("  {label}"),
        base_us / arm_us
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
        dump("gate_up_shipped", &h_shipped);
        dump("gate_up_w4", &h_arm);

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
        let ship_us = median(&mut t_ship);
        report(
            "moe_gate_up_preclamp8_q8 (w4)",
            ship_us,
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );

        // MEMRA_MOE_EXPERT_RP arm (memra#147): the SAME expert bytes, split-plane repacked per
        // expert (nvfp4_matrix_v2_permute, the QT_NVFP4_V2 form) and read by the _w4 twin's QT_NVFP4_V2 branch.
        // Bit-identity is against the SHIPPED interleaved kernel on copy 0; the perturbation is
        // applied before the repack so every copy's content matches its interleaved twin.
        let mk_experts_rp = |seed: u32| -> Result<Vec<CudaSlice<u8>>, Box<dyn std::error::Error>> {
            (0..n_used)
                .map(|j| {
                    let mut d = row_bytes_data.clone();
                    if let Some(b) = d.first_mut() {
                        *b ^= (seed.wrapping_add(j as u32) & 0xFF) as u8;
                    }
                    e.htod_bytes(&nvfp4_matrix_v2_permute(&d, n_ff, n_embd))
                })
                .collect()
        };
        let gate_rp: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_experts_rp(c as u32 * 97))
            .collect::<Result<_, _>>()?;
        let up_rp: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_experts_rp(c as u32 * 197 + 7))
            .collect::<Result<_, _>>()?;
        let g0 = wptr8(&gate_rp[0], &stream);
        let u0 = wptr8(&up_rp[0], &stream);
        let rp_act = e.moe_gate_up_preclamp8_q8_w4(
            g0,
            u0,
            &aq,
            &ad,
            gs,
            us,
            limit,
            n_embd,
            n_ff,
            n_used,
            QT_NVFP4_V2,
            QT_NVFP4_V2,
            row_bytes,
            row_bytes,
        )?;
        e.stream().synchronize()?;
        let h_rp = e.dtoh(&rp_act)?;
        let (mism_rp, maxd_rp) = compare(&h_shipped, &h_rp);
        dump("gate_up_rp", &h_rp);
        for (i, (a, b)) in h_shipped
            .iter()
            .zip(h_rp.iter())
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .take(3)
        {
            println!(
                "    rp mismatch gate_up idx={i} (slot={} row={}) shipped={a:e} ({:08x}) rp={b:e} ({:08x})",
                i / n_ff,
                i % n_ff,
                a.to_bits(),
                b.to_bits()
            );
        }
        let mut t_rp = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let g = wptr8(&gate_rp[c], &stream);
            let u = wptr8(&up_rp[c], &stream);
            let t1 = Instant::now();
            let _ = e.moe_gate_up_preclamp8_q8_w4(
                g,
                u,
                &aq,
                &ad,
                gs,
                us,
                limit,
                n_embd,
                n_ff,
                n_used,
                QT_NVFP4_V2,
                QT_NVFP4_V2,
                row_bytes,
                row_bytes,
            )?;
            e.stream().synchronize()?;
            t_rp.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        report_arm(
            "moe_gate_up_preclamp8_q8",
            "w4_v2",
            ship_us,
            median(&mut t_rp),
            bytes,
            mism_rp,
            maxd_rp,
        );

        // MEMRA_B200_GEMV_V2 arm: 8 warps/block + g-walk unrolled by two.
        {
            let g0 = wptr8(&gate_copies[0], &stream);
            let u0 = wptr8(&up_copies[0], &stream);
            let v2_act = e.moe_gate_up_preclamp8_q8_v2(
                g0, u0, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4,
                row_bytes, row_bytes,
            )?;
            e.stream().synchronize()?;
            let h_v2 = e.dtoh(&v2_act)?;
            let (mv, dv) = compare(&h_shipped, &h_v2);
            let mut t_v2 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let g = wptr8(&gate_copies[c], &stream);
                let u = wptr8(&up_copies[c], &stream);
                let t0 = Instant::now();
                let _ = e.moe_gate_up_preclamp8_q8_v2(
                    g, u, &aq, &ad, gs, us, limit, n_embd, n_ff, n_used, QT_NVFP4, QT_NVFP4,
                    row_bytes, row_bytes,
                )?;
                e.stream().synchronize()?;
                t_v2.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            report_arm(
                "moe_gate_up_preclamp8_q8_v2",
                "v2",
                ship_us,
                median(&mut t_v2),
                bytes,
                mv,
                dv,
            );
        }
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
        dump("down_shipped", &h_ship);
        dump("down_w4", &h_arm);

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
        let ship_us = median(&mut t_ship);
        report(
            "moe_down8_fma_q8 (w4)",
            ship_us,
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );
        // MEMRA_MOE_EXPERT_RP arm (memra#147): split-plane down slabs, rows = n_embd.
        let mk_down_rp = |seed: u32| -> Result<Vec<CudaSlice<u8>>, Box<dyn std::error::Error>> {
            (0..n_used)
                .map(|j| {
                    let mut d = row_bytes_data.clone();
                    if let Some(b) = d.first_mut() {
                        *b ^= (seed.wrapping_add(j as u32) & 0xFF) as u8;
                    }
                    e.htod_bytes(&nvfp4_matrix_v2_permute(&d, n_embd, n_ff))
                })
                .collect()
        };
        let down_rp: Vec<Vec<CudaSlice<u8>>> = (0..copies)
            .map(|c| mk_down_rp(c as u32 * 53 + 3))
            .collect::<Result<_, _>>()?;
        let mut y_rp = e.zeros(n_embd)?;
        let d0 = wptr8(&down_rp[0], &stream);
        {
            let mut dst = y_rp.slice_mut(0..n_embd);
            e.moe_down8_fma_q8_w4(
                d0,
                w,
                &aq2,
                &ad2,
                &mut dst,
                n_ff,
                n_embd,
                n_used,
                QT_NVFP4_V2,
                row_bytes,
            )?;
        }
        e.stream().synchronize()?;
        let h_rp = e.dtoh(&y_rp)?;
        let (mism_rp, maxd_rp) = compare(&h_ship, &h_rp);
        dump("down_rp", &h_rp);
        for (i, (a, b)) in h_ship
            .iter()
            .zip(h_rp.iter())
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .take(3)
        {
            println!(
                "    rp mismatch down row={i} shipped={a:e} ({:08x}) rp={b:e} ({:08x})",
                a.to_bits(),
                b.to_bits()
            );
        }
        let mut t_rp = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let mut y = e.zeros(n_embd)?;
            let d = wptr8(&down_rp[c], &stream);
            let t1 = Instant::now();
            {
                let mut dst = y.slice_mut(0..n_embd);
                e.moe_down8_fma_q8_w4(
                    d,
                    w,
                    &aq2,
                    &ad2,
                    &mut dst,
                    n_ff,
                    n_embd,
                    n_used,
                    QT_NVFP4_V2,
                    row_bytes,
                )?;
            }
            e.stream().synchronize()?;
            t_rp.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        report_arm(
            "moe_down8_fma_q8",
            "w4_v2",
            ship_us,
            median(&mut t_rp),
            bytes,
            mism_rp,
            maxd_rp,
        );

        // MEMRA_B200_GEMV_V2 arm: one block per output row, warp j owning expert slot j.
        {
            let mut y_v2 = e.zeros(n_embd)?;
            let d0 = wptr8(&down_copies[0], &stream);
            {
                let mut dst = y_v2.slice_mut(0..n_embd);
                e.moe_down8_fma_q8_v2(
                    d0, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
                )?;
            }
            e.stream().synchronize()?;
            let h_v2 = e.dtoh(&y_v2)?;
            let (mv, dv) = compare(&h_ship, &h_v2);
            let mut t_v2 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut y = e.zeros(n_embd)?;
                let d = wptr8(&down_copies[c], &stream);
                let t0 = Instant::now();
                {
                    let mut dst = y.slice_mut(0..n_embd);
                    e.moe_down8_fma_q8_v2(
                        d, w, &aq2, &ad2, &mut dst, n_ff, n_embd, n_used, QT_NVFP4, row_bytes,
                    )?;
                }
                e.stream().synchronize()?;
                t_v2.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            report_arm(
                "moe_down8_fma_q8_v2",
                "v2",
                ship_us,
                median(&mut t_v2),
                bytes,
                mv,
                dv,
            );
        }
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
        let ship_us = median(&mut t_ship);
        report(
            &format!("matvec_bf16_f32acc_x4_rows (pf) {label}"),
            ship_us,
            median(&mut t_arm),
            bytes,
            mism,
            maxd,
        );

        // MEMRA_B200_GEMV_V2 arm: 8 rows/block accumulated concurrently on one activation
        // load, ten 16 B `ld.global.nc` loads in flight before the first fma, one barrier
        // chain per block. ksplit comes from the same chooser the dispatch uses, and is
        // printed so a sub-2-wave shape's `bf16_gemv_v2_splitk` class is never silent.
        {
            let ksplit = e.gemv_v2_ksplit(in_f, out_f, 1);
            let mut y_v2 = e.zeros(out_f)?;
            e.matvec_bf16_v2_raw(&wcopies[0], &xd, &mut y_v2, in_f, out_f, 1, ksplit)?;
            e.stream().synchronize()?;
            let h_v2 = e.dtoh(&y_v2)?;
            let (mv, dv) = compare(&h_ship, &h_v2);
            let mut t_v2 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut y = e.zeros(out_f)?;
                let t0 = Instant::now();
                e.matvec_bf16_v2_raw(&wcopies[c], &xd, &mut y, in_f, out_f, 1, ksplit)?;
                e.stream().synchronize()?;
                t_v2.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            report_arm(
                &format!("matvec_bf16_v2 {label} (ksplit={ksplit})"),
                "v2",
                ship_us,
                median(&mut t_v2),
                bytes,
                mv,
                dv,
            );
        }

        // v3 (MEMRA_B200_GEMV_V2=2): the same walk with its weight tiles staged through shared
        // memory by cp.async, so the in-flight budget is smem-bound instead of register-bound.
        // Skipped when its 36 KB (at mmv_block()=128) would not fit the 48 KB default cap.
        if e.gemv_v3_fits() {
            let mut y_v3 = e.zeros(out_f)?;
            e.matvec_bf16_v3_raw(&wcopies[0], &xd, &mut y_v3, in_f, out_f, 1)?;
            e.stream().synchronize()?;
            let h_v3 = e.dtoh(&y_v3)?;
            let (mv, dv) = compare(&h_ship, &h_v3);
            let mut t_v3 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut y = e.zeros(out_f)?;
                let t0 = Instant::now();
                e.matvec_bf16_v3_raw(&wcopies[c], &xd, &mut y, in_f, out_f, 1)?;
                e.stream().synchronize()?;
                t_v3.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            report_arm(
                &format!("matvec_bf16_v3 {label} (cp.async staged)"),
                "v3",
                ship_us,
                median(&mut t_v3),
                bytes,
                mv,
                dv,
            );
        } else {
            println!("  matvec_bf16_v3 {label}: SKIPPED (v3 smem over the 48 KB default cap)");
        }

        // cuBLASLt REFERENCE arm (MEMRA_B200_BF16_GEMV_LT, lane/b200-gemv-hbm-20260902).
        // Called directly (the door is a process-wide OnceLock, so it cannot flip mid-run).
        // NAMED NUMERIC CLASS `bf16_gemv_lt`: the activation is cast f32 -> bf16 and the K
        // summation order is the library's, so a byte mismatch here is EXPECTED and the max
        // abs diff is the number to read, not the mismatch count.
        {
            let mut y_lt = e.zeros(out_f)?;
            let took = e.bf16_gemv_lt_into(&wcopies[0], &xd, &mut y_lt, in_f, out_f, 1)?;
            e.stream().synchronize()?;
            if !took {
                println!(
                    "{:<40} cuBLASLt DECLINED this shape (see the [b200-bf16-gemv-lt] line above)",
                    format!("  bf16_gemv_lt {label}")
                );
            } else {
                let h_lt = e.dtoh(&y_lt)?;
                let (mism_lt, maxd_lt) = compare(&h_ship, &h_lt);
                let mut t_lt = Vec::with_capacity(iters);
                for i in 0..iters {
                    let c = i % copies;
                    let mut y = e.zeros(out_f)?;
                    let t0 = Instant::now();
                    e.bf16_gemv_lt_into(&wcopies[c], &xd, &mut y, in_f, out_f, 1)?;
                    e.stream().synchronize()?;
                    t_lt.push(t0.elapsed().as_secs_f64() * 1e6);
                }
                let lt_us = median(&mut t_lt);
                let gbs = bytes / lt_us / 1e3;
                let cls = if mism_lt == 0 {
                    "bit-identical (unexpected for this class)".to_string()
                } else {
                    format!("class bf16_gemv_lt: n={mism_lt} max_abs_diff={maxd_lt:.3e}")
                };
                println!(
                    "{:<40} LT-ref={lt_us:>9.2}us ({gbs:>7.1} GB/s)  vs shipped={:>6.3}x  {cls}",
                    format!("  bf16_gemv_lt {label}"),
                    ship_us / lt_us
                );
            }
        }
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

    // -----------------------------------------------------------------------------------
    // Family 6: qmatvec_kda6_bf16f32 / _v2 (the census's single hottest launch: 93.8us for
    // ~200 MB of bf16 reads = 2.1 TB/s, 26% of the 8 TB/s HBM3e wall). GLM-5.3-Flash KDA
    // stage-1 six-projection group: three bf16 [8192, 4096] mixer projections (that is where
    // the ~192 MB lives) plus the three small f32 low-rank/beta rows.
    // -----------------------------------------------------------------------------------
    {
        let in_f = 4096usize;
        let dims = [8192usize, 8192, 8192, 128, 128, 64];
        let h_bf = synth_bf16(dims[0], in_f, 0x2468_ACE0);
        let x = synth_f32(in_f, 0x1357_9BDF);
        let xd = e.htod(&x)?;
        let f32w: Vec<CudaSlice<f32>> = (3..6)
            .map(|k| e.htod(&synth_f32(dims[k] * in_f, 0xA0A0 + k as u32)))
            .collect::<Result<_, _>>()?;
        // Distinct device copies per iteration so no run is served a warm L2 the others paid for.
        let bf_copies: Vec<[CudaSlice<u8>; 3]> = (0..copies)
            .map(
                |_| -> Result<[CudaSlice<u8>; 3], Box<dyn std::error::Error>> {
                    Ok([
                        e.htod_bytes(&h_bf)?,
                        e.htod_bytes(&h_bf)?,
                        e.htod_bytes(&h_bf)?,
                    ])
                },
            )
            .collect::<Result<_, _>>()?;
        let mk_outs = || -> Result<[CudaSlice<f32>; 6], Box<dyn std::error::Error>> {
            Ok([
                e.zeros(dims[0])?,
                e.zeros(dims[1])?,
                e.zeros(dims[2])?,
                e.zeros(dims[3])?,
                e.zeros(dims[4])?,
                e.zeros(dims[5])?,
            ])
        };
        let run = |c: usize,
                   outs: &mut [CudaSlice<f32>; 6],
                   arm: u8|
         -> Result<(), Box<dyn std::error::Error>> {
            let b = &bf_copies[c];
            e.kda_proj_fused6_bf16_arm_raw(
                &b[0], &b[1], &b[2], &f32w[0], &f32w[1], &f32w[2], &xd, outs, in_f, dims, 1, arm,
            )
        };
        // Per-projection identity, not just a total: the first cut's y5 bug (a dropped
        // `b -= nb` before the last range) showed up as 64 mismatches with the other five
        // ranges clean, and a single summed count made that one number instead of a location.
        let cmp6 = |a: &[CudaSlice<f32>; 6],
                    b: &[CudaSlice<f32>; 6]|
         -> Result<(usize, f32, String), Box<dyn std::error::Error>> {
            let mut mism = 0usize;
            let mut maxd = 0f32;
            let mut per = Vec::new();
            for k in 0..6 {
                let (m, d) = compare(&e.dtoh(&a[k])?, &e.dtoh(&b[k])?);
                mism += m;
                maxd = maxd.max(d);
                per.push(format!("y{k}={m}"));
            }
            Ok((mism, maxd, per.join(" ")))
        };

        let mut o_ship = mk_outs()?;
        run(0, &mut o_ship, 0)?;
        let mut o_v2 = mk_outs()?;
        run(0, &mut o_v2, 1)?;
        e.stream().synchronize()?;
        let (mism, maxd, per2) = cmp6(&o_ship, &o_v2)?;

        let mut t_ship = Vec::with_capacity(iters);
        let mut t_v2 = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let mut outs = mk_outs()?;
            let t0 = Instant::now();
            run(c, &mut outs, 0)?;
            e.stream().synchronize()?;
            t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

            let mut outs = mk_outs()?;
            let t1 = Instant::now();
            run(c, &mut outs, 1)?;
            e.stream().synchronize()?;
            t_v2.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bf_bytes = 3 * dims[0] * in_f * 2;
        let f32_bytes = (dims[3] + dims[4] + dims[5]) * in_f * 4;
        let bytes = (bf_bytes + f32_bytes) as f64;
        let ship_us = median(&mut t_ship);
        report(
            "qmatvec_kda6_bf16f32 (v2)",
            ship_us,
            median(&mut t_v2),
            bytes,
            mism,
            maxd,
        );
        println!("  per-projection mismatch counts (v2): {per2}");

        if e.gemv_v3_fits() {
            let mut o_v3 = mk_outs()?;
            run(0, &mut o_v3, 2)?;
            e.stream().synchronize()?;
            let (m3, d3, per3) = cmp6(&o_ship, &o_v3)?;
            let mut t_v3 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut outs = mk_outs()?;
                let t0 = Instant::now();
                run(c, &mut outs, 2)?;
                e.stream().synchronize()?;
                t_v3.push(t0.elapsed().as_secs_f64() * 1e6);
            }
            report_arm(
                "qmatvec_kda6_bf16f32_v3 (cp.async staged)",
                "v3",
                ship_us,
                median(&mut t_v3),
                bytes,
                m3,
                d3,
            );
            println!("  per-projection mismatch counts (v3): {per3}");
        } else {
            println!("  qmatvec_kda6_bf16f32_v3: SKIPPED (v3 smem over the 48 KB default cap)");
        }
        println!(
            "  (kda6 bytes: {:.1} MB bf16 + {:.1} MB f32; shipped GB/s above is over the sum)",
            bf_bytes as f64 / 1e6,
            f32_bytes as f64 / 1e6
        );
    }

    // -----------------------------------------------------------------------------------
    // Families 7-9: the W8 POSTURE. These are the kernels that actually serve t=1 decode and
    // the t<=8 verify walk once MEMRA_GLM5_W8 is on (PR #86), which is the posture the pair
    // serves — the bf16 families above are unreachable there. See the lane doc's section 11.
    // GLM-5.3-Flash KDA shapes; weights are synthesised as bf16 and pushed through the SAME
    // encode + rp4 split the W8 mirror uses, so the bytes under test are the real mirror form.
    // -----------------------------------------------------------------------------------
    for (in_f, out_f, label) in [
        (4096usize, 8192usize, "kda up 4096->8192"),
        (8192, 4096, "kda down 8192->4096"),
    ] {
        let nblk = in_f / 32;
        let row_bytes = nblk * 34;
        let h_bf = synth_bf16(out_f, in_f, 0x0BAD_C0DE);
        let x = synth_f32(in_f, 0x0FED_CBA9);
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        // one bf16 source per copy -> one distinct rp4 mirror per copy (cold L2 per arm)
        let mirrors: Vec<CudaSlice<u8>> = (0..copies)
            .map(|_| -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
                let src = e.htod_bytes(&h_bf)?;
                let mut inter = e.alloc_u8_uninit(out_f * row_bytes)?;
                e.encode_q8_0_from_bf16(&src, &mut inter, in_f, out_f)?;
                e.build_q8_rp4_raw(&inter, in_f, out_f)
            })
            .collect::<Result<_, _>>()?;
        let bytes = (out_f * row_bytes) as f64;

        // Family 7: t=1 trunk. shipped qmatvec_q8_0_mmvq_rp vs the v2 twin.
        {
            let mut y_ship = e.zeros(out_f)?;
            e.qmatvec_mmvq_into(
                &mirrors[0],
                &aq,
                &ad,
                1,
                in_f,
                out_f,
                QT_Q8_0,
                row_bytes,
                1.0,
                true,
                &mut y_ship,
            )?;
            let mut y_v2 = e.zeros(out_f)?;
            e.qmatvec_q8_0_rp_v2_raw(&mirrors[0], &aq, &ad, &mut y_v2, in_f, out_f, 1)?;
            e.stream().synchronize()?;
            let h_ship = e.dtoh(&y_ship)?;
            let (mism, maxd) = compare(&h_ship, &e.dtoh(&y_v2)?);
            let mut t_ship = Vec::with_capacity(iters);
            let mut t_v2 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut y = e.zeros(out_f)?;
                let t0 = Instant::now();
                e.qmatvec_mmvq_into(
                    &mirrors[c],
                    &aq,
                    &ad,
                    1,
                    in_f,
                    out_f,
                    QT_Q8_0,
                    row_bytes,
                    1.0,
                    true,
                    &mut y,
                )?;
                e.stream().synchronize()?;
                t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

                let mut y = e.zeros(out_f)?;
                let t1 = Instant::now();
                e.qmatvec_q8_0_rp_v2_raw(&mirrors[c], &aq, &ad, &mut y, in_f, out_f, 1)?;
                e.stream().synchronize()?;
                t_v2.push(t1.elapsed().as_secs_f64() * 1e6);
            }
            report(
                &format!("qmatvec_q8_0_mmvq_rp (v2) {label}"),
                median(&mut t_ship),
                median(&mut t_v2),
                bytes,
                mism,
                maxd,
            );
        }

        // Family 8: verify width t=8. shipped qmatvec_q8_0_rows_tw vs the v2 twin.
        {
            let tv = 8usize;
            let xt = synth_f32(tv * in_f, 0x1234_ABCD);
            let xtd = e.htod(&xt)?;
            let (aqt, adt) = e.quantize_q8_1(&xtd, tv, in_f)?;
            let mut y_ship = e.zeros(tv * out_f)?;
            e.qmatvec_q8_0_rows_tw_arm_raw(
                &mirrors[0],
                &aqt,
                &adt,
                &mut y_ship,
                in_f,
                out_f,
                tv,
                false,
            )?;
            let mut y_v2 = e.zeros(tv * out_f)?;
            e.qmatvec_q8_0_rows_tw_arm_raw(
                &mirrors[0],
                &aqt,
                &adt,
                &mut y_v2,
                in_f,
                out_f,
                tv,
                true,
            )?;
            e.stream().synchronize()?;
            let (mism, maxd) = compare(&e.dtoh(&y_ship)?, &e.dtoh(&y_v2)?);
            let mut t_ship = Vec::with_capacity(iters);
            let mut t_v2 = Vec::with_capacity(iters);
            for i in 0..iters {
                let c = i % copies;
                let mut y = e.zeros(tv * out_f)?;
                let t0 = Instant::now();
                e.qmatvec_q8_0_rows_tw_arm_raw(
                    &mirrors[c],
                    &aqt,
                    &adt,
                    &mut y,
                    in_f,
                    out_f,
                    tv,
                    false,
                )?;
                e.stream().synchronize()?;
                t_ship.push(t0.elapsed().as_secs_f64() * 1e6);

                let mut y = e.zeros(tv * out_f)?;
                let t1 = Instant::now();
                e.qmatvec_q8_0_rows_tw_arm_raw(
                    &mirrors[c],
                    &aqt,
                    &adt,
                    &mut y,
                    in_f,
                    out_f,
                    tv,
                    true,
                )?;
                e.stream().synchronize()?;
                t_v2.push(t1.elapsed().as_secs_f64() * 1e6);
            }
            report(
                &format!("qmatvec_q8_0_rows_tw (v2) t=8 {label}"),
                median(&mut t_ship),
                median(&mut t_v2),
                bytes,
                mism,
                maxd,
            );
        }
    }

    // Family 9: the FUSED W8 six-projection group. Reference = the three mirrored projections
    // launched SEPARATELY through the v2 kernel (what the W8 path does today, one launch each
    // plus one redundant activation quantize each); arm = one fused launch that also covers the
    // three f32 low-rank/beta ranges. Identity is checked on y0..y2, where both sides run the
    // same per-row program; the f32 ranges are the fused kernel's pinned warp-tree class and
    // have no unfused counterpart here.
    {
        let in_f = 4096usize;
        let dims = [8192usize, 8192, 8192, 128, 128, 64];
        let nblk = in_f / 32;
        let row_bytes = nblk * 34;
        let h_bf = synth_bf16(dims[0], in_f, 0x5EED_1234);
        let x = synth_f32(in_f, 0x4321_DEEF);
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        let f32w: Vec<CudaSlice<f32>> = (3..6)
            .map(|k| e.htod(&synth_f32(dims[k] * in_f, 0xB0B0 + k as u32)))
            .collect::<Result<_, _>>()?;
        let bf_copies: Vec<[CudaSlice<u8>; 3]> = (0..copies)
            .map(
                |_| -> Result<[CudaSlice<u8>; 3], Box<dyn std::error::Error>> {
                    Ok([
                        e.htod_bytes(&h_bf)?,
                        e.htod_bytes(&h_bf)?,
                        e.htod_bytes(&h_bf)?,
                    ])
                },
            )
            .collect::<Result<_, _>>()?;
        // separate rp4 mirrors for the unfused reference (the fused entry builds its own,
        // keyed on the bf16 source pointer)
        let sep: Vec<[CudaSlice<u8>; 3]> = (0..copies)
            .map(
                |c| -> Result<[CudaSlice<u8>; 3], Box<dyn std::error::Error>> {
                    let mut out = Vec::new();
                    for k in 0..3 {
                        let mut inter = e.alloc_u8_uninit(dims[k] * row_bytes)?;
                        e.encode_q8_0_from_bf16(&bf_copies[c][k], &mut inter, in_f, dims[k])?;
                        out.push(e.build_q8_rp4_raw(&inter, in_f, dims[k])?);
                    }
                    let Ok(m3): Result<[CudaSlice<u8>; 3], _> = out.try_into() else {
                        return Err("expected exactly 3 mirrors".into());
                    };
                    Ok(m3)
                },
            )
            .collect::<Result<_, _>>()?;
        let mk_outs = || -> Result<[CudaSlice<f32>; 6], Box<dyn std::error::Error>> {
            Ok([
                e.zeros(dims[0])?,
                e.zeros(dims[1])?,
                e.zeros(dims[2])?,
                e.zeros(dims[3])?,
                e.zeros(dims[4])?,
                e.zeros(dims[5])?,
            ])
        };
        let unfused =
            |c: usize, outs: &mut [CudaSlice<f32>; 6]| -> Result<(), Box<dyn std::error::Error>> {
                for k in 0..3 {
                    e.qmatvec_q8_0_rp_v2_raw(&sep[c][k], &aq, &ad, &mut outs[k], in_f, dims[k], 1)?;
                }
                Ok(())
            };
        let fused =
            |c: usize, outs: &mut [CudaSlice<f32>; 6]| -> Result<(), Box<dyn std::error::Error>> {
                let b = &bf_copies[c];
                e.kda_proj_fused6_q8rp_raw(
                    &b[0], &b[1], &b[2], &f32w[0], &f32w[1], &f32w[2], &xd, outs, in_f, dims, 1,
                )
            };

        let mut o_sep = mk_outs()?;
        unfused(0, &mut o_sep)?;
        let mut o_fus = mk_outs()?;
        fused(0, &mut o_fus)?;
        e.stream().synchronize()?;
        let mut mism = 0usize;
        let mut maxd = 0f32;
        let mut per = Vec::new();
        for k in 0..3 {
            let (m, d) = compare(&e.dtoh(&o_sep[k])?, &e.dtoh(&o_fus[k])?);
            mism += m;
            maxd = maxd.max(d);
            per.push(format!("y{k}={m}"));
        }

        let mut t_sep = Vec::with_capacity(iters);
        let mut t_fus = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            let mut outs = mk_outs()?;
            let t0 = Instant::now();
            unfused(c, &mut outs)?;
            e.stream().synchronize()?;
            t_sep.push(t0.elapsed().as_secs_f64() * 1e6);

            let mut outs = mk_outs()?;
            let t1 = Instant::now();
            fused(c, &mut outs)?;
            e.stream().synchronize()?;
            t_fus.push(t1.elapsed().as_secs_f64() * 1e6);
        }
        let bytes = (3 * dims[0] * row_bytes) as f64;
        report(
            "kda6 W8: 3x separate -> qmatvec_kda6_q8f32_rp_v2",
            median(&mut t_sep),
            median(&mut t_fus),
            bytes,
            mism,
            maxd,
        );
        println!(
            "  per-projection mismatch counts (q8 ranges): {}  (the fused arm additionally \
             covers the three f32 ranges the separate reference does not launch)",
            per.join(" ")
        );
    }

    println!("temp_out: {}", gpu_temp());
    Ok(())
}
