//! mla-coalesce-bench: structural-profiling target for the MLA absorb/decompress kernel family
//! (lane/glm5-b200, 2026-09-04). NOT a gate and NOT a timing instrument: it exists so `ncu` can
//! read sectors-per-request, occupancy and stall reasons for the SHIPPED thread-per-row kernels,
//! the decode-SPLIT twins, and the warp-per-row COALESCED twins (`MEMRA_MLA_COALESCE`) on one
//! process at the served GLM-5.3-Flash geometry (64 heads, kv_lora_rank 512, qk_nope_head_dim
//! 256, v_head_dim 256). `mla-decode-arm-gate` calls the raw launchers directly and cannot route
//! the coalesce door, which is why this bin exists. Random operands; outputs are not compared.
//!
//! usage: mla-coalesce-bench [device=0] [iters=1]
use memra_engine::Engine;

fn randf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let e = Engine::new(dev)?;
    let (t_q, n_head, d_nope, kv_rank, d_v) = (1usize, 64usize, 256usize, 512usize, 256usize);
    let q_nope = e.htod(&randf(t_q * n_head * d_nope, 1))?;
    let wk_b = e.htod(&randf(n_head * kv_rank * d_nope, 2))?;
    let mut q_lat = e.htod(&vec![0f32; t_q * n_head * kv_rank])?;
    let o_lat = e.htod(&randf(t_q * n_head * kv_rank, 3))?;
    let wv_b = e.htod(&randf(n_head * d_v * kv_rank, 4))?;
    let mut out = e.htod(&vec![0f32; t_q * n_head * d_v])?;
    for _ in 0..iters {
        // shipped (thread per output row), split=16 (grid only), wp split=1, wp split=16
        e.mla_absorb_q_raw_arm(
            &q_nope, &wk_b, &mut q_lat, t_q, n_head, d_nope, kv_rank, 0, 1,
        )?;
        e.mla_absorb_q_raw_arm(
            &q_nope, &wk_b, &mut q_lat, t_q, n_head, d_nope, kv_rank, 1, 16,
        )?;
        e.mla_absorb_q_raw_arm(
            &q_nope, &wk_b, &mut q_lat, t_q, n_head, d_nope, kv_rank, 2, 1,
        )?;
        e.mla_absorb_q_raw_arm(
            &q_nope, &wk_b, &mut q_lat, t_q, n_head, d_nope, kv_rank, 2, 16,
        )?;
        e.mla_decompress_v_raw_arm(&o_lat, &wv_b, &mut out, t_q, n_head, d_v, kv_rank, 0, 1)?;
        e.mla_decompress_v_raw_arm(&o_lat, &wv_b, &mut out, t_q, n_head, d_v, kv_rank, 1, 8)?;
        e.mla_decompress_v_raw_arm(&o_lat, &wv_b, &mut out, t_q, n_head, d_v, kv_rank, 2, 1)?;
        e.mla_decompress_v_raw_arm(&o_lat, &wv_b, &mut out, t_q, n_head, d_v, kv_rank, 2, 8)?;
    }
    // hc pre-chain, served geometry: hc=4 streams, d=4096, one row (t=1), sinkhorn iters=20.
    let (hc, d, rows_mix, it) = (4usize, 4096usize, (2 + 4) * 4usize, 20usize);
    let hx = e.htod(&randf(hc * d, 5))?;
    let hmix = e.htod(&randf(rows_mix, 6))?;
    let hscale = e.htod(&[0.5f32, 0.5, 0.5])?;
    let hbase = e.htod(&randf(rows_mix, 7))?;
    let mut hpre = e.htod(&vec![0f32; hc])?;
    let mut hpost = e.htod(&vec![0f32; hc])?;
    let mut hcomb = e.htod(&vec![0f32; hc * hc])?;
    let mut hy = e.htod(&vec![0f32; d])?;
    let mut rounds = [0i32; 3];
    for _ in 0..iters {
        for (arm, blk) in [(0u8, 128i32), (1, 512), (2, 512)] {
            let mut nit = e.htod_i32(&[0i32])?;
            e.hc_pre_raw_arm(
                &hx,
                &hmix,
                &hscale,
                &hbase,
                &mut hpre,
                &mut hpost,
                &mut hcomb,
                &mut hy,
                1,
                hc,
                d,
                it,
                1e-6,
                arm,
                blk,
                Some(&mut nit),
            )?;
            e.stream().synchronize()?;
            rounds[arm as usize] = e.dtoh_i32(&nit)?[0];
        }
    }
    e.stream().synchronize()?;
    println!(
        "hc pre-chain Sinkhorn rounds actually run (of {it}): v2@128={} v3@512-shared={} v3@512-registers={}  (early exit fires if < {it})",
        rounds[0], rounds[1], rounds[2]
    );
    println!(
        "mla-coalesce-bench: {iters} iteration(s) of 8 MLA + 3 hc-pre launches on device {dev}; profile with ncu"
    );
    Ok(())
}
