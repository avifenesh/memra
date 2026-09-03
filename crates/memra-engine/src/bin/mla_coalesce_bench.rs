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
    e.stream().synchronize()?;
    println!(
        "mla-coalesce-bench: {iters} iteration(s) of 8 launches on device {dev}; profile with ncu"
    );
    Ok(())
}
