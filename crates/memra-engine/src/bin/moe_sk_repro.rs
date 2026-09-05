//! MOE SK REPRO / RATE PROBE.
//!
//! Born 2026-08-27 to reproduce the f16g_sk "invalid argument" at the exact failing geometry the
//! [moe-sk-err] receipt named; values are irrelevant to a launch-geometry failure, shapes are
//! everything. Rebuilt 2026-08-28 into the standalone RATE probe for the prime's MoE wall.
//!
//! Why it exists in this form. In the server the grouped MoE is 44% of a 4096-token prime chunk
//! and [grp-prof] attributes 93% of that to GPU time, so it is not host churn. Nine kernel-shape
//! levers (tile form, occupancy, padding, B double-buffering, spills, prefix truncation, host
//! router, host CSR, host issue) all came back null. ncu would name the stall directly but this
//! box refuses GPU performance counters (ERR_NVGPUCTRPERM, a host driver setting), so the
//! decomposition has to come from timing the real shapes instead.
//!
//! Two things the first timed version got wrong, both fixed here:
//!   * its synthetic router put a fifth of all pairs on one expert (max_m=6639 against a mean of
//!     114), which is nothing like production's max_m~934 over 284 active experts — and tile
//!     padding, the thing under suspicion, is entirely a function of that distribution.
//!   * it timed only the gate shape (in_f=4096, out_f=640). Production runs THREE grouped calls
//!     per layer per rank, and the down projection is the odd one (in_f=640, out_f=4096), whose
//!     64 n-tiles per expert make it a different regime.
//!
//! usage: moe-sk-repro [reps]
use memra_engine::Engine;

fn gate_modelopt_split_layout(e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;

    let (n_expert, rows, k, pairs) = (2usize, 64usize, 128usize, 128usize);
    let mut mo_q = vec![0u8; n_expert * rows * (k / 2)];
    let mut mo_s = vec![0u8; n_expert * rows * (k / 16)];
    let mut gg = vec![0u8; n_expert * rows * (k / 64 * 36)];
    for ex in 0..n_expert {
        for row in 0..rows {
            let q0 = (ex * rows + row) * (k / 2);
            let s0 = (ex * rows + row) * (k / 16);
            let g0 = (ex * rows + row) * (k / 64 * 36);
            for sb in 0..k / 16 {
                let sc = 8 + ((ex * 17 + row * 3 + sb * 5) % 80) as u8;
                mo_s[s0 + sb] = sc;
                let gb = g0 + (sb / 4) * 36;
                gg[gb + sb % 4] = sc;
                for j in 0..16 {
                    let code = ((ex * 11 + row * 7 + sb * 3 + j * 5) % 16) as u8;
                    let kidx = sb * 16 + j;
                    let mb = q0 + kidx / 2;
                    if j & 1 == 0 {
                        mo_q[mb] = (mo_q[mb] & 0xf0) | code;
                    } else {
                        mo_q[mb] = (mo_q[mb] & 0x0f) | (code << 4);
                    }
                    let qb = gb + 4 + (sb % 4) * 8 + (j & 7);
                    if j < 8 {
                        gg[qb] = (gg[qb] & 0xf0) | code;
                    } else {
                        gg[qb] = (gg[qb] & 0x0f) | (code << 4);
                    }
                }
            }
        }
    }
    let mo_q_d = e.htod_bytes(&mo_q)?;
    let mo_s_d = e.htod_bytes(&mo_s)?;
    let gg_d = e.htod_bytes(&gg)?;
    let stream = e.stream();
    let (mqp, _g0) = mo_q_d.device_ptr(&stream);
    let (msp, _g1) = mo_s_d.device_ptr(&stream);
    let (ggp, _g2) = gg_d.device_ptr(&stream);
    let mut mo_tab = vec![0u64; 6 * n_expert];
    let mut gg_tab = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        let mq = mqp + (ex * rows * (k / 2)) as u64;
        let ms = msp + (ex * rows * (k / 16)) as u64;
        let gp = ggp + (ex * rows * (k / 64 * 36)) as u64;
        for proj in 0..3 {
            mo_tab[(2 * proj) * n_expert + ex] = mq;
            mo_tab[(2 * proj + 1) * n_expert + ex] = ms;
            gg_tab[proj * n_expert + ex] = gp;
        }
    }
    let mo_tab_d = e.htod_u64(&mo_tab)?;
    let gg_tab_d = e.htod_u64(&gg_tab)?;
    let ex_ids = e.htod_i32(&[0, 1])?;
    let ex_off_h = [0i32, 64, 128];
    let ex_off = e.htod_i32(&ex_off_h)?;
    let x: Vec<f32> = (0..pairs * k)
        .map(|i| (((i * 37 + 11) % 257) as f32 - 128.0) / 64.0)
        .collect();
    let x_d = e.htod(&x)?;
    let (x16, xs) = e.moe_f16g_act(&x_d, None, k, pairs)?;
    let mo = e.moe_f16_grouped(
        &mo_tab_d,
        0,
        n_expert,
        &ex_ids,
        &ex_off_h,
        &ex_off,
        &x16,
        &xs,
        k,
        rows,
        n_expert,
        pairs,
        memra_engine::QT_NVFP4_MODELOPT,
        k / 2,
    )?;
    let gg = e.moe_f16_grouped(
        &gg_tab_d,
        0,
        n_expert,
        &ex_ids,
        &ex_off_h,
        &ex_off,
        &x16,
        &xs,
        k,
        rows,
        n_expert,
        pairs,
        memra_engine::QT_NVFP4,
        k / 64 * 36,
    )?;
    let (a, b) = (e.dtoh(&mo)?, e.dtoh(&gg)?);
    let mismatches = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    if mismatches != 0 {
        return Err(format!(
            "ModelOpt split-layout twin mismatches={mismatches}/{}",
            a.len()
        )
        .into());
    }
    println!(
        "modelopt-split-layout gate: BIT-EXACT vs interleaved NVFP4, {} outputs",
        a.len()
    );
    Ok(())
}

/// Deterministic production-shaped routing: mean m_e ~ n_pairs/n_expert with a long tail, so
/// n_active and max_m land near the [moe-sk-form] receipt (n_active=284, max_m=934) instead of
/// the degenerate single-hot-expert shape. Pure function of the pair index — no RNG, so every
/// run of this probe measures the identical distribution.
fn route(p: usize, n_expert: usize) -> i32 {
    let mut h = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    // u in [0,1); u^1.35 tilts mass toward the low expert ids and gives a ~8x hot/mean ratio.
    let u = (h >> 11) as f64 / (1u64 << 53) as f64;
    let x = u.powf(1.35);
    ((x * n_expert as f64) as usize).min(n_expert - 1) as i32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let e = Engine::new(0)?;
    let dsv4_shape = std::env::var("MEMRA_DSV4_MOE_SHAPE").as_deref() == Ok("1");
    let n_expert = if dsv4_shape { 256usize } else { 288usize };
    let n_used = if dsv4_shape { 6usize } else { 8usize };
    let hidden = 4096usize;
    let local_ff = if dsv4_shape { 2_048usize } else { 640usize };
    if dsv4_shape {
        gate_modelopt_split_layout(&e)?;
    }

    // One synthetic bank per projection shape. NVFP4 stores 64 values per 36-byte block.
    let mk_bank = |in_f: usize, out_f: usize| -> Result<_, Box<dyn std::error::Error>> {
        let row_bytes = if dsv4_shape { in_f / 2 } else { in_f / 64 * 36 };
        let bank = e.alloc_u8(n_expert * out_f * row_bytes)?;
        let scales = e.alloc_u8(if dsv4_shape {
            n_expert * out_f * (in_f / 16)
        } else {
            1
        })?;
        let mut tab = vec![0u64; if dsv4_shape { 6 } else { 3 } * n_expert];
        {
            use cudarc::driver::DevicePtr;
            let stream = e.stream();
            let (p, _g) = bank.device_ptr(&stream);
            let (ps, _gs) = scales.device_ptr(&stream);
            for ex in 0..n_expert {
                let a = p + (ex * out_f * row_bytes) as u64;
                if dsv4_shape {
                    let s = ps + (ex * out_f * (in_f / 16)) as u64;
                    for proj in 0..3 {
                        tab[(2 * proj) * n_expert + ex] = a;
                        tab[(2 * proj + 1) * n_expert + ex] = s;
                    }
                } else {
                    tab[ex] = a;
                    tab[n_expert + ex] = a;
                    tab[2 * n_expert + ex] = a;
                }
            }
        }
        Ok((bank, scales, e.htod_u64(&tab)?, row_bytes))
    };
    // gate and up share a shape; down transposes it.
    let (_bg, _sg, tab_gu, rb_gu) = mk_bank(hidden, local_ff)?;
    let (_bd, _sd, tab_d, rb_d) = mk_bank(local_ff, hidden)?;
    let qtype = if dsv4_shape {
        memra_engine::QT_NVFP4_MODELOPT
    } else {
        memra_engine::QT_NVFP4
    };

    println!(
        "shape: family={} hidden={hidden} local_ff={local_ff} n_expert={n_expert} top_k={n_used} reps={reps}",
        if dsv4_shape { "dsv4" } else { "step37-tp2" }
    );
    let widths: &[usize] = if dsv4_shape {
        &[32, 128, 512, 1024, 2048, 4096]
    } else {
        &[512, 2048, 4096]
    };
    for &t in widths {
        let n_pairs = t * n_used;
        let sel: Vec<i32> = (0..n_pairs).map(|p| route(p, n_expert)).collect();
        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
        for (p, &s_id) in sel.iter().enumerate() {
            buckets[s_id as usize].push(p as i32);
        }
        let mut ex_ids = Vec::new();
        let mut ex_off = vec![0i32];
        let mut ex_pairs: Vec<i32> = Vec::new();
        for (id, b) in buckets.iter().enumerate() {
            if !b.is_empty() {
                ex_ids.push(id as i32);
                ex_pairs.extend_from_slice(b);
                ex_off.push(ex_pairs.len() as i32);
            }
        }
        let n_active = ex_ids.len();
        let ms: Vec<i64> = ex_off.windows(2).map(|w| (w[1] - w[0]) as i64).collect();
        let max_m = ms.iter().copied().max().unwrap_or(0);
        let mean_m = ms.iter().sum::<i64>() as f64 / n_active.max(1) as f64;
        // Tile padding the 128-row form pays on THIS distribution: rows processed / rows useful.
        let padded: i64 = ms.iter().map(|&m| (m + 127) / 128 * 128).sum();
        let pad = padded as f64 / ms.iter().sum::<i64>().max(1) as f64;
        let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
        let csr_tok_d = e.htod_i32(&csr_tok)?;
        let exi_d = e.htod_i32(&ex_ids)?;
        let exoff_d = e.htod_i32(&ex_off)?;
        println!(
            "\nt={t} n_pairs={n_pairs} n_active={n_active} max_m={max_m} mean_m={mean_m:.0} \
             pad128={pad:.2}x"
        );

        // gate/up gather from the token rows (csr_tok); down consumes pair-major rows already.
        let z = e.htod(&vec![0.01f32; t * hidden])?;
        let (z16, zs) = e.moe_f16g_act(&z, Some(&csr_tok_d), hidden, n_pairs)?;
        let a = e.htod(&vec![0.01f32; n_pairs * local_ff])?;
        let (a16, a_s) = e.moe_f16g_act(&a, None, local_ff, n_pairs)?;

        let mut total_ms = 0.0f64;
        for (name, tab, rb, in_f, out_f, act, sc) in [
            ("gate", &tab_gu, rb_gu, hidden, local_ff, &z16, &zs),
            ("up  ", &tab_gu, rb_gu, hidden, local_ff, &z16, &zs),
            ("down", &tab_d, rb_d, local_ff, hidden, &a16, &a_s),
        ] {
            let call = || {
                e.moe_f16_grouped(
                    tab, 0, n_expert, &exi_d, &ex_off, &exoff_d, act, sc, in_f, out_f, n_active,
                    n_pairs, qtype, rb,
                )
            };
            let r = call();
            e.stream().synchronize()?;
            if let Err(err) = &r {
                println!("  {name}: ERR {err}");
                continue;
            }
            let useful: f64 = n_pairs as f64 * out_f as f64 * in_f as f64 * 2.0;
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                let _ = call();
                e.stream().synchronize()?;
                best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            }
            total_ms += best;
            let ntx = out_f.div_ceil(64);
            println!(
                "  {name} in_f={in_f:<5} out_f={out_f:<5} ntx={ntx:<3} best={best:6.2}ms \
                 useful={:6.1}GFLOP rate={:6.1}TFLOP/s",
                useful / 1e9,
                useful / (best * 1e-3) / 1e12
            );
        }
        println!("  layer-total (gate+up+down, one rank) = {total_ms:.2}ms");
    }
    Ok(())
}
