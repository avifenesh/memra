//! DECODE KERNEL CENSUS (2026-08-25, the 90 tok/s goal): times the bf16 GEMV kernels the
//! step37 TP2 decode tick actually spends its bytes in, at their PER-CARD serving shapes,
//! and converts each into "ms per token over 45 layers" plus achieved weight bandwidth.
//!
//! Why a standalone bench: the banked verdict for single-stream was "eager GPU-bound at 76,
//! kernels at/near roofline", but a byte census disagrees — ~6.5 GB per card per token at
//! 1.79 TB/s is a 3.6 ms floor against a 13.2 ms measured token. This binary decides which
//! of the two is wrong by measuring the kernels alone, with no launch chain, no attention
//! and no routing around them. It is a SPEED receipt only; every arm here is bit-exact
//! (identical per-row FP order), so nothing it measures is a numeric-class question.
//!
//! usage: decode-kernel-census [--reps N]
use memra_engine::Engine;

const N_LAYERS: f64 = 45.0;
const PEAK_TBS: f64 = 1.79;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--reps")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(300);
    let e = Engine::new(0)?;
    println!("decode-kernel-census reps={reps} peak={PEAK_TBS} TB/s layers={N_LAYERS}");

    // QKV+gate, fused, PER CARD under STEP_TP=0-44@0,1: n_embd 4096 in; 32 local heads x
    // 128 = 4096 q out; 4 local kv heads x 128 = 512 each for k and v; 32 gate rows.
    qkvg(&e, reps, 4096, 4096, 512, 32)?;
    // o_proj, HEAD_SPLIT 4 blocks: 4 x 1024 local columns -> 4096 rows.
    b4(&e, reps, 1024, 4096)?;
    // Shared-expert down (the short-row shape MEMRA_DOWN_X4 exists for).
    plain(&e, reps, 1280, 4096, "shexp down 1280->4096")?;
    // The head, split across the pair: 4096 -> 128896/2.
    plain(&e, reps, 4096, 64448, "head 4096->64448 (HEAD_SPLIT half)")?;
    // SIZING THE ONE DOOR THAT REACHES 90 (2026-08-25): q/k/v/o are bf16 and are ~3.4 GB of
    // the ~6.5 GB this card streams per token. These q8_0 rows are the SPEED half of that
    // decision — same shapes, half the weight bytes. Numerics are NOT claimed here; a q8
    // attention-projection class would need its own argmax gate and owner ratification.
    q8(&e, reps, 4096, 5152, "q8_0 qkv-equivalent 4096->5152")?;
    q8(&e, reps, 4096, 4096, "q8_0 o_proj-equivalent 4096->4096")?;
    // The other two lines of the same budget, so none of it stays an estimate.
    q8(
        &e,
        reps,
        1280,
        4096,
        "q8_0 shexp-down-equivalent 1280->4096",
    )?;
    q8(&e, reps, 4096, 64448, "q8_0 head-equivalent 4096->64448")?;
    // The budget's last ASSUMED line: the NVFP4 expert sweep sits at 0.80 TB/s while every
    // wide member of this family is at 1.5-1.8. This row runs the same byte volume through
    // the q8 mmvq path at the stacked all-8-experts gate+up shape, which bounds how much of
    // that gap is the shape and how much is the NVFP4 unpack itself.
    q8(
        &e,
        reps,
        4096,
        20480,
        "q8_0 expert gate+up stacked 4096->20480",
    )?;
    Ok(())
}

/// q8_0 GEMV at an attention-projection shape: the bf16 twin of the same shape reads 2 bytes
/// per weight, this reads 1 plus a 2-byte scale per 32 (34/32 bytes per 32 weights).
fn q8(
    e: &Engine,
    reps: usize,
    in_f: usize,
    out_f: usize,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    const QK: usize = 32;
    let row_bytes = in_f / QK * (QK + 2);
    let w = e.alloc_u8(out_f * row_bytes)?;
    // q8_1 activation: one i8 per column plus a per-32-block scale pair.
    let aq = e.htod_i8(&vec![1i8; in_f])?;
    let ad = e.htod(&vec![0.01f32; 2 * in_f / QK])?;
    let mut y = e.htod(&vec![0f32; out_f])?;
    let mut run = || -> Result<(), Box<dyn std::error::Error>> {
        e.qmatvec_mmvq_into(
            &w,
            &aq,
            &ad,
            1,
            in_f,
            out_f,
            memra_engine::QT_Q8_0,
            row_bytes,
            1.0,
            true,
            &mut y,
        )
    };
    for _ in 0..30 {
        run()?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    // Same rule as the bf16 arm: the head runs once per token, everything else per layer.
    let per_layer = if out_f > 60000 { 1.0 / N_LAYERS } else { 1.0 };
    report(label, us, (out_f * row_bytes) as f64, per_layer);
    Ok(())
}

fn report(label: &str, us: f64, bytes: f64, per_token_calls: f64) {
    let tbs = bytes / us / 1e6;
    println!(
        "[{label}] {us:8.1} us/call  {tbs:5.2} TB/s ({:4.1}% of peak)  {:6.2} ms/token @ {per_token_calls} call(s)/layer",
        100.0 * tbs / PEAK_TBS,
        us * per_token_calls * N_LAYERS / 1000.0
    );
}

fn qkvg(
    e: &Engine,
    reps: usize,
    in_f: usize,
    out_q: usize,
    out_kv: usize,
    out_g: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let rows = out_q + 2 * out_kv + out_g;
    let wq = e.alloc_u8(out_q * in_f * 2)?;
    let wk = e.alloc_u8(out_kv * in_f * 2)?;
    let wv = e.alloc_u8(out_kv * in_f * 2)?;
    let wg = e.alloc_u8(out_g * in_f * 2)?;
    let x = e.htod(&vec![0.01f32; in_f])?;
    let mut yq = e.htod(&vec![0f32; out_q])?;
    let mut yk = e.htod(&vec![0f32; out_kv])?;
    let mut yv = e.htod(&vec![0f32; out_kv])?;
    let mut yg = e.htod(&vec![0f32; out_g])?;
    let mut run = || -> Result<(), Box<dyn std::error::Error>> {
        e.matvec_bf16_qkvg_into(
            &wq, &wk, &wv, &wg, &x, &mut yq, &mut yk, &mut yv, &mut yg, in_f, out_q, out_kv, out_g,
        )
    };
    for _ in 0..30 {
        run()?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    report(
        &format!("qkvg {in_f}->{rows} (per card)"),
        us,
        (rows * in_f * 2) as f64,
        1.0,
    );
    Ok(())
}

fn b4(
    e: &Engine,
    reps: usize,
    block_cols: usize,
    out_f: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let w: Vec<_> = (0..4)
        .map(|_| e.alloc_u8(out_f * block_cols * 2))
        .collect::<Result<_, _>>()?;
    let x = e.htod(&vec![0.01f32; 4 * block_cols])?;
    let mut y = e.htod(&vec![0f32; out_f])?;
    let mut run = || -> Result<(), Box<dyn std::error::Error>> {
        e.matvec_bf16_b4_into([&w[0], &w[1], &w[2], &w[3]], &x, &mut y, block_cols, out_f)
    };
    for _ in 0..30 {
        run()?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    report(
        &format!("b4 o_proj 4x{block_cols}->{out_f}"),
        us,
        (4 * out_f * block_cols * 2) as f64,
        1.0,
    );
    Ok(())
}

fn plain(
    e: &Engine,
    reps: usize,
    in_f: usize,
    out_f: usize,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let w = e.alloc_u8(in_f * out_f * 2)?;
    let x = e.htod(&vec![0.01f32; in_f])?;
    let mut y = e.htod(&vec![0f32; out_f])?;
    let mut run = || -> Result<(), Box<dyn std::error::Error>> {
        e.matvec_bf16_into(&w, &x, &mut y, in_f, out_f)
    };
    for _ in 0..30 {
        run()?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        run()?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    // The head runs once per token, not once per layer.
    let per_layer = if out_f > 60000 { 1.0 / N_LAYERS } else { 1.0 };
    report(label, us, (in_f * out_f * 2) as f64, per_layer);
    Ok(())
}
