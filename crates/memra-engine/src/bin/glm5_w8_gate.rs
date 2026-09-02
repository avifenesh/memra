//! glm5-w8-gate: acceptance instrument for `MEMRA_GLM5_W8` (the q8_0 decode mirror for the
//! glm5_next KDA/MLA bf16-resident trunk — docs/FLAGS.md, lane/b200-glm5-w8-20260902).
//!
//! TWO INDEPENDENT PROBES, printed together:
//!
//! 1. SYNTHETIC PER-SHAPE PROBE (always runs; safe on any card, incl. the local RTX 5090 rig
//!    — no real checkpoint needed, exactness-only per the rig's role). For each of a handful
//!    of representative KDA/MLA projection shapes (taken from the glm5_next residency census
//!    in docs/FLAGS.md's `MEMRA_BF16_MMV` acceptance row: "34 x kda_q/k/v/out at 33.5M
//!    elements", "12 x indexer.attn_q_b at 6.3M elements"): build a RANDOM bf16 weight
//!    matrix, its q8_0 mirror (the SAME `encode_q8_0_from_bf16` + `build_q8_rp4_raw` path the
//!    engine's decode-time mirror cache uses), and for N random f32 activation vectors run
//!    BOTH matvec classes — `Engine::matvec_bf16_into` (the unmirrored class
//!    `matvec_bf16_f32acc_x4_rows` runs) and the q8_0 mirror class (`quantize_q8_1_into` +
//!    `qmatvec_mmvq_into`, `QT_Q8_0`, `rp=true` — the exact call `matvec_bf16_via_q8_mirror`
//!    makes internally). Reports max abs error and argmax agreement over the N activations,
//!    per shape. This is a WIRING sanity probe, not a quantization-tightness bound: random
//!    unstructured weights are not the model's trained distribution, so it fails only on
//!    NaN/Inf or an error large enough to indicate a block-layout bug, never on ordinary
//!    quantization noise.
//!
//! 2. REAL-ARTIFACT ARGMAX-TAPE PROBE (only when `GLM5_ARTIFACT` is set — box-only; the real
//!    checkpoint does not fit a one-card rig). Loads the real glm5_next model TWICE, in two
//!    FRESH child processes (`MEMRA_GLM5_W8` is read into a per-process `OnceLock`, so an
//!    in-process toggle would not be a fresh boot per arm — the pin-against-truth law), once
//!    with the door off and once on, and greedy-decodes 32 tokens from the SAME fixed prompt
//!    in each. Reports the token-by-token argmax agreement between the two tapes.
//!
//! Usage:
//!   glm5-w8-gate [N]                                   synthetic probe only (N random
//!                                                       activations/shape, default 200)
//!   GLM5_ARTIFACT=<safetensors dir | .gguf> glm5-w8-gate [N]
//!                                                       synthetic probe + real-artifact tape
//!
//! Internal worker mode (`GLM5_W8_GATE_WORKER=1`, set by the orchestrator's re-exec, not by a
//! caller): loads `GLM5_ARTIFACT` under the process's own `MEMRA_GLM5_W8` and prints one
//! `GLM5_W8_TAPE <csv token ids>` line.

use memra_engine::Engine;

/// Census-representative glm5_next KDA/MLA projection shapes (in_f, out_f, label). Sourced
/// from docs/FLAGS.md's `MEMRA_BF16_MMV` GLM5_NEXT ACCEPTANCE row, not measured here.
const SHAPES: &[(usize, usize, &str)] = &[
    // "34 x kda_q/k/v/out at 33.5M elements" (8192*4096 = 33,554,432)
    (4096, 8192, "kda_qkvo_4096x8192"),
    (8192, 4096, "kda_qkvo_8192x4096"),
    // "12 x indexer.attn_q_b at 6.3M elements" (4096*1536 = 6,291,456)
    (4096, 1536, "mla_indexer_attn_q_b_4096x1536"),
    // a small gate/beta-class width, well below the trunk's dominant shapes
    (2048, 128, "small_gate_2048x128"),
];

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform f32 in [-1, 1). No external RNG dependency — this is a wiring probe, not a
/// statistical one, so a simple xorshift stream is sufficient and keeps the bin dependency-free.
fn rand_f32(state: &mut u64) -> f32 {
    let bits = (xorshift64(state) >> 40) as u32; // top 24 bits
    (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

/// f32 -> bf16 by truncation (upper 16 bits of the IEEE-754 bit pattern), the same
/// round-toward-zero rule the checkpoint loader's bf16 admission uses.
fn f32_to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for &f in v {
        let hi = (f.to_bits() >> 16) as u16;
        out.extend_from_slice(&hi.to_le_bytes());
    }
    out
}

fn synthetic_probe(e: &Engine, n: usize) -> Result<bool, Box<dyn std::error::Error>> {
    println!("=== MEMRA_GLM5_W8 synthetic per-shape probe (N={n} random activations/shape) ===");
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut ok = true;
    for &(in_f, out_f, label) in SHAPES {
        let w_host: Vec<f32> = (0..in_f * out_f).map(|_| rand_f32(&mut seed)).collect();
        let w_bytes = f32_to_bf16_bytes(&w_host);
        let w_bf16 = e.upload_u8(&w_bytes)?;

        // q8_0 mirror: identical sequence to `Engine::matvec_bf16_via_q8_mirror`'s build step.
        let mut interleaved = e.alloc_u8_uninit(out_f * Engine::q8_0_row_bytes(in_f))?;
        e.encode_q8_0_from_bf16(&w_bf16, &mut interleaved, in_f, out_f)?;
        let mirror = e.build_q8_rp4_raw(&interleaved, in_f, out_f)?;

        let mut max_abs_err = 0f32;
        let mut argmax_matches = 0usize;
        let mut nan_or_inf = false;
        let nblk = in_f / 32;
        for _ in 0..n {
            let x_host: Vec<f32> = (0..in_f).map(|_| rand_f32(&mut seed)).collect();
            let x_d = e.htod(&x_host)?;

            let mut y_bf16 = e.uninit(out_f)?;
            e.matvec_bf16_into(&w_bf16, &x_d, &mut y_bf16, in_f, out_f)?;
            let y_bf16_host = e.dtoh(&y_bf16)?;

            let mut aq = e.alloc_i8_uninit(in_f)?;
            let mut ad = e.uninit(nblk)?;
            e.quantize_q8_1_into(&x_d, 1, in_f, &mut aq, &mut ad)?;
            let mut y_q8 = e.uninit(out_f)?;
            e.qmatvec_mmvq_into(
                &mirror,
                &aq,
                &ad,
                1,
                in_f,
                out_f,
                memra_engine::QT_Q8_0,
                Engine::q8_0_row_bytes(in_f),
                1.0,
                true,
                &mut y_q8,
            )?;
            let y_q8_host = e.dtoh(&y_q8)?;

            let mut am_bf16 = 0usize;
            let mut am_q8 = 0usize;
            let mut best_bf16 = f32::NEG_INFINITY;
            let mut best_q8 = f32::NEG_INFINITY;
            for i in 0..out_f {
                let a = y_bf16_host[i];
                let b = y_q8_host[i];
                if !a.is_finite() || !b.is_finite() {
                    nan_or_inf = true;
                }
                let d = (a - b).abs();
                if d > max_abs_err {
                    max_abs_err = d;
                }
                if a > best_bf16 {
                    best_bf16 = a;
                    am_bf16 = i;
                }
                if b > best_q8 {
                    best_q8 = b;
                    am_q8 = i;
                }
            }
            if am_bf16 == am_q8 {
                argmax_matches += 1;
            }
        }
        let agree_pct = 100.0 * argmax_matches as f64 / n as f64;
        println!(
            "shape={label} in_f={in_f} out_f={out_f} max_abs_err={max_abs_err:.6e} \
             argmax_agree={argmax_matches}/{n} ({agree_pct:.1}%) nan_or_inf={nan_or_inf}"
        );
        if nan_or_inf || !max_abs_err.is_finite() || max_abs_err > 50.0 {
            println!(
                "  -> WIRING FAILURE on shape {label} (NaN/Inf or a block-layout-sized error)"
            );
            ok = false;
        }
    }
    Ok(ok)
}

fn worker_main(artifact: &str) -> Result<(), Box<dyn std::error::Error>> {
    use memra_engine::cache::Cache;
    use memra_engine::forward::argmax;
    use memra_engine::hybrid::HybridModel;

    let e = Engine::new(0)?;
    let path = std::path::Path::new(artifact);
    let model = if path.is_dir() {
        let src: Box<dyn memra_gguf::source::TensorSource> = if path.join("manifest.json").exists()
        {
            Box::new(memra_gguf::source::Hy3RepackSource::open(path)?)
        } else {
            Box::new(memra_gguf::source::SafetensorsSource::open(path)?)
        };
        HybridModel::load_from_source_without_mtp(&e, src.as_ref())?
    } else {
        let g = memra_gguf::GgufFile::open(artifact)?;
        HybridModel::load_without_mtp(&e, &g)?
    };

    const PROMPT_LEN: usize = 16;
    const GEN_LEN: usize = 32;
    // Deterministic fixed prompt (argmax-gate's convention), so both arms see byte-identical
    // input tokens — the ONLY thing allowed to differ between the two child processes is
    // MEMRA_GLM5_W8 itself.
    let prompt: Vec<u32> = (0..PROMPT_LEN)
        .map(|i| (100 + (i * 7) % 900) as u32)
        .collect();
    let mut cache = Cache::new(&e, &model.cfg, PROMPT_LEN + GEN_LEN + 8)?;
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = model.decode_step(&e, t, &mut cache)?;
    }
    let mut tape = Vec::with_capacity(GEN_LEN);
    let mut tok = argmax(&logits) as u32;
    for _ in 0..GEN_LEN {
        tape.push(tok);
        logits = model.decode_step(&e, tok, &mut cache)?;
        tok = argmax(&logits) as u32;
    }
    let csv: Vec<String> = tape.iter().map(|t| t.to_string()).collect();
    println!("GLM5_W8_TAPE {}", csv.join(","));
    Ok(())
}

fn run_worker(
    exe: &std::path::Path,
    artifact: &str,
    w8: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let out = std::process::Command::new(exe)
        .env("GLM5_W8_GATE_WORKER", "1")
        .env("GLM5_ARTIFACT", artifact)
        .env("MEMRA_GLM5_W8", w8)
        .env("NVIDIA_TF32_OVERRIDE", "0")
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "worker (MEMRA_GLM5_W8={w8}) exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("GLM5_W8_TAPE ") {
            return Ok(rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect());
        }
    }
    Err(format!(
        "worker (MEMRA_GLM5_W8={w8}) printed no GLM5_W8_TAPE line; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    )
    .into())
}

fn real_artifact_probe(artifact: &str) -> Result<bool, Box<dyn std::error::Error>> {
    println!("=== MEMRA_GLM5_W8 real-artifact 32-token greedy tape (fresh boot per arm) ===");
    let exe = std::env::current_exe()?;
    let off = run_worker(&exe, artifact, "0")?;
    let on = run_worker(&exe, artifact, "1")?;
    println!("door-off tape ({} tokens): {:?}", off.len(), off);
    println!("door-on  tape ({} tokens): {:?}", on.len(), on);
    let n = off.len().min(on.len());
    let matches = off.iter().zip(on.iter()).filter(|(a, b)| a == b).count();
    let agree_pct = 100.0 * matches as f64 / n.max(1) as f64;
    println!("argmax agreement: {matches}/{n} ({agree_pct:.1}%)");
    if matches != n || off.len() != 32 || on.len() != 32 {
        println!("  -> MISMATCH: door-off and door-on greedy tapes diverge (or a tape is short)");
    }
    Ok(matches == n && off.len() == 32 && on.len() == 32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(200);

    if std::env::var("GLM5_W8_GATE_WORKER").as_deref() == Ok("1") {
        let artifact = std::env::var("GLM5_ARTIFACT")
            .map_err(|_| "GLM5_W8_GATE_WORKER=1 requires GLM5_ARTIFACT")?;
        return worker_main(&artifact);
    }

    let e = Engine::new(0)?;
    let synth_ok = synthetic_probe(&e, n)?;
    drop(e);

    let artifact_ok = match std::env::var("GLM5_ARTIFACT") {
        Ok(artifact) => Some(real_artifact_probe(&artifact)?),
        Err(_) => {
            println!(
                "GLM5_ARTIFACT not set: skipping the real-artifact greedy-tape probe (box-only \
                 — the checkpoint does not fit a one-card rig). The synthetic probe above is \
                 the only receipt from this run."
            );
            None
        }
    };

    let ok = synth_ok && artifact_ok.unwrap_or(true);
    if ok {
        println!("glm5-w8-gate: PASS");
    } else {
        println!("glm5-w8-gate: FAIL");
        std::process::exit(1);
    }
    Ok(())
}
