//! OPTIPIPE increment-1 forced state-fork and increment-2 controller gate.
//!
//! Usage: optipipe-gate <model.gguf> <off|hit|miss|alternate|abort|ring|stream|controller>
//! Environment: MEMRA_MTP_DRAFT, MEMRA_NGEN (default 64), MEMRA_OPTI_STRESS_NGEN
//! (alternate default 256), MEMRA_OPTI_CAP (optional session capacity),
//! MEMRA_OPTI_CONTROLLER_Q (controller threshold, default 0), and MEMRA_OPTI_PROMPT
//! (optional real-text prompt; MEMRA_CHAT wraps it in the model chat template). PP-2
//! placement is supplied through the normal MEMRA_PP_* env.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::spec::{
    OptiForkGateMode, optipipe_gate_stats, reset_optipipe_gate_stats,
    set_optipipe_controller_threshold, set_optipipe_gate_mode,
};
use memra_gguf::GgufFile;

fn parse_mode(s: &str) -> Result<OptiForkGateMode, String> {
    match s {
        "off" => Ok(OptiForkGateMode::Disabled),
        "hit" | "ring" | "stream" => Ok(OptiForkGateMode::Hit),
        "miss" => Ok(OptiForkGateMode::Miss),
        "alternate" => Ok(OptiForkGateMode::Alternate),
        "abort" => Ok(OptiForkGateMode::Abort),
        "controller" => Ok(OptiForkGateMode::Controller),
        _ => Err(format!(
            "unknown mode {s:?}; expected off|hit|miss|alternate|abort|ring|stream|controller"
        )),
    }
}

/// The pinned Step artifact currently identifies itself as step35 and therefore has no live GDN
/// buffers. Exercise the exact conditional f32 primitive used for recurrent state and seed restore
/// directly so that a zero-byte model-backed recurrent comparison cannot leave the miss arm dark.
fn recurrent_restore_primitive_gate(e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 1 << 18;
    const BONUS: u32 = 777;
    let snapshot: Vec<f32> = (0..N).map(|i| (i as f32) * 0.25 - 32768.0).collect();
    let mutated: Vec<f32> = (0..N).map(|i| 16384.0 - (i as f32) * 0.125).collect();
    let snapshot_d = e.htod(&snapshot)?;

    for (name, acc_h, expected_valid, expected) in [
        ("hit", [1u32, BONUS], 1u32, &mutated),
        ("count-miss", [0u32, BONUS], 0u32, &snapshot),
        ("bonus-miss", [1u32, BONUS + 1], 0u32, &snapshot),
    ] {
        let acc_d = e.htod_u32_v(&acc_h)?;
        let mut valid_d = e.alloc_u32_zeroed(1)?;
        let mut state_d = e.htod(&mutated)?;
        e.spec_fork_valid(&acc_d, BONUS, &mut valid_d)?;
        e.spec_fork_restore_f32(&snapshot_d, &mut state_d, &valid_d)?;
        let valid = e.dtoh_u32_one(&valid_d)?;
        let state = e.dtoh(&state_d)?;
        if valid != expected_valid || state != *expected {
            return Err(format!(
                "recurrent restore primitive {name} failed: valid={valid} expected={expected_valid}"
            )
            .into());
        }
    }
    println!("RECURRENT RESTORE PRIMITIVE: PASS elements={N} hit-skip=count/bonus-miss-restore");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or(
        "usage: optipipe-gate <model.gguf> <off|hit|miss|alternate|abort|ring|stream|controller>",
    )?;
    let mode_name = args.next().ok_or(
        "usage: optipipe-gate <model.gguf> <off|hit|miss|alternate|abort|ring|stream|controller>",
    )?;
    if args.next().is_some() {
        return Err("optipipe-gate accepts exactly two arguments".into());
    }
    if mode_name == "ring" && std::env::var("MEMRA_SWA_RING").as_deref() != Ok("1") {
        return Err("ring mode requires MEMRA_SWA_RING=1 before model load".into());
    }
    if mode_name == "stream" && std::env::var("MEMRA_SPEC_STREAM").as_deref() != Ok("1") {
        return Err("stream mode requires MEMRA_SPEC_STREAM=1 before model load".into());
    }
    let mode = parse_mode(&mode_name)?;
    let controller_q = if mode == OptiForkGateMode::Controller {
        let threshold = std::env::var("MEMRA_OPTI_CONTROLLER_Q")
            .unwrap_or_else(|_| "0".into())
            .parse::<f32>()
            .map_err(|err| format!("invalid MEMRA_OPTI_CONTROLLER_Q: {err}"))?;
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "MEMRA_OPTI_CONTROLLER_Q={threshold} is outside the inclusive [0,1] range"
            )
            .into());
        }
        Some(threshold)
    } else {
        None
    };
    let n_new: usize = if mode == OptiForkGateMode::Alternate {
        std::env::var("MEMRA_OPTI_STRESS_NGEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    } else {
        std::env::var("MEMRA_NGEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    };

    let e = Engine::new(0)?;
    let gguf = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &gguf)?;
    if model.mtp.is_none() {
        return Err("model has no MTP head; set MEMRA_MTP_DRAFT for the pinned draft GGUF".into());
    }
    let prompt: Vec<u32> = if let Ok(text) = std::env::var("MEMRA_OPTI_PROMPT") {
        let tok = memra_tokenizer::Tokenizer::from_gguf(&gguf)?;
        let rendered = if std::env::var_os("MEMRA_CHAT").is_some() {
            tok.apply_chat_template(&[("user", text.as_str())], true)
        } else {
            text
        };
        let ids = tok.encode(&rendered, true);
        if ids.is_empty() {
            return Err("MEMRA_OPTI_PROMPT tokenized to an empty prompt".into());
        }
        eprintln!("[opti-gate] real prompt tokens={}", ids.len());
        ids
    } else {
        (101..229).collect()
    };
    let min_cap = prompt.len() + n_new * 3 + 128;
    let cap = std::env::var("MEMRA_OPTI_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(min_cap);
    if cap < min_cap {
        return Err(
            format!("MEMRA_OPTI_CAP={cap} is below required gate capacity {min_cap}").into(),
        );
    }

    set_optipipe_gate_mode(OptiForkGateMode::Disabled);
    let mut reference = model.new_session(&e, cap)?;
    let (reference_out, _, _) =
        model.generate_spec_session(&e, &mut reference, &prompt, n_new, 1)?;

    if mode == OptiForkGateMode::Abort {
        reset_optipipe_gate_stats();
        set_optipipe_gate_mode(mode);
        let mut aborted = model.new_session(&e, cap)?;
        let err = model
            .generate_spec_session(&e, &mut aborted, &prompt, n_new, 1)
            .expect_err("forced abort unexpectedly completed")
            .to_string();
        let abort_stats = optipipe_gate_stats();
        if abort_stats.attempts != 1 || abort_stats.abort_drains != 1 {
            return Err(format!("abort teardown counters wrong: {abort_stats:?}").into());
        }
        println!("ABORT DRAIN: PASS error={err:?} stats={abort_stats:?}");
        // Prove the CUDA contexts and shared PP boundary runtime remain usable after teardown.
        reset_optipipe_gate_stats();
        set_optipipe_gate_mode(OptiForkGateMode::Hit);
    } else {
        reset_optipipe_gate_stats();
        if let Some(threshold) = controller_q {
            set_optipipe_controller_threshold(threshold);
        } else {
            set_optipipe_gate_mode(mode);
        }
    }

    let mut candidate = model.new_session(&e, cap)?;
    let (candidate_out, _, _) =
        model.generate_spec_session(&e, &mut candidate, &prompt, n_new, 1)?;
    if candidate_out != reference_out {
        let at = candidate_out
            .iter()
            .zip(&reference_out)
            .position(|(a, b)| a != b)
            .unwrap_or(candidate_out.len().min(reference_out.len()));
        return Err(format!(
            "forced {mode_name} output mismatch at {at}: candidate={:?} reference={:?}",
            candidate_out.get(at),
            reference_out.get(at),
        )
        .into());
    }
    let first = model.optipipe_compare_session_state(&e, &reference, &candidate)?;
    let stats = optipipe_gate_stats();
    match mode_name.as_str() {
        "hit" if stats.attempts == 0 || stats.hits != stats.attempts || stats.misses != 0 => {
            return Err(format!("forced-hit liveness failed: {stats:?}").into());
        }
        "miss" if stats.attempts == 0 || stats.misses != stats.attempts || stats.hits != 0 => {
            return Err(format!("forced-miss liveness failed: {stats:?}").into());
        }
        "alternate" if stats.attempts < 16 || stats.hits == 0 || stats.misses == 0 => {
            return Err(format!("alternating stress liveness failed: {stats:?}").into());
        }
        "ring" | "stream" if stats.attempts != 0 || stats.refusals == 0 => {
            return Err(format!("{mode_name} exclusion failed: {stats:?}").into());
        }
        "controller"
            if stats.gate_checks == 0
                || stats.gate_checks != stats.gate_admits + stats.gate_rejects
                || stats.attempts != stats.gate_admits
                || stats.attempts != stats.hits + stats.misses + stats.abort_drains
                || stats.reconciles != stats.misses =>
        {
            return Err(format!("controller liveness/accounting failed: {stats:?}").into());
        }
        "off" if stats != Default::default() => {
            return Err(
                format!("disabled baseline unexpectedly touched fork state: {stats:?}").into(),
            );
        }
        _ => {}
    }

    // Continue from both states with the diagnostic disabled. Matching the next burst proves
    // the compared state is not merely an output-prefix coincidence.
    set_optipipe_gate_mode(OptiForkGateMode::Disabled);
    let (reference_next, _, _) = model.generate_spec_session(&e, &mut reference, &[], 16, 1)?;
    let (candidate_next, _, _) = model.generate_spec_session(&e, &mut candidate, &[], 16, 1)?;
    if candidate_next != reference_next {
        return Err(format!(
            "post-{mode_name} serial continuation mismatch: candidate={candidate_next:?} \
             reference={reference_next:?}",
        )
        .into());
    }
    let continued = model.optipipe_compare_session_state(&e, &reference, &candidate)?;
    recurrent_restore_primitive_gate(&e)?;
    println!(
        "STATE IDENTITY: PASS mode={mode_name} q={controller_q:?} cap={cap} generated={} continuation={} \
         first={first:?} continued={continued:?} stats={stats:?}",
        candidate_out.len(),
        candidate_next.len(),
    );
    Ok(())
}
