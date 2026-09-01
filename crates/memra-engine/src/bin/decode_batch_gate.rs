//! decode-batch-gate: the batched decode step's exactness battery (ARCHITECTURE-H100.md §3 B2').
//!
//! decode_step_batch is a NEW NUMERIC CONFIG vs decode_step_h's fused m=1 tier: the fused
//! path folds q8_1 scales as a separable post-op (matmul_pre_noscale + silu_mul_scaled),
//! which is m==1-only by construction; the batched path folds scales inside the matvec
//! (matmul_pre) + plain silu_mul. Same math, different FP composition — the GDN-chunked
//! prefill precedent. PROOF the plumbing is exact: under the equalized composition
//! (MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1, both paths on dp4a + unfused norms) the battery is
//! BIT-IDENTICAL at B=1 and B=N-vs-isolated (verified 2026-07-26 on H100).
//!
//! STRICT-MODE EQUALIZATION NOW COVERS NVFP4 (lane/nvfp4-strict, 2026-08-05). History: the
//! equalizing env was Q8/dp4a-shaped — MEMRA_MMVQ=0 steered the Q8_0-class arms (their
//! fused launches all sit behind `q8_fused_params`, which refuses under MMVQ=0 by the
//! FP-order law) but the NVFP4 gate+up/beta+alpha pair door (`matmul_pre_dual_noscale`)
//! had no such check: the oracle kept dispatching `qmatvec_nvfp4_mmvq_dual_mr2` (MMVQ
//! 32-thread warp-reduce family) while the batched side fell to dp4a (128-thread
//! two-level reduce) — a mixed-family comparison, so `--mode strict` FAILED on ANY NVFP4
//! model at pristine trees (train-HEAD receipts: gate1 maxdiff 1.639e-1 @ step 2 on q9,
//! research/servepath-p2-20260805/logs/dbg-strict-b4-TRAINHEAD.log; gate2 step-6 token
//! divergence on q27 at 93420980, research/nvfp4-strict-20260805/repro.log). The fix
//! applies the SAME law to the NVFP4 arm: `matmul_pre_dual_noscale` returns None when
//! `mmvq_supports(QT_NVFP4)` is false, so under the equalized env BOTH sides ride dp4a
//! and strict bit-identity holds (default env is dispatch-unchanged). A strict FAIL on an
//! NVFP4 model is a REAL failure again.
//!
//! Modes (--mode, default "config"):
//!   strict — opt-in eager-path bit-identity gates; run with MEMRA_SERVE_B1FAST=1 and
//!            under the EQUALIZED env or expect gate1 bit-diffs:
//!     gate1: eager B=1 logits bit-identical to decode_step_h, every step.
//!     gate2: B=N per-seq streams == isolated decode_step_h streams (argmax).
//!   config — the live default battery (one generic batched program at every width):
//!     gate1: N/A while B1FAST is off. With explicit B1FAST=1, calibrates the eager B=1
//!            argmax stream vs decode_step_h over 6 prompt draws.
//!     gate2: B=N per-seq LOGITS bit-identical to isolated decode_step_batch B=1 runs —
//!            the serving isolation contract (batchmates must not change your stream),
//!            enforced at full bit strength WITHIN the config.
//!   pp     — THE BATCHED STAGE-SPLIT EXACTNESS GATE (pp2-batch 2026-08-06). See
//!            `pp_battery` below. Opens `MEMRA_PP_STAGES` BEFORE load (weight sharding is
//!            a load-time decision), then proves `decode_step_batch_ppn` is BIT-IDENTICAL
//!            to the unsplit batched body over the same weights, per row, per step.
//!            gate1/2/3 are skipped in this mode — they are single-device jurisdiction and
//!            run in their own invocations.
//!   ppspec — THE SPEC-VERIFY STAGE-SPLIT EXACTNESS GATE (pp2-spec 2026-08-06). Same
//!            method, different forward: `decode_step_t_core_ppn` (T = K+1 verify) must be
//!            BIT-IDENTICAL to the unsplit verify trunk, per logit COLUMN, per round, plus
//!            the h_seed column the drafter is re-seeded from. See `ppspec_battery`.
//!
//! Usage: decode-batch-gate <model.gguf> [--steps 32] [--batch 4]
//!        [--mode config|strict|pp|ppspec]
//!        pp mode also honours: --batch 1,4,8 (list), --stages N (default 2), --reps R
//!        (default 2 — the split arm repeats, because the shared-scratch race class this
//!        gate must catch was a 35% FLAKE, so one green replay is not evidence), and passes
//!        MEMRA_PP_DEVICES / MEMRA_PP_SPLITS / MEMRA_PP_SHARD through from the caller.
//!        ppspec mode honours --stages/--reps/--steps the same way and takes --ts 2,5,9
//!        (the verify widths T=K+1; default 2,5,9 = K=1,4,8).

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::decode_batch::{DevPenalty, DevSamp};
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;
use sha2::{Digest, Sha256};

/// Restore a process environment variable when a temporary in-process gate arm exits. Rust drops
/// the guard on both `Ok` and `?`, so a failed decode cannot leak a diagnostic route into the next
/// arm.
struct EnvVarRestore {
    key: &'static str,
    value: Option<std::ffi::OsString>,
}

impl EnvVarRestore {
    fn set(key: &'static str, value: &str) -> Self {
        let restore = Self {
            key,
            value: std::env::var_os(key),
        };
        // SAFETY: decode-batch-gate owns its process environment. PP wave workers are scoped
        // inside the guarded decode call and finish before this guard restores the variable.
        unsafe {
            std::env::set_var(key, value);
        }
        restore
    }
}

impl Drop for EnvVarRestore {
    fn drop(&mut self) {
        // SAFETY: see `set`; no gate worker outlives the decode call guarded by this value.
        unsafe {
            match &self.value {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: decode-batch-gate <model.gguf> [--steps N] [--batch B]");
    let rest: Vec<String> = args.collect();
    let steps: usize = rest
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    // --batch takes a comma list in pp mode (one battery per width in ONE process, so all
    // widths are measured against the SAME loaded weights); the other modes use the first.
    let batches: Vec<usize> = rest
        .iter()
        .position(|a| a == "--batch")
        .and_then(|i| rest.get(i + 1))
        .map(|v| {
            v.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![4]);
    let b_n: usize = batches[0];
    let mode: String = rest
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| rest.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "config".into());
    let strict: bool = mode == "strict";
    let rewrite_receipt_path = std::env::var_os("MEMRA_REWRITE_RECEIPT");
    let ppspec_mode: bool = mode == "ppspec";
    // Both stage-split modes need the door open BEFORE load; only the battery differs.
    let pp_mode: bool = mode == "pp" || ppspec_mode;
    // ppspec: the verify widths. T = K+1, so 2,5,9 covers K=1,4,8 — the same K range the
    // run-spec self-consistency gate walks, and 9 crosses the t>=3 batched-linear window.
    let ts: Vec<usize> = rest
        .iter()
        .position(|a| a == "--ts")
        .and_then(|i| rest.get(i + 1))
        .map(|v| {
            v.split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![2, 5, 9]);
    let stages: usize = rest
        .iter()
        .position(|a| a == "--stages")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let reps: usize = rest
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    // --plen: base synthetic-prompt length (default 20 = the historic prompts, row i adds
    // 5i). Exists for SWA archs (lane/step35-batched-decode): step35's window is 512, so
    // the 20-token prompts leave every session INSIDE the window and the per-session view
    // offset (`off = len - win`, the mechanism the batched arm adds) never fires — the
    // chunkinv35 lesson (a gate whose prompts sit inside the window compares one kernel
    // against itself). The step35 battery passes --plen 520 so row 0 crosses the window
    // during decode and later rows start past it.
    let plen: u32 = rest
        .iter()
        .position(|a| a == "--plen")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    // PIN THE PRIME CONFIG (2026-07-26): this gate compares DECODE configs
    // (decode_step_batch vs decode_step_h) from a shared primed state — the GDN prime
    // config is a nuisance variable here. The K4/K5-MMA prime (Hopper default) shifts
    // near-tie logits enough to flip the config-mode step-16 threshold on the fixed
    // prompt (observed: step-1 argmax flip; STRICT bit-gate and gate2 both still PASS,
    // proving decode itself is untouched). The mma prime's own correctness is covered
    // by its kernel-check pin + the state-carry battery + run-gen argmax gates.
    // SAFETY: single-threaded gate binary; the GDN seam reads the env per call.
    unsafe {
        std::env::set_var("MEMRA_GDN_MMA", "0");
    }
    // Same rationale for the l2 prefill v2 config (round 27): its primed state shifts
    // the same near-tie logits at step 1. Gate tests DECODE; prime stays pinned f32-class.
    unsafe {
        std::env::set_var("MEMRA_L2_V2", "0");
    }
    unsafe {
        std::env::set_var("MEMRA_FA3", "0");
    }
    unsafe {
        std::env::set_var("MEMRA_GDN_WGMMA", "0");
    }
    // Round 49: the grouped f16 expert-prefill door (Hopper default mode 1; sm_120a naked
    // default mode 2 since lane/f16g-default-rearb 2026-08-02 — "0" fully closes the door
    // under every mode semantics, so this pin is default-flip-invariant) is another PRIME
    // nuisance — same signature as the K4/K5-MMA precedent (gate1 seed flip at step 0,
    // gate2+gate3 bit-strength PASS, and pinning it off restores 6/6 seeds). The door's
    // own correctness is covered by kernel-check + run-gen argmax + run-spec gates.
    unsafe {
        std::env::set_var("MEMRA_MOE_F16G", "0");
    }
    // PP MODE OPENS THE DOOR BEFORE LOAD (ppn-gate's method): weight sharding is a
    // LOAD-TIME decision — `hybrid.rs` asks `pp::layer_engine` per layer, so a door opened
    // after load would test a split walk over unsharded weights, i.e. not the serving
    // placement at all. The primary device follows MEMRA_PP_DEVICES[0] for the same reason
    // (stage 0's engine IS the primary engine).
    let primary_dev: usize = if pp_mode {
        unsafe {
            std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
        }
        std::env::var("MEMRA_PP_DEVICES")
            .ok()
            .and_then(|v| v.split(',').next().and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0)
    } else {
        0
    };
    let e = Engine::new(primary_dev)?;
    // DIRECTORY path = safetensors HF checkpoint (lane/rp-on-st, 2026-08-06). This gate was
    // GGUF-only, and `Mmap::map` on a directory fd returns ENODEV — so pointing it at an ST
    // checkpoint died with `Os { code: 19, ... "No such device" }` BEFORE loading anything,
    // which reads like a GPU-acquisition failure and is not one. That mattered: the exact-16
    // serve tier now admits both FP8-ST scale classes (QT_F8_E4M3 via the new b16 twin,
    // QT_F8_E4M3_BLK via its first batched family), and the ONLY model-level exactness gate
    // for decode_step_batch could not be pointed at either of them. Same branch as run_gen.
    let (model, arch) = if std::path::Path::new(&path).is_dir() {
        let dir = std::path::Path::new(&path);
        let src: Box<dyn memra_gguf::source::TensorSource> = if dir.join("manifest.json").exists() {
            Box::new(memra_gguf::source::Hy3RepackSource::open(dir)?)
        } else {
            Box::new(memra_gguf::source::SafetensorsSource::open(dir)?)
        };
        let m = HybridModel::load_from_source_without_mtp(&e, src.as_ref())?;
        (m, "safetensors".to_string())
    } else {
        let g = GgufFile::open(&path)?;
        let a = g.arch().unwrap_or("?").to_string();
        (HybridModel::load_without_mtp(&e, &g)?, a)
    };
    println!(
        "loaded {} ({} layers); steps={steps} batch={b_n}",
        arch,
        model.layers.len()
    );

    if pp_mode {
        let seed: u32 = std::env::var("MEMRA_GATE_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if ppspec_mode {
            let fails = ppspec_battery(&e, &model, stages, steps, &ts, reps, seed)?;
            if fails == 0 {
                println!("ALL GREEN: spec-verify PP-{stages} stage-split exactness battery");
                return Ok(());
            }
            return Err("decode-batch-gate --mode ppspec FAILED".into());
        }
        let fails = pp_battery(&e, &model, stages, steps, &batches, reps, seed, plen)?;
        if fails == 0 {
            println!("ALL GREEN: batched PP-{stages} stage-split exactness battery");
            return Ok(());
        }
        return Err("decode-batch-gate --mode pp FAILED".into());
    }

    // Distinct prompts per lane so caches/states genuinely diverge; length >= 16
    // (PRIME_MIN_T floor) and deliberately uneven so positions differ across the batch.
    // MEMRA_GATE_SEED offsets the token pattern — the cross-config drift class is a
    // near-tie roulette on any single synthetic prompt, so calibration sweeps need
    // several draws (the 2026-07-31 shexp-dot re-sweep; default 0 = the historic prompt).
    let seed: u32 = std::env::var("MEMRA_GATE_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let prompts: Vec<Vec<u32>> = (0..b_n.max(2))
        .map(|i| {
            (0..plen + i as u32 * 5)
                .map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31)
                .collect()
        })
        .collect();
    let ctx = 512 + steps + 64;

    // ---- Gate 1: B=1 vs decode_step_h ----
    // strict: bit-identity on the seed prompt (run under the EQUALIZED env).
    // config: MULTI-SEED calibration. The two decode configs carry an ACCEPTED
    // FP-composition gap, and on any single synthetic prompt the first argmax divergence
    // is a near-tie roulette — the H100 re-sweep (2026-07-31, the shexp-dot dig) failed
    // the old single-prompt step-16 rule on 3/6 draws (steps 7/8/15), and its
    // replacement ("FAIL iff ANY seed diverges before step 3") assumed tie flips start
    // at step 6+ — an H100-only observation. The 5090 re-sweep (2026-08-02,
    // research/gate1-recal-20260802/: 18 draws x {q9j Q8_0, q35 IQ4_XS}) saw legal dice
    // at steps 0/1/3/4 (q35 seeds 16/17 flip at step 0; q9j seed 0 at step 1), each
    // PROVEN dice by bit-identity under the equalized strict env on the very same draws.
    // The per-draw step threshold carries no rig-invariant signal; the FRACTION does:
    // plumbing (wrong token fed, KV misindexed) diverges at step 0-2 on EVERY draw,
    // observed dice reach at most 2 early draws per 6-window. FAIL iff >= G1_EARLY_K of
    // the 6 draws diverge before step G1_EARLY_STEP — margin 2 above the observed dice
    // maximum, margin 2 below the plumbing floor (6/6). Teeth verified by the
    // MEMRA_GATE_CANARY=1 wrong-token canary (must FAIL 6/6). Strict gate1 + gate2 +
    // gate3 keep full bit strength — they remain the hard exactness floor.
    const G1_EARLY_STEP: usize = 3; // plumbing window: wrong token/KV shows at step 0-2
    const G1_EARLY_K: usize = 4; // FAIL iff this many draws diverge inside the window
    let canary = std::env::var("MEMRA_GATE_CANARY")
        .map(|v| v == "1")
        .unwrap_or(false);
    // The live B=1 contract is now the batched body for every architecture, specifically to
    // prevent a serving row changing numeric class as peers arrive or retire. The default-config
    // eager comparison is therefore not a serving gate; gate2 is the load-transition proof at
    // bit strength. Explicit B1FAST=1 keeps the eager calibration reachable. Strict mode still
    // runs gate1 under its equalized kernel composition, and the wrong-token canary still runs so
    // the default policy cannot hide broken token/cache plumbing.
    let b1_fast_configured = HybridModel::b1_fast_on();
    let g1_live_eager_inapplicable =
        !strict && !canary && !(b1_fast_configured && model.b1_fast_plan_eligible());
    let mut g1_fail = 0usize;
    let mut g1_early = 0usize;
    let g1_seeds: u32 = if g1_live_eager_inapplicable {
        0
    } else if strict {
        1
    } else {
        6
    };
    for gs in 0..g1_seeds {
        let p0: Vec<u32> = (0..20).map(|j| 55 + (seed + gs) * 13 + j * 31).collect();
        let mut c_ref = Cache::new(&e, &model.cfg, ctx)?;
        let mut c_bat = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, &p0, &mut c_ref, 0)?;
        let _ = model.prime_cache(&e, &p0, &mut c_bat, 0)?;
        let mut t_ref = *p0.last().unwrap();
        let mut t_bat = t_ref;
        let mut diverged: Option<usize> = None;
        for s in 0..steps {
            // TEST-ONLY plumbing canary (MEMRA_GATE_CANARY=1): feed the batched lane one
            // wrong token — the class the fraction rule must keep catching (FAIL 6/6).
            if canary && s == 1 {
                t_bat = if t_bat == 0 { 1 } else { t_bat - 1 };
            }
            let (l_ref, _) = model.decode_step_h(&e, t_ref, &mut c_ref)?;
            let l_bat = {
                let mut caches = [&mut c_bat];
                model
                    .decode_step_batch(&e, &[t_bat], &mut caches)?
                    .remove(0)
            };
            if strict {
                let bits_equal = l_ref.len() == l_bat.len()
                    && l_ref
                        .iter()
                        .zip(l_bat.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                if !bits_equal {
                    let md = l_ref
                        .iter()
                        .zip(l_bat.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!("gate1 step {s}: BIT-DIFF (maxdiff {md:.3e}) FAIL");
                    g1_fail += 1;
                    if g1_fail > 3 {
                        break;
                    }
                }
            }
            t_ref = argmax(&l_ref) as u32;
            t_bat = argmax(&l_bat) as u32;
            if t_ref != t_bat {
                diverged = Some(s);
                break;
            }
        }
        match diverged {
            Some(s) if strict => {
                println!("gate1 seed {gs} step {s}: token diverged FAIL");
                g1_fail += 1;
            }
            Some(s) if s < G1_EARLY_STEP => {
                g1_early += 1;
                println!(
                    "gate1 seed {gs} step {s}: token diverged EARLY \
                          (step < {G1_EARLY_STEP}; plumbing iff every draw)"
                );
            }
            Some(s) => println!(
                "gate1 seed {gs} step {s}: token diverged — accepted \
                                 cross-config drift (WARN)"
            ),
            None => println!("gate1 seed {gs}: agreement all {steps} steps"),
        }
    }
    if g1_live_eager_inapplicable {
        println!(
            "gate1 (B=1 vs decode_step_h): N/A for the live default; B=1 uses the \
                  batched numeric class and gate2 checks B=1 vs B={b_n}"
        );
    } else if !strict {
        println!(
            "gate1 early draws (step < {G1_EARLY_STEP}): {g1_early}/{g1_seeds} \
                  (FAIL threshold >= {G1_EARLY_K}; plumbing class = every draw)"
        );
        if g1_early >= G1_EARLY_K {
            g1_fail += 1;
        }
        println!(
            "gate1 (B=1 argmax agreement vs decode_step_h, {steps} steps, \
                  {g1_seeds} seed(s)): {}",
            if g1_fail == 0 { "PASS" } else { "FAIL" }
        );
    } else {
        println!(
            "gate1 (B=1 bit-identity vs decode_step_h, {steps} steps, \
                  {g1_seeds} seed(s)): {}",
            if g1_fail == 0 { "PASS" } else { "FAIL" }
        );
    }

    // ---- Gate 2: B=N vs isolated (the serving isolation contract) ----
    // Reference = isolated runs of the SAME config: strict mode uses decode_step_h,
    // config mode uses decode_step_batch at B=1 — within-config, bit strength applies.
    //
    // Gate2 and gate3b pin the B=1 REFERENCE ARM to the generic batched body. Reason: an
    // explicitly enabled B=1 fast path routes solo sequences onto the m=1 FUSED trunk, so an
    // unpinned `decode_step_batch(&[t])` reference would no longer run the code these gates exist
    // to test — their bit/stream checks would silently degrade
    // from "batchmates must not perturb your logits" (the real teeth) into a cross-config
    // FP-composition comparison, which gate1's config mode already tolerates by design.
    // Pinning keeps their jurisdiction exactly where it was: the BATCHED m>=2 body.
    // The fast path's own exactness is gate1's job (STRICT gate1 = bit-identity to
    // decode_step_h, which PASSes ONLY with the fast path ON — verified on-box: OFF fails
    // at maxdiff 1.591e-1). The explicit seam lets one process cover either configured program
    // without a memoized environment read changing the reference arm.
    let b1_fast_setting = HybridModel::b1_fast_on();
    let b1_fast_live = b1_fast_setting && model.b1_fast_plan_eligible();
    HybridModel::set_b1_fast(false);
    println!(
        "gate2/gate3 B=1 reference arm: batched body (B=1 fast path pinned OFF; \
              global setting = {}; effective for this architecture = {})",
        if b1_fast_setting { "ON" } else { "OFF" },
        if b1_fast_live { "ON" } else { "OFF" }
    );
    let mut ref_streams: Vec<Vec<u32>> = Vec::new();
    let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    for p in prompts.iter().take(b_n) {
        let mut c = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, p, &mut c, 0)?;
        let mut t = *p.last().unwrap();
        let mut out = Vec::with_capacity(steps);
        let mut ls = Vec::with_capacity(steps);
        for _ in 0..steps {
            let l = if strict {
                model.decode_step_h(&e, t, &mut c)?.0
            } else {
                let mut caches = [&mut c];
                model.decode_step_batch(&e, &[t], &mut caches)?.remove(0)
            };
            t = argmax(&l) as u32;
            out.push(t);
            ls.push(l);
        }
        ref_streams.push(out);
        ref_logits.push(ls);
    }
    // Batched run over fresh caches primed identically.
    let mut caches: Vec<Cache> = Vec::new();
    for p in prompts.iter().take(b_n) {
        let mut c = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, p, &mut c, 0)?;
        caches.push(c);
    }
    let mut toks: Vec<u32> = prompts
        .iter()
        .take(b_n)
        .map(|p| *p.last().unwrap())
        .collect();
    let mut g2_fail = 0usize;
    let mut candidate_logits_flat = Vec::new();
    for s in 0..steps {
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let logits = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
        for (bi, l) in logits.iter().enumerate() {
            if rewrite_receipt_path.is_some() {
                candidate_logits_flat.extend_from_slice(l);
            }
            toks[bi] = argmax(l) as u32;
            if toks[bi] != ref_streams[bi][s] {
                println!("gate2 seq {bi}: token DIVERGED from isolated at step {s} FAIL");
                g2_fail += 1;
            } else if !strict {
                // within-config: batchmates must not perturb even one bit of your logits
                let r = &ref_logits[bi][s];
                if !(r.len() == l.len()
                    && r.iter()
                        .zip(l.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()))
                {
                    let md = r
                        .iter()
                        .zip(l.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "gate2 seq {bi} step {s}: LOGIT BIT-DIFF vs isolated \
                              (maxdiff {md:.3e}) FAIL"
                    );
                    g2_fail += 1;
                }
            }
        }
        if g2_fail > 6 {
            break;
        }
    }
    println!(
        "gate2 (B={b_n} vs isolated {}, {steps} steps): {}",
        if strict {
            "decode_step_h"
        } else {
            "batched-B=1, bit-checked"
        },
        if g2_fail == 0 { "PASS" } else { "FAIL" }
    );
    let rewrite_receipt = if g2_fail == 0 && rewrite_receipt_path.is_some() {
        let mut reference_logits_flat = Vec::with_capacity(candidate_logits_flat.len());
        for step in 0..steps {
            for rows in ref_logits.iter().take(b_n) {
                reference_logits_flat.extend_from_slice(&rows[step]);
            }
        }
        let rewrite = memra_engine::plan_backend::execution_rewrites(&model.plan)
            .into_iter()
            .find(|rewrite| {
                rewrite.surface == memra_engine::plan_backend::RewriteSurface::DecodeBatch
            })
            .ok_or("decode-batch rewrite manifest is missing")?;
        let executable = std::fs::read(std::env::current_exe()?)?;
        let executable_sha256 = Sha256::digest(&executable)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let receipt = rewrite.verify_logits(
            &executable_sha256,
            &reference_logits_flat,
            &candidate_logits_flat,
            memra_engine::plan_backend::RewriteParityPolicy {
                max_abs: 0.0,
                max_rel: 0.0,
                require_argmax: true,
            },
        )?;
        let receipt = memra_engine::plan_backend::bind_rewrite_artifact(receipt)?;
        receipt.validate_for(&rewrite)?;
        Some(receipt.to_tsv())
    } else {
        None
    };
    // 24GB-card capacity (inc3, 2026-08-01): free gate2's cache herd + host logits before
    // gate3 allocates its own (B=16 with the q8rp mirror OOM'd gate3 on the 5090 while
    // every verdict was green — harness footprint, not model state).
    drop(caches);
    drop(ref_logits);
    drop(ref_streams);

    // ---- Gate 3: DEVICE-SIDE SAMPLING isolation + greedy identity (2026-08-01 lever) ----
    // (a) greedy device rows: decode_step_batch_sampled's device argmax token must equal the
    //     host argmax of the SAME returned logits row, every row, every step (the argmax-gate
    //     contract surfaced at the batched-tick API).
    // (b) sampled isolation: per-seq (temp=0.7, seed=seq, ctr=step) device draws at B=N must
    //     equal the SAME metas' draws at B=1 over identically-primed caches — the serving
    //     isolation contract for the device sampler (batchmates must not change your stream).
    let mut g3_fail = 0usize;
    {
        // (a) greedy identity inside the batch. (Own block: the cache herd frees before (b) —
        // the 24GB-card capacity rule above.)
        {
            let mut caches: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> = prompts
                .iter()
                .take(b_n)
                .map(|p| *p.last().unwrap())
                .collect();
            let samp_g: Vec<Option<DevSamp>> =
                vec![Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0)); b_n];
            for _s in 0..steps.min(16) {
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let (rows, next) =
                    model.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp_g)?;
                for (bi, l) in rows.iter().enumerate() {
                    let host_am = argmax(l) as u32;
                    let dev = next[bi].expect("greedy device row missing token");
                    if dev != host_am {
                        println!(
                            "gate3a seq {bi}: device argmax {dev} != host argmax {host_am} FAIL"
                        );
                        g3_fail += 1;
                    }
                    toks[bi] = host_am;
                }
                if g3_fail > 4 {
                    break;
                }
            }
        }
        // (b) sampled isolation: B=N vs B=1, same per-seq (seed, ctr) schedule.
        let n_s = steps.min(16);
        let mut iso: Vec<Vec<u32>> = Vec::with_capacity(b_n);
        #[allow(clippy::needless_range_loop)]
        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
        for bi in 0..b_n {
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompts[bi], &mut c, 0)?;
            let mut t = *prompts[bi].last().unwrap();
            let mut out = Vec::with_capacity(n_s);
            for s in 0..n_s {
                let mut refs = [&mut c];
                let samp = [Some(DevSamp::new(
                    0.7,
                    bi as u64 + 1,
                    s as u32,
                    0,
                    1.0,
                    0.0,
                ))];
                let (_, nx) = model.decode_step_batch_sampled(&e, &[t], &mut refs, &samp)?;
                t = nx[0].expect("sampled row missing token");
                out.push(t);
            }
            iso.push(out);
        }
        let mut bat: Vec<Vec<u32>> = vec![Vec::with_capacity(n_s); b_n];
        {
            let mut caches: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> = prompts
                .iter()
                .take(b_n)
                .map(|p| *p.last().unwrap())
                .collect();
            for s in 0..n_s {
                let samp: Vec<Option<DevSamp>> = (0..b_n)
                    .map(|bi| Some(DevSamp::new(0.7, bi as u64 + 1, s as u32, 0, 1.0, 0.0)))
                    .collect();
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let (_, nx) = model.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp)?;
                for bi in 0..b_n {
                    toks[bi] = nx[bi].expect("sampled row missing token");
                    bat[bi].push(toks[bi]);
                }
            }
        }
        for bi in 0..b_n {
            if iso[bi] != bat[bi] {
                let d = iso[bi].iter().zip(&bat[bi]).position(|(a, b)| a != b);
                println!(
                    "gate3b seq {bi}: sampled stream DIVERGED batched-vs-isolated at \
                          step {d:?} FAIL"
                );
                g3_fail += 1;
            }
        }
        // (c) LEAN-LOGITS identity (inc2 component 3): the lean tick must (i) produce the
        //     SAME device tokens as the full tick, (ii) park every sampled row's logits
        //     on-device BIT-IDENTICALLY to the full tick's returned host row, (iii) leave
        //     unsampled rows' returned host rows bit-identical. Mixed metas (alternating
        //     vendor-filtered + presence-penalized device rows / host rows) exercise the
        //     sparse-penalty mutation, pristine-logit park, and partial-D2H path together.
        {
            let n_s = steps.min(8);
            let mut caches_f: Vec<Cache> = Vec::new();
            let mut caches_l: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
                caches_f.push(c);
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
                caches_l.push(c);
            }
            let mut toks: Vec<u32> = prompts
                .iter()
                .take(b_n)
                .map(|p| *p.last().unwrap())
                .collect();
            for _s in 0..n_s {
                let samp: Vec<Option<DevSamp>> = (0..b_n)
                    .map(|bi| {
                        if bi % 2 == 0 {
                            Some(
                                DevSamp::new(0.7, 7 + bi as u64, _s as u32, 20, 0.8, 0.0)
                                    .with_penalty(
                                        DevPenalty::try_new(1.0, 0.0, 1.5, vec![(toks[bi], 1)])
                                            .unwrap(),
                                    ),
                            )
                        } else {
                            None
                        }
                    })
                    .collect();
                let (rows_f, next_f) = {
                    let mut refs: Vec<&mut Cache> = caches_f.iter_mut().collect();
                    model.decode_step_batch_sampled_lean(&e, &toks, &mut refs, &samp, false)?
                };
                let (rows_l, next_l) = {
                    let mut refs: Vec<&mut Cache> = caches_l.iter_mut().collect();
                    model.decode_step_batch_sampled_lean(&e, &toks, &mut refs, &samp, true)?
                };
                for bi in 0..b_n {
                    if samp[bi].is_some() {
                        if next_f[bi] != next_l[bi] {
                            println!(
                                "gate3c seq {bi}: lean token {:?} != full token {:?} FAIL",
                                next_l[bi], next_f[bi]
                            );
                            g3_fail += 1;
                        }
                        if !rows_l[bi].is_empty() {
                            println!("gate3c seq {bi}: lean sampled row NOT empty FAIL");
                            g3_fail += 1;
                        }
                        let parked = e.dtoh(
                            caches_l[bi]
                                .last_logits_dev
                                .as_ref()
                                .expect("lean row missing device park"),
                        )?;
                        let r = &rows_f[bi];
                        if !(parked.len() == r.len()
                            && parked
                                .iter()
                                .zip(r.iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits()))
                        {
                            println!("gate3c seq {bi}: parked device logits != full host row FAIL");
                            g3_fail += 1;
                        }
                        toks[bi] = next_f[bi].unwrap();
                    } else {
                        let (r, l) = (&rows_f[bi], &rows_l[bi]);
                        if !(r.len() == l.len()
                            && r.iter()
                                .zip(l.iter())
                                .all(|(a, b)| a.to_bits() == b.to_bits()))
                        {
                            println!("gate3c seq {bi}: unsampled row lean != full FAIL");
                            g3_fail += 1;
                        }
                        toks[bi] = argmax(r) as u32;
                    }
                }
                if g3_fail > 8 {
                    break;
                }
            }
        }
        // (d) PENALTY-DISPATCH TOOTH: use the direct device argmax so exact ties cannot make
        // the causal check probabilistic. Penalize the raw winner beyond the runner-up and
        // require the returned token to match the CPU house-rule argmax while differing from
        // the unpenalized winner. Deleting or bypassing the sparse launch must make this fail.
        {
            let p = &prompts[0];
            let token = *p.last().unwrap();
            let mut raw_cache = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, p, &mut raw_cache, 0)?;
            let raw = {
                let mut refs = [&mut raw_cache];
                let (rows, _) =
                    model.decode_step_batch_sampled(&e, &[token], &mut refs, &[None])?;
                rows.into_iter().next().unwrap()
            };
            let raw_winner = argmax(&raw);
            let mut runner_up = if raw_winner == 0 { 1 } else { 0 };
            for i in 0..raw.len() {
                if i != raw_winner && raw[i] > raw[runner_up] {
                    runner_up = i;
                }
            }
            let gap = (raw[raw_winner] - raw[runner_up]).max(0.0);
            let margin = (raw[raw_winner].abs() + raw[runner_up].abs() + 1.0) * 1.0e-3 + 1.0;
            let present = gap + margin;
            let mut cpu_penalized = raw.clone();
            cpu_penalized[raw_winner] -= present;
            let expected = argmax(&cpu_penalized) as u32;

            let mut penalty_cache = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, p, &mut penalty_cache, 0)?;
            let meta = [Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0).with_penalty(
                DevPenalty::try_new(1.0, 0.0, present, vec![(raw_winner as u32, 1)]).unwrap(),
            ))];
            let got = {
                let mut refs = [&mut penalty_cache];
                let (rows, next) =
                    model.decode_step_batch_sampled(&e, &[token], &mut refs, &meta)?;
                let pristine = rows[0].len() == raw.len()
                    && rows[0]
                        .iter()
                        .zip(&raw)
                        .all(|(candidate, reference)| candidate.to_bits() == reference.to_bits());
                (
                    next[0].expect("penalty tooth device row missing token"),
                    pristine,
                )
            };
            let ok = expected != raw_winner as u32 && got.0 == expected && got.1;
            println!(
                "gate3d penalty-dispatch tooth: raw={} cpu-penalized={} device={} \
                 raw-pristine={} {}",
                raw_winner,
                expected,
                got.0,
                got.1,
                if ok {
                    "PASS"
                } else {
                    g3_fail += 1;
                    "FAIL"
                }
            );
        }
        // (e) FILTERED device rows (hermes finding, fixed 2026-08-23): top-k/top-p/min-p
        //     rows route through devsample_filtered_col + gumbel_perturb_filtered_col,
        //     default-ON since lane/devsample-topkp — and shipped WITHOUT the gates the
        //     FLAGS row claims for the device sampler. Two claims, now toothed:
        //     (e1) batch-composition independence: filtered draws inside a MIXED B=N batch
        //          (greedy / pure-temp / top-k / top-p / combined+min_p rows) must equal
        //          the same metas' draws at B=1 over identically-primed caches.
        //     (e2) filter admission (can't-hallucinate): every filtered draw must lie in
        //          the survivor set of its OWN returned logits row — top-k rank (ties
        //          allowed), min_p floor, top-p nucleus with boundary slack — computed on
        //          the host in f64.
        {
            let n_s = steps.min(16);
            let meta_for = |bi: usize, s: usize| -> DevSamp {
                let seed = bi as u64 + 101;
                let (temp, seed, ctr, top_k, top_p, min_p) = match bi % 5 {
                    0 => (0.0, 0, 0, 0i32, 1.0f32, 0.0f32),
                    1 => (0.7, seed, s as u32, 0i32, 1.0f32, 0.0f32),
                    2 => (0.7, seed, s as u32, 40i32, 1.0f32, 0.0f32),
                    3 => (0.8, seed, s as u32, 0i32, 0.95f32, 0.0f32),
                    _ => (0.9, seed, s as u32, 50i32, 0.9f32, 0.05f32),
                };
                DevSamp::new(temp, seed, ctr, top_k, top_p, min_p)
            };
            let survivor_ok = |row: &[f32], tok: usize, meta: &DevSamp| -> (bool, String) {
                let (temp, top_k, top_p, min_p) = (meta.temp, meta.top_k, meta.top_p, meta.min_p);
                if temp <= 0.0 || (top_k == 0 && top_p >= 1.0 && min_p <= 0.0) {
                    return (true, String::new());
                }
                let mx = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let probs: Vec<f64> = {
                    let e_: Vec<f64> = row
                        .iter()
                        .map(|&v| ((v as f64 - mx) / temp as f64).exp())
                        .collect();
                    let z: f64 = e_.iter().sum();
                    e_.iter().map(|v| v / z).collect()
                };
                let p_tok = probs[tok];
                if top_k > 0 {
                    let above = probs.iter().filter(|&&p| p > p_tok).count();
                    if above >= top_k as usize {
                        return (false, format!("rank {above} >= top_k {top_k}"));
                    }
                }
                if min_p > 0.0 {
                    let pmax = probs.iter().cloned().fold(0.0f64, f64::max);
                    if p_tok < 0.999 * min_p as f64 * pmax {
                        return (false, format!("p={p_tok:.3e} < min_p floor"));
                    }
                }
                if top_p < 1.0 {
                    let cum_above: f64 = probs.iter().filter(|&&p| p > p_tok).sum();
                    if cum_above >= top_p as f64 + 1e-4 {
                        return (false, format!("nucleus mass above token {cum_above:.4}"));
                    }
                }
                (true, String::new())
            };
            // isolated (B=1) reference streams.
            let mut iso: Vec<Vec<u32>> = Vec::with_capacity(b_n);
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for bi in 0..b_n {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, &prompts[bi], &mut c, 0)?;
                let mut t = *prompts[bi].last().unwrap();
                let mut out = Vec::with_capacity(n_s);
                for s in 0..n_s {
                    let mut refs = [&mut c];
                    let samp = [Some(meta_for(bi, s))];
                    let (rows, nx) = model.decode_step_batch_sampled(&e, &[t], &mut refs, &samp)?;
                    t = nx[0].expect("filtered row missing token");
                    // (d2) on the isolated arm too — the row is right here.
                    let (ok, why) = survivor_ok(&rows[0], t as usize, &meta_for(bi, s));
                    if !ok {
                        println!(
                            "gate3d seq {bi} step {s}: B=1 filtered draw OUTSIDE survivor \
                             set ({why}) FAIL"
                        );
                        g3_fail += 1;
                    }
                    out.push(t);
                }
                iso.push(out);
            }
            // batched, mixed metas.
            let mut caches: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> = prompts
                .iter()
                .take(b_n)
                .map(|p| *p.last().unwrap())
                .collect();
            let mut bat: Vec<Vec<u32>> = vec![Vec::with_capacity(n_s); b_n];
            for s in 0..n_s {
                let samp: Vec<Option<DevSamp>> = (0..b_n).map(|bi| Some(meta_for(bi, s))).collect();
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let (rows, nx) =
                    model.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp)?;
                for bi in 0..b_n {
                    let t = nx[bi].expect("filtered row missing token");
                    let (ok, why) = survivor_ok(&rows[bi], t as usize, &meta_for(bi, s));
                    if !ok {
                        println!(
                            "gate3d seq {bi} step {s}: batched filtered draw OUTSIDE \
                             survivor set ({why}) FAIL"
                        );
                        g3_fail += 1;
                    }
                    toks[bi] = t;
                    bat[bi].push(t);
                }
                if g3_fail > 12 {
                    break;
                }
            }
            for bi in 0..b_n {
                if iso[bi] != bat[bi] {
                    let d = iso[bi].iter().zip(&bat[bi]).position(|(a, b)| a != b);
                    println!(
                        "gate3d seq {bi} (meta class {}): filtered stream DIVERGED \
                         batched-vs-isolated at step {d:?} FAIL",
                        bi % 5
                    );
                    g3_fail += 1;
                }
            }
        }
    }
    println!(
        "gate3 (device sampling: greedy==host-argmax + sampled B={b_n} vs isolated \
              + lean-logits identity + penalty dispatch \
              + filtered top-k/p/min_p isolation+admission): {}",
        if g3_fail == 0 { "PASS" } else { "FAIL" }
    );

    if g1_fail + g2_fail + g3_fail == 0 {
        if let (Some(path), Some(receipt)) = (rewrite_receipt_path, rewrite_receipt) {
            std::fs::write(&path, receipt)?;
            println!("rewrite receipt: {}", std::path::Path::new(&path).display());
        }
        println!("ALL GREEN: decode_step_batch exactness battery");
        Ok(())
    } else {
        Err("decode-batch-gate FAILED".into())
    }
}

/// Per-arm bit ledger (the ppn-gate `ArmCheck` pattern, widened to per-row): a batched tick
/// returns B logit rows, so a mismatch is located by (step, row, index) — the row index is
/// what tells a stage-split bug (every row wrong at one step) apart from a per-row cache
/// bug (one row wrong from its own step onward).
struct BitCheck {
    name: String,
    bad: usize,
    first: Option<(usize, usize, usize, f32, f32)>, // (step, row, idx, ref, got)
    compared: usize,
}

/// Serial oracle for a worker-combined PP tick. Widths that fit one exact wave retain the historic
/// one-call reference. Wider widths run the live schedule's row ranges sequentially with the PP
/// door unavailable, then concatenate in request order. This isolates the pipeline schedule from
/// numeric-width changes: each oracle call and each live wave use the same exact kernel tier.
fn decode_batch_serial_waves(
    e: &Engine,
    model: &HybridModel,
    toks: &[u32],
    caches: &mut [Cache],
    wave_ranges: Option<&[(usize, usize)]>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let Some(wave_ranges) = wave_ranges else {
        let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
        return model.decode_step_batch(e, toks, &mut refs);
    };
    let mut rows = Vec::with_capacity(toks.len());
    let mut cache_tail = caches;
    for &(lo, hi) in wave_ranges {
        let width = hi - lo;
        let (wave_caches, tail) = cache_tail.split_at_mut(width);
        cache_tail = tail;
        let mut refs: Vec<&mut Cache> = wave_caches.iter_mut().collect();
        rows.extend(model.decode_step_batch(e, &toks[lo..hi], &mut refs)?);
    }
    assert!(
        cache_tail.is_empty(),
        "serial PP oracle did not consume every cache row"
    );
    Ok(rows)
}

impl BitCheck {
    fn new(name: String) -> Self {
        BitCheck {
            name,
            bad: 0,
            first: None,
            compared: 0,
        }
    }
    fn check(&mut self, step: usize, row: usize, got: &[f32], r: &[f32]) {
        assert_eq!(
            got.len(),
            r.len(),
            "row length mismatch (ref {} vs got {})",
            r.len(),
            got.len()
        );
        self.compared += got.len();
        let diffs = got
            .iter()
            .zip(r.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if diffs > 0 {
            self.bad += 1;
            let (idx, (a, b)) = got
                .iter()
                .zip(r.iter())
                .enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i, (*b, *a)))
                .unwrap();
            if self.first.is_none() {
                self.first = Some((step, row, idx, a, b));
            }
            if self.bad <= 5 {
                println!(
                    "[{}] MISMATCH step {step} row {row}: {diffs}/{} logits differ, \
                          first @[{idx}] ref={a:?} pp={b:?}",
                    self.name,
                    r.len()
                );
            }
        }
    }
    /// Returns 1 on failure (the caller's fail counter increments), 0 on pass.
    fn verdict(&self) -> usize {
        if self.bad == 0 {
            println!(
                "pp gate PASS [{}]: {} f32 logits BIT-IDENTICAL (0 differing bits)",
                self.name, self.compared
            );
            0
        } else {
            let (s, row, i, a, b) = self.first.unwrap();
            println!(
                "pp gate FAIL [{}]: {} rows mismatched of {} f32 compared (first @ step \
                      {s} row {row} idx {i}: ref={a:?} pp={b:?})",
                self.name, self.bad, self.compared
            );
            1
        }
    }
}

/// PP3/PP4 wavefront serving-tail qualification. Compare the live sampled/masked batch call with
/// only `MEMRA_PP_WAVE` changed: both arms retain the same PP placement, numeric-width cap,
/// inputs, and independently primed cache state.
#[allow(clippy::too_many_arguments)]
fn pp_wave_sampled_masked_arm(
    e: &Engine,
    model: &HybridModel,
    stages: usize,
    steps: usize,
    seed: u32,
    plen: u32,
    ctx: usize,
    exact_wave_cap: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    assert!(matches!(stages, 3 | 4));
    let b = exact_wave_cap;
    let n_s = steps.clamp(2, 8);
    let n_vocab = model.cfg.n_vocab as usize;
    if b < 8 || n_vocab < 32 {
        return Err(format!(
            "PP wave sampled/masked gate requires B>=8 and vocab>=32 (B={b}, vocab={n_vocab})"
        )
        .into());
    }

    // Eight deliberately heterogeneous rows, repeated once when the dense exact tier is B=16:
    // greedy; vendor-like .7/20/.8; min-p; sampled all-three penalties; two host rows; and two
    // grammar rows. Filtered/penalized rows stay unmasked, matching the server's admitted device
    // composition; masks exercise greedy and pure-temperature device sampling.
    let prompt_vocab = model.cfg.n_vocab.saturating_sub(16).max(1) as u64;
    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|bi| {
            (0..plen + bi as u32 * 5)
                .map(|j| {
                    ((55u64 + seed as u64 * 13 + bi as u64 * 97 + j as u64 * 31) % prompt_vocab)
                        as u32
                })
                .collect()
        })
        .collect();

    // Prime both cache herds under the same serial PP setting. The guard restores the caller's
    // MEMRA_PP_WAVE=1 even if allocation or prime returns early through `?`.
    let (mut serial_caches, mut wave_caches) = {
        let _wave_env = EnvVarRestore::set("MEMRA_PP_WAVE", "0");
        let mut serial_caches = Vec::with_capacity(b);
        let mut wave_caches = Vec::with_capacity(b);
        for prompt in &prompts {
            let mut serial = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, prompt, &mut serial, 0)?;
            let mut wave = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, prompt, &mut wave, 0)?;
            serial_caches.push(serial);
            wave_caches.push(wave);
        }
        (serial_caches, wave_caches)
    };

    let mut semantic_bad = 0usize;
    for bi in 0..b {
        if serial_caches[bi].pos != wave_caches[bi].pos {
            println!(
                "pp wave sampled/masked row {bi}: identically primed cache positions differ: serial={} wave={} FAIL",
                serial_caches[bi].pos, wave_caches[bi].pos
            );
            semantic_bad += 1;
        }
    }
    let mut toks: Vec<u32> = prompts
        .iter()
        .map(|prompt| *prompt.last().unwrap())
        .collect();
    let mut logits = BitCheck::new(format!(
        "wave sampled/masked serial-vs-wave PP-{stages} B={b}"
    ));
    let mask_words = n_vocab.div_ceil(32);
    let filter_survivor_ok = |row: &[f32], token: usize, meta: &DevSamp| -> bool {
        if token >= row.len()
            || meta.temp <= 0.0
            || (meta.top_k == 0 && meta.top_p >= 1.0 && meta.min_p <= 0.0)
        {
            return token < row.len();
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp: Vec<f64> = row
            .iter()
            .map(|&value| ((value as f64 - max) / meta.temp as f64).exp())
            .collect();
        let z: f64 = exp.iter().sum();
        let probabilities: Vec<f64> = exp.iter().map(|value| value / z).collect();
        let p_token = probabilities[token];
        if meta.top_k > 0
            && probabilities.iter().filter(|&&p| p > p_token).count() >= meta.top_k as usize
        {
            return false;
        }
        if meta.min_p > 0.0 {
            let p_max = probabilities.iter().copied().fold(0.0f64, f64::max);
            if p_token < 0.999 * meta.min_p as f64 * p_max {
                return false;
            }
        }
        if meta.top_p < 1.0 {
            let mass_above: f64 = probabilities.iter().filter(|&&p| p > p_token).sum();
            if mass_above >= meta.top_p as f64 + 1.0e-4 {
                return false;
            }
        }
        true
    };

    for step in 0..n_s {
        let mut mask_hosts: Vec<Option<Vec<u32>>> = Vec::with_capacity(b);
        for bi in 0..b {
            if !matches!(bi % 8, 5 | 6) {
                mask_hosts.push(None);
                continue;
            }
            // Eleven allowed ids spread across the vocabulary: neither an all-ones pass-through
            // nor a one-token degenerate mask. Row and step offsets make every grammar row's
            // packed bitset distinct while preserving deterministic replay between the arms.
            let allowed = 11usize.min(n_vocab);
            let stride = (n_vocab / allowed).max(1);
            let offset = (seed as usize % n_vocab + bi + step * b) % n_vocab;
            let mut words = vec![0u32; mask_words];
            for slot in 0..allowed {
                let id = (offset + slot * stride) % n_vocab;
                words[id / 32] |= 1u32 << (id % 32);
            }
            let set_bits: usize = words.iter().map(|word| word.count_ones() as usize).sum();
            assert!(
                set_bits > 1 && set_bits < n_vocab,
                "grammar mask must be sparse and non-degenerate"
            );
            mask_hosts.push(Some(words));
        }
        let masked: Vec<&Vec<u32>> = mask_hosts.iter().flatten().collect();
        assert!(masked.len() >= 2, "gate must carry multiple grammar rows");
        assert!(
            masked.windows(2).all(|pair| pair[0] != pair[1]),
            "grammar masks must be distinct per row"
        );

        // Allocate through the primary engine, exactly like worker-side grammar staging. The
        // PP epilogue consumes these buffers on the last stage through the live UVA/peer seam.
        let mut mask_devs = Vec::with_capacity(b);
        for words in &mask_hosts {
            mask_devs.push(match words {
                Some(words) => Some(e.htod_u32_v(words)?),
                None => None,
            });
        }
        let masks: Vec<Option<(&cudarc::driver::CudaSlice<u32>, usize)>> = mask_devs
            .iter()
            .map(|mask| mask.as_ref().map(|mask| (mask, mask_words)))
            .collect();

        let mut samp = Vec::with_capacity(b);
        for bi in 0..b {
            let row_seed = 0x6000_0000_0000_0000u64 ^ ((seed as u64) << 16) ^ bi as u64;
            let ctr = u32::try_from(step * b + bi).expect("bounded gate counter fits u32");
            let meta = match bi % 8 {
                0 => Some(DevSamp::new(0.0, row_seed, ctr, 0, 1.0, 0.0)),
                1 => Some(DevSamp::new(0.7, row_seed, ctr, 20, 0.8, 0.0)),
                2 => Some(DevSamp::new(0.9, row_seed, ctr, 0, 1.0, 0.05)),
                3 => {
                    let first = toks[bi];
                    let mut second = prompts[bi][(step + bi) % prompts[bi].len()];
                    if second == first {
                        second = (second + 1) % model.cfg.n_vocab;
                    }
                    let penalty = DevPenalty::try_new(
                        1.1,
                        0.5,
                        0.25,
                        vec![(first, (step % 3 + 1) as u32), (second, 2)],
                    )
                    .map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
                    Some(DevSamp::new(0.7, row_seed, ctr, 20, 0.8, 0.0).with_penalty(penalty))
                }
                4 | 7 => None,
                5 => Some(DevSamp::new(0.0, row_seed, ctr, 0, 1.0, 0.0)),
                6 => Some(DevSamp::new(0.75, row_seed, ctr, 0, 1.0, 0.0)),
                _ => unreachable!(),
            };
            samp.push(meta);
        }

        let serial_before: Vec<usize> = serial_caches.iter().map(|cache| cache.pos).collect();
        let wave_before: Vec<usize> = wave_caches.iter().map(|cache| cache.pos).collect();
        let serial_result = {
            let _wave_env = EnvVarRestore::set("MEMRA_PP_WAVE", "0");
            let mut refs: Vec<&mut Cache> = serial_caches.iter_mut().collect();
            model.decode_step_batch_sampled_lean_masked(e, &toks, &mut refs, &samp, &masks, false)
        };
        let (serial_rows, serial_next) = match serial_result {
            Ok(result) => result,
            Err(error) => {
                println!(
                    "pp wave sampled/masked FAIL [serial PP-{stages} B={b} step={step}]: {error}"
                );
                return Ok(1);
            }
        };
        let wave_liveness_before = memra_engine::pp::pp_wave_snapshot();
        let wave_result = {
            let _wave_env = EnvVarRestore::set("MEMRA_PP_WAVE", "1");
            let mut refs: Vec<&mut Cache> = wave_caches.iter_mut().collect();
            model.decode_step_batch_sampled_lean_masked(e, &toks, &mut refs, &samp, &masks, false)
        };
        let (wave_rows, wave_next) = match wave_result {
            Ok(result) => result,
            Err(error) => {
                println!(
                    "pp wave sampled/masked FAIL [wave PP-{stages} B={b} step={step}]: {error}"
                );
                return Ok(1);
            }
        };
        let wave_liveness_after = memra_engine::pp::pp_wave_snapshot();
        let ticks = wave_liveness_after.0 - wave_liveness_before.0;
        let cells = wave_liveness_after.1 - wave_liveness_before.1;
        let overlaps = wave_liveness_after.2 - wave_liveness_before.2;
        if ticks != 1 || cells < stages || overlaps == 0 {
            println!(
                "pp wave sampled/masked FAIL [wave PP-{stages} B={b} step={step}]: liveness ticks=+{ticks} cells=+{cells} overlaps=+{overlaps}"
            );
            semantic_bad += 1;
        }
        if serial_rows.len() != b
            || wave_rows.len() != b
            || serial_next.len() != b
            || wave_next.len() != b
        {
            println!(
                "pp wave sampled/masked FAIL [PP-{stages} B={b} step={step}]: result shape serial rows/tokens={}/{} wave rows/tokens={}/{}",
                serial_rows.len(),
                serial_next.len(),
                wave_rows.len(),
                wave_next.len()
            );
            return Ok(1);
        }

        for bi in 0..b {
            // Indexed comparison is also the row-order gate: prompts, seeds, counters, masks,
            // penalties, and cache depths are all row-distinct, so a wave concatenation reorder
            // cannot hide behind equal metadata.
            logits.check(step, bi, &wave_rows[bi], &serial_rows[bi]);
            if serial_next[bi] != wave_next[bi] {
                println!(
                    "pp wave sampled/masked row {bi} step {step}: sampled token serial={:?} wave={:?} FAIL",
                    serial_next[bi], wave_next[bi]
                );
                semantic_bad += 1;
            }
            match (&samp[bi], serial_next[bi], wave_next[bi]) {
                (None, None, None) => {
                    toks[bi] = argmax(&serial_rows[bi]) as u32;
                }
                (None, _, _) => {
                    println!(
                        "pp wave sampled/masked row {bi} step {step}: unsampled row returned a device token FAIL"
                    );
                    semantic_bad += 1;
                    toks[bi] = argmax(&serial_rows[bi]) as u32;
                }
                (Some(_), Some(serial_token), Some(wave_token)) => {
                    if let Some(meta) = &samp[bi]
                        && meta.penalty.is_none()
                        && (meta.top_k != 0 || meta.top_p < 1.0 || meta.min_p > 0.0)
                        && (!filter_survivor_ok(&serial_rows[bi], serial_token as usize, meta)
                            || !filter_survivor_ok(&wave_rows[bi], wave_token as usize, meta))
                    {
                        println!(
                            "pp wave sampled/masked row {bi} step {step}: filtered draw escaped its top-k/top-p/min-p survivor set FAIL"
                        );
                        semantic_bad += 1;
                    }
                    if let Some(words) = &mask_hosts[bi] {
                        let allows = |token: u32| {
                            let token = token as usize;
                            token < n_vocab && words[token / 32] & (1u32 << (token % 32)) != 0
                        };
                        if !allows(serial_token) || !allows(wave_token) {
                            println!(
                                "pp wave sampled/masked row {bi} step {step}: grammar emitted disallowed token serial={serial_token} wave={wave_token} FAIL"
                            );
                            semantic_bad += 1;
                        }
                    }
                    // Replay the serial stream into both cache herds. A sampled-id mismatch is
                    // already recorded above; keeping future inputs equal localizes later steps.
                    toks[bi] = serial_token;
                }
                (Some(_), _, _) => {
                    println!(
                        "pp wave sampled/masked row {bi} step {step}: sampled row omitted a device token (serial={:?}, wave={:?}) FAIL",
                        serial_next[bi], wave_next[bi]
                    );
                    semantic_bad += 1;
                    toks[bi] = argmax(&serial_rows[bi]) as u32;
                }
            }

            let expected_serial = serial_before[bi] + 1;
            let expected_wave = wave_before[bi] + 1;
            if serial_caches[bi].pos != expected_serial
                || wave_caches[bi].pos != expected_wave
                || serial_caches[bi].pos != wave_caches[bi].pos
            {
                println!(
                    "pp wave sampled/masked row {bi} step {step}: cache pos serial={} (expected {expected_serial}) wave={} (expected {expected_wave}) FAIL",
                    serial_caches[bi].pos, wave_caches[bi].pos
                );
                semantic_bad += 1;
            }
        }
    }

    let logit_bad = logits.verdict() != 0;
    let failed = logit_bad || semantic_bad != 0;
    println!(
        "pp wave sampled/masked {} [serial-vs-wave PP-{stages} B={b} steps={n_s}]: pristine logits, sampled ids, cache positions, and original row order; greedy + vendor(.7/20/.8) + min-p + checked repeat/frequency/presence penalty + unsampled + per-row packed masks",
        if failed { "FAIL" } else { "PASS" }
    );
    Ok(usize::from(failed))
}

/// THE BATCHED STAGE-SPLIT EXACTNESS GATE (`--mode pp`, pp2-batch 2026-08-06).
///
/// `decode_step_batch_ppn` runs each stage's layer range on its own engine/stream with a
/// `[B, n_embd]` boundary copy between them. That copy is exact (dtod / cudaMemcpyPeerAsync,
/// no conversion) and every stage runs the SAME kernels on the SAME bytes in the same order,
/// so PP-N adds ZERO deviation: the split MUST be BIT-IDENTICAL to the unsplit batched body,
/// per row, per step. The batched analogue of the eager arm's bar (48 steps x 248,320 f32
/// logits, zero differing bits — research/pp2-hardening-20260806).
///
/// METHOD (ppn-gate's, widened to B rows): the door opens BEFORE LOAD so the weights are
/// genuinely sharded, then the door is CLOSED for the reference walk. That reference is the
/// unsplit batched body over the SAME sharded placement — it peer-reads the remote stages'
/// weights, which is slow (13.9-28x) but BYTE-EXACT, which is precisely why the placement
/// needed a refusal rather than a gate. The recorded inputs come from the reference's own
/// greedy stream, so a mismatch can never desync the comparison.
///
/// THREE ARMS, and the middle one is the localizer:
///   1. `split`      — door ON, ppN caches, the stage split. Repeated `reps` times: the
///      shared-Engine scratch race this design avoids was a 35% FLAKE
///      (2026-08-02), so ONE green replay is not evidence of absence.
///   2. `unsplit@ppncache` — door ON, ppN caches (identical placement to arm 1), but
///      MEMRA_BATCH_PP=0 forces the UNSPLIT walk, with
///      MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 to pass the fail-closed guard. This
///      holds cache placement constant and varies ONLY the walk, so an arm-1
///      failure with arm 2 green localizes to the stage split, and both
///      failing localizes to stage-owned cache allocation.
///   3. `epilogue`   — the last-stage epilogue: device-sampled greedy rows must equal the
///      host argmax of their own returned row, and a lean tick must park
///      logits bit-identically to the full tick's host row. New jurisdiction
///      for this lane, because under the split that epilogue (mask ->
///      sampler -> `cache.last_logits_dev`) runs on the LAST stage's engine
///      and device, not the primary's.
///
/// The B=1 FAST PATH IS PINNED OFF for the whole battery: with the door shut its condition
/// is satisfied, so the reference at B=1 would be the m=1 FUSED trunk instead of the batched
/// body — an accepted cross-config FP-composition gap (gate1's jurisdiction) that would show
/// up here as a fake stage-split failure. Pinned through the explicit seam.
#[allow(clippy::too_many_arguments)]
fn pp_battery(
    e: &Engine,
    model: &HybridModel,
    stages: usize,
    steps: usize,
    batches: &[usize],
    reps: usize,
    seed: u32,
    plen: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let n_layers = model.layers.len();
    let fence = memra_engine::pp::pp_cuts(n_layers).unwrap_or_else(|| {
        panic!("pp mode: door failed to open (n_layers={n_layers}, stages={stages})")
    });
    assert_eq!(
        fence.len() - 1,
        stages,
        "fence {fence:?} != stages {stages}"
    );
    let devices = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let knobs = format!(
        "stages={stages} fence={fence:?} devices={} splits={} shard={} streams={}",
        if devices.is_empty() {
            "default(primary)".into()
        } else {
            devices.clone()
        },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() {
            "OFF(all-primary)"
        } else {
            "per-stage"
        },
        if memra_engine::pp::pp2_streams_off() {
            "OFF(same-stream)"
        } else {
            "per-stage"
        },
    );
    println!("pp mode: batched stage-split exactness battery over {n_layers} layers; {knobs}");
    println!("pp mode: batches={batches:?} steps={steps} reps={reps} (split arm)");
    // See the fn doc: the reference must be the BATCHED body at every B, including B=1.
    let b1_live = HybridModel::b1_fast_on();
    HybridModel::set_b1_fast(false);
    if model.uses_sliding_gated_moe_program() {
        println!(
            "pp mode: B=1 fast path inapplicable for Step35 \
                  (live correctness default = batched; batched reference pinned)"
        );
    } else if !model.b1_fast_plan_eligible() {
        println!(
            "pp mode: B=1 fast path inapplicable for Qwen35-MoE \
                  (live load-stable default = batched; batched reference pinned)"
        );
    } else {
        println!(
            "pp mode: B=1 fast path pinned OFF (live default = {})",
            if b1_live { "ON" } else { "OFF" }
        );
    }

    // widest row's prompt is plen + 5*(maxB-1); the historic 512+steps+64 floor stands.
    let max_b = *batches.iter().max().unwrap_or(&1);
    let ctx = (plen as usize + 5 * max_b) + 512 + steps + 64;
    let mut fails = 0usize;
    let rewrite_receipt_path = std::env::var_os("MEMRA_REWRITE_RECEIPT");
    let mut rewrite_reference = Vec::new();
    let mut rewrite_candidate = Vec::new();
    let dual = stages == 2 && memra_engine::pp::dual_pp_on();
    let wave = matches!(stages, 3 | 4)
        && memra_engine::pp::pp_wave_on()
            .map_err(|reason| -> Box<dyn std::error::Error> { reason.into() })?;
    let exact_wave_cap = if model.uses_sliding_gated_moe_program() {
        8
    } else if model.decode_batch_exact16_ok() {
        16
    } else {
        8
    };
    let overlap_env = std::env::var_os("MEMRA_PP_OVERLAP");
    let host_bounce_env = std::env::var_os("MEMRA_PP_HOST_BOUNCE");
    let dual_env = std::env::var_os("MEMRA_DUAL_PP");

    // BINDING-AMENDMENT NEGATIVE CELL: dual ON with the operational single-slot boundary
    // must return the exact refusal before any row is produced or cache position advances.
    // Run it before opening the double-slot positive matrix in this same loaded process.
    // Since the 2026-08-11 default flip this combination is only reachable through the
    // explicit-request mode (Auto degrades serially and unset overlap follows the mode),
    // so the cell pins MEMRA_DUAL_PP=1 with an explicit overlap=0 rather than unsetting.
    if dual {
        unsafe {
            std::env::set_var("MEMRA_DUAL_PP", "1");
        }
        unsafe {
            std::env::set_var("MEMRA_PP_OVERLAP", "0");
        }
        let prime_pipe_env = std::env::var_os("MEMRA_PRIME_PIPE");
        unsafe {
            std::env::set_var("MEMRA_PRIME_PIPE", "0");
        }
        let prompts: Vec<Vec<u32>> = (0..2)
            .map(|i| {
                (0..plen + i * 5)
                    .map(|j| 55 + seed * 13 + i * 97 + j * 31)
                    .collect()
            })
            .collect();
        let mut caches: Vec<Cache> = Vec::with_capacity(2);
        for p in &prompts {
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, p, &mut c, 0)?;
            caches.push(c);
        }
        match prime_pipe_env {
            Some(value) => unsafe {
                std::env::set_var("MEMRA_PRIME_PIPE", value);
            },
            None => unsafe {
                std::env::remove_var("MEMRA_PRIME_PIPE");
            },
        }
        let before: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let toks: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
        let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let result = model.decode_step_batch(e, &toks, &mut refs);
        drop(refs);
        match result {
            Err(err)
                if err.to_string() == memra_engine::pp::DUAL_PP_SINGLE_SLOT_REFUSAL
                    && caches.iter().map(|c| c.pos).eq(before.iter().copied()) =>
            {
                println!("dual pp negative PASS: {}", err);
            }
            Err(err) => {
                println!("dual pp negative FAIL: wrong refusal or cache mutation: {err}");
                fails += 1;
            }
            Ok(rows) => {
                println!(
                    "dual pp negative FAIL: single-slot cell produced {} token row(s)",
                    rows.len()
                );
                fails += 1;
            }
        }
        unsafe {
            std::env::set_var("MEMRA_PP_OVERLAP", "1");
        }

        // POST-REVIEW NEGATIVE CELL: double-slot does not license the unvalidated
        // host-bounce transport. Refuse with the quoted reason and preserve every cache.
        unsafe {
            std::env::set_var("MEMRA_PP_HOST_BOUNCE", "1");
        }
        let before: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let result = model.decode_step_batch(e, &toks, &mut refs);
        drop(refs);
        match result {
            Err(err)
                if err.to_string() == memra_engine::pp::DUAL_PP_HOST_BOUNCE_REFUSAL
                    && caches.iter().map(|c| c.pos).eq(before.iter().copied()) =>
            {
                println!("dual pp host-bounce negative PASS: {}", err);
            }
            Err(err) => {
                println!(
                    "dual pp host-bounce negative FAIL: wrong refusal or cache mutation: {err}"
                );
                fails += 1;
            }
            Ok(rows) => {
                println!(
                    "dual pp host-bounce negative FAIL: cell produced {} token row(s)",
                    rows.len()
                );
                fails += 1;
            }
        }
        match &host_bounce_env {
            Some(value) => unsafe {
                std::env::set_var("MEMRA_PP_HOST_BOUNCE", value);
            },
            None => unsafe {
                std::env::remove_var("MEMRA_PP_HOST_BOUNCE");
            },
        }
    }

    for &b in batches {
        let oracle_ranges = if (dual || wave) && b > exact_wave_cap {
            let ranges = if dual {
                let mid = memra_engine::pp::dual_pp_wave_mid(b)
                    .expect("a width above the exact wave cap must have two waves");
                vec![(0, mid), (mid, b)]
            } else {
                memra_engine::pp::pp_wave_ranges(b, stages)
            };
            assert!(
                ranges.iter().all(|(lo, hi)| hi - lo <= exact_wave_cap),
                "B={b} cannot fit exact oracle waves capped at {exact_wave_cap}: {ranges:?}"
            );
            Some(ranges)
        } else {
            None
        };
        // Uneven prompt lengths => uneven cache.pos across rows, which is the real serving
        // shape (per-row t_kv, so the split's per-stage pointer tables and the t_kv_max
        // padding path are both exercised rather than a degenerate all-equal-pos batch).
        // `plen` base (default 20): SWA archs pass a length past their window so the
        // per-session view-offset arm actually fires (see the --plen doc at parse).
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|i| {
                (0..plen + i as u32 * 5)
                    .map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31)
                    .collect()
            })
            .collect();

        // ---- REFERENCE: door OFF, unsplit batched body, primary-allocated caches ----
        // Sharded weights stay where the loader put them; peer reads are byte-exact.
        unsafe {
            std::env::remove_var("MEMRA_PP_STAGES");
        }
        let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(steps);
        let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(steps);
        {
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = Cache::new(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c, 0)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
            for _ in 0..steps {
                inputs.push(toks.clone());
                let rows = decode_batch_serial_waves(
                    e,
                    model,
                    &toks,
                    &mut caches,
                    oracle_ranges.as_deref(),
                )?;
                for (bi, l) in rows.iter().enumerate() {
                    toks[bi] = argmax(l) as u32;
                }
                ref_logits.push(rows);
            }
        }
        let n_vocab = ref_logits[0][0].len();
        println!(
            "-- B={b}: reference recorded ({steps} steps x {b} rows x {n_vocab} f32, \
                  door OFF over the sharded placement{})",
            oracle_ranges
                .as_ref()
                .map(|ranges| {
                    format!(
                        ", serial waves {}",
                        ranges
                            .iter()
                            .map(|(lo, hi)| (hi - lo).to_string())
                            .collect::<Vec<_>>()
                            .join("+")
                    )
                })
                .unwrap_or_default()
        );

        // ---- ARM 1: THE SPLIT (door ON, ppN caches), repeated for the flake class ----
        unsafe {
            std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
        }
        for rep in 0..reps.max(1) {
            let overlaps0 = memra_engine::pp::dual_pp_overlaps();
            let wave0 = memra_engine::pp::pp_wave_snapshot();
            let mut chk = BitCheck::new(format!("split B={b} rep{rep}"));
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c, 0)?;
                caches.push(c);
            }
            for (s, toks) in inputs.iter().enumerate() {
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let rows = model.decode_step_batch(e, toks, &mut refs)?;
                for (bi, l) in rows.iter().enumerate() {
                    if rep == 0 && rewrite_receipt_path.is_some() {
                        rewrite_reference.extend_from_slice(&ref_logits[s][bi]);
                        rewrite_candidate.extend_from_slice(l);
                    }
                    chk.check(s, bi, l, &ref_logits[s][bi]);
                }
            }
            fails += chk.verdict();
            if dual && b >= 2 {
                let overlaps = memra_engine::pp::dual_pp_overlaps() - overlaps0;
                if overlaps == 0 {
                    println!(
                        "dual pp liveness FAIL [B={b} rep{rep}]: DUAL_PP_OVERLAPS did not advance"
                    );
                    fails += 1;
                } else {
                    println!(
                        "dual pp liveness PASS [B={b} rep{rep}]: DUAL_PP_OVERLAPS +{overlaps}"
                    );
                }
            }
            if wave && b >= 2 {
                let wave1 = memra_engine::pp::pp_wave_snapshot();
                let ticks = wave1.0 - wave0.0;
                let cells = wave1.1 - wave0.1;
                let overlaps = wave1.2 - wave0.2;
                if ticks == 0 || cells < stages || overlaps == 0 {
                    println!(
                        "pp wave liveness FAIL [B={b} rep{rep}]: ticks=+{ticks} cells=+{cells} overlaps=+{overlaps}"
                    );
                    fails += 1;
                } else {
                    println!(
                        "pp wave liveness PASS [B={b} rep{rep}]: ticks=+{ticks} cells=+{cells} overlaps=+{overlaps}"
                    );
                }
            }
        }

        // ---- ARM 2: UNSPLIT WALK over the SAME ppN cache placement (the localizer) ----
        {
            let mut chk = BitCheck::new(format!("unsplit@ppncache B={b}"));
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c, 0)?;
                caches.push(c);
            }
            // MEMRA_BATCH_PP=0 selects the unsplit body; the ALLOW override is required
            // because that body is exactly what `refuse_unsplit_if_remote` fails closed on
            // under a sharded cross-device placement. Both are restored right after.
            unsafe {
                std::env::set_var("MEMRA_BATCH_PP", "0");
                std::env::set_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH", "1");
            }
            let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                for (s, toks) in inputs.iter().enumerate() {
                    let rows = decode_batch_serial_waves(
                        e,
                        model,
                        toks,
                        &mut caches,
                        oracle_ranges.as_deref(),
                    )?;
                    for (bi, l) in rows.iter().enumerate() {
                        chk.check(s, bi, l, &ref_logits[s][bi]);
                    }
                }
                Ok(())
            })();
            unsafe {
                std::env::remove_var("MEMRA_BATCH_PP");
                std::env::remove_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH");
            }
            r?;
            fails += chk.verdict();
        }
    }

    if wave {
        fails +=
            pp_wave_sampled_masked_arm(e, model, stages, steps, seed, plen, ctx, exact_wave_cap)?;
    }

    // ---- ARM 4: B=1 PER-STAGE FAST PATH vs the EAGER stage-split (decode_step_h_ppn) ----
    // Its own reference, because its bar is a DIFFERENT one. Arms 1-2 pin b1_fast OFF so B=1
    // compares batched-vs-batched; that is what makes them a clean stage-split test, but it
    // means they never execute the path a solo serving session actually takes once the door is
    // open. The B=1 stage-fast path routes each stage's range through `decode_layers_eager` —
    // the SAME per-stage call `decode_step_h_ppn` makes on the same fence with the same
    // engines/streams/slots — so the bar here is BIT-IDENTITY TO THE EAGER SPLIT ARM, not to
    // the batched body (against which it carries the accepted m=1 fusion FP gap by design; see
    // decode_batch.rs `b1_stage_fast`). Both arms run over pp::new_cache placements, and both
    // have the door open, so the only difference is which public entry point is called.
    // WHY THIS ARM EARNS ITS KEEP: it is the only gate that would catch the stage-fast branch
    // wiring the wrong fence range, reusing stage 0's engine for a later range, or advancing
    // pos twice — mistakes that leave arms 1-3 fully green because they never run it.
    // Step3.5/Step3.7 deliberately refuse this path on PP-N: their live B=1 contract is the
    // batched body already covered by arms 1-2, and the serve transition gate proves the
    // live-width invariant. Keep this arm for every model that can still select stage-fast.
    if !model.uses_sliding_gated_moe_program() && model.b1_fast_plan_eligible() {
        let mut chk = BitCheck::new("b1-stagefast vs eager-ppn B=1".to_string());
        let prompt: Vec<u32> = (0..24u32).map(|j| 55 + seed * 13 + j * 31).collect();
        let n_s = steps.min(16);
        // b1_fast ON is the whole point of the arm (arms 1-2 left it OFF).
        HybridModel::set_b1_fast(true);
        let mut c_eager = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
        let _ = model.prime_cache(e, &prompt, &mut c_eager, 0)?;
        let mut c_batch = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
        let _ = model.prime_cache(e, &prompt, &mut c_batch, 0)?;
        let mut tok = *prompt.last().unwrap();
        for s in 0..n_s {
            let (ref_row, _) = model.decode_step_h(e, tok, &mut c_eager)?;
            let got = {
                let mut refs: Vec<&mut Cache> = vec![&mut c_batch];
                model.decode_step_batch(e, &[tok], &mut refs)?
            };
            chk.check(s, 0, &got[0], &ref_row);
            // Advance on the REFERENCE's argmax so both arms stay on one token stream; a
            // divergence shows up as differing bits, not as two arms exploring different text.
            tok = argmax(&ref_row) as u32;
            assert_eq!(
                c_eager.pos, c_batch.pos,
                "b1-stagefast pos {} != eager pos {} at step {s} — one arm advanced \
                        the cache differently",
                c_batch.pos, c_eager.pos
            );
        }
        fails += chk.verdict();
        // Re-pin OFF: arm 3 (below) compares batched-vs-batched sampled/lean rows and must not
        // have one of its two caches on the m=1 fusion side of the accepted FP gap.
        HybridModel::set_b1_fast(false);
    } else if model.uses_sliding_gated_moe_program() {
        println!(
            "pp gate: b1-stagefast arm N/A for Step35 (correctness default is batched at every width)"
        );
    } else {
        println!(
            "pp gate: b1-stagefast arm N/A for Qwen35-MoE (load-stable correctness default is batched at every width)"
        );
    }

    // ---- ARM 3: the LAST-STAGE epilogue (device sampling + lean park) ----
    // Runs at the widest requested B, on the split path, with MIXED metas so the partial-D2H
    // path is exercised: even rows device-sampled greedy, odd rows host rows.
    {
        let b = *batches.iter().max().unwrap();
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|i| {
                (0..plen + i as u32 * 5)
                    .map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31)
                    .collect()
            })
            .collect();
        let n_s = steps.min(8);
        let mut ep_fail = 0usize;
        let mut caches_f: Vec<Cache> = Vec::with_capacity(b);
        let mut caches_l: Vec<Cache> = Vec::with_capacity(b);
        for p in prompts.iter() {
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, p, &mut c, 0)?;
            caches_f.push(c);
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, p, &mut c, 0)?;
            caches_l.push(c);
        }
        let mut toks: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
        for _ in 0..n_s {
            let samp: Vec<Option<DevSamp>> = (0..b)
                .map(|bi| {
                    if bi % 2 == 0 {
                        Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0))
                    } else {
                        None
                    }
                })
                .collect();
            let (rows_f, next_f) = {
                let mut refs: Vec<&mut Cache> = caches_f.iter_mut().collect();
                model.decode_step_batch_sampled_lean(e, &toks, &mut refs, &samp, false)?
            };
            let (rows_l, next_l) = {
                let mut refs: Vec<&mut Cache> = caches_l.iter_mut().collect();
                model.decode_step_batch_sampled_lean(e, &toks, &mut refs, &samp, true)?
            };
            for bi in 0..b {
                if samp[bi].is_some() {
                    let host_am = argmax(&rows_f[bi]) as u32;
                    let dev = next_f[bi].expect("split greedy row missing device token");
                    if dev != host_am {
                        println!(
                            "pp gate epilogue row {bi}: device argmax {dev} != host \
                                  argmax {host_am} FAIL"
                        );
                        ep_fail += 1;
                    }
                    if next_l[bi] != next_f[bi] {
                        println!(
                            "pp gate epilogue row {bi}: lean token {:?} != full token \
                                  {:?} FAIL",
                            next_l[bi], next_f[bi]
                        );
                        ep_fail += 1;
                    }
                    if !rows_l[bi].is_empty() {
                        println!("pp gate epilogue row {bi}: lean sampled row NOT empty FAIL");
                        ep_fail += 1;
                    }
                    // Parked on the LAST STAGE's device under the split — the D2H reads it
                    // through UVA from the primary context, which is the same thing the
                    // server's retire path does.
                    let parked = e.dtoh(
                        caches_l[bi]
                            .last_logits_dev
                            .as_ref()
                            .expect("lean row missing device park"),
                    )?;
                    let r = &rows_f[bi];
                    if !(parked.len() == r.len()
                        && parked
                            .iter()
                            .zip(r.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()))
                    {
                        println!(
                            "pp gate epilogue row {bi}: parked device logits != full \
                                  host row FAIL"
                        );
                        ep_fail += 1;
                    }
                    toks[bi] = host_am;
                } else {
                    let (r, l) = (&rows_f[bi], &rows_l[bi]);
                    if !(r.len() == l.len()
                        && r.iter()
                            .zip(l.iter())
                            .all(|(a, b)| a.to_bits() == b.to_bits()))
                    {
                        println!("pp gate epilogue row {bi}: unsampled row lean != full FAIL");
                        ep_fail += 1;
                    }
                    toks[bi] = argmax(r) as u32;
                }
            }
            if ep_fail > 8 {
                break;
            }
        }
        println!(
            "pp gate {} [epilogue B={b}]: last-stage device sampling + lean park",
            if ep_fail == 0 { "PASS" } else { "FAIL" }
        );
        fails += usize::from(ep_fail > 0);
    }

    // Restore the live default (arms 1-3 pinned it OFF, arm 4 flipped it): this process may
    // run further gates, and a leaked pin would silently re-tier them.
    HybridModel::set_b1_fast(b1_live);
    if dual {
        match overlap_env {
            Some(value) => unsafe {
                std::env::set_var("MEMRA_PP_OVERLAP", value);
            },
            None => unsafe {
                std::env::remove_var("MEMRA_PP_OVERLAP");
            },
        }
        match dual_env {
            Some(value) => unsafe {
                std::env::set_var("MEMRA_DUAL_PP", value);
            },
            None => unsafe {
                std::env::remove_var("MEMRA_DUAL_PP");
            },
        }
    }
    println!("pp mode verdict: {fails} failing arm(s); {knobs}");
    if fails == 0
        && let Some(path) = rewrite_receipt_path
    {
        let rewrite = memra_engine::plan_backend::execution_rewrites(&model.plan)
            .into_iter()
            .find(|rewrite| rewrite.surface == memra_engine::plan_backend::RewriteSurface::Pipeline)
            .ok_or("pipeline rewrite manifest is missing")?;
        let executable = std::fs::read(std::env::current_exe()?)?;
        let executable_sha256 = Sha256::digest(&executable)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let receipt = rewrite.verify_logits(
            &executable_sha256,
            &rewrite_reference,
            &rewrite_candidate,
            memra_engine::plan_backend::RewriteParityPolicy {
                max_abs: 0.0,
                max_rel: 0.0,
                require_argmax: true,
            },
        )?;
        let receipt = memra_engine::plan_backend::bind_rewrite_artifact(receipt)?;
        receipt.validate_for(&rewrite)?;
        std::fs::write(&path, receipt.to_tsv())?;
        println!("rewrite receipt: {}", std::path::Path::new(&path).display());
    }
    Ok(fails)
}

/// THE SPEC-VERIFY STAGE-SPLIT EXACTNESS GATE (`--mode ppspec`, pp2-spec 2026-08-06).
///
/// `decode_step_t_core_ppn` runs each stage's layer range of the T=K+1 VERIFY forward on its own
/// engine/stream with a `[T, n_embd]` boundary copy between them. Same argument as the batched
/// twin: every stage runs the SAME kernels (`verify_layers`, the one range-scoped body the
/// unsplit trunk also calls) on the SAME bytes in the same order, and the boundary is a straight
/// f32 copy — so the split MUST be BIT-IDENTICAL to the unsplit verify, per logit COLUMN, per
/// round. This is the gate that licenses lifting `refuse_unsplit_if_remote` on the spec path.
///
/// WHY IT IS NOT COVERED BY `--mode pp`: the verify forward is a different numeric config from
/// batched decode. It runs T columns through the GDN recurrence SEQUENTIALLY, its own
/// `t >= 3 || (t == 2 && spec_m2())` batched-linear window, and FA at m=T — and it allocates
/// more per-stage Engine scratch than any decode path (the FA partial pool at m=T plus the
/// per-layer retains). The scratch-race class this design prevents therefore has MORE surface
/// here, not less, which is why the split arm repeats (`reps`).
///
/// WHAT IS CHECKED, per round:
///   - ALL T logit columns, bit-by-bit (`T * n_vocab` f32 per round) — not just the last. A
///     stage-split bug that only perturbs interior columns would still change the accept walk,
///     because greedy accept argmaxes every column.
///   - the `h_seed` hidden ([n_embd], the LAST column pre/post-norm per MEMRA_SPEC_HPOST) — the
///     drafter is re-seeded from it every round, so a wrong h_seed silently degrades acceptance
///     without ever changing a verify logit.
///   - `cache.pos` parity against the reference at every round (asserted): the verify step
///     advances position by T, and a stage that advanced it twice would otherwise show up only
///     as a slow drift in a long run.
///
/// TWO ARMS (the epilogue/b1-fast arms have no analogue here — verify has no device-sampling
/// epilogue and no m=1 fusion tier):
///   1. `split` — door ON, ppN caches, the stage split. Repeated `reps` times for the flake
///      class.
///   2. `unsplit@ppncache` — door ON, ppN caches (identical placement to arm 1), MEMRA_SPEC_PP=0
///      to force the unsplit trunk walk, MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 to pass the
///      fail-closed guard. Holds cache placement constant and varies ONLY the walk,
///      so arm1-FAIL/arm2-PASS localizes to the stage split and both-FAIL localizes
///      to stage-owned cache allocation.
///
/// BOTH PLACEMENT ORDERS (`MEMRA_PP_DEVICES=0,1` and `1,0`) are two INVOCATIONS of this binary,
/// not two arms: the primary device follows `MEMRA_PP_DEVICES[0]` and the door opens before load,
/// so the order is a load-time property. The lane runner drives both.
fn ppspec_battery(
    e: &Engine,
    model: &HybridModel,
    stages: usize,
    rounds: usize,
    ts: &[usize],
    reps: usize,
    seed: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let n_layers = model.layers.len();
    let fence = memra_engine::pp::pp_cuts(n_layers).unwrap_or_else(|| {
        panic!("ppspec mode: door failed to open (n_layers={n_layers}, stages={stages})")
    });
    assert_eq!(
        fence.len() - 1,
        stages,
        "fence {fence:?} != stages {stages}"
    );
    let devices = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let knobs = format!(
        "stages={stages} fence={fence:?} devices={} splits={} shard={} streams={}",
        if devices.is_empty() {
            "default(primary)".into()
        } else {
            devices.clone()
        },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() {
            "OFF(all-primary)"
        } else {
            "per-stage"
        },
        if memra_engine::pp::pp2_streams_off() {
            "OFF(same-stream)"
        } else {
            "per-stage"
        },
    );
    println!("ppspec mode: verify stage-split exactness battery over {n_layers} layers; {knobs}");
    println!("ppspec mode: T={ts:?} rounds={rounds} reps={reps} (split arm)");

    let mut fails = 0usize;
    for &t in ts {
        assert!(t >= 1, "verify width T must be >= 1");
        let prompt: Vec<u32> = (0..24u32).map(|j| 55 + seed * 13 + j * 31).collect();
        let ctx = 512 + rounds * t + 64;

        // ---- REFERENCE: door OFF, unsplit verify trunk, primary-allocated cache ----
        // Sharded weights stay where the loader put them; peer reads are byte-exact.
        unsafe {
            std::env::remove_var("MEMRA_PP_STAGES");
        }
        let mut inputs: Vec<(usize, Vec<u32>)> = Vec::with_capacity(rounds);
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(rounds);
        let mut ref_seed: Vec<Vec<f32>> = Vec::with_capacity(rounds);
        let mut ref_pos: Vec<usize> = Vec::with_capacity(rounds);
        {
            let mut c = Cache::new(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, &prompt, &mut c, 0)?;
            // Round 0's chunk = the prompt's last token repeated forward through the reference's
            // own argmax columns; every later chunk is derived from the PREVIOUS round's logits,
            // so the token stream is the reference's and the arms replay it exactly (a mismatch
            // can never desync the comparison into comparing different text).
            let mut chunk: Vec<u32> = vec![*prompt.last().unwrap(); t];
            for _ in 0..rounds {
                let pos0 = c.pos;
                inputs.push((pos0, chunk.clone()));
                let (l, hs) = model.decode_step_t_h(e, &chunk, pos0, &mut c)?;
                let n_vocab = l.len() / t;
                // next chunk: column j's argmax (deterministic, and every column participates,
                // so a column-specific perturbation would change the replayed stream too).
                chunk = (0..t)
                    .map(|j| argmax(&l[j * n_vocab..(j + 1) * n_vocab]) as u32)
                    .collect();
                ref_logits.push(l);
                ref_seed.push(e.dtoh(&hs)?);
                ref_pos.push(c.pos);
            }
        }
        let n_vocab = ref_logits[0].len() / t;
        println!(
            "-- T={t}: reference recorded ({rounds} rounds x {t} cols x {n_vocab} f32 \
                  + h_seed, door OFF over the sharded placement)"
        );

        // ---- ARM 1: THE SPLIT (door ON, ppN cache), repeated for the flake class ----
        unsafe {
            std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
        }
        for rep in 0..reps.max(1) {
            let mut chk = BitCheck::new(format!("verify-split T={t} rep{rep}"));
            let mut hchk = BitCheck::new(format!("verify-split h_seed T={t} rep{rep}"));
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, &prompt, &mut c, 0)?;
            for (r, (pos0, chunk)) in inputs.iter().enumerate() {
                assert_eq!(
                    c.pos, *pos0,
                    "verify-split pos {} != reference pos {pos0} at round {r} — one arm \
                            advanced the cache differently",
                    c.pos
                );
                let (l, hs) = model.decode_step_t_h(e, chunk, *pos0, &mut c)?;
                for j in 0..t {
                    chk.check(
                        r,
                        j,
                        &l[j * n_vocab..(j + 1) * n_vocab],
                        &ref_logits[r][j * n_vocab..(j + 1) * n_vocab],
                    );
                }
                hchk.check(r, 0, &e.dtoh(&hs)?, &ref_seed[r]);
                assert_eq!(
                    c.pos, ref_pos[r],
                    "verify-split advanced pos to {} vs reference {} \
                                               at round {r}",
                    c.pos, ref_pos[r]
                );
            }
            fails += chk.verdict();
            fails += hchk.verdict();
        }

        // ---- ARM 2: UNSPLIT WALK over the SAME ppN cache placement (the localizer) ----
        {
            let mut chk = BitCheck::new(format!("verify-unsplit@ppncache T={t}"));
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, &prompt, &mut c, 0)?;
            // MEMRA_SPEC_PP=0 selects the unsplit trunk; the ALLOW override is required because
            // that trunk is exactly what `refuse_unsplit_if_remote` fails closed on under a
            // sharded cross-device placement. Both are restored right after.
            unsafe {
                std::env::set_var("MEMRA_SPEC_PP", "0");
                std::env::set_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH", "1");
            }
            let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                for (r, (pos0, chunk)) in inputs.iter().enumerate() {
                    let (l, _) = model.decode_step_t_h(e, chunk, *pos0, &mut c)?;
                    for j in 0..t {
                        chk.check(
                            r,
                            j,
                            &l[j * n_vocab..(j + 1) * n_vocab],
                            &ref_logits[r][j * n_vocab..(j + 1) * n_vocab],
                        );
                    }
                }
                Ok(())
            })();
            unsafe {
                std::env::remove_var("MEMRA_SPEC_PP");
                std::env::remove_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH");
            }
            r?;
            fails += chk.verdict();
        }
    }

    println!("ppspec mode verdict: {fails} failing arm(s); {knobs}");
    Ok(fails)
}
