//! #433 split the eager TP/EP decode vocab head across both ranks.
//!
//! WHY THIS DOOR EXISTS, IN BYTES. The 2026-09-09 replay maps put the commit and logits
//! tail at 1.151404 ms/step on rank 0, and inside it the BF16 vocab head 129280x4096 runs
//! ONCE per step on rank 1 alone at 646.792 us and 1638 GB/s, which is 91.4% of the
//! 1792 GB/s nominal. Of the launch's 1,059,595,264 modeled bytes, 1,059,061,760 (99.95%)
//! is the weight slab and 517,120 B is the f32 logits vector. A kernel at 91.4% of nominal
//! has no bandwidth headroom to find, and the vector it writes is 0.05% of its traffic, so
//! the only thing that can move this row is a SMALLER READ. Splitting the output rows is
//! that: each rank reads 529,530,880 B and 258,560 B crosses the copy engine.
//!
//! WHAT THIS BIN PROVES, AND WHAT IT DOES NOT. Splitting over N leaves every row's
//! K-reduction, accumulator and rounding point exactly where the unsplit kernel put them,
//! so the door is SAME-CLASS and owes bit equality rather than drift rows. This bin is the
//! component instrument for that claim plus the serving-shape twin; it scores no timing
//! row and admits nothing. Timing lives in `--qualify` on a relayed slot, one arm per
//! process, and never crosses boxes.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg, dsv4_prof_on};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const PRIME: usize = 256;
const OUTPUT: usize = 256;
const CAPACITY: usize = PRIME + OUTPUT + 8;
const VOCAB: usize = 129280;
/// Committed positions the component cell re-heads at. Every one of them is a DIFFERENT
/// hidden state, so a pass is not one lucky vector.
const COMPONENT_STEPS: usize = 32;

/// Every OTHER DSV4 gate bin pins this door 0 and refuses an exported 1. This bin is the
/// one whose rows may depend on it, so it reads the environment on purpose and prints
/// what it read: `--qualify` selects its arm from it, one arm per process.
fn door_env() -> String {
    std::env::var("MEMRA_DSV4_TP_HEAD_SPLIT").unwrap_or_else(|_| "unset".into())
}

/// The three merged flips are the CURRENT default program and this bin measures on top of
/// them, so it asserts them rather than pinning them off. The expectation follows the
/// resolved door accessors the engine itself reads, never a kernel symbol named here
/// (memra #438: two gates asserted a displaced split-K symbol and panicked on every
/// current tree).
fn default_program() {
    for name in [
        "MEMRA_DSV4_DENSE_FAST",
        "MEMRA_DSV4_NORM_FUSE",
        "MEMRA_DSV4_NORM_FUSE2",
        "MEMRA_DSV4_NORM2_WIDE",
        "MEMRA_DSV4_SPLITK_FAST",
        "MEMRA_DSV4_DENSE_EXACT_TAIL",
        "MEMRA_DSV4_REPLAY_CADENCE",
    ] {
        assert!(
            std::env::var(name).is_err() || std::env::var(name).as_deref() == Ok("1"),
            "default ON required: {name}"
        );
    }
    assert!(
        memra_engine::moe_m1_splitk_fast_on(),
        "the #425 paired-fetch split-K default is this bin's program"
    );
    assert!(memra_engine::moe_m1_graph_splitk_on());
    // memra #435's gate-only AR phase instrument is NOT this bin's arm: pin the resolved
    // state instead of inheriting an exported instrument or null collective, and refuse an
    // exported 1 rather than scoring a split-head row under a delayed all-reduce.
    assert_ne!(
        std::env::var("MEMRA_DSV4_AR_PHASE").as_deref(),
        Ok("1"),
        "MEMRA_DSV4_AR_PHASE=1 is not this bin's arm"
    );
    // SAFETY: single-threaded process start, before any engine state exists.
    unsafe { std::env::set_var("MEMRA_DSV4_AR_PHASE", "0") };
    unsafe extern "C" {
        fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
    }
    assert_eq!(
        unsafe { memra_dsv4_hc_dot_split_slices_for_gate() },
        16,
        "default HC S16 required"
    );
}

fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(CAPACITY, 1)
        .expect("independent state")
}

fn sha_f32(row: &[f32]) -> String {
    assert!(row.iter().all(|v| v.is_finite()), "finite logits");
    let mut h = Sha256::new();
    for v in row {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn sha_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for v in tokens {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

type Identity = (String, [u64; 2], [u64; 2]);
fn identity(gpu: &Dsv4Gpu, st: &DecodeState) -> Identity {
    let logits = gpu.read_decode_logits_for_gate(st).expect("logits");
    let cache = gpu.tp_ep_cache_digest_for_gate(st).expect("cache digest");
    let hidden = gpu.tp_ep_hidden_digest_for_gate(st).expect("hidden digest");
    assert_eq!(cache[0], cache[1], "cache rank symmetry");
    // The load-bearing invariant of this whole door: rank 0's hidden state is
    // bit-identical to rank 1's, because every cross-rank join in the walk is
    // `memra_tp_ar_1stage` reading its operands in GLOBAL rank order on both sides. If
    // this ever parts, rank 0 must not be computing logit rows.
    assert_eq!(hidden[0], hidden[1], "hidden rank symmetry");
    (sha_f32(&logits[..VOCAB]), cache, hidden)
}

fn epochs(gpu: &Dsv4Gpu, before: &[Vec<u32>; 2], steps: u32) {
    let after = gpu.full_token_ar_epochs_for_gate().expect("AR epochs");
    let attention = memra_engine::tp_ar::ar_blocks_for(4096) as usize;
    let expert = memra_engine::tp_ar::ar_blocks_for(6 * 4096) as usize;
    for rank in 0..2 {
        assert_eq!(before[rank].len(), 72);
        assert_eq!(after[rank].len(), 72);
        for block in 0..72 {
            let per_step = 43 * (u32::from(block < attention) + u32::from(block < expert));
            assert_eq!(
                after[rank][block].wrapping_sub(before[rank][block]),
                per_step * steps,
                "AR epoch rank {rank} block {block}"
            );
        }
    }
}

fn select_arm(gpu: &Dsv4Gpu, on: bool) {
    gpu.set_tp_head_split_for_gate(on).unwrap();
    assert_eq!(gpu.tp_head_split_enabled_for_gate(), on);
}

/// The first index at which two logits vectors differ in RAW BITS, or `None`. Bits and
/// not values: two NaN encodings and the two zeros compare equal as f32 and are not the
/// same logits vector.
fn first_differing_bit(a: &[f32], b: &[f32]) -> Option<usize> {
    assert_eq!(a.len(), b.len(), "compared vectors must be the same width");
    (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
}

/// One re-head under one arm, with the door's own counters read across it. Returns the
/// full vocab row and the counter deltas, so the caller can prove the arm it asked for is
/// the arm that ran.
fn rehead(gpu: &Dsv4Gpu, st: &mut DecodeState, split: bool) -> (Vec<f32>, [u64; 2], u64) {
    select_arm(gpu, split);
    let d0 = gpu.tp_head_split_dispatches();
    let p0 = gpu.tp_head_split_pulls();
    // SAFETY: the model outlives this state, no walk runs concurrently in this bin.
    let row = unsafe { gpu.rehead_logits_for_gate(st, split) }.expect("re-head");
    assert_eq!(row.len() % VOCAB, 0, "logits workspace width");
    let d1 = gpu.tp_head_split_dispatches();
    (
        row[..VOCAB].to_vec(),
        [d1[0] - d0[0], d1[1] - d0[1]],
        gpu.tp_head_split_pulls() - p0,
    )
}

/// Bit equality on a shared input, ABBA, at `COMPONENT_STEPS` different hidden states,
/// with the red arms that make the comparison a check rather than decoration.
fn component_cell(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg, output: &Path) {
    // A split arm requested with the door disarmed must REFUSE, or every "OFF" row below
    // could silently be a split row.
    select_arm(gpu, false);
    let mut probe = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut probe, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut probe).unwrap();
    }
    // SAFETY: as above.
    assert!(
        unsafe { gpu.rehead_logits_for_gate(&mut probe, true) }.is_err(),
        "RED: the split arm must refuse while the door is disarmed"
    );

    let mut sampler = gpu.device_sampler().unwrap();
    let mut carry = gpu
        .sample_device_logits(&probe, &mut sampler, &cfg, &[], None)
        .unwrap();
    let mut rows = Vec::new();
    let mut emitted = Vec::new();
    for step in 0..COMPONENT_STEPS {
        let ident = identity(gpu, &probe);
        // ABBA inside one committed hidden state: A B B A, so an ordering effect or a
        // stale workspace shows up as an A-to-A difference before it can hide in an
        // A-to-B one.
        let (a0, da0, pa0) = rehead(gpu, &mut probe, false);
        let (b0, db0, pb0) = rehead(gpu, &mut probe, true);
        let (b1, db1, pb1) = rehead(gpu, &mut probe, true);
        let (a1, da1, pa1) = rehead(gpu, &mut probe, false);
        // Non-vacuity, both directions: the split arm dispatches on BOTH ranks and pulls
        // once; the unsplit arm dispatches on neither and never pulls. A counter that
        // never moves would let an arm that silently did nothing pass.
        assert_eq!(
            da0,
            [0, 0],
            "unsplit arm dispatched a split head, step {step}"
        );
        assert_eq!(
            da1,
            [0, 0],
            "unsplit arm dispatched a split head, step {step}"
        );
        assert_eq!(pa0, 0, "unsplit arm pulled, step {step}");
        assert_eq!(pa1, 0, "unsplit arm pulled, step {step}");
        assert_eq!(
            db0,
            [1, 1],
            "split arm must dispatch on both ranks, step {step}"
        );
        assert_eq!(
            db1,
            [1, 1],
            "split arm must dispatch on both ranks, step {step}"
        );
        assert_eq!(pb0, 1, "split arm must pull once, step {step}");
        assert_eq!(pb1, 1, "split arm must pull once, step {step}");
        for (name, lhs, rhs) in [
            ("A/A", &a0, &a1),
            ("B/B", &b0, &b1),
            ("A/B", &a0, &b0),
            ("A/B reverse", &a1, &b1),
        ] {
            if let Some(i) = first_differing_bit(lhs, rhs) {
                panic!(
                    "COMPONENT_REFUSED {name} step={step} first_differing_row={i} \
                     off_bits=0x{:08x} on_bits=0x{:08x}",
                    lhs[i].to_bits(),
                    rhs[i].to_bits()
                );
            }
        }
        rows.push(sha_f32(&a0));
        assert_eq!(rows[step], ident.0, "re-head reproduced the committed row");

        // RED ARM. Slide the weight rows rank 0 reads by one. The composed vector must
        // then differ from the unsplit arm at row 0, because rank 0's whole half is off
        // by a row. A check that has never failed is not a check.
        gpu.set_tp_head_split_red_shift_for_gate(1).unwrap();
        let (red, dred, pred) = rehead(gpu, &mut probe, true);
        gpu.set_tp_head_split_red_shift_for_gate(0).unwrap();
        assert_eq!(dred, [1, 1], "the red arm must still run on both ranks");
        assert_eq!(pred, 1, "the red arm must still pull");
        let hit = first_differing_bit(&a0, &red)
            .expect("RED: a one-row weight shift produced a bit-identical vector");
        assert_eq!(
            hit, 0,
            "RED: the shift moves rank 0's half, so row 0 differs"
        );
        // The high half is rank 1's and the red arm never touches it: a red step has
        // exactly one cause.
        assert!(
            first_differing_bit(&a0[VOCAB / 2..], &red[VOCAB / 2..]).is_none(),
            "RED: the shift must not disturb rank 1's half"
        );
        // A restored shift must restore bit equality, or the red arm is sticky and every
        // later row is scored under it.
        let (after, _, _) = rehead(gpu, &mut probe, true);
        assert!(
            first_differing_bit(&a0, &after).is_none(),
            "RED: clearing the shift must restore bit equality"
        );

        select_arm(gpu, false);
        emitted.push(carry);
        gpu.decode_step_device_logits(carry, &mut probe).unwrap();
        carry = gpu
            .sample_device_logits(&probe, &mut sampler, &cfg, &[], None)
            .unwrap();
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    }
    select_arm(gpu, false);
    std::fs::write(output.join("component-rows.txt"), rows.join("\n")).unwrap();
    println!(
        "COMPONENT_PASS steps={COMPONENT_STEPS} vocab={VOCAB} comparisons={} \
         emitted_sha256={} door={}",
        COMPONENT_STEPS * 4 * VOCAB,
        sha_tokens(&emitted),
        door_env()
    );
}

/// The serving-shape twin: the same 256 sampled positions under both arms, in ONE
/// process, interleaved per step so a box clock or a warm cache cannot pick a winner.
/// Identity, not timing: this cell asserts the emitted stream, the cache and hidden
/// digests, the AR epochs and the refusal words.
fn identity_cell(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg) -> (String, Identity) {
    select_arm(gpu, false);
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    let mut off = state(gpu);
    let mut on = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut off, &prefix)
        .unwrap();
    gpu.restore_full_token_prefix_for_gate(&mut on, &prefix)
        .unwrap();
    let mut carry = first;
    let mut emitted = Vec::new();
    for step in 0..OUTPUT {
        emitted.push(carry);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        select_arm(gpu, false);
        gpu.decode_step_device_logits(carry, &mut off).unwrap();
        let next = gpu
            .sample_device_logits(&off, &mut sampler, &cfg, &[], None)
            .unwrap();
        epochs(gpu, &before, 1);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        select_arm(gpu, true);
        gpu.decode_step_device_logits(carry, &mut on).unwrap();
        let split = gpu
            .sample_device_logits(&on, &mut sampler, &cfg, &[], None)
            .unwrap();
        epochs(gpu, &before, 1);
        assert_eq!(split, next, "same-class sampled token step={step}");
        assert_eq!(
            identity(gpu, &on),
            identity(gpu, &off),
            "same-class state step={step}"
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        carry = next;
    }
    select_arm(gpu, false);
    let ident = identity(gpu, &on);
    println!(
        "IDENTITY_PASS positions={OUTPUT} generated_sha256={} final_logits_sha256={} \
         dispatches={:?} pulls={}",
        sha_tokens(&emitted),
        ident.0,
        gpu.tp_head_split_dispatches(),
        gpu.tp_head_split_pulls()
    );
    (sha_tokens(&emitted), ident)
}

/// The door is EAGER-ONLY. Arming full-token replay while it is set must refuse, or a
/// retained graph would silently bake the single-rank head under a door that reports ON.
fn replay_refusal_cell(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg) {
    select_arm(gpu, true);
    let mut st = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut st, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut st).unwrap();
    }
    // SAFETY: gate-only lifetime lease; the model outlives this state.
    let armed = unsafe { gpu.arm_full_token_replay_for_gate(&mut st, cfg) };
    let refused = match armed {
        Err(_) => true,
        Ok(()) => gpu
            .decode_sample_full_token_for_gate(prompt[1], &mut st)
            .is_err(),
    };
    assert!(
        refused,
        "RED: replay must refuse while the split-head door is armed"
    );
    select_arm(gpu, false);
    println!("REPLAY_REFUSAL_PASS door=1");
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5
                && matches!(
                    args[4].as_str(),
                    "--component" | "--identity" | "--replay-refusal" | "--defaults"
                )),
        "usage: dsv4-tp-head-split-gate <model-dir> <source.txt> <new-output-dir> \
         [--component|--identity|--replay-refusal|--defaults]"
    );
    assert!(!dsv4_prof_on(), "profiling rejected in these rows");
    for key in [
        "NSYS_PROFILING_SESSION_ID",
        "CUDA_INJECTION64_PATH",
        "NV_INJECTION64_PATH",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "profiler injection rejected: {key}"
        );
    }
    default_program();
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);
    println!("DOOR MEMRA_DSV4_TP_HEAD_SPLIT={}", door_env());
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260910,
    };
    let source = std::fs::read_to_string(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("tokenizer");
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= PRIME);
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir(&output).expect("new output directory");
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu = Box::new(
        Dsv4Gpu::load(
            Path::new(&args[1]),
            &[0, 1],
            ActQuantVariant::RefFp8Round,
            PRIME + OUTPUT + 32,
        )
        .expect("pinned TP2 model"),
    );
    assert!(gpu.topology().is_tp_ep());
    assert_eq!(gpu.topology().layers, 43);
    assert!(gpu.attention_tp_geometry().is_some() && gpu.small_kernel_diet_enabled());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);

    match args.get(4).map(String::as_str) {
        Some("--defaults") => {
            // A DOOR: an ordinary process must get OFF. The arm here comes from the
            // environment policy at load, not from a gate setter, so this is the only
            // cell that proves what a serving process actually gets.
            let env = std::env::var("MEMRA_DSV4_TP_HEAD_SPLIT").ok();
            assert!(
                env.is_none() || env.as_deref() == Some("0"),
                "default engagement requires unset or explicit 0, got {env:?}"
            );
            assert!(
                !gpu.tp_head_split_enabled_for_gate(),
                "the split-head door must be OFF before any gate override"
            );
            assert_eq!(gpu.tp_head_split_dispatches(), [0, 0]);
            assert_eq!(gpu.tp_head_split_pulls(), 0);
            println!(
                "DEFAULT_ENGAGEMENT tp_head_split=false env={} scored_rows=0",
                env.as_deref().unwrap_or("unset")
            );
        }
        Some("--component") => component_cell(&gpu, &prompt[..PRIME], cfg, &output),
        Some("--replay-refusal") => replay_refusal_cell(&gpu, &prompt[..PRIME], cfg),
        _ => {
            let (tokens, ident) = identity_cell(&gpu, &prompt[..PRIME], cfg);
            std::fs::write(
                output.join("identity.txt"),
                format!("{tokens}\n{}\n{:?}\n{:?}\n", ident.0, ident.1, ident.2),
            )
            .unwrap();
        }
    }
    assert!(
        !gpu.tp_head_split_enabled_for_gate(),
        "the door is left disarmed at exit"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit comparator must see what an f32 comparison cannot: -0.0 against 0.0 and
    /// two NaN encodings are different logits vectors, and a comparator that missed them
    /// would report bit equality on a vector that is not bit-equal.
    #[test]
    fn the_comparator_compares_bits_not_values() {
        assert_eq!(first_differing_bit(&[1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(first_differing_bit(&[1.0, 2.0], &[1.0, 2.5]), Some(1));
        assert_eq!(first_differing_bit(&[0.0], &[-0.0]), Some(0));
        let a = f32::from_bits(0x7fc0_0001);
        let b = f32::from_bits(0x7fc0_0002);
        assert!(a.is_nan() && b.is_nan());
        assert_eq!(first_differing_bit(&[a], &[b]), Some(0));
        // The first differing row, not any differing row.
        assert_eq!(
            first_differing_bit(&[1.0, 2.0, 3.0], &[9.0, 2.0, 9.0]),
            Some(0)
        );
    }

    /// The red arm's expectation, restated against the same helper the engine uses: a
    /// one-row shift moves rank 0's half and leaves rank 1's alone, so the composed
    /// vector must differ at row 0 and agree from the midpoint up.
    #[test]
    fn the_red_shift_expectation_matches_the_row_plan() {
        let (base0, rows) = memra_engine::dsv4_gpu::tp_head_split_rows(VOCAB, 0, 0).unwrap();
        let (red0, red_rows) = memra_engine::dsv4_gpu::tp_head_split_rows(VOCAB, 0, 1).unwrap();
        let (base1, _) = memra_engine::dsv4_gpu::tp_head_split_rows(VOCAB, 1, 0).unwrap();
        assert_eq!(base0, 0);
        assert_eq!(rows, VOCAB / 2);
        assert_eq!(red_rows, rows);
        assert_ne!(red0, base0, "the red arm must move rank 0's weight rows");
        assert_eq!(base1, VOCAB / 2, "rank 1 keeps the high half unshifted");
    }
}
