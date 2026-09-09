//! Gate-only launch-boundary instrument. Build with tools/build-dsv4-launch-boundary.sh.
use memra_engine::dsv4_gpu::{
    DecodeState, Dsv4Gpu, Dsv4SampleCfg, dense_exact_tail_enabled_for_gate, dsv4_prof_on,
    dsv4_replay_cadence_default, restore_dense_exact_tail_default_for_gate,
};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, c_void},
    path::Path,
};

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
fn symbol(name: &std::ffi::CStr) -> *mut c_void {
    let p = unsafe { dlsym(std::ptr::null_mut(), name.as_ptr()) };
    assert!(
        !p.is_null(),
        "missing gate wrapper; use tools/build-dsv4-launch-boundary.sh"
    );
    p
}
fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(528, 1).unwrap()
}
fn identity(gpu: &Dsv4Gpu, s: &DecodeState) -> (String, [u64; 2], [u64; 2]) {
    let mut hash = Sha256::new();
    for v in gpu.read_decode_logits_for_gate(s).unwrap() {
        hash.update(v.to_bits().to_le_bytes());
    }
    (
        format!("{:x}", hash.finalize()),
        gpu.tp_ep_cache_digest_for_gate(s).unwrap(),
        gpu.tp_ep_hidden_digest_for_gate(s).unwrap(),
    )
}
fn census(gpu: &Dsv4Gpu, s: &DecodeState) {
    assert_eq!(gpu.full_token_replay_captures_for_gate(s).unwrap(), [3, 1]);
    for rank in gpu.full_token_replay_variant_census_for_gate(s).unwrap() {
        for (slot, kernels) in [(0, 2741), (2, 3140), (3, 3240)] {
            assert_eq!(
                [
                    rank[slot][1],
                    rank[slot][2],
                    rank[slot][3],
                    rank[slot][4],
                    rank[slot][6]
                ],
                [kernels, 86, 1, 86, 0]
            );
        }
    }
}
fn main() {
    // Missing wrappers fail before model allocation. They are linked only into this gate.
    let prepare: unsafe extern "C" fn() =
        unsafe { std::mem::transmute(symbol(c"memra_launch_boundary_prepare")) };
    let begin: unsafe extern "C" fn() =
        unsafe { std::mem::transmute(symbol(c"memra_launch_boundary_begin")) };
    let finish: unsafe extern "C" fn(u32) =
        unsafe { std::mem::transmute(symbol(c"memra_launch_boundary_finish")) };
    let cleanup: unsafe extern "C" fn() =
        unsafe { std::mem::transmute(symbol(c"memra_launch_boundary_cleanup")) };
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "gate <model-dir> <source-tape>");
    for key in [
        "MEMRA_DSV4_REPLAY_CADENCE",
        "MEMRA_DSV4_DENSE_EXACT_TAIL",
        "MEMRA_DSV4_NVTX",
        "MEMRA_DSV4_ROUND_PROFILE",
        "LD_PRELOAD",
    ] {
        assert!(std::env::var_os(key).is_none(), "{key} must be unset");
    }
    assert!(!dsv4_prof_on());
    restore_dense_exact_tail_default_for_gate();
    assert!(dense_exact_tail_enabled_for_gate() && dsv4_replay_cadence_default());
    assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);
    for (key, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_DRAFTER", "off"),
    ] {
        assert_eq!(std::env::var(key).as_deref(), Ok(value));
    }
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded"
    );
    let tokenizer = Tokenizer::from_hf_dir(dir).unwrap();
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= 256);
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    memra_engine::set_moe_m1_splitk_for_gate(false);
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 544).unwrap();
    assert!(gpu.topology().is_tp_ep() && gpu.attention_tp_geometry().is_some());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);
    gpu.set_small_kernel_diet_for_gate(true).unwrap();
    let cfg = Dsv4SampleCfg {
        temperature: 1.,
        top_p: 1.,
        top_k: 0,
        seed: 20260907,
    };
    let mut prefix = state(&gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &token in &prompt[1..256] {
        gpu.decode_step_device_logits(token, &mut prefix).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let mut first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    while prefix.pos < 368 {
        assert_ne!(first, tokenizer.eos_id());
        gpu.decode_step_device_logits(first, &mut prefix).unwrap();
        first = gpu
            .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
            .unwrap();
    }
    let mut measured = state(&gpu);
    gpu.restore_full_token_prefix_for_gate(&mut measured, &prefix)
        .unwrap();
    // The model, weights and request allocations remain stable throughout both arms.
    unsafe {
        gpu.arm_full_token_replay_for_gate(&mut measured, cfg)
            .unwrap();
    }
    // Capture the default variants and commit graph, outside the 64 steps.
    let warm = gpu
        .decode_sample_full_token_for_gate(first, &mut measured)
        .unwrap();
    census(&gpu, &measured);
    unsafe { prepare() };
    gpu.restore_full_token_prefix_for_gate(&mut measured, &prefix)
        .unwrap();
    let before = gpu.full_token_ar_epochs_for_gate().unwrap();
    let counts = gpu
        .full_token_replay_variant_counts_for_gate(&measured)
        .unwrap();
    let mut carry = first;
    let mut tokens = Vec::new();
    for position in 368..432 {
        assert_ne!(carry, tokenizer.eos_id());
        tokens.push(carry);
        unsafe { begin() };
        carry = gpu
            .decode_sample_full_token_for_gate(carry, &mut measured)
            .unwrap();
        unsafe { finish(position) };
        assert_eq!(measured.pos, position as usize + 1);
    }
    let actual = (carry, identity(&gpu, &measured));
    census(&gpu, &measured);
    let after = gpu
        .full_token_replay_variant_counts_for_gate(&measured)
        .unwrap();
    for (r, a) in after.iter().enumerate() {
        for (s, n) in [48, 64, 15, 1].iter().enumerate() {
            assert_eq!(a[s] - counts[r][s], *n);
        }
    }
    let epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
    let attention = memra_engine::tp_ar::ar_blocks_for(4096) as usize;
    let expert = memra_engine::tp_ar::ar_blocks_for(6 * 4096) as usize;
    for rank in 0..2 {
        for block in 0..72 {
            assert_eq!(
                epochs[rank][block].wrapping_sub(before[rank][block]),
                64 * 43 * (u32::from(block < attention) + u32::from(block < expert))
            );
        }
    }
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    // Same retained state and same inputs, wrappers disarmed: exact default-replay twin.
    gpu.restore_full_token_prefix_for_gate(&mut measured, &prefix)
        .unwrap();
    carry = first;
    for &token in &tokens {
        assert_eq!(carry, token);
        carry = gpu
            .decode_sample_full_token_for_gate(carry, &mut measured)
            .unwrap();
    }
    assert_eq!((carry, identity(&gpu, &measured)), actual);
    let mut hash = Sha256::new();
    for token in &tokens {
        hash.update(token.to_le_bytes());
    }
    println!(
        "BOUNDARY_PASS {{\"steps\":64,\"warm_next\":{warm},\"identity\":true,\"census\":true,\"ar_epochs\":true,\"tokens_sha256\":\"{:x}\",\"profiled\":false,\"first_ar_wait\":null,\"decision_threshold_ms\":1.0}}",
        hash.finalize()
    );
    drop(measured);
    drop(prefix);
    unsafe { cleanup() };
}
