//! One environment-selected program per process, compared against eager OFF.
//! Two invocations (unset and explicit zero) form ten sanity rows, not a new claim.
use super::*;

pub(super) fn observed_program() -> Program {
    let cadence_env = std::env::var("MEMRA_DSV4_REPLAY_CADENCE").ok();
    let dense_env = std::env::var("MEMRA_DSV4_DENSE_EXACT_TAIL").ok();
    assert!(
        (cadence_env.is_none() && dense_env.is_none())
            || (cadence_env.as_deref() == Some("0") && dense_env.as_deref() == Some("0")),
        "default engagement admits only both unset or both explicit zero"
    );
    let program = Program {
        cadence: memra_engine::dsv4_gpu::dsv4_replay_cadence_default(),
        dense: memra_engine::dsv4_gpu::dense_exact_tail_enabled_for_gate(),
    };
    assert_eq!(program.cadence, cadence_env.is_none());
    assert_eq!(program.dense, dense_env.is_none());
    println!(
        r#"DEFAULT_POLICY {{"cadence_env_unset":{},"dense_env_unset":{},"cadence_env_zero":{},"dense_env_zero":{},"cadence_on":{},"dense_on":{},"observed_before_override":true,"rows":5,"sanity_only":true}}"#,
        cadence_env.is_none(),
        dense_env.is_none(),
        cadence_env.as_deref() == Some("0"),
        dense_env.as_deref() == Some("0"),
        program.cadence,
        program.dense
    );
    program
}

pub(super) fn run(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    tokenizer: &Tokenizer,
    output: &Path,
    program: Program,
    cfg: Dsv4SampleCfg,
) {
    select(gpu, false);
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("prime first");
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("prime control");
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .expect("initial carry");
    assert_eq!(prefix.pos, PRIME);
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    let prefix_identity = identity(gpu, &prefix);
    let mut eager = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
        .unwrap();
    let mut selected = Arm::new_default(gpu, &prefix, cfg, program);
    let mut inputs = Vec::with_capacity(OUTPUT);
    let mut carry = first;
    for step in 0..OUTPUT {
        assert_ne!(carry, tokenizer.eos_id(), "default correctness early EOS");
        inputs.push(carry);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let next = eager_step(gpu, &mut eager, &mut sampler, &cfg, carry);
        epochs(gpu, &before, 1);
        let capture = selected.prepare(gpu);
        let host = enqueues();
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let actual = gpu
            .decode_sample_full_token_for_gate(carry, &mut selected.state)
            .expect("environment replay");
        assert_eq!(actual, next, "default/eager sampled identity step={step}");
        selected.check_enqueues(host, capture);
        epochs(gpu, &before, 1);
        assert_eq!(selected.state.pos, PRIME + step + 1);
        assert_eq!(selected.state.pos, eager.pos);
        assert_eq!(
            identity(gpu, &selected.state),
            identity(gpu, &eager),
            "default/eager state step={step}"
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        assert_eq!(
            gpu.full_token_replay_counts_for_gate(&selected.state)
                .unwrap(),
            [[step as u64 + 1; 2]; 2]
        );
        assert_eq!(
            gpu.full_token_replay_variant_counts_for_gate(&selected.state)
                .unwrap(),
            [expected_variants(program.cadence, PRIME, PRIME + step + 1, true); 2]
        );
        captures_once(gpu, &selected.state, program.cadence);
        if step == 0 {
            selected.census(gpu, &output.join("default-qual"));
        }
        carry = next;
        println!(
            r#"DEFAULT_EXACT {{"step":{step},"position":{},"eager_identity":true,"token":{carry},"cadence_on":{},"dense_on":{}}}"#,
            selected.state.pos, program.cadence, program.dense
        );
    }
    assert!(!looped(&inputs));
    let expected_tokens = sha_tokens(&inputs);
    let expected_identity = identity(gpu, &eager);
    let expected_next = carry;
    drop(selected);
    drop(eager);
    refusal_cells(gpu, &prefix, cfg, &inputs, program, output);
    println!(
        "PASS environment-selected identity steps=256 refusal_cells=8; five sanity rows follow"
    );

    // Fresh environment-armed state; first forward/commit captures stay timed.
    let mut active = Arm::new_default(gpu, &prefix, cfg, program);
    let mut total_wall = 0u128;
    for row in 0..5 {
        gpu.restore_full_token_prefix_for_gate(&mut active.state, &prefix)
            .unwrap();
        assert_eq!(identity(gpu, &active.state), prefix_identity);
        let first_capture = active.prepare(gpu);
        assert_eq!(first_capture, row == 0);
        let host = enqueues();
        let counts = gpu
            .full_token_replay_counts_for_gate(&active.state)
            .unwrap();
        let variants = gpu
            .full_token_replay_variant_counts_for_gate(&active.state)
            .unwrap();
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id(), "sanity early EOS");
            tokens.push(carry);
            carry = gpu
                .decode_sample_full_token_for_gate(carry, &mut active.state)
                .expect("default sanity replay");
        }
        drain(gpu);
        let wall = start.elapsed().as_nanos();
        assert!(!looped(&tokens));
        assert_eq!(sha_tokens(&tokens), expected_tokens);
        assert_eq!(carry, expected_next);
        assert_eq!(active.state.pos, 512);
        assert_eq!(identity(gpu, &active.state), expected_identity);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        active.check_enqueues(host, first_capture);
        let counts = replay_delta(gpu, &active.state, counts, OUTPUT as u64);
        let variants = variant_delta(
            gpu,
            &active.state,
            variants,
            expected_variants(program.cadence, PRIME, PRIME + OUTPUT, true),
        );
        epochs(gpu, &before, OUTPUT as u32);
        captures_once(gpu, &active.state, program.cadence);
        if row == 0 || row == 4 {
            active.census(gpu, &output.join(format!("default-row-{row}")));
        }
        let mut control_hash = Sha256::new();
        for (i, &token) in tokens.iter().enumerate() {
            control_hash.update((u64::from(token) | (((PRIME + i) as u64) << 32)).to_le_bytes());
            control_hash.update(
                dsv4_pos_uniform(cfg.seed, PRIME + i + 1)
                    .to_bits()
                    .to_le_bytes(),
            );
            control_hash.update(0u64.to_le_bytes());
        }
        total_wall += wall;
        println!(
            r#"DEFAULT_SANITY {{"row":{row},"rows":5,"cadence_on":{},"dense_on":{},"generated_tokens":256,"decode_wall_ns":{wall},"decode_tok_s":{},"first_capture":{first_capture},"timing_scope":"sample_plus_forward_envelope","sanity_only":true,"eligible":true,"identity":true,"generated_sha256":"{expected_tokens}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"device_replays":{counts:?},"variant_device_counts":{variants:?},"control_sha256":"{:x}","control_hash_provenance":"host_reconstructed_intended_sequence"}}"#,
            program.cadence,
            program.dense,
            256e9 / wall as f64,
            expected_identity.0,
            expected_identity.1,
            expected_identity.2,
            control_hash.finalize()
        );
    }
    println!(
        r#"DEFAULT_SUMMARY {{"rows":5,"cadence_on":{},"dense_on":{},"pooled_tok_s":{},"identity":true,"first_capture_rows":1,"sanity_only":true,"performance_claim":false,"environment_arming":true}}"#,
        program.cadence,
        program.dense,
        1280e9 / total_wall as f64
    );
}
