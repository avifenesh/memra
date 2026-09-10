//! One-load scalar/tiled/scalar attention-score comparison on the complete model.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4SampleCfg, Dsv4SinkScore, Dsv4VerifyTopk, Dsv4Vt, dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn classes(hash: &mut Sha256, items: Vec<(String, Vec<f32>)>) {
    for (name, values) in items {
        hash.update(name.as_bytes());
        hash.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
}
fn floats(hash: &mut Sha256, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()));
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
}

fn run(
    gpu: &Dsv4Gpu,
    source: &[u32],
    count: usize,
    width: usize,
    suffix: usize,
    c4: bool,
) -> Vec<u8> {
    let start = Instant::now();
    let capacity = count + suffix + 96;
    let transient = width.max(gpu.verify_tmax());
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, transient)
        .expect("state");
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
    let before = gpu.ep_calls();
    let mut row = gpu
        .dspark_prefill_prime_chunked(&source[..count], &mut state, &mut draft, width)
        .expect("prime");
    let mut hash = Sha256::new();
    floats(&mut hash, &row);
    classes(&mut hash, gpu.cache_classes(&state).expect("prime cache"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("prime draft"),
    );
    let parked = gpu.snapshot_decode_state(&state).expect("park");
    let parked_draft = gpu.snapshot_dspark_state(&draft).expect("park draft");
    drop(state);
    drop(draft);
    state = gpu
        .restore_decode_state_for_transient(&parked, capacity, transient)
        .expect("restore");
    draft = gpu
        .restore_dspark_state(&parked_draft)
        .expect("restore draft");
    if c4 {
        gpu.offload_c4_decode_state(&mut state, &verify)
            .expect("C4 offload");
    }
    if suffix > 0 {
        row = gpu
            .dspark_continue_prefix_chunked(
                &source[count..count + suffix],
                &mut state,
                &mut draft,
                width,
            )
            .expect("warm suffix");
    }
    floats(&mut hash, &row);
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260905,
    };
    let mut prompt = source[..count + suffix].to_vec();
    for _ in 0..9 {
        let token = dsv4_sample_row(&row, state.pos, &cfg).expect("sample");
        prompt.push(token);
        let pos = state.pos;
        row = gpu
            .decode_step_tap(token, &mut state, &mut draft, 0)
            .expect("plain");
        gpu.dspark_write_rings(&mut draft, 0, pos)
            .expect("plain rings");
        hash.update(token.to_le_bytes());
        floats(&mut hash, &row);
    }
    let run = gpu
        .spec_sampled_batched_pen_restored(
            &prompt,
            &row,
            32,
            &mut state,
            &mut draft,
            &mut verify,
            usize::MAX,
            Dsv4Vt::Off,
            &cfg,
            None,
            None,
        )
        .expect("spec");
    assert_eq!(run.tokens.len(), 32);
    assert!(!run.rounds.is_empty());
    for token in run.tokens {
        hash.update(token.to_le_bytes());
    }
    for round in &run.rounds {
        for value in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.emitted,
        ] {
            hash.update((value as u64).to_le_bytes());
        }
        floats(&mut hash, &round.confidence);
    }
    classes(&mut hash, gpu.cache_classes(&state).expect("final cache"));
    classes(
        &mut hash,
        gpu.dspark_ring_classes(&draft).expect("final draft"),
    );
    let calls = gpu.ep_calls() - before;
    println!(
        "CASE count={count} width={width} suffix={suffix} c4={c4} sampled=41 rounds={} ep_calls={calls} gate_wall_with_checks_s={:.6}",
        run.rounds.len(),
        start.elapsed().as_secs_f64()
    );
    hash.finalize().to_vec()
}

fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        // Gate-only AR phase instrument: pinned off here so no other bin can inherit
        // an exported instrument or null collective from the environment.
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_sink_score_gate <model-dir> <source.txt>"
    );
    for (name, value) in [
        ("MEMRA_DSV4_SINK_SCORE", "scalar"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    let dir = Path::new(&args[1]);
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let text = std::fs::read_to_string(&args[2]).expect("source");
    let tokens = tokenizer.encode(
        &format!("Review this inference engine source:\n{text}"),
        true,
    );
    assert!(tokens.len() > 5000);
    println!("SOURCE sha256={:x}", Sha256::digest(text.as_bytes()));
    // memra #458: this is a bench process, so it may run the matrix expert program
    // with the default-ON split-K arm; a serving process cannot arm it and refuses
    // that combination at load instead of failing every request.
    memra_engine::arm_matrix_splitk_door_for_gate();
    let mut gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8192).expect("matrix EP load");
    gpu.set_verify_topk_for_gate(Dsv4VerifyTopk::Device)
        .expect("selector");
    let cases = [
        (1, 32, 0, false),
        (32, 1, 17, false),
        (160, 32, 33, true),
        (1025, 512, 129, true),
        (4097, 512, 33, false),
    ];
    let mut baseline = Vec::new();
    for &(count, width, suffix, c4) in &cases {
        baseline.push(run(&gpu, &tokens, count, width, suffix, c4));
    }
    assert_eq!(
        gpu.sink_tiled_calls(),
        0,
        "scalar baseline must not use tiled scorer"
    );
    for arm in [Dsv4SinkScore::Tiled, Dsv4SinkScore::Scalar] {
        gpu.set_sink_score_for_gate(arm).expect("sink score arm");
        for (case, &(count, width, suffix, c4)) in cases.iter().enumerate() {
            let before = gpu.sink_tiled_calls();
            assert_eq!(
                baseline[case],
                run(&gpu, &tokens, count, width, suffix, c4),
                "sink-score model mismatch arm={arm:?} count={count} width={width} c4={c4}"
            );
            let calls = gpu.sink_tiled_calls() - before;
            assert_eq!(
                calls > 0,
                arm == Dsv4SinkScore::Tiled,
                "sink scorer engagement"
            );
            println!("EXACT full-model sink arm={arm:?} case={case} tiled_calls={calls}");
        }
    }
    println!(
        "PASS full-model scalar/tiled/scalar sink scores, sampled output, warm restore and C4 identity; serving/performance admission remains separate"
    );
}
