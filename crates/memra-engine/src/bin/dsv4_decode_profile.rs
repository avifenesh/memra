//! Explicit sampled plain-decode capture after warmup; not a throughput cell.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4SampleCfg, Dsv4SamplerOrder, dsv4_sample_row, dsv4_sampler_order,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage
            .gpu
            .stream()
            .synchronize()
            .expect("profile boundary drain");
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_decode_profile <model-dir> <source.txt>"
    );
    assert_eq!(
        dsv4_sampler_order().expect("sampler order"),
        Dsv4SamplerOrder::Radix
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("native tokenizer");
    let mut prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= 8192);
    prompt.truncate(8192);
    let mut hash = Sha256::new();
    for token in &prompt {
        hash.update(token.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "3f371a0e7d56a4846b28869a780756ceea7dc3a5b1a6426b700e01e4de34593f"
    );
    println!(
        "INPUT count=8192 source_sha256={:x} chunk=512 active_C4=true sampler=radix capture_steps=32..64",
        Sha256::digest(source.as_bytes())
    );
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8544).expect("model");
    assert!(gpu.matrix_moe_enabled());
    // Profile-only composition arm: the caller explicitly requests the already
    // gated GU_FUSE + device-route/half-mirror validation-off posture.  Keep
    // this opt-in so the ordinary frozen profile remains the shipped shape.
    if std::env::var("MEMRA_DSV4_PROFILE_COMPOSE").as_deref() == Ok("1") {
        gpu.set_grouped_route_validation_for_gate(false);
        gpu.set_grouped_mirror_validation_for_gate(false);
        gpu.set_grouped_gu_fuse_for_gate(true);
        println!("PROFILE_COMPOSITION gu_fuse=true route_validate=false mirror_validate=false");
    }
    let c4_elide = std::env::var("MEMRA_DSV4_PROFILE_C4_ELIDE").as_deref() == Ok("1");
    // 8544 / ratio-4 is the complete compressed history for this bounded
    // profile. A full recent sidecar makes host publication elision safe for
    // the one-way plain decode capture; older rows remain in canonical host
    // storage and newly emitted rows are served by absolute recent tags.
    let c4_recent_rows = 8544 / 4;
    let mut state = if c4_elide {
        gpu.alloc_decode_state_host_c4_recent(8544, 512, c4_recent_rows)
            .expect("host C4 recent")
    } else {
        gpu.alloc_decode_state_host_c4(8544, 512).expect("host C4")
    };
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let mut row = gpu
        .dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, 512)
        .expect("prefill");
    let host = gpu.snapshot_decode_state(&state).expect("snapshot");
    let _host_draft = gpu.snapshot_dspark_state(&draft).expect("draft snapshot");
    drop(state);
    drop(draft);
    let mut state = if c4_elide {
        gpu.restore_decode_state_host_c4_recent(&host, 8544, 512, c4_recent_rows)
            .expect("restore recent C4")
    } else {
        gpu.restore_decode_state_host_c4(&host, 8544, 512)
            .expect("restore")
    };
    if c4_elide {
        gpu.set_c4_host_copy_elision_for_gate(true);
        println!(
            "PROFILE_C4_HOST_COPY_ELISION enabled=true recent_rows={c4_recent_rows} full_capacity=true"
        );
    }
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
    let mut tokens = Vec::new();
    let mut capture_timer = None;
    for step in 0..96 {
        if step == 32 {
            drain(&gpu);
            gpu.stages[0]
                .gpu
                .ctx
                .bind_to_thread()
                .expect("profile start context");
            cudarc::driver::safe::profiler_start().expect("profile start");
            capture_timer = Some(Instant::now());
        }
        let token = dsv4_sample_row(&row, prompt.len() + step, &cfg).expect("sample");
        assert_ne!(
            token,
            tokenizer.eos_id(),
            "early EOS: profile protocol refused"
        );
        tokens.push(token);
        row = gpu.decode_step(token, &mut state).expect("plain step");
        if step == 63 {
            drain(&gpu);
            let seconds = capture_timer.take().unwrap().elapsed().as_secs_f64();
            gpu.stages[0]
                .gpu
                .ctx
                .bind_to_thread()
                .expect("profile stop context");
            cudarc::driver::safe::profiler_stop().expect("profile stop");
            println!(
                "PROFILE_COMPLETE steps=32..64 absolute_positions=8224..8256 seconds={seconds:.6} both_stages_drained=true"
            );
        }
    }
    if c4_elide {
        gpu.set_c4_host_copy_elision_for_gate(false);
        println!("PROFILE_C4_HOST_COPY_ELISION enabled=false");
    }
    let mut hash = Sha256::new();
    for token in &tokens {
        hash.update(token.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "ef41ad6a5caeebf4b5fb54b93d7054b628ace34892b59309b954e20d0d2fb81d",
        "frozen sampled stream"
    );
    assert!(gpu.ep_calls() > 0 && gpu.sink_tiled_calls() > 0);
    println!(
        "PASS radix sampled plain profile with frozen 96-token stream; not a throughput measurement"
    );
}
