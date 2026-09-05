//! Compare emitted tokens and persistent state with plain position-keyed sampling.
//! Proposal acceptance/timing is characterization, not a serving promotion.
use memra_engine::dsv4_gpu::{
    Dsv4DraftProposal, Dsv4Gpu, Dsv4PenaltyCfg, Dsv4SampleCfg, Dsv4Vt, dsv4_penalize_row,
    dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;

fn digest(classes: Vec<(String, Vec<f32>)>) -> Vec<u8> {
    let mut hash = Sha256::new();
    for (name, values) in classes {
        hash.update(name.as_bytes());
        hash.update((values.len() as u64).to_le_bytes());
        let mask = name.ends_with(".cmp_pend_score") || name.ends_with(".idx_pend_score");
        for value in values {
            assert!(value.is_finite() || (mask && value == f32::NEG_INFINITY));
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    hash.finalize().to_vec()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_coupled_proposal_gate <model-dir> <real-source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(
        &format!("Review this inference engine source:\n{source}"),
        true,
    );
    assert!(tokens.len() >= 1025, "real source must not be padded");
    let mut gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    gpu.set_prefill_grouped_for_gate(false)
        .expect("reference math");
    for count in [160, 1025] {
        let prompt = &tokens[..count];
        let capacity = count + 96;
        let mut state = gpu
            .alloc_decode_state_for_transient(capacity, 32)
            .expect("state");
        let mut draft = gpu.dspark_alloc_state().expect("draft");
        let logits = gpu
            .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, 32)
            .expect("prime");
        let host = gpu.snapshot_decode_state(&state).expect("snapshot trunk");
        let host_draft = gpu.snapshot_dspark_state(&draft).expect("snapshot draft");
        drop(state);
        drop(draft);
        for (seed, penalized) in [(20260905, false), (73, false), (911, true)] {
            let cfg = Dsv4SampleCfg {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed,
            };
            let penalty = penalized.then_some(Dsv4PenaltyCfg {
                last_n: 64,
                repeat: 1.1,
                freq: 0.2,
                present: 0.1,
            });
            let (plain, plain_cache, plain_draft) = {
                let mut state = gpu
                    .restore_decode_state_for_transient(&host, capacity, 32)
                    .expect("plain state");
                let mut draft = gpu.restore_dspark_state(&host_draft).expect("plain draft");
                let mut row = logits.clone();
                let mut emitted = Vec::new();
                for step in 0..64 {
                    let mut sampled = row.clone();
                    if let Some(pc) = &penalty {
                        let mut window = prompt.to_vec();
                        window.extend_from_slice(&emitted);
                        dsv4_penalize_row(&mut sampled, &window, pc);
                    }
                    let token =
                        dsv4_sample_row(&sampled, count + step, &cfg).expect("plain sample");
                    emitted.push(token);
                    if step < 63 {
                        let pos = state.pos;
                        row = gpu
                            .decode_step_tap(token, &mut state, &mut draft, 0)
                            .expect("plain decode");
                        gpu.dspark_write_rings(&mut draft, 0, pos)
                            .expect("plain rings");
                    }
                }
                (
                    emitted,
                    digest(gpu.cache_classes(&state).expect("plain cache")),
                    digest(gpu.dspark_ring_classes(&draft).expect("plain draft")),
                )
            };
            for arm in [
                Dsv4DraftProposal::Greedy,
                Dsv4DraftProposal::Coupled,
                Dsv4DraftProposal::Greedy,
            ] {
                gpu.set_draft_proposal_for_gate(arm).expect("proposal arm");
                let mut state = gpu
                    .restore_decode_state_for_transient(&host, capacity, 32)
                    .expect("spec state");
                let mut draft = gpu.restore_dspark_state(&host_draft).expect("spec draft");
                let mut verify = gpu.alloc_verify_state_for(capacity).expect("verify");
                let before = gpu.coupled_draft_draws();
                let start = Instant::now();
                let run = gpu
                    .spec_sampled_batched_pen_restored(
                        prompt,
                        &logits,
                        64,
                        &mut state,
                        &mut draft,
                        &mut verify,
                        usize::MAX,
                        Dsv4Vt::Off,
                        &cfg,
                        penalty.as_ref(),
                        None,
                    )
                    .expect("spec sample");
                let seconds = start.elapsed().as_secs_f64();
                assert_eq!(
                    plain, run.tokens,
                    "output count={count} seed={seed} arm={arm:?}"
                );
                assert_eq!(
                    plain_cache,
                    digest(gpu.cache_classes(&state).expect("spec cache")),
                    "persistent trunk state"
                );
                assert_eq!(
                    plain_draft,
                    digest(gpu.dspark_ring_classes(&draft).expect("spec draft")),
                    "persistent draft state"
                );
                let draws = gpu.coupled_draft_draws() - before;
                let expected_draws = if arm == Dsv4DraftProposal::Coupled {
                    run.rounds.len() as u64 * (gpu.verify_tmax() - 1) as u64
                } else {
                    0
                };
                assert_eq!(
                    draws, expected_draws,
                    "every coupled draft slot must engage"
                );
                let accepted: usize = run.rounds.iter().map(|round| round.accepts).sum();
                println!(
                    "EXACT count={count} seed={seed} penalized={penalized} arm={arm:?} tokens=64 rounds={} accepted={accepted} coupled_draws={draws} decode_gate_s={seconds:.6}",
                    run.rounds.len()
                );
            }
        }
    }
    println!(
        "PASS sampled proposal coupling and persistent-state identity; performance qualification remains separate"
    );
}
