//! Sampled engine decode-only timing, independent of prefill and restoration.
//! ABBA x3, one model load, exact restored prompt per row, EOS respected.
//! This is not HTTP latency or concurrent-serving qualification.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4HostDecodeState, Dsv4HostDsparkState, Dsv4SampleCfg, Dsv4SamplerOrder, Dsv4Vt,
    dsv4_sample_row, dsv4_sampler_order, resolve_vt, set_dsv4_sampler_order_for_gate,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("UTC")
        .as_millis()
}

fn token_hash(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Conservative exclusion: at least 32 tokens and four repetitions of a block.
/// The raw row remains recorded; this is never used to stop or change generation.
fn looped(tokens: &[u32]) -> bool {
    (1..=32).any(|width| {
        let repetitions = 4usize.max(32usize.div_ceil(width));
        let len = width * repetitions;
        tokens.windows(len).any(|span| {
            span.chunks_exact(width)
                .all(|block| block == &span[..width])
        })
    })
}

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage
            .gpu
            .stream()
            .synchronize()
            .expect("timing boundary drain");
    }
}

struct ReadyPrompt<'a> {
    gpu: &'a Dsv4Gpu,
    tokenizer: &'a Tokenizer,
    tokens: &'a [u32],
    logits: &'a [f32],
    trunk: &'a Dsv4HostDecodeState,
    draft: &'a Dsv4HostDsparkState,
    capacity: usize,
    new_tokens: usize,
}

impl ReadyPrompt<'_> {
    #[allow(clippy::too_many_arguments)]
    fn measure(
        &self,
        spec: bool,
        greedy: bool,
        ordinal: usize,
        warmup: bool,
        recent_rows: usize,
        vt: Dsv4Vt,
        _mirror_validate: bool,
    ) -> Vec<u32> {
        let restore_start = Instant::now();
        let mut state = self
            .gpu
            .restore_decode_state_host_c4_recent(self.trunk, self.capacity, 512, recent_rows)
            .expect("direct host restore");
        let recent_gpu_bytes = self.gpu.c4_recent_gpu_bytes(&state).unwrap();
        assert_eq!(recent_gpu_bytes.iter().sum::<u64>() > 0, recent_rows > 0);
        let mut draft = spec.then(|| {
            self.gpu
                .restore_dspark_state(self.draft)
                .expect("draft restore")
        });
        let mut verify = spec.then(|| {
            self.gpu
                .alloc_verify_state_for(self.capacity)
                .expect("verify")
        });
        let mut row = self.logits.to_vec();
        drain(self.gpu);
        let restore_ns = restore_start.elapsed().as_nanos();
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 20260906,
        };
        let mut tokens = Vec::with_capacity(self.new_tokens);
        let mut commits = Vec::with_capacity(self.new_tokens);
        let mut eos = false;
        let arm = if spec { "dspark" } else { "plain" };
        let mode = if greedy { "greedy" } else { "sampled" };
        let sampler_order = match dsv4_sampler_order().expect("sampler order") {
            Dsv4SamplerOrder::Comparison => "comparison",
            Dsv4SamplerOrder::Radix => "radix",
        };
        let vt_label = match vt {
            Dsv4Vt::Off => "off".to_string(),
            Dsv4Vt::Slot { tau_logit, floor } => {
                format!("slot_logit={tau_logit:.6},floor={floor}")
            }
        };
        let ep_before = self.gpu.ep_calls();
        let recent_before = self.gpu.c4_recent_gather_calls();
        let sink_before = self.gpu.sink_tiled_calls();
        let start_ms = unix_ms();
        println!(
            "START prompt={} mode={mode} arm={arm} sampler_order={sampler_order} vt={vt_label} recent_rows={recent_rows} ordinal={ordinal} warmup={warmup} start_unix_ms={start_ms}",
            self.tokens.len()
        );
        let timer = Instant::now();
        let (rounds, drafted, accepted) = {
            let mut commit = |new: &[u32]| {
                let ns = timer.elapsed().as_nanos();
                for &token in new {
                    if token == self.tokenizer.eos_id() {
                        eos = true;
                        return false;
                    }
                    tokens.push(token);
                    commits.push(ns);
                    if tokens.len() == self.new_tokens {
                        return false;
                    }
                }
                true
            };
            if spec {
                let run = if greedy {
                    self.gpu
                        .spec_greedy_batched_stream_restored(
                            self.tokens.len(),
                            self.logits,
                            self.new_tokens,
                            &mut state,
                            draft.as_mut().unwrap(),
                            verify.as_mut().unwrap(),
                            usize::MAX,
                            vt,
                            Some(&mut commit),
                        )
                        .expect("greedy DSpark")
                } else {
                    self.gpu
                        .spec_sampled_batched_pen_restored(
                            self.tokens,
                            self.logits,
                            self.new_tokens,
                            &mut state,
                            draft.as_mut().unwrap(),
                            verify.as_mut().unwrap(),
                            usize::MAX,
                            vt,
                            &cfg,
                            None,
                            Some(&mut commit),
                        )
                        .expect("sampled DSpark")
                };
                (
                    run.rounds.len(),
                    run.rounds
                        .iter()
                        .map(|r| r.t_batch.saturating_sub(1))
                        .sum::<usize>(),
                    run.rounds.iter().map(|r| r.accepts).sum::<usize>(),
                )
            } else {
                let mut token = if greedy {
                    argmax(&row)
                } else {
                    dsv4_sample_row(&row, self.tokens.len(), &cfg).expect("sample")
                };
                for i in 0..self.new_tokens {
                    if !commit(&[token]) {
                        break;
                    }
                    if i + 1 < self.new_tokens {
                        if greedy {
                            token = self
                                .gpu
                                .decode_step_greedy(token, &mut state)
                                .expect("greedy plain decode");
                        } else {
                            row = self
                                .gpu
                                .decode_step(token, &mut state)
                                .expect("sampled plain decode");
                            token = dsv4_sample_row(&row, self.tokens.len() + i + 1, &cfg)
                                .expect("sample");
                        }
                    }
                }
                (0, 0, 0)
            }
        };
        drain(self.gpu);
        let decode_wall_ns = timer.elapsed().as_nanos();
        let end_ms = unix_ms();
        let ep_calls = self.gpu.ep_calls() - ep_before;
        let recent_gathers = self.gpu.c4_recent_gather_calls() - recent_before;
        assert_eq!(
            recent_gathers > 0,
            recent_rows > 0,
            "cache-aware gather engagement"
        );
        let sink_calls = self.gpu.sink_tiled_calls() - sink_before;
        assert!(
            ep_calls > 0 && sink_calls > 0,
            "required EP/tiled scorer not engaged"
        );
        assert!(!spec || rounds > 0, "DSpark did not engage");
        assert_eq!(tokens.len(), commits.len());
        let looped = looped(&tokens);
        let eligible = !warmup && !looped && tokens.len() >= 32;
        let sha = token_hash(&tokens);
        // Hashes, detokenization and stdout are deliberately outside timing.
        let text = self.tokenizer.decode(&tokens);
        // Log the live process gates, not the caller's intended arm. This is
        // critical for best-ab: the baseline and tuned selections change all
        // three grouped doors in one closure, and a stale argument can make a
        // receipt claim a gate state that never actually ran.
        let mirror_validate = self.gpu.grouped_mirror_validation_for_gate();
        println!(
            "MEASURE {{\"prompt\":{},\"mode\":\"{mode}\",\"arm\":\"{arm}\",\"sampler_order\":\"{sampler_order}\",\"vt\":\"{vt_label}\",\"mirror_validate\":{mirror_validate},\"route_validate\":{},\"gu_fuse\":{},\"m1_tc\":{},\"recent_rows\":{recent_rows},\"recent_gpu_bytes\":{recent_gpu_bytes:?},\"recent_gathers\":{recent_gathers},\"ordinal\":{ordinal},\"warmup\":{warmup},\"eligible\":{eligible},\"looped\":{looped},\"eos\":{eos},\"output_tokens\":{},\"restore_ns\":{restore_ns},\"decode_wall_ns\":{decode_wall_ns},\"commit_ns\":{commits:?},\"start_unix_ms\":{start_ms},\"end_unix_ms\":{end_ms},\"ep_calls\":{ep_calls},\"sink_calls\":{sink_calls},\"rounds\":{rounds},\"drafted\":{drafted},\"accepted\":{accepted},\"token_sha256\":\"{sha}\",\"text_sha256\":\"{:x}\",\"tokens\":{tokens:?}}}",
            self.tokens.len(),
            self.gpu.grouped_route_validation_for_gate(),
            self.gpu.grouped_gu_fuse_for_gate(),
            self.gpu.grouped_m1_tc_for_gate(),
            tokens.len(),
            Sha256::digest(text.as_bytes())
        );
        println!(
            "OUTPUT prompt={} arm={arm} ordinal={ordinal} text={text:?}",
            self.tokens.len()
        );
        tokens
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len())
            && args.get(3).is_none_or(|arg| matches!(
                arg.as_str(),
                "sampler-ab"
                    | "recent-ab"
                    | "vt-ab"
                    | "greedy-vt-ab"
                    | "mirror-ab"
                    | "route-ab"
                    | "gu-ab"
                    | "gu-compose-ab"
                    | "m1-tc-compose-ab"
                    | "best-ab"
                    | "route-stats"
            )),
        "usage: dsv4_decode_rate_gate <model-dir> <source.txt> [sampler-ab|recent-ab|vt-ab|greedy-vt-ab|mirror-ab|route-ab|gu-ab|gu-compose-ab|m1-tc-compose-ab|best-ab|route-stats]"
    );
    let sampler_ab = args.get(3).is_some_and(|arg| arg == "sampler-ab");
    let recent_ab = args.get(3).is_some_and(|arg| arg == "recent-ab");
    let vt_ab = args.get(3).is_some_and(|arg| arg == "vt-ab");
    let greedy_vt_ab = args.get(3).is_some_and(|arg| arg == "greedy-vt-ab");
    let mirror_ab = args.get(3).is_some_and(|arg| arg == "mirror-ab");
    let route_ab = args.get(3).is_some_and(|arg| arg == "route-ab");
    let gu_ab = args.get(3).is_some_and(|arg| arg == "gu-ab");
    let gu_compose_ab = args.get(3).is_some_and(|arg| arg == "gu-compose-ab");
    let m1_tc_compose_ab = args.get(3).is_some_and(|arg| arg == "m1-tc-compose-ab");
    let best_ab = args.get(3).is_some_and(|arg| arg == "best-ab");
    let route_stats = args.get(3).is_some_and(|arg| arg == "route-stats");
    let gate_repeats = std::env::var("MEMRA_DSV4_GATE_REPEATS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&repeats| repeats > 0)
        .unwrap_or(3);
    if recent_ab
        || vt_ab
        || greedy_vt_ab
        || mirror_ab
        || route_ab
        || gu_ab
        || gu_compose_ab
        || m1_tc_compose_ab
        || best_ab
        || route_stats
    {
        assert_eq!(
            dsv4_sampler_order().unwrap(),
            Dsv4SamplerOrder::Radix,
            "grid comparison keeps radix sampling fixed"
        );
    }
    for (name, expected) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_SINK_SCORE", "tiled"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "requires {name}"
        );
    }
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(tokens.len() >= 8192, "do not pad or repeat source");
    println!(
        "PROTOCOL engine_decode_only=true HTTP=false counts=[256,8192] new_tokens=256 chunk=512 active_C4=true ABBAx3_N6_each=true gate_repeats={gate_repeats} sampler_ab={sampler_ab} recent_ab={recent_ab} vt_ab={vt_ab} greedy_vt_ab={greedy_vt_ab} mirror_ab={mirror_ab} route_ab={route_ab} gu_ab={gu_ab} gu_compose_ab={gu_compose_ab} m1_tc_compose_ab={m1_tc_compose_ab} best_ab={best_ab} route_stats={route_stats} warmups=each_arm sample_T=1 top_p=1 top_k=0 seed=20260906 EOS=respected loops=excluded source_sha256={:x}",
        Sha256::digest(source.as_bytes())
    );
    let gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8192 + 256 + 96).expect("model");
    assert!(gpu.matrix_moe_enabled());
    for count in [256, 8192] {
        let prompt = &tokens[..count];
        let capacity = count + 256 + 96;
        println!("INPUT count={count} token_sha256={}", token_hash(prompt));
        let mut state = gpu
            .alloc_decode_state_host_c4(capacity, 512)
            .expect("host state");
        let mut draft = gpu.dspark_alloc_state().expect("draft state");
        drain(&gpu);
        let prefill_start = Instant::now();
        let logits = gpu
            .dspark_prefill_prime_chunked(prompt, &mut state, &mut draft, 512)
            .expect("prefill");
        drain(&gpu);
        println!(
            "PREFILL count={count} seconds={:.6} not_in_decode_measurement=true",
            prefill_start.elapsed().as_secs_f64()
        );
        let host = gpu.snapshot_decode_state(&state).expect("trunk snapshot");
        let host_draft = gpu.snapshot_dspark_state(&draft).expect("draft snapshot");
        drop(state);
        drop(draft);
        let ready = ReadyPrompt {
            gpu: &gpu,
            tokenizer: &tokenizer,
            tokens: prompt,
            logits: &logits,
            trunk: &host,
            draft: &host_draft,
            capacity,
            new_tokens: 256,
        };
        if route_stats {
            let before = gpu.ep_route_stats();
            let tokens = ready.measure(false, false, 0, false, 0, Dsv4Vt::Off, true);
            let stats = gpu.ep_route_stats().delta(before);
            println!(
                "EP_ROUTE_STATS prompt={} output_tokens={} calls={} one_row_calls={} local_slots={} peer_slots={} busier_slots={} local_hist={:?} busier_hist={:?}",
                count,
                tokens.len(),
                stats.calls,
                stats.one_row_calls,
                stats.local_slots,
                stats.peer_slots,
                stats.busier_slots,
                stats.local_hist,
                stats.busier_hist,
            );
            println!(
                "EXACT prompt={count} route-stats plain output captured; timing is diagnostic only"
            );
            continue;
        }
        let gu_toggle = gu_ab || gu_compose_ab || m1_tc_compose_ab || best_ab;
        let select = |tuned: bool| {
            drain(&gpu);
            if sampler_ab {
                set_dsv4_sampler_order_for_gate(Some(if tuned {
                    Dsv4SamplerOrder::Radix
                } else {
                    Dsv4SamplerOrder::Comparison
                }));
            }
            if best_ab {
                gpu.set_grouped_mirror_validation_for_gate(!tuned);
                gpu.set_grouped_route_validation_for_gate(!tuned);
                gpu.set_grouped_gu_fuse_for_gate(tuned);
                gpu.set_grouped_m1_tc_for_gate(tuned);
                gpu.set_c4_host_copy_elision_for_gate(tuned);
            } else if m1_tc_compose_ab {
                // Composition posture: GU fusion is held ON for both arms,
                // while the identity/perf variable is only the tensor-core
                // m_e=1 tail. Validation is held OFF on both arms so the
                // receipt prices the same production-shaped boundary.
                gpu.set_grouped_mirror_validation_for_gate(false);
                gpu.set_grouped_route_validation_for_gate(false);
                gpu.set_grouped_gu_fuse_for_gate(true);
                gpu.set_grouped_m1_tc_for_gate(tuned);
            } else {
                if mirror_ab || gu_compose_ab {
                    gpu.set_grouped_mirror_validation_for_gate(!tuned);
                }
                if route_ab || gu_compose_ab {
                    gpu.set_grouped_route_validation_for_gate(!tuned);
                }
                if gu_toggle {
                    gpu.set_grouped_gu_fuse_for_gate(tuned);
                }
                gpu.set_grouped_m1_tc_for_gate(false);
            }
        };
        let mirror_for = |tuned: bool| {
            if best_ab {
                !tuned
            } else if m1_tc_compose_ab {
                false
            } else if mirror_ab || gu_compose_ab {
                !tuned
            } else {
                true
            }
        };
        let vt_for = |tuned: bool| {
            if (vt_ab || greedy_vt_ab) && tuned {
                resolve_vt(Some("slot"), Some("0.5"), None).expect("slot@0.5")
            } else {
                Dsv4Vt::Off
            }
        };
        // Keep every mode that changes `select(tuned)` in this branch. Omitting
        // one silently falls into the default six-row schedule and produces
        // an incomplete receipt with no tuned arm (the best-ab failure mode).
        if sampler_ab
            || vt_ab
            || mirror_ab
            || route_ab
            || gu_ab
            || gu_compose_ab
            || m1_tc_compose_ab
            || best_ab
        {
            select(false);
        }
        let greedy_mode = greedy_vt_ab;
        let expected = ready.measure(
            false,
            greedy_mode,
            0,
            true,
            0,
            vt_for(false),
            mirror_for(false),
        );
        assert_eq!(
            expected,
            ready.measure(
                true,
                greedy_mode,
                1,
                true,
                0,
                vt_for(false),
                mirror_for(false),
            ),
            "warmup plain/DSpark output identity"
        );
        if sampler_ab
            || recent_ab
            || vt_ab
            || greedy_vt_ab
            || mirror_ab
            || route_ab
            || gu_ab
            || gu_compose_ab
            || m1_tc_compose_ab
            || best_ab
        {
            if !greedy_vt_ab {
                let frozen = if count == 256 {
                    "6a7d95b9ebe8714ab3ca75b6cbd8feced699937f08d605fd3f822a37bbdd115b"
                } else {
                    "7af072c1cdbde293dfa88b1dff10326d33ae6e7531eecb7b43585e3ba9146f33"
                };
                assert_eq!(
                    token_hash(&expected),
                    frozen,
                    "pre-rewrite frozen output stream"
                );
            }
            select(true);
            let tuned_recent = if recent_ab { 512 } else { 0 };
            assert_eq!(
                expected,
                ready.measure(
                    false,
                    greedy_mode,
                    2,
                    true,
                    tuned_recent,
                    vt_for(true),
                    mirror_for(true),
                )
            );
            assert_eq!(
                expected,
                ready.measure(
                    true,
                    greedy_mode,
                    3,
                    true,
                    tuned_recent,
                    vt_for(true),
                    mirror_for(true),
                )
            );
            let schedule = [
                (false, false),
                (true, false),
                (true, true),
                (false, true),
                (false, true),
                (true, true),
                (true, false),
                (false, false),
            ]
            .repeat(gate_repeats);
            for (ordinal, (tuned, spec)) in schedule.into_iter().enumerate() {
                select(tuned);
                assert_eq!(
                    expected,
                    ready.measure(
                        spec,
                        greedy_mode,
                        ordinal + 4,
                        false,
                        if recent_ab && tuned { 512 } else { 0 },
                        vt_for(tuned),
                        mirror_for(tuned)
                    ),
                    "sampler/decoder output identity"
                );
            }
            if sampler_ab {
                set_dsv4_sampler_order_for_gate(None);
            }
            if mirror_ab {
                gpu.set_grouped_mirror_validation_for_gate(true);
            }
            if route_ab {
                gpu.set_grouped_route_validation_for_gate(true);
            }
            if gu_compose_ab {
                gpu.set_grouped_mirror_validation_for_gate(true);
                gpu.set_grouped_route_validation_for_gate(true);
            }
            if m1_tc_compose_ab || best_ab {
                gpu.set_grouped_m1_tc_for_gate(false);
                gpu.set_grouped_mirror_validation_for_gate(true);
                gpu.set_grouped_route_validation_for_gate(true);
                if best_ab {
                    gpu.set_c4_host_copy_elision_for_gate(false);
                }
            }
            if gu_toggle {
                gpu.clear_grouped_gu_fuse_for_gate();
            }
        } else {
            for (ordinal, spec) in [false, true, true, false].repeat(3).into_iter().enumerate() {
                assert_eq!(
                    expected,
                    ready.measure(spec, false, ordinal + 2, false, 0, Dsv4Vt::Off, true,),
                    "sampled output identity"
                );
            }
        }
        println!(
            "EXACT prompt={count} all sampled outputs identical; inspect exclusions before computing speed"
        );
    }
    println!(
        "PASS sampled plain/DSpark decode timing and output identity; HTTP and concurrency remain separate"
    );
}

#[cfg(test)]
mod tests {
    use super::looped;
    #[test]
    fn loop_exclusion_needs_a_sustained_repeated_block() {
        assert!(!looped(&[1, 2, 1, 2, 1, 2]));
        assert!(looped(&[1; 32]));
        assert!(looped(&[1, 2, 3, 4, 5, 6, 7, 8].repeat(4)));
        assert!(!looped(&(0..256).collect::<Vec<_>>()));
        let mut prefixed = vec![99, 98, 97];
        prefixed.extend([1, 2, 3, 4].repeat(8));
        assert!(looped(&prefixed));
    }
}
