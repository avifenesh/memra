//! Paired task accuracy for the DSV4 expert program: reference against matrix.
//!
//! WHY THIS EXISTS AND WHY IT IS NOT OPTIONAL. Drift rows say two programs differ.
//! They cannot say whether a changed answer is a WORSE answer. The house precedent
//! for shipping a numeric-class change (HC S16, owner accepted 2026-09-09) was
//! accepted on drift rows AND a 300-item paired task-accuracy control, 177/300
//! against 177/300 with exact McNemar p=1.00 (darklanes #545). Bringing a larger
//! decision with less evidence than the smaller one that set the precedent is not a
//! defensible ask, so this control runs alongside `dsv4_moe_program_class_gate`.
//!
//! DELIBERATELY THE SAME INSTRUMENT AS #545, so the two results are comparable:
//! the same frozen 300 items, the same prompts rendered by the vendor template, the
//! same greedy argmax with ties to the lowest id, the same 128-token bound, the same
//! stop-on-EOS rule, the same per-item output file plus row hash. Greedy here is the
//! INSTRUMENT, never a serving shape: a paired accuracy comparison needs a
//! deterministic decode or the arms differ by sampling noise instead of by program.
//!
//! ONE NUMERIC CLASS PER INVOCATION. The expert program is fixed for the whole
//! process, so no in-process setter can smear the arms together.
//!
//! WHAT IS DIFFERENT FROM #545, and why. #545 ran the TP/EP topology, which REFUSES
//! the reference program outright (`dsv4_gpu.rs:3072`), so it could only ever compare
//! arms WITHIN matrix. This runs the served PP-2 topology at `EP=off`, which is the
//! owner-ruled canonical arm: the criterion is whatever a default flip would actually
//! serve, and today's served program has both `MOE_PROGRAM` and `EP` unset.
//!
//! The matrix program's admission prerequisites are asserted in BOTH arms. They are
//! inert on the reference walk, which never enters the grouped executor, so holding
//! them fixed is what makes the expert program the only thing that differs.
//!
//! Usage: dsv4_program_accuracy <model> <prompts-dir> <new-out-dir> <REF|MAT>
use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};

/// #545's bound, kept so a truncation is a truncation in both campaigns.
const LIMIT: usize = 128;

/// Value descending, index ascending: the house tie ordering. A `fold` with `>=`
/// would take the LAST maximum and silently disagree with the served candidate
/// order on every tie.
fn argmax(row: &[f32]) -> u32 {
    assert!(!row.is_empty() && row.iter().all(|v| v.is_finite()));
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    // The same doors this lane's class gate pins, for the same reason: the axis under
    // test is the expert program, and a second moving arm makes the answer
    // un-attributable. Process startup, before any worker thread exists.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
        // memra #458: matrix under PP-2 boots clean on default-ON split-K and then
        // fails every request. Pinned off so this control measures the expert
        // program and not a second, separately gated numeric arm.
    }

    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: dsv4_program_accuracy <model> <prompts-dir> <new-out-dir> <REF|MAT>"
    );
    let arm = args[4].as_str();
    assert!(arm == "REF" || arm == "MAT", "arm must be REF or MAT");

    // The served program shape at the canonical realization, asserted rather than
    // assumed. Everything here is identical between the two arms EXCEPT the program.
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_DSV4_EP", "off"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    // Refused at load under PP-2 ("small-kernel diet requires all-layer TP/EP and
    // f32x"), so its presence in a bench env must not leak into a serving-shaped run.
    assert!(std::env::var_os("MEMRA_DSV4_SMALL_KERNEL_DIET").is_none());
    // The expert program stopped being an environment door on 2026-09-11
    // (memra #461): matrix is what loads, and the REF arm of this control is
    // selected by the gate arm, which is the only way in. The assertion that the
    // arm actually took is `grouped_device_route_calls` further down, which is
    // engagement rather than a restatement of what we asked for.
    let want_program = if arm == "MAT" { "matrix" } else { "reference" };
    if arm == "REF" {
        memra_engine::arm_reference_expert_program_for_gate();
    } else {
        memra_engine::disarm_reference_expert_program_for_gate();
    }

    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("tokenizer");
    assert_eq!(tokenizer.eos_id(), 1);
    let mut paths: Vec<_> = std::fs::read_dir(&args[2])
        .expect("prompts dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 300, "the frozen #545 set is exactly 300 items");
    let prompts: Vec<_> = paths
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).expect("prompt");
            // The vendor render, asserted: a prompt that lost its template would
            // score a different task and the comparison would still look fine.
            assert!(text.starts_with("<｜begin▁of▁sentence｜>"));
            assert!(text.ends_with("<｜Assistant｜></think>"));
            let ids = tokenizer.encode(&text, false);
            assert_eq!(ids[0], 0);
            assert!(!ids.contains(&tokenizer.eos_id()));
            ids
        })
        .collect();
    let capacity = prompts.iter().map(Vec::len).max().unwrap() + LIMIT + 8;
    assert!(capacity <= 4096);

    let out = Path::new(&args[3]);
    std::fs::create_dir(out).expect("create a new, nonexisting receipt directory");
    println!(
        "PROTOCOL arm={arm} program={want_program} ep=off topology=pp2 prompts={} max_new={LIMIT} \
         capacity={capacity} sampler=argmax ties=lowest_id drafter=off stop=eos \
         instrument=greedy_only_never_a_serving_shape",
        prompts.len()
    );

    let gpu = Dsv4Gpu::load(
        Path::new(&args[1]),
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        capacity,
    )
    .expect("native model");
    assert_eq!(
        gpu.matrix_moe_enabled(),
        arm == "MAT",
        "the loaded program must be the one the arm asked for"
    );
    assert!(
        !gpu.topology().is_tp_ep(),
        "this control runs the served PP-2 topology; TP/EP refuses the reference arm"
    );
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);

    let routes_at_start = gpu.grouped_device_route_calls();
    let mut rows = std::fs::File::create(out.join("rows.jsonl")).expect("rows");
    for (i, tokens) in prompts.iter().enumerate() {
        let mut state = gpu
            .alloc_decode_state_for_transient(capacity, 1)
            .expect("fresh state per item; no cache crosses items");
        gpu.prefill_with_cache_chunked(&tokens[..1], &mut state, 1)
            .expect("prime");
        for &token in &tokens[1..] {
            gpu.decode_step_device_logits(token, &mut state)
                .expect("prompt walk");
        }
        assert_eq!(state.pos, tokens.len());
        let mut carry = argmax(&gpu.read_decode_logits_for_gate(&state).expect("logits"));
        let mut generated = Vec::new();
        let mut stopped = false;
        for step in 0..LIMIT {
            generated.push(carry);
            if carry == tokenizer.eos_id() {
                stopped = true;
                break;
            }
            if step + 1 < LIMIT {
                carry = gpu
                    .decode_step_greedy(carry, &mut state)
                    .expect("greedy step");
            }
        }
        let content = if stopped {
            &generated[..generated.len() - 1]
        } else {
            &generated[..]
        };
        let text = tokenizer.decode(content);
        std::fs::write(out.join(format!("{i:03}.txt")), &text).expect("item output");
        let raw: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let row = format!(
            "{{\"index\":{i},\"arm\":\"{arm}\",\"prompt_tokens\":{tokens:?},\"prompt_tokens_sha256\":\"{:x}\",\"tokens\":{generated:?},\"eos\":{stopped},\"output_sha256\":\"{:x}\"}}",
            Sha256::digest(&raw),
            Sha256::digest(text.as_bytes())
        );
        writeln!(rows, "{row}").expect("row");
        rows.flush().expect("flush");
        println!(
            "ITEM_DONE arm={arm} item={i} input={} generated={} eos={stopped}",
            tokens.len(),
            generated.len()
        );
    }

    // Engagement, in both directions: a reference arm that touched the grouped
    // executor, or a matrix arm that never did, is a PASS that lies about which
    // program produced these 300 answers.
    let routes = gpu.grouped_device_route_calls() - routes_at_start;
    if arm == "MAT" {
        assert!(routes > 0, "matrix arm never engaged the grouped executor");
    } else {
        assert_eq!(routes, 0, "reference arm engaged the grouped executor");
    }
    println!("ENGAGEMENT arm={arm} program={want_program} grouped_device_route_calls={routes}");
    println!(
        "SUMMARY_TASK_EVIDENCE_PASS arm={arm} items={} program={want_program}; \
         accuracy is scored out of process, no threshold is chosen here",
        prompts.len()
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn argmax_takes_the_lowest_id_on_a_tie() {
        assert_eq!(super::argmax(&[1.0, 3.0, 3.0, -1.0]), 1);
        assert_eq!(super::argmax(&[-0.0, 0.0]), 0);
    }
    #[test]
    #[should_panic]
    fn nonfinite_logits_are_refused() {
        super::argmax(&[0.0, f32::NAN]);
    }
    #[test]
    #[should_panic]
    fn empty_logits_are_refused() {
        super::argmax(&[]);
    }
}
