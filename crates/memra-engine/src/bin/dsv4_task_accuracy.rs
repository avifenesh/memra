//! Paired task accuracy on ANY frozen short-answer set, at the served realization.
//!
//! WHY A SECOND ACCURACY BINARY. `dsv4_quality_twin` (darklanes #545) and the matrix
//! lane's `dsv4_program_accuracy` (memra #467) each hard-wire one comparison: a fixed
//! 300-item count, one arm axis, and `MEMRA_DSV4_HC_DOT_SPLIT` pinned to a constant
//! inside the binary. Neither can answer the question the Hebrew accuracy set was
//! built for, because that set is 100 items and because HC S16 is one of the arms
//! rather than a fixed background. This binary keeps every measurement rule of those
//! two identical and moves exactly three things into the caller's hands: the item
//! count, the expert program, and the HC dot-split slice count.
//!
//! Greedy is the INSTRUMENT, never a serving shape. A paired accuracy comparison
//! needs a deterministic decode or the arms differ by sampling noise instead of by
//! the numeric class under test. Prod serves the vendor-recommended sampled default.
//!
//! WHY IT ALSO EMITS NLL. A null on a paired accuracy set is only meaningful if the
//! set could have registered a flip. The reference negative log likelihood of each
//! emitted token is the set's own entropy receipt, in the same units as the drift
//! tape's per-panel `nll_a_mean`, so a low-entropy set that could never have flipped
//! is visible in the receipt instead of being read as equivalence.
use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};

unsafe extern "C" {
    fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
}

const LIMIT: usize = 128;

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

/// Negative log likelihood of `token` under `row`, in nats, computed with the
/// max-shifted log-sum-exp so a large logit cannot overflow the exponential.
fn nll(row: &[f32], token: u32) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
    let sum: f64 = row.iter().map(|v| (*v as f64 - max).exp()).sum();
    assert!(sum.is_finite() && sum > 0.0);
    max + sum.ln() - row[token as usize] as f64
}

fn main() {
    // Pinned at process startup, before any worker thread exists, so a stray bench
    // export cannot make one arm a different numeric class than the label says.
    // MEMRA_DSV4_HC_DOT_SPLIT is deliberately NOT pinned here: it is an arm.
    unsafe {
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
        // memra #458: matrix under PP-2 boots clean on default-ON split-K and then
        // fails every request.
        std::env::set_var("MEMRA_DSV4_MOE_M1_SPLITK", "0");
    }

    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: dsv4_task_accuracy <model> <prompts-dir> <new-out-dir> <arm-label>"
    );
    let arm = args[4].as_str();
    assert!(
        !arm.is_empty() && arm.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
        "arm label is written into every row; keep it a plain token"
    );

    // The served realization, asserted rather than assumed. Identical across arms.
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
    // f32x"), so a bench export must not leak into a serving-shaped run.
    assert!(std::env::var_os("MEMRA_DSV4_SMALL_KERNEL_DIET").is_none());

    // Arm axis 1: the expert program. Named explicitly in both arms, because a
    // receipt that says "the default" stops being readable the day the default moves.
    let program = std::env::var("MEMRA_DSV4_MOE_PROGRAM")
        .expect("name the expert program explicitly, even when it is the default");
    assert!(
        program == "reference" || program == "matrix",
        "MEMRA_DSV4_MOE_PROGRAM must be reference or matrix, got {program}"
    );
    // Arm axis 2: HC dot split slices. The kernel's own parser maps unset AND "1"
    // AND "16" to 16 and anything unrecognised to 0, so only the two explicit
    // values this control means are accepted.
    let hc = std::env::var("MEMRA_DSV4_HC_DOT_SPLIT")
        .expect("name the HC dot-split slice count explicitly, even at its default");
    assert!(
        hc == "0" || hc == "16",
        "MEMRA_DSV4_HC_DOT_SPLIT must be 0 or 16"
    );
    let slices: i32 = hc.parse().unwrap();

    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("tokenizer");
    assert_eq!(tokenizer.eos_id(), 1);
    let mut paths: Vec<_> = std::fs::read_dir(&args[2])
        .expect("prompts dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "an empty prompt set would PASS while measuring nothing"
    );
    let prompts: Vec<_> = paths
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).expect("prompt");
            // A prompt that lost its template would score a different task and the
            // comparison would still look fine.
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
        "PROTOCOL arm={arm} program={program} hc_dot_split={slices} ep=off topology=pp2 \
         prompts={} max_new={LIMIT} capacity={capacity} sampler=argmax ties=lowest_id \
         drafter=off splitk=0 stop=eos instrument=greedy_only_never_a_serving_shape",
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
        program == "matrix",
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

    let mut rows = std::fs::File::create(out.join("rows.jsonl")).expect("rows");
    let routes_at_start = gpu.grouped_device_route_calls();
    let mut nll_sum = 0.0f64;
    let mut nll_n = 0usize;
    for (i, tokens) in prompts.iter().enumerate() {
        let mut state = gpu
            .alloc_decode_state_for_transient(capacity, 1)
            .expect("fresh state per item; no cache crosses items");
        // The slice count is thread local in the kernel translation unit, so the
        // value that matters is the one on the thread that runs this decode.
        assert_eq!(
            unsafe { memra_dsv4_hc_dot_split_slices_for_gate() },
            slices,
            "HC dot-split slices on the decode thread do not match the arm"
        );
        gpu.prefill_with_cache_chunked(&tokens[..1], &mut state, 1)
            .expect("prime");
        for &token in &tokens[1..] {
            gpu.decode_step_device_logits(token, &mut state)
                .expect("prompt walk");
        }
        assert_eq!(state.pos, tokens.len());
        let mut row = gpu.read_decode_logits_for_gate(&state).expect("logits");
        let mut carry = argmax(&row);
        let mut generated = Vec::new();
        let mut nlls: Vec<f64> = Vec::new();
        let mut stopped = false;
        for step in 0..LIMIT {
            nlls.push(nll(&row, carry));
            generated.push(carry);
            if carry == tokenizer.eos_id() {
                stopped = true;
                break;
            }
            if step + 1 < LIMIT {
                gpu.decode_step_device_logits(carry, &mut state)
                    .expect("generation step");
                row = gpu.read_decode_logits_for_gate(&state).expect("logits");
                carry = argmax(&row);
            }
        }
        assert_eq!(nlls.len(), generated.len());
        // The device greedy path is what #545 and #467 measured. Item 0 is decoded a
        // second time through it, from a fresh state, and must produce the same tape:
        // without this the NLL column would be free to come from a different decode
        // than the accuracy column.
        if i == 0 {
            let mut check = gpu
                .alloc_decode_state_for_transient(capacity, 1)
                .expect("state");
            gpu.prefill_with_cache_chunked(&tokens[..1], &mut check, 1)
                .expect("prime");
            for &token in &tokens[1..] {
                gpu.decode_step_device_logits(token, &mut check)
                    .expect("prompt walk");
            }
            let mut c = argmax(&gpu.read_decode_logits_for_gate(&check).expect("logits"));
            let mut device_tape = Vec::new();
            for step in 0..LIMIT {
                device_tape.push(c);
                if c == tokenizer.eos_id() {
                    break;
                }
                if step + 1 < LIMIT {
                    c = gpu.decode_step_greedy(c, &mut check).expect("greedy step");
                }
            }
            assert_eq!(
                device_tape, generated,
                "host argmax over read logits and the device greedy step disagree"
            );
            println!(
                "GREEDY_PATH_EQUIVALENT arm={arm} item=0 tokens={}",
                generated.len()
            );
        }
        let content = if stopped {
            &generated[..generated.len() - 1]
        } else {
            &generated[..]
        };
        let text = tokenizer.decode(content);
        std::fs::write(out.join(format!("{i:03}.txt")), &text).expect("item output");
        let raw: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
        nll_sum += nlls.iter().sum::<f64>();
        nll_n += nlls.len();
        let nll_text: Vec<String> = nlls.iter().map(|v| format!("{v:.6}")).collect();
        let row_json = format!(
            "{{\"index\":{i},\"arm\":\"{arm}\",\"program\":\"{program}\",\"hc_dot_split\":{slices},\
             \"prompt_tokens\":{tokens:?},\"prompt_tokens_sha256\":\"{:x}\",\"tokens\":{generated:?},\
             \"eos\":{stopped},\"output_sha256\":\"{:x}\",\"nll\":[{}],\"mean_nll\":{mean:.6}}}",
            Sha256::digest(&raw),
            Sha256::digest(text.as_bytes()),
            nll_text.join(",")
        );
        writeln!(rows, "{row_json}").expect("row");
        rows.flush().expect("flush");
        println!(
            "ITEM_DONE arm={arm} item={i} input={} generated={} eos={stopped} mean_nll={mean:.6}",
            tokens.len(),
            generated.len()
        );
    }

    // Engagement in both directions: a reference arm that touched the grouped
    // executor, or a matrix arm that never did, is a PASS that lies about which
    // program produced these answers.
    let routes = gpu.grouped_device_route_calls() - routes_at_start;
    if program == "matrix" {
        assert!(routes > 0, "matrix arm never engaged the grouped executor");
    } else {
        assert_eq!(routes, 0, "reference arm engaged the grouped executor");
    }
    println!("ENGAGEMENT arm={arm} program={program} grouped_device_route_calls={routes}");
    println!(
        "SET_ENTROPY arm={arm} generated_tokens={nll_n} mean_nll={:.6}",
        nll_sum / nll_n as f64
    );
    println!(
        "SUMMARY_TASK_EVIDENCE_PASS arm={arm} items={} program={program} hc_dot_split={slices}; \
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
    #[test]
    fn nll_of_a_uniform_row_is_the_log_of_its_width() {
        let row = [0.0f32; 8];
        assert!((super::nll(&row, 3) - 8f64.ln()).abs() < 1e-12);
    }
    #[test]
    fn nll_survives_logits_that_would_overflow_a_naive_exponential() {
        // exp(400) is inf in f64; the max shift is what keeps this finite.
        let row = [400.0f32, 399.0, 0.0];
        let v = super::nll(&row, 0);
        assert!(v.is_finite() && v > 0.0 && v < 1.0, "{v}");
    }
    #[test]
    fn nll_is_larger_for_the_less_likely_token() {
        let row = [5.0f32, 1.0, 1.0];
        assert!(super::nll(&row, 1) > super::nll(&row, 0));
    }
}
