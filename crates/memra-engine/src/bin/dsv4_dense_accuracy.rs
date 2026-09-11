//! Paired task accuracy on ANY frozen short-answer set, scalar dense against CUTLASS
//! dense, both arms in ONE process over ONE load, one item at a time.
//!
//! WHY A THIRD ACCURACY BINARY. `dsv4_quality_twin` (darklanes #545) and
//! `dsv4_task_accuracy` (memra #485) each run one arm per process: their axes are
//! load-time (expert program, HC slices), so two loads is the only pairing they can
//! offer. The dense axis flips through the process-local gate arm, so this binary
//! pairs strictly harder: arm A then arm B on every item, fresh state each, nothing
//! able to drift between arms except the arm itself.
//!
//! WHY THE PRIME DIFFERS FROM #545. That protocol primes decode-shaped (one token,
//! then a token-at-a-time walk), under which the dense path can never engage: it
//! admits exactly m=32 and declines everything else. An accuracy control run that
//! way would compare the scalar kernel to itself and report a vacuous null. This
//! driver primes the whole prompt chunked at width 32, which is the only engaging
//! width; both arms share the protocol, and per-item split-K deltas prove which arm
//! ran. Greedy decode stays token-at-a-time (scalar in both arms, as served); the
//! drift under test enters through the primed state.
//!
//! Greedy is the INSTRUMENT, never a serving shape. A paired accuracy comparison
//! needs a deterministic decode or the arms differ by sampling noise instead of by
//! the numeric class under test. Prod serves the vendor-recommended sampled default.
//!
//! WHY IT ALSO EMITS NLL. A null on a paired accuracy set is only meaningful if the
//! set could have registered a flip. The negative log likelihood of each emitted
//! token is the set's own entropy receipt, in the same units as the drift rows'
//! per-panel means, so a low-entropy set that could never have flipped is visible
//! in the receipt instead of being read as equivalence. NLL is scored out of
//! process with the set's own frozen rules; this binary prints no threshold.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, arm_dense_cutlass_for_gate, dense_cutlass_armed_for_gate,
    dense_cutlass_counts_for_gate,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};

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
        4,
        "usage: dsv4_dense_accuracy <model> <prompts-dir> <new-out-dir>"
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

    // The dense path must be PRESENT (linked archive) before any item runs: a null
    // weak symbol reports `None`, which is a different fact from "off", and a cell
    // whose two arms are one program must fail, not report a perfect null.
    match dense_cutlass_armed_for_gate() {
        None => {
            println!("ACC_FAIL this binary does not contain the dense CUTLASS path");
            std::process::exit(1);
        }
        Some(armed) => println!("PRESENT armed_at_start={armed}"),
    }
    // RED ARM. A check that has never failed is not a check, so this env secretly
    // arms the cutlass path behind the scalar arm's back on item 0, and the run must
    // refuse at the scalar engagement assert: a scalar arm that ran split-K calls is
    // not the arm its label says. Without this, both arms could be cutlass and the
    // paired comparison would report a perfect null, which is the answer everyone is
    // hoping for and the worst possible thing to report by accident.
    let red_arm = std::env::var_os("MEMRA_DENSE_ACC_RED_ARM_FORCE_CUTLASS").is_some();
    if red_arm {
        println!("RED_ARM force_cutlass=1; the scalar engagement assert MUST refuse");
    }

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
    for arm in ["scalar", "cutlass"] {
        std::fs::create_dir(out.join(arm)).expect("arm receipt directory");
    }
    println!(
        "PROTOCOL arms=scalar,cutlass paired=same_process_same_load_per_item prime=chunked_width32 \
         ep=off topology=pp2 prompts={} max_new={LIMIT} capacity={capacity} sampler=argmax \
         ties=lowest_id drafter=off stop=eos instrument=greedy_only_never_a_serving_shape",
        prompts.len()
    );

    let gpu = Dsv4Gpu::load(
        Path::new(&args[1]),
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        capacity,
    )
    .expect("native model");
    assert!(
        !gpu.topology().is_tp_ep(),
        "this control runs the served PP-2 topology"
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

    let mut nll_sum = 0.0f64;
    let mut nll_n = 0usize;
    let mut arm_splitk = [0u64; 2];
    let mut arm_flips = [0u64; 2];
    for (i, tokens) in prompts.iter().enumerate() {
        let mut arm_out: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        let mut arm_nll: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut arm_eos = [false, false];
        for (a, cutlass) in [false, true].iter().enumerate() {
            let arm = if *cutlass { "cutlass" } else { "scalar" };
            arm_dense_cutlass_for_gate(*cutlass).expect("select arm");
            if red_arm && i == 0 && !cutlass {
                // Behind the scalar arm's back: the engagement assert below must fire.
                arm_dense_cutlass_for_gate(true).expect("red arm");
            }
            let mut state = gpu
                .alloc_decode_state_for_transient(capacity, 1)
                .expect("fresh state per item per arm; no cache crosses items or arms");
            let c0 = dense_cutlass_counts_for_gate().expect("counts");
            // Chunked prime at width 32, the only width the dense path admits.
            // A decode-shaped prime (width 1) would compare the scalar kernel to
            // itself; see the header.
            gpu.prefill_with_cache_chunked(tokens, &mut state, 32)
                .expect("prime");
            assert_eq!(state.pos, tokens.len());
            let c1 = dense_cutlass_counts_for_gate().expect("counts");
            let splitk = c1.splitk - c0.splitk;
            let flips = c1.ws_device_flips - c0.ws_device_flips;
            arm_splitk[a] += splitk;
            arm_flips[a] += flips;
            if *cutlass {
                assert!(
                    splitk > 0,
                    "item {i}: the cutlass arm ran no split-K calls, so the arms are one program"
                );
                assert!(
                    flips > 0,
                    "item {i}: the cutlass arm crossed no device boundary, so the item \
                     never tested the device-keyed workspace"
                );
            } else {
                assert_eq!(
                    splitk, 0,
                    "item {i}: the scalar arm ran {splitk} split-K calls, so the arm is \
                     not what its label says"
                );
            }
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
            // The device greedy path is what #545 and #467 measured. Item 0 of the
            // scalar arm is decoded a second time through it, from a fresh state,
            // and must produce the same tape: without this the NLL column would be
            // free to come from a different decode than the accuracy column.
            if i == 0 && !cutlass {
                arm_dense_cutlass_for_gate(false).expect("reselect scalar");
                let mut check = gpu
                    .alloc_decode_state_for_transient(capacity, 1)
                    .expect("state");
                gpu.prefill_with_cache_chunked(tokens, &mut check, 32)
                    .expect("prime");
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
            arm_out[a] = generated;
            arm_nll[a] = nlls;
            arm_eos[a] = stopped;
        }
        for (a, cutlass) in [false, true].iter().enumerate() {
            let arm = if *cutlass { "cutlass" } else { "scalar" };
            let generated = &arm_out[a];
            let nlls = &arm_nll[a];
            let stopped = arm_eos[a];
            let content = if stopped {
                &generated[..generated.len() - 1]
            } else {
                &generated[..]
            };
            let text = tokenizer.decode(content);
            let dir = out.join(arm);
            std::fs::write(dir.join(format!("{i:03}.txt")), &text).expect("item output");
            let raw: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
            nll_sum += nlls.iter().sum::<f64>();
            nll_n += nlls.len();
            let nll_text: Vec<String> = nlls.iter().map(|v| format!("{v:.6}")).collect();
            let row_json = format!(
                "{{\"index\":{i},\"arm\":\"{arm}\",\"axis\":\"dense_cutlass\",\
                 \"prompt_tokens\":{tokens:?},\"prompt_tokens_sha256\":\"{:x}\",\"tokens\":{generated:?},\
                 \"eos\":{stopped},\"output_sha256\":\"{:x}\",\"nll\":[{}],\"mean_nll\":{mean:.6}}}",
                Sha256::digest(&raw),
                Sha256::digest(text.as_bytes()),
                nll_text.join(",")
            );
            // One rows.jsonl per arm, in the arm's own directory, in the shape the
            // frozen scorers read: {arm}/rows.jsonl plus {arm}/{i:03}.txt.
            let mut rows = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out.join(arm).join("rows.jsonl"))
                .expect("rows");
            writeln!(rows, "{row_json}").expect("row");
            println!(
                "ITEM_DONE arm={arm} item={i} input={} generated={} eos={stopped} mean_nll={mean:.6}",
                tokens.len(),
                generated.len()
            );
        }
    }

    // Engagement totals: the scalar arm must never have run the path, the cutlass
    // arm must have run it on every item and crossed cards doing so. A paired
    // comparison whose arms are not what the labels say is a PASS that lies.
    assert_eq!(
        arm_splitk[0], 0,
        "scalar arm ran {} split-K calls over the run",
        arm_splitk[0]
    );
    assert!(
        arm_splitk[1] > 0 && arm_flips[1] > 0,
        "cutlass arm ran {} split-K calls with {} flips over the run",
        arm_splitk[1],
        arm_flips[1]
    );
    println!(
        "ENGAGEMENT scalar_splitk=0 cutlass_splitk={} cutlass_flips={}",
        arm_splitk[1], arm_flips[1]
    );
    println!(
        "SET_ENTROPY generated_tokens={nll_n} mean_nll={:.6}",
        nll_sum / nll_n as f64
    );
    println!(
        "SUMMARY_TASK_EVIDENCE_PASS arms=scalar,cutlass items={} paired=same_process; \
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
