//! Bounded plain greedy task evidence. Inputs are vendor-rendered UTF-8 chat prompts.
//! Each invocation is one numeric class, with fresh cache state for every item.
use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{io::Write, path::Path};

unsafe extern "C" {
    fn memra_dsv4_hc_dot_split_set_for_gate(slices: i32) -> i32;
    fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
}
const LIMIT: usize = 128;
fn argmax(row: &[f32]) -> u32 {
    assert!(!row.is_empty() && row.iter().all(|v| v.is_finite()));
    let mut best = 0;
    for i in 1..row.len() {
        if row[i] > row[best] { best = i; }
    }
    best as u32
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "usage: dsv4_quality_twin <model> <prompts-dir> <new-output-dir> <A|B|C>");
    let arm = args[4].as_str();
    assert!(matches!(arm, "A" | "B" | "C"));
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"), ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"), ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_DSV4_EP", "pair"), ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"), ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"), ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"), ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] { assert_eq!(std::env::var(name).as_deref(), Ok(value), "{name}"); }
    assert!(!memra_engine::dsv4_gpu::dsv4_prof_on());
    if arm == "A" {
        assert_eq!(std::env::var("MEMRA_DSV4_MOE_M1_SPLITK").as_deref(), Ok("0"));
    } else {
        assert!(std::env::var_os("MEMRA_DSV4_MOE_M1_SPLITK").is_none());
    }
    assert_eq!(memra_engine::moe_m1_graph_splitk_on(), arm != "A");
    let slices = if arm == "C" { 16 } else { 0 };
    assert_eq!(std::env::var("MEMRA_DSV4_HC_DOT_SPLIT").unwrap(), slices.to_string());
    // The experimental env parser maps only "1" to 16. Use its existing gate
    // selector explicitly, on the same host thread that executes every decode.
    unsafe {
        assert_eq!(memra_dsv4_hc_dot_split_set_for_gate(slices), 0);
        assert_eq!(memra_dsv4_hc_dot_split_slices_for_gate(), slices);
    }
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).unwrap();
    assert_eq!(tokenizer.eos_id(), 1);
    let mut paths: Vec<_> = std::fs::read_dir(&args[2]).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "txt")).collect();
    paths.sort();
    assert!(paths.len() >= 300);
    let prompts: Vec<_> = paths.iter().map(|p| {
        let text = std::fs::read_to_string(p).unwrap();
        assert!(text.starts_with("<｜begin▁of▁sentence｜>"));
        assert!(text.ends_with("<｜Assistant｜></think>"));
        let ids = tokenizer.encode(&text, false);
        assert_eq!(ids[0], 0);
        assert!(!ids.contains(&tokenizer.eos_id()));
        ids
    }).collect();
    let capacity = prompts.iter().map(Vec::len).max().unwrap() + LIMIT + 8;
    assert!(capacity <= 4096);
    let out = Path::new(&args[3]);
    std::fs::create_dir(out).unwrap();
    println!("PROTOCOL arm={arm} graph_splitk={} hc_slices={slices} prompts={} max_new={LIMIT} capacity={capacity} sampler=argmax ties=lowest_id dspark=off stop=eos", arm != "A", prompts.len());
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu = Dsv4Gpu::load(Path::new(&args[1]), &[0,1], ActQuantVariant::RefFp8Round, capacity).unwrap();
    assert!(gpu.topology().is_tp_ep() && gpu.topology().layers == 43 && gpu.small_kernel_diet_enabled());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);
    let mut rows = std::fs::File::create(out.join("rows.jsonl")).unwrap();
    for (i, tokens) in prompts.iter().enumerate() {
        assert_eq!(unsafe { memra_dsv4_hc_dot_split_slices_for_gate() }, slices);
        let mut state = gpu.alloc_decode_state_for_transient(capacity, 1).unwrap();
        gpu.prefill_with_cache_chunked(&tokens[..1], &mut state, 1).unwrap();
        for &token in &tokens[1..] { gpu.decode_step_device_logits(token, &mut state).unwrap(); }
        assert_eq!(state.pos, tokens.len());
        let mut carry = argmax(&gpu.read_decode_logits_for_gate(&state).unwrap());
        let mut generated = Vec::new();
        let mut stopped = false;
        for step in 0..LIMIT {
            generated.push(carry);
            if carry == tokenizer.eos_id() { stopped = true; break; }
            if step + 1 < LIMIT { carry = gpu.decode_step_greedy(carry, &mut state).unwrap(); }
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0,0]);
        }
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0,0]);
        let content = if stopped { &generated[..generated.len()-1] } else { &generated[..] };
        let text = tokenizer.decode(content);
        std::fs::write(out.join(format!("{i:03}.txt")), &text).unwrap();
        let raw: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let row = format!("{{\"index\":{i},\"arm\":\"{arm}\",\"prompt_tokens\":{tokens:?},\"prompt_tokens_sha256\":\"{:x}\",\"tokens\":{generated:?},\"eos\":{stopped},\"output_sha256\":\"{:x}\"}}", Sha256::digest(&raw), Sha256::digest(text.as_bytes()));
        writeln!(rows, "{row}").unwrap(); rows.flush().unwrap();
        println!("ITEM_DONE arm={arm} item={i} input={} generated={} eos={stopped}", tokens.len(), generated.len());
    }
    println!("SUMMARY_TASK_EVIDENCE_PASS arm={arm} items={}", prompts.len());
}
#[cfg(test)]
mod tests {
    #[test]
    fn argmax_lowest_id_on_ties() { assert_eq!(super::argmax(&[1.0,3.0,3.0,-1.0]), 1); }
    #[test]
    #[should_panic]
    fn nonfinite_logits_refused() { super::argmax(&[0.0,f32::NAN]); }
}
