//! dsv4-greedy-verify: CPU teacher-forcing verification of the GPU greedy continuation
//! (lane 4, gate (b) CPU side).
//!
//! Protocol (banked in RECEIPTS.md "Lane 4" before any run): greedy decoding is
//! causal-deterministic, so ONE CPU prefill over [prompt + gpu_tokens] with per-position
//! argmax is EQUIVALENT to generating the CPU greedy continuation and comparing:
//!   - if argmax(cpu_logits[p]) == seq[p+1] for every position, the CPU greedy sequence
//!     IS the GPU sequence (agreement proven for all n_new tokens);
//!   - the first disagreement position is exactly the first greedy divergence (the
//!     prefix up to it is identical), and both logit rows exist for analysis: the CPU
//!     row from this run, the GPU row from dsv4-gpu-greedy's banked logits bin.
//! The CPU forward is the lane-3 oracle (memra_gguf::dsv4_forward) — the same program
//! that passed the fixture gate 15/15 at max-abs ≤ 1.9e-5.
//!
//! Usage: dsv4-greedy-verify <model-dir> <gpu_greedy.json> <out-dir>
//!   exit 0 = full agreement; exit 1 = divergence found (rows banked, analysis printed)

use memra_gguf::config::JsonObj;
use memra_gguf::dsv4_forward::{
    BlockW, Dsv4Model, HcSet, drift_coeff, expert_arm_native, hc_expand, hc_head, matmul, rmsnorm,
};
use std::io::Write;
use std::path::Path;

const MAX_SEQ: usize = 4096;

/// Lane 7 native class: GPU-vs-CPU-quantized-oracle pair coefficient (triangle through
/// the f32 ideal; both realizations carry the quantizer noise independently). Every
/// disagreement is adjudicated by band = 3·√2·C_pair·|cpu top1| — in-band near-ties are
/// legitimate realization flips of the class, out-of-band is a REAL bug. The bf16 class
/// keeps the lane-4 exit-1-on-any-divergence behavior (its instrument was raw identity;
/// divergences there were adjudicated downstream by dsv4-decode-oracle-check).
fn native_band(top1: f32) -> f64 {
    let c_pair = drift_coeff(86.0, 86.0) + drift_coeff(0.0, 86.0);
    3.0 * 2f64.sqrt() * c_pair * (top1.abs() as f64)
}

fn top_k(v: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
        .into_iter()
        .take(k)
        .map(|i| (i as u32, v[i]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dsv4-greedy-verify <model-dir> <gpu_greedy.json> <out-dir>");
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let gj_path = Path::new(&args[2]);
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");

    let gj = JsonObj::parse(&std::fs::read_to_string(gj_path).expect("read gpu json"));
    let variant_tag = gj.string("variant").expect("variant");
    let variant = memra_gguf::dsv4_forward::ActQuantVariant::from_fixture_tag(&variant_tag);
    let prompt = gj.u32_array("prompt").expect("prompt");
    let gpu_tokens = gj.u32_array("tokens_run0").expect("tokens_run0");
    let n_new = gpu_tokens.len();
    let seq: Vec<u32> = prompt.iter().chain(gpu_tokens.iter()).cloned().collect();
    let s = seq.len();
    println!(
        "dsv4-greedy-verify | model {} | variant {variant_tag} | prompt {} + gpu tokens {n_new} = {s}",
        dir.display(),
        prompt.len()
    );

    let model = Dsv4Model::open(dir);
    let d = model.cfg();
    let hc = d.hc_mult as usize;
    let hidden = model.mc.n_embd as usize;
    let n_trunk = model.mc.n_layer - model.mc.nextn_predict_layers;
    let hc_eps = d.hc_eps;
    let eps = model.mc.rms_eps;

    // ---- one teacher-forced prefill over the full sequence (lane-3 oracle program)
    let e = model.embed_rows(&seq);
    let mut h = hc_expand(&e, s, hc, hidden);
    for lid in 0..n_trunk {
        let blk = BlockW::load(&model, &format!("layers.{lid}"), lid, MAX_SEQ);
        h = blk.forward(&model, &h, s, &seq, variant, None);
        if lid % 4 == 3 || lid + 1 == n_trunk {
            println!("layer {lid} done t={:.0}s", t0.elapsed().as_secs_f64());
        }
    }

    // trunk head over ALL positions (hc_head + final norm), then batched logits for the
    // predictive positions prompt_len-1 .. s-2 (each predicts seq[p+1]).
    let set = HcSet {
        rows: model.tensor_f32("hc_head_fn").0[0],
        fn_w: model.tensor_f32("hc_head_fn").1,
        base: model.tensor_f32("hc_head_base").1,
        scale: model.tensor_f32("hc_head_scale").1,
    };
    let collapsed = hc_head(&h, s, hc, hidden, &set, eps, hc_eps);
    let final_h = rmsnorm(&collapsed, &model.tensor_f32("norm.weight").1, eps);
    let p0 = prompt.len() - 1;
    let positions: Vec<usize> = (p0..s - 1).collect();
    let mut xsel = Vec::with_capacity(positions.len() * hidden);
    for &p in &positions {
        xsel.extend_from_slice(&final_h[p * hidden..(p + 1) * hidden]);
    }
    println!(
        "head: decoding + {} position logits (t={:.0}s)",
        positions.len(),
        t0.elapsed().as_secs_f64()
    );
    let (hshape, head_w) = model.tensor_f32("head");
    let vocab = hshape[0];
    let logits = matmul(&xsel, positions.len(), hidden, &head_w, vocab);
    println!("logits done t={:.0}s", t0.elapsed().as_secs_f64());

    // ---- per-position agreement
    let native = expert_arm_native();
    if native {
        println!(
            "class: NATIVE expert arm — oracle ran the quantized expert GEMMs; disagreements adjudicated in-band vs band = 3·√2·(C_gpu + C_cpu)·|top1|"
        );
    }
    let mut first_div: Option<usize> = None;
    let mut out_of_band = 0usize;
    let mut agree = 0usize;
    let mut report = String::new();
    for (ri, &p) in positions.iter().enumerate() {
        let row = &logits[ri * vocab..(ri + 1) * vocab];
        let top = top_k(row, 5);
        let want = seq[p + 1];
        let got = top[0].0;
        if got == want {
            agree += 1;
        } else if native {
            let margin = (top[0].1 - row[want as usize]) as f64;
            let band = native_band(top[0].1);
            let in_band = margin <= band;
            if !in_band {
                out_of_band += 1;
            }
            if first_div.is_none() {
                first_div = Some(p);
            }
            println!(
                "  disagreement at position {p} (gpu step {}): cpu argmax {} vs gpu token {} | cpu margin {margin:.4} vs band {band:.4} -> {}",
                p - p0,
                got,
                want,
                if in_band {
                    "in-band"
                } else {
                    "OUT OF BAND — REAL BUG"
                }
            );
        } else if first_div.is_none() {
            first_div = Some(p);
            let gap = top[0].1 - row[want as usize];
            println!(
                "FIRST DIVERGENCE at position {p} (gpu step {}): cpu argmax {} (logit {:.4}) vs gpu token {} (cpu logit {:.4}), cpu top1-vs-gpu-token gap {:.4}",
                p - p0,
                got,
                top[0].1,
                want,
                row[want as usize],
                gap
            );
            println!("  cpu top-5 at divergence: {top:?}");
            // bank the CPU logits row
            let mut blob = Vec::with_capacity(vocab * 4);
            for v in row {
                blob.extend_from_slice(&v.to_le_bytes());
            }
            let path = out_dir.join(format!("cpu_logits_div_pos{p}.bin"));
            std::fs::write(&path, &blob).expect("write div row");
            println!("  banked {}", path.display());
        }
        report.push_str(&format!(
            "{{\"pos\": {p}, \"gpu_token\": {want}, \"cpu_argmax\": {}, \"cpu_top1_logit\": {:.6}, \"cpu_logit_of_gpu_token\": {:.6}}}{}\n",
            got,
            top[0].1,
            row[want as usize],
            if p + 2 < s { "," } else { "" }
        ));
    }

    // bank the reference (per-position record + full logits matrix)
    let mut blob = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out_dir.join("cpu_logits_all.bin"), &blob).expect("write all");
    let mut f = std::fs::File::create(out_dir.join("cpu_verify.json")).expect("json");
    write!(
        f,
        "{{\n  \"variant\": \"{variant_tag}\",\n  \"positions\": {},\n  \"agree\": {agree},\n  \"first_divergence\": {},\n  \"records\": [\n{}  ]\n}}\n",
        positions.len(),
        first_div.map(|p| p.to_string()).unwrap_or("null".into()),
        report
    )
    .expect("write json");

    println!(
        "\nCPU TEACHER-FORCING VERIFY [{variant_tag}]: {agree}/{} positions agree | first divergence: {} | elapsed {:.0}s",
        positions.len(),
        first_div
            .map(|p| format!("position {p} (gpu step {})", p - p0))
            .unwrap_or_else(|| "NONE — CPU greedy == GPU greedy for all tokens".into()),
        t0.elapsed().as_secs_f64()
    );
    if native {
        println!(
            "NATIVE-CLASS VERDICT: {} ({} disagreements, {} out-of-band)",
            if out_of_band == 0 { "PASS" } else { "FAIL" },
            positions.len() - agree,
            out_of_band
        );
        std::process::exit(if out_of_band == 0 { 0 } else { 1 });
    }
    if first_div.is_some() {
        std::process::exit(1);
    }
}
