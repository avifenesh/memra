//! dsv4-mtp-probe: pin the MTP (NextN) token-shift convention empirically.
//!
//! Lane-2 flagged this open: the reference repo never drives mtp (generate.py has no
//! drafter; model.py's own smoke test feeds the SAME ids to trunk and MTP, model.py:826),
//! so the spec-decode alignment is not pinned by any code in the artifact. The V3-paper
//! convention says MTP position i fuses trunk state h_i with Emb(t_{i+1}) — the
//! embedding stream leads the trunk by one — and predicts t_{i+2}.
//!
//! Decisive probe (can't-hallucinate shape): a counting prompt whose next TWO tokens
//! are near-deterministic and DISTINCT (`… 11 12` → next = the space token, two-ahead =
//! the "13" token, ids differ). Run the trunk once, then the MTP block two ways over
//! the same trunk hc-state:
//!   A. UNSHIFTED (model.py:826 call shape): input_ids = prompt ids.
//!   B. SHIFTED (V3 convention): input_ids = prompt ids shifted left by one, with the
//!      trunk's greedy next token appended at the last position.
//! If A's top-1 is the trunk's next token while B's top-1 is the token AFTER it, the
//! V3 shift is confirmed: the drafter must be fed the one-ahead embedding stream.
//!
//! Usage:
//!   dsv4-mtp-probe <model-dir> <ref|clamp-only> <comma-separated token ids>
//!
//! The variant only selects the QAT act-quant fork; the alignment conclusion must not
//! depend on it (run under the engine's contract variant).

use memra_gguf::dsv4_forward::{
    ActQuantVariant, BlockW, Dsv4Model, hc_expand, mtp_logits_last, trunk_logits_last,
};
use std::path::Path;

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.into_iter().take(5).map(|i| (i, v[i])).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dsv4-mtp-probe <model-dir> <ref|clamp-only> <ids,comma,separated>");
        std::process::exit(2);
    }
    let variant = ActQuantVariant::from_fixture_tag(&args[2]);
    let ids: Vec<u32> = args[3]
        .split(',')
        .map(|x| x.trim().parse::<u32>().expect("token id"))
        .collect();
    let s = ids.len();
    let t0 = std::time::Instant::now();
    let model = Dsv4Model::open(Path::new(&args[1])).unwrap_or_else(|error| {
        eprintln!("dsv4 model load failed: {error}");
        std::process::exit(1);
    });
    let hc = model.cfg().hc_mult as usize;
    let hidden = model.mc.n_embd as usize;
    let n_trunk = model.mc.n_layer - model.mc.nextn_predict_layers;
    println!(
        "dsv4-mtp-probe | model {} | variant {:?} | {} tokens",
        args[1], variant, s
    );

    let e = model.embed_rows(&ids);
    let mut h = hc_expand(&e, s, hc, hidden);
    for lid in 0..n_trunk {
        let blk = BlockW::load(&model, &format!("layers.{lid}"), lid, 4096);
        h = blk.forward(&model, &h, s, &ids, variant, None);
        if lid % 8 == 7 {
            println!(
                "  trunk layer {lid} done t={:.1}s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let logits = trunk_logits_last(&model, &h, s);
    let trunk_top = top5(&logits);
    println!("trunk next-token top5: {trunk_top:?}");
    let t1 = trunk_top[0].0 as u32;

    // A. unshifted (model.py:826 call shape)
    let la = mtp_logits_last(&model, &h, s, &ids, variant, 4096);
    let a_top = top5(&la);
    println!("MTP UNSHIFTED (same ids)   top5: {a_top:?}");

    // B. shifted (V3 convention): embedding stream leads by one, trunk greedy appended
    let mut shifted = ids[1..].to_vec();
    shifted.push(t1);
    let lb = mtp_logits_last(&model, &h, s, &shifted, variant, 4096);
    let b_top = top5(&lb);
    println!("MTP SHIFTED (ids[1..]+greedy) top5: {b_top:?}");

    println!(
        "\nsummary: trunk t+1 = {t1}; unshifted mtp top1 = {}; shifted mtp top1 = {}",
        a_top[0].0, b_top[0].0
    );
    println!("elapsed: {:.1}s", t0.elapsed().as_secs_f64());
}
