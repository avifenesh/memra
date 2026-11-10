//! memra#1146 scratch probe 2 (not for landing): the first differing value.
//! Prime turn 2's history with `prime_cache` (what `generate` runs), then take the next step
//! three ways from that same primed state and compare the logits: plain `decode_step`, the
//! verify forward over one row, and the verify forward over four rows (row 0).
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::memra_gguf::GgufFile;

fn top(l: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = l.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(n);
    v
}

fn cmp(name: &str, a: &[f32], b: &[f32]) {
    let n = a.len().min(b.len());
    let mut worst = (0usize, 0f32);
    let mut differing = 0usize;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if a[i].to_bits() != b[i].to_bits() {
            differing += 1;
        }
        if d > worst.1 {
            worst = (i, d);
        }
    }
    println!(
        "{name}: {differing} of {n} logits differ in bits; max |diff| {:.6} at id {} ({} vs {})",
        worst.1, worst.0, a[worst.0], b[worst.0]
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: probe <model.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let k: usize = std::env::var("MEMRA_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let n_new: usize = std::env::var("MEMRA_NGEN").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let n_vocab = model.cfg.n_vocab as usize;
    let turn1: Vec<u32> = (101..229).collect();
    let turn2: Vec<u32> = (350..415).collect();
    let mut sess = model.new_session(&e, 4096)?;
    let (out1, _, _) = model.generate_spec_session(&e, &mut sess, &turn1, n_new, k)?;
    drop(sess);
    let mut hist = turn1.clone();
    hist.extend_from_slice(&out1);
    hist.extend_from_slice(&turn2);
    println!("history: {} tokens", hist.len());

    let prime = |c: &mut memra_engine::cache::Cache| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (l, _h, _hid) = model.prime_cache(&e, &hist, c, 0)?;
        Ok(l)
    };
    let mut ca = memra_engine::pp::new_cache(&e, &model.cfg, hist.len() + 64)?;
    let la = prime(&mut ca)?;
    let t1 = top(&la[..n_vocab], 1)[0].0 as u32;
    println!("prime: next token {t1}, top5 {:?}", top(&la[..n_vocab], 5));
    let mut cb = memra_engine::pp::new_cache(&e, &model.cfg, hist.len() + 64)?;
    let lb = prime(&mut cb)?;
    cmp("prime twice", &la[..n_vocab], &lb[..n_vocab]);
    let mut cc = memra_engine::pp::new_cache(&e, &model.cfg, hist.len() + 64)?;
    let _ = prime(&mut cc)?;
    let mut cd = memra_engine::pp::new_cache(&e, &model.cfg, hist.len() + 64)?;
    let _ = prime(&mut cd)?;

    // (i) plain decode step
    let pos = ca.pos;
    let plain = model.decode_step(&e, t1, &mut ca)?;
    println!("plain decode_step at pos {pos}: top5 {:?}", top(&plain[..n_vocab], 5));
    // (ii) verify forward, one row, in the exact scope spec runs it in
    let v1 = {
        let _exact = e.exact_scope(true);
        model.decode_step_t_h(&e, &[t1], pos, &mut cb)?.0
    };
    println!("verify 1 row (exact scope): top5 {:?}", top(&v1[..n_vocab], 5));
    // (iii) verify forward, four rows; row 0 is the same token at the same position
    let v4 = {
        let _exact = e.exact_scope(true);
        model.decode_step_t_h(&e, &[t1, 15, 220, 15], pos, &mut cc)?.0
    };
    println!("verify 4 rows, row 0: top5 {:?}", top(&v4[..n_vocab], 5));
    // (iv) verify forward, one row, outside the exact scope
    let v1n = model.decode_step_t_h(&e, &[t1], pos, &mut cd)?.0;
    println!("verify 1 row (no exact scope): top5 {:?}", top(&v1n[..n_vocab], 5));
    cmp("plain vs verify-1 (exact)", &plain[..n_vocab], &v1[..n_vocab]);
    cmp("plain vs verify-4 row 0 (exact)", &plain[..n_vocab], &v4[..n_vocab]);
    cmp("verify-1 vs verify-4 row 0", &v1[..n_vocab], &v4[..n_vocab]);
    cmp("plain vs verify-1 (no exact)", &plain[..n_vocab], &v1n[..n_vocab]);
    for id in [15usize, 201, 220] {
        println!(
            "id {id}: plain {} verify1 {} verify4 {} verify1-noexact {}",
            plain[id], v1[id], v4[id], v1n[id]
        );
    }
    // a second plain step from each, to see whether the states stay together
    let p2 = model.decode_step(&e, 201, &mut ca)?;
    let q2 = model.decode_step(&e, 201, &mut cb)?;
    cmp("second step (201) after plain vs after verify-1", &p2[..n_vocab], &q2[..n_vocab]);
    Ok(())
}
