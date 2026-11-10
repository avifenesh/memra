//! memra#1146 scratch probe (not for landing): which step moves turn 2's second token?
//! Runs turn 1 as the session gate does, then turn 2's full history through several routes and
//! prints each route's first tokens against plain greedy `generate()`.
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::hybrid_forward::PRIME_MIN_T;
use memra_engine::memra_gguf::GgufFile;
use memra_engine::spec::SpecSampling;

fn stable_boundary(start: usize, len: usize, grid: usize) -> Option<usize> {
    let mut b = len / grid * grid;
    while b >= grid && len - b < PRIME_MIN_T {
        b -= grid;
    }
    (b > start).then_some(b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: probe <model.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let k: usize = std::env::var("MEMRA_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let n_new: usize = std::env::var("MEMRA_NGEN").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let grid = Engine::gdn_chunk_size();
    let turn1: Vec<u32> = (101..229).collect();
    let turn2: Vec<u32> = (350..415).collect();
    let show = |name: &str, out: &[u32], reference: &[u32]| {
        let n = out.len().min(reference.len());
        let first = (0..n).find(|&i| out[i] != reference[i]);
        println!(
            "{name:<34} {:?} first_diff={first:?}",
            &out[..out.len().min(10)]
        );
    };

    // turn 1 exactly as the gate: cold session, prime with a grid stop.
    let mut sess = model.new_session(&e, 4096)?;
    let b1 = stable_boundary(0, turn1.len(), grid);
    sess.ckpt_at = b1;
    let (out1, _, _) = model.generate_spec_session_sampled_prime_split(
        &e, &mut sess, &turn1, n_new, k, Some(SpecSampling::from_cli_env()), b1, None,
    )?;
    let ref1 = model.generate(&e, &turn1, out1.len())?;
    show("turn1 session (split)", &out1, &ref1);
    let mut hist = turn1.clone();
    hist.extend_from_slice(&out1);
    hist.extend_from_slice(&turn2);
    println!("turn 2 history: {} tokens (turn 1 generated {})", hist.len(), out1.len());
    let n2 = 16usize;
    let reference = model.generate(&e, &hist, n2)?;
    show("R  generate (plain greedy)", &reference, &reference);
    let again = model.generate(&e, &hist, n2)?;
    show("R' generate again", &again, &reference);

    for kk in [1usize, 2, 3] {
        let (a, _, _) = model.generate_spec(&e, &hist, n2, kk)?;
        show(&format!("A{kk} generate_spec cold K={kk}"), &a, &reference);
    }
    for kk in [1usize, 2, 3] {
        let mut s = model.new_session(&e, 4096)?;
        let (b, _, _) = model.generate_spec_session(&e, &mut s, &hist, n2, kk)?;
        show(&format!("B{kk} fresh session, no split K={kk}"), &b, &reference);
    }
    {
        let mut s = model.new_session(&e, 4096)?;
        let bd = stable_boundary(0, hist.len(), grid);
        s.ckpt_at = bd;
        let (c, _, _) = model.generate_spec_session_sampled_prime_split(
            &e, &mut s, &hist, n2, k, Some(SpecSampling::from_cli_env()), bd, None,
        )?;
        show(&format!("C  fresh session, split {bd:?} K={k}"), &c, &reference);
    }
    {
        // D: the turn-1 session rewound to its on-grid checkpoint, the rest primed with a stop.
        let pos = model.spec_rewind_to_checkpoint(&e, &mut sess)?.expect("checkpoint");
        let prime = hist[pos..].to_vec();
        let bd = stable_boundary(pos, hist.len(), grid);
        sess.ckpt_at = bd.or(sess.ckpt_at);
        let (d, _, _) = model.generate_spec_session_sampled_prime_split(
            &e, &mut sess, &prime, n2, k, Some(SpecSampling::from_cli_env()),
            bd.map(|b| b - pos), None,
        )?;
        show(&format!("D  rewound to {pos}, split {bd:?} K={k}"), &d, &reference);
    }
    {
        // E: a second turn-1 session resumed at its parked end (the old gate).
        let mut s = model.new_session(&e, 4096)?;
        let (o1, _, _) = model.generate_spec_session(&e, &mut s, &turn1, n_new, k)?;
        println!("E  turn 1 again equal to first run: {}", o1 == out1);
        let (ee, _, _) = model.generate_spec_session(&e, &mut s, &turn2, n2, k)?;
        show(&format!("E  parked end + suffix K={k}"), &ee, &reference);
    }
    // plain greedy through the reference twice more, after the sessions ran
    let last = model.generate(&e, &hist, n2)?;
    show("R\" generate after the sessions", &last, &reference);
    Ok(())
}
