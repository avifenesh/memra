//! prime-fold-gate: the boundary tail fold against the prime split it replaces
//! (research/gb10-prefill-20261106/TAIL-FOLD-DESIGN.md).
//!
//! Per case (prompt rows `n`, boundary `b` on the GDN chunk grid) on deterministic token ids:
//!   split: `prime_cache` of rows [0, b), `Cache::snapshot()` there, then rows [b, n);
//!   fold:  one `prime_cache` of rows [0, n) with a GDN tap at `b`, `Cache::snapshot_from_tap`,
//!          and `prime_row_logits` at row b - 1.
//! Bit-compares: every hidden row, the final logits, the logits at b - 1 (the split's first
//! call's), the boundary snapshot (conv rings, SSM states, KV lengths, pos), the final
//! recurrent state of every GDN layer and the K/V bytes of rows [0, n) of every
//! full-attention layer. Also checks that `prime_tail_fold_ok` accepts each case, so the gate
//! covers what the serving path folds, and refuses a stop on a call boundary. Exit 1 on any
//! difference.
//!
//! usage: prime-fold-gate <model.gguf> [--cases n:b,n:b,...]

use memra_engine::Engine;
use memra_engine::cache::{Cache, CacheSnapshot, GdnTap};
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn bits_eq(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + a.len().abs_diff(b.len())
}

fn snap_diff(
    e: &Engine,
    a: &CacheSnapshot,
    b: &CacheSnapshot,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut bad = usize::from(a.pos != b.pos) + usize::from(a.kv_len != b.kv_len);
    for (x, y) in a.conv.iter().zip(&b.conv).chain(a.ssm.iter().zip(&b.ssm)) {
        match (x, y) {
            (Some(x), Some(y)) => bad += bits_eq(&e.dtoh(x)?, &e.dtoh(y)?),
            (None, None) => {}
            _ => bad += 1,
        }
    }
    Ok(bad)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: prime-fold-gate <model.gguf> [--cases n:b,...]");
    let rest: Vec<String> = args.collect();
    let cases: Vec<(usize, usize)> = rest
        .iter()
        .position(|a| a == "--cases")
        .and_then(|i| rest.get(i + 1))
        .map(|s| {
            s.split(',')
                .filter_map(|c| {
                    let (n, b) = c.split_once(':')?;
                    Some((n.parse().ok()?, b.parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_else(|| {
            vec![
                (8171, 8128), // the served 131k shape: 4,032 rows into a 4,075-row last call, 43 after
                (8139, 8096), // the same with the stop between two 64-row tiles (r = 4,000)
                (4171, 4128), // 32 rows into the last call, 43 after
                (4396, 4352), // 256 rows into the last call, 44 after
                (4176, 4160), // the shortest tail, 16 rows
                (1000, 960),  // one call
                (8232, 8160), // a middle call (32 rows left in it), the tail crosses into the last call
                (8292, 8192), // the stop on a call boundary: refused, the split already ends a call there
            ]
        });

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let vocab = model.cfg.n_vocab as u64;
    let n_embd = model.cfg.n_embd as usize;
    println!(
        "prime-fold-gate: {} ({} layers), gdn chunk {}",
        g.arch().unwrap_or("?"),
        model.layers.len(),
        Engine::gdn_chunk_size()
    );
    let mut fails = 0usize;
    for (ci, &(n, b)) in cases.iter().enumerate() {
        let toks: Vec<u32> = (0..n as u64)
            .map(|i| ((i * 2_654_435_761 + 97 * ci as u64 + 11) % (vocab - 1000) + 500) as u32)
            .collect();
        let accepted = model.prime_tail_fold_ok(&e, n, b);
        let ranges = memra_engine::hybrid_forward::prime_chunk_ranges(
            n,
            model.layers.len(),
            model.gdn_prime_grid_on(),
            model.swa_ring_on(),
        );
        if ranges.iter().any(|&(s, _)| s == b) {
            let ok = !accepted;
            fails += usize::from(!ok);
            println!(
                "PRIME-FOLD n={n} b={b}: the stop is a call boundary, fold_ok={accepted} {}",
                if ok {
                    "OK (refused)"
                } else {
                    "FAIL (accepted)"
                }
            );
            continue;
        }
        let ctx = n + 64;

        let mut cs = Cache::new(&e, &model.cfg, ctx)?;
        let (l1, _, h1) = model.prime_cache(&e, &toks[..b], &mut cs, n - b)?;
        let snap_split = cs.snapshot(&e)?;
        let (l2, _, h2) = model.prime_cache(&e, &toks[b..], &mut cs, 0)?;
        let mut h_split = e.dtoh(&h1)?;
        h_split.extend(e.dtoh(&h2)?);

        let mut cf = Cache::new(&e, &model.cfg, ctx)?;
        cf.gdn_tap = Some(GdnTap::new(b, cf.recur.len()));
        let (lf, _, hf) = model.prime_cache(&e, &toks, &mut cf, 0)?;
        let tap = cf.gdn_tap.take().ok_or("the tap vanished")?;
        let recurrent = cf.recur.iter().flatten().count();
        let taken = tap.taken;
        let snap_fold = cf.snapshot_from_tap(tap);
        let lb = model.prime_row_logits(&e, &hf, b - 1)?;

        let d_h = bits_eq(&h_split, &e.dtoh(&hf)?);
        let d_l = bits_eq(&l2, &lf);
        let d_lb = bits_eq(&l1, &lb);
        let d_snap = match &snap_fold {
            Ok(s) => snap_diff(&e, &snap_split, s)?,
            Err(_) => usize::MAX,
        };
        let mut d_state = 0usize;
        for (x, y) in cs.recur.iter().zip(&cf.recur) {
            if let (Some(x), Some(y)) = (x, y) {
                d_state += bits_eq(&e.dtoh(&x.conv_state)?, &e.dtoh(&y.conv_state)?);
                d_state += bits_eq(&e.dtoh(&x.ssm_state)?, &e.dtoh(&y.ssm_state)?);
            }
        }
        let mut d_kv = 0usize;
        for (x, y) in cs.kv.iter().zip(&cf.kv) {
            if let (Some(x), Some(y)) = (x, y) {
                d_kv += usize::from(x.len != y.len);
                let rows = x.len.min(y.len);
                for (px, py, tb) in [(&x.k, &y.k, x.k_tok_bytes), (&x.v, &y.v, x.v_tok_bytes)] {
                    let (bx, by) = (e.dtoh_u8(px)?, e.dtoh_u8(py)?);
                    d_kv += bx[..rows * tb]
                        .iter()
                        .zip(&by[..rows * tb])
                        .filter(|(p, q)| p != q)
                        .count();
                }
            }
        }
        let ok = accepted
            && taken == recurrent
            && d_h == 0
            && d_l == 0
            && d_lb == 0
            && d_snap == 0
            && d_state == 0
            && d_kv == 0;
        fails += usize::from(!ok);
        println!(
            "PRIME-FOLD n={n} b={b} tail={}: fold_ok={accepted} tapped {taken}/{recurrent} | \
             hidden {d_h}/{} final-logits {d_l} boundary-logits {d_lb} snapshot {} state {d_state} \
             kv {d_kv} {}",
            n - b,
            n * n_embd,
            if d_snap == usize::MAX {
                "REFUSED".to_string()
            } else {
                d_snap.to_string()
            },
            if ok { "OK" } else { "FAIL" }
        );
    }
    println!(
        "prime-fold-gate: {} cases, {fails} failed {}",
        cases.len(),
        if fails == 0 { "ALL GREEN" } else { "FAIL" }
    );
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}
