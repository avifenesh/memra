//! tp-ar-bench: what a TP-2 join should cost, against what it costs today.
//!
//! The served pair's TP-2 arm measured 3.1x SLOWER than the pipeline split it would replace,
//! and the whole gap is the join: `tp_transport`'s default is `host-canonical`, every hop is
//! `dtoh` -> host -> `htod`, and `Engine::dtoh` ends in `stream().synchronize()`. A drain costs
//! whatever the stream has pending, not bytes, so each of the ~90 joins per token pays that
//! layer's compute twice over. The link is NV18, eighteen NVLink links at 53.125 GB/s.
//!
//! This times the `tp_ar` one-shot all-reduce at decode payload sizes against the host-bounce
//! shape it replaces, on the same two engines in the same process, interleaved so a clock drift
//! cannot favour either. Correctness first: every size is checked against the host sum before it
//! is timed.
//!
//! Usage: `tp-ar-bench [dev_a] [dev_b] [reps]`. Two DEVICES are required: this primitive stores
//! into the peer's buffer from inside a kernel, and two contexts on one card share no address
//! space, so `ArLink::new` refuses that pairing (its constructor note has the detail).
use memra_engine::Engine;
use memra_engine::tp_ar::ArLink;
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn main() -> Res<()> {
    let arg = |i: usize, d: usize| -> usize {
        std::env::args()
            .nth(i)
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (da, db, reps) = (arg(1, 0), arg(2, 1), arg(3, 200));
    let ea = Engine::new(da)?;
    let eb = Engine::new(db)?;
    if da != db {
        memra_engine::tp::grant_peer_access(&ea, &eb, "tp-ar-bench")?;
        memra_engine::tp::grant_peer_access(&eb, &ea, "tp-ar-bench")?;
    } else {
        println!(
            "[tp-ar] SAME-DEVICE emulation on ordinal {da}: correctness only, timing is not a receipt"
        );
    }
    let engines = [&ea, &eb];

    for &n in &[1024usize, 4096, 16384, 65536] {
        let ha = vecf(n, 11 + n as u64);
        let hb = vecf(n, 77 + n as u64);
        let want: Vec<f32> = ha.iter().zip(&hb).map(|(x, y)| x + y).collect();

        // correctness, once per size, before anything is timed
        let mut link = ArLink::new(&engines)?;
        let mut xa = ea.htod(&ha)?;
        let mut xb = eb.htod(&hb)?;
        link.all_reduce(&engines, &mut [&mut xa, &mut xb], n)?;
        let ga = ea.dtoh_view(&xa.slice(0..n))?;
        let gb = eb.dtoh_view(&xb.slice(0..n))?;
        let bad = (0..n)
            .filter(|&i| {
                ga[i].to_bits() != want[i].to_bits() || gb[i].to_bits() != want[i].to_bits()
            })
            .count();
        assert_eq!(bad, 0, "all-reduce differs from the host sum at n={n}");

        // correctness of the one-shot arm at this size, before it is timed
        let mut ya = ea.htod(&ha)?;
        let mut yb = eb.htod(&hb)?;
        link.all_reduce_1stage(&engines, &mut [&mut ya, &mut yb], n)?;
        let oa = ea.dtoh_view(&ya.slice(0..n))?;
        let ob = eb.dtoh_view(&yb.slice(0..n))?;
        let bad1 = (0..n)
            .filter(|&i| {
                oa[i].to_bits() != want[i].to_bits() || ob[i].to_bits() != want[i].to_bits()
            })
            .count();
        assert_eq!(
            bad1, 0,
            "one-shot all-reduce differs from the host sum at n={n}"
        );
        assert_eq!(
            link.barrier_errors(&engines)?,
            vec![0, 0],
            "barrier refused at n={n}"
        );

        // timed: the pipeline, the one-shot, and the host-bounce shape they replace, interleaved
        let mut t_ar = f64::MAX;
        let mut t_one = f64::MAX;
        let mut t_host = f64::MAX;
        for _ in 0..5 {
            ea.stream().synchronize()?;
            eb.stream().synchronize()?;
            let t0 = Instant::now();
            for _ in 0..reps {
                link.all_reduce(&engines, &mut [&mut xa, &mut xb], n)?;
            }
            ea.stream().synchronize()?;
            eb.stream().synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if us < t_ar {
                t_ar = us;
            }

            ea.stream().synchronize()?;
            eb.stream().synchronize()?;
            let t2 = Instant::now();
            for _ in 0..reps {
                link.all_reduce_1stage(&engines, &mut [&mut ya, &mut yb], n)?;
            }
            ea.stream().synchronize()?;
            eb.stream().synchronize()?;
            let us = t2.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if us < t_one {
                t_one = us;
            }

            let t1 = Instant::now();
            for _ in 0..reps {
                // The shape `host-canonical` runs: a draining dtoh each way plus the htod back.
                let sa = ea.dtoh_view(&xa.slice(0..n))?;
                let sb = eb.dtoh_view(&xb.slice(0..n))?;
                let _ba = eb.htod(&sa)?;
                let _bb = ea.htod(&sb)?;
            }
            ea.stream().synchronize()?;
            eb.stream().synchronize()?;
            let us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
            if us < t_host {
                t_host = us;
            }
        }
        println!(
            "[tp-ar] n={n:6} ({:5.1} KiB)  1stage {t_one:7.2} us   push-fold {t_ar:7.2} us   \
             host-bounce {t_host:8.2} us   1stage is {:.1}x the pipeline, {:.1}x the bounce",
            (n * 4) as f64 / 1024.0,
            t_ar / t_one.max(1e-9),
            t_host / t_one.max(1e-9)
        );
    }
    Ok(())
}
