//! Gate for the TP decode all-reduce (lane/tp-allreduce-20260906). Every rank must end holding
//! the elementwise sum, bitwise against the host sum, at the payload sizes a TP-2 decode join
//! actually moves (hidden 4096 f32 = 16 KiB) and either side of them. Runs on two devices when
//! the box has them and otherwise on the two-context same-device emulation the TP gates use, so
//! CI covers the program even on one card.
//!
//! RED ARM, and it is the reason the emulation is worth running: the first cut of this primitive
//! had the fold spin on a peer-armed flag, which is correct on two cards and deadlocks on one
//! (the spinning fold fills the SMs and the peer's push can never be scheduled). It passed at
//! 4 KiB and 64 KiB and failed every element at 256 KiB. The sweep here reaches 256 KiB for
//! exactly that reason, and `all_reduce_without_the_peer_push_diverges` proves the check is not
//! vacuous by dropping one direction.
use cudarc::driver::DevicePtr;
use memra_engine::Engine;
use memra_engine::tp_ar::{ArLink, memra_tp_ar_fold};
use std::os::raw::c_void;

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

/// Two real devices, or nothing. See the module note.
fn pair() -> Option<(Engine, Engine)> {
    let a = Engine::new(0).ok()?;
    let b = Engine::new(1).ok()?;
    memra_engine::tp::grant_peer_access(&a, &b, "tp-ar-gate").ok()?;
    memra_engine::tp::grant_peer_access(&b, &a, "tp-ar-gate").ok()?;
    Some((a, b))
}

#[test]
fn all_reduce_matches_the_host_sum_bitwise() {
    let Some((ea, eb)) = pair() else {
        eprintln!("needs two CUDA devices; skipping");
        return;
    };
    let engines = [&ea, &eb];
    for &n in &[1usize, 4, 1024, 4096, 16384, 65536] {
        let ha = vecf(n, 11 + n as u64);
        let hb = vecf(n, 77 + n as u64);
        let want: Vec<f32> = ha.iter().zip(&hb).map(|(x, y)| x + y).collect();
        let mut link = ArLink::new(&engines).unwrap();
        let mut xa = ea.htod(&ha).unwrap();
        let mut xb = eb.htod(&hb).unwrap();
        link.all_reduce(&engines, &mut [&mut xa, &mut xb], n)
            .unwrap();
        let ga = ea.dtoh_view(&xa.slice(0..n)).unwrap();
        let gb = eb.dtoh_view(&xb.slice(0..n)).unwrap();
        for i in 0..n {
            assert_eq!(ga[i].to_bits(), want[i].to_bits(), "rank 0 at n={n} i={i}");
            assert_eq!(gb[i].to_bits(), want[i].to_bits(), "rank 1 at n={n} i={i}");
        }
        // Repeating on the same link must keep working: the staging buffers are reused and the
        // events are re-recorded, so a stale-event or stale-stage bug shows up on the second call.
        link.all_reduce(&engines, &mut [&mut xa, &mut xb], n)
            .unwrap();
        // Both ranks now hold the sum, so a second reduce doubles it.
        let ga2 = ea.dtoh_view(&xa.slice(0..n)).unwrap();
        for i in 0..n {
            let twice = want[i] + want[i];
            assert_eq!(
                ga2[i].to_bits(),
                twice.to_bits(),
                "second call at n={n} i={i}"
            );
        }
    }
}

#[test]
fn all_reduce_without_the_peer_push_diverges() {
    let Some((ea, eb)) = pair() else {
        eprintln!("needs two CUDA devices; skipping");
        return;
    };
    let engines = [&ea, &eb];
    let n = 4096usize;
    let ha = vecf(n, 5);
    let hb = vecf(n, 6);
    let link = ArLink::new(&engines).unwrap();
    let xa = ea.htod(&ha).unwrap();
    // The fold alone, with nothing pushed: the staging buffer is zeros, so rank 0 keeps its own
    // partial instead of the sum. If this matched, the passing test above would prove nothing.
    let s = ea.stream();
    let dst = xa.device_ptr(&s).0 as *mut f32;
    let stage = ea.zeros(n).unwrap();
    let stage_ptr = stage.device_ptr(&s).0 as *const f32;
    let rc = unsafe { memra_tp_ar_fold(dst, stage_ptr, n as i64, s.cu_stream() as *mut c_void) };
    assert_eq!(rc, 0, "fold rc");
    let got = ea.dtoh_view(&xa.slice(0..n)).unwrap();
    let want: Vec<f32> = ha.iter().zip(&hb).map(|(x, y)| x + y).collect();
    let same = (0..n)
        .filter(|&i| got[i].to_bits() == want[i].to_bits())
        .count();
    assert!(
        same < n,
        "red arm: a fold with no peer push must not reproduce the sum"
    );
    drop(link);
}

/// Broadcast and all-gather are PURE MOVEMENT, which is the property the TP walk's byte-identity
/// rests on: the glm5 MLA layer's three hops move bytes and nothing else, so a transport swap
/// must reproduce them exactly, not approximately. Both are checked bitwise, and the all-gather
/// is checked on BOTH ranks because each rank fills its own slot locally and its peer's slot
/// over the link, and only reading both proves the two directions do not collide.
#[test]
fn broadcast_and_all_gather_move_bytes_exactly() {
    let Some((ea, eb)) = pair() else {
        eprintln!("needs two CUDA devices; skipping");
        return;
    };
    let engines = [&ea, &eb];
    for &span in &[1usize, 1024, 16384] {
        let mut link = ArLink::new(&engines).unwrap();

        // broadcast: rank 0's buffer lands on rank 1 byte for byte
        let hsrc = vecf(span, 31 + span as u64);
        let src = ea.htod(&hsrc).unwrap();
        let mut dst = eb.htod(&vec![0.0f32; span]).unwrap();
        link.broadcast(&engines, 0, &src, &mut dst, span).unwrap();
        let got = eb.dtoh_view(&dst.slice(0..span)).unwrap();
        for i in 0..span {
            assert_eq!(
                got[i].to_bits(),
                hsrc[i].to_bits(),
                "broadcast span={span} i={i}"
            );
        }

        // all-gather: rank order concatenation, identical on both ranks
        let pa = vecf(span, 41 + span as u64);
        let pb = vecf(span, 42 + span as u64);
        let da = ea.htod(&pa).unwrap();
        let db = eb.htod(&pb).unwrap();
        let mut fa = ea.htod(&vec![0.0f32; 2 * span]).unwrap();
        let mut fb = eb.htod(&vec![0.0f32; 2 * span]).unwrap();
        link.all_gather(&engines, &[&da, &db], &mut [&mut fa, &mut fb], span)
            .unwrap();
        let ga = ea.dtoh_view(&fa.slice(0..2 * span)).unwrap();
        let gb = eb.dtoh_view(&fb.slice(0..2 * span)).unwrap();
        for i in 0..span {
            assert_eq!(
                ga[i].to_bits(),
                pa[i].to_bits(),
                "gather rank0 slot0 span={span} i={i}"
            );
            assert_eq!(
                ga[span + i].to_bits(),
                pb[i].to_bits(),
                "gather rank0 slot1 span={span} i={i}"
            );
            assert_eq!(
                gb[i].to_bits(),
                pa[i].to_bits(),
                "gather rank1 slot0 span={span} i={i}"
            );
            assert_eq!(
                gb[span + i].to_bits(),
                pb[i].to_bits(),
                "gather rank1 slot1 span={span} i={i}"
            );
        }
        ea.stream().synchronize().unwrap();
        eb.stream().synchronize().unwrap();
    }
}

/// The one-shot arm, which is the shape that matters: one kernel per rank, no CUDA events. Same
/// bitwise bar as the pipeline arm, plus the barrier's refusal word, plus repeat calls (the
/// alternating start/end counters are exactly what a second call exercises, and getting them wrong
/// shows up as a hang or a stale flag rather than a wrong number).
#[test]
fn one_shot_all_reduce_matches_the_host_sum_bitwise() {
    let Some((ea, eb)) = pair() else {
        eprintln!("needs two CUDA devices; skipping");
        return;
    };
    let engines = [&ea, &eb];
    for &n in &[1usize, 1024, 4096, 16384, 65536] {
        let ha = vecf(n, 11 + n as u64);
        let hb = vecf(n, 77 + n as u64);
        let want: Vec<f32> = ha.iter().zip(&hb).map(|(x, y)| x + y).collect();
        let mut link = ArLink::new(&engines).unwrap();
        let mut xa = ea.htod(&ha).unwrap();
        let mut xb = eb.htod(&hb).unwrap();
        for round in 0..3 {
            link.all_reduce_1stage(&engines, &mut [&mut xa, &mut xb], n)
                .unwrap();
            // The barrier words FIRST: an expired barrier returns with `x` untouched, and read
            // as numbers that is "rank 0 got its own operand back", not "the barrier expired".
            assert_eq!(
                link.barrier_errors(&engines).unwrap(),
                [0, 0],
                "one-shot barrier expired at n={n} round={round} (40043 entry, 40044 exit)"
            );
            let ga = ea.dtoh_view(&xa.slice(0..n)).unwrap();
            let gb = eb.dtoh_view(&xb.slice(0..n)).unwrap();
            // Round r doubles the previous sum, so the expected value is want * 2^r.
            let scale = (1u32 << round) as f32;
            for i in 0..n {
                let w = want[i] * scale;
                assert_eq!(
                    ga[i].to_bits(),
                    w.to_bits(),
                    "rank 0 n={n} round={round} i={i}"
                );
                assert_eq!(
                    gb[i].to_bits(),
                    w.to_bits(),
                    "rank 1 n={n} round={round} i={i}"
                );
            }
        }
        assert_eq!(
            link.barrier_errors(&engines).unwrap(),
            vec![0, 0],
            "a barrier refused at n={n}"
        );
        ea.stream().synchronize().unwrap();
        eb.stream().synchronize().unwrap();
    }
}
