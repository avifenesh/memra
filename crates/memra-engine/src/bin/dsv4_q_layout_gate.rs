//! Byte-identity gate for the DSV4 q layout change (lane/dsv4-qlayout-20260911,
//! darklanes `research/dsv4f-attn-floor-20260911/LANE.md`).
//!
//! `dsv4_sink_scores_mq_f32acc_kernel` and `dsv4_indexer_score_f32acc_pos_m_kernel` now read q
//! as `[nq][hd][heads]`, staged by `dsv4_q_transpose_m_kernel`, instead of `[nq][heads][hd]`.
//! Every product, its order, and the single f32 accumulator are unchanged, so the change owes
//! BIT-EQUALITY and not drift rows. This gate is where that is asserted rather than assumed.
//!
//! Three bars, and the second is what makes the first mean anything:
//!
//! 1. **IDENTITY.** For each shape, the shipped kernel fed the transposed q must produce a score
//!    plane BYTE-IDENTICAL to the reference kernel (`*_ref`, the `[heads][hd]` form the change
//!    replaced) fed the original q. Compared as raw bits, so a `-INFINITY` hole compares equal
//!    only to a `-INFINITY` hole and a `-0.0` never passes for a `+0.0`.
//! 2. **RED ARM.** Before any identity result counts, the same comparison is run once with a
//!    single element of the transposed q perturbed by one ULP, and it MUST differ. A comparator
//!    that cannot see a one-ULP difference would report identity for two kernels that disagree
//!    everywhere, so if the red arm matches the gate exits 1 no matter how green the rest is.
//! 3. **TRANSPOSE FIDELITY.** The device transpose is checked against a host transpose of the
//!    same buffer, elementwise and by bits, so a broken stager cannot hide inside a comparison
//!    that only ever reads what the stager produced.
//!
//! Shapes cover what the served program actually launches, not just the convenient one: the
//! saturated prefill chunk (`nq = 512`, `slots = 640`), a narrow slot list (`slots = 128`, the
//! SWA window alone), the `t == 1` decode row, and a plane with `-1` index holes so the
//! `-INFINITY` early-out branch is inside the comparison. The indexer is swept at its own
//! `hd = 128` with a causal `pos0` that leaves some rows short.
//!
//! usage: dsv4-q-layout-gate [device_ordinal]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi::{
    memra_dsv4_indexer_score_f32acc_pos_m, memra_dsv4_indexer_score_f32acc_pos_m_ref,
    memra_dsv4_q_transpose_m, memra_dsv4_sink_attn_dec_mq_f32acc,
    memra_dsv4_sink_scores_mq_f32acc_ref,
};
use std::os::raw::c_void;
use std::sync::Arc;

const HEADS: usize = 64;
const HD: usize = 512;
const IHD: usize = 128;
const NCAND: usize = 4096;

type Stream = Arc<cudarc::driver::CudaStream>;

fn dp(s: &CudaSlice<f32>, stream: &Stream) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, stream: &Stream) -> *mut f32 {
    s.device_ptr_mut(stream).0 as *mut f32
}
fn dpi(s: &CudaSlice<i32>, stream: &Stream) -> *const i32 {
    s.device_ptr(stream).0 as *const i32
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 1000.0 - 1.0
    }
}

/// `qt[p][x][h] = q[p][h][x]`, on the host, so the device stager is checked against something
/// that is not itself.
fn host_transpose(q: &[f32], nq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    for p in 0..nq {
        for h in 0..heads {
            for x in 0..hd {
                out[p * heads * hd + x * heads + h] = q[p * heads * hd + h * hd + x];
            }
        }
    }
    out
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn first_difference(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let e = Engine::new(dev)?;
    let stream = e.stream();
    let cu = stream.cu_stream() as *mut c_void;
    let scale = (HD as f32).powf(-0.5);

    println!(
        "dsv4-q-layout-gate: device {dev}, heads={HEADS} hd={HD} ihd={IHD}; \
         shipped scorers read q as [nq][hd][heads], reference arms read [nq][heads][hd]"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut red_arm_proved = false;
    let mut cells = 0usize;

    // slot lists: (nq, slots, holes). holes puts -1 entries in the list so the -INFINITY
    // early-out branch is inside the byte comparison rather than beside it.
    let shapes: [(usize, usize, bool); 4] = [
        (512, 640, true),
        (512, 128, false),
        (64, 640, true),
        (1, 640, true),
    ];

    for (nq, slots, holes) in shapes {
        let mut rng = Rng(0x5111_0911 ^ (nq as u64) << 20 ^ slots as u64);
        let nq_q = nq * HEADS * HD;
        let q: Vec<f32> = (0..nq_q).map(|_| rng.unit()).collect();
        let kv: Vec<f32> = (0..NCAND * HD).map(|_| rng.unit()).collect();
        let idx: Vec<i32> = (0..nq * slots)
            .map(|i| {
                if holes && i % 97 == 3 {
                    -1
                } else {
                    (rng.next() as usize % NCAND) as i32
                }
            })
            .collect();

        let dq = e.htod(&q)?;
        let dkv = e.htod(&kv)?;
        let didx = e.htod_i32(&idx)?;
        let mut dqt = e.uninit(nq_q)?;
        let mut sc_ship = e.uninit(nq * HEADS * slots)?;
        let mut sc_ref = e.uninit(nq * HEADS * slots)?;
        // the rest of the fused launcher's outputs; only the score plane is compared, but the
        // shipped path is exercised through the launcher the server actually calls.
        let sink: Vec<f32> = (0..HEADS).map(|_| rng.unit()).collect();
        let dsink = e.htod(&sink)?;
        let mut evals = e.uninit(nq * HEADS * slots)?;
        let mut den = e.uninit(nq * HEADS)?;
        let mut o = e.uninit(nq * HEADS * HD)?;

        let rc = unsafe {
            memra_dsv4_q_transpose_m(
                dp(&dq, &stream),
                dpm(&mut dqt, &stream),
                nq as i32,
                HEADS as i32,
                HD as i32,
                cu,
            )
        };
        assert_eq!(rc, 0, "q transpose rc={rc}");
        stream.synchronize()?;

        // BAR 3: the stager itself, against a host transpose.
        let got = e.dtoh(&dqt)?;
        let want = host_transpose(&q, nq, HEADS, HD);
        if !bits_equal(&got, &want) {
            let at = first_difference(&got, &want).unwrap_or(0);
            failures.push(format!(
                "transpose nq={nq}: device stager differs from host transpose at element {at} \
                 ({} vs {})",
                got[at], want[at]
            ));
        }

        let rc = unsafe {
            memra_dsv4_sink_attn_dec_mq_f32acc(
                dp(&dqt, &stream),
                dp(&dkv, &stream),
                dpi(&didx, &stream),
                dp(&dsink, &stream),
                dpm(&mut sc_ship, &stream),
                dpm(&mut evals, &stream),
                dpm(&mut den, &stream),
                dpm(&mut o, &stream),
                nq as i32,
                HEADS as i32,
                HD as i32,
                slots as i32,
                slots as i32,
                scale,
                cu,
            )
        };
        assert_eq!(rc, 0, "sink_attn_dec_mq_f32acc rc={rc}");
        let rc = unsafe {
            memra_dsv4_sink_scores_mq_f32acc_ref(
                dp(&dq, &stream),
                dp(&dkv, &stream),
                dpi(&didx, &stream),
                dpm(&mut sc_ref, &stream),
                nq as i32,
                HEADS as i32,
                HD as i32,
                slots as i32,
                slots as i32,
                scale,
                cu,
            )
        };
        assert_eq!(rc, 0, "sink_scores ref rc={rc}");
        stream.synchronize()?;

        let a = e.dtoh(&sc_ship)?;
        let b = e.dtoh(&sc_ref)?;
        cells += 1;
        if bits_equal(&a, &b) {
            println!(
                "  IDENTITY sink nq={nq} slots={slots} holes={holes}: {} values bit-equal",
                a.len()
            );
        } else {
            let at = first_difference(&a, &b).unwrap_or(0);
            failures.push(format!(
                "sink nq={nq} slots={slots}: first difference at {at}, shipped {:e} vs ref {:e}",
                a[at], b[at]
            ));
        }

        // BAR 2: the red arm, once, on the widest shape. One ULP on one element of the staged
        // q, and the comparison above MUST see it.
        if !red_arm_proved && nq == 512 && slots == 640 {
            let mut poisoned = got.clone();
            let target = poisoned.len() / 3;
            poisoned[target] = f32::from_bits(poisoned[target].to_bits() ^ 1);
            let dbad = e.htod(&poisoned)?;
            let mut sc_bad = e.uninit(nq * HEADS * slots)?;
            let rc = unsafe {
                memra_dsv4_sink_attn_dec_mq_f32acc(
                    dp(&dbad, &stream),
                    dp(&dkv, &stream),
                    dpi(&didx, &stream),
                    dp(&dsink, &stream),
                    dpm(&mut sc_bad, &stream),
                    dpm(&mut evals, &stream),
                    dpm(&mut den, &stream),
                    dpm(&mut o, &stream),
                    nq as i32,
                    HEADS as i32,
                    HD as i32,
                    slots as i32,
                    slots as i32,
                    scale,
                    cu,
                )
            };
            assert_eq!(rc, 0, "red arm rc={rc}");
            stream.synchronize()?;
            let bad = e.dtoh(&sc_bad)?;
            if bits_equal(&bad, &b) {
                failures.push(
                    "RED ARM VACUOUS: a one-ULP perturbation of the staged q produced a \
                     byte-identical score plane, so the identity comparison proves nothing"
                        .into(),
                );
            } else {
                let at = first_difference(&bad, &b).unwrap_or(0);
                println!(
                    "  RED ARM sink: one-ULP q perturbation moves the plane at element {at} \
                     ({:e} vs {:e}), so the comparator is live",
                    bad[at], b[at]
                );
                red_arm_proved = true;
            }
        }
    }

    // ---- indexer scorer, its own hd and its own causal limit ------------------------------
    for (s_rows, nb, pos0) in [(512usize, 512usize, 4096usize), (8, 256, 31usize)] {
        let mut rng = Rng(0x1DEC_0911 ^ s_rows as u64);
        let iq: Vec<f32> = (0..s_rows * HEADS * IHD).map(|_| rng.unit()).collect();
        let ckv: Vec<f32> = (0..nb * IHD).map(|_| rng.unit()).collect();
        let w: Vec<f32> = (0..s_rows * HEADS).map(|_| rng.unit()).collect();
        let wscale = ((IHD as f64).powf(-0.5) * (HEADS as f64).powf(-0.5)) as f32;
        let ratio = 4i32;

        let diq = e.htod(&iq)?;
        let dckv = e.htod(&ckv)?;
        let dw = e.htod(&w)?;
        let mut diqt = e.uninit(iq.len())?;
        let mut ship = e.uninit(s_rows * nb)?;
        let mut refr = e.uninit(s_rows * nb)?;

        let rc = unsafe {
            memra_dsv4_q_transpose_m(
                dp(&diq, &stream),
                dpm(&mut diqt, &stream),
                s_rows as i32,
                HEADS as i32,
                IHD as i32,
                cu,
            )
        };
        assert_eq!(rc, 0, "qi transpose rc={rc}");
        let rc = unsafe {
            memra_dsv4_indexer_score_f32acc_pos_m(
                dp(&diqt, &stream),
                dp(&dckv, &stream),
                dp(&dw, &stream),
                wscale,
                dpm(&mut ship, &stream),
                s_rows as i32,
                HEADS as i32,
                IHD as i32,
                nb as i32,
                ratio,
                pos0 as i32,
                cu,
            )
        };
        assert_eq!(rc, 0, "indexer pos_m rc={rc}");
        let rc = unsafe {
            memra_dsv4_indexer_score_f32acc_pos_m_ref(
                dp(&diq, &stream),
                dp(&dckv, &stream),
                dp(&dw, &stream),
                wscale,
                dpm(&mut refr, &stream),
                s_rows as i32,
                HEADS as i32,
                IHD as i32,
                nb as i32,
                ratio,
                pos0 as i32,
                cu,
            )
        };
        assert_eq!(rc, 0, "indexer ref rc={rc}");
        stream.synchronize()?;

        let a = e.dtoh(&ship)?;
        let b = e.dtoh(&refr)?;
        cells += 1;
        // A plane that is entirely -INFINITY would compare equal for the wrong reason, so the
        // cell asserts that the causal limit actually admitted work before it counts.
        let finite = a.iter().filter(|v| v.is_finite()).count();
        if finite == 0 {
            failures.push(format!(
                "indexer s={s_rows} nb={nb} pos0={pos0}: every score is -INFINITY, so this cell \
                 compares nothing"
            ));
        } else if bits_equal(&a, &b) {
            println!(
                "  IDENTITY indexer s={s_rows} nb={nb} pos0={pos0}: {} values bit-equal, \
                 {finite} finite",
                a.len()
            );
        } else {
            let at = first_difference(&a, &b).unwrap_or(0);
            failures.push(format!(
                "indexer s={s_rows} nb={nb}: first difference at {at}, shipped {:e} vs ref {:e}",
                a[at], b[at]
            ));
        }
    }

    assert!(cells >= 6, "gate swept {cells} cells, expected every shape");
    if !red_arm_proved {
        failures.push("RED ARM NEVER RAN: the identity result is unproven".into());
    }
    if failures.is_empty() {
        println!("dsv4-q-layout-gate: PASS ({cells} cells, red arm proved)");
        Ok(())
    } else {
        for f in &failures {
            println!("FAIL {f}");
        }
        println!("dsv4-q-layout-gate: FAIL ({} problems)", failures.len());
        std::process::exit(1);
    }
}
