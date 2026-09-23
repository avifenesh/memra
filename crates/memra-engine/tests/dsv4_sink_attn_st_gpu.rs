//! Kernel-boundary bit gate for the two-launch DSV4 sink attention program (memra #683).
//!
//! `memra_dsv4_sink_attn_st_f32acc` replaces the three-kernel f32acc split (scores, soft,
//! out) on every launch whose geometry it admits: the batched/verify entry
//! (`memra_dsv4_sink_attn_dec_mq_f32acc`, q transposed), the graph replay entry
//! (`memra_dsv4_replay_attention`, live slot count read from the device position) and the
//! single-query eager entry (`memra_dsv4_sink_attn_dec_f32acc`). The claim is bit identity
//! with each, so each is the oracle here: `expf` is not correctly rounded, so a CPU oracle
//! would not be exact, and the former kernels are the program being preserved.
//!
//! Fixtures cover the tile edges (1, 7, 8, 9, 127..129, 255..257 slots), a second and third
//! slot tile of the output kernel (640, 1100), -1 pads including an all-pad query, head and
//! column tiles narrower than the served 64 x 512, and a wide-magnitude regime where most
//! `expf` terms underflow to 0 so the `ev == 0` skip is exercised.
//!
//! Red arm: one element of a live kv row changed by 2^-10 relative must change `o`.
//!
//! `sink_attn_st_timing` is the component instrument: CUDA event chains at the served
//! decode shapes, the three-kernel replay program (plus the q transpose it needs) against
//! the two-launch program.
//!
//! Rig law: run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made in this process yet, and call_once
            // serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 17
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 8) as u32 as f32) / (1u32 << 31) as f32 - 1.0
    }
}

/// Deterministic values in [-1, 1) times a per-element power of two in 2^lo .. 2^hi.
fn fixture(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<f32> {
    let mut r = Lcg(seed | 1);
    (0..n)
        .map(|_| {
            let u = r.unit();
            let e = lo + r.below((hi - lo + 1) as usize) as i32;
            u * 2f32.powi(e)
        })
        .collect()
}

/// `nq` rows of `stride` indices: the first `slots` of each row select kv rows (with
/// repeats, as the window and the compressed top-k can), a `pad_pct` share of them -1.
/// Query 1 of a multi-query case is all -1 when `all_pad_q1`.
fn indices(
    nq: usize,
    slots: usize,
    stride: usize,
    rows: usize,
    pad_pct: usize,
    all_pad_q1: bool,
    seed: u64,
) -> Vec<i32> {
    let mut r = Lcg(seed | 1);
    let mut v = vec![-7i32; nq * stride];
    for p in 0..nq {
        for s in 0..slots {
            let pad = (all_pad_q1 && p == 1) || r.below(100) < pad_pct;
            v[p * stride + s] = if pad { -1 } else { r.below(rows) as i32 };
        }
    }
    v
}

fn dp(s: &CudaSlice<f32>, e: &Engine) -> *const f32 {
    s.device_ptr(&e.stream()).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, e: &Engine) -> *mut f32 {
    s.device_ptr_mut(&e.stream()).0 as *mut f32
}
fn ip(s: &CudaSlice<i32>, e: &Engine) -> *const i32 {
    s.device_ptr(&e.stream()).0 as *const i32
}
fn sv(e: &Engine) -> *mut c_void {
    e.stream().cu_stream() as *mut c_void
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits_eq(what: &str, got: &[u32], want: &[u32], case: &str) {
    assert_eq!(got.len(), want.len(), "{case}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{case}: {what}[{i}] st {:#010x} != three-kernel {:#010x}",
            got[i], want[i]
        );
    }
}

const SENTINEL: f32 = f32::from_bits(0x7fc0_dead);

struct Fix {
    q: CudaSlice<f32>,
    kv: CudaSlice<f32>,
    sink: CudaSlice<f32>,
    idx: CudaSlice<i32>,
}

fn fix(
    e: &Engine,
    nq: usize,
    heads: usize,
    hd: usize,
    rows: usize,
    idx: &[i32],
    mag: (i32, i32),
    seed: u64,
) -> Fix {
    Fix {
        q: e.htod(&fixture(nq * heads * hd, seed, mag.0, mag.1))
            .unwrap(),
        kv: e
            .htod(&fixture(rows * hd, seed ^ 0x77, mag.0, mag.1))
            .unwrap(),
        sink: e.htod(&fixture(heads, seed ^ 0x5151, -2, 3)).unwrap(),
        idx: e.htod_i32(idx).unwrap(),
    }
}

/// The three-kernel batched program: q transpose, then scores/soft/out.
fn mq_old(
    e: &Engine,
    f: &Fix,
    nq: usize,
    heads: usize,
    hd: usize,
    slots: usize,
    stride: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut qt = e.uninit(nq * heads * hd).unwrap();
    let mut scores = e.htod(&vec![SENTINEL; nq * heads * slots]).unwrap();
    let mut evals = e.uninit(nq * heads * slots).unwrap();
    let mut den = e.uninit(nq * heads).unwrap();
    let mut o = e.htod(&vec![SENTINEL; nq * heads * hd]).unwrap();
    let scale = (hd as f64).powf(-0.5) as f32;
    unsafe {
        let rc = k::memra_dsv4_q_transpose_m(
            dp(&f.q, e),
            dpm(&mut qt, e),
            nq as i32,
            heads as i32,
            hd as i32,
            sv(e),
        );
        assert_eq!(rc, 0, "q_transpose_m");
        let rc = k::memra_dsv4_sink_attn_dec_mq_f32acc(
            dp(&qt, e),
            dp(&f.kv, e),
            ip(&f.idx, e),
            dp(&f.sink, e),
            dpm(&mut scores, e),
            dpm(&mut evals, e),
            dpm(&mut den, e),
            dpm(&mut o, e),
            nq as i32,
            heads as i32,
            hd as i32,
            slots as i32,
            stride as i32,
            scale,
            sv(e),
        );
        assert_eq!(rc, 0, "sink_attn_dec_mq_f32acc");
    }
    (e.dtoh(&scores).unwrap(), e.dtoh(&o).unwrap())
}

#[allow(clippy::too_many_arguments)]
fn st(
    e: &Engine,
    f: &Fix,
    nq: usize,
    heads: usize,
    hd: usize,
    slots: usize,
    stride: usize,
    replay: Option<(&CudaSlice<i32>, usize, usize, i32)>,
) -> (Vec<f32>, Vec<f32>) {
    let mut scores = e.htod(&vec![SENTINEL; nq * heads * slots]).unwrap();
    let mut o = e.htod(&vec![SENTINEL; nq * heads * hd]).unwrap();
    let scale = (hd as f64).powf(-0.5) as f32;
    let (pos, win, ratio, topk) = replay.map_or((std::ptr::null(), 0, 0, 0), |(p, w, r, t)| {
        (ip(p, e), w as i32, r as i32, t)
    });
    let rc = unsafe {
        k::memra_dsv4_sink_attn_st_f32acc(
            dp(&f.q, e),
            dp(&f.kv, e),
            ip(&f.idx, e),
            dp(&f.sink, e),
            dpm(&mut scores, e),
            dpm(&mut o, e),
            nq as i32,
            heads as i32,
            hd as i32,
            slots as i32,
            stride as i32,
            scale,
            pos,
            win,
            ratio,
            topk,
            sv(e),
        )
    };
    assert_eq!(rc, 0, "sink_attn_st_f32acc");
    (e.dtoh(&scores).unwrap(), e.dtoh(&o).unwrap())
}

const SLOTS: [usize; 13] = [1, 7, 8, 9, 127, 128, 129, 187, 255, 256, 257, 640, 1100];

#[test]
#[ignore]
fn sink_attn_st_matches_three_kernel_batched() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let rows = 1536;
    let mut cases = 0;
    for (hi, &(heads, hd)) in [(64usize, 512usize), (32, 512), (16, 256), (48, 64)]
        .iter()
        .enumerate()
    {
        assert_ne!(
            unsafe { k::memra_dsv4_sink_attn_st_admits(heads as i32, hd as i32) },
            0
        );
        for (si, &slots) in SLOTS.iter().enumerate() {
            for nq in [1usize, 3] {
                for (mi, mag) in [(-2, 1), (-1, 4)].into_iter().enumerate() {
                    let pad = [0usize, 30][(si + mi) % 2];
                    let stride = slots + 5;
                    let seed = 0x5a_0000 + (hi * 4096 + si * 64 + nq * 8 + mi) as u64;
                    let idx = indices(nq, slots, stride, rows, pad, nq == 3, seed);
                    let f = fix(&e, nq, heads, hd, rows, &idx, mag, seed);
                    let case = format!(
                        "heads={heads} hd={hd} slots={slots} nq={nq} mag=2^{}..2^{} pad={pad}%",
                        mag.0, mag.1
                    );
                    let (s_old, o_old) = mq_old(&e, &f, nq, heads, hd, slots, stride);
                    let (s_new, o_new) = st(&e, &f, nq, heads, hd, slots, stride, None);
                    assert_bits_eq("scores", &bits(&s_new), &bits(&s_old), &case);
                    assert_bits_eq("o", &bits(&o_new), &bits(&o_old), &case);
                    cases += 1;
                }
            }
        }
    }
    println!("sink_attn_st batched: {cases} cases bit-identical to the three-kernel program");
}

#[test]
#[ignore]
fn sink_attn_st_matches_three_kernel_replay() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let rows = 4096;
    let win = 128usize;
    let mut cases = 0;
    for &heads in &[64usize, 32] {
        let hd = 512usize;
        // (ratio, topk, replay_limit): the served ratio-0 window layers, ratio-4 layers with
        // the 512 top-k cap, and ratio-128 layers whose compressed rows are uncapped.
        for &(ratio, topk, limit) in &[
            (0usize, i32::MAX, 1024usize),
            (4, 512, 1024),
            (4, 512, 4096),
            (128, i32::MAX, 65536),
        ] {
            let slots_max = win + limit.checked_div(ratio).map_or(0, |n| n.min(topk as usize));
            let stride = slots_max + 3;
            let seed = 0x7e_0000 + (heads * 131 + ratio * 7 + limit) as u64;
            let idx = indices(1, slots_max, stride, rows, 10, false, seed);
            let f = fix(&e, 1, heads, hd, rows, &idx, (-2, 2), seed);
            for &pos in &[
                0usize, 2, 3, 126, 127, 235, 236, 511, 1023, 2046, 2047, 4095, 65535,
            ] {
                if pos >= limit {
                    continue;
                }
                let pos_d = e.htod_i32(&[pos as i32]).unwrap();
                let live = win
                    + if ratio == 0 {
                        0
                    } else {
                        ((pos + 1) / ratio).min(topk as usize)
                    };
                let case = format!(
                    "replay heads={heads} ratio={ratio} topk={topk} slots_max={slots_max} pos={pos} live={live}"
                );
                let mut qt = e.uninit(heads * hd).unwrap();
                let mut scores = e.htod(&vec![SENTINEL; heads * slots_max]).unwrap();
                let mut evals = e.uninit(heads * slots_max).unwrap();
                let mut den = e.uninit(heads).unwrap();
                let mut o = e.htod(&vec![SENTINEL; heads * hd]).unwrap();
                let scale = (hd as f64).powf(-0.5) as f32;
                unsafe {
                    let rc = k::memra_dsv4_q_transpose_m(
                        dp(&f.q, &e),
                        dpm(&mut qt, &e),
                        1,
                        heads as i32,
                        hd as i32,
                        sv(&e),
                    );
                    assert_eq!(rc, 0);
                    let rc = k::memra_dsv4_replay_attention(
                        dp(&qt, &e),
                        dp(&f.kv, &e),
                        ip(&f.idx, &e),
                        dp(&f.sink, &e),
                        dpm(&mut scores, &e),
                        dpm(&mut evals, &e),
                        dpm(&mut den, &e),
                        dpm(&mut o, &e),
                        ip(&pos_d, &e),
                        heads as i32,
                        hd as i32,
                        slots_max as i32,
                        stride as i32,
                        scale,
                        win as i32,
                        ratio as i32,
                        topk,
                        sv(&e),
                    );
                    assert_eq!(rc, 0, "replay_attention");
                }
                let s_old = e.dtoh(&scores).unwrap();
                let o_old = e.dtoh(&o).unwrap();
                let (s_new, o_new) = st(
                    &e,
                    &f,
                    1,
                    heads,
                    hd,
                    slots_max,
                    stride,
                    Some((&pos_d, win, ratio, topk)),
                );
                // Both programs lay scores out with the live count as the row stride.
                let n = heads * live;
                assert_bits_eq("scores", &bits(&s_new[..n]), &bits(&s_old[..n]), &case);
                assert_bits_eq("o", &bits(&o_new), &bits(&o_old), &case);
                cases += 1;
            }
        }
    }
    println!("sink_attn_st replay: {cases} cases bit-identical to the three-kernel program");
}

#[test]
#[ignore]
fn sink_attn_st_matches_three_kernel_single_query() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let rows = 1536;
    let mut cases = 0;
    for &(heads, hd) in &[(64usize, 512usize), (32, 512)] {
        for (si, &slots) in SLOTS.iter().enumerate() {
            let seed = 0x51_0000 + (heads * 97 + si) as u64;
            let idx = indices(1, slots, slots, rows, [0, 20][si % 2], false, seed);
            let f = fix(&e, 1, heads, hd, rows, &idx, (-2, 2), seed);
            let case = format!("single heads={heads} hd={hd} slots={slots}");
            let mut scores = e.htod(&vec![SENTINEL; heads * slots]).unwrap();
            let mut evals = e.uninit(heads * slots).unwrap();
            let mut den = e.uninit(heads).unwrap();
            let mut o = e.htod(&vec![SENTINEL; heads * hd]).unwrap();
            let scale = (hd as f64).powf(-0.5) as f32;
            let rc = unsafe {
                k::memra_dsv4_sink_attn_dec_f32acc(
                    dp(&f.q, &e),
                    dp(&f.kv, &e),
                    ip(&f.idx, &e),
                    dp(&f.sink, &e),
                    dpm(&mut scores, &e),
                    dpm(&mut evals, &e),
                    dpm(&mut den, &e),
                    dpm(&mut o, &e),
                    heads as i32,
                    hd as i32,
                    slots as i32,
                    scale,
                    sv(&e),
                )
            };
            assert_eq!(rc, 0, "sink_attn_dec_f32acc");
            let (s_new, o_new) = st(&e, &f, 1, heads, hd, slots, slots, None);
            assert_bits_eq(
                "scores",
                &bits(&s_new),
                &bits(&e.dtoh(&scores).unwrap()),
                &case,
            );
            assert_bits_eq("o", &bits(&o_new), &bits(&e.dtoh(&o).unwrap()), &case);
            cases += 1;
        }
    }
    println!("sink_attn_st single query: {cases} cases bit-identical to the three-kernel program");
}

#[test]
#[ignore]
fn sink_attn_st_red_arm() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let (nq, heads, hd, slots, rows) = (1usize, 64usize, 512usize, 187usize, 1536usize);
    let idx = indices(nq, slots, slots, rows, 0, false, 0xbad);
    let f = fix(&e, nq, heads, hd, rows, &idx, (-2, 1), 0xbad);
    let (_, o_ref) = st(&e, &f, nq, heads, hd, slots, slots, None);
    let mut kv = e.dtoh(&f.kv).unwrap();
    let live = idx[slots / 2] as usize;
    kv[live * hd + 17] *= 1.0 + 2f32.powi(-10);
    let g = Fix {
        q: f.q.clone(),
        kv: e.htod(&kv).unwrap(),
        sink: f.sink.clone(),
        idx: f.idx.clone(),
    };
    let (_, o_pert) = st(&e, &g, nq, heads, hd, slots, slots, None);
    let (_, o_pert_old) = mq_old(&e, &g, nq, heads, hd, slots, slots);
    assert_bits_eq(
        "o",
        &bits(&o_pert),
        &bits(&o_pert_old),
        "red arm perturbed input",
    );
    assert_ne!(
        bits(&o_ref),
        bits(&o_pert),
        "a perturbed live kv element must change o"
    );
    println!("sink_attn_st red arm: a 2^-10 change to one live kv element is caught");
}

fn chain_us(e: &Engine, launches: usize, mut f: impl FnMut()) -> f64 {
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    for _ in 0..64 {
        f();
    }
    let start = e.stream().record_event(flags).unwrap();
    for _ in 0..launches {
        f();
    }
    let end = e.stream().record_event(flags).unwrap();
    e.stream().synchronize().unwrap();
    1000.0 * f64::from(start.elapsed_ms(&end).unwrap()) / launches as f64
}

/// Served decode shapes on the graph replay entry: window layers (128 live slots), a
/// ratio-4 layer early in a request (187 live) and at the top-k cap (640 live), and the
/// TP/EP attention shard (32 heads). `old` is the q transpose plus the three-kernel replay
/// program; `st` is the two-launch program, which reads q untransposed.
#[test]
#[ignore]
fn sink_attn_st_timing() {
    force_true_f32();
    let _g = gpu_guard();
    let e = Engine::new(0).unwrap();
    let launches = 2000;
    let reps = 5;
    let tree = std::env::var("DSV4_LATENCY_TREE").unwrap_or_else(|_| "unlabeled".into());
    let (hd, win, rows) = (512usize, 128usize, 4096usize);
    for &heads in &[64usize, 32] {
        for &(ratio, topk, limit, pos) in &[
            (0usize, i32::MAX, 1024usize, 236usize),
            (4, 512, 1024, 236),
            (4, 512, 4096, 4095),
        ] {
            let slots_max = win + limit.checked_div(ratio).map_or(0, |n| n.min(topk as usize));
            let live = win
                + if ratio == 0 {
                    0
                } else {
                    ((pos + 1) / ratio).min(topk as usize)
                };
            let idx = indices(1, slots_max, slots_max, rows, 0, false, 9);
            let f = fix(&e, 1, heads, hd, rows, &idx, (-2, 2), 9);
            let pos_d = e.htod_i32(&[pos as i32]).unwrap();
            let mut qt = e.uninit(heads * hd).unwrap();
            let mut scores = e.uninit(heads * slots_max).unwrap();
            let mut evals = e.uninit(heads * slots_max).unwrap();
            let mut den = e.uninit(heads).unwrap();
            let mut o = e.uninit(heads * hd).unwrap();
            let scale = (hd as f64).powf(-0.5) as f32;
            let (qp, kvp, ixp, snp, pp) = (
                dp(&f.q, &e),
                dp(&f.kv, &e),
                ip(&f.idx, &e),
                dp(&f.sink, &e),
                ip(&pos_d, &e),
            );
            let (qtp, scp, evp, dnp, op) = (
                dpm(&mut qt, &e),
                dpm(&mut scores, &e),
                dpm(&mut evals, &e),
                dpm(&mut den, &e),
                dpm(&mut o, &e),
            );
            let shape = format!("h{heads}_r{ratio}_live{live}");
            for rep in 0..reps {
                let old = chain_us(&e, launches, || unsafe {
                    assert_eq!(
                        k::memra_dsv4_q_transpose_m(qp, qtp, 1, heads as i32, hd as i32, sv(&e)),
                        0
                    );
                    assert_eq!(
                        k::memra_dsv4_replay_attention(
                            qtp,
                            kvp,
                            ixp,
                            snp,
                            scp,
                            evp,
                            dnp,
                            op,
                            pp,
                            heads as i32,
                            hd as i32,
                            slots_max as i32,
                            slots_max as i32,
                            scale,
                            win as i32,
                            ratio as i32,
                            topk,
                            sv(&e),
                        ),
                        0
                    );
                });
                let new = chain_us(&e, launches, || unsafe {
                    assert_eq!(
                        k::memra_dsv4_sink_attn_st_f32acc(
                            qp,
                            kvp,
                            ixp,
                            snp,
                            scp,
                            op,
                            1,
                            heads as i32,
                            hd as i32,
                            slots_max as i32,
                            slots_max as i32,
                            scale,
                            pp,
                            win as i32,
                            ratio as i32,
                            topk,
                            sv(&e),
                        ),
                        0
                    );
                });
                println!(
                    "TIMING tree={tree} kernel=sink_attn shape={shape} rep={rep} old_us={old:.3} st_us={new:.3}"
                );
            }
        }
    }
}
