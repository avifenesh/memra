//! Kernel-boundary bit gate for BF16 island storage (compressor wkv/wgate).
//!
//! The DSv4 checkpoint stores every compressor projection as BF16. The engine kept them as f32
//! islands; it now keeps the BF16 bytes and passes `w_is_bf16 = 1`. The claim: every dots entry
//! point the compressor reaches gives the same bits from the BF16 plane as from its exact f32
//! widening, at every row count the served program runs (one decode row, the DSpark verify
//! width, a prefill chunk), and the f32acc entry's dense-fast and exact-tail arms both hold.
//!
//! Red arm: one row's BF16 weights moved by one ulp must move that row's output and no other, so
//! a PASS cannot come from both launches reading the same plane.
//!
//! Rig law: correctness-only, one CUDA card under the rig's lock, `-- --ignored --test-threads=1`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

/// Compressor shapes of DSV4-Flash: ratio-4 attention (latent 1024), ratio-128 attention
/// (latent 512), indexer (latent 256), all over hidden 4096.
const SHAPES: [(usize, usize); 3] = [(1024, 4096), (512, 4096), (256, 4096)];
const ROWS: [usize; 4] = [1, 2, 6, 33];

type Dots = unsafe extern "C" fn(
    *const f32,
    *const c_void,
    i32,
    *mut f32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;

fn fixture(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Finite BF16 over many exponents, both signs, no NaN/Inf.
            let exp = 100 + ((s >> 40) % 40) as u16;
            let sign = ((s >> 63) as u16) << 15;
            let mant = ((s >> 20) & 0x7f) as u16;
            sign | (exp << 7) | mant
        })
        .collect()
}

fn widen(w: &[u16]) -> Vec<f32> {
    w.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

fn activations(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            ((s >> 33) as u32 as f32 / u32::MAX as f32 - 0.5)
                * 2f32.powi(((s >> 13) % 9) as i32 - 4)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    e: &Engine,
    f: Dots,
    x: &CudaSlice<f32>,
    w: *const c_void,
    bf16: i32,
    s: usize,
    kdim: usize,
    n: usize,
) -> Vec<u32> {
    let mut y: CudaSlice<f32> = e.htod(&vec![f32::from_bits(0x7fc0_1234); s * n]).unwrap();
    let stream = e.stream();
    let rc = unsafe {
        f(
            x.device_ptr(&stream).0 as *const f32,
            w,
            bf16,
            y.device_ptr_mut(&stream).0 as *mut f32,
            s as i32,
            kdim as i32,
            n as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    assert_eq!(rc, 0, "dots launch");
    e.dtoh(&y).unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs a CUDA device; run under the rig's GPU lock"]
fn bf16_island_storage_is_bit_identical_to_the_f32_widening() {
    if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
        // SAFETY: before any CUDA call in this single-test process.
        unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
    }
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let entries: [(&str, Dots); 4] = [
        ("dots_f32", k::memra_dsv4_dots_f32),
        ("dots_f32acc", k::memra_dsv4_dots_f32acc),
        ("dots_f32_mrow", k::memra_dsv4_dots_f32_mrow),
        ("dots_f32acc_mrow", k::memra_dsv4_dots_f32acc_mrow),
    ];
    let mut cases = 0;
    for (si, &(n, kdim)) in SHAPES.iter().enumerate() {
        let wb = fixture(n * kdim, 0xB16 ^ si as u64);
        let wf = widen(&wb);
        let wb_dev: CudaSlice<u16> = e.stream().clone_htod(&wb).unwrap();
        let wf_dev: CudaSlice<f32> = e.htod(&wf).unwrap();
        let stream = e.stream();
        let pb = wb_dev.device_ptr(&stream).0 as *const c_void;
        let pf = wf_dev.device_ptr(&stream).0 as *const c_void;
        for &s in &ROWS {
            let x: CudaSlice<f32> = e
                .htod(&activations(s * kdim, 0xA11 ^ (s as u64) << 8 ^ si as u64))
                .unwrap();
            for (name, f) in entries {
                let from_bf16 = run(&e, f, &x, pb, 1, s, kdim, n);
                let from_f32 = run(&e, f, &x, pf, 0, s, kdim, n);
                if let Some(i) = (0..from_bf16.len()).find(|&i| from_bf16[i] != from_f32[i]) {
                    panic!(
                        "{name} n={n} k={kdim} rows={s}: out[{i}] bf16 {:#010x} != f32 {:#010x}",
                        from_bf16[i], from_f32[i]
                    );
                }
                assert!(
                    from_bf16.iter().all(|&b| b != 0x7fc0_1234),
                    "{name}: unwritten output"
                );
                cases += 1;
            }
        }
    }
    // Red arm: move row 7's BF16 weights by one ulp; row 7 must move, row 6 must not.
    let (n, kdim) = SHAPES[2];
    let mut wb = fixture(n * kdim, 0xB16 ^ 2);
    let x: CudaSlice<f32> = e.htod(&activations(kdim, 0xA11 ^ 2)).unwrap();
    let clean_dev: CudaSlice<u16> = e.stream().clone_htod(&wb).unwrap();
    let stream = e.stream();
    let clean = run(
        &e,
        k::memra_dsv4_dots_f32acc,
        &x,
        clean_dev.device_ptr(&stream).0 as *const c_void,
        1,
        1,
        kdim,
        n,
    );
    // The whole row: one weight of a 4096-term dot can sit below the sum's ulp.
    for w in &mut wb[7 * kdim..8 * kdim] {
        *w = w.wrapping_add(1);
    }
    let red_dev: CudaSlice<u16> = e.stream().clone_htod(&wb).unwrap();
    let red = run(
        &e,
        k::memra_dsv4_dots_f32acc,
        &x,
        red_dev.device_ptr(&stream).0 as *const c_void,
        1,
        1,
        kdim,
        n,
    );
    assert_ne!(
        red[7], clean[7],
        "red arm: a moved weight did not move its row"
    );
    assert_eq!(red[6], clean[6], "red arm: a neighbouring row moved");
    println!("DSV4_ISLAND_BF16 EXACT cases={cases} entries=5 shapes=3 rows=1,2,6,33 red_arm=1");
}
