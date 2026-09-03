//! THE SERVING-SHAPE REPRODUCTION for the decode-graph door's T=1 MoE consumer
//! (lane/b200-glm5-graph-20260902).
//!
//! WHY THIS EXISTS, and why it is shaped exactly like this. Eleven box takes narrowed the
//! token-0 tape to one choice: with `MEMRA_GLM5_GRAPH_HOST_MOE=1` (host-oracle MoE) the tape is
//! correct, and with the T=1 device-table arm it is wrong from step 1. Both arms are handed
//! IDENTICAL inputs (take 9 diffed `sel`, `w`, macro scales, strides, row_bytes, qtypes, limit
//! and `gu_il` and found nothing). Two rig gates already clear the pieces in isolation: the
//! table VALUES match the host arithmetic bit for bit even at strides past 4 GiB
//! (`glm5_vrows_dev_tables_gpu`), and the kernel pair matches the sequential chain bit for bit at
//! t=1 (`glm5_verify_batch_gpu`, loop extended to t=1 by this lane).
//!
//! What neither of those reaches is the SERVING SCALE. This one does: 288 experts, `in_f` 4096
//! gate/up and 2048 down, `row_bytes` 2304/2304/1152 and `expert_stride` 4 718 592 — the exact
//! numbers the box dumped — so each plane's pointer span is 1.36 GB and the whole bank is 4.1 GB.
//! And it drives it with the REAL ROUTING from the box (`box/sel-slice-50mb.bin`, records of
//! `u8 layer, u8 n_sel, n_sel x (u16 expert, f32 w)`), not a synthetic `(p*5) % n_expert` pattern,
//! because a selection that never repeats an expert and never spreads across the bank is not the
//! selection the failure happens under.
//!
//! WHAT IT COMPARES, and why that is the right cut. Both arms run the SAME kernel pair on the
//! SAME bytes; the ONLY difference is where the pointer/scale tables came from — the host loop
//! (`VrowsSel::Host`) or `moe_vrows_tables_from_sel` (`VrowsSel::Dev`). That is precisely the one
//! choice the door makes differently from the shipped decode path, reduced to two launches and no
//! model. If the outputs differ here, the defect is reproduced on the rig and debuggable locally;
//! if they agree, the consumer is clean at serving scale and the fault is in the capture/replay
//! binding instead — which is the other live hypothesis, and this gate is what excludes this one.
//!
//! The slab bytes are generated directly rather than through `f32_to_nvfp4` per row: 2.4M row
//! conversions would dominate the runtime and buy nothing, since both arms read the same bytes
//! and the comparison is bit identity, not realism. Bytes are kept in a range that dequantizes
//! finite and O(1) so a fixture-induced inf/NaN cannot be mistaken for the failure under test.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`,
//! `-- --ignored --test-threads=1`. Needs ~4.1 GB free VRAM.

use cudarc::driver::DevicePtr;
use memra_engine::{Engine, QT_NVFP4};

const N_EXPERT: usize = 288;
const N_USED: usize = 8;
const IN_F: usize = 4096; // gate/up in, and the down projection's out
const N_FF: usize = 2048; // gate/up out, and the down projection's in

fn nvfp4_row_bytes(in_f: usize) -> usize {
    in_f / 2 + in_f / 16
}

struct Lcg(u64);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / 16777216.0) * 2.0 - 1.0
    }
}

/// One expert bank at the serving shape. Bytes are held in the 0x30..0x3F range so every e4m3
/// scale and every nvfp4 nibble dequantizes finite and O(1): a fixture that produced its own
/// inf/NaN could not tell us anything about the failure under test.
fn serving_slab(
    e: &Engine,
    out_f: usize,
    in_f: usize,
    seed: u64,
) -> (cudarc::driver::CudaSlice<u8>, usize, usize) {
    let rb = nvfp4_row_bytes(in_f);
    let stride = out_f * rb;
    let mut bytes = vec![0u8; N_EXPERT * stride];
    let mut r = Lcg(seed);
    for b in bytes.iter_mut() {
        *b = 0x30 | (r.byte() & 0x0F);
    }
    let slab = e.htod_bytes(&bytes).expect("serving slab upload");
    (slab, rb, stride)
}

/// `(layer, sel, w)` records from the box's routing dump: `u8 layer, u8 n_sel,
/// n_sel x (u16 expert little-endian, f32 w little-endian)`.
fn read_sel_slice(path: &std::path::Path, want: usize) -> Vec<(u8, Vec<u32>, Vec<f32>)> {
    let Ok(raw) = std::fs::read(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 2 <= raw.len() && out.len() < want {
        let layer = raw[i];
        let n = raw[i + 1] as usize;
        let need = 2 + n * 6;
        if n == 0 || i + need > raw.len() {
            break;
        }
        let (mut sel, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for j in 0..n {
            let o = i + 2 + j * 6;
            sel.push(u16::from_le_bytes([raw[o], raw[o + 1]]) as u32);
            w.push(f32::from_le_bytes([
                raw[o + 2],
                raw[o + 3],
                raw[o + 4],
                raw[o + 5],
            ]));
        }
        if n == N_USED && sel.iter().all(|&x| (x as usize) < N_EXPERT) {
            out.push((layer, sel, w));
        }
        i += need;
    }
    out
}

fn sum(v: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let (mut nz, mut absmax) = (0usize, 0f32);
    for x in v {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
        if *x != 0.0 {
            nz += 1;
        }
        if x.abs() > absmax {
            absmax = x.abs();
        }
    }
    format!("0x{h:016x} nz={nz}/{} absmax={absmax:.6e}", v.len())
}

/// The kernel pair, run over tables the caller supplies. Both arms differ ONLY in how `ptrs_d`
/// and `scl_d` were filled — same bytes, same launches, same everything else.
#[allow(clippy::too_many_arguments)]
fn run_pair(
    e: &Engine,
    ptrs_d: &cudarc::driver::CudaSlice<u64>,
    scl_d: &cudarc::driver::CudaSlice<f32>,
    z_d: &cudarc::driver::CudaSlice<f32>,
    rb_gu: usize,
    rb_d: usize,
    limit: f32,
) -> Vec<f32> {
    let n_pairs = N_USED;
    let mut zq = e.alloc_i8_uninit(IN_F).expect("zq");
    let mut zd = e.zeros(IN_F / 32).expect("zd");
    e.quantize_q8_1_into(z_d, 1, IN_F, &mut zq, &mut zd)
        .expect("quantize z");
    let act = e
        .moe_gate_up_preclamp8_q8_rows(
            ptrs_d, scl_d, &zq, &zd, limit, IN_F, N_FF, N_USED, n_pairs, QT_NVFP4, QT_NVFP4, rb_gu,
            rb_gu,
        )
        .expect("gate/up rows");
    let mut aq2 = e.alloc_i8_uninit(n_pairs * N_FF).expect("aq2");
    let mut ad2 = e.zeros(n_pairs * (N_FF / 32)).expect("ad2");
    e.quantize_q8_1_into(&act, n_pairs, N_FF, &mut aq2, &mut ad2)
        .expect("quantize act");
    let mut out = e.zeros(IN_F).expect("out");
    e.moe_down8_fma_q8_rows(
        ptrs_d, scl_d, &aq2, &ad2, &mut out, N_FF, IN_F, N_USED, n_pairs, QT_NVFP4, rb_d,
    )
    .expect("down rows");
    e.dtoh(&out).expect("out readback")
}

#[test]
#[ignore = "needs a CUDA device with ~4.1 GB free — run under flock /tmp/memra-5090.lock"]
fn vrows_t1_device_tables_match_host_tables_at_serving_scale()
-> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("[vrows-t1-serving] GPU0: {}", e.ctx().name()?);

    let slice = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/b200-glm5-graph-20260902/box/sel-slice-50mb.bin");
    let records = read_sel_slice(&slice, 6);
    // NON-VACUITY: this gate exists to run the BOX's routing. A synthetic fallback would let it
    // pass while testing a selection the failure never happens under, so it refuses instead.
    assert!(
        !records.is_empty(),
        "no usable routing records at {} — this gate must run the box's real selection",
        slice.display()
    );
    println!(
        "[vrows-t1-serving] {} routing records, layers {:?}",
        records.len(),
        records.iter().map(|(l, _, _)| *l).collect::<Vec<_>>()
    );

    let (rb_gu, rb_d) = (nvfp4_row_bytes(IN_F), nvfp4_row_bytes(N_FF));
    assert_eq!(
        (rb_gu, rb_d),
        (2304, 1152),
        "row_bytes must match the box dump"
    );
    let (gate, _, sg) = serving_slab(&e, N_FF, IN_F, 0x51A0);
    let (up, _, su) = serving_slab(&e, N_FF, IN_F, 0x51A1);
    let (down, _, sd) = serving_slab(&e, IN_F, N_FF, 0x51A2);
    assert_eq!(
        (sg, su, sd),
        (4718592, 4718592, 4718592),
        "expert_stride must match the box dump"
    );
    println!(
        "[vrows-t1-serving] banks up: 3 x {:.2} GB, stride {sg}",
        (N_EXPERT * sg) as f64 / 1e9
    );

    let st = e.stream();
    let (pg, _g0) = gate.device_ptr(&st);
    let (pu, _g1) = up.device_ptr(&st);
    let (pd, _g2) = down.device_ptr(&st);

    // Live macro planes: dropping or mis-indexing any fold has to be visible.
    let mg: Vec<f32> = (0..N_EXPERT).map(|i| 0.5 + 0.003 * i as f32).collect();
    let mu: Vec<f32> = (0..N_EXPERT).map(|i| 1.6 - 0.002 * i as f32).collect();
    let md: Vec<f32> = (0..N_EXPERT).map(|i| 0.8 + 0.004 * i as f32).collect();
    let limit = 10.0f32; // the box's own clamp (limit=Some(Pre(10.0)))

    let mut r = Lcg(0xBEEF);
    let z: Vec<f32> = (0..IN_F).map(|_| r.unit()).collect();
    let z_d = e.htod(&z)?;

    let n_pairs = N_USED;
    let mut bad = 0usize;
    for (rec, (layer, sel, w)) in records.iter().enumerate() {
        // ---- arm HOST: the `VrowsSel::Host` table arithmetic, verbatim ----
        let mut hp = vec![0u64; 3 * n_pairs];
        let mut hs = vec![0f32; 3 * n_pairs];
        for (p, (&ex, &wt)) in sel.iter().zip(w).enumerate() {
            let ex = ex as usize;
            hp[p] = pg + (ex * sg) as u64;
            hp[n_pairs + p] = pu + (ex * su) as u64;
            hp[2 * n_pairs + p] = pd + (ex * sd) as u64;
            hs[p] = mg[ex];
            hs[n_pairs + p] = mu[ex];
            hs[2 * n_pairs + p] = wt * md[ex];
        }
        let ptrs_h = e.htod_u64(&hp)?;
        let scl_h = e.htod(&hs)?;
        let out_host = run_pair(&e, &ptrs_h, &scl_h, &z_d, rb_gu, rb_d, limit);

        // ---- arm DEVICE: `moe_vrows_tables_from_sel`, the door's provenance ----
        let sel_i: Vec<i32> = sel.iter().map(|&x| x as i32).collect();
        let sel_d = e.htod_i32(&sel_i)?;
        let selw_d = e.htod(w)?;
        let mut ptrs_v = e.htod_u64(&vec![0u64; 3 * n_pairs])?;
        let mut scl_v = e.htod(&vec![0f32; 3 * n_pairs])?;
        e.moe_vrows_tables_from_sel(
            &sel_d,
            &selw_d,
            *layer as u16,
            Some((&mg, &mu, &md)),
            (pg, pu, pd),
            (sg, su, sd),
            n_pairs,
            &mut ptrs_v,
            &mut scl_v,
        )?;
        let out_dev = run_pair(&e, &ptrs_v, &scl_v, &z_d, rb_gu, rb_d, limit);

        let diffs = out_host
            .iter()
            .zip(&out_dev)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "  rec{rec} layer={layer} sel={:?}\n    host {}\n    dev  {}\n    diffs {diffs}/{}",
            sel,
            sum(&out_host),
            sum(&out_dev),
            out_host.len()
        );
        if diffs > 0 {
            bad += 1;
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad}/{} routing records diverge between host and device tables at serving scale",
        records.len()
    );
    println!("[vrows-t1-serving] all records bit-identical between table provenances");
    Ok(())
}
