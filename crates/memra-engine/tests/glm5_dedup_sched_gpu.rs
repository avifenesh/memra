//! glm5 EXPERT-SLAB DEDUP schedule door (lane/glm5-dedup, 2026-08-31): rig exactness gates.
//!
//! WHY THIS LANE EXISTS, in one measured number: the struct-battery instrument booted
//! `MEMRA_MOE_VROWS_DEDUP_STAT=1` on the real artifact and the ship recipe and read a **21.96%
//! cumulative repeat fraction** over 99,751 vrows layer-calls / 2.55M expert visits, mode-stable
//! (22.27% greedy / 21.53% vendor-default sampled), i.e. **6.9x the 3.21% independent-routing
//! bound**. About a fifth of the pair's (verify row, expert) visits re-read a slab a sibling row
//! of the SAME layer-call already read. The pair is at 90.2% / 89.9% of this card class's
//! theoretical DRAM peak (moe-loc LANE.md §1.3), so there is no efficiency to win on the gate/up
//! half: the only lever there was READING LESS via a scheduled visit order.
//!
//! Door E-down (`MEMRA_MOE_VROWS_DOWN_TMAJ`): the down launch takes `_tmaj`: grid transposed to
//! `(t, out_f)`, token fastest, so a repeated expert's down row is read once for every token that
//! shares it. The slot-ordered `__fmaf_rn` chain lives INSIDE the block and keeps its original
//! slot order; only the grid moves.
//!
//! Door E (`MEMRA_MOE_VROWS_DEDUP_ORDER`, the gate/up expert-major order-plane twin) was REMOVED
//! 2026-09-28 (memra#886, door hygiene): neutral on the pair (-0.18%, one-boot outlier aside;
//! darklanes verdicts-ledger `glm5-b200-vrows-pack-plus-0-70-ord-neutral`), never armed in the
//! served launcher, past its decide-by with no positive receipt. Its dedicated gates (the device
//! order-plane build, the expert-major gate/up bitwise suite, and the shuffle-inert /
//! non-permutation reds) moved with it. This file now covers only door E-down, which stays.
//!
//! THE BAR for door E-down: it is a pure visit-ORDER change on the down launch: every output is
//! a function of its `(o, tok)` coordinate and no block communicates, so re-indexing which block
//! computes which output must move ZERO bits, while the slot-ordered accumulation INSIDE the
//! block (which the transposition never touches) stays exactly the shipped chain.
//!
//! Everything here is exactness or counters. The WIN is a scheduling property and the rig is
//! exactness-only (rig law), so this door ships default OFF and the box prices the wall.
//!
//! Every OFF arm PINS its flag `=0` and never leaves it unset: doors T/X/K/W are default ON at
//! this base, and the moe-loc lane found two VACUOUS GREENS the moment "unset" stopped meaning
//! "off" (its §4.5). The same rule is applied here to this lane's own door.

use memra_engine::Engine;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|p| p.into_inner())
}

fn force_true_f32() {
    unsafe {
        std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
    }
}

/// Run `f` with `keys` pinned to the given values, restoring the prior values afterwards.
/// PINNING, not unsetting: an arm that leaves a flag unset is only an OFF arm while the flag's
/// default stays OFF, and this family has already flipped four doors to default ON.
fn with_flags<T>(keys: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let prior: Vec<(String, Option<String>)> = keys
        .iter()
        .map(|(k, _)| ((*k).to_string(), std::env::var(k).ok()))
        .collect();
    for (k, v) in keys {
        unsafe { std::env::set_var(k, v) };
    }
    let out = f();
    for (k, v) in &prior {
        match v {
            Some(v) => unsafe { std::env::set_var(k, v) },
            None => unsafe { std::env::remove_var(k) },
        }
    }
    out
}

/// The shipped schedule: door E-down pinned `=0`.
const SHIPPED: &[(&str, &str)] = &[("MEMRA_MOE_VROWS_DOWN_TMAJ", "0")];

/// The down dedup schedule armed.
const DOWN_TMAJ: &[(&str, &str)] = &[("MEMRA_MOE_VROWS_DOWN_TMAJ", "1")];

fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * spread
        })
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "compared buffers differ in length");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn nvfp4_slab(
    e: &Engine,
    n_expert: usize,
    out_f: usize,
    in_f: usize,
    seed: u64,
) -> (cudarc::driver::CudaSlice<u8>, usize, usize) {
    let mut bytes = Vec::new();
    let mut row_bytes = 0usize;
    for ex in 0..n_expert {
        for o in 0..out_f {
            let row = varied(in_f, seed ^ ((ex * out_f + o) as u64) << 8, 2.0);
            let rb = memra_gguf::nvfp4_repack::f32_to_nvfp4(&row);
            row_bytes = rb.len();
            bytes.extend_from_slice(&rb);
        }
    }
    let stride = out_f * row_bytes;
    let slab = e.htod_bytes(&bytes).expect("nvfp4 slab upload");
    (slab, row_bytes, stride)
}

/// A selection with the properties the dedup schedule must survive: REPEATED experts across rows
/// (the whole point of the lane), expert 0 (so `base + 0*stride` is exercised), the top expert id,
/// and full-row collisions at some `t` (two verify rows routing identically).
fn planted_sel(t: usize, n_used: usize, n_expert: usize) -> Vec<u32> {
    (0..t * n_used)
        .map(|p| {
            let (tok, j) = (p / n_used, p % n_used);
            // Rows 0 and 1 share every expert; the rest overlap partially.
            let row = if tok == 1 { 0 } else { tok };
            ((row * 3 + j * 5) % n_expert) as u32
        })
        .collect()
}

/// The HOST table build, verbatim from `moe_vrows_pairs_q8`'s host arm (three planes; door E's
/// fourth plane went with door E).
fn host_tables(
    (base_g, base_u, base_d): (u64, u64, u64),
    (sg, su, sd): (usize, usize, usize),
    macros: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    sel: &[u32],
    w: &[f32],
) -> (Vec<u64>, Vec<f32>) {
    let n_pairs = sel.len();
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for (p, (&ex, &wj)) in sel.iter().zip(w).enumerate() {
        let ex = ex as usize;
        ptrs[p] = base_g + (ex * sg) as u64;
        ptrs[n_pairs + p] = base_u + (ex * su) as u64;
        ptrs[2 * n_pairs + p] = base_d + (ex * sd) as u64;
        let (mg, mu, md) = match macros {
            Some(m) => (m.0[ex], m.1[ex], m.2[ex]),
            None => (1.0, 1.0, 1.0),
        };
        scl[p] = mg;
        scl[n_pairs + p] = mu;
        scl[2 * n_pairs + p] = wj * md;
    }
    (ptrs, scl)
}

fn macro_planes(n_expert: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        (0..n_expert)
            .map(|i| 0.5 + 0.07 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 1.6 - 0.05 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 0.8 + 0.04 * i as f32)
            .collect::<Vec<_>>(),
    )
}

/// Shape class: the vrest gate-4 shape (NVFP4 block 64, live macro plane, biting PRE clamp).
struct PairFixture {
    in_f: usize,
    n_ff: usize,
    n_expert: usize,
    n_used: usize,
    limit: f32,
    rb_gu: usize,
    stride_gu: usize,
    rb_dn: usize,
    stride_dn: usize,
    base_gu: u64,
    base_dn: u64,
    macros: (Vec<f32>, Vec<f32>, Vec<f32>),
    // Kept alive: the pointer tables address these slabs.
    _slab_gu: cudarc::driver::CudaSlice<u8>,
    _slab_dn: cudarc::driver::CudaSlice<u8>,
}

fn fixture(e: &Engine) -> PairFixture {
    let (in_f, n_ff) = (128usize, 64usize);
    let (n_expert, n_used) = (16usize, 8usize);
    let gu = nvfp4_slab(e, n_expert, n_ff, in_f, 0x6A7E);
    let dn = nvfp4_slab(e, n_expert, in_f, n_ff, 0xD003);
    // The address guards must drop before the slabs move into the fixture.
    let (base_gu, base_dn) = {
        use cudarc::driver::DevicePtr;
        let stream = e.stream();
        let (g, _g0) = gu.0.device_ptr(&stream);
        let (d, _g1) = dn.0.device_ptr(&stream);
        (g, d)
    };
    PairFixture {
        in_f,
        n_ff,
        n_expert,
        n_used,
        limit: 0.75,
        rb_gu: gu.1,
        stride_gu: gu.2,
        rb_dn: dn.1,
        stride_dn: dn.2,
        base_gu,
        base_dn,
        macros: macro_planes(n_expert),
        _slab_gu: gu.0,
        _slab_dn: dn.0,
    }
}

/// The full pair (gate/up -> pair quantize -> down/FMA), read back.
fn run_pair(
    e: &Engine,
    fx: &PairFixture,
    ptrs: &cudarc::driver::CudaSlice<u64>,
    scl: &cudarc::driver::CudaSlice<f32>,
    z_d: &cudarc::driver::CudaSlice<f32>,
    t: usize,
    n_pairs: usize,
) -> Vec<f32> {
    let (zq, zd) = e.quantize_q8_1(z_d, t, fx.in_f).expect("token quantize");
    let act = e
        .moe_gate_up_preclamp8_q8_rows(
            ptrs,
            scl,
            &zq,
            &zd,
            fx.limit,
            fx.in_f,
            fx.n_ff,
            fx.n_used,
            n_pairs,
            memra_engine::QT_NVFP4,
            memra_engine::QT_NVFP4,
            fx.rb_gu,
            fx.rb_gu,
        )
        .expect("gate/up rows launch");
    let (aq2, ad2) = e
        .quantize_q8_1(&act, n_pairs, fx.n_ff)
        .expect("pair act quantize");
    let mut out = e.uninit(t * fx.in_f).expect("out");
    e.moe_down8_fma_q8_rows(
        ptrs,
        scl,
        &aq2,
        &ad2,
        &mut out,
        fx.n_ff,
        fx.in_f,
        fx.n_used,
        n_pairs,
        memra_engine::QT_NVFP4,
        fx.rb_dn,
    )
    .expect("down rows launch");
    e.dtoh(&out).expect("pair readback")
}

// -------------------------------------------------------------------------------------------
// Gate 1: the token-major down schedule, bitwise identity against the shipped pair.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_token_major_down_is_bit_identical_and_keeps_its_slot_order() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let fx = fixture(&e);
    let mut arms = 0usize;

    for t in 2..=8usize {
        let n_pairs = t * fx.n_used;
        let sel = planted_sel(t, fx.n_used, fx.n_expert);
        // Routing weights spanning several magnitudes: the slot-ordered __fmaf_rn chain is only
        // order-SENSITIVE when the addends differ in scale, so a flat weight vector would let a
        // reordered accumulation pass. These make the chain's order observable.
        let w: Vec<f32> = (0..n_pairs)
            .map(|p| (1.0 + p as f32) * 10f32.powi(((p % 5) as i32) - 2))
            .collect();
        let z = varied(t * fx.in_f, 0x5150 + t as u64, 2.0);
        let z_d = e.htod(&z).expect("z upload");

        for (mlabel, mac) in [("live macros", true), ("no macros", false)] {
            let macs = mac.then_some(&fx.macros);
            let (hp, hs) = host_tables(
                (fx.base_gu, fx.base_gu, fx.base_dn),
                (fx.stride_gu, fx.stride_gu, fx.stride_dn),
                macs,
                &sel,
                &w,
            );
            let want = with_flags(SHIPPED, || {
                run_pair(
                    &e,
                    &fx,
                    &e.htod_u64(&hp).expect("host ptrs"),
                    &e.htod(&hs).expect("host scl"),
                    &z_d,
                    t,
                    n_pairs,
                )
            });

            // Down-only arm: the token-major grid with the SHIPPED gate/up schedule, so any
            // divergence is attributable to the down transposition alone.
            let got_down = with_flags(DOWN_TMAJ, || {
                run_pair(
                    &e,
                    &fx,
                    &e.htod_u64(&hp).expect("host ptrs"),
                    &e.htod(&hs).expect("host scl"),
                    &z_d,
                    t,
                    n_pairs,
                )
            });
            assert_eq!(
                bit_diffs(&got_down, &want),
                0,
                "t={t} {mlabel}: the token-major down schedule moved bits, the slot-ordered \
                 FMA chain did not survive the grid transposition"
            );
            arms += 1;
        }
    }
    println!(
        "gate 1 PASS: token-major down bit-identical in {arms} arms (t=2..8 x {{live macros, none}})"
    );
}

// -------------------------------------------------------------------------------------------
// Gate 2: engagement counters, and the refusal by name.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_down_tmaj_counters_move_on_and_are_flat_on_every_refusal() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let fx = fixture(&e);
    let (t, n_used) = (3usize, 8usize);
    let n_pairs = t * n_used;
    let sel = planted_sel(t, n_used, fx.n_expert);
    let w: Vec<f32> = (0..n_pairs).map(|p| 0.2 + 0.05 * (p % 7) as f32).collect();
    let z = varied(t * fx.in_f, 0xC0DE, 2.0);
    let z_d = e.htod(&z).expect("z upload");

    let (hp, hs) = host_tables(
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        Some(&fx.macros),
        &sel,
        &w,
    );
    let scl_d = e.htod(&hs).expect("scl");
    let ptrs_d = e.htod_u64(&hp).expect("ptrs");

    // Deltas are taken around each arm; an absolute value would pass on a leftover count.
    let probe = |flags: &[(&str, &str)]| -> u64 {
        let d0 = memra_engine::moe_vrows_down_tmaj_dispatches();
        with_flags(flags, || {
            run_pair(&e, &fx, &ptrs_d, &scl_d, &z_d, t, n_pairs)
        });
        memra_engine::moe_vrows_down_tmaj_dispatches() - d0
    };

    let d_on = probe(DOWN_TMAJ);
    assert!(
        d_on > 0,
        "door armed: the counter did not move (down {d_on})"
    );

    let d_off = probe(SHIPPED);
    assert_eq!(
        d_off, 0,
        "the door pinned =0 still dispatched (down {d_off})"
    );

    println!("gate 2 PASS: counter moves ON (down {d_on}), flat with the door pinned =0");
}
