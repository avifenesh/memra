//! glm5 EXPERT-SLAB DEDUP schedule doors (lane/glm5-dedup, 2026-08-31) — rig exactness gates.
//!
//! WHY THIS LANE EXISTS, in one measured number: the struct-battery instrument booted
//! `MEMRA_MOE_VROWS_DEDUP_STAT=1` on the real artifact and the ship recipe and read a **21.96%
//! cumulative repeat fraction** over 99,751 vrows layer-calls / 2.55M expert visits — mode-stable
//! (22.27% greedy / 21.53% vendor-default sampled), i.e. **6.9x the 3.21% independent-routing
//! bound**. About a fifth of the pair's (verify row, expert) visits re-read a slab a sibling row
//! of the SAME layer-call already read. The pair is at 90.2% / 89.9% of this card class's
//! theoretical DRAM peak (moe-loc LANE.md §1.3), so there is no efficiency to win: the only lever
//! is READING LESS, and a repeat read is avoided only if it is SCHEDULED inside the reuse window.
//!
//! Door E (`MEMRA_MOE_VROWS_DEDUP_ORDER`): the gate/up launch takes `_ord` — grid transposed so
//! the pair index is the FASTEST dimension, walked in EXPERT-MAJOR order from a fourth plane
//! appended to the pointer table.
//! Door E-down (`MEMRA_MOE_VROWS_DOWN_TMAJ`): the down launch takes `_tmaj` — grid transposed to
//! `(t, out_f)`, token fastest. The slot-ordered `__fmaf_rn` chain lives INSIDE the block and
//! keeps its original slot order; only the grid moves.
//!
//! THE BAR. Both doors are pure visit-ORDER changes: every output is a function of its `(o, pr)` /
//! `(o, tok)` coordinate and no block communicates, so re-indexing which block computes which
//! output must move ZERO bits. This suite proves that three ways — bitwise identity against the
//! shipped pair across `t = 2..=8` x {live macros, none} x {host tables, device tables}, a
//! VALID-SHUFFLE arm that must stay bit-INERT (the arithmetic claim), and a NON-PERMUTATION order
//! plane that must BITE (proving the plane is live and every pair is computed exactly once).
//!
//! Everything here is exactness or counters. The WIN is a scheduling property and the rig is
//! exactness-only (rig law), so both doors ship default OFF and the box prices the wall.
//!
//! Every OFF arm PINS its flag `=0` and never leaves it unset: doors T/X/K/W are default ON at
//! this base, and the moe-loc lane found two VACUOUS GREENS the moment "unset" stopped meaning
//! "off" (its §4.5). The same rule is applied here to door M and to this lane's own two doors.

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

/// The shipped schedule: every door of this lane pinned `=0`, plus the refuted pack door.
const SHIPPED: &[(&str, &str)] = &[
    ("MEMRA_MOE_VROWS_DEDUP_ORDER", "0"),
    ("MEMRA_MOE_VROWS_DOWN_TMAJ", "0"),
    ("MEMRA_MOE_VROWS_PACK", "0"),
];

/// Both dedup schedules armed, pack pinned off.
const DEDUP: &[(&str, &str)] = &[
    ("MEMRA_MOE_VROWS_DEDUP_ORDER", "1"),
    ("MEMRA_MOE_VROWS_DOWN_TMAJ", "1"),
    ("MEMRA_MOE_VROWS_PACK", "0"),
];

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

/// The HOST table build, verbatim from `moe_vrows_pairs_q8`'s host arm, with door E's fourth
/// (expert-major order) plane appended when `order` is set.
fn host_tables(
    (base_g, base_u, base_d): (u64, u64, u64),
    (sg, su, sd): (usize, usize, usize),
    macros: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    sel: &[u32],
    w: &[f32],
    order: bool,
) -> (Vec<u64>, Vec<f32>) {
    let n_pairs = sel.len();
    let planes = if order { 4 } else { 3 };
    let mut ptrs = vec![0u64; planes * n_pairs];
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
    if order {
        ptrs[3 * n_pairs..].copy_from_slice(&memra_engine::vrows_expert_major_order_for_test(sel));
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

// -------------------------------------------------------------------------------------------
// Gate 1 — the expert-major order plane: device build vs the host stable sort.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_order_plane_device_build_matches_the_host_stable_sort() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (n_expert, n_used) = (16usize, 8usize);

    for t in 2..=8usize {
        let n_pairs = t * n_used;
        let sel = planted_sel(t, n_used, n_expert);
        let sel_d = e
            .htod_i32(&sel.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .expect("sel upload");
        let want = memra_engine::vrows_expert_major_order_for_test(&sel);

        let mut ptrs_d = e.htod_u64(&vec![u64::MAX; 4 * n_pairs]).expect("ptrs");
        e.moe_vrows_order_from_sel(&sel_d, n_pairs, &mut ptrs_d)
            .expect("device order build");
        let got_all = e.dtoh_u64(&ptrs_d).expect("ptrs readback");
        let got = &got_all[3 * n_pairs..];

        assert_eq!(
            got,
            &want[..],
            "t={t}: the device order plane diverged from the host stable sort"
        );

        // It is a PERMUTATION — every pair computed exactly once. Without this the identity
        // gates could pass for a plane that dropped a pair and duplicated another whose output
        // happened to match.
        let mut sorted = got.to_vec();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..n_pairs as u64).collect::<Vec<_>>(),
            "t={t}: the order plane is not a permutation of 0..n_pairs"
        );

        // It is EXPERT-MAJOR, and STABLE inside each expert's run (so per-token slot order
        // survives for a shared expert).
        for i in 1..n_pairs {
            let (a, b) = (got[i - 1] as usize, got[i] as usize);
            assert!(
                sel[a] < sel[b] || (sel[a] == sel[b] && a < b),
                "t={t}: order plane is not a stable expert-major order at position {i} \
                 (pair {a} expert {} then pair {b} expert {})",
                sel[a],
                sel[b]
            );
        }

        // The number of RUNS in the plane is exactly `distinct`, which is what makes
        // `visits - distinct` (the MOE_VROWS_SLAB_READS_AVOIDED receipt) the count of repeat
        // visits this schedule places inside the reuse window.
        let runs = 1
            + (1..n_pairs)
                .filter(|&i| sel[got[i] as usize] != sel[got[i - 1] as usize])
                .count();
        let (visits, distinct) = memra_engine::vrows_overlap_counts_for_test(&sel);
        assert_eq!(
            runs as u64, distinct,
            "t={t}: expert-major runs ({runs}) != distinct experts ({distinct})"
        );
        println!(
            "order plane PASS t={t}: {n_pairs} pairs, {distinct} distinct, \
             {} repeat visits made adjacent",
            visits - distinct
        );

        // RED — a DIFFERENT selection must produce a different plane, else `sel` is unread.
        let mut alt = sel.clone();
        alt[0] = ((alt[0] as usize + 7) % n_expert) as u32;
        let alt_d = e
            .htod_i32(&alt.iter().map(|&x| x as i32).collect::<Vec<_>>())
            .expect("alt sel upload");
        let mut ptrs_alt = e.htod_u64(&vec![u64::MAX; 4 * n_pairs]).expect("ptrs alt");
        e.moe_vrows_order_from_sel(&alt_d, n_pairs, &mut ptrs_alt)
            .expect("device order build (red)");
        let alt_all = e.dtoh_u64(&ptrs_alt).expect("readback");
        assert_ne!(
            &alt_all[3 * n_pairs..],
            got,
            "t={t}: the order plane did not move when the selection did — `sel` is unread"
        );
    }
    println!(
        "gate 1 PASS: device order plane == host stable sort, is a permutation, is stable \
              expert-major, run count == distinct; the changed-selection RED bites"
    );
}

// -------------------------------------------------------------------------------------------
// Gates 2-4 — bitwise identity of the two schedules against the shipped pair.
// -------------------------------------------------------------------------------------------

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

/// Build the device-provenance tables (door D's kernel), with door E's order plane when asked.
fn device_tables(
    e: &Engine,
    fx: &PairFixture,
    sel: &[u32],
    w: &[f32],
    macros: bool,
    order: bool,
) -> (
    cudarc::driver::CudaSlice<u64>,
    cudarc::driver::CudaSlice<f32>,
) {
    let n_pairs = sel.len();
    let planes = if order { 4 } else { 3 };
    let sel_d = e
        .htod_i32(&sel.iter().map(|&x| x as i32).collect::<Vec<_>>())
        .expect("sel upload");
    let w_d = e.htod(w).expect("w upload");
    let mut ptrs = e
        .htod_u64(&vec![0u64; planes * n_pairs])
        .expect("ptrs alloc");
    let mut scl = e.htod(&vec![0f32; 3 * n_pairs]).expect("scl alloc");
    e.moe_vrows_tables_from_sel(
        &sel_d,
        &w_d,
        3,
        macros.then_some((
            fx.macros.0.as_slice(),
            fx.macros.1.as_slice(),
            fx.macros.2.as_slice(),
        )),
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        n_pairs,
        &mut ptrs,
        &mut scl,
    )
    .expect("device table build");
    if order {
        e.moe_vrows_order_from_sel(&sel_d, n_pairs, &mut ptrs)
            .expect("device order build");
    }
    (ptrs, scl)
}

/// The gate/up rows launch, read back. The SCHEDULE is chosen by the ambient flags.
fn run_gate_up(
    e: &Engine,
    fx: &PairFixture,
    ptrs: &cudarc::driver::CudaSlice<u64>,
    scl: &cudarc::driver::CudaSlice<f32>,
    z_d: &cudarc::driver::CudaSlice<f32>,
    t: usize,
    n_pairs: usize,
) -> Vec<f32> {
    let (zq, zd) = e
        .quantize_q8_1(z_d, t, fx.in_f)
        .expect("batched token quantize");
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
    e.dtoh(&act).expect("act readback")
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

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_expert_major_gate_up_is_bit_identical_to_the_shipped_schedule() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let fx = fixture(&e);
    let mut arms = 0usize;

    for t in 2..=8usize {
        let n_pairs = t * fx.n_used;
        let sel = planted_sel(t, fx.n_used, fx.n_expert);
        let w: Vec<f32> = (0..n_pairs).map(|p| 0.1 + 0.03 * (p % 11) as f32).collect();
        let z = varied(t * fx.in_f, 0x2A + t as u64, 2.0);
        let z_d = e.htod(&z).expect("z upload");

        for (mlabel, mac) in [("live macros", true), ("no macros", false)] {
            let macs = mac.then_some(&fx.macros);
            // Reference: the SHIPPED schedule with a 3-plane table, this lane's doors pinned =0.
            let (hp, hs) = host_tables(
                (fx.base_gu, fx.base_gu, fx.base_dn),
                (fx.stride_gu, fx.stride_gu, fx.stride_dn),
                macs,
                &sel,
                &w,
                false,
            );
            let want = with_flags(SHIPPED, || {
                run_gate_up(
                    &e,
                    &fx,
                    &e.htod_u64(&hp).expect("host ptrs"),
                    &e.htod(&hs).expect("host scl"),
                    &z_d,
                    t,
                    n_pairs,
                )
            });

            // Arm A: door E with HOST-built tables and a HOST-built order plane.
            let (op, os) = host_tables(
                (fx.base_gu, fx.base_gu, fx.base_dn),
                (fx.stride_gu, fx.stride_gu, fx.stride_dn),
                macs,
                &sel,
                &w,
                true,
            );
            let got_host = with_flags(DEDUP, || {
                run_gate_up(
                    &e,
                    &fx,
                    &e.htod_u64(&op).expect("host ptrs+order"),
                    &e.htod(&os).expect("host scl"),
                    &z_d,
                    t,
                    n_pairs,
                )
            });
            assert_eq!(
                bit_diffs(&got_host, &want),
                0,
                "t={t} {mlabel} host tables: the expert-major gate/up schedule moved bits"
            );

            // Arm B: door E composed with door D — tables AND order plane built on device.
            let (dp, ds) = device_tables(&e, &fx, &sel, &w, mac, true);
            let got_dev = with_flags(DEDUP, || run_gate_up(&e, &fx, &dp, &ds, &z_d, t, n_pairs));
            assert_eq!(
                bit_diffs(&got_dev, &want),
                0,
                "t={t} {mlabel} device tables: the expert-major gate/up schedule moved bits"
            );
            arms += 2;
        }
    }
    println!(
        "gate 2 PASS: expert-major gate/up bit-identical in {arms} arms \
         (t=2..8 x {{live macros, none}} x {{host tables, device tables}})"
    );
}

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
                false,
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
            let got_down = with_flags(
                &[
                    ("MEMRA_MOE_VROWS_DEDUP_ORDER", "0"),
                    ("MEMRA_MOE_VROWS_DOWN_TMAJ", "1"),
                    ("MEMRA_MOE_VROWS_PACK", "0"),
                ],
                || {
                    run_pair(
                        &e,
                        &fx,
                        &e.htod_u64(&hp).expect("host ptrs"),
                        &e.htod(&hs).expect("host scl"),
                        &z_d,
                        t,
                        n_pairs,
                    )
                },
            );
            assert_eq!(
                bit_diffs(&got_down, &want),
                0,
                "t={t} {mlabel}: the token-major down schedule moved bits — the slot-ordered \
                 FMA chain did not survive the grid transposition"
            );

            // Both doors, both table provenances — the composed serving shape.
            let (op, os) = host_tables(
                (fx.base_gu, fx.base_gu, fx.base_dn),
                (fx.stride_gu, fx.stride_gu, fx.stride_dn),
                macs,
                &sel,
                &w,
                true,
            );
            let got_both = with_flags(DEDUP, || {
                run_pair(
                    &e,
                    &fx,
                    &e.htod_u64(&op).expect("host ptrs+order"),
                    &e.htod(&os).expect("host scl"),
                    &z_d,
                    t,
                    n_pairs,
                )
            });
            assert_eq!(
                bit_diffs(&got_both, &want),
                0,
                "t={t} {mlabel} host tables: the composed dedup schedule moved bits"
            );

            let (dp, ds) = device_tables(&e, &fx, &sel, &w, mac, true);
            let got_dev = with_flags(DEDUP, || run_pair(&e, &fx, &dp, &ds, &z_d, t, n_pairs));
            assert_eq!(
                bit_diffs(&got_dev, &want),
                0,
                "t={t} {mlabel} device tables: the composed dedup schedule moved bits"
            );
            arms += 3;
        }
    }
    println!(
        "gate 3 PASS: token-major down and the composed pair bit-identical in {arms} arms \
         (down-only, composed host tables, composed device tables)"
    );
}

// -------------------------------------------------------------------------------------------
// Gate 4 — the reds. A valid shuffle must be INERT; a non-permutation must BITE.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_order_plane_shuffles_are_inert_and_non_permutations_bite() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let fx = fixture(&e);
    let (t, n_used) = (4usize, 8usize);
    let n_pairs = t * n_used;
    let sel = planted_sel(t, n_used, fx.n_expert);
    let w: Vec<f32> = (0..n_pairs).map(|p| 0.1 + 0.03 * (p % 11) as f32).collect();
    let z = varied(t * fx.in_f, 0x9E37, 2.0);
    let z_d = e.htod(&z).expect("z upload");

    let (hp3, hs) = host_tables(
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        Some(&fx.macros),
        &sel,
        &w,
        false,
    );
    let want = with_flags(SHIPPED, || {
        run_pair(
            &e,
            &fx,
            &e.htod_u64(&hp3).expect("host ptrs"),
            &e.htod(&hs).expect("host scl"),
            &z_d,
            t,
            n_pairs,
        )
    });
    let scl_d = e.htod(&hs).expect("host scl");

    // A — VALID SHUFFLES ARE BIT-INERT. This is the arithmetic claim of the whole lane: the visit
    // order is a schedule, not an accumulation order, so ANY permutation must return the same
    // bytes. Three seeds, all bitwise.
    for seed in [0xA5A5u64, 0x1234, 0xFEED] {
        let mut ord: Vec<u64> = (0..n_pairs as u64).collect();
        let mut s = seed | 1;
        for i in (1..n_pairs).rev() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ord.swap(i, (s >> 33) as usize % (i + 1));
        }
        let mut ptrs = hp3.clone();
        ptrs.extend_from_slice(&ord);
        let got = with_flags(DEDUP, || {
            run_pair(
                &e,
                &fx,
                &e.htod_u64(&ptrs).expect("shuffled ptrs"),
                &scl_d,
                &z_d,
                t,
                n_pairs,
            )
        });
        assert_eq!(
            bit_diffs(&got, &want),
            0,
            "shuffle seed {seed:#x}: a VALID permutation of the visit order moved bits — the \
             schedule is entangled with the arithmetic"
        );
    }
    println!("gate 4A PASS: 3 valid shuffles of the visit order are bit-INERT");

    // B — a NON-PERMUTATION must BITE. This is the ONLY arm that proves the order plane is read
    // at all: 4A cannot, because a kernel that ignored the plane would compute `pr = blockIdx.x`
    // — itself a valid permutation — and every valid permutation is inert. A non-permutation
    // leaves a pair uncomputed, so an `act` row is never written.
    //
    // ORDERING IS LOAD-BEARING HERE, and its first pass was a FALSE NEGATIVE (found and fixed
    // in-lane): `act` is an uninit draw, and comparing the red against a reference computed
    // EARLIER FROM THE SAME `z` let the async allocator hand the red back the very block the
    // reference run had just freed — the dropped row read as exactly correct and the red did not
    // bite. So the red now runs on a `z` no earlier arm in this test has used, and it runs BEFORE
    // its own reference: no freed block can hold the correct row for the dropped pair, and a
    // fresh (zeroed) page cannot match a live-clamp epilogue's nonzero row either. The SECOND red
    // arm needs its OWN fresh `z` for the same reason — the first arm's reference run frees a
    // correct block of exactly the right size, and re-using one `z` across arms reproduced the
    // false negative one level deeper.
    for (label, seed, bad) in [
        ("duplicated entry (one pair dropped)", 0x0BADC0DEu64, {
            let mut b: Vec<u64> = (0..n_pairs as u64).collect();
            b[1] = b[0];
            b
        }),
        (
            "degenerate all-zero plane (n_pairs-1 dropped)",
            0x0DEFACED,
            vec![0u64; n_pairs],
        ),
    ] {
        let z_red = varied(t * fx.in_f, seed, 2.0);
        let z_red_d = e.htod(&z_red).expect("red z upload");
        let mut ptrs_bad = hp3.clone();
        ptrs_bad.extend_from_slice(&bad);
        let got_bad = with_flags(DEDUP, || {
            run_pair(
                &e,
                &fx,
                &e.htod_u64(&ptrs_bad).expect("bad ptrs"),
                &scl_d,
                &z_red_d,
                t,
                n_pairs,
            )
        });
        let want_red = with_flags(SHIPPED, || {
            run_pair(
                &e,
                &fx,
                &e.htod_u64(&hp3).expect("host ptrs"),
                &scl_d,
                &z_red_d,
                t,
                n_pairs,
            )
        });
        let d = bit_diffs(&got_bad, &want_red);
        assert!(
            d > 0,
            "RED {label}: the non-permutation order plane left the output identical — the order \
             plane is not being read, so gate 2/3's identity is vacuous"
        );
        println!("gate 4B PASS: RED {label} bites, {d} outputs differ");
    }

    // C — the identity is not vacuous about the SCALES either: dropping the macro planes must
    // still move the output THROUGH the dedup schedule (the vrest/moe-loc dropped-macro red,
    // re-bitten on the new kernels).
    let (hp_nm, hs_nm) = host_tables(
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        None,
        &sel,
        &w,
        true,
    );
    let got_nm = with_flags(DEDUP, || {
        run_pair(
            &e,
            &fx,
            &e.htod_u64(&hp_nm).expect("no-macro ptrs"),
            &e.htod(&hs_nm).expect("no-macro scl"),
            &z_d,
            t,
            n_pairs,
        )
    });
    let dn = bit_diffs(&got_nm, &want);
    assert!(
        dn > 0,
        "the dropped-macro RED did not bite through the dedup schedule — the scale planes are \
         not being consumed"
    );
    println!("gate 4C PASS: the dropped-macro RED bites through the dedup schedule, {dn} differ");
}

// -------------------------------------------------------------------------------------------
// Gate 5 — engagement counters, and the two refusals by name.
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_dedup_schedule_counters_move_on_and_are_flat_on_every_refusal() {
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

    let (hp3, hs) = host_tables(
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        Some(&fx.macros),
        &sel,
        &w,
        false,
    );
    let (hp4, _) = host_tables(
        (fx.base_gu, fx.base_gu, fx.base_dn),
        (fx.stride_gu, fx.stride_gu, fx.stride_dn),
        Some(&fx.macros),
        &sel,
        &w,
        true,
    );
    let scl_d = e.htod(&hs).expect("scl");
    let p3 = e.htod_u64(&hp3).expect("ptrs 3-plane");
    let p4 = e.htod_u64(&hp4).expect("ptrs 4-plane");

    // Deltas are taken around each arm; an absolute value would pass on a leftover count.
    let probe = |flags: &[(&str, &str)], ptrs: &cudarc::driver::CudaSlice<u64>| -> (u64, u64) {
        let (o0, d0) = (
            memra_engine::moe_vrows_dedup_order_dispatches(),
            memra_engine::moe_vrows_down_tmaj_dispatches(),
        );
        with_flags(flags, || run_pair(&e, &fx, ptrs, &scl_d, &z_d, t, n_pairs));
        (
            memra_engine::moe_vrows_dedup_order_dispatches() - o0,
            memra_engine::moe_vrows_down_tmaj_dispatches() - d0,
        )
    };

    let (o_on, d_on) = probe(DEDUP, &p4);
    assert!(
        o_on > 0 && d_on > 0,
        "both doors armed with a 4-plane table: counters did not move (order {o_on}, down {d_on})"
    );

    let (o_off, d_off) = probe(SHIPPED, &p3);
    assert_eq!(
        (o_off, d_off),
        (0, 0),
        "the doors pinned =0 still dispatched (order {o_off}, down {d_off})"
    );

    // REFUSAL 1 — the flag is on but the caller built no order plane. Door E must fall closed to
    // the shipped schedule rather than read past the table (the standing-gate call sites all look
    // exactly like this).
    let (o_noplane, d_noplane) = probe(DEDUP, &p3);
    assert_eq!(
        o_noplane, 0,
        "door E engaged on a 3-plane table — it read past the pointer table"
    );
    assert!(
        d_noplane > 0,
        "the down door needs no table plane and must still engage on a 3-plane table"
    );

    // REFUSAL 2 — door M (MEMRA_MOE_VROWS_PACK, refuted at 0.9959x) takes precedence, so the two
    // schedule families are never crossed. Named in FLAGS.md for both doors.
    let (o_pack, d_pack) = probe(
        &[
            ("MEMRA_MOE_VROWS_DEDUP_ORDER", "1"),
            ("MEMRA_MOE_VROWS_DOWN_TMAJ", "1"),
            ("MEMRA_MOE_VROWS_PACK", "1"),
        ],
        &p4,
    );
    assert_eq!(
        (o_pack, d_pack),
        (0, 0),
        "door M did not refuse the dedup schedules (order {o_pack}, down {d_pack})"
    );

    println!(
        "gate 5 PASS: counters move ON (order {o_on}, down {d_on}), flat with the doors pinned \
         =0, flat for door E on a 3-plane table, and flat for BOTH under door M"
    );
}

// -------------------------------------------------------------------------------------------
// Gate 6 — the avoided-slab-read receipt's arithmetic, on planted overlaps (CPU).
// -------------------------------------------------------------------------------------------

#[test]
fn avoided_slab_reads_equal_visits_minus_distinct_on_planted_overlaps() {
    // MOE_VROWS_SLAB_READS_AVOIDED is fed `visits - distinct` per layer-call, and the box turns it
    // into bytes (x 9.4372 MB gate+up, x 4.7186 MB down at the serving geometry). The counting is
    // the whole receipt, so it is gated on plants — never inferred from a live tape.
    for (label, sel, want_visits, want_distinct) in [
        ("disjoint rows", vec![0u32, 1, 2, 3, 4, 5, 6, 7], 8u64, 8u64),
        (
            "two rows sharing 3 experts",
            vec![0u32, 1, 2, 3, 0, 1, 2, 9],
            8,
            5,
        ),
        ("identically routed rows", vec![5u32, 6, 5, 6, 5, 6], 6, 2),
        ("one expert everywhere", vec![3u32; 12], 12, 1),
    ] {
        let (visits, distinct) = memra_engine::vrows_overlap_counts_for_test(&sel);
        assert_eq!((visits, distinct), (want_visits, want_distinct), "{label}");

        // The order plane's run count is what makes the difference the AVOIDABLE count: after the
        // expert-major sort, exactly `distinct` runs remain and the other `visits - distinct`
        // visits sit next to a block that already read their slab.
        let ord = memra_engine::vrows_expert_major_order_for_test(&sel);
        let runs = 1
            + (1..ord.len())
                .filter(|&i| sel[ord[i] as usize] != sel[ord[i - 1] as usize])
                .count();
        assert_eq!(runs as u64, distinct, "{label}: runs != distinct");
        println!(
            "{label}: visits={visits} distinct={distinct} avoided={} ({:.2}% repeat)",
            visits - distinct,
            100.0 * (1.0 - distinct as f64 / visits as f64)
        );
    }
}
