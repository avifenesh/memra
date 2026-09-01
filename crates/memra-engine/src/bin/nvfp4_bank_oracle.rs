//! NVFP4 BANK ORACLE — is the slot-major bank layout really a pure permutation, ON DEVICE,
//! per kernel arm? (2026-09-01, research/step37-bankv3-20260901)
//!
//! The named follow-up of `75bf4ce76`: "a device-side bank oracle is the named follow-up lane".
//! The slot-major layout shipped with a bit-identity claim, a passing host-side layout test, and
//! corrupt serving output. It has now failed twice, and BOTH failures were invisible to every
//! instrument that existed (DIAGNOSIS.md):
//!
//!   * the host-side `bank_v2_layout_tests` pins the byte map and passes — a host test cannot
//!     see a reader;
//!   * the "host-canonical oracle" branched on the same flag as the path under test, so it
//!     compared v2 against v2;
//!   * no v2 gate ever ran the PREFILL grouped GEMM, which is where both defects lived.
//!
//! So this oracle is built to be immune to all three. It is a DIFFERENTIAL gate:
//!
//!   1. one pseudo-random block_nvfp4 bank, `bank_v1`, is the only source of values;
//!   2. `bank_v2 = tp::nvfp4_matrix_v2_permute(bank_v1)` — the shipping permutation, not a copy
//!      of its logic;
//!   3. the SAME activations and the SAME CSR go through the SAME entry point
//!      (`Engine::moe_f16_grouped`) twice, once as `(bank_v1, QT_NVFP4)` and once as
//!      `(bank_v2, QT_NVFP4_V2)`;
//!   4. the two f32 outputs must be **bit-identical**. A permutation that is a permutation, read
//!      by a reader that reads it, cannot move a bit: the value stream into the mma is the same
//!      values in the same k-order, so this is an exactness gate and not a tolerance band.
//!
//! The v1 arm is the oracle and it is pinned to v1 bytes and the v1 qtype. Nothing in the gate is
//! a function of the layout under test.
//!
//! WHY RANDOM BYTES ARE THE RIGHT INPUT HERE, and zeros are not: an uninitialised or zero bank
//! dequants to zeros, the GEMM computes 0 everywhere, and any two accumulation orders agree
//! bit-for-bit on a sum of zeros — the trap `moe_tp2_repro` documents against itself. Random
//! codes make every product distinct. The four UE4M3 scale bytes per superblock are drawn from a
//! narrow non-degenerate band instead of uniformly, for a reason that is the whole point of the
//! gate: the defect this oracle was written to catch substitutes a CODE byte for a SCALE byte, so
//! the two byte populations must be distinguishable in magnitude or a wrong-scale read could
//! land on a value that happens to be plausible and the arms would agree by luck.
//!
//! ARM COVERAGE. Which GEMM tile form runs is a function of CSR group size against
//! `MEMRA_F16G_SK_CROSS`, and it is chosen inside the C launcher, so it cannot be asserted from
//! here — it is DRIVEN instead, from the caller, one arm per process (the policy is a OnceLock):
//!
//!   MEMRA_F16G_SK=128  -> cross=1, every group on the 128x64x64 3-stage form
//!   MEMRA_F16G_SK=32   -> cross=INT_MAX, every group on the 32x64 tail form
//!   unset              -> the shipping hybrid, cross=64
//!   MEMRA_F16G_TAIL=0  -> the tail form's 2-stage rollback instead of the deep 3-stage tail
//!
//! Both geometries of a step37 layer are exercised in one run, because "it works on gate/up"
//! says nothing about down: gate/up is in_f=4096 out_f=640 (row_bytes 2304, 16B-aligned rows),
//! down is in_f=640 out_f=4096 (row_bytes 360, 8-aligned only). Both give nkb > 1, which is
//! required: the defect that shipped was in the kb+1 software-prefetch, so a single-k-block
//! geometry would pass while corrupt.
//!
//! DECODE COVERAGE (added 2026-09-01, milestone 4). The prefill gate above is necessary and not
//! sufficient: the −23.7% decode prize lives in the SELECTED-EXPERTS SWEEP family, which the
//! grouped GEMM never touches. `run_decode_cell` therefore gates each restored program against a
//! v1-BYTES oracle, one arm per program, so each flag carries its own device-side bit receipt:
//!
//!   P1 `MEMRA_NVFP4_BANK_SM`   `_sel_v2`(v2 bytes)      vs `_sel`(v1 bytes)
//!   P2 `MEMRA_NVFP4_SEL_GU`    `_sel_v2_gu`(v2 gate+up) vs TWO `_sel`(v1) sweeps
//!   P3 `MEMRA_NVFP4_SEL_DOWN8` `_sel_v2_down8`(v2)      vs `_sel`(v1) + `axpy_rows_seq_md`
//!
//! Every oracle arm is pinned to v1 bytes and the v1 kernel, so no arm is a function of the
//! layout or the fusion under test — the mistake the original "host-canonical oracle" made when
//! it branched on the flag and compared v2 against v2. P2 uses TWO DIFFERENT random banks for
//! gate and up: with one bank a kernel that read the gate rows for the up half would pass.
//!
//! usage: nvfp4-bank-oracle
//!   MEMRA_ORACLE_EXPERTS (default 32)  MEMRA_ORACLE_T (default 64)  n_pairs = T*8
//!   MEMRA_ORACLE_HIDDEN  (default 4096)  MEMRA_ORACLE_FF (default 640)
//! exit 0 = every arm bit-identical; exit 1 = a deviation (the byte gate FAILED).

use cudarc::driver::{CudaSlice, DevicePtr};
use memra_engine::Engine;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// xorshift64, so the bank is reproducible across arms, processes and boxes without a file.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 24) as u8
    }
}

/// One pseudo-random but VALID block_nvfp4 bank: `n_expert * out_f` rows of
/// `(in_f/64)*36` bytes, each 36-byte superblock = 4 UE4M3 scale bytes then 32 packed e2m1
/// code bytes.
///
/// Scales are drawn from 0x38..0x48 — UE4M3 exponents around 1.0, so every superblock
/// contributes at a comparable magnitude and no row is dominated by one huge scale (which would
/// mask a wrong scale elsewhere in the row). Codes are uniform. The two populations are
/// deliberately disjoint in value so that reading a code byte AS a scale is loud.
fn random_v1_bank(seed: u64, rows: usize, in_f: usize) -> Vec<u8> {
    let mut rng = Rng(seed);
    let n_sb = in_f / 64;
    let mut out = Vec::with_capacity(rows * n_sb * 36);
    for _ in 0..rows {
        for _ in 0..n_sb {
            for _ in 0..4 {
                out.push(0x38 + (rng.byte() & 0x0f));
            }
            for _ in 0..32 {
                out.push(rng.byte());
            }
        }
    }
    out
}

/// A step37-shaped CSR: `n_pairs` (token, slot) pairs routed over `n_expert` experts with the
/// same skewed tilt `moe_tp2_repro` uses, bucketed expert-major. Returned as
/// (ex_ids, ex_off, ex_pairs, csr_tok).
fn build_csr(
    n_pairs: usize,
    n_expert: usize,
    n_used: usize,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let route = |p: usize| -> usize {
        let mut h = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        ((u.powf(1.35) * n_expert as f64) as usize).min(n_expert - 1)
    };
    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
    for p in 0..n_pairs {
        buckets[route(p)].push(p as i32);
    }
    let (mut ex_ids, mut ex_off, mut ex_pairs) = (Vec::new(), vec![0i32], Vec::<i32>::new());
    for (id, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            ex_ids.push(id as i32);
            ex_pairs.extend_from_slice(b);
            ex_off.push(ex_pairs.len() as i32);
        }
    }
    let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
    (ex_ids, ex_off, ex_pairs, csr_tok)
}

/// Pointer table in the layout the grouped GEMM expects: [gate(n_expert), up(n_expert),
/// down(n_expert)] absolute device addresses. One bank here, so all three thirds alias it and
/// the caller uses proj = 0.
fn pointer_table(
    e: &Engine,
    bank: &CudaSlice<u8>,
    n_expert: usize,
    expert_bytes: usize,
) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
    let mut tab = vec![0u64; 3 * n_expert];
    {
        let s = e.stream();
        let (p, _g) = bank.device_ptr(&s);
        for ex in 0..n_expert {
            let a = p + (ex * expert_bytes) as u64;
            tab[ex] = a;
            tab[n_expert + ex] = a;
            tab[2 * n_expert + ex] = a;
        }
    }
    e.htod_u64(&tab)
}

struct Cell {
    name: &'static str,
    in_f: usize,
    out_f: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    e: &Engine,
    cell: &Cell,
    n_expert: usize,
    t: usize,
    n_used: usize,
    seed: u64,
) -> Result<bool, Box<dyn std::error::Error>> {
    let n_pairs = t * n_used;
    let row_bytes = cell.in_f / 64 * 36;
    let rows = n_expert * cell.out_f;
    let expert_bytes = cell.out_f * row_bytes;

    // (1) the only source of values.
    let v1 = random_v1_bank(seed, rows, cell.in_f);
    assert_eq!(v1.len(), rows * row_bytes);
    // (2) the SHIPPING permutation, applied per expert matrix exactly as the bank builder does.
    let mut v2 = Vec::with_capacity(v1.len());
    for ex in 0..n_expert {
        let m = &v1[ex * expert_bytes..(ex + 1) * expert_bytes];
        v2.extend_from_slice(&memra_engine::tp::nvfp4_matrix_v2_permute(
            m, cell.out_f, cell.in_f,
        ));
    }
    assert_eq!(v2.len(), v1.len(), "permutation changed the byte count");
    // A permutation must not be the identity, or the gate would pass on a no-op reader.
    if v1 == v2 {
        return Err(format!(
            "{}: v2 bytes equal v1 bytes — the permutation did nothing, gate is vacuous",
            cell.name
        )
        .into());
    }

    let (ex_ids, ex_off, _ex_pairs, csr_tok) = build_csr(n_pairs, n_expert, n_used);
    let n_active = ex_ids.len();
    let group_sizes: Vec<i32> = ex_off.windows(2).map(|w| w[1] - w[0]).collect();
    let max_m = group_sizes.iter().copied().max().unwrap_or(0);
    let min_m = group_sizes.iter().copied().min().unwrap_or(0);

    let bank_v1 = e.htod_bytes(&v1)?;
    let bank_v2 = e.htod_bytes(&v2)?;
    let tab_v1 = pointer_table(e, &bank_v1, n_expert, expert_bytes)?;
    let tab_v2 = pointer_table(e, &bank_v2, n_expert, expert_bytes)?;
    let exi = e.htod_i32(&ex_ids)?;
    let exoff = e.htod_i32(&ex_off)?;
    let csr = e.htod_i32(&csr_tok)?;

    // (3) identical activations for both arms — non-trivial, deterministic, and NOT constant:
    // a constant activation row makes every expert's dot a scaled row-sum and can hide a
    // permutation error that only shows when neighbouring values differ.
    let mut rng = Rng(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    let z: Vec<f32> = (0..t * cell.in_f)
        .map(|_| (rng.next_u64() >> 40) as f32 / 8388608.0 - 0.0625)
        .collect();
    let z_d = e.htod(&z)?;
    let (act, act_scale) = e.moe_f16g_act(&z_d, Some(&csr), cell.in_f, n_pairs)?;

    let y1 = e.moe_f16_grouped(
        &tab_v1,
        0,
        n_expert,
        &exi,
        &ex_off,
        &exoff,
        &act,
        &act_scale,
        cell.in_f,
        cell.out_f,
        n_active,
        n_pairs,
        memra_engine::QT_NVFP4,
        row_bytes,
    )?;
    let y2 = e.moe_f16_grouped(
        &tab_v2,
        0,
        n_expert,
        &exi,
        &ex_off,
        &exoff,
        &act,
        &act_scale,
        cell.in_f,
        cell.out_f,
        n_active,
        n_pairs,
        memra_engine::QT_NVFP4_V2,
        row_bytes,
    )?;
    e.stream().synchronize()?;
    let h1 = e.dtoh(&y1)?;
    let h2 = e.dtoh(&y2)?;
    assert_eq!(h1.len(), h2.len());

    // An all-zero output means the arms agree on nothing: a sum of zeros is order-independent,
    // so it would be a green light that compared no arithmetic. Refuse it.
    let nz = h1.iter().filter(|v| **v != 0.0).count();
    let finite = h1.iter().all(|v| v.is_finite()) && h2.iter().all(|v| v.is_finite());
    let mut ndiff = 0usize;
    let mut maxabs = 0.0f64;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (a, b)) in h1.iter().zip(&h2).enumerate() {
        if a.to_bits() != b.to_bits() {
            ndiff += 1;
            let d = (*a as f64 - *b as f64).abs();
            if d > maxabs {
                maxabs = d;
            }
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
    }
    let rel = {
        let scale = h1.iter().fold(0.0f64, |m, v| m.max((*v as f64).abs()));
        if scale > 0.0 { maxabs / scale } else { 0.0 }
    };
    let ok = ndiff == 0 && nz > 0 && finite;
    println!(
        "[cell {}] in_f={} out_f={} n_expert={} n_pairs={} n_active={} group_m=[{}..{}] \
         nkb={} elems={} nonzero_v1={} finite={} DIFF={} maxabs={:.6e} maxrel={:.3e} => {}",
        cell.name,
        cell.in_f,
        cell.out_f,
        n_expert,
        n_pairs,
        n_active,
        min_m,
        max_m,
        cell.in_f / 64,
        h1.len(),
        nz,
        finite,
        ndiff,
        maxabs,
        rel,
        if ok { "IDENTICAL" } else { "DEVIATION" }
    );
    if nz == 0 {
        println!(
            "[cell {}] REFUSED: v1 output is all zeros — this gate would have compared nothing",
            cell.name
        );
    }
    if let Some((i, a, b)) = first {
        println!(
            "[cell {}] first deviation at element {i}: v1={a:.9e} ({:#010x}) v2={b:.9e} ({:#010x})",
            cell.name,
            a.to_bits(),
            b.to_bits()
        );
    }
    Ok(ok)
}

/// One decode-sweep cell: the three restored programs, each against a v1-bytes oracle.
///
/// `n_sel` is the routed pair count of ONE token (step37 decode: top-8), which is the shape the
/// serving decode sweep actually launches — and, for P3, the shape the reduce identity is argued
/// at (`n_sel <= 8`, `nsb <= 32`, launcher-enforced).
fn run_decode_cell(
    e: &Engine,
    cell: &Cell,
    n_expert: usize,
    n_sel: usize,
    seed: u64,
) -> Result<bool, Box<dyn std::error::Error>> {
    let row_bytes = cell.in_f / 64 * 36;
    let rows = n_expert * cell.out_f;
    let expert_bytes = cell.out_f * row_bytes;

    // Two INDEPENDENT banks. `a` plays gate (and the single-bank P1/P3 subject), `b` plays up.
    // Distinct value streams are what make a gate-rows-for-up-rows confusion visible in P2.
    let mut arms: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for k in 0..2u64 {
        let v1 = random_v1_bank(
            seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k + 1)),
            rows,
            cell.in_f,
        );
        let mut v2 = Vec::with_capacity(v1.len());
        for ex in 0..n_expert {
            let m = &v1[ex * expert_bytes..(ex + 1) * expert_bytes];
            v2.extend_from_slice(&memra_engine::tp::nvfp4_matrix_v2_permute(
                m, cell.out_f, cell.in_f,
            ));
        }
        if v1 == v2 {
            return Err(
                format!("{}: permutation is the identity — vacuous gate", cell.name).into(),
            );
        }
        arms.push((v1, v2));
    }
    let (a1, a2) = (e.htod_bytes(&arms[0].0)?, e.htod_bytes(&arms[0].1)?);
    let (b1, b2) = (e.htod_bytes(&arms[1].0)?, e.htod_bytes(&arms[1].1)?);

    // Selection: distinct experts, deterministic, and NOT the identity permutation 0..n_sel —
    // `sel[t]*expert_stride` addressing is exactly what a slot-major stride error would break,
    // and sel==t would leave that arithmetic untested.
    let mut rng = Rng(seed ^ 0x5DEE_CE66_D1EE_1234);
    let sel: Vec<i32> = (0..n_sel)
        .map(|_| (rng.next_u64() % n_expert as u64) as i32)
        .collect();
    let sel_d = e.htod_i32(&sel)?;

    // One shared activation row (the serving shape: one q8_1 quantization per rank reused across
    // every selected expert), so act_row_stride = ad_row_stride = 0 for P1/P2.
    let act: Vec<f32> = (0..cell.in_f)
        .map(|_| (rng.next_u64() >> 40) as f32 / 8388608.0 - 0.0625)
        .collect();
    let act_d = e.htod(&act)?;
    let (aq, ad) = e.quantize_q8_1(&act_d, 1, cell.in_f)?;

    let mut ok = true;
    let mut report = |name: &str, oracle: &[f32], arm: &[f32]| {
        assert_eq!(oracle.len(), arm.len());
        let nz = oracle.iter().filter(|v| **v != 0.0).count();
        let finite = oracle.iter().all(|v| v.is_finite()) && arm.iter().all(|v| v.is_finite());
        let mut ndiff = 0usize;
        let mut first: Option<(usize, f32, f32)> = None;
        for (i, (x, y)) in oracle.iter().zip(arm).enumerate() {
            if x.to_bits() != y.to_bits() {
                ndiff += 1;
                if first.is_none() {
                    first = Some((i, *x, *y));
                }
            }
        }
        let pass = ndiff == 0 && nz > 0 && finite;
        ok &= pass;
        println!(
            "[decode {} {}] in_f={} out_f={} n_sel={} elems={} nonzero_oracle={} finite={} \
             DIFF={} => {}",
            cell.name,
            name,
            cell.in_f,
            cell.out_f,
            n_sel,
            oracle.len(),
            nz,
            finite,
            ndiff,
            if pass { "IDENTICAL" } else { "DEVIATION" }
        );
        if nz == 0 {
            println!(
                "[decode {} {}] REFUSED: oracle output is all zeros",
                cell.name, name
            );
        }
        if let Some((i, x, y)) = first {
            println!(
                "[decode {} {}] first deviation at {i}: oracle={x:.9e} ({:#010x}) arm={y:.9e} ({:#010x})",
                cell.name,
                name,
                x.to_bits(),
                y.to_bits()
            );
        }
    };

    // ── P1: the slot-major sel reader against the v1 sel reader ────────────────────────────
    let mut y_v1 = e.uninit(n_sel * cell.out_f)?;
    let mut y_v2 = e.uninit(n_sel * cell.out_f)?;
    e.qmatvec_nvfp4_sel_into(
        &a1,
        &sel_d,
        &aq,
        &ad,
        &mut y_v1,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        0,
        0,
        false,
    )?;
    e.qmatvec_nvfp4_sel_into(
        &a2,
        &sel_d,
        &aq,
        &ad,
        &mut y_v2,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        0,
        0,
        true,
    )?;
    e.stream().synchronize()?;
    let (h_v1, h_v2) = (e.dtoh(&y_v1)?, e.dtoh(&y_v2)?);
    report("P1_sel_v2_vs_sel", &h_v1, &h_v2);

    // ── P2: the fused gate+up launch against TWO v1 sweeps ─────────────────────────────────
    let mut og = e.uninit(n_sel * cell.out_f)?;
    let mut ou = e.uninit(n_sel * cell.out_f)?;
    e.qmatvec_nvfp4_sel_into(
        &a1,
        &sel_d,
        &aq,
        &ad,
        &mut og,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        0,
        0,
        false,
    )?;
    e.qmatvec_nvfp4_sel_into(
        &b1,
        &sel_d,
        &aq,
        &ad,
        &mut ou,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        0,
        0,
        false,
    )?;
    let mut fg = e.uninit(n_sel * cell.out_f)?;
    let mut fu = e.uninit(n_sel * cell.out_f)?;
    e.qmatvec_nvfp4_sel_gu_into(
        &a2,
        &b2,
        &sel_d,
        &aq,
        &ad,
        &mut fg,
        &mut fu,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        true,
    )?;
    e.stream().synchronize()?;
    let (hog, hou) = (e.dtoh(&og)?, e.dtoh(&ou)?);
    let (hfg, hfu) = (e.dtoh(&fg)?, e.dtoh(&fu)?);
    report("P2_gu_gate_half", &hog, &hfg);
    report("P2_gu_up_half", &hou, &hfu);
    // The fusion must not have written the same rows twice: gate and up ride different banks, so
    // identical halves would mean one bank was read for both and BOTH comparisons above could
    // still pass if the oracle sweeps had also collapsed. Assert the banks really differ.
    if hog == hou {
        return Err(format!(
            "{}: the two oracle sweeps agree — the independent banks collapsed, P2 is vacuous",
            cell.name
        )
        .into());
    }

    // ── P3: the fused down+combine against sweep + axpy_rows_seq_md ────────────────────────
    // Only where the launcher's reduce-identity class holds (nsb <= 32 <=> in_f <= 1024).
    if (cell.in_f >> 5) <= 32 && n_sel <= 8 {
        let route_w: Vec<f32> = (0..n_sel)
            .map(|_| 0.05 + (rng.next_u64() >> 40) as f32 / 16777216.0)
            .collect();
        let md: Vec<f32> = (0..n_expert)
            .map(|_| 1.0e-4 + (rng.next_u64() >> 44) as f32 / 1.0e9)
            .collect();
        let rw_d = e.htod(&route_w)?;
        let md_d = e.htod(&md)?;
        // The oracle pair: v1 sweep into an n_sel x out_f partial, then the sequential
        // slot-ordered combine. This is the exact two-launch program down8 replaces.
        let mut partial = e.uninit(n_sel * cell.out_f)?;
        e.qmatvec_nvfp4_sel_into(
            &a1,
            &sel_d,
            &aq,
            &ad,
            &mut partial,
            n_sel,
            cell.in_f,
            cell.out_f,
            row_bytes,
            expert_bytes,
            0,
            0,
            false,
        )?;
        let mut acc_ref = e.uninit(cell.out_f)?;
        e.axpy_rows_seq_md_into(
            &partial,
            &rw_d,
            &md_d,
            &sel_d,
            &mut acc_ref,
            cell.out_f,
            n_sel,
        )?;
        let mut acc_f = e.uninit(cell.out_f)?;
        e.qmatvec_nvfp4_sel_down8_into(
            &a2,
            &sel_d,
            &aq,
            &ad,
            &rw_d,
            &md_d,
            &mut acc_f,
            n_sel,
            cell.in_f,
            cell.out_f,
            row_bytes,
            expert_bytes,
            0,
            0,
            true,
        )?;
        e.stream().synchronize()?;
        let (h_ref, h_f) = (e.dtoh(&acc_ref)?, e.dtoh(&acc_f)?);
        report("P3_down8_vs_sweep_plus_axpy", &h_ref, &h_f);
    } else {
        println!(
            "[decode {} P3_down8] SKIPPED: nsb={} > 32 — outside the launcher's reduce-identity \
             class (this is the down geometry's gate, not gate/up's)",
            cell.name,
            cell.in_f >> 5
        );
    }

    // The launchers must REFUSE v1 bytes rather than read them as slot-major. A silent
    // mis-read here is precisely the 2026-08-29 failure shape, so it is gated, not assumed.
    let mut junk_g = e.uninit(n_sel * cell.out_f)?;
    let mut junk_u = e.uninit(n_sel * cell.out_f)?;
    if e.qmatvec_nvfp4_sel_gu_into(
        &a1,
        &b1,
        &sel_d,
        &aq,
        &ad,
        &mut junk_g,
        &mut junk_u,
        n_sel,
        cell.in_f,
        cell.out_f,
        row_bytes,
        expert_bytes,
        false,
    )
    .is_ok()
    {
        return Err(format!(
            "{}: gu launcher ACCEPTED slot_major=false — the layout guard is not a guard",
            cell.name
        )
        .into());
    }
    println!(
        "[decode {}] guard: gu launcher refuses non-slot-major banks",
        cell.name
    );
    Ok(ok)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_expert = env_usize("MEMRA_ORACLE_EXPERTS", 32);
    let t = env_usize("MEMRA_ORACLE_T", 64);
    let hidden = env_usize("MEMRA_ORACLE_HIDDEN", 4096);
    let ff = env_usize("MEMRA_ORACLE_FF", 640);
    let n_used = 8usize;

    let e = Engine::new(0)?;
    // The grouped-MoE FFI's raw launches follow the RUNTIME API's current device, not the pushed
    // driver context (the same bind the server's prime does per rank).
    e.bind_runtime_device(e.ctx().ordinal() as i32)?;

    let (shape_sel, cross) = memra_engine::moe_f16g_sk_params();
    println!(
        "nvfp4-bank-oracle: v1(QT_NVFP4) vs v2(QT_NVFP4_V2) over ONE permuted bank\n\
         arm policy: shape_sel={shape_sel} cross={cross} deep_tail={} f16g_mode={} \
         direct_on={} | MEMRA_F16G_SK / _SK_CROSS / _TAIL select the tile form",
        memra_engine::moe_f16g_tail_on(),
        memra_engine::moe_f16g_mode(),
        memra_engine::moe_f16g_direct_on(memra_engine::QT_NVFP4_V2),
    );
    if memra_engine::moe_f16g_mode() < 2 || shape_sel < 0 {
        return Err(
            "this oracle gates the sk/direct lane; MEMRA_MOE_F16G>=2 and \
                    MEMRA_F16G_SK!=0 are required or the call falls back to the dequant \
                    workspace and the gate proves nothing about kq_fetch"
                .into(),
        );
    }

    // Both step37 layer geometries. gate/up and down differ in row alignment (2304B vs 360B)
    // and in which side of the k walk is short, so neither stands in for the other.
    let cells = [
        Cell {
            name: "gate_up",
            in_f: hidden,
            out_f: ff,
        },
        Cell {
            name: "down",
            in_f: ff,
            out_f: hidden,
        },
    ];
    let mut all_ok = true;
    for (i, cell) in cells.iter().enumerate() {
        let ok = run_cell(
            &e,
            cell,
            n_expert,
            t,
            n_used,
            0x243F_6A88_85A3_08D3 ^ (i as u64 + 1),
        )?;
        all_ok &= ok;
    }
    // DECODE SWEEP FAMILY — one arm per restored program (see the module doc). Independent of
    // the tile-form policy above, which only selects the PREFILL GEMM shape.
    println!(
        "nvfp4-bank-oracle decode: P1 _sel_v2 | P2 _sel_v2_gu | P3 _sel_v2_down8, \
         every oracle arm pinned to v1 bytes"
    );
    for (i, cell) in cells.iter().enumerate() {
        let ok = run_decode_cell(
            &e,
            cell,
            n_expert,
            n_used,
            0x13198A2E_03707344u64 ^ (i as u64 + 1),
        )?;
        all_ok &= ok;
    }
    if all_ok {
        println!("nvfp4-bank-oracle: PASS — every cell bit-identical");
        Ok(())
    } else {
        println!(
            "nvfp4-bank-oracle: FAIL — the slot-major bank is not a pure permutation on this arm"
        );
        std::process::exit(1);
    }
}
