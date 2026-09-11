//! The PP-2 port gate for the three norm doors: `MEMRA_DSV4_NORM_FUSE` (#404),
//! `MEMRA_DSV4_NORM_FUSE2` (#426) and `MEMRA_DSV4_NORM2_WIDE` (#430).
//!
//! WHAT THE PORT CHANGED, AND WHAT THIS GATE IS THEREFORE FOR. The doors were
//! admitted under `is_tp_ep() && chains_f32` and their call sites additionally
//! required `t == 1`. The topology term was an assumption; the shape term was a
//! kernel limit, because every fused launcher pinned grid 1. The launchers now
//! take `rows`, one CTA per row, and the admission keeps only `chains_f32`. The
//! served program (PP-2, DSpark resident, chunked prefill) presents `t > 1` on
//! every spec verify round and every prefill chunk, so `rows > 1` is the shape
//! that decides whether the doors pay anything at all, and it is the shape no
//! merged receipt ever exercised.
//!
//! WHY THE CHECK IS BIT EQUALITY AND NOT A TOLERANCE. Rows are independent in
//! every one of these kernels. Each row's CTA repeats the identical 128-thread
//! eight-load accumulation and the identical `dsv4_block_sum_f32` tree that
//! `dsv4_rmsnorm_f32acc_kernel` runs for that row in the unfused arm, and the
//! epilogue is a pure function of (column, rsq) with the same `__float2bfloat16`
//! round-to-nearest the separate `dsv4_cvt_bf16` applies. The reduction tree and
//! every rounding point are unchanged, so the port is SAME-CLASS and one
//! differing bit is a defect. This gate refuses on the first one and names it:
//! row, column, and the bit index inside the word.
//!
//! NON-VACUITY. A multi-row sweep over identical rows would pass with a kernel
//! that reads row 0 for every CTA, which is exactly the defect a row port
//! introduces. So every row here carries DISTINCT data and a DISTINCT RoPE
//! position, and `red_arms` proves the sweep can see that: it feeds the fused
//! arm a constant position vector and requires the comparator to refuse, then
//! flips a single bf16 bit in the middle of the last row and requires the
//! comparator to name that exact offset.
use super::*;

const COLS_KV: usize = 512;
const RD: usize = 64;
const COLS_H: usize = 4096;
const COLS_SH: usize = 2048;
const CANARY: u32 = 0x4b123456;

/// Row counts the sweep walks. 1 is the pre-port geometry, and it stays so a
/// port that only works for `t > 1` fails here rather than on the box. 2/3/4/8
/// are DSpark verify widths. 129 is deliberately not a multiple of anything.
///
/// The top of the sweep is `DSV4_BATCH_WIDTH_MAX`, the kernel's own transaction
/// width, because that is the widest row count a served request can present: it
/// is both the ceiling `resolve_prefill_chunk` enforces and, since memra #460,
/// the DEFAULT chunk. It moved from 64 to 512 on 2026-09-11 while this port was
/// in flight, which is the argument for taking it from the constant instead of
/// typing the number: a row port correct at 129 and wrong at 512 would pass a
/// hand-written sweep and fail in production. `MAX - 1` rides along because the
/// wide arm's grid is `rows * tiles` and an off-by-one in the row/tile split
/// shows up at a non-multiple, not at the round number.
pub const ROW_SWEEP: [usize; 9] = [
    1,
    2,
    3,
    4,
    8,
    64,
    129,
    DSV4_BATCH_WIDTH_MAX - 1,
    DSV4_BATCH_WIDTH_MAX,
];
/// Every tile count `memra_dsv4_norm2_pack_wide` admits at 4096 columns with a
/// 128-thread block, so the wide arm is swept over its whole legal domain at
/// every row count rather than only at the pinned constant.
pub const TILE_SWEEP: [i32; 6] = [1, 2, 4, 8, 16, 32];

/// How many byte comparisons a complete run owes, derived from the sweeps rather
/// than counted after the fact.
///
/// This exists because of a vacuity class the memra #482 audit named and neither
/// lane had: not a check that cannot fail, but a check whose SUBJECT SET can
/// become empty. `for rows in ROW_SWEEP` over an empty sweep passes, prints
/// PASS, and has compared nothing. A `> 0` guard would catch the empty sweep and
/// nothing else; an exact expectation also catches a cell that silently stopped
/// contributing, which is the same defect one notch smaller.
///
/// Per row: the KV cell compares `rows * COLS_KV` f32 words; the norm2 cell
/// compares f32 AND bf16 for the single-CTA symbol plus every tile count, so
/// `2 * rows * COLS_H` per arm across `1 + TILE_SWEEP.len()` arms; the SwiGLU
/// cell compares `rows * COLS_SH`.
pub fn expected_comparisons(rows_sweep: &[usize], tiles_sweep: &[i32]) -> usize {
    let arms = 1 + tiles_sweep.len();
    rows_sweep
        .iter()
        .map(|&rows| rows * COLS_KV + arms * 2 * rows * COLS_H + rows * COLS_SH)
        .sum()
}

/// Deterministic, distinct per (row, col). No RNG crate and no file: the gate
/// must be reproducible from its own source.
fn operand(row: usize, col: usize, salt: u64) -> f32 {
    let mut h = (row as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (col as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ salt;
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    // Values in [-2, 2), never denormal, never zero: a row of zeros would make
    // rsq a constant and hide a row-offset defect behind equal answers.
    let unit = ((h >> 40) as f32) / (((1u64 << 24) - 1) as f32);
    let v = unit * 4.0 - 2.0;
    if v.abs() < 1e-3 { v + 0.5 } else { v }
}

fn rows_of(rows: usize, cols: usize, salt: u64) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| operand(i / cols, i % cols, salt))
        .collect()
}

/// The first differing BIT between two byte strings, as (byte offset, bit index
/// in that byte). `None` when the strings are equal.
pub fn first_differing_bit(a: &[u8], b: &[u8]) -> Option<(usize, u32)> {
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), 0));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            return Some((i, (x ^ y).trailing_zeros()));
        }
    }
    None
}

/// Refuse on the first differing bit, naming the row and column it lands in so
/// the failure points at the port seam rather than at "the arms differ".
fn refuse_on_first_bit(
    what: &str,
    rows: usize,
    cols: usize,
    width: usize,
    control: &[u8],
    arm: &[u8],
) -> Res<()> {
    if control.is_empty() || arm.is_empty() {
        // Same class of defect one level down: two empty buffers "match", so a
        // cell that produced nothing would report bit equality.
        return Err(format!(
            "{what} compared an EMPTY buffer at rows={rows} cols={cols}: nothing was checked"
        ));
    }
    match first_differing_bit(control, arm) {
        None => Ok(()),
        Some((byte, bit)) => {
            let elem = byte / width;
            Err(format!(
                "{what} raw-bit mismatch rows={rows} cols={cols} row={} col={} byte={byte} bit={bit} control=0x{:02x} arm=0x{:02x}",
                elem / cols,
                elem % cols,
                control[byte],
                arm[byte]
            ))
        }
    }
}

fn read_f32(stream: &std::sync::Arc<CudaStream>, d: &CudaSlice<f32>) -> Res<Vec<u8>> {
    let host = dtoh_f32(stream, d)?;
    let mut out = Vec::with_capacity(host.len() * 4);
    for v in host {
        out.extend(v.to_bits().to_le_bytes());
    }
    Ok(out)
}

impl Dsv4Gpu {
    /// The whole port gate. Device-only, no model, no GPU lock contention beyond
    /// one context: it is a kernel contract check, and it runs in seconds.
    pub fn run_norm_pp2_port_gate_for_gate(device: usize) -> Res<()> {
        let ctx = cudarc::driver::CudaContext::new(device).map_err(e("pp2 port context"))?;
        let stream = ctx.new_stream().map_err(e("pp2 port stream"))?;
        println!(
            "PP2_PORT_PROTOCOL doors=MEMRA_DSV4_NORM_FUSE,MEMRA_DSV4_NORM_FUSE2,MEMRA_DSV4_NORM2_WIDE class=same device={device} rows={ROW_SWEEP:?} tiles={TILE_SWEEP:?}"
        );
        // The empty-subject guard, before the loop rather than after it. An
        // empty sweep is not a fast PASS, it is a gate that tested nothing.
        if ROW_SWEEP.is_empty() || TILE_SWEEP.is_empty() {
            return Err(
                "PP2 port gate has an empty sweep: it would PASS having compared nothing".into(),
            );
        }
        let owed = expected_comparisons(&ROW_SWEEP, &TILE_SWEEP);
        let mut comparisons = 0usize;
        for rows in ROW_SWEEP {
            comparisons += Self::pp2_norm_rope_cell(&stream, rows)?;
            comparisons += Self::pp2_norm2_pack_cell(&stream, rows)?;
            comparisons += Self::pp2_swiglu_pack_cell(&stream, rows)?;
        }
        // And the count is CHECKED against what the sweeps owe, so a cell that
        // silently stopped contributing fails here instead of shrinking a number
        // nobody reads.
        if comparisons != owed {
            return Err(format!(
                "PP2 port gate compared {comparisons} bytes, sweeps owe {owed}: a cell did not run"
            ));
        }
        let red = Self::pp2_red_arms(&stream)?;
        if red == 0 {
            return Err("PP2 port gate ran no red arms".into());
        }
        println!(
            "PP2_PORT_PASS comparisons={comparisons} owed={owed} red_arms={red} bits_equal=true class=same"
        );
        Ok(())
    }

    /// `memra_dsv4_norm_rope_f32_fixed_order` over `rows` against the unfused
    /// pair the call site falls back to: `rmsnorm_f32acc` then `dsv4_rope` with
    /// `n_pos = rows`, `n_vec = 1`. Positions are distinct per row.
    fn pp2_norm_rope_cell(stream: &std::sync::Arc<CudaStream>, rows: usize) -> Res<usize> {
        let x = rows_of(rows, COLS_KV, 0x11);
        let w = rows_of(1, COLS_KV, 0x12);
        // cos/sin table: distinct per position, so two rows that read the same
        // row of it produce different bytes.
        let positions: Vec<i32> = (0..rows).map(|r| (7 * r + 3) as i32).collect();
        let max_pos = *positions.iter().max().unwrap() as usize;
        let cs = rows_of(max_pos + 1, RD, 0x13);
        let eps = 1e-6f32;

        let wd = upload_f32(stream, &w)?;
        let cd = upload_f32(stream, &cs)?;
        let pd = upload_i32(stream, &positions)?;
        let mut control = upload_f32(stream, &x)?;
        let mut arm = upload_f32(stream, &x)?;
        let mut scratch = stream
            .alloc_zeros::<f32>(rows * COLS_KV)
            .map_err(e("pp2 rope scratch"))?;

        unsafe {
            // control: the shipped unfused pair, in the shipped order.
            ck(
                "pp2 control rmsnorm",
                k::memra_dsv4_rmsnorm_f32acc(
                    control.device_ptr(stream).0 as *const f32,
                    wd.device_ptr(stream).0 as *const f32,
                    scratch.device_ptr_mut(stream).0 as *mut f32,
                    rows as i32,
                    COLS_KV as i32,
                    eps,
                    sp(stream),
                ),
            )?;
            ck(
                "pp2 control rope",
                k::memra_dsv4_rope(
                    scratch.device_ptr_mut(stream).0 as *mut f32,
                    rows as i32,
                    1,
                    COLS_KV as i32,
                    RD as i32,
                    cd.device_ptr(stream).0 as *const f32,
                    pd.device_ptr(stream).0 as *const i32,
                    0,
                    sp(stream),
                ),
            )?;
            ck(
                "pp2 arm norm rope fused",
                k::memra_dsv4_norm_rope_f32_fixed_order(
                    arm.device_ptr_mut(stream).0 as *mut f32,
                    wd.device_ptr(stream).0 as *const f32,
                    rows as i32,
                    COLS_KV as i32,
                    eps,
                    RD as i32,
                    cd.device_ptr(stream).0 as *const f32,
                    pd.device_ptr(stream).0 as *const i32,
                    sp(stream),
                ),
            )?;
        }
        stream.synchronize().map_err(e("pp2 rope drain"))?;
        let control_bytes = read_f32(stream, &scratch)?;
        let arm_bytes = read_f32(stream, &arm)?;
        refuse_on_first_bit("norm-fuse", rows, COLS_KV, 4, &control_bytes, &arm_bytes)?;
        println!(
            "PP2_CELL door=MEMRA_DSV4_NORM_FUSE rows={rows} cols={COLS_KV} rd={RD} distinct_positions={rows} bits_equal=true"
        );
        Ok(rows * COLS_KV)
    }

    /// `memra_dsv4_norm2_pack` and its wide twin over `rows`, against
    /// `rmsnorm_f32acc` + `cvt_bf16`, which is exactly what the call site runs
    /// with the door off (`rmsnorm ffn batch` then `cvt xb batch`). Both outputs
    /// are compared: the retained f32 row AND the bf16 pack.
    fn pp2_norm2_pack_cell(stream: &std::sync::Arc<CudaStream>, rows: usize) -> Res<usize> {
        let x = rows_of(rows, COLS_H, 0x21);
        let w = rows_of(1, COLS_H, 0x22);
        let eps = 1e-6f32;
        let n = rows * COLS_H;

        let xd = upload_f32(stream, &x)?;
        let wd = upload_f32(stream, &w)?;
        let mut ctrl_y = upload_f32(stream, &vec![f32::from_bits(CANARY); n])?;
        let mut ctrl_b = upload_u8(stream, &vec![0xa5; 2 * n])?;
        let mut arm_y = upload_f32(stream, &vec![f32::from_bits(CANARY); n])?;
        let mut arm_b = upload_u8(stream, &vec![0xa5; 2 * n])?;

        unsafe {
            ck(
                "pp2 control norm2 rmsnorm",
                k::memra_dsv4_rmsnorm_f32acc(
                    xd.device_ptr(stream).0 as *const f32,
                    wd.device_ptr(stream).0 as *const f32,
                    ctrl_y.device_ptr_mut(stream).0 as *mut f32,
                    rows as i32,
                    COLS_H as i32,
                    eps,
                    sp(stream),
                ),
            )?;
            ck(
                "pp2 control norm2 cvt",
                k::memra_dsv4_cvt_bf16(
                    ctrl_y.device_ptr(stream).0 as *const f32,
                    ctrl_b.device_ptr_mut(stream).0 as *mut c_void,
                    n as i64,
                    sp(stream),
                ),
            )?;
        }
        stream.synchronize().map_err(e("pp2 norm2 control drain"))?;
        let control_y = read_f32(stream, &ctrl_y)?;
        let control_b = stream.clone_dtoh(&ctrl_b).map_err(e("pp2 control pack"))?;

        let mut comparisons = 0usize;
        // `None` is the single-CTA symbol, then every legal tile count of the
        // wide symbol. Both must land on the SAME control bytes: the wide arm is
        // a same-class twin of the pack, and the pack is a same-class twin of
        // the unfused pair, so the whole family is one equivalence class.
        for tiles in std::iter::once(None).chain(TILE_SWEEP.map(Some)) {
            stream
                .memcpy_htod(&vec![f32::from_bits(CANARY); n], &mut arm_y)
                .map_err(e("pp2 arm reset y"))?;
            stream
                .memcpy_htod(&vec![0xa5u8; 2 * n], &mut arm_b)
                .map_err(e("pp2 arm reset pack"))?;
            unsafe {
                let xp = xd.device_ptr(stream).0 as *const f32;
                let wp = wd.device_ptr(stream).0 as *const f32;
                let yp = arm_y.device_ptr_mut(stream).0 as *mut f32;
                let bp = arm_b.device_ptr_mut(stream).0 as *mut c_void;
                match tiles {
                    None => ck(
                        "pp2 arm norm2 pack",
                        k::memra_dsv4_norm2_pack(
                            xp,
                            wp,
                            yp,
                            bp,
                            rows as i32,
                            COLS_H as i32,
                            eps,
                            sp(stream),
                        ),
                    )?,
                    Some(tiles) => ck(
                        "pp2 arm norm2 pack wide",
                        k::memra_dsv4_norm2_pack_wide(
                            xp,
                            wp,
                            yp,
                            bp,
                            rows as i32,
                            COLS_H as i32,
                            eps,
                            tiles,
                            sp(stream),
                        ),
                    )?,
                }
            }
            stream.synchronize().map_err(e("pp2 norm2 arm drain"))?;
            let arm_y_bytes = read_f32(stream, &arm_y)?;
            let arm_b_bytes = stream.clone_dtoh(&arm_b).map_err(e("pp2 arm pack"))?;
            let label = match tiles {
                None => "norm-fuse2".to_string(),
                Some(t) => format!("norm2-wide-tiles{t}"),
            };
            refuse_on_first_bit(&label, rows, COLS_H, 4, &control_y, &arm_y_bytes)?;
            refuse_on_first_bit(&label, rows, COLS_H, 2, &control_b, &arm_b_bytes)?;
            comparisons += 2 * n;
            println!(
                "PP2_CELL door=MEMRA_DSV4_NORM_FUSE2 arm={label} rows={rows} cols={COLS_H} bits_equal=true f32_and_bf16=true"
            );
        }
        Ok(comparisons)
    }

    /// `memra_dsv4_norm2_swiglu_pack` over `rows` against `dsv4_swiglu` +
    /// `cvt_bf16`, the pair the call site runs with the door off.
    fn pp2_swiglu_pack_cell(stream: &std::sync::Arc<CudaStream>, rows: usize) -> Res<usize> {
        let g = rows_of(rows, COLS_SH, 0x31);
        let u = rows_of(rows, COLS_SH, 0x32);
        let limit = 7.0f32;
        let n = rows * COLS_SH;

        let gd = upload_f32(stream, &g)?;
        let ud = upload_f32(stream, &u)?;
        let mut ctrl_h = upload_f32(stream, &vec![f32::from_bits(CANARY); n])?;
        let mut ctrl_b = upload_u8(stream, &vec![0xa5; 2 * n])?;
        let mut arm_b = upload_u8(stream, &vec![0xa5; 2 * n])?;

        unsafe {
            ck(
                "pp2 control swiglu",
                k::memra_dsv4_swiglu(
                    gd.device_ptr(stream).0 as *const f32,
                    ud.device_ptr(stream).0 as *const f32,
                    ctrl_h.device_ptr_mut(stream).0 as *mut f32,
                    rows as i32,
                    COLS_SH as i32,
                    limit,
                    std::ptr::null(),
                    sp(stream),
                ),
            )?;
            ck(
                "pp2 control swiglu cvt",
                k::memra_dsv4_cvt_bf16(
                    ctrl_h.device_ptr(stream).0 as *const f32,
                    ctrl_b.device_ptr_mut(stream).0 as *mut c_void,
                    n as i64,
                    sp(stream),
                ),
            )?;
            ck(
                "pp2 arm swiglu pack",
                k::memra_dsv4_norm2_swiglu_pack(
                    gd.device_ptr(stream).0 as *const f32,
                    ud.device_ptr(stream).0 as *const f32,
                    arm_b.device_ptr_mut(stream).0 as *mut c_void,
                    rows as i32,
                    COLS_SH as i32,
                    limit,
                    sp(stream),
                ),
            )?;
        }
        stream.synchronize().map_err(e("pp2 swiglu drain"))?;
        let control_b = stream
            .clone_dtoh(&ctrl_b)
            .map_err(e("pp2 swiglu control"))?;
        let arm_b_bytes = stream.clone_dtoh(&arm_b).map_err(e("pp2 swiglu arm"))?;
        refuse_on_first_bit(
            "norm-fuse2-swiglu",
            rows,
            COLS_SH,
            2,
            &control_b,
            &arm_b_bytes,
        )?;
        println!(
            "PP2_CELL door=MEMRA_DSV4_NORM_FUSE2 arm=swiglu-pack rows={rows} cols={COLS_SH} bits_equal=true"
        );
        Ok(n)
    }

    /// The arms that must FAIL. Without these the sweep above is a check that
    /// has never failed, which is not a check.
    ///
    /// R1 is the defect a row port actually introduces: a fused kernel that
    /// keeps reading `positions[0]` for every CTA. Feeding the fused arm a
    /// constant position vector while the control keeps distinct ones must
    /// produce a mismatch; if it does not, the rows in the sweep above are not
    /// actually distinguishable and every PASS there is vacuous.
    ///
    /// R2 proves the comparator inspects the whole buffer and reports the right
    /// place: one bf16 bit is flipped in the LAST row, and the reported byte
    /// offset and bit index must be exactly that one.
    fn pp2_red_arms(stream: &std::sync::Arc<CudaStream>) -> Res<usize> {
        let rows = 8usize;
        let x = rows_of(rows, COLS_KV, 0x11);
        let w = rows_of(1, COLS_KV, 0x12);
        let positions: Vec<i32> = (0..rows).map(|r| (7 * r + 3) as i32).collect();
        let flat: Vec<i32> = vec![positions[0]; rows];
        let cs = rows_of(*positions.iter().max().unwrap() as usize + 1, RD, 0x13);
        let eps = 1e-6f32;

        let wd = upload_f32(stream, &w)?;
        let cd = upload_f32(stream, &cs)?;
        let pd = upload_i32(stream, &positions)?;
        let fd = upload_i32(stream, &flat)?;
        let mut honest = upload_f32(stream, &x)?;
        let mut flattened = upload_f32(stream, &x)?;
        unsafe {
            for (dst, pos) in [(&mut honest, &pd), (&mut flattened, &fd)] {
                ck(
                    "pp2 red arm fused",
                    k::memra_dsv4_norm_rope_f32_fixed_order(
                        dst.device_ptr_mut(stream).0 as *mut f32,
                        wd.device_ptr(stream).0 as *const f32,
                        rows as i32,
                        COLS_KV as i32,
                        eps,
                        RD as i32,
                        cd.device_ptr(stream).0 as *const f32,
                        pos.device_ptr(stream).0 as *const i32,
                        sp(stream),
                    ),
                )?;
            }
        }
        stream.synchronize().map_err(e("pp2 red drain"))?;
        let honest_bytes = read_f32(stream, &honest)?;
        let flat_bytes = read_f32(stream, &flattened)?;
        let r1 = first_differing_bit(&honest_bytes, &flat_bytes)
            .ok_or("RED ARM DEAD: a constant position vector produced identical bytes, so the row sweep cannot see a row-offset defect")?;
        // It must differ in a row OTHER than row 0: row 0 reads positions[0] in
        // both arms, so a difference there would mean something else broke.
        let r1_row = (r1.0 / 4) / COLS_KV;
        if r1_row == 0 {
            return Err(format!(
                "RED ARM WRONG PLACE: first difference in row 0 (byte {}), which both arms compute identically",
                r1.0
            ));
        }

        let mut poisoned = honest_bytes.clone();
        let target = (rows - 1) * COLS_KV * 4 + 17;
        poisoned[target] ^= 0b0000_1000;
        let r2 = first_differing_bit(&honest_bytes, &poisoned)
            .ok_or("RED ARM DEAD: a flipped bit compared equal")?;
        if r2 != (target, 3) {
            return Err(format!(
                "RED ARM MISREPORTS: flipped ({target}, 3), reported {r2:?}"
            ));
        }
        // And the refusal the gate would raise must name that row and column.
        let refusal = refuse_on_first_bit("red", rows, COLS_KV, 4, &honest_bytes, &poisoned)
            .expect_err("poisoned bytes must refuse");
        let want_row = (target / 4) / COLS_KV;
        let want_col = (target / 4) % COLS_KV;
        if !refusal.contains(&format!("row={want_row} col={want_col}")) {
            return Err(format!("RED ARM MISNAMES: {refusal}"));
        }
        println!(
            "PP2_RED_ARM r1_constant_positions_differ_at_row={r1_row} r2_flipped_bit_reported_at={r2:?} refusal={refusal:?}"
        );
        Ok(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The comparator itself, off the device: equal strings pass, and a flip at
    /// a known place is reported at exactly that place and no earlier.
    #[test]
    fn the_comparator_names_the_first_differing_bit() {
        let a = vec![0x00u8, 0xff, 0x0f, 0x10];
        assert_eq!(first_differing_bit(&a, &a), None);
        let mut b = a.clone();
        b[2] ^= 0b0100_0000;
        assert_eq!(first_differing_bit(&a, &b), Some((2, 6)));
        // Two differences: the FIRST is reported, which is the contract.
        let mut c = b.clone();
        c[1] ^= 0b0000_0001;
        assert_eq!(first_differing_bit(&a, &c), Some((1, 0)));
        // Length disagreement is a refusal too, never a silent prefix compare.
        assert!(first_differing_bit(&a, &a[..3]).is_some());
    }

    /// The sweep must reach the widest row count a served request can present.
    /// This is the check that would have caught the 64 -> 512 chunk default move
    /// (memra #460) landing under a hand-written sweep that stopped at 129.
    #[test]
    fn the_sweep_reaches_the_served_transaction_ceiling() {
        assert!(
            ROW_SWEEP.contains(&DSV4_BATCH_WIDTH_MAX),
            "sweep {ROW_SWEEP:?} never reaches the kernel transaction width {DSV4_BATCH_WIDTH_MAX}"
        );
        assert!(
            ROW_SWEEP.contains(&1),
            "the pre-port geometry left the sweep"
        );
        // And the wide arm's grid at the ceiling must be a legal launch
        // dimension, which is the shape of bug the extra rows are hunting.
        for tiles in TILE_SWEEP {
            let grid = DSV4_BATCH_WIDTH_MAX as u64 * tiles as u64;
            assert!(grid <= u32::MAX as u64, "grid {grid} at tiles={tiles}");
        }
    }

    /// The empty-subject vacuity class, tested directly (memra #482 audit,
    /// 2026-09-11). A loop over a derived subject list PASSES when the list is
    /// empty, so a check whose whole job is to compare things can end up proving
    /// nothing about nothing. An exact expectation, rather than a `> 0` guard,
    /// also catches the smaller version: a cell that stopped contributing.
    #[test]
    fn an_empty_sweep_owes_nothing_and_the_gate_would_notice() {
        assert_eq!(expected_comparisons(&[], &TILE_SWEEP), 0);
        assert_eq!(expected_comparisons(&ROW_SWEEP, &[]), {
            // Even with no wide tiles the single-CTA arm still owes its bytes,
            // so an empty TILE_SWEEP is a smaller subject set, not an empty one.
            ROW_SWEEP
                .iter()
                .map(|&r| r * COLS_KV + 2 * r * COLS_H + r * COLS_SH)
                .sum::<usize>()
        });
        assert!(expected_comparisons(&ROW_SWEEP, &TILE_SWEEP) > 0);
        // Dropping one row from the sweep must change the expectation, or the
        // expectation is not actually derived from the sweep.
        assert_ne!(
            expected_comparisons(&ROW_SWEEP, &TILE_SWEEP),
            expected_comparisons(&ROW_SWEEP[..ROW_SWEEP.len() - 1], &TILE_SWEEP)
        );
    }

    /// The comparator refuses an empty buffer rather than reporting equality,
    /// which is the same vacuity one level down: two empty strings "match".
    #[test]
    fn the_comparator_refuses_an_empty_buffer() {
        let err = refuse_on_first_bit("x", 0, 512, 4, &[], &[]).unwrap_err();
        assert!(err.contains("EMPTY"), "{err}");
        // And it still passes real equal buffers, so the guard did not just
        // break the ordinary path.
        assert!(refuse_on_first_bit("x", 1, 2, 4, &[0u8; 8], &[0u8; 8]).is_ok());
    }

    /// The operand generator must actually distinguish rows, or the whole sweep
    /// is vacuous before a kernel ever runs.
    #[test]
    fn every_row_of_a_cell_is_distinct() {
        let rows = rows_of(8, 512, 0x11);
        for r in 1..8 {
            assert_ne!(
                &rows[0..512],
                &rows[r * 512..(r + 1) * 512],
                "row {r} equals row 0"
            );
        }
        assert!(rows.iter().all(|v| v.is_finite() && v.abs() >= 1e-3));
    }

    /// The refusal string is what an operator reads. It must carry the row and
    /// column, not just "the arms differ".
    #[test]
    fn the_refusal_names_the_row_and_column() {
        let a = vec![0u8; 4 * 512 * 3];
        let mut b = a.clone();
        b[2 * 512 * 4 + 40] = 1;
        let err = refuse_on_first_bit("x", 3, 512, 4, &a, &b).unwrap_err();
        assert!(err.contains("row=2"), "{err}");
        assert!(err.contains("col=10"), "{err}");
        assert!(refuse_on_first_bit("x", 3, 512, 4, &a, &a).is_ok());
    }
}
