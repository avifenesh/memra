//! Two-card persistent replicated-row all-reduce gate.
//!
//! Exercises the same 16 KiB hidden-row shape needed by TP2 replicated-residual decode: two
//! peer pushes, two reusable event waits, and the same `(rank0 + rank1)` add on both devices.

use memra_engine::tp::TpE4m3HostBounce;

fn bf16_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

fn signed_unit(seed: u64) -> f32 {
    let mut mixed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^= mixed >> 31;
    let unit = (mixed >> 40) as f32 / (1u32 << 24) as f32;
    unit - 0.5
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);
    let repetitions = std::env::args()
        .nth(2)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1000);
    if repetitions == 0 {
        return Err("repetitions must be nonzero".into());
    }

    let runtime = TpE4m3HostBounce::new_native_p2p(&[0, 1])?;
    let rank0 = runtime.rank_engine(0).ok_or("missing rank 0")?;
    let rank1 = runtime.rank_engine(1).ok_or("missing rank 1")?;
    let input0 = (0..width)
        .map(|index| index as f32 * 0.25 - 17.0)
        .collect::<Vec<_>>();
    let input1 = (0..width)
        .map(|index| 9.0 - index as f32 * 0.125)
        .collect::<Vec<_>>();
    let expected = input0
        .iter()
        .zip(&input1)
        .map(|(&left, &right)| left + right)
        .collect::<Vec<_>>();
    let partial0 = rank0.htod(&input0)?;
    let partial1 = rank1.htod(&input1)?;
    let mut output0 = rank0.zeros(width)?;
    let mut output1 = rank1.zeros(width)?;
    let mut join = runtime.prepare_tp2_replicated_row_join(width)?;

    for _ in 0..20 {
        runtime.tp2_replicated_row_join(
            &mut join,
            &partial0,
            &partial1,
            &mut output0,
            &mut output1,
        )?;
    }
    rank0.stream().synchronize()?;
    rank1.stream().synchronize()?;

    let started = std::time::Instant::now();
    for _ in 0..repetitions {
        runtime.tp2_replicated_row_join(
            &mut join,
            &partial0,
            &partial1,
            &mut output0,
            &mut output1,
        )?;
    }
    rank0.stream().synchronize()?;
    rank1.stream().synchronize()?;
    let elapsed = started.elapsed().as_secs_f64();

    let host0 = rank0.dtoh(&output0)?;
    let host1 = rank1.dtoh(&output1)?;
    if host0 != expected || host1 != expected || host0 != host1 {
        return Err("replicated-row join output mismatch".into());
    }
    let microseconds = elapsed * 1.0e6 / repetitions as f64;
    println!(
        "TP2_REPLICATED_ROW_JOIN PASS width={width} bytes={} repetitions={repetitions} \
         us_per_join={microseconds:.3} outputs=bit-identical",
        width * std::mem::size_of::<f32>()
    );

    // Real HY3 shared-expert-down geometry. Each rank reads one aligned K half, then the same
    // persistent join publishes the row on both devices. Compare against the unsplit t=1 BF16
    // program; TP changes only the cross-half association, so this is a tolerance gate.
    let (in_f, out_f) = (1536usize, 4096usize);
    let half = in_f / 2;
    let weight_bytes =
        bf16_bytes((0..out_f).flat_map(|row| {
            (0..in_f).map(move |col| signed_unit((row * in_f + col) as u64) * 0.5)
        }));
    let input = (0..in_f)
        .map(|index| signed_unit(index as u64 ^ 0xd1b5_4a32_d192_ed03))
        .collect::<Vec<_>>();
    let weight0 = rank0.htod_bytes(&weight_bytes)?;
    let weight1 = rank1.htod_bytes(&weight_bytes)?;
    let reference_input = rank0.htod(&input)?;
    let input0 = rank0.htod(&input[..half])?;
    let input1 = rank1.htod(&input[half..])?;
    let mut reference = rank0.zeros(out_f)?;
    let mut partial0 = rank0.zeros(out_f)?;
    let mut partial1 = rank1.zeros(out_f)?;
    let mut joined0 = rank0.zeros(out_f)?;
    let mut joined1 = rank1.zeros(out_f)?;
    let mut row_join = runtime.prepare_tp2_replicated_row_join(out_f)?;
    rank0.matvec_bf16_rows_into(&weight0, &reference_input, &mut reference, in_f, out_f, 1)?;
    rank0.matvec_bf16_col_range_into(&weight0, &input0, &mut partial0, in_f, out_f, 0, half)?;
    rank1.matvec_bf16_col_range_into(&weight1, &input1, &mut partial1, in_f, out_f, half, half)?;
    runtime.tp2_replicated_row_join(
        &mut row_join,
        &partial0,
        &partial1,
        &mut joined0,
        &mut joined1,
    )?;
    rank0.stream().synchronize()?;
    rank1.stream().synchronize()?;
    let reference = rank0.dtoh(&reference)?;
    let joined0 = rank0.dtoh(&joined0)?;
    let joined1 = rank1.dtoh(&joined1)?;
    if joined0 != joined1 {
        return Err("row-parallel joined outputs differ between ranks".into());
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&expected, &actual) in reference.iter().zip(&joined0) {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(1.0e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > 1.0e-4 + 1.0e-4 * expected.abs() {
            return Err(format!(
                "row-parallel BF16 tolerance exceeded: expected={expected} actual={actual} \
                 abs={abs} rel={rel}"
            )
            .into());
        }
    }
    let reference_argmax = reference
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("empty reference row")?;
    let joined_argmax = joined0
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .ok_or("empty joined row")?;
    if reference_argmax != joined_argmax {
        return Err(format!(
            "row-parallel BF16 argmax mismatch: reference={reference_argmax} joined={joined_argmax}"
        )
        .into());
    }
    println!(
        "TP2_BF16_ROW_PARALLEL PASS geometry={out_f}x{in_f} split={half}+{half} \
         argmax={reference_argmax} max_abs={max_abs:.8} max_rel={max_rel:.8}"
    );
    Ok(())
}
