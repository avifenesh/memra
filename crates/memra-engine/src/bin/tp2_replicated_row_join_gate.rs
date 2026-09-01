//! Two-card persistent replicated-row all-reduce gate.
//!
//! Exercises the same 16 KiB hidden-row shape needed by TP2 replicated-residual decode: two
//! peer pushes, two reusable event waits, and the same `(rank0 + rank1)` add on both devices.

use memra_engine::tp::TpE4m3HostBounce;

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
    Ok(())
}
