//! Step TP DEVICE-RESIDENT activation gate (lane/hermes-perf-fixes, 2026-08-23).
//!
//! THE FINDING THIS GATES: the full-attention TP QKV/O seams used to DtoH the full activation,
//! run the projection from a host copy, gather to host vectors and re-upload — a host
//! round-trip per layer per step, on the very path native P2P exists to remove. The lane
//! added device-input twins (`bf16_column_parallel_resident_native_device`,
//! `step_bf16_row_parallel_resident_native_device`). Their claim is BYTE-IDENTITY to the
//! host-canonical arms, and this gate is that claim's teeth: same synthetic BF16 weights,
//! same activation bytes, host arm vs device arm, ANY bit mismatch = FAIL.
//!
//! Deliberately artifact-free (synthetic weights, deterministic): the numeric claim is about
//! the TRANSPORT of bytes across ranks, not about any checkpoint's values, so a 300 GB
//! official Step download proves nothing extra here. The official-checkpoint shape gate
//! (`tp-step-bf16-gate`) remains the artifact-backed oracle for the projection itself.
//!
//! usage: tp-step-resident-device-gate            (MEMRA_TP_DEVICES=0,1 default)
//!        MEMRA_TP_TOKENS=<n> MEMRA_TP_REPS=<n> tp-step-resident-device-gate
//!
//! Emits per-arm wall time as a SECONDARY reading (host round-trip removed vs kept) with the
//! interleaved protocol: reps alternate host/device so box clock drift cannot fake a win.

use memra_engine::tp::{Bf16Matrix, TpE4m3HostBounce};

fn devices() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_TP_DEVICES").unwrap_or_else(|_| "0,1".to_string());
    let devices = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    if !(2..=8).contains(&devices.len()) {
        return Err(format!("device-resident TP gate needs 2..=8 ranks, got {raw:?}").into());
    }
    Ok(devices)
}

/// Deterministic BF16 weight bytes (exact bf16 values — the low 16 bits are the payload).
fn bf16_weights(out_features: usize, in_features: usize, salt: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(out_features * in_features * 2);
    for index in 0..out_features * in_features {
        let mixed = (index as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(salt.wrapping_mul(97));
        // small, exactly representable bf16 magnitudes around +-1
        let value = ((mixed % 4096) as f32 - 2048.0) / 2048.0;
        let hi = (value.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&hi.to_le_bytes());
    }
    bytes
}

fn activations(tokens: usize, width: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|index| {
            let mixed = index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((mixed % 8191) as f32 - 4095.0) / 2048.0
        })
        .collect()
}

fn bitdiff(a: &[f32], b: &[f32]) -> usize {
    if a.len() != b.len() {
        return a.len().abs_diff(b.len()).max(1);
    }
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = devices()?;
    let tokens: usize = std::env::var("MEMRA_TP_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let reps: usize = std::env::var("MEMRA_TP_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // Step-3.7-Flash attention geometry class: hidden 7168, q 7168, kv 1024 (divisible by
    // the TP8 product envelope, which the canonical row/column chunking requires).
    let hidden = 7168usize;
    let q_out = 7168usize;
    let kv_out = 1024usize;

    let runtime = TpE4m3HostBounce::new_native_p2p(&devices)?;
    let names = runtime.device_names()?;
    println!(
        "tp-step-resident-device-gate: devices={devices:?} names={names:?} \
         transport={} native_p2p={} bulk_p2p={} tokens={tokens} reps={reps}",
        runtime.transport_label(),
        runtime.native_p2p(),
        runtime.bulk_p2p(),
    );

    let q_bytes = bf16_weights(q_out, hidden, 1);
    let k_bytes = bf16_weights(kv_out, hidden, 2);
    let v_bytes = bf16_weights(kv_out, hidden, 3);
    let o_bytes = bf16_weights(hidden, q_out, 4);
    fn mk(bytes: &[u8], out_features: usize, in_features: usize) -> Bf16Matrix<'_> {
        Bf16Matrix {
            bytes,
            out_features,
            in_features,
        }
    }

    let q = runtime.upload_step_bf16_column_parallel(mk(&q_bytes, q_out, hidden))?;
    let k = runtime.upload_step_bf16_column_parallel(mk(&k_bytes, kv_out, hidden))?;
    let v = runtime.upload_step_bf16_column_parallel(mk(&v_bytes, kv_out, hidden))?;
    let o = runtime.upload_step_bf16_row_parallel(mk(&o_bytes, hidden, q_out))?;

    let input = activations(tokens, hidden);
    let attn = activations(tokens, q_out);
    let root = runtime
        .rank_engine(0)
        .ok_or("TP runtime has no root rank")?;
    let input_dev = root.htod(&input)?;
    let attn_dev = root.htod(&attn)?;
    root.stream().synchronize()?;

    let mut fails = 0usize;
    let mut host_ms = Vec::new();
    let mut dev_ms = Vec::new();

    for rep in 0..reps {
        // INTERLEAVED (box-drift law): host arm then device arm, every rep.
        let t0 = std::time::Instant::now();
        let hq = runtime.bf16_column_parallel_resident_native(&q, &input, tokens)?;
        let hk = runtime.bf16_column_parallel_resident_native(&k, &input, tokens)?;
        let hv = runtime.bf16_column_parallel_resident_native(&v, &input, tokens)?;
        let ho = runtime.step_bf16_row_parallel_resident_native(&o, &attn, tokens)?;
        host_ms.push(t0.elapsed().as_secs_f64() * 1e3);

        let t1 = std::time::Instant::now();
        let dq = runtime.bf16_column_parallel_resident_native_device(&q, &input_dev, tokens)?;
        let dk = runtime.bf16_column_parallel_resident_native_device(&k, &input_dev, tokens)?;
        let dv = runtime.bf16_column_parallel_resident_native_device(&v, &input_dev, tokens)?;
        let d_o = runtime.step_bf16_row_parallel_resident_native_device(&o, &attn_dev, tokens)?;
        dev_ms.push(t1.elapsed().as_secs_f64() * 1e3);

        let (dqh, dkh, dvh, doh) = (
            root.dtoh(&dq)?,
            root.dtoh(&dk)?,
            root.dtoh(&dv)?,
            root.dtoh(&d_o)?,
        );
        for (label, host, device) in [
            ("q", &hq, &dqh),
            ("k", &hk, &dkh),
            ("v", &hv, &dvh),
            ("o", &ho, &doh),
        ] {
            let n = bitdiff(host, device);
            if n != 0 {
                fails += 1;
            }
            if rep == 0 || n != 0 {
                println!(
                    "  rep{rep} {label}: bitdiff={n}/{} {}",
                    host.len(),
                    if n == 0 {
                        "OK (byte-identical)"
                    } else {
                        "FAIL"
                    },
                );
            }
        }
    }

    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (hm, dm) = (median(host_ms.clone()), median(dev_ms.clone()));
    println!("host-canonical  median {hm:.3} ms/qkvo  samples {host_ms:?}");
    println!("device-resident median {dm:.3} ms/qkvo  samples {dev_ms:?}");
    println!(
        "wall: device/host = {:.4}x ({:+.1}%) — SECONDARY reading, interleaved x{reps}",
        dm / hm,
        (hm - dm) / hm * 100.0,
    );
    println!(
        "verdict: {}",
        if fails == 0 {
            "GREEN — device-resident twins are BYTE-IDENTICAL to the host-canonical arms"
        } else {
            "*** RED: device-resident output differs from the host-canonical oracle ***"
        }
    );
    if fails != 0 {
        std::process::exit(1);
    }
    Ok(())
}
