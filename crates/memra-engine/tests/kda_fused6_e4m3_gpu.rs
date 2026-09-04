//! BIT-IDENTITY GATE for the e4m3 six-group KDA arm (lane/glm5-b200-mint-consume, 2026-09-04).
//!
//! `qmatvec_e4m3_mmvq_fused6` exists so the GLM-5.3-Flash B200 hybrid mint's six per-tensor e4m3
//! KDA projections run as ONE launch on ONE shared q8_1 activation instead of six launches with
//! six redundant activation quantizes. Its whole claim is that this costs nothing numerically:
//! per (range, row) the body is `e4m3_mmvq_row1`, the same body a separate m=1 launch runs, so
//! every output float must match BIT for bit — not approximately, exactly.
//!
//! A gate that only checked the fused path against itself would pass on a kernel that read the
//! wrong weight for every range, so each arm here is anchored on the SEPARATE per-tensor program
//! (`qmatvec_mmvq_into` at `QT_F8_E4M3`), and the last two arms are RED: they mutate the inputs
//! and assert the gate notices. Exactness only, no timing — this runs on the rig 5090.
use memra_engine::{Engine, QT_F8_E4M3};

/// Deterministic e4m3 code bytes. Every value stays inside the normal range (no 0x7F/0xFF NaN
/// magnitudes, which the loader refuses for this class anyway) so the kernel and the reference
/// decode the same finite numbers.
fn e4m3_plane(out_f: usize, in_f: usize, seed: u64) -> Vec<u8> {
    (0..out_f * in_f)
        .map(|i| {
            // The seed is MULTIPLIED into the hash, not added: an added seed differing only in
            // its low bits vanishes under the shift below, which silently makes every plane
            // identical -- caught by red arm 2 on 2026-09-04, which is what red arms are for.
            let h = (i as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
            let byte = (h >> 27) as u8;
            // Keep |exponent| moderate and away from the NaN magnitude 0x7F.
            let m = byte & 0x0F;
            let e = 0x30u8 + ((byte >> 4) & 0x07) * 8;
            let v = e | m;
            if v & 0x7F == 0x7F { v & 0xFE } else { v }
        })
        .collect()
}

fn activation(in_f: usize) -> Vec<f32> {
    (0..in_f)
        .map(|i| {
            let t = i as f32 * 0.017_3;
            (t.sin() * 1.7 + (t * 0.37).cos() * 0.6) * 0.25
        })
        .collect()
}

#[test]
fn e4m3_fused_six_is_bit_identical_to_six_separate_launches() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    // Unequal out_f across the group, exactly like the real KDA six (q/k/v wide, f_a/g_a/b thin),
    // so a kernel that assumed a uniform range width cannot pass.
    // in_f 4096 is the served KDA width AND the smallest size that exercises the ILP main loop
    // (nblk = in_f/32 = 128 > 32*(ILP-1) = 96); a smaller in_f would run only the serial tail and
    // the ILP arm below would pass without ever executing the code it is there to gate.
    let in_f = 4096usize;
    let dims = [128usize, 128, 128, 32, 32, 16];
    let ws = [1.0f32, 0.5, 2.25, 0.125, 3.0, 0.031_25];

    let planes: Vec<Vec<u8>> = dims
        .iter()
        .enumerate()
        .map(|(i, &o)| e4m3_plane(o, in_f, 0xC0FF_EE00 + i as u64))
        .collect();
    // Self-check: distinct planes are a PRECONDITION of the red arms below. Identical planes
    // would make "a different plane changes the output" untestable while still reading green.
    for i in 0..planes.len() {
        for j in (i + 1)..planes.len() {
            if dims[i] == dims[j] {
                assert_ne!(
                    planes[i], planes[j],
                    "planes {i} and {j} are byte-identical"
                );
            }
        }
    }
    let dev: Vec<_> = planes
        .iter()
        .map(|p| e.htod_bytes(p).expect("weight to device"))
        .collect();
    let x = e.htod(&activation(in_f)).expect("activation to device");

    // REFERENCE: six separate per-tensor launches on one shared activation quantize. This is the
    // program the door runs today when the fused arm declines.
    let (aq, ad) = e.quantize_q8_1(&x, 1, in_f).expect("q8_1");
    let mut want: Vec<Vec<f32>> = Vec::new();
    for (i, &o) in dims.iter().enumerate() {
        let mut y = e.uninit(o).expect("out");
        e.qmatvec_mmvq_into(
            &dev[i], &aq, &ad, 1, in_f, o, QT_F8_E4M3, in_f, ws[i], false, &mut y,
        )
        .expect("separate launch");
        want.push(e.dtoh(&y).expect("dtoh"));
    }

    let w6 = [&dev[0], &dev[1], &dev[2], &dev[3], &dev[4], &dev[5]];
    let got = e
        .qmatvec_e4m3_fused6_raw(w6, &x, in_f, dims, in_f, ws, 0)
        .expect("fused six");
    for (i, &o) in dims.iter().enumerate() {
        let g = e.dtoh(&got[i]).expect("dtoh");
        assert_eq!(g.len(), o);
        for (r, (a, b)) in g.iter().zip(want[i].iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "range {i} row {r}: fused {a} != separate {b} (bit identity is the contract)"
            );
        }
    }

    // RED ARM 1 — a per-range weight scale must actually reach its own range. Swapping two
    // scales has to change the output; if it does not, the kernel is ignoring `ws` and the
    // green arm above proved nothing about scale routing.
    let mut swapped = ws;
    swapped.swap(1, 4);
    let alt = e
        .qmatvec_e4m3_fused6_raw(w6, &x, in_f, dims, in_f, swapped, 0)
        .expect("fused six, swapped scales");
    let a1 = e.dtoh(&alt[1]).expect("dtoh");
    assert!(
        a1.iter()
            .zip(want[1].iter())
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "swapping two per-range scales left range 1 unchanged: the kernel is not applying ws"
    );

    // RED ARM 2 — each range must read ITS OWN weight plane. Feeding plane 0 in slot 2 has to
    // change range 2's output, which a kernel that mixed up its base pointers would fail.
    let w_mixed = [&dev[0], &dev[1], &dev[0], &dev[3], &dev[4], &dev[5]];
    let mixed = e
        .qmatvec_e4m3_fused6_raw(w_mixed, &x, in_f, dims, in_f, ws, 0)
        .expect("fused six, mixed planes");
    let m2 = e.dtoh(&mixed[2]).expect("dtoh");
    assert!(
        m2.iter()
            .zip(want[2].iter())
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "range 2 produced the same output from a different weight plane: pointers are not per-range"
    );
    // MEMRA_E4M3_ROW_ILP ARM. The ILP walk changes only WHEN loads issue: four blocks in flight
    // per lane, folded into `acc` in the same ascending order. It never touches a row's own
    // arithmetic, so it must match the serial walk BIT for bit, which also makes it match the six
    // separate per-tensor launches above. (The wider-block and staged-activation arms this loop
    // once carried were measured and removed; see the verdict in qmatvec.cu.)
    for arm in [1u32] {
        let got = e
            .qmatvec_e4m3_fused6_raw(w6, &x, in_f, dims, in_f, ws, arm)
            .unwrap_or_else(|err| panic!("fused six arm {arm}: {err}"));
        for (i, &o) in dims.iter().enumerate() {
            let g = e.dtoh(&got[i]).expect("dtoh");
            assert_eq!(g.len(), o);
            for (r, (a, b)) in g.iter().zip(want[i].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "arm {arm} range {i} row {r}: {a} != serial {b} \
                     (these arms move loads and memory spaces, never arithmetic)"
                );
            }
        }
    }
}
