//! A logits row with no finite value (every logit NaN or -inf) has no argmax. The device argmax
//! (`argmax_partial_f32` + `argmax_final_f32`) writes `TOKEN_NONE` for it, the host argmax gives
//! the same id, and `check_token` refuses it. Before this, the graph-replay chain fed that id
//! straight into the device embed gather: a read 2^31 rows past the table.
//!
//! Every kernel that indexes memory by a device token id now bounds it: the embed gathers
//! `embed_gather_u32` / `embed_gather_u32_t` (NaN row), the draft-confidence
//! `prob_of_token_partial_f32` (p = 0), the dspark `gather_row_bf16_f32` (NaN row) and the
//! FR-Spec `u32_map_k` (the id stays `TOKEN_NONE`), and the sampled-spec gathers
//! `softmax_gather_f32` / `softmax_gather_filtered_f32` (p = 0). The residual samplers' fallback
//! gives `TOKEN_NONE` for a row with no finite value instead of token 0. In-range ids keep their
//! bits.
//!
//! One test per kernel: on code without the bound, the first out-of-bounds read faults the
//! CUDA context, which would fail every later test in the same process. Run a single test with
//! `-- --exact NAME` to see one kernel's failure on its own.
use memra_engine::forward::argmax;
use memra_engine::{Engine, TOKEN_NONE, check_token, check_tokens};

const QT_F32: i32 = 8;
const N_VOCAB: usize = 151_936;

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 16.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn engine() -> Option<Engine> {
    match Engine::new(0) {
        Ok(e) => Some(e),
        Err(_) => {
            eprintln!("SKIP[CUDA device]: GPU test not run");
            None
        }
    }
}

/// Rows with no candidate: all NaN, all -inf, a mix.
fn none_rows() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("all NaN", vec![f32::NAN; N_VOCAB]),
        ("all -inf", vec![f32::NEG_INFINITY; N_VOCAB]),
        (
            "NaN and -inf",
            (0..N_VOCAB)
                .map(|i| {
                    if i % 3 == 0 {
                        f32::NAN
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect(),
        ),
    ]
}

/// A resident f32 embed table [rows, n_embd] and its host copy.
fn f32_table(e: &Engine, rows: usize, n_embd: usize) -> (cudarc::driver::CudaSlice<u8>, Vec<f32>) {
    let table = vecf(rows * n_embd, 21);
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    (e.upload_u8(&bytes).unwrap(), table)
}

#[test]
fn host_argmax_without_a_finite_logit_is_token_none() {
    for (name, row) in none_rows() {
        assert_eq!(argmax(&row) as u32, TOKEN_NONE, "{name}");
    }
    assert_eq!(argmax(&[]) as u32, TOKEN_NONE);
    assert_eq!(
        argmax(&[f32::NAN, f32::NEG_INFINITY, -f32::MAX, -f32::MAX]),
        2
    );
    assert_eq!(argmax(&[1.0, 3.0, f32::NAN, 3.0]), 1);
    assert_eq!(check_token(5, 6).unwrap(), 5);
    let outside = check_token(6, 6).unwrap_err().to_string();
    assert!(outside.contains("outside the vocabulary"), "{outside}");
    let none = check_token(TOKEN_NONE, 6).unwrap_err().to_string();
    assert!(none.contains("no finite logit"), "{none}");
    assert!(check_tokens(&[0, 1, TOKEN_NONE], 6).is_err());
    assert!(check_tokens(&[0, 1, 5], 6).is_ok());
}

#[test]
fn device_argmax_matches_host_and_gives_token_none_without_a_candidate() {
    let Some(e) = engine() else { return };
    let device_argmax = |row: &[f32]| -> u32 {
        let d = e.htod(row).unwrap();
        let tok = e.argmax_token_device(&d, row.len()).unwrap();
        e.dtoh_u32_one(&tok).unwrap()
    };
    for (name, row) in none_rows() {
        let dev = device_argmax(&row);
        assert_eq!(dev, TOKEN_NONE, "{name}: device argmax");
        assert!(check_token(dev, N_VOCAB).is_err(), "{name}");
    }
    // rows with a candidate: the host argmax's id, ties to the smallest index
    let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();
    for seed in 1..=4 {
        cases.push(("random", vecf(N_VOCAB, seed)));
    }
    let mut tie = vecf(N_VOCAB, 9);
    tie[100_001] = 99.0;
    tie[70_000] = 99.0;
    cases.push(("tie", tie));
    let mut one = vec![f32::NAN; N_VOCAB];
    one[123_456] = -3.0;
    cases.push(("one finite among NaN", one));
    let mut lowest = vec![f32::NEG_INFINITY; N_VOCAB];
    lowest[5] = -f32::MAX;
    lowest[150_000] = -f32::MAX;
    cases.push(("-FLT_MAX among -inf", lowest));
    let mut inf = vecf(N_VOCAB, 11);
    inf[42] = f32::NAN;
    inf[77_777] = f32::INFINITY;
    cases.push(("+inf with a NaN", inf));
    for (name, row) in &cases {
        let dev = device_argmax(row);
        assert_eq!(dev, argmax(row) as u32, "{name}: device vs host argmax");
        assert_eq!(check_token(dev, N_VOCAB).unwrap(), dev, "{name}");
    }
}

#[test]
fn embed_gather_of_an_id_past_the_table_is_a_nan_row() {
    let Some(e) = engine() else { return };
    let (n_rows, n_embd) = (1000usize, 96usize);
    let (embd, table) = f32_table(&e, n_rows, n_embd);
    let row_bytes = n_embd * 4;
    for tok in [
        0u32,
        7,
        (n_rows - 1) as u32,
        n_rows as u32,
        TOKEN_NONE,
        u32::MAX,
    ] {
        let tok_d = e.htod_u32_v(&[tok]).unwrap();
        let x = e
            .embed_gather_device(&embd, &tok_d, n_embd, QT_F32, row_bytes)
            .unwrap();
        let got = e.dtoh(&x).unwrap();
        if (tok as usize) < n_rows {
            let t = tok as usize;
            assert_eq!(
                bits(&got),
                bits(&table[t * n_embd..(t + 1) * n_embd]),
                "gather of {tok}"
            );
        } else {
            assert!(
                got.iter().all(|v| v.is_nan()),
                "gather of {tok}: not a NaN row"
            );
        }
    }
    // the device argmax's own buffer, as graph replay chains it
    let none_d = e.htod(&vec![f32::NAN; N_VOCAB]).unwrap();
    let tok_d = e.argmax_token_device(&none_d, N_VOCAB).unwrap();
    let x = e
        .embed_gather_device(&embd, &tok_d, n_embd, QT_F32, row_bytes)
        .unwrap();
    assert!(e.dtoh(&x).unwrap().iter().all(|v| v.is_nan()));
    e.stream().synchronize().unwrap();
}

#[test]
fn embed_gather_rows_keeps_in_table_rows_and_nans_the_rest() {
    let Some(e) = engine() else { return };
    let (n_rows, n_embd) = (1000usize, 96usize);
    let (embd, table) = f32_table(&e, n_rows, n_embd);
    let row_bytes = n_embd * 4;
    let toks = [3u32, TOKEN_NONE, 999, n_rows as u32, 0];
    let toks_d = e.htod_u32_v(&toks).unwrap();
    let mut out = e.uninit(toks.len() * n_embd).unwrap();
    e.embed_gather_rows(
        &embd,
        &toks_d.slice(0..toks.len()),
        &mut out.slice_mut(0..toks.len() * n_embd),
        toks.len(),
        n_embd,
        QT_F32,
        row_bytes,
    )
    .unwrap();
    let got = e.dtoh(&out).unwrap();
    for (i, &tok) in toks.iter().enumerate() {
        let row = &got[i * n_embd..(i + 1) * n_embd];
        if (tok as usize) < n_rows {
            let t = tok as usize;
            assert_eq!(
                bits(row),
                bits(&table[t * n_embd..(t + 1) * n_embd]),
                "row {i} (token {tok})"
            );
        } else {
            assert!(
                row.iter().all(|v| v.is_nan()),
                "row {i} (token {tok}): not a NaN row"
            );
        }
    }
}

#[test]
fn prob_of_an_id_past_the_row_is_zero() {
    let Some(e) = engine() else { return };
    let row = vecf(N_VOCAB, 31);
    let row_d = e.htod(&row).unwrap();
    for (tok, finite) in [
        (argmax(&row) as u32, true),
        (TOKEN_NONE, false),
        (N_VOCAB as u32, false),
    ] {
        let tok_d = e.htod_u32_v(&[tok]).unwrap();
        let p = e
            .dtoh(&e.prob_of_token_device(&row_d, &tok_d, N_VOCAB).unwrap())
            .unwrap()[0];
        if finite {
            assert!(p > 0.0 && p <= 1.0, "p({tok}) = {p}");
        } else {
            assert_eq!(p, 0.0, "p({tok}) = {p}, want 0");
        }
    }
}

#[test]
fn trim_map_keeps_an_id_past_the_map_as_token_none() {
    let Some(e) = engine() else { return };
    let map: Vec<u32> = (0..100u32).map(|i| i * 3).collect();
    let map_d = e.htod_u32_v(&map).unwrap();
    let mut buf = e.htod_u32_v(&[5, TOKEN_NONE, 100, 99]).unwrap();
    for idx in 0..4 {
        e.u32_map_k(&mut buf, &map_d, idx).unwrap();
    }
    assert_eq!(
        e.dtoh_u32(&buf).unwrap(),
        vec![15, TOKEN_NONE, TOKEN_NONE, 297]
    );
}

#[test]
fn bf16_row_gather_of_an_id_past_the_table_is_a_nan_row() {
    let Some(e) = engine() else { return };
    let (rows, ncols) = (50usize, 64usize);
    let tbf: Vec<u16> = vecf(rows * ncols, 41)
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let tbytes: Vec<u8> = tbf.iter().flat_map(|v| v.to_le_bytes()).collect();
    let table_d = e.upload_u8(&tbytes).unwrap();
    let toks_d = e.htod_u32_v(&[7, TOKEN_NONE, rows as u32]).unwrap();
    for (idx, want) in [(0usize, Some(7usize)), (1, None), (2, None)] {
        let mut dst = e.uninit(ncols).unwrap();
        e.gather_row_bf16(&table_d, &toks_d, idx, &mut dst, ncols)
            .unwrap();
        let got = e.dtoh(&dst).unwrap();
        match want {
            Some(r) => {
                let host: Vec<u32> = tbf[r * ncols..(r + 1) * ncols]
                    .iter()
                    .map(|&h| (h as u32) << 16)
                    .collect();
                assert_eq!(bits(&got), host, "bf16 gather of row {r}");
            }
            None => assert!(
                got.iter().all(|v| v.is_nan()),
                "bf16 gather slot {idx}: not a NaN row"
            ),
        }
    }
}

#[test]
fn sampled_spec_gather_gives_an_id_past_the_row_probability_zero() {
    let Some(e) = engine() else { return };
    let n = 1000usize;
    let x_d = e.htod(&vecf(n, 51)).unwrap();
    let ids_d = e.htod_u32_v(&[5, TOKEN_NONE, n as u32]).unwrap();
    let rows_d = e.htod_i32(&[0, 0, 0]).unwrap();
    let mut out = e.zeros(3).unwrap();
    e.softmax_gather(&x_d, n, &ids_d, &rows_d, &mut out, n, 3, 1.0)
        .unwrap();
    let p = e.dtoh(&out).unwrap();
    assert!(p[0] > 0.0 && p[0] < 1.0, "p(5) = {}", p[0]);
    assert_eq!(&p[1..], &[0.0, 0.0], "p past the row");
    let th_d = e.htod(&[0.0f32; 3]).unwrap();
    let z_d = e.htod(&[1.0f32; 3]).unwrap();
    let mut outf = e.zeros(3).unwrap();
    e.softmax_gather_filtered(&x_d, n, &ids_d, &rows_d, &th_d, &z_d, &mut outf, n, 3, 1.0)
        .unwrap();
    let pf = e.dtoh(&outf).unwrap();
    assert!(pf[0] > 0.0, "filtered p(5) = {}", pf[0]);
    assert_eq!(&pf[1..], &[0.0, 0.0], "filtered p past the row");
}

#[test]
fn residual_sample_of_a_row_without_a_finite_value_is_token_none() {
    let Some(e) = engine() else { return };
    let n = 4096usize;
    for (name, row) in [
        ("all NaN", vec![f32::NAN; n]),
        ("all -inf", vec![f32::NEG_INFINITY; n]),
    ] {
        let p_d = e.htod(&row).unwrap();
        let mut tok = e.alloc_u32_zeroed(1).unwrap();
        e.residual_sample(&p_d, None, n, 1.0, 7, 0, &mut tok)
            .unwrap();
        assert_eq!(
            e.dtoh_u32(&tok).unwrap()[0],
            TOKEN_NONE,
            "{name}: residual_sample"
        );
        let mut tokf = e.alloc_u32_zeroed(1).unwrap();
        e.residual_sample_filtered(
            &p_d,
            None,
            n,
            1.0,
            7,
            0,
            (-3.4e38, 0.0, 0.0),
            (-3.4e38, 0.0, 0.0),
            &mut tokf,
        )
        .unwrap();
        assert_eq!(
            e.dtoh_u32(&tokf).unwrap()[0],
            TOKEN_NONE,
            "{name}: residual_sample_filtered"
        );
    }
    // a finite row still samples an id in the row
    let p_d = e.htod(&vecf(n, 61)).unwrap();
    let mut tok = e.alloc_u32_zeroed(1).unwrap();
    e.residual_sample(&p_d, None, n, 1.0, 7, 0, &mut tok)
        .unwrap();
    assert!((e.dtoh_u32(&tok).unwrap()[0] as usize) < n);
}
