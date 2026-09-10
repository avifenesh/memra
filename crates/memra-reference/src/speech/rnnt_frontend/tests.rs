//! Unit tests for the RNNT frontend's geometry and its refusals. Parity against the pinned
//! NeMo capture is `tools/check_rnnt_frontend.py`, which needs the checkpoint on the machine.

use super::*;
use memra_gguf::model_packs::nemotron_rnnt::HEBREW_GEOMETRY;

fn frontend() -> RnntFrontend {
    // Synthetic buffers: the geometry checks do not depend on their values, and the real ones
    // are bound from the checkpoint rather than derived anywhere.
    let window: Vec<f32> = (0..WINDOW)
        .map(|i| (i as f32 / WINDOW as f32).sin())
        .collect();
    let filters =
        vec![0.5f32; HEBREW_GEOMETRY.mel_bins as usize * HEBREW_GEOMETRY.fft_bins as usize];
    RnntFrontend::load(HEBREW_GEOMETRY, window, filters).unwrap()
}

#[test]
fn the_transform_is_five_hundred_twelve_over_a_four_hundred_sample_window() {
    // Whisper's frontend is a 400-point transform over a 400-sample window. This one is not,
    // and one shared implementation would have to be one or the other.
    assert_eq!(FFT, 512);
    assert_eq!(WINDOW, 400);
    assert_eq!(HOP, 160);
    assert_eq!(HEBREW_GEOMETRY.fft_bins as usize, FFT / 2 + 1);
    assert_eq!((FFT - WINDOW) / 2, 56);
}

#[test]
fn a_mismatched_window_or_filterbank_is_refused() {
    let filters =
        vec![0.0f32; HEBREW_GEOMETRY.mel_bins as usize * HEBREW_GEOMETRY.fft_bins as usize];
    assert!(RnntFrontend::load(HEBREW_GEOMETRY, vec![0.0; 399], filters).is_err());
    assert!(RnntFrontend::load(HEBREW_GEOMETRY, vec![0.0; WINDOW], vec![0.0; 10]).is_err());
}

#[test]
fn frames_past_the_valid_length_are_the_pad_value_not_a_log_floor() {
    let frontend = frontend();
    let pcm = vec![0.25f32; 16000];
    let (mel, frames) = frontend.compute(&pcm).unwrap();
    let valid = frontend.valid_frames(pcm.len());
    assert_eq!(valid, 100);
    assert_eq!(frames, 101);
    let bins = HEBREW_GEOMETRY.mel_bins as usize;
    assert_eq!(mel.len(), bins * frames);
    for bin in 0..bins {
        assert_eq!(mel[bin * frames + frames - 1], 0.0, "masked column");
    }
    assert!(mel.iter().all(|x| x.is_finite()));
}

#[test]
fn silence_reaches_the_log_guard_and_not_negative_infinity() {
    let frontend = frontend();
    let (mel, frames) = frontend.compute(&vec![0.0f32; 4000]).unwrap();
    assert!(mel.iter().all(|x| x.is_finite()));
    // log(0 + 2^-24) is the floor a silent bin lands on.
    let floor = LOG_GUARD.ln();
    assert!((mel[0] - floor).abs() < 1e-3, "{} vs {floor}", mel[0]);
    assert_eq!(frames, 26);
}

#[test]
fn a_short_or_non_finite_waveform_is_refused() {
    let frontend = frontend();
    assert!(frontend.compute(&[0.0; 10]).is_err());
    assert!(frontend.compute(&[f32::NAN; 4000]).is_err());
}
