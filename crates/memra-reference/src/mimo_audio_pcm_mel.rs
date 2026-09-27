//! Bounded mono F32 PCM to log-mel component for the pinned MiMo V2.6 codec.
//!
//! XiaomiMiMo/MiMo-Audio-Tokenizer at
//! b62b59922979bf9f389b373169298a251587653f, `utils.py::mel_spectrogram`,
//! uses torchaudio's MelSpectrogram with the separate V2.6 codec config.
//! The same frontend geometry is in the V2.6 `processor_config`. Torchaudio
//! 2.6 defaults supply a periodic Hann window, centered reflect padding,
//! unnormalized magnitude STFT, HTK triangular filters, and no area norm.
//! The publisher then applies `ln(max(mel, 1e-7))`.
//!
//! This accepts already resampled mono PCM. Audio decoding, channel mixing,
//! resampling, batching, and codec inference are separate operations.
//! The direct F64 DFT does not reproduce torch's F32 FFT roundoff in nearly
//! empty frequency bins. The public operation rejects that unstable region
//! rather than returning a large log-mel error.

use crate::ReferenceTensor;
use std::f64::consts::TAU;

pub const SAMPLE_RATE: u32 = 24_000;
pub const FFT_SIZE: usize = 960;
pub const HOP_LENGTH: usize = 240;
pub const MEL_BINS: usize = 128;
/// A CPU component bound, not the pinned V2.6 config's 300-second codec limit.
pub const MAX_COMPONENT_PCM_SAMPLES: usize = SAMPLE_RATE as usize;

const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1;
const MEL_FLOOR: f32 = 1e-7;
/// The empirically qualified mel-magnitude floor relative to PCM peak.
/// Non-silent input with any raw mel below this threshold is rejected.
pub const MIN_MEL_MAGNITUDE_PER_PCM_PEAK: f32 = 1e-3;

pub struct MiMoPcmMelFrontend {
    window: Vec<f32>,
    twiddles: Vec<(f64, f64)>,
    filters: Vec<f32>,
}

impl Default for MiMoPcmMelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MiMoPcmMelFrontend {
    pub fn new() -> Self {
        // torch.hann_window(960) in the pinned 2.6.0 CPU publisher oracle.
        // Its F32 cosine rounding differs from evaluating Hann in F64.
        let window = include_bytes!("fixtures/mimo-pcm-mel-hann960.f32")
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let twiddles = (0..SPECTRUM_BINS)
            .flat_map(|bin| {
                (0..FFT_SIZE).map(move |sample| {
                    let phase = TAU * bin as f64 * sample as f64 / FFT_SIZE as f64;
                    (phase.cos(), -phase.sin())
                })
            })
            .collect();
        // Exact F32 output of torchaudio 2.6 melscale_fbanks for this pinned
        // geometry, transposed to `[mel, FFT bin]` for the reference loop.
        let filters: Vec<f32> = include_bytes!("fixtures/mimo-pcm-mel-htk128x481.f32")
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        debug_assert_eq!(filters.len(), MEL_BINS * SPECTRUM_BINS);
        Self {
            window,
            twiddles,
            filters,
        }
    }

    /// Return row-major `[128, floor(samples / 240) + 1]` log-mel features.
    ///
    /// Centered reflect padding requires more than 480 input samples. The
    /// caller supplies the rate so a wrong-rate waveform cannot be silently
    /// interpreted as 24 kHz. This bounded component rejects longer clips.
    ///
    /// For non-silent PCM, every pre-log mel must be at least `1e-3` times
    /// the peak absolute PCM sample. Lower-energy bins amplify differences
    /// between native and torch F32 FFT rounding after the logarithm.
    /// Silence is exact because all magnitudes are zero and hit the source's
    /// `1e-7` floor. The cutoff is a fail-closed scope limit, not a universal
    /// proof of numerical equivalence for all possible waveforms.
    pub fn compute(&self, pcm: &[f32], sample_rate: u32) -> Result<ReferenceTensor, String> {
        self.compute_inner(pcm, sample_rate, true)
    }

    fn compute_inner(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        enforce_fidelity: bool,
    ) -> Result<ReferenceTensor, String> {
        if sample_rate != SAMPLE_RATE {
            return Err("MiMo PCM frontend requires 24 kHz mono input".into());
        }
        if pcm.len() <= FFT_SIZE / 2
            || pcm.len() > MAX_COMPONENT_PCM_SAMPLES
            || pcm.iter().any(|value| !value.is_finite())
        {
            return Err("MiMo PCM must be finite and contain 481..=24000 samples".into());
        }

        let frames = pcm.len() / HOP_LENGTH + 1;
        let peak = pcm
            .iter()
            .fold(0.0_f32, |maximum, &value| maximum.max(value.abs()));
        if peak == 0.0 {
            return ReferenceTensor::new(
                vec![MEL_BINS, frames],
                vec![MEL_FLOOR.ln(); MEL_BINS * frames],
            )
            .map_err(|error| error.to_string());
        }
        let fidelity_floor = MIN_MEL_MAGNITUDE_PER_PCM_PEAK * peak;
        let mut windowed = vec![0.0_f32; FFT_SIZE];
        let mut magnitudes = vec![0.0_f32; SPECTRUM_BINS];
        let mut output = vec![0.0_f32; MEL_BINS * frames];
        for frame in 0..frames {
            for (sample, value) in windowed.iter_mut().enumerate() {
                let position = frame * HOP_LENGTH + sample;
                let mut position = position as isize - (FFT_SIZE / 2) as isize;
                if position < 0 {
                    position = -position;
                }
                if position >= pcm.len() as isize {
                    position = 2 * pcm.len() as isize - 2 - position;
                }
                *value = pcm[position as usize] * self.window[sample];
            }
            for (bin, magnitude) in magnitudes.iter_mut().enumerate() {
                let mut real = 0.0_f64;
                let mut imag = 0.0_f64;
                for (&sample, &(cos, sin)) in windowed
                    .iter()
                    .zip(&self.twiddles[bin * FFT_SIZE..(bin + 1) * FFT_SIZE])
                {
                    real += f64::from(sample) * cos;
                    imag += f64::from(sample) * sin;
                }
                *magnitude = (real as f32).hypot(imag as f32);
            }
            for mel in 0..MEL_BINS {
                let mut energy = 0.0_f32;
                for (&weight, &magnitude) in self.filters
                    [mel * SPECTRUM_BINS..(mel + 1) * SPECTRUM_BINS]
                    .iter()
                    .zip(&magnitudes)
                {
                    energy += weight * magnitude;
                }
                if enforce_fidelity && energy < fidelity_floor {
                    return Err(format!(
                        "MiMo PCM mel magnitude below source-fidelity floor at mel {mel}, frame {frame}: {energy} < {fidelity_floor}"
                    ));
                }
                output[mel * frames + frame] = energy.max(MEL_FLOOR).ln();
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err("MiMo PCM frontend produced a non-finite mel value".into());
        }
        ReferenceTensor::new(vec![MEL_BINS, frames], output).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn matches_publisher_torchaudio_2_6_0_corpus() {
        let frontend = MiMoPcmMelFrontend::new();
        for (name, pcm_bytes, expected_bytes, frames) in [
            (
                "broadband-1680",
                include_bytes!("fixtures/mimo-pcm-mel-1680.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-1680.logmel.f32") as &[u8],
                8,
            ),
            (
                "broadband-721",
                include_bytes!("fixtures/mimo-pcm-mel-721.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-721.logmel.f32") as &[u8],
                4,
            ),
            (
                "silence",
                include_bytes!("fixtures/mimo-pcm-mel-silence-960.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-silence-960.logmel.f32") as &[u8],
                5,
            ),
            (
                "sparse",
                include_bytes!("fixtures/mimo-pcm-mel-sparse-1680.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-sparse-1680.logmel.f32") as &[u8],
                8,
            ),
            (
                "tone",
                include_bytes!("fixtures/mimo-pcm-mel-tone-1680.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-tone-1680.logmel.f32") as &[u8],
                8,
            ),
            (
                "noise",
                include_bytes!("fixtures/mimo-pcm-mel-noise-1537.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-noise-1537.logmel.f32") as &[u8],
                7,
            ),
            (
                "voiced",
                include_bytes!("fixtures/mimo-pcm-mel-voiced-2048.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-voiced-2048.logmel.f32") as &[u8],
                9,
            ),
            (
                "chirp",
                include_bytes!("fixtures/mimo-pcm-mel-chirp-1201.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-chirp-1201.logmel.f32") as &[u8],
                6,
            ),
            (
                "quiet-tone",
                include_bytes!("fixtures/mimo-pcm-mel-quiet-tone-1680.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-quiet-tone-1680.logmel.f32") as &[u8],
                8,
            ),
        ] {
            let pcm = floats(pcm_bytes);
            let expected = floats(expected_bytes);
            // Keep the raw numerical diagnostic for rejected cases, while
            // the public operation fails closed on unstable mel magnitudes.
            let output = frontend.compute_inner(&pcm, SAMPLE_RATE, false).unwrap();
            assert_eq!(output.shape, vec![MEL_BINS, frames]);
            assert_eq!(output.data.len(), expected.len());
            let (worst_index, worst) = output
                .data
                .iter()
                .zip(&expected)
                .enumerate()
                .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap();
            let max_relative = output
                .data
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs() / expected.abs().max(1e-6))
                .fold(0.0_f32, f32::max);
            println!(
                "case={name} samples={} max_abs_error={worst} max_relative_error={max_relative} mel={} frame={}",
                pcm.len(),
                worst_index / frames,
                worst_index % frames
            );
            if matches!(name, "sparse" | "tone" | "quiet-tone") {
                let error = frontend.compute(&pcm, SAMPLE_RATE).unwrap_err();
                assert!(error.contains("below source-fidelity floor"), "{error}");
            } else {
                let guarded = frontend.compute(&pcm, SAMPLE_RATE).unwrap();
                assert_eq!(guarded, output);
                assert!(
                    worst < 5e-5 && max_relative < 0.01,
                    "{name} max absolute {worst}, max relative {max_relative} at mel {}, frame {}: actual {}, expected {}",
                    worst_index / frames,
                    worst_index % frames,
                    output.data[worst_index],
                    expected[worst_index]
                );
            }
        }
    }

    #[test]
    fn silence_has_natural_log_floor_and_centered_frame_count() {
        let frontend = MiMoPcmMelFrontend::new();
        for samples in [481, 720, 721, 960] {
            let output = frontend.compute(&vec![0.0; samples], SAMPLE_RATE).unwrap();
            assert_eq!(output.shape, vec![MEL_BINS, samples / HOP_LENGTH + 1]);
            assert!(output.data.iter().all(|&value| value == MEL_FLOOR.ln()));
        }
    }

    #[test]
    fn fidelity_floor_tracks_pcm_amplitude() {
        let frontend = MiMoPcmMelFrontend::new();
        let tone = floats(include_bytes!("fixtures/mimo-pcm-mel-tone-1680.pcm.f32"));
        let noise = floats(include_bytes!("fixtures/mimo-pcm-mel-noise-1537.pcm.f32"));
        for scale in [0.1_f32, 1.0, 10.0] {
            let scaled_tone: Vec<f32> = tone.iter().map(|&sample| sample * scale).collect();
            let scaled_noise: Vec<f32> = noise.iter().map(|&sample| sample * scale).collect();
            assert!(
                frontend
                    .compute(&scaled_tone, SAMPLE_RATE)
                    .unwrap_err()
                    .contains("below source-fidelity floor")
            );
            assert!(frontend.compute(&scaled_noise, SAMPLE_RATE).is_ok());
        }
    }

    #[test]
    fn refuses_wrong_rate_invalid_pcm_and_unbounded_work() {
        let frontend = MiMoPcmMelFrontend::new();
        let good = vec![0.0; FFT_SIZE];
        assert!(frontend.compute(&good, 16_000).is_err());
        assert!(
            frontend
                .compute(&good[..FFT_SIZE / 2], SAMPLE_RATE)
                .is_err()
        );
        assert!(
            frontend
                .compute(&vec![0.0; MAX_COMPONENT_PCM_SAMPLES + 1], SAMPLE_RATE)
                .is_err()
        );
        let mut bad = good;
        bad[10] = f32::NAN;
        assert!(frontend.compute(&bad, SAMPLE_RATE).is_err());
    }
}
