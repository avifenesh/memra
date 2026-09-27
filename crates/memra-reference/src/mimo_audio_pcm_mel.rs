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
//! The direct F64 DFT follows the same STFT program but has different
//! roundoff from torch's F32 FFT in nearly empty frequency bins.

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
        let window = (0..FFT_SIZE)
            .map(|sample| (0.5 - 0.5 * (TAU * sample as f64 / FFT_SIZE as f64).cos()) as f32)
            .collect();
        let twiddles = (0..SPECTRUM_BINS)
            .flat_map(|bin| {
                (0..FFT_SIZE).map(move |sample| {
                    let phase = TAU * bin as f64 * sample as f64 / FFT_SIZE as f64;
                    (phase.cos(), -phase.sin())
                })
            })
            .collect();

        // torchaudio.functional.melscale_fbanks with mel_scale="htk",
        // norm=None, f_min=0, and f_max=sample_rate/2. The frequency bins
        // and mel points are F32, as in the default torch.linspace calls.
        let mel_max = (2595.0_f64 * (1.0_f64 + 12_000.0 / 700.0).log10()) as f32;
        let mel_step = mel_max / (MEL_BINS + 1) as f32;
        let points: Vec<f32> = (0..MEL_BINS + 2)
            .map(|index| 700.0 * (10.0_f32.powf(index as f32 * mel_step / 2595.0) - 1.0))
            .collect();
        let mut filters = Vec::with_capacity(MEL_BINS * SPECTRUM_BINS);
        for mel in 0..MEL_BINS {
            let (left, center, right) = (points[mel], points[mel + 1], points[mel + 2]);
            for bin in 0..SPECTRUM_BINS {
                let hz = bin as f32 * (SAMPLE_RATE as f32 / FFT_SIZE as f32);
                let rising = (hz - left) / (center - left);
                let falling = (right - hz) / (right - center);
                filters.push(rising.min(falling).max(0.0));
            }
        }
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
    pub fn compute(&self, pcm: &[f32], sample_rate: u32) -> Result<ReferenceTensor, String> {
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
    fn matches_publisher_torchaudio_2_6_0_fixture() {
        let frontend = MiMoPcmMelFrontend::new();
        for (pcm_bytes, expected_bytes, frames) in [
            (
                include_bytes!("fixtures/mimo-pcm-mel-1680.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-1680.logmel.f32") as &[u8],
                8,
            ),
            (
                include_bytes!("fixtures/mimo-pcm-mel-721.pcm.f32") as &[u8],
                include_bytes!("fixtures/mimo-pcm-mel-721.logmel.f32") as &[u8],
                4,
            ),
        ] {
            let pcm = floats(pcm_bytes);
            let expected = floats(expected_bytes);
            let output = frontend.compute(&pcm, SAMPLE_RATE).unwrap();
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
            println!(
                "samples={} max_abs_error={worst} mel={} frame={}",
                pcm.len(),
                worst_index / frames,
                worst_index % frames
            );
            assert!(
                worst < 5e-4,
                "maximum absolute log-mel error {worst} at mel {}, frame {}: actual {}, expected {}",
                worst_index / frames,
                worst_index % frames,
                output.data[worst_index],
                expected[worst_index]
            );
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
