//! Whisper PCM to log-mel reference operation, with no external DSP/runtime dependency.
//! The direct real DFT is deliberately simple; it is the oracle for later FFT rewrites.

use crate::ReferenceTensor;
use memra_gguf::model_plan::speech::LogMelPlan;
use std::f64::consts::TAU;

pub struct WhisperFrontend {
    plan: LogMelPlan,
    window: Vec<f32>,
    twiddles: Vec<(f64, f64)>,
    filters: Vec<f32>,
}

impl WhisperFrontend {
    pub fn new(plan: &LogMelPlan) -> Result<Self, String> {
        if plan.sample_rate != 16000
            || plan.fft_size != 400
            || plan.hop_length != 160
            || ![80, 128].contains(&plan.mel_bins)
            || plan.max_samples == 0
            || plan.max_samples > 480000
            || !plan.max_samples.is_multiple_of(plan.hop_length)
            || plan.max_frames != plan.max_samples / plan.hop_length
            || !plan.periodic_hann
            || !plan.centered_reflect_padding
            || !plan.drop_last_stft_frame
            || !plan.slaney_mel_filters
            || plan.power != 2
            || plan.log10_floor != 1e-10
            || plan.dynamic_range != 8.0
            || plan.normalization_offset != 4.0
            || plan.normalization_divisor != 4.0
        {
            return Err("unsupported Whisper log-mel program".into());
        }
        // Reflect padding needs a source longer than n_fft/2, after right padding.
        if plan.max_samples <= plan.fft_size / 2 {
            return Err("Whisper padded waveform is too short for reflect padding".into());
        }
        let n = plan.fft_size as usize;
        let window = (0..n)
            .map(|i| (0.5 - 0.5 * (TAU * i as f64 / n as f64).cos()) as f32)
            .collect();
        let twiddles = (0..=n / 2)
            .flat_map(|bin| {
                (0..n).map(move |sample| {
                    let phase = TAU * bin as f64 * sample as f64 / n as f64;
                    (phase.cos(), -phase.sin())
                })
            })
            .collect();
        let mel_max = 15.0 + (8000.0_f64 / 1000.0).ln() * 27.0 / 6.4_f64.ln();
        let edges: Vec<f64> = (0..plan.mel_bins + 2)
            .map(|i| {
                let mel = mel_max * f64::from(i) / f64::from(plan.mel_bins + 1);
                if mel < 15.0 {
                    200.0 * mel / 3.0
                } else {
                    1000.0 * (6.4_f64.ln() * (mel - 15.0) / 27.0).exp()
                }
            })
            .collect();
        let mut filters = Vec::with_capacity(plan.mel_bins as usize * (n / 2 + 1));
        for mel in 0..plan.mel_bins as usize {
            let (left, center, right) = (edges[mel], edges[mel + 1], edges[mel + 2]);
            for bin in 0..=n / 2 {
                let hz = bin as f64 * f64::from(plan.sample_rate) / n as f64;
                filters.push(
                    (((hz - left) / (center - left))
                        .min((right - hz) / (right - center))
                        .max(0.0)
                        * 2.0
                        / (right - left)) as f32,
                );
            }
        }
        Ok(Self {
            plan: plan.clone(),
            window,
            twiddles,
            filters,
        })
    }

    /// Mono normalized PCM. Pad with zeroes BEFORE centered STFT, never reflect at the
    /// unpadded utterance boundary. Oversized/non-finite input is an error, not truncation.
    pub fn compute(&self, pcm: &[f32]) -> Result<ReferenceTensor, String> {
        if pcm.is_empty()
            || pcm.len() > self.plan.max_samples as usize
            || pcm.iter().any(|x| !x.is_finite())
        {
            return Err("Whisper PCM must be nonempty, finite and within the padded window".into());
        }
        self.features(
            pcm,
            self.plan.max_frames as usize,
            self.plan.max_samples as usize,
        )
    }

    /// Clip-scoped features, the shape faster-whisper actually decodes from. It builds one
    /// feature array per utterance, so both the centered reflection boundary and the
    /// dynamic-range clamp floor belong to the clip. Slicing 30-second windows out of this is
    /// not the same program as calling `compute` on each isolated window: the clamp floor
    /// moves and the frame at a window edge loses the audio that follows it.
    pub fn compute_clip(&self, pcm: &[f32]) -> Result<ReferenceTensor, String> {
        const MAX_CLIP_SAMPLES: usize = 16000 * 60 * 60;
        if pcm.is_empty() || pcm.len() > MAX_CLIP_SAMPLES || pcm.iter().any(|x| !x.is_finite()) {
            return Err("Whisper clip PCM must be nonempty, finite and at most one hour".into());
        }
        if pcm.len() <= self.plan.fft_size as usize / 2 {
            return Err("Whisper clip is too short for reflect padding".into());
        }
        let frames = pcm.len() / self.plan.hop_length as usize + 1;
        // The reference extractor reflects only at the start of the utterance. Past the last
        // sample it reads zeros, the same way `compute` treats the tail of a padded window.
        self.features(pcm, frames, usize::MAX)
    }

    fn features(
        &self,
        pcm: &[f32],
        frames: usize,
        reflect_len: usize,
    ) -> Result<ReferenceTensor, String> {
        let n = self.plan.fft_size as usize;
        let bins = n / 2 + 1;
        let mels = self.plan.mel_bins as usize;
        let mut windowed = vec![0.0f32; n];
        let mut power = vec![0.0f32; bins];
        let mut output = vec![0.0f32; mels * frames];
        let mut maximum = f32::NEG_INFINITY;
        for frame in 0..frames {
            for (j, value) in windowed.iter_mut().enumerate() {
                let mut index =
                    (frame * self.plan.hop_length as usize + j) as isize - (n / 2) as isize;
                if index < 0 {
                    index = -index;
                }
                if reflect_len != usize::MAX && index >= reflect_len as isize {
                    index = 2 * reflect_len as isize - 2 - index;
                }
                *value = pcm.get(index as usize).copied().unwrap_or(0.0) * self.window[j];
            }
            if windowed.iter().all(|&x| x == 0.0) {
                power.fill(0.0);
            } else {
                for (bin, magnitude) in power.iter_mut().enumerate() {
                    let mut real = 0.0f64;
                    let mut imag = 0.0f64;
                    for (&x, &(cos, sin)) in
                        windowed.iter().zip(&self.twiddles[bin * n..(bin + 1) * n])
                    {
                        real += f64::from(x) * cos;
                        imag += f64::from(x) * sin;
                    }
                    let (real, imag) = (real as f32, imag as f32);
                    *magnitude = real * real + imag * imag;
                }
            }
            for mel in 0..mels {
                let mut energy = 0.0f32;
                for (&weight, &value) in self.filters[mel * bins..(mel + 1) * bins]
                    .iter()
                    .zip(&power)
                {
                    energy += weight * value;
                }
                let log = energy.max(self.plan.log10_floor).log10();
                output[mel * frames + frame] = log;
                maximum = maximum.max(log);
            }
        }
        for value in &mut output {
            *value = (value.max(maximum - self.plan.dynamic_range)
                + self.plan.normalization_offset)
                / self.plan.normalization_divisor;
        }
        ReferenceTensor::new(vec![mels, frames], output).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(mel_bins: u32) -> LogMelPlan {
        LogMelPlan {
            sample_rate: 16000,
            fft_size: 400,
            hop_length: 160,
            mel_bins,
            max_samples: 32000,
            max_frames: 200,
            periodic_hann: true,
            centered_reflect_padding: true,
            drop_last_stft_frame: true,
            slaney_mel_filters: true,
            power: 2,
            log10_floor: 1e-10,
            dynamic_range: 8.0,
            normalization_offset: 4.0,
            normalization_divisor: 4.0,
        }
    }

    #[test]
    fn two_second_waveform_matches_the_pinned_hf_fp32_frontend() {
        fn floats(bytes: &[u8]) -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        }
        let pcm = floats(include_bytes!("fixtures/whisper-2s.pcm.f32"));
        let expected = floats(include_bytes!("fixtures/whisper-2s.mel.f32"));
        let mut full = plan(128);
        full.max_samples = 480000;
        full.max_frames = 3000;
        let got = WhisperFrontend::new(&full).unwrap().compute(&pcm).unwrap();
        assert_eq!(got.data.len(), expected.len());
        let max_abs = got
            .data
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-3, "mel max_abs={max_abs}, threshold=0.001");
    }

    /// Clip features are not window features stitched together. The rental oracle only agrees
    /// with the clip program: all 364 real windows match at 1.1920929e-7 when each window is
    /// sliced out of `compute_clip`, and the last window of a clip disagrees by up to 0.2507
    /// when the clip end reflects instead of reading zeros.
    #[test]
    fn clip_features_use_a_clip_wide_clamp_and_read_zeros_past_the_last_sample() {
        let mut full = plan(128);
        full.max_samples = 480000;
        full.max_frames = 3000;
        let frontend = WhisperFrontend::new(&full).unwrap();

        // A quiet tone inside the first window and a loud burst past it. The clip-wide maximum
        // then comes from audio the first window never sees, which is what moves the clamp.
        let mut pcm = vec![0.0f32; 480000 + 96000];
        for (i, x) in pcm[8000..16000].iter_mut().enumerate() {
            *x = ((i as f32) * 0.05).sin() * 0.002;
        }
        for (i, x) in pcm[500000..508000].iter_mut().enumerate() {
            *x = ((i as f32) * 0.05).sin() * 0.8;
        }
        let clip = frontend.compute_clip(&pcm).unwrap();
        assert_eq!(clip.shape, vec![128, pcm.len() / 160 + 1]);

        // The clamp floor is a property of the whole clip, so the quiet tail is held at the
        // clip-wide floor rather than being renormalized against its own local maximum.
        let frames = clip.shape[1];
        let floor = clip.data.iter().copied().fold(f32::INFINITY, f32::min);
        let tail: Vec<f32> = (0..128)
            .map(|m| clip.data[m * frames + frames - 2])
            .collect();
        assert!(
            tail.iter().all(|&x| (x - floor).abs() < 1e-6),
            "silent tail should sit on the clip floor {floor}, got {:?}",
            &tail[..4]
        );

        // Isolating the first window is a different program and must not be assumed equal.
        let window = frontend.compute(&pcm[..480000]).unwrap();
        let widest = (0..128 * 3000)
            .map(|i| {
                let (m, f) = (i / 3000, i % 3000);
                (window.data[m * 3000 + f] - clip.data[m * frames + f]).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            widest > 1e-3,
            "a window computed in isolation matched the clip slice at {widest}; if the reference \
             program changed, re-derive the rental-oracle contract before relaxing this"
        );

        // Past the final sample the clip reads zeros. Reflecting there moved the real oracle's
        // last window by up to 0.2507, so a silent run-out must stay on the floor.
        let silent = frontend.compute_clip(&vec![0.0; 32000]).unwrap();
        assert!(silent.data.iter().all(|&x| x == -1.5));
    }

    #[test]
    fn silence_has_the_analytic_floor_in_both_whisper_mel_geometries() {
        for mels in [80, 128] {
            let frontend = WhisperFrontend::new(&plan(mels)).unwrap();
            let output = frontend.compute(&[0.0; 1000]).unwrap();
            assert_eq!(output.shape, vec![mels as usize, 200]);
            assert!(output.data.iter().all(|&x| x == -1.5));
        }
    }

    #[test]
    fn padding_happens_before_reflection_and_invalid_audio_is_rejected() {
        let frontend = WhisperFrontend::new(&plan(128)).unwrap();
        let mut short = vec![0.0; 2000];
        short[1999] = 0.5;
        let padded = {
            let mut x = short.clone();
            x.resize(32000, 0.0);
            x
        };
        assert_eq!(
            frontend.compute(&short).unwrap(),
            frontend.compute(&padded).unwrap()
        );
        assert!(frontend.compute(&[]).is_err());
        assert!(frontend.compute(&[f32::NAN]).is_err());
        assert!(frontend.compute(&vec![0.0; 32001]).is_err());
        let mut wrong = plan(128);
        wrong.fft_size = 512;
        assert!(WhisperFrontend::new(&wrong).is_err());
    }
}
