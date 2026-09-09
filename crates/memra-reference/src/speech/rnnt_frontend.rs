//! Native log-mel frontend for the streaming RNNT path.
//!
//! This is not the Whisper frontend with different constants. It is a different program: a
//! 512-point transform over a 400-sample window, a pre-emphasis filter ahead of it, the
//! checkpoint's own window and filterbank buffers rather than recomputed ones, an additive log
//! guard instead of a dynamic-range clamp, and no normalization at all. Sharing code with the
//! Whisper frontend would mean sharing a program the two do not share.
//!
//! The window and the mel filterbank are bound from the checkpoint, not derived. They ship as
//! tensors, so deriving them would be a second opinion about a thing the artifact already
//! states.

use memra_gguf::model_packs::nemotron_rnnt::{BoundRnnt, RnntGeometry};

/// Samples between frames. 10 ms at 16 kHz.
pub const HOP: usize = 160;
/// Analysis window, 25 ms at 16 kHz, centred inside the transform.
pub const WINDOW: usize = 400;
/// Transform size. Not the window size: the window is zero-padded into it.
pub const FFT: usize = 512;
/// First-order pre-emphasis coefficient applied before the transform.
pub const PREEMPHASIS: f32 = 0.97;
/// Additive guard under the log, so a zero-power bin has a finite value.
pub const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24

pub struct RnntFrontend {
    geometry: RnntGeometry,
    /// Checkpoint window buffer, `WINDOW` samples.
    window: Vec<f32>,
    /// Checkpoint mel filterbank, `[mel_bins, fft_bins]`.
    filters: Vec<f32>,
    cosines: Vec<f32>,
    sines: Vec<f32>,
}

impl RnntFrontend {
    pub fn load(
        geometry: RnntGeometry,
        window: Vec<f32>,
        filters: Vec<f32>,
    ) -> Result<Self, String> {
        let bins = geometry.fft_bins as usize;
        if window.len() != WINDOW {
            return Err(format!("checkpoint window is {} samples", window.len()));
        }
        if filters.len() != geometry.mel_bins as usize * bins {
            return Err("checkpoint filterbank does not match the declared geometry".into());
        }
        if bins != FFT / 2 + 1 {
            return Err("frequency bin count does not match the transform size".into());
        }
        let mut cosines = vec![0.0f32; bins * FFT];
        let mut sines = vec![0.0f32; bins * FFT];
        for bin in 0..bins {
            for sample in 0..FFT {
                let angle =
                    -2.0 * std::f64::consts::PI * (bin as f64) * (sample as f64) / FFT as f64;
                cosines[bin * FFT + sample] = angle.cos() as f32;
                sines[bin * FFT + sample] = angle.sin() as f32;
            }
        }
        Ok(Self {
            geometry,
            window,
            filters,
            cosines,
            sines,
        })
    }

    /// Bind the frontend buffers out of an already-verified checkpoint.
    pub fn from_bound(
        geometry: RnntGeometry,
        bound: &BoundRnnt,
        archive: &[u8],
        storages: &std::collections::BTreeMap<String, (usize, usize)>,
    ) -> Result<Self, String> {
        let read = |name: &str| -> Result<Vec<f32>, String> {
            let entry = bound
                .tensors
                .get(name)
                .ok_or_else(|| format!("bound checkpoint is missing {name}"))?;
            let (at, len) = storages
                .get(&entry.storage_key)
                .ok_or_else(|| format!("{name} has no storage member"))?;
            Ok(archive[*at..*at + *len]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        Self::load(
            geometry,
            read("preprocessor.featurizer.window")?,
            read("preprocessor.featurizer.fb")?,
        )
    }

    /// Frames the reference declares valid for this many samples.
    ///
    /// The centred transform emits one more column than this; the reference masks it to the
    /// pad value, so it is a column of the output but not audio.
    pub fn valid_frames(&self, samples: usize) -> usize {
        samples / HOP
    }

    /// Log-mel for a whole waveform, `[mel_bins, frames]`, columns past the valid length
    /// filled with the reference's pad value.
    pub fn compute(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize), String> {
        if pcm.len() < HOP {
            return Err("waveform is shorter than one frame".into());
        }
        if pcm.iter().any(|x| !x.is_finite()) {
            return Err("waveform is not finite".into());
        }
        let bins = self.geometry.fft_bins as usize;
        let mel_bins = self.geometry.mel_bins as usize;

        // Pre-emphasis keeps the first sample and differences the rest.
        let mut emphasized = Vec::with_capacity(pcm.len());
        emphasized.push(pcm[0]);
        for i in 1..pcm.len() {
            emphasized.push(PREEMPHASIS.mul_add(-pcm[i - 1], pcm[i]));
        }

        // Centred analysis, padded with zeros rather than a reflection. The reference asks for
        // constant padding explicitly; reflecting instead moves the first two and the last
        // frame and nothing else, which is exactly the shape of that mistake.
        let pad = FFT / 2;
        let mut padded = vec![0.0f32; emphasized.len() + 2 * pad];
        padded[pad..pad + emphasized.len()].copy_from_slice(&emphasized);

        let frames = emphasized.len() / HOP + 1;
        let valid = self.valid_frames(pcm.len());
        let offset = (FFT - WINDOW) / 2;
        let mut power = vec![0.0f32; bins];
        let mut mel = vec![0.0f32; mel_bins * frames];
        let mut frame = vec![0.0f32; FFT];
        for t in 0..frames {
            if t >= valid {
                // Masked to the pad value; the reference writes zero here, not a log floor.
                for bin in 0..mel_bins {
                    mel[bin * frames + t] = 0.0;
                }
                continue;
            }
            frame.iter_mut().for_each(|x| *x = 0.0);
            for (i, value) in self.window.iter().enumerate() {
                frame[offset + i] = padded[t * HOP + offset + i] * value;
            }
            for (bin, slot) in power.iter_mut().enumerate() {
                let mut real = 0.0f32;
                let mut imaginary = 0.0f32;
                for (sample, &value) in frame.iter().enumerate() {
                    real = value.mul_add(self.cosines[bin * FFT + sample], real);
                    imaginary = value.mul_add(self.sines[bin * FFT + sample], imaginary);
                }
                // Magnitude then squared, which is what the reference's power spectrum is.
                let magnitude = (real * real + imaginary * imaginary).sqrt();
                *slot = magnitude * magnitude;
            }
            for bin in 0..mel_bins {
                let mut sum = 0.0f32;
                for (f, &value) in power.iter().enumerate() {
                    sum = self.filters[bin * bins + f].mul_add(value, sum);
                }
                mel[bin * frames + t] = (sum + LOG_GUARD).ln();
            }
        }
        Ok((mel, frames))
    }
}

#[cfg(test)]
mod tests;
