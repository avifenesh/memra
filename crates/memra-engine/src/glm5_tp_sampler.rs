//! GLM TP-2 plain sampling on the root head stream.
//!
//! Reuses the DSV4 nucleus kernels and the existing two-pass raw argmax.
//! Sampled draws use DSV4's position-keyed SplitMix64(seed, absolute cache pos)
//! and device-f64-exp-tree-cdf-v1 arithmetic, including the first/restore row.
//! This deliberately differs from the host sampler's stateful f32 draw program.
//! Temperature zero or top-k one uses raw argmax, lowest token ID on ties.

use crate::Engine;
use crate::dsv4_gpu::Dsv4SampleCfg;
use crate::dsv4_sampler::Dsv4DeviceSampler;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use memra_sampling::SamplerConfig;
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn parse_requested(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("expected 0 or 1"),
    }
}

pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var_os("MEMRA_GLM5_TP_DEVICE_SAMPLE");
        let parsed = match &value {
            None => parse_requested(None),
            Some(v) => v
                .to_str()
                .ok_or("non-Unicode flag value")
                .and_then(|v| parse_requested(Some(v))),
        };
        parsed.unwrap_or_else(|reason| {
            eprintln!("[glm5-tp-device-sample] refused: {reason}; host fallback");
            false
        })
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Glm5TpSampleConfig {
    cfg: Dsv4SampleCfg,
    greedy: bool,
}

impl Glm5TpSampleConfig {
    /// A refusal leaves the request on its existing host path. `host_logits`
    /// covers constraints, logprobs and other consumers of the full host row.
    pub fn for_request(
        cfg: &SamplerConfig,
        glm_tp_plain: bool,
        host_logits: bool,
    ) -> Result<Self, &'static str> {
        if !glm_tp_plain {
            return Err("route is not GLM TP-2 plain");
        }
        if host_logits {
            return Err("request requires host logits (constraints, logprobs or capture)");
        }
        // Refuse non-neutral coefficients even with a zero history window.
        if cfg.penalty_repeat != 1.0 || cfg.penalty_freq != 0.0 || cfg.penalty_present != 0.0 {
            return Err("non-neutral penalties");
        }
        if cfg.min_p != 0.0 {
            return Err("min-p is not supported by the nucleus device sampler");
        }
        if !cfg.temperature.is_finite() || !(cfg.top_p > 0.0 && cfg.top_p <= 1.0) {
            return Err("invalid device temperature or top-p");
        }
        Ok(Self {
            cfg: Dsv4SampleCfg {
                temperature: cfg.temperature,
                top_p: cfg.top_p,
                top_k: cfg.top_k,
                seed: cfg.seed,
            },
            greedy: cfg.temperature <= 0.0 || cfg.top_k == 1,
        })
    }
}

enum Scratch {
    Greedy(CudaSlice<u32>),
    Sampled(Box<Dsv4DeviceSampler>),
}

pub struct Glm5TpDeviceSampler {
    stream: Arc<CudaStream>,
    n: usize,
    cfg: Glm5TpSampleConfig,
    scratch: Scratch,
    calls: u64,
}

impl Glm5TpDeviceSampler {
    pub fn new(e: &Engine, n: usize, cfg: Glm5TpSampleConfig) -> Res<Self> {
        if n == 0 || n > (1 << 24) {
            return Err("GLM device sampler vocabulary outside 1..=2^24".into());
        }
        let stream = e.stream();
        let scratch = if cfg.greedy {
            Scratch::Greedy(stream.alloc_zeros(1)?)
        } else {
            Scratch::Sampled(Box::new(Dsv4DeviceSampler::new(stream.clone(), n)?))
        };
        Ok(Self {
            stream,
            n,
            cfg,
            scratch,
            calls: 0,
        })
    }

    pub fn engagements(&self) -> u64 {
        self.calls
    }

    /// Prefill/restore currently returns host logits. Upload that boundary row
    /// once so cold and restored requests use the same device RNG program.
    pub fn sample_host_row(&mut self, e: &Engine, row: &[f32], pos: usize) -> Res<u32> {
        let logits = e.htod(row)?;
        self.sample(e, &logits, pos)
    }

    /// The producer retains `logits` on this exact root engine stream. Both
    /// arms read back only one u32 and drain that stream before worker emission.
    pub fn sample(&mut self, e: &Engine, logits: &CudaSlice<f32>, pos: usize) -> Res<u32> {
        if !Arc::ptr_eq(&self.stream, &e.stream())
            || !Arc::ptr_eq(&self.stream, logits.stream())
            || logits.len() != self.n
        {
            return Err("GLM device sampler stream or vocabulary mismatch".into());
        }
        let token = match &mut self.scratch {
            Scratch::Greedy(out) => {
                e.argmax_token_device_into(logits, out, self.n)?;
                let mut token = [0u32];
                self.stream.memcpy_dtoh(out, &mut token)?;
                self.stream.synchronize()?;
                token[0]
            }
            Scratch::Sampled(sampler) => {
                sampler.validate_source(&self.stream, self.n)?;
                let (ptr, _record) = logits.device_ptr(&self.stream);
                // SAFETY: borrowed allocation and producer share this stream,
                // and sample_ptr drains it before the borrow/record is released.
                unsafe { sampler.sample_ptr(ptr as *const f32, pos, &self.cfg.cfg, &[], None)? }
            }
        };
        if token as usize >= self.n {
            return Err("GLM device sampler returned an invalid token (nonfinite row)".into());
        }
        self.calls += 1;
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm5_tp_sample_args_and_refusals() {
        assert_eq!(parse_requested(None), Ok(false));
        assert_eq!(parse_requested(Some("0")), Ok(false));
        assert_eq!(parse_requested(Some("1")), Ok(true));
        assert!(parse_requested(Some("device")).is_err());
        let cfg = SamplerConfig {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            seed: 20260908,
            ..Default::default()
        };
        let parsed = Glm5TpSampleConfig::for_request(&cfg, true, false).unwrap();
        assert_eq!(parsed.cfg.temperature, cfg.temperature);
        assert_eq!(parsed.cfg.top_p, cfg.top_p);
        assert_eq!(parsed.cfg.top_k, cfg.top_k);
        assert_eq!(parsed.cfg.seed, cfg.seed);
        assert!(!parsed.greedy);
        assert!(Glm5TpSampleConfig::for_request(&cfg, false, false).is_err());
        assert!(Glm5TpSampleConfig::for_request(&cfg, true, true).is_err());
        for bad in [
            SamplerConfig {
                penalty_repeat: 1.1,
                ..cfg.clone()
            },
            SamplerConfig {
                penalty_freq: 0.1,
                ..cfg.clone()
            },
            SamplerConfig {
                penalty_present: 0.1,
                ..cfg.clone()
            },
            SamplerConfig {
                min_p: 0.1,
                ..cfg.clone()
            },
            SamplerConfig {
                temperature: f32::NAN,
                ..cfg.clone()
            },
            SamplerConfig {
                top_p: 0.0,
                ..cfg.clone()
            },
            SamplerConfig {
                top_p: f32::NAN,
                ..cfg.clone()
            },
            SamplerConfig {
                top_p: 1.1,
                ..cfg.clone()
            },
        ] {
            assert!(Glm5TpSampleConfig::for_request(&bad, true, false).is_err());
        }
        for greedy in [
            SamplerConfig {
                temperature: 0.0,
                ..cfg.clone()
            },
            SamplerConfig { top_k: 1, ..cfg },
        ] {
            assert!(
                Glm5TpSampleConfig::for_request(&greedy, true, false)
                    .unwrap()
                    .greedy
            );
        }
    }

    // Reproduce the actual host f32 filter order for the finite, distinct test row.
    fn host_nucleus(row: &[f32], cfg: &SamplerConfig) -> Vec<u32> {
        let inv = 1.0 / cfg.temperature;
        let mut p: Vec<_> = row
            .iter()
            .enumerate()
            .map(|(i, x)| (i as u32, x * inv))
            .collect();
        if cfg.top_k > 0 && cfg.top_k < p.len() {
            p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            p.truncate(cfg.top_k);
        }
        let max = p.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in &mut p {
            x.1 = (x.1 - max).exp();
            sum += x.1;
        }
        let inv_sum = 1.0 / sum;
        for x in &mut p {
            x.1 *= inv_sum;
        }
        p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut mass = 0.0f32;
        let mut ids = Vec::new();
        for (id, probability) in p {
            ids.push(id);
            mass += probability;
            if mass >= cfg.top_p {
                break;
            }
        }
        ids
    }

    #[test]
    #[ignore = "requires an exclusively owned CUDA box; never run on the local rig"]
    fn glm5_tp_device_sampler_gpu_oracles() -> Res<()> {
        let e = Engine::new(0)?;
        let mut state = 20260908u64;
        let mut row: Vec<f32> = (0..154_880)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 40) as f32 / (1u32 << 24) as f32) * 16.0 - 8.0
            })
            .collect();
        for tied in [false, true] {
            if tied {
                row[17] = 10.0;
                row[4097] = 10.0;
            }
            let host = memra_sampling::Sampler::new(SamplerConfig::default()).sample(&row);
            let logits = e.htod(&row)?;
            for cfg in [
                SamplerConfig::default(),
                SamplerConfig {
                    temperature: 1.0,
                    top_k: 1,
                    ..Default::default()
                },
            ] {
                let cfg = Glm5TpSampleConfig::for_request(&cfg, true, false)?;
                let mut device = Glm5TpDeviceSampler::new(&e, row.len(), cfg)?;
                assert_eq!(device.sample(&e, &logits, 23)?, host);
                assert_eq!(device.sample_host_row(&e, &row, 23)?, host);
                assert_eq!(device.engagements(), 2);
            }
        }
        for (temperature, top_p, top_k) in [(1.0, 0.95, 0), (0.7, 0.8, 40), (1.3, 0.5, 0)] {
            let cfg = SamplerConfig {
                temperature,
                top_p,
                top_k,
                seed: 20260908,
                ..Default::default()
            };
            let nucleus = host_nucleus(&row, &cfg);
            let parsed = Glm5TpSampleConfig::for_request(&cfg, true, false)?;
            let mut device = Glm5TpDeviceSampler::new(&e, row.len(), parsed)?;
            let logits = e.htod(&row)?;
            let mut draws = std::collections::HashSet::new();
            for pos in 0..256 {
                let id = device.sample(&e, &logits, pos)?;
                assert!(
                    nucleus.contains(&id),
                    "token {id} outside host nucleus at {pos}"
                );
                assert_eq!(
                    device.sample(&e, &logits, pos)?,
                    id,
                    "seed/position determinism"
                );
                if pos < 4 {
                    assert_eq!(
                        device.sample_host_row(&e, &row, pos)?,
                        id,
                        "prefill/restore adapter"
                    );
                }
                draws.insert(id);
            }
            assert!(draws.len() > 1, "sampled arm must not collapse to greedy");
            eprintln!(
                "GLM sampler oracle: T={temperature} top_p={top_p} top_k={top_k} nucleus={} distinct_draws={} engagements={}",
                nucleus.len(),
                draws.len(),
                device.engagements()
            );
        }
        Ok(())
    }
}
