//! Magnitude census of the Gated DeltaNet recurrent state (research/gdn-state-fp16-20261107).
//!
//! Diagnostic only. `MEMRA_GDN_STATE_CENSUS=<dir>` makes memra-server copy each live session's GDN
//! states (`RecurLayer::ssm_state`, fp32) to the host at a cadence and write, per layer and per
//! value head, what fp16 and bf16 storage would do to them: the largest magnitude against fp16's
//! 65504, the elements in fp16's subnormal range (below 2^-14) and those that would flush to zero,
//! the energy those small elements carry, and the relative L2 error of one fp16 and one bf16
//! rounding of the whole head. Nothing in the numeric program changes: the copy reads the state
//! between ticks and the statistics run on the host.
//!
//! Cadence: a session is sampled the first time it is seen, at most once per
//! `MEMRA_GDN_STATE_CENSUS_PRIME_SECS` (default 10) while it primes, on its first decode tick, and
//! then every `MEMRA_GDN_STATE_CENSUS_DECODE_STRIDE` (default 128) committed rows. Output:
//! `<dir>/census.jsonl` (one row per sample, columns in `<dir>/columns.json`) and `<dir>/agg.json`
//! (per layer and head extremes over every sample, rewritten after each sample).

use memra_engine::Engine;
use memra_engine::cache::Cache;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// d_state = d_v = 128: one head's state is 128 x 128 contiguous f32 (`gdn_scan_s128`).
const HEAD: usize = 128 * 128;
/// fp16's smallest normal, 2^-14.
const F16_MIN_NORMAL: f32 = 6.103_515_6e-5;
/// fp16 subnormals are the multiples of 2^-24.
const F16_SUB_SCALE: f32 = 16_777_216.0;
/// log2 histogram: bucket 0 holds exponents <= HIST_LO, the last >= HIST_LO + HIST_N - 1.
const HIST_LO: i32 = -40;
const HIST_N: usize = 58;

/// `x` rounded to the nearest fp16 value (ties to even), as f32; +-inf past fp16's range
/// (|x| >= 65520), subnormals below 2^-14, zero at or below 2^-25.
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    if x.abs() < F16_MIN_NORMAL {
        return (x * F16_SUB_SCALE).round_ties_even() / F16_SUB_SCALE;
    }
    let b = x.to_bits();
    let r = f32::from_bits((b + 0x0fff + ((b >> 13) & 1)) & !0x1fff);
    if r.abs() >= 65_536.0 {
        f32::INFINITY.copysign(x)
    } else {
        r
    }
}

/// `x` rounded to bf16 (ties to even), as f32: `__float2bfloat16_rn` on finite values.
pub fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

#[derive(Clone)]
struct HeadStats {
    max_abs: f32,
    nonfinite: u32,
    zero: u32,
    sub: u32,
    ftz: u32,
    ovf: u32,
    e_tot: f64,
    e_sub: f64,
    e_r16: f64,
    e_rb16: f64,
    hist: [u32; HIST_N],
}

fn head_stats(xs: &[f32]) -> HeadStats {
    let mut s = HeadStats {
        max_abs: 0.0,
        nonfinite: 0,
        zero: 0,
        sub: 0,
        ftz: 0,
        ovf: 0,
        e_tot: 0.0,
        e_sub: 0.0,
        e_r16: 0.0,
        e_rb16: 0.0,
        hist: [0; HIST_N],
    };
    for &x in xs {
        if !x.is_finite() {
            s.nonfinite += 1;
            continue;
        }
        let a = x.abs();
        if a == 0.0 {
            s.zero += 1;
            continue;
        }
        s.max_abs = s.max_abs.max(a);
        let x2 = f64::from(x) * f64::from(x);
        s.e_tot += x2;
        if a < F16_MIN_NORMAL {
            s.sub += 1;
            s.e_sub += x2;
        }
        let h = round_f16(x);
        if h.is_infinite() {
            s.ovf += 1;
        } else {
            if h == 0.0 {
                s.ftz += 1;
            }
            let d = f64::from(h) - f64::from(x);
            s.e_r16 += d * d;
        }
        let d = f64::from(round_bf16(x)) - f64::from(x);
        s.e_rb16 += d * d;
        let exp = ((x.to_bits() >> 23) & 0xff) as i32 - 127;
        let bucket = (exp.clamp(HIST_LO, HIST_LO + HIST_N as i32 - 1) - HIST_LO) as usize;
        s.hist[bucket] += 1;
    }
    s
}

fn rel(err: f64, tot: f64) -> f64 {
    if tot > 0.0 { (err / tot).sqrt() } else { 0.0 }
}

/// Per (layer, head) extremes over every sample of the process.
#[derive(Clone)]
struct HeadAgg {
    max_abs: f32,
    max_r16: f64,
    max_rb16: f64,
    max_sub_frac: f64,
    max_esub: f64,
    min_rms: f64,
    samples: u64,
}

impl Default for HeadAgg {
    fn default() -> Self {
        Self {
            max_abs: 0.0,
            max_r16: 0.0,
            max_rb16: 0.0,
            max_sub_frac: 0.0,
            max_esub: 0.0,
            min_rms: f64::INFINITY,
            samples: 0,
        }
    }
}

struct LayerAgg {
    layer: usize,
    heads: Vec<HeadAgg>,
    hist: [u64; HIST_N],
    zero: u64,
    nonfinite: u64,
    ovf: u64,
    ftz: u64,
    sub: u64,
    elems: u64,
}

/// Where a sampled session stands.
pub struct At<'a> {
    pub id: &'a str,
    pub decoding: bool,
    pub rows: usize,
    pub n_prompt: usize,
    pub generated: usize,
}

struct Seen {
    decoding: bool,
    rows: usize,
    at: Instant,
}

/// One census per server worker (`Census::from_env` once at worker start).
pub struct Census {
    dir: PathBuf,
    out: std::io::BufWriter<std::fs::File>,
    prime_every: Duration,
    decode_stride: usize,
    seen: HashMap<String, Seen>,
    agg: Vec<LayerAgg>,
    samples: u64,
    started: Instant,
}

/// Columns of one layer's entry in a `census.jsonl` row.
const COLUMNS: [&str; 15] = [
    "layer",
    "max_abs",
    "max_abs_head",
    "sub_frac",
    "ftz_frac",
    "zero_frac",
    "ovf",
    "nonfinite",
    "esub_share",
    "r16_layer",
    "r16_worst_head",
    "r16_worst_head_id",
    "rb16_layer",
    "rb16_worst_head",
    "min_head_rms",
];

impl Census {
    /// `Some` when `MEMRA_GDN_STATE_CENSUS` names a directory (created if missing).
    pub fn from_env() -> Option<Census> {
        let dir = PathBuf::from(std::env::var_os("MEMRA_GDN_STATE_CENSUS")?);
        let num = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(d)
        };
        let prime_every = Duration::from_secs_f64(num("MEMRA_GDN_STATE_CENSUS_PRIME_SECS", 10.0));
        let decode_stride = num("MEMRA_GDN_STATE_CENSUS_DECODE_STRIDE", 128.0).max(1.0) as usize;
        let open = || -> std::io::Result<std::fs::File> {
            std::fs::create_dir_all(&dir)?;
            let cols = serde_json::json!({
                "row": ["t_s", "id", "phase", "rows", "n_prompt", "generated", "layers"],
                "layer": COLUMNS,
                "hist": {"lo_exp": HIST_LO, "buckets": HIST_N},
                "thresholds": {"f16_min_normal": F16_MIN_NORMAL, "f16_max": 65504.0},
            });
            std::fs::write(dir.join("columns.json"), cols.to_string())?;
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("census.jsonl"))
        };
        match open() {
            Ok(f) => {
                eprintln!(
                    "[gdn-census] on: dir={} prime_every={:.1}s decode_stride={decode_stride} \
                     (diagnostic: copies GDN states to the host between ticks)",
                    dir.display(),
                    prime_every.as_secs_f64()
                );
                Some(Census {
                    dir,
                    out: std::io::BufWriter::new(f),
                    prime_every,
                    decode_stride,
                    seen: HashMap::new(),
                    agg: Vec::new(),
                    samples: 0,
                    started: Instant::now(),
                })
            }
            Err(err) => {
                eprintln!("[gdn-census] off: cannot open {}: {err}", dir.display());
                None
            }
        }
    }

    /// Whether the session `id` at `rows` rows is due a sample now.
    pub fn due(&self, id: &str, decoding: bool, rows: usize) -> bool {
        match self.seen.get(id) {
            None => rows > 0,
            Some(s) if decoding != s.decoding => true,
            Some(s) if decoding => rows >= s.rows + self.decode_stride,
            Some(s) => s.at.elapsed() >= self.prime_every && rows != s.rows,
        }
    }

    /// Copy `cache`'s GDN states to the host and record one row. Errors are printed, never fatal.
    pub fn sample(&mut self, e: &Engine, cache: &Cache, at: &At<'_>) {
        self.seen.insert(
            at.id.to_string(),
            Seen {
                decoding: at.decoding,
                rows: at.rows,
                at: Instant::now(),
            },
        );
        if let Err(err) = self.sample_inner(e, cache, at) {
            eprintln!(
                "[gdn-census] sample of {} at {} rows failed: {err}",
                at.id, at.rows
            );
        }
    }

    fn sample_inner(
        &mut self,
        e: &Engine,
        cache: &Cache,
        at: &At<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t0 = Instant::now();
        let mut states: Vec<(usize, Vec<f32>)> = Vec::new();
        for (il, r) in cache.recur.iter().enumerate() {
            let Some(r) = r else { continue };
            let v = e.dtoh(&r.ssm_state)?;
            if v.is_empty() || v.len() % HEAD != 0 {
                return Err(format!(
                    "layer {il}: ssm_state of {} words is not whole 128x128 heads",
                    v.len()
                )
                .into());
            }
            states.push((il, v));
        }
        if states.is_empty() {
            return Ok(());
        }
        let t_copy = t0.elapsed();
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .min(16);
        let per = states.len().div_ceil(threads);
        let stats: Vec<(usize, Vec<HeadStats>)> = std::thread::scope(|sc| {
            let handles: Vec<_> = states
                .chunks(per)
                .map(|part| {
                    sc.spawn(move || {
                        part.iter()
                            .map(|(il, v)| (*il, v.chunks(HEAD).map(head_stats).collect()))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        if self.agg.is_empty() {
            self.agg = stats
                .iter()
                .map(|(il, hs)| LayerAgg {
                    layer: *il,
                    heads: vec![HeadAgg::default(); hs.len()],
                    hist: [0; HIST_N],
                    zero: 0,
                    nonfinite: 0,
                    ovf: 0,
                    ftz: 0,
                    sub: 0,
                    elems: 0,
                })
                .collect();
        }
        let mut layers = Vec::with_capacity(stats.len());
        for (k, (il, hs)) in stats.iter().enumerate() {
            let n = (hs.len() * HEAD) as f64;
            let sum = |f: &dyn Fn(&HeadStats) -> f64| hs.iter().map(f).sum::<f64>();
            let (max_h, max_abs) = hs.iter().enumerate().fold((0, 0.0f32), |a, (h, s)| {
                if s.max_abs > a.1 { (h, s.max_abs) } else { a }
            });
            let e_tot = sum(&|s| s.e_tot);
            let (r16w_h, r16w) = hs
                .iter()
                .enumerate()
                .map(|(h, s)| (h, rel(s.e_r16, s.e_tot)))
                .fold((0, 0.0f64), |a, b| if b.1 > a.1 { b } else { a });
            let rb16w = hs
                .iter()
                .map(|s| rel(s.e_rb16, s.e_tot))
                .fold(0.0f64, f64::max);
            let min_rms = hs
                .iter()
                .map(|s| (s.e_tot / HEAD as f64).sqrt())
                .fold(f64::INFINITY, f64::min);
            let g = |v: f64| format!("{v:.4e}").parse::<f64>().unwrap_or(v);
            layers.push(serde_json::json!([
                il,
                g(f64::from(max_abs)),
                max_h,
                g(sum(&|s| f64::from(s.sub)) / n),
                g(sum(&|s| f64::from(s.ftz)) / n),
                g(sum(&|s| f64::from(s.zero)) / n),
                sum(&|s| f64::from(s.ovf)) as u64,
                sum(&|s| f64::from(s.nonfinite)) as u64,
                g(if e_tot > 0.0 {
                    sum(&|s| s.e_sub) / e_tot
                } else {
                    0.0
                }),
                g(rel(sum(&|s| s.e_r16), e_tot)),
                g(r16w),
                r16w_h,
                g(rel(sum(&|s| s.e_rb16), e_tot)),
                g(rb16w),
                g(min_rms),
            ]));
            let la = &mut self.agg[k];
            if la.layer != *il || la.heads.len() != hs.len() {
                return Err(format!("layer set changed at layer {il}").into());
            }
            for (a, s) in la.heads.iter_mut().zip(hs) {
                a.max_abs = a.max_abs.max(s.max_abs);
                a.max_r16 = a.max_r16.max(rel(s.e_r16, s.e_tot));
                a.max_rb16 = a.max_rb16.max(rel(s.e_rb16, s.e_tot));
                a.max_sub_frac = a.max_sub_frac.max(f64::from(s.sub) / HEAD as f64);
                a.max_esub = a.max_esub.max(if s.e_tot > 0.0 {
                    s.e_sub / s.e_tot
                } else {
                    0.0
                });
                a.min_rms = a.min_rms.min((s.e_tot / HEAD as f64).sqrt());
                a.samples += 1;
                for (t, c) in la.hist.iter_mut().zip(s.hist.iter()) {
                    *t += u64::from(*c);
                }
                la.zero += u64::from(s.zero);
                la.nonfinite += u64::from(s.nonfinite);
                la.ovf += u64::from(s.ovf);
                la.ftz += u64::from(s.ftz);
                la.sub += u64::from(s.sub);
                la.elems += HEAD as u64;
            }
        }
        self.samples += 1;
        let phase = if at.decoding { "decode" } else { "prime" };
        let row = serde_json::json!([
            (self.started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
            at.id,
            phase,
            at.rows,
            at.n_prompt,
            at.generated,
            layers,
        ]);
        writeln!(self.out, "{row}")?;
        self.out.flush()?;
        self.write_agg()?;
        eprintln!(
            "[gdn-census] sample {} {} {phase} rows={} copy={:.0}ms total={:.0}ms",
            self.samples,
            at.id,
            at.rows,
            t_copy.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    fn write_agg(&self) -> std::io::Result<()> {
        let layers: Vec<_> = self
            .agg
            .iter()
            .map(|la| {
                let col = |f: &dyn Fn(&HeadAgg) -> serde_json::Value| {
                    la.heads.iter().map(f).collect::<Vec<_>>()
                };
                serde_json::json!({
                    "layer": la.layer,
                    "elems": la.elems,
                    "zero": la.zero,
                    "nonfinite": la.nonfinite,
                    "ovf": la.ovf,
                    "ftz": la.ftz,
                    "sub": la.sub,
                    "hist": la.hist.to_vec(),
                    "head_max_abs": col(&|a| serde_json::json!(a.max_abs)),
                    "head_max_r16": col(&|a| serde_json::json!(a.max_r16)),
                    "head_max_rb16": col(&|a| serde_json::json!(a.max_rb16)),
                    "head_max_sub_frac": col(&|a| serde_json::json!(a.max_sub_frac)),
                    "head_max_esub": col(&|a| serde_json::json!(a.max_esub)),
                    "head_min_rms": col(&|a| serde_json::json!(a.min_rms)),
                })
            })
            .collect();
        let doc = serde_json::json!({"samples": self.samples, "layers": layers});
        let tmp = self.dir.join("agg.json.tmp");
        std::fs::write(&tmp, doc.to_string())?;
        std::fs::rename(tmp, self.dir.join("agg.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_rounding_matches_ieee_binary16() {
        // normal range: 10 mantissa bits, ties to even
        assert_eq!(round_f16(1.0 + 2f32.powi(-11)), 1.0);
        assert_eq!(round_f16(1.0 + 3.0 * 2f32.powi(-11)), 1.0 + 2f32.powi(-9));
        assert_eq!(round_f16(-std::f32::consts::PI), -3.140_625);
        // range top: 65504 is the largest finite value, ties at 65520 go to infinity
        assert_eq!(round_f16(65_519.0), 65_504.0);
        assert!(round_f16(65_520.0).is_infinite());
        assert!(round_f16(-1e6).is_infinite() && round_f16(-1e6) < 0.0);
        // subnormals: multiples of 2^-24; 2^-25 ties to zero, just above it rounds up
        assert_eq!(round_f16(2f32.powi(-14)), 2f32.powi(-14));
        assert_eq!(round_f16(3.0 * 2f32.powi(-24)), 3.0 * 2f32.powi(-24));
        assert_eq!(round_f16(2f32.powi(-25)), 0.0);
        assert_eq!(round_f16(1.5 * 2f32.powi(-25)), 2f32.powi(-24));
        assert_eq!(round_f16(1e-9), 0.0);
        // the largest subnormal rounds up into the normal range
        assert_eq!(round_f16(2f32.powi(-14) - 2f32.powi(-26)), 2f32.powi(-14));
    }

    #[test]
    fn bf16_rounding_keeps_seven_mantissa_bits() {
        assert_eq!(round_bf16(1.0 + 2f32.powi(-8)), 1.0);
        assert_eq!(round_bf16(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
        assert_eq!(round_bf16(1e-9).to_bits() & 0xffff, 0);
    }

    #[test]
    fn head_stats_counts_small_elements_and_energy() {
        let mut xs = vec![0.5f32; HEAD];
        xs[0] = 1e-6; // subnormal in fp16, kept nonzero
        xs[1] = 1e-9; // flushes to zero in fp16
        xs[2] = 0.0;
        xs[3] = 70_000.0; // overflows fp16
        let s = head_stats(&xs);
        assert_eq!((s.sub, s.ftz, s.zero, s.ovf), (2, 1, 1, 1));
        assert_eq!(s.max_abs, 70_000.0);
        assert!(s.e_sub > 0.0 && s.e_sub < 1e-11);
        assert!(s.e_r16 < s.e_rb16);
    }
}
