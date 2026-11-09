//! Engram on the host: the n-gram hash (vendor `engram.py` `NgramHashState`) and the gather of
//! the hashed table rows into pinned staging for one H2D copy per step.
//!
//! The tables stay in host memory (two fp8 tables of 384M x 256 B plus ue8m0 scales, 203 GB);
//! a token at an Engram layer reads `(max_ngram - 1) * heads` = 24 rows of 256 B. The hash
//! constants (prime bucket sizes, offsets, per-layer multipliers, the compressed token map) are the
//! ones the vendor code derives at load; they are read from its dump (`engram_constants.json` and
//! `.token_map.i32`), checked against the config, and the hash ids are gated against the vendor's.

use super::{Buf, Ops};
use cudarc::driver::PinnedHostSlice;
use memra_gguf::safetensors::StModel;

/// The vendor-derived hashing constants for every Engram layer.
#[derive(Debug, Clone)]
pub struct EngramConsts {
    pub layer_ids: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// `[layer][ngram - 2][head]` prime bucket sizes
    pub primes: Vec<Vec<Vec<i64>>>,
    /// `[layer][(ngram - 2) * heads + head]` row offsets of each bucket range
    pub offsets: Vec<Vec<i64>>,
    /// `[layer][lookback]` odd multipliers
    pub multipliers: Vec<Vec<i64>>,
    pub num_embeddings: Vec<i64>,
    /// compressed id of the pad token
    pub pad: i64,
    pub compressed_vocab: usize,
    pub token_map: Vec<i32>,
}

fn jarr(v: &serde_json::Value, k: &str) -> Result<Vec<serde_json::Value>, String> {
    v.get(k)
        .and_then(|x| x.as_array())
        .cloned()
        .ok_or_else(|| format!("engram constants: no array {k}"))
}
fn ji64(v: &serde_json::Value) -> Result<i64, String> {
    v.as_i64()
        .ok_or_else(|| format!("engram constants: {v} is not an integer"))
}

impl EngramConsts {
    pub fn load(
        json_path: &std::path::Path,
        token_map_path: &std::path::Path,
    ) -> Result<Self, String> {
        let s = std::fs::read_to_string(json_path)
            .map_err(|e| format!("{}: {e}", json_path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&s).map_err(|e| format!("{}: {e}", json_path.display()))?;
        let usz = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("engram constants: no {k}"))
        };
        let layer_ids = jarr(&v, "layer_ids")?
            .iter()
            .map(|x| ji64(x).map(|y| y as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let primes = jarr(&v, "primes")?
            .iter()
            .map(|l| {
                l.as_array()
                    .ok_or("primes layer")
                    .map_err(String::from)
                    .and_then(|ng| {
                        ng.iter()
                            .map(|h| {
                                h.as_array()
                                    .ok_or("primes ngram")
                                    .map_err(String::from)
                                    .and_then(|hs| {
                                        hs.iter().map(ji64).collect::<Result<Vec<_>, _>>()
                                    })
                            })
                            .collect::<Result<Vec<_>, _>>()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mat = |k: &str| -> Result<Vec<Vec<i64>>, String> {
            jarr(&v, k)?
                .iter()
                .map(|r| {
                    r.as_array()
                        .ok_or(format!("{k} row"))
                        .and_then(|xs| xs.iter().map(ji64).collect())
                })
                .collect()
        };
        let offsets = mat("offsets")?;
        let multipliers = mat("multipliers")?;
        let num_embeddings = jarr(&v, "num_embeddings")?
            .iter()
            .map(ji64)
            .collect::<Result<Vec<_>, _>>()?;
        let raw = std::fs::read(token_map_path)
            .map_err(|e| format!("{}: {e}", token_map_path.display()))?;
        let token_map: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let c = Self {
            layer_ids,
            max_ngram: usz("max_ngram_size")?,
            n_heads: usz("n_heads")?,
            head_dim: usz("head_dim")?,
            primes,
            offsets,
            multipliers,
            num_embeddings,
            pad: ji64(
                v.get("pad_compressed_id")
                    .ok_or("engram constants: no pad_compressed_id")?,
            )?,
            compressed_vocab: usz("compressed_vocab")?,
            token_map,
        };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<(), String> {
        let nl = self.layer_ids.len();
        if self.primes.len() != nl
            || self.offsets.len() != nl
            || self.multipliers.len() != nl
            || self.num_embeddings.len() != nl
        {
            return Err("engram constants: per-layer arrays disagree in length".into());
        }
        for l in 0..nl {
            let flat: Vec<i64> = self.primes[l].iter().flatten().copied().collect();
            if self.primes[l].len() != self.max_ngram - 1
                || flat.len() != self.cols()
                || self.multipliers[l].len() != self.max_ngram
            {
                return Err(format!("engram constants: layer {l} shape"));
            }
            // offsets are the running sums of the bucket sizes; the table holds exactly their total
            let mut acc = 0i64;
            for (i, &p) in flat.iter().enumerate() {
                if self.offsets[l][i] != acc {
                    return Err(format!(
                        "engram constants: layer {l} offset {i} is not the prefix sum"
                    ));
                }
                acc += p;
            }
            if acc != self.num_embeddings[l] {
                return Err(format!(
                    "engram constants: layer {l} buckets sum to {acc}, table has {}",
                    self.num_embeddings[l]
                ));
            }
        }
        if self
            .token_map
            .iter()
            .any(|&t| t < 0 || t as usize >= self.compressed_vocab)
        {
            return Err("engram constants: token map leaves the compressed vocab".into());
        }
        Ok(())
    }

    /// Hash columns per Engram layer: (max_ngram - 1) n-gram sizes x heads.
    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    /// Hash ids `[seq][n_layers][cols]` of the n-grams ending at each position of a text prompt
    /// starting at position 0 (vendor `NgramHashState.forward` with no image spans): look-back
    /// stops at the sequence start and pads with the compressed pad id.
    pub fn hash_prefill(&self, ids: &[u32]) -> Result<Vec<i64>, String> {
        let comp: Vec<i64> = ids
            .iter()
            .map(|&t| {
                self.token_map
                    .get(t as usize)
                    .map(|&c| c as i64)
                    .ok_or_else(|| format!("engram: token {t} outside the token map"))
            })
            .collect::<Result<_, _>>()?;
        let nl = self.layer_ids.len();
        let cols = self.cols();
        let mut out = vec![0i64; ids.len() * nl * cols];
        let mut toks = vec![0i64; self.max_ngram];
        for t in 0..ids.len() {
            for (shift, tk) in toks.iter_mut().enumerate() {
                *tk = if t < shift { self.pad } else { comp[t - shift] };
            }
            for l in 0..nl {
                let m = &self.multipliers[l];
                let mut rolling = toks[0].wrapping_mul(m[0]);
                for i in 1..self.max_ngram {
                    rolling ^= toks[i].wrapping_mul(m[i]);
                    for h in 0..self.n_heads {
                        let c = (i - 1) * self.n_heads + h;
                        let p = self.primes[l][i - 1][h];
                        out[(t * nl + l) * cols + c] = rolling.rem_euclid(p) + self.offsets[l][c];
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Copy the hashed rows of one Engram table out of the host mmap: `(rows [n][hd] e4m3, scales
/// [n][hd/32] ue8m0)`, in hash order.
pub fn gather_host(
    stm: &StModel,
    table: &str,
    n_rows: usize,
    hd: usize,
    hashes: &[i64],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (iw, w) = stm
        .raw(&format!("{table}.weight"))
        .ok_or_else(|| format!("missing {table}.weight"))?;
    let (is, s) = stm
        .raw(&format!("{table}.scale"))
        .ok_or_else(|| format!("missing {table}.scale"))?;
    if iw.dtype != "F8_E4M3"
        || iw.shape != [n_rows as u64, hd as u64]
        || is.dtype != "F8_E8M0"
        || is.shape != [n_rows as u64, (hd / 32) as u64]
    {
        return Err(format!(
            "{table}: unexpected layout {} {:?} / {} {:?}",
            iw.dtype, iw.shape, is.dtype, is.shape
        ));
    }
    let sb = hd / 32;
    let mut rows = Vec::with_capacity(hashes.len() * hd);
    let mut scales = Vec::with_capacity(hashes.len() * sb);
    for &h in hashes {
        if h < 0 || h as usize >= n_rows {
            return Err(format!("{table}: hash id {h} outside {n_rows} rows"));
        }
        let r = h as usize;
        rows.extend_from_slice(&w[r * hd..(r + 1) * hd]);
        scales.extend_from_slice(&s[r * sb..(r + 1) * sb]);
    }
    Ok((rows, scales))
}

/// Pinned staging for one Engram layer's step: the gathered rows land in page-locked host memory
/// and go to the device in one async copy each.
pub struct Stage {
    rows: PinnedHostSlice<u8>,
    sc: PinnedHostSlice<u8>,
    pub d_rows: Buf,
    pub d_sc: Buf,
    pub cap: usize,
    hd: usize,
}

impl Stage {
    /// Room for `cap` hashed rows of `hd` bytes.
    pub fn new(ops: &Ops, cap: usize, hd: usize) -> Result<Self, String> {
        let ctx = ops.st.context().clone();
        let rows =
            unsafe { ctx.alloc_pinned::<u8>(cap * hd) }.map_err(|e| format!("pinned: {e:?}"))?;
        let sc = unsafe { ctx.alloc_pinned::<u8>(cap * hd / 32) }
            .map_err(|e| format!("pinned: {e:?}"))?;
        Ok(Self {
            rows,
            sc,
            d_rows: ops.buf(cap * hd)?,
            d_sc: ops.buf(cap * hd / 32)?,
            cap,
            hd,
        })
    }

    /// Gather `hashes` from the host table into the pinned buffers, then queue the H2D copies.
    pub fn load(
        &mut self,
        ops: &Ops,
        stm: &StModel,
        table: &str,
        n_rows: usize,
        hashes: &[i64],
    ) -> Result<(), String> {
        if hashes.len() > self.cap {
            return Err(format!(
                "engram stage: {} rows, capacity {}",
                hashes.len(),
                self.cap
            ));
        }
        let hd = self.hd;
        let sb = hd / 32;
        let (iw, w) = stm
            .raw(&format!("{table}.weight"))
            .ok_or_else(|| format!("missing {table}.weight"))?;
        let (_, s) = stm
            .raw(&format!("{table}.scale"))
            .ok_or_else(|| format!("missing {table}.scale"))?;
        if iw.shape != [n_rows as u64, hd as u64] {
            return Err(format!("{table}: shape {:?}", iw.shape));
        }
        {
            let pr = self
                .rows
                .as_mut_slice()
                .map_err(|e| format!("pinned: {e:?}"))?;
            for (i, &h) in hashes.iter().enumerate() {
                if h < 0 || h as usize >= n_rows {
                    return Err(format!("{table}: hash id {h} outside {n_rows} rows"));
                }
                let r = h as usize;
                pr[i * hd..(i + 1) * hd].copy_from_slice(&w[r * hd..(r + 1) * hd]);
            }
        }
        {
            let ps = self
                .sc
                .as_mut_slice()
                .map_err(|e| format!("pinned: {e:?}"))?;
            for (i, &h) in hashes.iter().enumerate() {
                let r = h as usize;
                ps[i * sb..(i + 1) * sb].copy_from_slice(&s[r * sb..(r + 1) * sb]);
            }
        }
        ops.st
            .memcpy_htod(&self.rows, &mut self.d_rows.s)
            .map_err(|e| format!("engram h2d: {e:?}"))?;
        ops.st
            .memcpy_htod(&self.sc, &mut self.d_sc.s)
            .map_err(|e| format!("engram h2d: {e:?}"))?;
        Ok(())
    }
}
