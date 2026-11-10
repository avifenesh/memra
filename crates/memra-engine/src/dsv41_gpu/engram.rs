//! Engram on the host: the n-gram hash (vendor `engram.py` `NgramHashState`) and the gather of
//! the hashed table rows into pinned staging for one H2D copy per step.
//!
//! The tables stay in host memory (two fp8 tables of 384M x 256 B plus ue8m0 scales, 203 GB);
//! a token at an Engram layer reads `(max_ngram - 1) * heads` = 24 rows of 256 B. The bucket
//! primes and the per-layer multipliers come from `memra_gguf::dsv41_forward` (the primes drawn as
//! the vendor draws them, numpy's `SeedSequence` + `PCG64` + Lemire draw ported bit-exactly for the
//! multipliers); the compressed token map is still an input (the vendor builds it from tokenizer
//! normalizers at load). The gates compare the resulting hash ids with the vendor's own.

use super::{Buf, Geo, Ops};
use cudarc::driver::{CudaEvent, PinnedHostSlice};
use memra_gguf::safetensors::StModel;

/// The hashing constants for every Engram layer.
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

/// The compressed token map as raw little-endian i32, one entry per vocabulary id.
pub fn token_map_from_file(path: &std::path::Path) -> Result<Vec<i32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if raw.len() % 4 != 0 {
        return Err(format!("{}: not an i32 array", path.display()));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl EngramConsts {
    /// Derive the constants from the config and the token map, then check them against the
    /// config's table sizes.
    pub fn derive(g: &Geo, token_map: Vec<i32>) -> Result<Self, String> {
        use memra_gguf::dsv41_forward::{engram_multipliers, engram_primes};
        let nl = g.engram_layers.len();
        let primes_u = engram_primes(nl, g.engram_ngram, g.engram_heads, g.engram_bucket_base);
        let primes: Vec<Vec<Vec<i64>>> = primes_u
            .iter()
            .map(|l| {
                l.iter()
                    .map(|n| n.iter().map(|&p| p as i64).collect())
                    .collect()
            })
            .collect();
        let offsets = primes
            .iter()
            .map(|l| {
                let mut acc = 0i64;
                l.iter()
                    .flatten()
                    .map(|&p| {
                        let o = acc;
                        acc += p;
                        o
                    })
                    .collect()
            })
            .collect();
        let multipliers = g
            .engram_layers
            .iter()
            .map(|&l| engram_multipliers(l as u32, g.engram_ngram, g.engram_compressed_vocab))
            .collect::<Result<Vec<_>, _>>()?;
        let pad = *token_map
            .get(g.engram_pad_id)
            .ok_or("engram: pad id outside the token map")? as i64;
        let c = Self {
            layer_ids: g.engram_layers.clone(),
            max_ngram: g.engram_ngram,
            n_heads: g.engram_heads,
            head_dim: g.engram_head_dim,
            primes,
            offsets,
            multipliers,
            num_embeddings: g.engram_rows.iter().map(|&r| r as i64).collect(),
            pad,
            compressed_vocab: g.engram_compressed_vocab as usize,
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
            let total: i64 = flat.iter().sum();
            if total != self.num_embeddings[l] {
                return Err(format!(
                    "engram constants: layer {l} buckets sum to {total}, table has {}",
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

    /// Hash ids `[ids.len() - from][n_layers][cols]` of the positions `from..` of the text
    /// sequence `ids` (prompt and fed tokens from position 0): the n-grams ending there, each
    /// looking back over at most `max_ngram - 1` earlier tokens. The rows equal those of
    /// [`Self::hash_prefill`] over the whole sequence (the vendor's decode step hashes one new
    /// token against the tokens before it).
    pub fn hash_span(&self, ids: &[u32], from: usize) -> Result<Vec<i64>, String> {
        if from > ids.len() {
            return Err(format!("engram: position {from} past {} tokens", ids.len()));
        }
        let start = from.saturating_sub(self.max_ngram - 1);
        let all = self.hash_prefill(&ids[start..])?;
        let row = self.layer_ids.len() * self.cols();
        Ok(all[(from - start) * row..].to_vec())
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
    /// recorded behind a load's copies; the next load waits for it before it writes the pinned
    /// buffers (the copies take a call's rows, not the whole buffers, so they go around the
    /// pinned slices' own events)
    done: CudaEvent,
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
            done: ctx.new_event(None).map_err(|e| format!("event: {e:?}"))?,
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
        // The copies below take only this call's rows. The buffers are sized for the largest
        // prefill chunk so far (50 MB after one of 8,192 tokens), and copying all of them on every
        // call cost a decode step 0.9 ms per Engram layer.
        self.done
            .synchronize()
            .map_err(|e| format!("engram stage: {e:?}"))?;
        let cu = ops.sv() as cudarc::driver::sys::CUstream;
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
            // SAFETY: the source is pinned memory this stage owns; `done` (recorded below) is
            // waited on before the next write to it, and the destination holds `cap` rows
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    self.d_rows.p(&ops.st).addr(),
                    &pr[..hashes.len() * hd],
                    cu,
                )
            }
            .map_err(|e| format!("engram h2d: {e:?}"))?;
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
            // SAFETY: as above
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    self.d_sc.p(&ops.st).addr(),
                    &ps[..hashes.len() * sb],
                    cu,
                )
            }
            .map_err(|e| format!("engram h2d: {e:?}"))?;
        }
        self.done
            .record(&ops.st)
            .map_err(|e| format!("engram stage: {e:?}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small hashing constants (2 layers, 4-grams, 3 heads, 20 compressed ids).
    fn consts() -> EngramConsts {
        let primes = vec![
            vec![vec![7, 11, 13], vec![17, 19, 23], vec![29, 31, 37]],
            vec![vec![41, 43, 47], vec![53, 59, 61], vec![67, 71, 73]],
        ];
        let offsets = primes
            .iter()
            .map(|l: &Vec<Vec<i64>>| {
                let mut o = 0;
                l.iter()
                    .flatten()
                    .map(|&p| {
                        let x = o;
                        o += p;
                        x
                    })
                    .collect()
            })
            .collect();
        EngramConsts {
            layer_ids: vec![1, 14],
            max_ngram: 4,
            n_heads: 3,
            head_dim: 8,
            num_embeddings: primes.iter().map(|l| l.iter().flatten().sum()).collect(),
            primes,
            offsets,
            multipliers: vec![vec![3, 5, 9, 17], vec![33, 65, 129, 257]],
            pad: 2,
            compressed_vocab: 20,
            token_map: (0..20).map(|i| (i * 7) % 20).collect(),
        }
    }

    #[test]
    fn a_decode_span_hashes_like_the_whole_sequence() {
        let c = consts();
        c.validate().unwrap();
        let ids: Vec<u32> = vec![5, 0, 19, 3, 3, 8, 11, 2, 17, 6];
        let all = c.hash_prefill(&ids).unwrap();
        let row = c.layer_ids.len() * c.cols();
        for from in 0..=ids.len() {
            assert_eq!(
                c.hash_span(&ids, from).unwrap(),
                all[from * row..].to_vec(),
                "from {from}"
            );
        }
        // one decode token: its row only, from the look-back window
        let last = c.hash_span(&ids, ids.len() - 1).unwrap();
        assert_eq!(last.len(), row);
        assert!(c.hash_span(&ids, ids.len() + 1).is_err());
    }
}
