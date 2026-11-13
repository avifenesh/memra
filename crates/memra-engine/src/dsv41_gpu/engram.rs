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

    /// The hash rows of `tokens` at positions `ids.len()..`, without appending them to `ids`:
    /// [`Self::hash_span`] of the whole sequence from `ids.len()`, which reads at most
    /// `max_ngram - 1` tokens back. A caller checks a call's tokens before its sequence takes them.
    pub fn hash_after(&self, ids: &[u32], tokens: &[u32]) -> Result<Vec<i64>, String> {
        let keep = ids.len().min(self.max_ngram.saturating_sub(1));
        let mut tail = Vec::with_capacity(keep + tokens.len());
        tail.extend_from_slice(&ids[ids.len() - keep..]);
        tail.extend_from_slice(tokens);
        self.hash_span(&tail, keep)
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

/// Map an Engram table's pages into the process up front (`madvise(MADV_POPULATE_READ)` on the
/// checkpoint mapping), once a process a table. The table is 98 GB of 256-byte rows and a call
/// reads rows scattered over it, so without this nearly every row is a first touch of its page:
/// measured 6.8 us a row cold against 0.03 to 0.1 us once the page is mapped (a 4,096-token
/// prefill's 98,000 rows 666 ms against 3.5 to 10 ms; a decode step's 24 rows a table 0.16 ms
/// against microseconds). Populating runs at 38 GB/s, 2.6 s a table, and takes no memory of its
/// own: the pages are the page cache's. A table the page cache does not hold is read from the
/// disk here, which one thread did at 1.3 GB/s (74 s a table, measured after another job dropped
/// the cache), so the range is split over [`POPULATE_THREADS`] threads: the disk serves them
/// side by side. Best effort: where the kernel has no such advice the rows fault in as before.
pub fn populate(stm: &StModel, table: &str) {
    static DONE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut done = DONE.lock().unwrap_or_else(|e| e.into_inner());
    if done.iter().any(|t| t == table) {
        return;
    }
    done.push(table.to_string());
    // SAFETY: sysconf has no preconditions
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(POPULATE_THREADS);
    for part in ["weight", "scale"] {
        if let Some((_, b)) = stm.raw(&format!("{table}.{part}")) {
            let a = b.as_ptr() as usize & !(page - 1);
            let end = b.as_ptr() as usize + b.len();
            // whole pages a thread, at least 1 GiB each
            let per = ((end - a).div_ceil(threads).max(1 << 30) + page - 1) & !(page - 1);
            std::thread::scope(|sc| {
                for lo in (a..end).step_by(per) {
                    let len = per.min(end - lo);
                    // SAFETY: [lo, lo + len) lies inside the checkpoint's read-only mapping that
                    // `b` borrows from; the advice only maps its pages
                    sc.spawn(move || unsafe {
                        libc::madvise(lo as *mut libc::c_void, len, libc::MADV_POPULATE_READ)
                    });
                }
            });
        }
    }
}

/// The most threads that map an Engram table ([`populate`]).
const POPULATE_THREADS: usize = 8;

/// The fewest rows a gather thread takes: under it one thread does the whole gather (a decode
/// step's rows are a few dozen).
const GATHER_ROWS_A_THREAD: usize = 4096;

/// `dst[i] = table[hashes[i]]`, rows of `w` bytes. The table is a memory-mapped file of 98 GB a
/// layer and the rows are scattered over it, so nearly every row is a page the process has not
/// touched: a 4,096-token prefill's 98,000 rows took 267 ms on one thread, about the time of
/// eight layers of kernels (measured: every sample of the host inside a prefill's forward was
/// here). The rows are independent, so they are split over the cores the process may use.
fn gather_rows(dst: &mut [u8], table: &[u8], hashes: &[i64], w: usize) {
    let copy = |d: &mut [u8], hs: &[i64]| {
        for (o, &h) in d.chunks_exact_mut(w).zip(hs) {
            let r = h as usize;
            o.copy_from_slice(&table[r * w..(r + 1) * w]);
        }
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(hashes.len() / GATHER_ROWS_A_THREAD)
        .max(1);
    if threads == 1 {
        return copy(dst, hashes);
    }
    let per = hashes.len().div_ceil(threads);
    std::thread::scope(|sc| {
        for (d, hs) in dst.chunks_mut(per * w).zip(hashes.chunks(per)) {
            sc.spawn(move || copy(d, hs));
        }
    });
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
        let (is, s) = stm
            .raw(&format!("{table}.scale"))
            .ok_or_else(|| format!("missing {table}.scale"))?;
        // the checks of `gather_host`: the rows and their per-32 scales as the table stores them
        if iw.dtype != "F8_E4M3"
            || iw.shape != [n_rows as u64, hd as u64]
            || is.dtype != "F8_E8M0"
            || is.shape != [n_rows as u64, sb as u64]
        {
            return Err(format!(
                "{table}: {} {:?} rows and {} {:?} scales, want F8_E4M3 [{n_rows}, {hd}] and \
                 F8_E8M0 [{n_rows}, {sb}]",
                iw.dtype, iw.shape, is.dtype, is.shape
            ));
        }
        {
            let pr = self
                .rows
                .as_mut_slice()
                .map_err(|e| format!("pinned: {e:?}"))?;
            if let Some(&h) = hashes.iter().find(|&&h| h < 0 || h as usize >= n_rows) {
                return Err(format!("{table}: hash id {h} outside {n_rows} rows"));
            }
            gather_rows(&mut pr[..hashes.len() * hd], w, hashes, hd);
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
            gather_rows(&mut ps[..hashes.len() * sb], s, hashes, sb);
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

    /// The rows of tokens not yet appended are the whole sequence's rows from where they start,
    /// at every split (a prefill from nothing, decode tokens, a drafted block), and a token the
    /// map does not hold is refused before any sequence takes it.
    #[test]
    fn hash_after_is_the_span_of_the_appended_sequence() {
        let c = consts();
        let ids: Vec<u32> = vec![5, 0, 19, 3, 3, 8, 11, 2, 17, 6];
        for at in 0..=ids.len() {
            assert_eq!(
                c.hash_after(&ids[..at], &ids[at..]).unwrap(),
                c.hash_span(&ids, at).unwrap(),
                "split at {at}"
            );
        }
        assert!(c.hash_after(&ids, &[20]).is_err());
    }
}
