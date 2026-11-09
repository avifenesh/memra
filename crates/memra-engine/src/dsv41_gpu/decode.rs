//! The V4.1 trunk as a decode loop: a prefill, then one token per call, every sequence's state
//! carried as the vendor carries it (`inference/model.py`, `generate.py`).
//!
//! What a sequence carries between calls (the vendor's buffer in brackets):
//!   - per layer: the sliding-window ring of fp8-quantized KV, position p at slot p % window
//!     (`Attention.window_kv_cache`), and on a ratio-2 kv source the open group's fp32 kv and
//!     gate rows (`Compressor.kv_state` / `score_state`);
//!   - per kv-source group: the compressed KV (the source's `Attention.compress_kv_cache`) and the
//!     index keys (the key-owning `Indexer.k_cache`), appended as groups complete;
//!   - the token history, for the Engram n-gram hash of each new token (`NgramHashState`).
//!
//! Each call recomputes the per-forward half of the shared runtime (the selection, the candidate
//! pool) at its index sources, as the vendor's forward does.
//!
//! A call at position 0 is the vendor's prefill program; every later call is one token, the
//! vendor's decode program (the whole ring oldest slot first, the compressor's slot update, the
//! indexer over `end_pos / ratio` keys, one Engram row). Below the window the two programs differ
//! for the same row (another key order into the blocked softmax), so a generated token is always
//! fed as a decode call, never prefilled again. More than one token at a position past 0 (chunked
//! continuation) is not a vendor program and is refused.
//!
//! [`Runner::forward`] runs a list of calls layer by layer: each layer's weights are loaded once
//! (or kept resident) and every call runs through it before the next layer, a sequence's calls in
//! order. A call reads only state written by its own and earlier calls (compressed rows below its
//! end position, its own selection), so a sequence's tokens fed as several calls of one forward
//! (teacher forcing, or replaying a saved generation) give the bytes of feeding them one forward
//! at a time, and the weights stream once per forward instead of once per token. Every call runs
//! alone through each block (one sequence, its own rows), so running several sequences in one
//! forward never changes a sequence's numbers.
//!
//! The ranks are a [`Group`]: one rank is the one-card program (with the grouped expert dispatch,
//! byte-identical to the per-expert loop), N ranks the TP/EP partition, which writes the same
//! bytes.

use super::engram::EngramConsts;
use super::tp::Group;
use super::{BW, Buf, Geo, LayerState, LayerW, Shared, Trace};
use memra_gguf::safetensors::StModel;
use std::time::Instant;

/// One call: the next tokens of sequence `seq` (the whole prompt at position 0, then one token).
#[derive(Debug, Clone)]
pub struct Call {
    pub seq: usize,
    pub tokens: Vec<u32>,
}

/// A call's final stream and carried pre-mix on every rank, ready for the head.
pub struct Out {
    pub h: Vec<Buf>,
    pub pre: Vec<Buf>,
    /// the call's first position and its token count
    pub sp: usize,
    pub s: usize,
}

/// One sequence's state on every rank.
pub struct Seq {
    /// the tokens in the cache, in order
    pub ids: Vec<u32>,
    /// `[layer][rank]`: window ring and the ratio-2 compressor's open group
    ls: Vec<Vec<LayerState>>,
    /// `[kv source][rank]`: each kv-source group's compressed KV and index keys
    store: Vec<Vec<Shared>>,
}

impl Seq {
    /// Tokens in the cache (the next call's position).
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Where the time of the forwards went.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub forwards: usize,
    pub calls: usize,
    pub layers_loaded: usize,
    pub load_s: f64,
    pub run_s: f64,
}

/// The trunk on a rank group, its weights streamed per forward from the checkpoint mmap (the
/// first `keep` layers kept on the card once loaded).
pub struct Runner<'a> {
    pub grp: Group,
    pub g: Geo,
    stm: &'a StModel,
    ec: Option<EngramConsts>,
    pub n_layers: usize,
    /// positions the RoPE tables and the cache capacities cover
    pub max_pos: usize,
    keep: usize,
    resident: Vec<Option<Vec<LayerW>>>,
    head: Option<(Vec<Buf>, Vec<BW>)>,
    pub stats: Stats,
}

impl<'a> Runner<'a> {
    /// Ranks on `devices` covering positions below `max_pos`, running the first `n_layers` trunk
    /// layers with prefill slices of `chunk` rows. `ec` is required when an Engram layer runs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        stm: &'a StModel,
        g: Geo,
        ec: Option<EngramConsts>,
        devices: &[usize],
        max_pos: usize,
        n_layers: usize,
        keep: usize,
        chunk: usize,
    ) -> Result<Self, String> {
        if n_layers > g.compress_ratios.len() {
            return Err(format!(
                "{n_layers} layers: the config has {}",
                g.compress_ratios.len()
            ));
        }
        if ec.is_none() && g.engram_layers.iter().any(|&l| l < n_layers) {
            return Err("the Engram layers need the hashing constants".into());
        }
        let mut grp = Group::new(devices, &g, max_pos)?;
        for c in grp.cx.iter_mut() {
            c.chunk = chunk.max(1);
        }
        Ok(Self {
            grp,
            g,
            stm,
            ec,
            n_layers,
            max_pos,
            keep: keep.min(n_layers),
            resident: (0..n_layers).map(|_| None).collect(),
            head: None,
            stats: Stats::default(),
        })
    }

    /// Keep the first `keep` layers on the card from the next forward on (fewer: free the rest).
    /// A long prompt's prefill wants the memory; the decode forwards after it want the layers.
    pub fn set_keep(&mut self, keep: usize) {
        self.keep = keep.min(self.n_layers);
        for w in self.resident.iter_mut().skip(self.keep) {
            *w = None;
        }
    }

    /// An empty sequence for this runner's ranks and layers.
    pub fn seq(&self) -> Seq {
        let n = self.grp.n();
        let per_rank =
            |_: usize| -> Vec<LayerState> { (0..n).map(|_| LayerState::default()).collect() };
        Seq {
            ids: Vec::new(),
            ls: (0..self.n_layers).map(per_rank).collect(),
            store: (0..self.g.kv_sources.len())
                .map(|_| (0..n).map(|_| Shared::default()).collect())
                .collect(),
        }
    }

    /// The kv-source group whose compressed KV and index keys layer `li` reads: the latest kv
    /// source at or below it (vendor: the source that last pointed `shared_attn` at its cache).
    pub fn group_of(g: &Geo, li: usize) -> Option<usize> {
        if g.compress_ratios[li] == 0 {
            return None;
        }
        g.kv_sources
            .iter()
            .enumerate()
            .filter(|&(_, &k)| k <= li)
            .max_by_key(|&(_, &k)| k)
            .map(|(i, _)| i)
    }

    /// The bf16 embedding of `tokens`, `hc` copies per row, and the one-hot carried pre-mix.
    fn embed(&self, tokens: &[u32]) -> Result<(Vec<u8>, Vec<f32>), String> {
        let (hc, d) = (self.g.hc, self.g.dim);
        let (ei, eb) = self.stm.raw("embed.weight").ok_or("embed.weight missing")?;
        if ei.dtype != "BF16" || ei.shape != [self.g.vocab as u64, d as u64] {
            return Err(format!("embed.weight: {} {:?}", ei.dtype, ei.shape));
        }
        let s = tokens.len();
        let mut hb = vec![0u8; s * hc * d * 2];
        for (t, &id) in tokens.iter().enumerate() {
            let id = id as usize;
            if id >= self.g.vocab {
                return Err(format!("token {id} outside the vocabulary"));
            }
            let row = &eb[id * d * 2..(id + 1) * d * 2];
            for k in 0..hc {
                hb[(t * hc + k) * d * 2..(t * hc + k + 1) * d * 2].copy_from_slice(row);
            }
        }
        let mut pm = vec![0f32; s * hc];
        for t in 0..s {
            pm[t * hc] = 1.0;
        }
        Ok((hb, pm))
    }

    /// Run `calls` through the trunk, layer by layer (see the module doc). A sequence's calls run
    /// in the given order from its current length. Returns each call's final stream for
    /// [`Self::logits`]. An error part-way leaves the sequences it touched unusable.
    pub fn forward(&mut self, seqs: &mut [Seq], calls: &[Call]) -> Result<Vec<Out>, String> {
        let n = self.grp.n();
        let g = self.g.clone();
        // positions first, so a refused call leaves every sequence as it was
        let mut lens: Vec<usize> = seqs.iter().map(|q| q.len()).collect();
        let mut pos = Vec::with_capacity(calls.len());
        for c in calls {
            let len = lens
                .get_mut(c.seq)
                .ok_or_else(|| format!("call on sequence {} of {}", c.seq, seqs.len()))?;
            let (sp, s) = (*len, c.tokens.len());
            if s == 0 {
                return Err("a call needs at least one token".into());
            }
            if sp > 0 && s != 1 {
                return Err(format!(
                    "{s} tokens at position {sp}: past the prefill the vendor program takes one token per call"
                ));
            }
            if sp + s > self.max_pos {
                return Err(format!(
                    "position {} past the runner's {} (RoPE tables and caches)",
                    sp + s,
                    self.max_pos
                ));
            }
            *len += s;
            pos.push((sp, s));
        }
        let engram = g.engram_layers.iter().any(|&l| l < self.n_layers);
        let mut hashes: Vec<Option<Vec<i64>>> = Vec::with_capacity(calls.len());
        let mut h: Vec<Vec<Buf>> = Vec::with_capacity(calls.len());
        let mut pre: Vec<Vec<Buf>> = Vec::with_capacity(calls.len());
        for (c, &(sp, _)) in calls.iter().zip(&pos) {
            let q = &mut seqs[c.seq];
            q.ids.extend_from_slice(&c.tokens);
            hashes.push(match (&self.ec, engram) {
                (Some(ec), true) => Some(ec.hash_span(&q.ids, sp)?),
                _ => None,
            });
            let (hb, pm) = self.embed(&c.tokens)?;
            let mut hr = Vec::with_capacity(n);
            let mut pr = Vec::with_capacity(n);
            for cx in &self.grp.cx {
                hr.push(Buf::from_bytes(&cx.ops.st, &hb)?);
                pr.push(Buf::from_f32(&cx.ops.st, &pm)?);
            }
            h.push(hr);
            pre.push(pr);
        }
        let mut fwd: Vec<Vec<Shared>> = calls
            .iter()
            .map(|_| (0..n).map(|_| Shared::default()).collect())
            .collect();
        let mut tr: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
        for li in 0..self.n_layers {
            let tl = Instant::now();
            let lw = match self.resident[li].take() {
                Some(w) => w,
                None => {
                    self.stats.layers_loaded += 1;
                    self.grp.load_layer(self.stm, &g, li, false)?
                }
            };
            self.stats.load_s += tl.elapsed().as_secs_f64();
            let tr0 = Instant::now();
            let slot = match (&self.ec, lw[0].engram.is_some()) {
                (Some(ec), true) => Some(
                    ec.layer_ids
                        .iter()
                        .position(|&x| x == li)
                        .ok_or("engram layer not in the constants")?,
                ),
                _ => None,
            };
            let gi = Self::group_of(&g, li);
            for (ci, c) in calls.iter().enumerate() {
                let (sp, s) = pos[ci];
                let q = &mut seqs[c.seq];
                if let Some(sl) = slot {
                    let ec = self.ec.as_ref().unwrap();
                    let (nl, cols) = (ec.layer_ids.len(), ec.cols());
                    let all = hashes[ci].as_ref().ok_or("engram hashes missing")?;
                    let mine: Vec<i64> = (0..s)
                        .flat_map(|t| all[(t * nl + sl) * cols..(t * nl + sl + 1) * cols].to_vec())
                        .collect();
                    for r in 0..n {
                        self.grp.cx[r].engram(self.stm, &lw[r], &h[ci][r], &mine, s, &mut tr[r])?;
                    }
                }
                if let Some(gi) = gi {
                    for (f, st) in fwd[ci].iter_mut().zip(q.store[gi].iter_mut()) {
                        f.swap_group(st);
                    }
                }
                let res = self.grp.block_forward(
                    &lw,
                    &h[ci],
                    &pre[ci],
                    sp,
                    s,
                    &mut fwd[ci],
                    &mut q.ls[li],
                    &mut tr,
                );
                if let Some(gi) = gi {
                    for (f, st) in fwd[ci].iter_mut().zip(q.store[gi].iter_mut()) {
                        f.swap_group(st);
                    }
                }
                let (h2, p2) = res?;
                h[ci] = h2;
                pre[ci] = p2;
            }
            self.stats.run_s += tr0.elapsed().as_secs_f64();
            if li < self.keep {
                self.resident[li] = Some(lw);
            }
        }
        self.stats.forwards += 1;
        self.stats.calls += calls.len();
        Ok(h.into_iter()
            .zip(pre)
            .zip(pos)
            .map(|((h, pre), (sp, s))| Out { h, pre, sp, s })
            .collect())
    }

    /// Logits `[rows][vocab]` of rows `[r0, r0 + rows)` of a call: the collapse with the carried
    /// pre-mix, the final norm and the fp32 head (each rank's vocab rows, gathered).
    pub fn logits(&mut self, out: &Out, r0: usize, rows: usize) -> Result<Vec<f32>, String> {
        if r0 + rows > out.s {
            return Err(format!(
                "rows {r0}..{} of a {}-token call",
                r0 + rows,
                out.s
            ));
        }
        if self.head.is_none() {
            self.head = Some(self.grp.load_head(self.stm, &self.g)?);
        }
        let (nw, hw) = self.head.as_ref().unwrap();
        self.grp.head_logits(&out.h, &out.pre, r0, rows, nw, hw)
    }

    /// The scored next-token distribution after a call: its last row's logits reduced.
    pub fn score_last(&mut self, out: &Out, keep_logits: bool) -> Result<Scored, String> {
        let lg = self.logits(out, out.s - 1, 1)?;
        Scored::of(lg, keep_logits)
    }
}

/// Logits kept per step, as many as the oracle tape's `top_ids`/`top_vals`.
pub const TOPK: usize = 32;

/// One next-token distribution, reduced: the greedy token, log-sum-exp and top-[`TOPK`].
#[derive(Debug, Clone)]
pub struct Scored {
    pub argmax: u32,
    pub lse: f64,
    pub top: Vec<(usize, f32)>,
    pub logits: Option<Vec<f32>>,
}

impl Scored {
    pub fn of(lg: Vec<f32>, keep_logits: bool) -> Result<Self, String> {
        if lg.iter().any(|x| !x.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(Self {
            argmax: argmax(&lg) as u32,
            lse: lse(&lg),
            top: top_k(&lg),
            logits: keep_logits.then_some(lg),
        })
    }
}

/// log(sum(exp(x))) in f64.
pub fn lse(x: &[f32]) -> f64 {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    m + x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
}

/// The [`TOPK`] largest logits of a row, largest first (lower id first among equal values).
pub fn top_k(x: &[f32]) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = x.iter().take(TOPK).cloned().enumerate().collect();
    let worse = |a: &(usize, f32), b: &(usize, f32)| a.1 < b.1 || (a.1 == b.1 && a.0 > b.0);
    let mut low = (0..best.len())
        .reduce(|a, b| if worse(&best[b], &best[a]) { b } else { a })
        .unwrap_or(0);
    for (i, &v) in x.iter().enumerate().skip(TOPK) {
        if v > best[low].1 {
            best[low] = (i, v);
            low = (0..TOPK)
                .reduce(|a, b| if worse(&best[b], &best[a]) { b } else { a })
                .unwrap_or(0);
        }
    }
    best.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    best
}

/// The lowest index among equal maxima, like the oracle's argmax.
pub fn argmax(x: &[f32]) -> usize {
    let mut b = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[b] {
            b = i;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_is_the_sorted_largest_with_lower_ids_first_among_ties() {
        let mut x: Vec<f32> = (0..200).map(|i| ((i * 37) % 101) as f32).collect();
        x[150] = 1000.0;
        x[7] = 1000.0;
        let got = top_k(&x);
        let mut want: Vec<(usize, f32)> = x.iter().cloned().enumerate().collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        want.truncate(TOPK);
        assert_eq!(got, want);
        assert_eq!(got[0], (7, 1000.0));
        assert_eq!(got[0].0, argmax(&x));
    }

    #[test]
    fn each_layer_reads_the_latest_kv_source_at_or_below_it() {
        // the V4.1-Flash layout: window-only 0, 1; ratio 2 from 2 (sources 2, 8, 14); ratio 1 from 20
        let mut ratios = vec![0, 0];
        ratios.extend(std::iter::repeat_n(2, 18));
        ratios.extend(std::iter::repeat_n(1, 20));
        let g = Geo {
            compress_ratios: ratios,
            kv_sources: vec![2, 8, 14, 20],
            ..Geo::default()
        };
        let got: Vec<Option<usize>> = (0..40).map(|l| Runner::group_of(&g, l)).collect();
        assert_eq!(&got[..3], &[None, None, Some(0)]);
        assert_eq!(got[7], Some(0));
        assert_eq!(got[8], Some(1));
        assert_eq!(got[13], Some(1));
        assert_eq!(got[14], Some(2));
        assert_eq!(got[19], Some(2));
        assert!(got[20..].iter().all(|&x| x == Some(3)));
    }

    #[test]
    fn lse_is_stable_far_from_zero() {
        let x = [1000.0f32, 1000.0, 999.0];
        let want = 1000.0 + (2.0 + (-1.0f64).exp()).ln();
        assert!((lse(&x) - want).abs() < 1e-12);
    }
}
