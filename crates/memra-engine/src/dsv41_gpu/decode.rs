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
use super::experts::{CacheStats, ExpertCache, HostExperts};
use super::tp::Group;
use super::{BW, Buf, Geo, LayerState, LayerW, Part, Shared, Trace};
use memra_gguf::safetensors::StModel;
use std::sync::Arc;
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
    /// positions its compressed KV and index keys are sized for at their first allocation
    pub cap: usize,
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
/// first `keep` layers kept on the card once loaded), or, after [`Runner::experts_off_card`],
/// every layer's own weights resident and the routed experts behind each rank's expert cache.
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
    /// the routed experts in pinned host memory, when they are off the card
    host: Option<Arc<HostExperts>>,
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
            host: None,
            stats: Stats::default(),
        })
    }

    /// Keep the first `keep` layers on the card from the next forward on (fewer: free the rest).
    /// A long prompt's prefill wants the memory; the decode forwards after it want the layers.
    /// With the experts off the card every layer stays resident and this does nothing.
    pub fn set_keep(&mut self, keep: usize) {
        if self.host.is_some() {
            return;
        }
        self.keep = keep.min(self.n_layers);
        for w in self.resident.iter_mut().skip(self.keep) {
            *w = None;
        }
    }

    /// Free and total bytes on each rank's device as the allocator sees them: the driver's free
    /// memory plus the blocks the async pool keeps cached (`Engine::pool_cached_bytes`).
    pub fn device_mem(&self) -> Result<Vec<(usize, usize)>, String> {
        let mut out = Vec::with_capacity(self.grp.n());
        for (r, e) in self.grp.engines.iter().enumerate() {
            let (free, total) = e
                .ctx()
                .mem_get_info()
                .map_err(|x| format!("rank {r} mem_get_info: {x:?}"))?;
            out.push((free + e.pool_cached_bytes(), total));
        }
        Ok(out)
    }

    /// The distinct devices of the group with their ranks, in rank order of first use.
    pub fn device_ranks(&self) -> Vec<(usize, Vec<usize>)> {
        let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
        for (r, &d) in self.grp.devices.iter().enumerate() {
            match out.iter_mut().find(|x| x.0 == d) {
                Some(x) => x.1.push(r),
                None => out.push((d, vec![r])),
            }
        }
        out
    }

    /// Load the first `keep` layers' shares and the head onto the ranks now and keep them, so no
    /// forward streams them. Before each layer, every device must still hold `reserve` bytes plus
    /// what the layers loaded so far projected over the ones left; otherwise this refuses before
    /// anything runs out of memory. `reserve` is what the runs after the load need (the longest
    /// prefill's workspace and the sequences' state, [`prefill_workspace_bytes`],
    /// [`seq_state_bytes`]). Returns the report it prints.
    pub fn load_resident(&mut self, keep: usize, reserve: usize) -> Result<Residency, String> {
        let keep = keep.min(self.n_layers);
        self.keep = keep;
        let devs = self.device_ranks();
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        let start = self.device_mem()?;
        let t0 = Instant::now();
        let first_rank = |d: usize| devs.iter().find(|x| x.0 == d).map(|x| x.1[0]).unwrap();
        for li in 0..keep {
            if self.resident[li].is_some() {
                continue;
            }
            let now = self.device_mem()?;
            if li >= 2 {
                for (d, _) in &devs {
                    let r = first_rank(*d);
                    let used = start[r].0.saturating_sub(now[r].0);
                    let per = used / li;
                    let need = per * (keep - li) + reserve;
                    if now[r].0 < need {
                        return Err(format!(
                            "device {d}: {} layers resident would need {:.1} GiB more ({:.2} GiB a layer for its ranks) plus {:.1} GiB reserve; {:.1} GiB free: does not fit",
                            keep,
                            gib(per * (keep - li)),
                            gib(per),
                            gib(reserve),
                            gib(now[r].0)
                        ));
                    }
                }
            }
            let lw = self.grp.load_layer(self.stm, &self.g, li, false)?;
            self.stats.layers_loaded += 1;
            self.resident[li] = Some(lw);
        }
        if self.head.is_none() {
            self.head = Some(self.grp.load_head(self.stm, &self.g)?);
        }
        self.grp.sync_all()?;
        let load_s = t0.elapsed().as_secs_f64();
        let after = self.device_mem()?;
        let mut rep = Residency {
            layers: keep,
            of: self.n_layers,
            load_s,
            reserve,
            devices: Vec::new(),
        };
        for (d, ranks) in &devs {
            let r = ranks[0];
            let resident = start[r].0.saturating_sub(after[r].0);
            // layers left to stream need room for one layer's shares at a time
            let stream = if keep < self.n_layers && keep > 0 {
                resident / keep
            } else {
                0
            };
            rep.devices.push(DeviceBudget {
                device: *d,
                ranks: ranks.clone(),
                total: after[r].1,
                free_before: start[r].0,
                resident,
                free_after: after[r].0,
            });
            if after[r].0 < reserve + stream {
                return Err(format!(
                    "device {d}: {:.1} GiB free after the load, the runs need {:.1} GiB (+{:.1} GiB for a streamed layer): does not fit",
                    gib(after[r].0),
                    gib(reserve),
                    gib(stream)
                ));
            }
        }
        for b in &rep.devices {
            eprintln!(
                "[dsv41-resident] device {} ranks {:?}: {} of {} layers + head resident, {:.2} GiB ({:.2} GiB a rank); free {:.2} -> {:.2} GiB of {:.2} GiB; reserve {:.2} GiB; load {:.1}s",
                b.device,
                b.ranks,
                keep,
                self.n_layers,
                gib(b.resident),
                gib(b.resident / b.ranks.len().max(1)),
                gib(b.free_before),
                gib(b.free_after),
                gib(b.total),
                gib(reserve),
                load_s
            );
        }
        Ok(rep)
    }

    /// The single-card residency: load every trunk layer's own weights (attention, compressor,
    /// indexer, router, shared expert, mHC, Engram projections) and the head onto the card for
    /// good, leave the routed experts in `host`, and give each rank an expert cache of its range
    /// with no slabs yet ([`Self::grow_caches`] sizes them). Returns the device bytes the
    /// resident weights took on each rank.
    pub fn experts_off_card(&mut self, host: Arc<HostExperts>) -> Result<Vec<u64>, String> {
        if host.n_layers < self.n_layers || host.n_experts != self.g.n_experts {
            return Err(format!(
                "the host store holds {} layers of {} experts; the runner needs {} of {}",
                host.n_layers, host.n_experts, self.n_layers, self.g.n_experts
            ));
        }
        let g = self.g.clone();
        let n = self.grp.n();
        let mut before = Vec::with_capacity(n);
        for c in &self.grp.cx {
            before.push(super::experts::device_memory(&c.ops)?.0);
        }
        for li in 0..self.n_layers {
            self.resident[li] = Some(self.grp.load_layer_with(self.stm, &g, li, false, false)?);
        }
        if self.head.is_none() {
            self.head = Some(self.grp.load_head(self.stm, &g)?);
        }
        self.keep = self.n_layers;
        // one slice can route to every expert of the rank's range: that many slots is the floor
        let per_rank = g.n_experts / n;
        for (r, c) in self.grp.cx.iter_mut().enumerate() {
            debug_assert_eq!(Part { rank: r, n }.range(g.n_experts).len(), per_rank);
            c.experts = Some(ExpertCache::new(&c.ops, host.clone(), per_rank)?);
        }
        self.host = Some(host);
        let mut used = Vec::with_capacity(n);
        for (c, b) in self.grp.cx.iter().zip(before) {
            used.push(b.saturating_sub(super::experts::device_memory(&c.ops)?.0));
        }
        Ok(used)
    }

    /// Whether the routed experts are off the card.
    pub fn offloaded(&self) -> bool {
        self.host.is_some()
    }

    /// Grow every rank's expert cache into its effective free memory less `reserve` bytes, up to
    /// `max_slots` slots per rank (at least the floor). Returns the slots per rank.
    pub fn grow_caches(&mut self, reserve: u64, max_slots: usize) -> Result<Vec<usize>, String> {
        let mut out = Vec::with_capacity(self.grp.n());
        for c in self.grp.cx.iter_mut() {
            let (free, _) = super::experts::device_memory(&c.ops)?;
            let budget = free.saturating_sub(reserve) as usize;
            let Some(cache) = c.experts.as_mut() else {
                return Err("the experts are on the card: no expert cache".into());
            };
            cache.grow(&c.ops, budget, max_slots)?;
            if cache.slots() < cache.floor_slots {
                return Err(format!(
                    "the expert cache needs {} slots ({} bytes) for one slice and {} fit beside the \
                     resident weights and the {reserve}-byte reserve",
                    cache.floor_slots,
                    cache.floor_slabs() * cache.slab_bytes(),
                    cache.slots()
                ));
            }
            out.push(cache.slots());
        }
        Ok(out)
    }

    /// Give back expert-cache slabs on every rank until each has `need` effective free bytes, or
    /// only the floor is left. Returns each rank's effective free bytes after.
    pub fn make_room(&mut self, need: u64) -> Result<Vec<u64>, String> {
        let mut out = Vec::with_capacity(self.grp.n());
        for c in self.grp.cx.iter_mut() {
            let (free, _) = super::experts::device_memory(&c.ops)?;
            if free < need
                && let Some(cache) = c.experts.as_mut()
            {
                cache.release(&c.ops, (need - free) as usize)?;
            }
            out.push(super::experts::device_memory(&c.ops)?.0);
        }
        Ok(out)
    }

    /// Every rank's expert-cache counters, summed, and the slots per rank.
    pub fn cache_stats(&self) -> (CacheStats, Vec<usize>) {
        let mut st = CacheStats::default();
        let mut slots = Vec::new();
        for c in &self.grp.cx {
            if let Some(cache) = c.experts.as_ref() {
                st.add(&cache.stats);
                slots.push(cache.slots());
            }
        }
        (st, slots)
    }

    /// The device bytes the cache can give back on each rank (its slabs above the floor).
    pub fn cache_reclaimable(&self) -> Vec<u64> {
        self.grp
            .cx
            .iter()
            .map(|c| {
                c.experts.as_ref().map_or(0, |cache| {
                    (cache
                        .slots()
                        .saturating_sub(cache.floor_slabs() * super::experts::SLAB_SLOTS)
                        * cache.slab_bytes()
                        / super::experts::SLAB_SLOTS) as u64
                })
            })
            .collect()
    }

    /// An empty sequence for this runner's ranks and layers, its state sized for `max_pos`.
    pub fn seq(&self) -> Seq {
        self.seq_with_capacity(self.max_pos)
    }

    /// An empty sequence whose compressed KV and index keys are first allocated for `cap`
    /// positions (a short request in a long-context runner holds only what it uses).
    pub fn seq_with_capacity(&self, cap: usize) -> Seq {
        let n = self.grp.n();
        let per_rank =
            |_: usize| -> Vec<LayerState> { (0..n).map(|_| LayerState::default()).collect() };
        Seq {
            cap: cap.clamp(1, self.max_pos),
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

    /// The bf16 embedding rows of `tokens` `[s][dim]` (the `hc` copies are made on the device)
    /// and the one-hot carried pre-mix `[s][hc]`.
    fn embed(&self, tokens: &[u32]) -> Result<(Vec<u8>, Vec<f32>), String> {
        let (hc, d) = (self.g.hc, self.g.dim);
        let (ei, eb) = self.stm.raw("embed.weight").ok_or("embed.weight missing")?;
        if ei.dtype != "BF16" || ei.shape != [self.g.vocab as u64, d as u64] {
            return Err(format!("embed.weight: {} {:?}", ei.dtype, ei.shape));
        }
        let s = tokens.len();
        let mut hb = vec![0u8; s * d * 2];
        for (t, &id) in tokens.iter().enumerate() {
            let id = id as usize;
            if id >= self.g.vocab {
                return Err(format!("token {id} outside the vocabulary"));
            }
            hb[t * d * 2..(t + 1) * d * 2].copy_from_slice(&eb[id * d * 2..(id + 1) * d * 2]);
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
            let (eb, pm) = self.embed(&c.tokens)?;
            let s = c.tokens.len();
            let (hc, d) = (g.hc, g.dim);
            // each rank uploads the rows once and copies them into the hc stream slots
            let up = self.grp.each(|_, cx| {
                let e = Buf::from_bytes(&cx.ops.st, &eb)?;
                let hh = cx.ops.buf(s * hc * d * 2)?;
                for k in 0..hc {
                    super::copy_2d(&cx.ops, &e, 0, d * 2, &hh, k * d * 2, hc * d * 2, d * 2, s)?;
                }
                Ok((hh, Buf::from_f32(&cx.ops.st, &pm)?))
            })?;
            let (hr, pr): (Vec<Buf>, Vec<Buf>) = up.into_iter().unzip();
            h.push(hr);
            pre.push(pr);
        }
        let mut fwd: Vec<Vec<Shared>> = calls
            .iter()
            .map(|_| (0..n).map(|_| Shared::default()).collect())
            .collect();
        let mut tr: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
        let rows: usize = pos.iter().map(|&(_, s)| s).sum();
        for li in 0..self.n_layers {
            let tl = Instant::now();
            // a resident layer is borrowed, so an error part-way never loses it
            let mut streamed: Option<Vec<LayerW>> = None;
            let lw: &Vec<LayerW> = match self.resident[li].as_ref() {
                Some(w) => w,
                None if self.host.is_some() => {
                    return Err(format!(
                        "layer {li} is not resident with the experts off the card"
                    ));
                }
                None => {
                    self.stats.layers_loaded += 1;
                    streamed.insert(self.grp.load_layer(self.stm, &g, li, false)?)
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
                for cx in self.grp.cx.iter_mut() {
                    cx.rows_hint = q.cap;
                }
                if let Some(sl) = slot {
                    let ec = self.ec.as_ref().unwrap();
                    let (nl, cols) = (ec.layer_ids.len(), ec.cols());
                    let all = hashes[ci].as_ref().ok_or("engram hashes missing")?;
                    let mine: Vec<i64> = (0..s)
                        .flat_map(|t| all[(t * nl + sl) * cols..(t * nl + sl + 1) * cols].to_vec())
                        .collect();
                    let stm = self.stm;
                    let hc_ = &h[ci];
                    let mut trs: Vec<&mut Trace> = tr.iter_mut().collect();
                    self.grp.each_mut(&mut trs, |r, cx, t| {
                        cx.engram(stm, &lw[r], &hc_[r], &mine, s, t)
                    })?;
                }
                if let Some(gi) = gi {
                    for (f, st) in fwd[ci].iter_mut().zip(q.store[gi].iter_mut()) {
                        f.swap_group(st);
                    }
                }
                let res = self.grp.block_forward_owned(
                    lw,
                    std::mem::take(&mut h[ci]),
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
            // a serving thread's progress sink sees every layer pass of the forward's rows
            crate::progress::note_prime_rows(rows);
            if let Some(w) = streamed
                && li < self.keep
            {
                self.resident[li] = Some(w);
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

    /// The greedy next token after a call (its last row's logits, lowest id among equal maxima),
    /// without the log-sum-exp and top-k a scored step also keeps.
    pub fn argmax_last(&mut self, out: &Out) -> Result<u32, String> {
        let lg = self.logits(out, out.s - 1, 1)?;
        if lg.iter().any(|x| !x.is_finite()) {
            return Err("non-finite logits".into());
        }
        Ok(argmax(&lg) as u32)
    }

    /// The scored next-token distribution after a call: its last row's logits reduced.
    pub fn score_last(&mut self, out: &Out, keep_logits: bool) -> Result<Scored, String> {
        let lg = self.logits(out, out.s - 1, 1)?;
        Scored::of(lg, keep_logits)
    }
}

/// What [`Runner::load_resident`] placed, per device.
#[derive(Debug, Clone)]
pub struct Residency {
    pub layers: usize,
    pub of: usize,
    pub load_s: f64,
    pub reserve: usize,
    pub devices: Vec<DeviceBudget>,
}

#[derive(Debug, Clone)]
pub struct DeviceBudget {
    pub device: usize,
    pub ranks: Vec<usize>,
    pub total: usize,
    pub free_before: usize,
    /// bytes the resident layers and the head took on this device (all its ranks)
    pub resident: usize,
    pub free_after: usize,
}

/// Device bytes one sequence's carried state takes on each rank when it is allocated for `cap`
/// positions (`Seq::cap`): every kv-source group's compressed KV (`cap / ratio` rows of
/// `head_dim` bf16) and, where the source owns index keys, its index keys (`cap / ratio` rows of
/// `index_dim` bf16); every layer's window ring (`window` rows of `head_dim` bf16); each ratio-2
/// source's open pair (two fp32 rows of kv and of gate). Replicated: the same on every rank.
pub fn seq_state_bytes(g: &Geo, n_layers: usize, cap: usize) -> usize {
    let hd = g.head_dim;
    let mut b = n_layers * g.window * hd * 2;
    for &k in &g.kv_sources {
        if k >= n_layers {
            continue;
        }
        let r = g.compress_ratios[k];
        if r == 0 {
            continue;
        }
        b += (cap / r) * hd * 2;
        if g.index_sources.contains(&k) {
            b += (cap / r) * g.index_dim * 2;
        }
        if r > 1 {
            b += 2 * 2 * hd * 4;
        }
    }
    b
}

/// An upper estimate of the device bytes one rank needs at once for a prefill of `s` tokens on
/// `n` ranks with query slices of `chunk` rows, besides the weights and the carried state: the
/// `[s][hc][dim]` stream (the mHC posts write in place), the larger of the attention phase (its
/// input and output rows, the query latent, the KV rows the heads read, the selection, this
/// rank's candidate rows, one slice's index score and head buffers) and the MoE phase (its input
/// and output rows, one slice's expert pairs and buffers), the mHC mixes, Engram's slice buffers,
/// plus 10% and 512 MiB for the allocator and the kernels' own scratch. The runs measure the
/// real peak (the pool's high-water mark) next to it.
pub fn prefill_workspace_bytes(g: &Geo, n: usize, s: usize, chunk: usize) -> usize {
    let (d, hd, hc) = (g.dim, g.head_dim, g.hc);
    let c = chunk.min(s).max(1);
    let n_c = s; // ratio 1 has the most compressed rows
    let stream = s * hc * d * 2;
    let rows = s * d * 2;
    let attn = 2 * rows
        + s * g.q_lora * 2
        + (s + n_c) * hd * 2
        + s * g.index_topk * 4
        + s.div_ceil(n) * n_c.div_ceil(g.cand_block.max(1))
        + c * n_c * 2
        + c * g.index_heads * g.index_dim * 2
        + 2 * c * (g.heads / n) * hd * 2
        + c * g.o_groups * g.o_lora * 2
        + c * (g.window + g.index_topk) * 4;
    let pairs = c * g.topk;
    let moe = 2 * rows
        + pairs * d * 2
        + pairs.div_ceil(n)
            * 2
            * (d + d / 32 + 3 * g.moe_inter * 2 + g.moe_inter + g.moe_inter / 32)
        + c * (3 * g.moe_inter * 2 + d * 2 + g.n_experts * 4);
    let mixes = 2 * s * ((2 + hc) * hc + 2 * hc + hc * hc) * 4;
    let engram = c * (d * (hc + 1) * 2 + g.engram_cols() * g.engram_head_dim * 3);
    let peak = stream + attn.max(moe) + mixes + engram;
    peak + peak / 10 + (512 << 20)
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
