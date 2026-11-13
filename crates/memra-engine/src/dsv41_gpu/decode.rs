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
//! fed as a decode call, never prefilled again.
//!
//! Several tokens at a position past 0 are a drafted block's verification: every row runs the
//! one-token decode program of its own position (its ring order, the compressor's pair, the
//! indexer over its own visible keys, its Engram row) in one pass through each block, reading the
//! state as it stood plus the rows before it. Every kernel acts per row, so a row's bytes are
//! those of feeding the tokens one call at a time (`dsv41-gate verify-rows`). Such a call writes
//! no carried state: the sequence is pending until [`Runner::commit`] keeps a prefix of the rows
//! (the accepted tokens) and drops the rest, so a rejected draft never reaches the ring, the
//! compressor's open pair, the row counts or the token history.
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
//! Several sequences can take their next token in one call ([`Runner::forward_batch`]): the
//! rows of the call are the sequences' one-token steps, each at its own position against its own
//! ring, compressor pair, compressed rows, index keys, candidate pool and token history, run
//! through each block's launches together. What acts per row runs once for all rows; what reads
//! one sequence's buffers (its ring copy, a ratio-2 pair's pooling, the index score over its
//! keys, its candidates and top-k, its compressed rows' copy) runs per sequence
//! ([`super::Ctx::attn_keys_batch`]). A row's bytes are those of its sequence's own one-token
//! call (`dsv41-spec batch-rows`).
//!
//! The ranks are a [`Group`]: one rank is the one-card program (with the grouped expert dispatch,
//! byte-identical to the per-expert loop), N ranks the TP/EP partition, which writes the same
//! bytes.

use super::engram::EngramConsts;
use super::experts::{CacheStats, ExpertCache, HostExperts};
use super::tp::Group;
use super::{BW, BatchRow, BatchSel, Buf, Dp, Geo, LayerState, LayerW, Part, Shared, Trace};
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
    /// with [`Runner::set_taps`]: on every rank the draft head's `main_hidden` rows, bf16
    /// `[tap_rows][taps * dim]`, of the call's last `tap_rows` positions (from row `tap_row0`)
    pub taps: Vec<Buf>,
    pub tap_row0: usize,
    pub tap_rows: usize,
    /// a batched call ([`Runner::forward_batch`]): each row's own position (its sequence's);
    /// empty for a call of one sequence, whose rows are consecutive from `sp`
    pub batch: Vec<usize>,
}

impl Out {
    /// The position of the call's row `row`.
    pub fn pos_of(&self, row: usize) -> usize {
        self.batch.get(row).copied().unwrap_or(self.sp + row)
    }
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
    /// `(position, rows)` of a several-token decode call not yet committed
    pending: Option<(usize, usize)>,
}

impl Seq {
    /// Tokens in the cache (the next call's position).
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// The rows of a several-token decode call waiting for [`Runner::commit`].
    pub fn pending(&self) -> Option<(usize, usize)> {
        self.pending
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
    pub(super) stm: &'a StModel,
    ec: Option<EngramConsts>,
    pub n_layers: usize,
    /// positions the RoPE tables and the cache capacities cover
    pub max_pos: usize,
    keep: usize,
    resident: Vec<Option<Vec<LayerW>>>,
    pub(super) head: Option<(Vec<Buf>, Vec<BW>)>,
    /// the routed experts in pinned host memory, when they are off the card
    host: Option<Arc<HostExperts>>,
    /// the draft head's target layers: a forward keeps the mean over the stream copies of each
    /// one's input ([`Out::taps`])
    taps: Vec<usize>,
    /// f32 `[window][hc]` of 1 / hc on every rank (the mean as a collapse)
    tap_mix: Vec<Buf>,
    /// with the ranks' trace on: the router's selection of the last batched call, i32
    /// `[layers][rows][top-k]` (a measurement: how many experts a batch reads)
    pub last_routes: Vec<i32>,
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
            taps: Vec::new(),
            tap_mix: Vec::new(),
            last_routes: Vec::new(),
            stats: Stats::default(),
        })
    }

    /// Keep, from the next forward on, the draft head's inputs: for each call the mean over the
    /// `hc` stream copies at the input of each layer of `layers` (after its Engram, before its
    /// attention; vendor `Transformer.forward`: `h.mean(dim=2)` at `dspark_target_layer_ids`), for
    /// the call's last `window` positions. Empty: none.
    pub fn set_taps(&mut self, layers: &[usize]) -> Result<(), String> {
        if layers.iter().any(|&l| l >= self.n_layers) || !layers.is_sorted() {
            return Err(format!(
                "tap layers {layers:?}: ascending trunk layers below {}",
                self.n_layers
            ));
        }
        let (hc, win) = (self.g.hc, self.g.window);
        self.taps = layers.to_vec();
        self.tap_mix = if layers.is_empty() {
            Vec::new()
        } else {
            let mix = vec![1.0f32 / hc as f32; win * hc];
            self.grp.each(|_, cx| Buf::from_f32(&cx.ops.st, &mix))?
        };
        Ok(())
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

    /// Whether the group lost its transport ([`super::tp::Group::lost`]): every later forward
    /// refuses, so a serving route stops instead of failing each step.
    pub fn transport_lost(&self) -> Option<&str> {
        self.grp.lost.as_deref()
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
            pending: None,
        }
    }

    /// [`Self::seq_with_capacity`] with its state allocated now on every rank, as the forwards
    /// allocate it at its first use (zeroed, the same sizes): each layer's window ring, a ratio-2
    /// kv source's open pair, and each kv-source group's compressed KV and its index keys sized
    /// for `cap` positions. A serving route books a request's state at admission ([`seq_state_bytes`]);
    /// holding it from there keeps the expert cache's regrowth between steps from taking memory
    /// the request's first emissions need. The bytes a forward writes are the same either way.
    pub fn seq_alloc(&self, cap: usize) -> Result<Seq, String> {
        let mut q = self.seq_with_capacity(cap);
        let g = &self.g;
        let (hd, win, idim) = (g.head_dim, g.window, g.index_dim);
        for (li, per) in q.ls.iter_mut().enumerate() {
            let k = super::LayerKind::of(g, li);
            for (ls, c) in per.iter_mut().zip(&self.grp.cx) {
                ls.ring = Some(c.ops.buf(win * hd * 2)?);
                if k.ratio == 2 && k.kv_source {
                    ls.c_kv = Some(c.ops.buf(2 * hd * 4)?);
                    ls.c_sc = Some(c.ops.buf(2 * hd * 4)?);
                }
            }
        }
        for (gi, per) in q.store.iter_mut().enumerate() {
            let src = g.kv_sources[gi];
            let k = super::LayerKind::of(g, src);
            let rows = q.cap.checked_div(k.ratio).unwrap_or(0);
            if src >= self.n_layers || rows == 0 {
                continue;
            }
            for (sh, c) in per.iter_mut().zip(&self.grp.cx) {
                super::ensure_rows(
                    &c.ops,
                    &mut sh.compress_kv,
                    &mut sh.compress_cap,
                    rows,
                    hd * 2,
                    rows,
                )?;
                if k.index_source {
                    super::ensure_rows(
                        &c.ops,
                        &mut sh.index_k,
                        &mut sh.index_k_cap,
                        rows,
                        idim * 2,
                        rows,
                    )?;
                }
            }
        }
        Ok(q)
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
    pub(super) fn embed(&self, tokens: &[u32]) -> Result<(Vec<u8>, Vec<f32>), String> {
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
        self.grp.live()?;
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
            if let Some((p, n)) = seqs[c.seq].pending {
                return Err(format!(
                    "sequence {}: {n} rows at position {p} wait for a commit",
                    c.seq
                ));
            }
            if sp > 0 && s > 1 && calls.iter().filter(|x| x.seq == c.seq).count() > 1 {
                return Err(format!(
                    "sequence {}: a call of {s} decode tokens is the sequence's only call of its forward",
                    c.seq
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
        for c in calls {
            let q = &mut seqs[c.seq];
            // the call's tokens are checked (Engram hashes, embedding rows) before the sequence
            // takes them
            hashes.push(match (&self.ec, engram) {
                (Some(ec), true) => Some(ec.hash_after(&q.ids, &c.tokens)?),
                _ => None,
            });
            let (eb, pm) = self.embed(&c.tokens)?;
            q.ids.extend_from_slice(&c.tokens);
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
        // the draft head's inputs: each call's last `window` rows at every target layer
        let nt = self.taps.len();
        let tap_span = |s: usize| (s - s.min(g.window), s.min(g.window));
        let mut taps: Vec<Vec<Buf>> = Vec::with_capacity(calls.len());
        for &(_, s) in &pos {
            taps.push(if nt == 0 {
                Vec::new()
            } else {
                let (_, tr_) = tap_span(s);
                self.grp.each(|_, cx| cx.ops.tmp(tr_ * nt * g.dim * 2))?
            });
        }
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
                if let Some(ti) = self.taps.iter().position(|&l| l == li) {
                    let (t0, t_rows) = tap_span(s);
                    let (hc, d) = (g.hc, g.dim);
                    let (hs, tb, mix) = (&h[ci], &taps[ci], &self.tap_mix);
                    self.grp.each(|r, cx| {
                        let st = cx.ops.st.clone();
                        super::ck("tap mean", unsafe {
                            super::ffi::memra_v41_hc_pre(
                                hs[r].at(&st, t0 * hc * d * 2).c(),
                                mix[r].p(&st).f(),
                                t_rows as i32,
                                hc as i32,
                                d as i32,
                                tb[r].at(&st, ti * d * 2).m(),
                                (nt * d) as i64,
                                cx.ops.sv(),
                            )
                        })
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
        for (c, &(sp, s)) in calls.iter().zip(&pos) {
            if sp > 0 && s > 1 {
                seqs[c.seq].pending = Some((sp, s));
            }
        }
        if self.grp.graphs.on {
            // a replayed layer's pre-mix is its capture's buffer, rewritten by the layer's next
            // replay: a call's result gets its own
            for p in pre.iter_mut() {
                let own = self.grp.each(|r, c| super::clone_buf(&c.ops, &p[r]))?;
                *p = own;
            }
        }
        Ok(h.into_iter()
            .zip(pre)
            .zip(pos)
            .zip(taps)
            .map(|(((h, pre), (sp, s)), taps)| {
                let (tap_row0, tap_rows) = if taps.is_empty() { (0, 0) } else { tap_span(s) };
                Out {
                    h,
                    pre,
                    sp,
                    s,
                    taps,
                    tap_row0,
                    tap_rows,
                    batch: Vec::new(),
                }
            })
            .collect())
    }

    /// One decode step of several sequences in one call: `rows[i]` is `(sequence, token)`, the
    /// sequences distinct and in ascending order, each past its prefill with no call open. The
    /// returned [`Out`] holds the rows' streams side by side (row i is `rows[i]`), so
    /// [`Self::logits`] and [`Self::argmax_rows`] take all of them in one head pass, and its
    /// taps hold every row. Each row is its sequence's one-token decode call, byte for byte.
    pub fn forward_batch(
        &mut self,
        seqs: &mut [Seq],
        rows: &[(usize, u32)],
    ) -> Result<Out, String> {
        self.grp.live()?;
        let n = self.grp.n();
        let g = self.g.clone();
        let nr = rows.len();
        if nr == 0 {
            return Err("a batched call needs at least one row".into());
        }
        let mut positions = Vec::with_capacity(nr);
        for (i, &(si, _)) in rows.iter().enumerate() {
            if i > 0 && rows[i - 1].0 >= si {
                return Err("a batched call takes distinct sequences in ascending order".into());
            }
            let q = seqs
                .get(si)
                .ok_or_else(|| format!("batched row on sequence {si} of {}", seqs.len()))?;
            if q.pending.is_some() {
                return Err(format!("sequence {si}: rows wait for a commit"));
            }
            let p = q.len();
            if p == 0 {
                return Err(format!("sequence {si}: a batched row follows the prefill"));
            }
            if p + 1 > self.max_pos {
                return Err(format!(
                    "position {} past the runner's {} (RoPE tables and caches)",
                    p + 1,
                    self.max_pos
                ));
            }
            positions.push(p);
        }
        let engram = g.engram_layers.iter().any(|&l| l < self.n_layers);
        let tokens: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let mut hashes: Vec<Vec<i64>> = Vec::with_capacity(nr);
        if let (Some(ec), true) = (&self.ec, engram) {
            for &(si, t) in rows {
                hashes.push(ec.hash_after(&seqs[si].ids, &[t])?);
            }
        }
        let (eb, pm) = self.embed(&tokens)?;
        // the rows' tokens are checked (Engram hashes, embedding rows): the sequences take them
        for &(si, t) in rows {
            seqs[si].ids.push(t);
        }
        let (hc, d) = (g.hc, g.dim);
        let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        // `[row][rank]`: each row's CSA2 state for this call (its group's store swapped in per layer)
        let mut fwd: Vec<Vec<Shared>> = (0..nr)
            .map(|_| (0..n).map(|_| Shared::default()).collect())
            .collect();
        // A step whose every layer is on the cards and that keeps no trace takes its host data
        // from one block a rank (`StepBlock`), written here before the first launch: positions,
        // pre-mix, embedding rows, every Engram layer's rows, every layer's tables.
        let stepped = self.grp.cx.iter().all(|c| c.keyed && !c.trace)
            && self.resident[..self.n_layers].iter().all(|w| w.is_some());
        let layer_tab = super::step_layer_bytes(nr);
        let (o_pos, o_pm) = (0usize, (nr * 4).next_multiple_of(8));
        // the embedding rows start on a row boundary of their own pitch (`copy_2d` takes a row)
        let o_emb = (o_pm + nr * hc * 4).next_multiple_of(d * 2);
        let mut o_next = o_emb + (nr * d * 2).next_multiple_of(8);
        // (layer, rows offset, scales offset) of each Engram layer's rows in the block
        let mut eng_at: Vec<(usize, usize, usize)> = Vec::new();
        let (ecols, ehd) = (g.engram_cols(), g.engram_head_dim);
        if stepped && self.ec.is_some() && engram {
            for &l in g.engram_layers.iter().filter(|&&l| l < self.n_layers) {
                let rows_at = o_next;
                let sc_at = rows_at + (nr * ecols * ehd).next_multiple_of(8);
                o_next = sc_at + (nr * ecols * ehd / 32).next_multiple_of(8);
                eng_at.push((l, rows_at, sc_at));
            }
        }
        let o_layers = o_next;
        let mut steps: Vec<super::StepLayer> = Vec::new();
        let mut blk: Vec<Dp> = Vec::new();
        if stepped {
            let len = o_layers + self.n_layers * layer_tab;
            let mut head = vec![0u8; o_layers];
            head[o_pos..o_pos + nr * 4].copy_from_slice(&super::i32_bytes(&pos_i32));
            head[o_pm..o_pm + nr * hc * 4].copy_from_slice(&super::f32_bytes(&pm));
            head[o_emb..o_emb + eb.len()].copy_from_slice(&eb);
            if let Some(ec) = &self.ec {
                let (nl, cols) = (ec.layer_ids.len(), ec.cols());
                for &(l, rows_at, sc_at) in &eng_at {
                    let sl = ec
                        .layer_ids
                        .iter()
                        .position(|&x| x == l)
                        .ok_or("engram layer not in the constants")?;
                    let mut mine: Vec<i64> = Vec::with_capacity(nr * cols);
                    for hs in &hashes {
                        debug_assert_eq!(hs.len(), nl * cols);
                        mine.extend_from_slice(&hs[sl * cols..(sl + 1) * cols]);
                    }
                    let ew = self.resident[l].as_ref().unwrap()[0]
                        .engram
                        .as_ref()
                        .ok_or("layer has no engram")?;
                    let (er, es) =
                        super::engram::gather_host(self.stm, &ew.table, ew.rows, ehd, &mine)?;
                    head[rows_at..rows_at + er.len()].copy_from_slice(&er);
                    head[sc_at..sc_at + es.len()].copy_from_slice(&es);
                }
            }
            let mut bytes: Vec<Vec<u8>> = (0..n)
                .map(|_| {
                    let mut b = head.clone();
                    b.resize(len, 0);
                    b
                })
                .collect();
            // every layer's tables, in layer order (a kv source sizes and counts what its rows
            // append before the layers of its group read the count)
            let mut carried: Vec<(usize, Vec<bool>)> = (0..n).map(|_| (0, Vec::new())).collect();
            for li in 0..self.n_layers {
                let kind = super::LayerKind::of(&g, li);
                let gi = Self::group_of(&g, li);
                let at = o_layers + li * layer_tab;
                let mut this = super::StepLayer {
                    off: at,
                    ..Default::default()
                };
                let mut next = 0usize;
                let mut by_rank: Vec<Vec<BatchRow<'_>>> =
                    (0..n).map(|_| Vec::with_capacity(nr)).collect();
                let mut fwd_it = fwd.iter_mut();
                for (si, q) in seqs.iter_mut().enumerate() {
                    if next >= nr || rows[next].0 != si {
                        continue;
                    }
                    let f = fwd_it.next().expect("one state per row");
                    if let Some(gi) = gi {
                        for (fr, st) in f.iter_mut().zip(q.store[gi].iter_mut()) {
                            fr.swap_group(st);
                        }
                    }
                    let cap = q.cap;
                    for (r, (l, fr)) in q.ls[li].iter_mut().zip(f.iter_mut()).enumerate() {
                        by_rank[r].push(BatchRow {
                            pos: positions[next],
                            cap,
                            ls: l,
                            sh: fr,
                        });
                    }
                    next += 1;
                }
                let mut res = Ok(());
                for (r, rr) in by_rank.iter_mut().enumerate() {
                    match self.grp.cx[r].plan_keys_step(
                        kind,
                        rr,
                        &mut carried[r],
                        &mut bytes[r][at..at + layer_tab],
                    ) {
                        Ok((tw, has)) => {
                            this.tw = tw;
                            this.has = has;
                        }
                        Err(e) => {
                            res = Err(e);
                            break;
                        }
                    }
                }
                drop(by_rank);
                if let Some(gi) = gi {
                    for (f, &(si, _)) in fwd.iter_mut().zip(rows) {
                        for (fr, st) in f.iter_mut().zip(seqs[si].store[gi].iter_mut()) {
                            fr.swap_group(st);
                        }
                    }
                }
                res?;
                steps.push(this);
            }
            let br = &bytes;
            // room for the widest small-M batch, so the block stays where it is while the
            // batch changes width (every section is linear in the rows, up to its padding)
            let cap = len.max(len.div_ceil(nr) * super::SMALL_M + 64 * (self.n_layers + 8) + d * 2);
            blk = self.grp.each(|r, cx| {
                if cx.step.as_ref().is_none_or(|b| b.cap() < len) {
                    cx.step = Some(super::StepBlock::new(&cx.ops, cap)?);
                }
                let b = cx.step.as_mut().unwrap();
                b.upload(&cx.ops, &br[r])?;
                Ok(b.dev.p(&cx.ops.st))
            })?;
        }
        let up = if stepped {
            let bl = &blk;
            self.grp.each(|r, cx| {
                let b = cx.step.as_ref().unwrap();
                let hh = cx.ops.tmp(nr * hc * d * 2)?;
                for k in 0..hc {
                    super::copy_2d(
                        &cx.ops,
                        &b.dev,
                        o_emb / (d * 2),
                        d * 2,
                        &hh,
                        k * d * 2,
                        hc * d * 2,
                        d * 2,
                        nr,
                    )?;
                }
                let pre = cx.ops.tmp(nr * hc * 4)?;
                super::copy_d2d(&cx.ops, &b.dev, o_pm, &pre, 0, nr * hc * 4)?;
                Ok((hh, pre, None, Dp(bl[r].addr() + o_pos as u64)))
            })?
        } else {
            self.grp.each(|_, cx| {
                let e = Buf::from_bytes(&cx.ops.st, &eb)?;
                let hh = cx.ops.buf(nr * hc * d * 2)?;
                for k in 0..hc {
                    super::copy_2d(&cx.ops, &e, 0, d * 2, &hh, k * d * 2, hc * d * 2, d * 2, nr)?;
                }
                let pb = Buf::from_i32(&cx.ops.st, &pos_i32)?;
                let pp = pb.p(&cx.ops.st);
                Ok((hh, Buf::from_f32(&cx.ops.st, &pm)?, Some(pb), pp))
            })?
        };
        let (mut h, mut pre, mut pos, mut pos_keep) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (a, b, keep, c) in up {
            h.push(a);
            pre.push(b);
            pos_keep.push(keep);
            pos.push(c);
        }
        let mut sel: Vec<BatchSel> = (0..n).map(|_| BatchSel::default()).collect();
        let mut tr: Vec<Trace> = (0..n).map(|_| Trace::default()).collect();
        let nt = self.taps.len();
        let taps: Vec<Buf> = if nt == 0 {
            Vec::new()
        } else {
            self.grp.each(|_, cx| cx.ops.tmp(nr * nt * d * 2))?
        };
        for li in 0..self.n_layers {
            super::tally::layer(li);
            let tl = Instant::now();
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
            if let (Some(ec), true) = (&self.ec, lw[0].engram.is_some()) {
                let sl = ec
                    .layer_ids
                    .iter()
                    .position(|&x| x == li)
                    .ok_or("engram layer not in the constants")?;
                let (nl, cols) = (ec.layer_ids.len(), ec.cols());
                let mut mine: Vec<i64> = Vec::with_capacity(nr * cols);
                for hs in &hashes {
                    mine.extend_from_slice(&hs[sl * cols..(sl + 1) * cols]);
                }
                debug_assert!(hashes.iter().all(|x| x.len() == nl * cols));
                let stm = self.stm;
                let hr = &h;
                if let Some(&(_, rows_at, sc_at)) = eng_at.iter().find(|x| x.0 == li) {
                    let bl = &blk;
                    self.grp.each(|r, cx| {
                        cx.engram_step(
                            &lw[r],
                            &hr[r],
                            Dp(bl[r].addr() + rows_at as u64),
                            Dp(bl[r].addr() + sc_at as u64),
                            nr,
                        )
                    })?;
                } else {
                    let mut trs: Vec<&mut Trace> = tr.iter_mut().collect();
                    self.grp.each_mut(&mut trs, |r, cx, t| {
                        cx.engram(stm, &lw[r], &hr[r], &mine, nr, t)
                    })?;
                }
            }
            if let Some(ti) = self.taps.iter().position(|&l| l == li) {
                let (hs, tb, mix) = (&h, &taps, &self.tap_mix);
                self.grp.each(|r, cx| {
                    let st = cx.ops.st.clone();
                    super::ck("tap mean", unsafe {
                        super::ffi::memra_v41_hc_pre(
                            hs[r].p(&st).c(),
                            mix[r].p(&st).f(),
                            nr as i32,
                            hc as i32,
                            d as i32,
                            tb[r].at(&st, ti * d * 2).m(),
                            (nt * d) as i64,
                            cx.ops.sv(),
                        )
                    })
                })?;
            }
            let gi = Self::group_of(&g, li);
            // every row's state of this layer, by rank: the sequences in order, their rows in order
            let mut by_rank: Vec<Vec<BatchRow<'_>>> =
                (0..n).map(|_| Vec::with_capacity(nr)).collect();
            let mut next = 0usize;
            let mut fwd_it = fwd.iter_mut();
            for (si, q) in seqs.iter_mut().enumerate() {
                if next >= nr || rows[next].0 != si {
                    continue;
                }
                let f = fwd_it.next().expect("one state per row");
                if let Some(gi) = gi {
                    for (fr, st) in f.iter_mut().zip(q.store[gi].iter_mut()) {
                        fr.swap_group(st);
                    }
                }
                let cap = q.cap;
                for (r, (l, fr)) in q.ls[li].iter_mut().zip(f.iter_mut()).enumerate() {
                    by_rank[r].push(BatchRow {
                        pos: positions[next],
                        cap,
                        ls: l,
                        sh: fr,
                    });
                }
                next += 1;
            }
            let res = self.grp.block_forward_batch(
                lw,
                std::mem::take(&mut h),
                &pre,
                &pos,
                &mut by_rank,
                &mut sel,
                steps.get(li).map(|sl| (blk.as_slice(), sl)),
                &mut tr,
            );
            drop(by_rank);
            if let Some(gi) = gi {
                for (f, &(si, _)) in fwd.iter_mut().zip(rows) {
                    for (fr, st) in f.iter_mut().zip(seqs[si].store[gi].iter_mut()) {
                        fr.swap_group(st);
                    }
                }
            }
            let (h2, p2) = res?;
            h = h2;
            pre = p2;
            self.stats.run_s += tr0.elapsed().as_secs_f64();
            crate::progress::note_prime_rows(nr);
            if let Some(w) = streamed
                && li < self.keep
            {
                self.resident[li] = Some(w);
            }
        }
        super::tally::layer(usize::MAX);
        drop(pos_keep);
        self.stats.forwards += 1;
        self.stats.calls += nr;
        if self.grp.graphs.on {
            // a replayed layer's pre-mix is its capture's buffer, rewritten by the layer's next
            // replay: the call's result gets its own
            pre = self.grp.each(|r, c| super::clone_buf(&c.ops, &pre[r]))?;
        }
        if self.grp.cx[0].trace {
            self.last_routes = std::mem::take(&mut tr[0].route_sel);
        }
        let tap_rows = if taps.is_empty() { 0 } else { nr };
        Ok(Out {
            h,
            pre,
            sp: positions[0],
            s: nr,
            taps,
            tap_row0: 0,
            tap_rows,
            batch: positions,
        })
    }

    /// Close a several-token decode call on `q`: its first `keep` rows (at least one: the first
    /// token of a verified block is the target's own) become the sequence's state on every
    /// rank, exactly as feeding those tokens one call at a time leaves it; the rows past them
    /// are dropped (ring and open-pair rows never written, the compressed rows and index keys
    /// past the kept length no longer counted, the token history cut).
    pub fn commit(&mut self, q: &mut Seq, keep: usize) -> Result<(), String> {
        let (sp, s) = q
            .pending
            .ok_or("commit: the sequence has no several-token call open")?;
        if keep == 0 || keep > s {
            return Err(format!("commit: {keep} of {s} rows"));
        }
        let g = self.g.clone();
        for li in 0..self.n_layers {
            let ls = &mut q.ls[li];
            let mut io: Vec<&mut LayerState> = ls.iter_mut().collect();
            self.grp
                .each_mut(&mut io, |_, cx, l| cx.commit_rows(l, sp, keep))?;
        }
        for (gi, &k) in g.kv_sources.iter().enumerate() {
            if k >= self.n_layers {
                continue;
            }
            let ratio = g.compress_ratios[k];
            if ratio == 0 {
                continue;
            }
            for st in q.store[gi].iter_mut() {
                st.n_c = (sp + keep) / ratio;
            }
        }
        q.ids.truncate(sp + keep);
        q.pending = None;
        Ok(())
    }

    /// Drop a several-token decode call on `q` whole: none of its rows becomes state and the
    /// sequence stands where it stood before the call.
    pub fn discard(&mut self, q: &mut Seq) -> Result<(), String> {
        let (sp, _) = q
            .pending
            .ok_or("discard: the sequence has no several-token call open")?;
        let g = self.g.clone();
        for ls in q.ls.iter_mut() {
            for l in ls.iter_mut() {
                l.pend_kv = None;
                l.pend_c = None;
            }
        }
        for (gi, &k) in g.kv_sources.iter().enumerate() {
            if k >= self.n_layers {
                continue;
            }
            let ratio = g.compress_ratios[k];
            if ratio == 0 {
                continue;
            }
            for st in q.store[gi].iter_mut() {
                st.n_c = sp / ratio;
            }
        }
        q.ids.truncate(sp);
        q.pending = None;
        Ok(())
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

    /// The greedy tokens of rows `[r0, r0 + rows)` of a call, reduced on the device: the head as
    /// [`Self::logits`] runs it (the collapse, the final norm, each rank's vocabulary rows of the
    /// fp32 head), then each row's largest logit and the lowest id holding it on every rank
    /// (`memra_v41_argmax_rows`), the ranks' slices compared on the host in ascending order, so
    /// the lowest id wins among equal maxima. The token of [`argmax`] over [`Self::logits`]
    /// without the logits' download. Refused as a whole, like there, when a rank's route flag is
    /// set (`Ctx::route_bad`) or a row's maximum is not finite: a caller that must not lose the
    /// other rows takes [`Self::logits`] for the call then and chooses row by row.
    pub fn argmax_rows(&mut self, out: &Out, r0: usize, rows: usize) -> Result<Vec<u32>, String> {
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
        let g = self.g.clone();
        let (hc, d) = (g.hc, g.dim);
        let (h, pre) = (&out.h, &out.pre);
        let parts = self.grp.each(|r, c| {
            let st = c.ops.st.clone();
            let x = c.ops.tmp(rows * d * 2)?;
            super::ck("hc_pre", unsafe {
                super::ffi::memra_v41_hc_pre(
                    h[r].at(&st, r0 * hc * d * 2).c(),
                    pre[r].at(&st, r0 * hc * 4).f(),
                    rows as i32,
                    hc as i32,
                    d as i32,
                    x.p(&st).m(),
                    d as i64,
                    c.ops.sv(),
                )
            })?;
            c.ops
                .rmsnorm(x.p(&st), d, &nw[r], x.p(&st), d, rows, d, g.norm_eps)?;
            let n = hw[r].n;
            let lg = c.ops.tmp(rows * n * 4)?;
            c.ops
                .linear_bf16(x.p(&st), d, rows, &hw[r], lg.p(&st), n, true)?;
            let val = c.ops.tmp(rows * 4)?;
            let idx = c.ops.tmp(rows * 4)?;
            super::ck("argmax_rows", unsafe {
                super::ffi::memra_v41_argmax_rows(
                    lg.p(&st).f(),
                    n as i64,
                    rows as i32,
                    n as i32,
                    val.p(&st).fm(),
                    idx.p(&st).im(),
                    c.ops.sv(),
                )
            })?;
            let got = (
                super::bytes_f32(&val.host(&st)?),
                super::bytes_i32(&idx.host(&st)?),
                n,
            );
            // the stream is drained: the route flags of the calls since the last read, read as
            // `Group::head_logits` reads them
            Ok((got, c.route_flags()?))
        })?;
        let (parts, flags): (Vec<_>, Vec<(i32, i32)>) = parts.into_iter().unzip();
        self.grp.route_verdict(&flags)?;
        let mut best = vec![(f32::NEG_INFINITY, 0u32); rows];
        let mut col = 0usize;
        for (val, idx, n) in parts {
            for i in 0..rows {
                if !val[i].is_finite() {
                    return Err("non-finite logits".into());
                }
                if val[i] > best[i].0 {
                    best[i] = (val[i], (col + idx[i] as usize) as u32);
                }
            }
            col += n;
        }
        Ok(best.into_iter().map(|b| b.1).collect())
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
    // experts split by rows (N ranks, experts on the card): a MoE slice of `moe_slice(chunk)` rows
    // holds the full hidden rows of its pairs and of the shared expert, the quantized pair inputs
    // and hidden rows, this rank's gate and up slices and output columns, and the routing
    let moe = if n > 1 {
        let cm = moe_slice(chunk).min(s).max(1);
        let pm = cm * g.topk;
        let (inter, hn, dn) = (g.moe_inter, g.moe_inter.div_ceil(n), d.div_ceil(n));
        let split = 2 * rows
            + (pm + cm) * (inter * 2 + 2 * hn * 2 + dn * 2)
            + pm * (d + d / 32 + inter + inter / 32)
            + cm * (d + d / 32 + inter + inter / 32 + g.n_experts * 4)
            + pm * 4 * 4;
        moe.max(split)
    } else {
        moe
    };
    let mixes = 2 * s * ((2 + hc) * hc + 2 * hc + hc * hc) * 4;
    let engram = c * (d * (hc + 1) * 2 + g.engram_cols() * g.engram_head_dim * 3);
    let peak = stream + attn.max(moe) + mixes + engram;
    peak + peak / 10 + (512 << 20)
}

/// The MoE slice of a forward whose query slices are `chunk` rows: four of them, so that a routed
/// expert's rows fill whole tiles (384 experts, 6 a token: 128 rows an expert at 8,192 rows).
pub fn moe_slice(chunk: usize) -> usize {
    4 * chunk
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
