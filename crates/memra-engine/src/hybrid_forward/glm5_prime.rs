//! Saved hyper prime ranges. A PP pipeline keeps its one-range lookahead in an
//! owned receive buffer, never in a runtime slot that another request can reuse.
use super::*;
use crate::prime_walker::PrimeError;

type ChunkRows = (Vec<f32>, CudaSlice<f32>, CudaSlice<f32>);

/// Process numeric configuration is immutable for a saved tape. Refuse an
/// in-process change rather than silently picking a different kernel on resume.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PrimeEnvironment(Vec<(std::ffi::OsString, std::ffi::OsString)>);
impl PrimeEnvironment {
    pub(crate) fn read() -> Self {
        let mut values: Vec<_> = std::env::vars_os()
            .filter(|(k, _)| {
                k.to_str()
                    .is_some_and(|k| k.starts_with("MEMRA_") || k == "NVIDIA_TF32_OVERRIDE")
            })
            .collect();
        values.sort();
        Self(values)
    }
    pub(crate) fn check(&self) -> Result<(), PrimeError> {
        if *self != Self::read() {
            return Err(
                "GLM5 prime numeric environment changed while its tape was suspended".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct PrimeRendezvous {
    completed: Vec<usize>,
    committed: usize,
}
impl PrimeRendezvous {
    pub(crate) fn new(ranks: usize) -> Self {
        Self {
            completed: vec![0; ranks],
            committed: 0,
        }
    }
    pub(crate) fn acknowledge(&mut self, rank: usize, chunk: usize) -> Result<(), PrimeError> {
        let value = self.completed.get_mut(rank).ok_or("unknown prime rank")?;
        if chunk != self.committed + 1 || *value != self.committed {
            return Err("GLM5 prime rank acknowledged a stale or repeated chunk".into());
        }
        *value = chunk;
        Ok(())
    }
    pub(crate) fn commit(&mut self, chunk: usize) -> Result<(), PrimeError> {
        if chunk != self.committed + 1 || self.completed.iter().any(|&c| c != chunk) {
            return Err(
                "GLM5 prime cannot yield before every rank completes the same chunk".into(),
            );
        }
        self.committed = chunk;
        Ok(())
    }
}

pub(crate) struct Glm5TrunkPrime {
    pub(crate) tokens: Vec<u32>,
    pub(crate) hiddens: CudaSlice<f32>,
    pub(crate) logits: Vec<f32>,
    ranges: Vec<(usize, usize)>,
    cursor: usize,
    base: usize,
    seq_end: usize,
    queued_after: usize,
    fence: Option<Vec<usize>>,
    pipeline: bool,
    stage0_pos: usize,
    ahead: Option<CudaSlice<f32>>,
    rendezvous: PrimeRendezvous,
}

impl Glm5TrunkPrime {
    pub(crate) fn new(
        m: &HybridModel,
        e: &Engine,
        tokens: &[u32],
        cache: &Cache,
    ) -> Result<Self, PrimeError> {
        cache.ensure_usable("glm5_prime_start")?;
        if tokens.is_empty() || cache.pos + tokens.len() > cache.max_ctx {
            return Err("GLM5 trunk prime requires nonempty tokens within capacity".into());
        }
        let ranges = hyper_prime_ranges(tokens.len(), m.layers.len(), m.gdn_prime_grid_on());
        let fence = crate::pp::pp_cuts(m.layers.len());
        let pipeline = ranges.len() > 1
            && b200_prime_v2_on()
            && !crate::pp::pp2_streams_off()
            && crate::pp::prime_pipe_on()
            && fence.as_ref().is_some_and(|f| f.len() == 3);
        // The existing device-ring/pipeline combination has no valid range ingest.
        if pipeline
            && cache
                .hc_taps
                .as_ref()
                .is_some_and(|s| s.ingest_state.is_some())
        {
            return Err("GLM5 pipelined prime with device tap ring is not qualified".into());
        }
        let ranks = fence
            .as_ref()
            .map_or_else(|| m.glm5_prime_rank_count(), |f| f.len() - 1);
        Ok(Self {
            tokens: tokens.to_vec(),
            hiddens: e.uninit(tokens.len() * m.cfg.n_embd as usize)?,
            logits: Vec::new(),
            ranges,
            cursor: 0,
            base: cache.pos,
            seq_end: cache.pos + tokens.len(),
            queued_after: 0,
            fence,
            pipeline,
            stage0_pos: cache.pos,
            ahead: None,
            rendezvous: PrimeRendezvous::new(ranks),
        })
    }
    pub(crate) fn with_queued_after(mut self, queued_after: usize) -> Self {
        self.seq_end += queued_after;
        self.queued_after = queued_after;
        self
    }
    pub(crate) fn remaining(&self) -> usize {
        self.ranges.len() - self.cursor
    }
    pub(crate) fn advance(
        &mut self,
        m: &HybridModel,
        e: &Engine,
        cache: &mut Cache,
    ) -> Result<usize, PrimeError> {
        let &(start, end) = self.ranges.get(self.cursor).ok_or("GLM5 trunk exhausted")?;
        if cache.pos != self.base + start {
            return Err("GLM5 prime cursor does not match its frozen range".into());
        }
        let topology = m.hyper.as_ref().ok_or("GLM5 prime has no hyper topology")?;
        let out = if self.pipeline {
            self.pipeline_chunk(m, e, cache, start, end)?
        } else {
            m.glm5_taps_range_begin(cache, start);
            let out = match &self.fence {
                Some(fence) => m.prime_cache_hyper_ppn(
                    e,
                    &self.tokens[start..end],
                    cache,
                    self.queued_after + self.tokens.len() - end,
                    topology,
                    fence,
                    None,
                )?,
                None => m.prime_chunk_hyper(
                    e,
                    &self.tokens[start..end],
                    cache,
                    self.seq_end,
                    start,
                    None,
                )?,
            };
            m.glm5_taps_range_done(e, cache, start, end)?;
            out
        };
        let n = m.cfg.n_embd as usize;
        e.copy_into(&mut self.hiddens, start * n, &out.2, (end - start) * n)?;
        self.logits = out.0;
        drop(out.1);
        drop(out.2);
        // All operations on all ranks have been issued BEFORE any host wait.
        m.glm5_prime_fence(e)?;
        for (il, layer) in m.layers.iter().enumerate() {
            if !matches!(layer.mixer, Mixer::Mla(_)) {
                continue;
            }
            let expected = if self.pipeline && self.fence.as_ref().is_some_and(|f| il < f[1]) {
                self.stage0_pos
            } else {
                cache.pos
            };
            if cache.latent[il].as_ref().is_some_and(|p| p.len != expected)
                || cache.glm5_tp_latent_peer[il]
                    .as_ref()
                    .is_some_and(|ps| ps.iter().any(|p| p.len != expected))
            {
                return Err(format!(
                    "GLM5 layer {il} rank frontier differs from expected {expected}"
                )
                .into());
            }
        }
        for rank in 0..self.rendezvous.completed.len() {
            self.rendezvous.acknowledge(rank, self.cursor + 1)?;
        }
        self.rendezvous.commit(self.cursor + 1)?;
        self.cursor += 1;
        if std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1") {
            eprintln!(
                "[glm5-prime-rendezvous] chunk={} ranks={:?} pos={} stage0_pos={} lookahead={}",
                self.cursor,
                self.rendezvous.completed,
                cache.pos,
                self.stage0_pos,
                self.ahead.is_some()
            );
        }
        crate::progress::note_prime_rows(end - start);
        Ok(end - start)
    }
    fn pipeline_chunk(
        &mut self,
        m: &HybridModel,
        e: &Engine,
        cache: &mut Cache,
        start: usize,
        end: usize,
    ) -> Result<ChunkRows, PrimeError> {
        let fence = self.fence.as_ref().ok_or("PP prime missing cuts")?;
        let topology = m.hyper.as_ref().ok_or("PP prime missing topology")?;
        let rt = crate::pp::PpNRt::get(e)?;
        let caller = e.stream();
        let n = m.cfg.n_embd as usize;
        let width = topology.streams * n;
        rt.fence_stages_behind(&caller)?;
        if self.cursor == 0 {
            let max = self
                .ranges
                .iter()
                .map(|(s, x)| (x - s) * width)
                .max()
                .unwrap();
            rt.prepare_overlap_slots(0, max)?;
        }
        let mut sink = cache.hc_taps.take();
        let result = (|| {
            let lock = sink.as_mut().map(std::sync::Mutex::new);
            let arm = |base| match &lock {
                Some(s) => HcTapArm::Shared(s, base),
                None => HcTapArm::FromCache,
            };
            let mut stages = PrimeCacheStages::new(cache, fence);
            let (c0, c1) = stages.pp2_parts();
            c0.pos = self.stage0_pos;
            let x = match self.ahead.take() {
                Some(x) => x,
                None => {
                    let slot = m.prime_hyper_pp2_stage0_enqueue(
                        e,
                        rt,
                        topology,
                        &self.tokens[start..end],
                        c0,
                        self.seq_end,
                        fence,
                        self.base + start,
                        arm(self.base + start),
                    )?;
                    c0.pos = self.base + end;
                    rt.rx(0, slot, (end - start) * width)?
                }
            };
            let next = self.ranges.get(self.cursor + 1).copied();
            let (out, next_slot) = std::thread::scope(|scope| -> Result<_, PrimeError> {
                let producer = next.map(|(ns, ne)| {
                    let tokens = &self.tokens[ns..ne];
                    let base = self.base;
                    let seq_end = self.seq_end;
                    let arm = &arm;
                    scope.spawn(move || -> Result<usize, String> {
                        let slot = m
                            .prime_hyper_pp2_stage0_enqueue(
                                e,
                                rt,
                                topology,
                                tokens,
                                c0,
                                seq_end,
                                fence,
                                base + ns,
                                arm(base + ns),
                            )
                            .map_err(|e| e.to_string())?;
                        c0.pos = base + ne;
                        Ok(slot)
                    })
                });
                let out = (|| {
                    rt.bind_stage(1)?;
                    let _stage = rt.enter(1);
                    let el = rt.engine(1, e);
                    let pos: Vec<i32> = (self.base + start..self.base + end)
                        .map(|p| p as i32)
                        .collect();
                    let pos_d = el.htod_i32(&pos)?;
                    let _overlap = crate::pp::enter_prime_pipe_stage();
                    let x = m.hyper_range_prime(
                        el,
                        topology,
                        x,
                        fence[1],
                        fence[2],
                        &pos_d,
                        end - start,
                        c1,
                        self.seq_end,
                        arm(self.base + start),
                    )?;
                    m.hyper_prime_tail(el, topology, &x, end - start, n, m.cfg.rms_eps, c1)
                })();
                let slot = match producer {
                    Some(t) => Some(
                        t.join()
                            .map_err(|_| "GLM5 PP producer panicked")?
                            .map_err(|s| -> PrimeError { s.into() })?,
                    ),
                    None => None,
                };
                Ok((out?, slot))
            })?;
            // rx returns an owned copy and records slot release. Retain that copy
            // across peer work; neither shared slot is leased by a saved request.
            if let (Some(slot), Some((ns, ne))) = (next_slot, next) {
                self.ahead = Some(rt.rx(0, slot, (ne - ns) * width)?);
                self.stage0_pos = self.base + ne;
            } else {
                self.stage0_pos = self.base + end;
            }
            stages.commit();
            drop(stages);
            rt.publish_all_to(&caller)?;
            Ok(out)
        })();
        if let Some(ref mut s) = sink {
            s.base = self.base + start;
        }
        cache.hc_taps = sink;
        result
    }
}

impl HybridModel {
    fn glm5_prime_rank_count(&self) -> usize {
        self.layers
            .iter()
            .find_map(|l| match &l.mixer {
                Mixer::Kda(k) => k.tp.as_ref().map(|t| t.rt.ranks()),
                Mixer::Mla(k) => k.tp.as_ref().map(|t| t.rt.ranks()),
                _ => None,
            })
            .unwrap_or(1)
    }
    pub(crate) fn glm5_prime_fence(&self, e: &Engine) -> Result<(), PrimeError> {
        if crate::pp::pp_cuts(self.layers.len()).is_some() && !crate::pp::pp2_streams_off() {
            let rt = crate::pp::PpNRt::get(e)?;
            rt.publish_all_to(&e.stream())?;
            for stage in 0..rt.n_stages() {
                rt.bind_stage(stage)?;
                {
                    let _scope = rt.enter(stage);
                    rt.engine(stage, e).stream().synchronize()?;
                }
                rt.engine(stage, e).stream().synchronize()?;
            }
        }
        // Each loaded group can own different streams. Do not infer group ownership
        // from decode-only symmetric glue or from mutable environment placement.
        let mut seen = std::collections::HashSet::new();
        for l in &self.layers {
            let rt = match &l.mixer {
                Mixer::Kda(k) => k.tp.as_ref().map(|t| &t.rt),
                Mixer::Mla(k) => k.tp.as_ref().map(|t| &t.rt),
                _ => None,
            };
            if let Some(rt) = rt
                && seen.insert(std::sync::Arc::as_ptr(rt) as usize)
            {
                rt.prime_completion_fence(e)?;
            }
        }
        e.ctx().bind_to_thread()?;
        e.bind_runtime_device(e.ctx().ordinal() as i32)?;
        e.stream().synchronize()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_ranks_acknowledge_before_peer_service() {
        let mut r = PrimeRendezvous::new(2);
        r.acknowledge(0, 1).unwrap();
        assert!(r.commit(1).is_err());
        assert!(r.acknowledge(1, 2).is_err());
        r.acknowledge(1, 1).unwrap();
        r.commit(1).unwrap();
        assert!(r.acknowledge(0, 1).is_err());
        r.acknowledge(1, 2).unwrap();
        assert!(r.commit(2).is_err());
        r.acknowledge(0, 2).unwrap();
        r.commit(2).unwrap();
    }
    #[test]
    fn numeric_environment_snapshot_is_immutable() {
        let saved = PrimeEnvironment::read();
        saved.check().unwrap();
        let mut changed = PrimeEnvironment::read();
        changed
            .0
            .push(("MEMRA_PRIME_CHUNK".into(), "different".into()));
        assert_ne!(saved, changed);
        assert!(changed.check().is_err());
    }
}
