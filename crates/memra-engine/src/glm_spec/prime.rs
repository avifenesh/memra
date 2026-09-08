//! GLM5 preparation: frozen trunk ranges, one boundary draw, then the original
//! native 512-row warm or host DFlash 256-row ingest. Both arms use this tape.
use super::*;
use crate::hybrid_forward::{Glm5TrunkPrime, PrimeEnvironment};
use crate::prime_walker::{PrimeChunk, PrimeWalker};

pub struct Glm5PrimeState {
    trunk: Option<Glm5TrunkPrime>,
    cache: Option<Cache>,
    session: Option<Glm5SpecSession>,
    committed: Vec<u32>,
    boundary_logits: Vec<f32>,
    restored_dkv: Option<DflashKv>,
    restored: bool,
    taps: Option<Vec<usize>>,
    device_taps: bool,
    sampling: Option<SpecSampling>,
    pen: Option<Glm5Penalty>,
    max_ctx: usize,
    mtp_il: Option<usize>,
    tp_sharded: bool,
    fill: Vec<(usize, usize)>,
    fill_cursor: usize,
    prof: Option<SpecFirstTokenProf>,
    environment: PrimeEnvironment,
    order: PreparationOrder,
}

pub struct Glm5PrimeWalker<'a> {
    model: &'a HybridModel,
    e: &'a Engine,
    state: &'a mut Option<Glm5PrimeState>,
}

fn fill_ranges(rows: usize, width: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(width)
        .map(|s| (s, (s + width).min(rows)))
        .collect()
}

/// The same phase order drives CUDA and the CPU tape oracle. Boundary sampling
/// happens after the trunk, before draft warm, and emission only after ready.
#[derive(Debug)]
struct PreparationOrder {
    trunk: usize,
    draft: usize,
    completed: usize,
    anchor: bool,
}
impl PreparationOrder {
    fn new(trunk: usize, draft: usize) -> Self {
        Self {
            trunk,
            draft,
            completed: 0,
            anchor: false,
        }
    }
    fn remaining(&self) -> usize {
        self.trunk + self.draft - self.completed
    }
    fn complete_chunk(&mut self) -> Res<()> {
        if self.remaining() == 0 || (self.completed >= self.trunk && !self.anchor) {
            return Err("GLM5 preparation chunk violates the anchor boundary".into());
        }
        self.completed += 1;
        Ok(())
    }
    fn draw_anchor(&mut self) -> Res<()> {
        if self.completed != self.trunk || self.anchor {
            return Err("GLM5 anchor must be drawn once at the completed trunk boundary".into());
        }
        self.anchor = true;
        Ok(())
    }
    fn ready(&self) -> bool {
        self.remaining() == 0 && self.anchor
    }
}

impl Glm5PrimeState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cold(
        m: &HybridModel,
        e: &Engine,
        mut cache: Cache,
        prompt: &[u32],
        max_ctx: usize,
        sampling: Option<SpecSampling>,
        pen: Option<Glm5Penalty>,
        mtp_il: Option<usize>,
        source: crate::spec::DraftSourceKind,
        tp_sharded: bool,
        prof: Option<SpecFirstTokenProf>,
    ) -> Res<Self> {
        let n = m.cfg.n_embd as usize;
        let eh = m.glm5_head_engine(e)?;
        let taps = m
            .glm5_dflash
            .as_ref()
            .map(|dr| glm5_dflash_tap_layers(&dr.draft, m.layers.len()))
            .transpose()?;
        let device_taps = taps.is_some() && glm5_draft_taps_device_on();
        if let Some(taps) = &taps {
            let mut sink = if device_taps {
                let ring = crate::hybrid_forward::hyper_prime_call_rows(
                    prompt.len(),
                    m.layers.len(),
                    m.gdn_prime_grid_on(),
                );
                let dr = m.glm5_dflash.as_ref().ok_or("GLM5 draft vanished")?;
                let kv = DflashKv::new(eh, &dr.draft.cfg, max_ctx)?;
                let mut sink = HcTapSink::new_device_staged_at(taps.clone(), n, ring, 0);
                sink.ingest_state = Some(Box::new(Glm5DraftPrimeInflight {
                    kv,
                    taps: taps.clone(),
                    n_embd: n,
                    ring,
                    stage: (0..taps.len()).map(|_| None).collect(),
                    rows_dev: None,
                    prof_on: prof.is_some(),
                    copy_ms: 0.0,
                    feat_ms: 0.0,
                    kv_ms: 0.0,
                    chunks: 0,
                }));
                sink
            } else {
                HcTapSink::new(taps.clone(), n, prompt.len())
            };
            sink.base = 0;
            cache.hc_taps = Some(sink);
        }
        let fill = match source {
            crate::spec::DraftSourceKind::NativeMtp => fill_ranges(prompt.len() - 1, 512),
            crate::spec::DraftSourceKind::Dflash2
                if !device_taps && !glm5_draft_prime_lazy_on() =>
            {
                fill_ranges(prompt.len(), 256)
            }
            crate::spec::DraftSourceKind::Dflash2 => Vec::new(),
        };
        let trunk = Glm5TrunkPrime::new(m, e, prompt, &cache)?;
        // A short cold arrival includes its single bounded warm in the final
        // trunk quantum, so the worker can emit its first token in that turn.
        let draft_quanta = if trunk.remaining() == 1 && fill.len() == 1 {
            0
        } else {
            fill.len()
        };
        let order = PreparationOrder::new(trunk.remaining(), draft_quanta);
        Ok(Self {
            trunk: Some(trunk),
            cache: Some(cache),
            session: None,
            committed: prompt.to_vec(),
            boundary_logits: Vec::new(),
            restored_dkv: None,
            restored: false,
            taps,
            device_taps,
            sampling,
            pen,
            max_ctx,
            mtp_il,
            tp_sharded,
            fill,
            fill_cursor: 0,
            prof,
            environment: PrimeEnvironment::read(),
            order,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restored(
        m: &HybridModel,
        e: &Engine,
        mut cache: Cache,
        fed: &[u32],
        suffix: &[u32],
        boundary_logits: &[f32],
        dkv: DflashKv,
        max_ctx: usize,
        sampling: Option<SpecSampling>,
        pen: Option<Glm5Penalty>,
    ) -> Res<Self> {
        let dr = m
            .glm5_dflash
            .as_ref()
            .ok_or("restored GLM5 draft missing")?;
        let taps = glm5_dflash_tap_layers(&dr.draft, m.layers.len())?;
        let trunk = if suffix.is_empty() {
            None
        } else {
            cache.hc_taps = Some(HcTapSink::new_at(
                taps.clone(),
                m.cfg.n_embd as usize,
                suffix.len(),
                fed.len(),
            ));
            Some(Glm5TrunkPrime::new(m, e, suffix, &cache)?)
        };
        let committed = fed.iter().chain(suffix).copied().collect();
        let order = PreparationOrder::new(trunk.as_ref().map_or(0, |t| t.remaining()), 0);
        Ok(Self {
            trunk,
            cache: Some(cache),
            session: None,
            committed,
            boundary_logits: boundary_logits.to_vec(),
            restored_dkv: Some(dkv),
            restored: true,
            taps: Some(taps),
            device_taps: false,
            sampling,
            pen,
            max_ctx,
            mtp_il: None,
            tp_sharded: false,
            fill: Vec::new(),
            fill_cursor: 0,
            prof: spec_prof_on().then(SpecFirstTokenProf::default),
            environment: PrimeEnvironment::read(),
            order,
        })
    }
    /// Exact boundary row for the resumed-prime diagnostic. Available after
    /// the trunk, before or after the independent draft preparation.
    pub fn boundary_logits(&self) -> &[f32] {
        self.trunk
            .as_ref()
            .map_or(&self.boundary_logits, |t| &t.logits)
    }
    fn remaining(&self) -> usize {
        self.order.remaining()
    }
    fn setup(&mut self, m: &HybridModel, e: &Engine) -> Res<()> {
        if self.session.is_some() {
            return Ok(());
        }
        if self.trunk.as_ref().is_some_and(|t| t.remaining() != 0) {
            return Err("GLM5 anchor cannot precede the last trunk range".into());
        }
        let eh = m.glm5_head_engine(e)?;
        let mut cache = self.cache.take().ok_or("GLM5 prime cache missing")?;
        let logits = self
            .trunk
            .as_ref()
            .map_or(&self.boundary_logits, |t| &t.logits);
        let mut clock = self.prof.as_ref().map(|_| ProfClock::start(e, eh));
        let prefix_capture = if glm5_spec_prefix_on() && self.taps.is_some() {
            self.trunk.as_ref().and_then(|t| {
                m.glm5_prefix_boundary_capture(
                    e,
                    eh,
                    &cache,
                    logits,
                    &t.hiddens,
                    self.committed.len(),
                    t.tokens.len(),
                )
            })
        } else {
            None
        };
        if let (Some(p), Some(c)) = (&mut self.prof, &mut clock) {
            p.capture_ms = c.lap(e, eh);
        }
        let pen_hist = glm5_pen_window_seed(self.pen.as_ref(), &self.committed);
        let mut sctr = 0;
        self.order.draw_anchor()?;
        let anchor = glm5_anchor(
            eh,
            logits,
            self.sampling.as_ref(),
            self.pen.as_ref(),
            &pen_hist,
            &mut sctr,
            if self.restored {
                "glm5-restore"
            } else {
                "glm5-prime"
            },
        )?;
        if let (Some(p), Some(c)) = (&mut self.prof, &mut clock) {
            p.anchor_ms = c.lap(e, eh);
        }
        let draft = if let Some(taps) = &self.taps {
            let dr = m.glm5_dflash.as_ref().ok_or("GLM5 draft missing")?;
            let sink = cache.hc_taps.take();
            let (kv, pending) = if self.device_taps {
                let state = sink
                    .ok_or("device tap sink missing")?
                    .ingest_state
                    .take()
                    .ok_or("device ingest state missing")?
                    .downcast::<Glm5DraftPrimeInflight>()
                    .map_err(|_| "wrong device ingest state")?;
                if state.kv.len != self.committed.len() {
                    return Err("device prime coverage mismatch".into());
                }
                (state.kv, Vec::new())
            } else {
                let pending = match sink {
                    Some(s) => s.rows,
                    None if self.restored && self.trunk.is_none() => Vec::new(),
                    None => return Err("GLM5 host tap sink missing".into()),
                };
                let kv = match self.restored_dkv.take() {
                    Some(k) => k,
                    None => DflashKv::new(eh, &dr.draft.cfg, self.max_ctx)?,
                };
                (kv, pending)
            };
            if let Some(p) = &mut self.prof {
                p.draft_kv_mb = dflash_kv_bytes(&dr.draft.cfg, self.max_ctx) as f64 / 1e6;
            }
            Glm5DraftState::Dflash2 {
                kv,
                pending,
                taps: taps.clone(),
            }
        } else {
            Glm5DraftState::NativeMtp
        };
        let prof = self.prof.take();
        self.session = Some(Glm5SpecSession {
            cache,
            committed: self.committed.clone(),
            anchor,
            anchor_emitted: false,
            pending: Vec::new(),
            draft,
            sampling: self.sampling,
            pen: self.pen.take(),
            pen_hist,
            sctr,
            uctr: 0,
            snap_pool: Vec::new(),
            verify_graphs: VerifyGraphPool::default(),
            rounds: 0,
            rank_trimmed_rounds: 0,
            done: false,
            max_ctx: self.max_ctx,
            mtp_il: self.mtp_il,
            prefix_capture,
            prof_rounds: prof.as_ref().map(|_| SpecRoundsLog::default()),
            prof,
        });
        Ok(())
    }
    fn fill_one(&mut self, m: &HybridModel, e: &Engine) -> Res<usize> {
        let &(start, end) = self
            .fill
            .get(self.fill_cursor)
            .ok_or("GLM5 draft fill exhausted")?;
        let session = self
            .session
            .as_mut()
            .ok_or("GLM5 fill before anchor setup")?;
        let started = std::time::Instant::now();
        let eh = m.glm5_head_engine(e)?;
        let n = m.cfg.n_embd as usize;
        match &mut session.draft {
            Glm5DraftState::NativeMtp => {
                let trunk = self
                    .trunk
                    .as_ref()
                    .ok_or("native warm needs hidden stack")?;
                m.glm5_mtp_plane_fill_chunk(
                    eh,
                    &self.committed[1..],
                    &trunk.hiddens,
                    self.committed.len() - 1,
                    start,
                    end - start,
                    &mut session.cache,
                )?;
                if self.fill_cursor + 1 == self.fill.len() {
                    session.pending.push((
                        session.anchor,
                        m.glm5_seed_row(
                            eh,
                            &trunk.hiddens,
                            self.committed.len(),
                            self.committed.len() - 1,
                        )?,
                    ));
                }
            }
            Glm5DraftState::Dflash2 { kv, pending, taps } => {
                let dr = m.glm5_dflash.as_ref().ok_or("GLM5 draft missing")?;
                let width = taps.len() * n;
                m.glm5_dflash_ingest_rows(
                    eh,
                    &dr.draft,
                    kv,
                    &pending[start * width..end * width],
                    width,
                    None,
                )?;
                if self.fill_cursor + 1 == self.fill.len() {
                    pending.clear();
                }
            }
        }
        m.glm5_prime_fence(e)?;
        if let Some(p) = &mut session.prof {
            p.draft_prime_ms += started.elapsed().as_secs_f64() * 1000.0;
            p.draft_prime_rows += end - start;
            p.draft_prime_chunks += 1;
            p.draft_prime_arm = if self.taps.is_some() {
                "eager"
            } else {
                "native"
            };
        }
        self.fill_cursor += 1;
        Ok(end - start)
    }
}

impl HybridModel {
    /// Pair/PP-1 diagnostic: independent old hyper loop versus a saved prime,
    /// with a plain peer exercising shared transfer slots and decode graphs.
    pub fn check_glm5_prime_walker(&self, e: &Engine, prompt: &[u32]) -> Res<()> {
        let cap = prompt.len() + 128;
        let mut oracle = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, cap)?;
        let (logits, seed, hidden) = self.prime_cache(e, prompt, &mut oracle, 0)?;
        self.glm5_prime_fence(e)?;
        let expected_hidden = e.dtoh(&hidden)?;
        drop(seed);
        drop(hidden);
        drop(oracle);
        let mut pending = Some(self.glm5_prime_start(e, prompt, cap, None)?);
        let small = &prompt[..prompt.len().min(17)];
        let mut yields = 0;
        loop {
            let mut walker = self.glm5_prime_walker(e, &mut pending);
            let progress = crate::prime_walker::advance_prime(
                &mut walker,
                true,
                crate::prime_walker::trace_chunk,
            )?;
            if progress.remaining_chunks == 0 {
                break;
            }
            // Another prime uses both PP slots, so a saved request that retained a
            // shared slot instead of its owned receive copy fails on resumption.
            let mut peer = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, 128)?;
            let (row, seed, hidden) = self.prime_cache(e, small, &mut peer, 0)?;
            drop(seed);
            drop(hidden);
            let mut token = argmax(&row) as u32;
            for _ in 0..4 {
                token = argmax(&self.decode_step(e, token, &mut peer)?) as u32;
            }
            self.glm5_prime_fence(e)?;
            yields += 1;
        }
        let state = pending.as_ref().ok_or("GLM5 diagnostic state missing")?;
        let trunk = state
            .trunk
            .as_ref()
            .ok_or("GLM5 diagnostic trunk missing")?;
        let bits_equal = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if yields == 0
            || !bits_equal(&trunk.logits, &logits)
            || !bits_equal(&e.dtoh(&trunk.hiddens)?, &expected_hidden)
        {
            return Err("GLM5 saved prime boundary/hidden mismatch or no yield".into());
        }
        let walker = self.glm5_prime_walker(e, &mut pending);
        let session = crate::prime_walker::finish_prime(walker)?;
        drop(session);
        self.glm5_prime_fence(e)?;
        eprintln!(
            "[glm5-prime-walker-check] PASS rows={} yields={yields} boundary_logits=identical hidden_stack=identical peer=prime-plus-graphed-decode",
            prompt.len()
        );
        Ok(())
    }

    pub fn glm5_prime_walker<'a>(
        &'a self,
        e: &'a Engine,
        state: &'a mut Option<Glm5PrimeState>,
    ) -> Glm5PrimeWalker<'a> {
        Glm5PrimeWalker {
            model: self,
            e,
            state,
        }
    }
}
impl PrimeWalker for Glm5PrimeWalker<'_> {
    type Output = Glm5SpecSession;
    fn remaining_chunks(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.remaining())
    }
    fn advance_chunk(&mut self) -> Res<PrimeChunk> {
        let s = self.state.as_mut().ok_or("GLM5 prime already finished")?;
        s.environment.check()?;
        if let Some(t) = s.trunk.as_mut()
            && t.remaining() > 0
        {
            let before = std::time::Instant::now();
            let rows = t.advance(
                self.model,
                self.e,
                s.cache.as_mut().ok_or("GLM5 cache missing")?,
            )?;
            if let Some(p) = &mut s.prof {
                p.prime_ms += before.elapsed().as_secs_f64() * 1000.0;
            }
            s.order.complete_chunk()?;
            if t.remaining() == 0 {
                s.setup(self.model, self.e)?;
                if s.order.draft == 0 && s.fill.len() == 1 {
                    s.fill_one(self.model, self.e)?;
                }
                self.model.glm5_prime_fence(self.e)?;
            }
            return Ok(PrimeChunk {
                phase: "glm5-trunk",
                rows,
            });
        }
        let rows = s.fill_one(self.model, self.e)?;
        s.order.complete_chunk()?;
        Ok(PrimeChunk {
            phase: "glm5-draft",
            rows,
        })
    }
    fn finish(self) -> Res<Self::Output> {
        let s = self.state.as_mut().ok_or("GLM5 prime already finished")?;
        s.environment.check()?;
        if s.remaining() != 0 {
            return Err("cannot finish incomplete GLM5 prime".into());
        }
        s.setup(self.model, self.e)?; // zero-trunk full cover: one boundary draw, no loop
        if !s.order.ready() {
            return Err("GLM5 prime not ready to emit".into());
        }
        self.model.glm5_prime_fence(self.e)?;
        let mut s = self.state.take().ok_or("GLM5 prime already finished")?;
        let session = s.session.take().ok_or("GLM5 prepared session missing")?;
        if s.restored {
            let suffix = s.trunk.as_ref().map_or(0, |t| t.tokens.len());
            eprintln!(
                "[glm5-spec] RESTORED session: {} prefix tokens + {} suffix from cache; arm {}",
                s.committed.len() - suffix,
                suffix,
                if suffix == 0 {
                    "full-cover"
                } else {
                    "suffix-prime"
                }
            );
        }
        if s.tp_sharded {
            eprintln!(
                "[glm5-spec] spec x TP composition ARMED (MEMRA_GLM5_SPEC_TP=1): verify rows ride the TP shards; rollback restores per-rank planes performance_claim=false"
            );
        }
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_and_first_token_keep_their_tape_position_off_and_on() {
        struct Fake {
            order: PreparationOrder,
            tape: Vec<&'static str>,
        }
        impl PrimeWalker for Fake {
            type Output = Vec<&'static str>;
            fn remaining_chunks(&self) -> usize {
                self.order.remaining()
            }
            fn advance_chunk(&mut self) -> Res<PrimeChunk> {
                let phase = if self.order.completed < self.order.trunk {
                    "trunk"
                } else {
                    "draft"
                };
                self.tape.push(phase);
                self.order.complete_chunk()?;
                if self.order.completed == self.order.trunk {
                    self.order.draw_anchor()?;
                    self.tape.push("anchor");
                }
                Ok(PrimeChunk { phase, rows: 1 })
            }
            fn finish(mut self) -> Res<Self::Output> {
                if self.order.trunk == 0 && !self.order.anchor {
                    self.order.draw_anchor()?;
                    self.tape.push("anchor");
                }
                if !self.order.ready() {
                    return Err("not ready".into());
                }
                self.tape.push("first-token");
                Ok(self.tape)
            }
        }
        for (trunk, draft) in [(3, 2), (1, 0), (2, 0), (0, 0)] {
            let mut tapes = Vec::new();
            for yielded in [false, true] {
                let mut f = Fake {
                    order: PreparationOrder::new(trunk, draft),
                    tape: Vec::new(),
                };
                while f.remaining_chunks() > 0 {
                    crate::prime_walker::advance_prime(&mut f, yielded, |_, _| {}).unwrap();
                    assert!(!f.tape.contains(&"first-token"));
                }
                tapes.push(crate::prime_walker::finish_prime(f).unwrap());
            }
            assert_eq!(tapes[0], tapes[1]);
            assert_eq!(tapes[0][trunk], "anchor");
            assert_eq!(tapes[0].last(), Some(&"first-token"));
        }
        let mut order = PreparationOrder::new(2, 1);
        assert!(order.draw_anchor().is_err());
        order.complete_chunk().unwrap();
        order.complete_chunk().unwrap();
        assert!(order.complete_chunk().is_err());
        order.draw_anchor().unwrap();
        assert!(order.draw_anchor().is_err());
    }
    #[test]
    fn draft_phase_ranges_keep_the_original_tail_and_offsets() {
        assert_eq!(
            fill_ranges(1025, 512),
            [(0, 512), (512, 1024), (1024, 1025)]
        );
        assert_eq!(fill_ranges(257, 256), [(0, 256), (256, 257)]);
        assert!(fill_ranges(0, 256).is_empty());
    }
}

/// Plain carried hyper segments retain the prefill tick's original range and
/// queued_after. Ownership is identical to the speculative trunk adapter.
pub struct Glm5PlainPrimeState {
    cache: Cache,
    trunk: Glm5TrunkPrime,
    environment: PrimeEnvironment,
}
impl Glm5PlainPrimeState {
    pub fn tokens_len(&self) -> usize {
        self.trunk.tokens.len()
    }
}
pub struct Glm5PlainPrimeWalker<'a> {
    model: &'a HybridModel,
    e: &'a Engine,
    state: &'a mut Option<Glm5PlainPrimeState>,
}
impl HybridModel {
    pub fn glm5_plain_prime_start(
        &self,
        e: &Engine,
        cache: Cache,
        tokens: &[u32],
        queued_after: usize,
    ) -> Res<Glm5PlainPrimeState> {
        let trunk = Glm5TrunkPrime::new(self, e, tokens, &cache)?.with_queued_after(queued_after);
        Ok(Glm5PlainPrimeState {
            cache,
            trunk,
            environment: PrimeEnvironment::read(),
        })
    }
    pub fn glm5_plain_prime_walker<'a>(
        &'a self,
        e: &'a Engine,
        state: &'a mut Option<Glm5PlainPrimeState>,
    ) -> Glm5PlainPrimeWalker<'a> {
        Glm5PlainPrimeWalker {
            model: self,
            e,
            state,
        }
    }
}
impl PrimeWalker for Glm5PlainPrimeWalker<'_> {
    type Output = (Cache, Vec<f32>, CudaSlice<f32>);
    fn remaining_chunks(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.trunk.remaining())
    }
    fn advance_chunk(&mut self) -> Res<PrimeChunk> {
        let s = self
            .state
            .as_mut()
            .ok_or("plain hyper prime already finished")?;
        s.environment.check()?;
        let rows = s.trunk.advance(self.model, self.e, &mut s.cache)?;
        Ok(PrimeChunk {
            phase: "glm5-plain-trunk",
            rows,
        })
    }
    fn finish(self) -> Res<Self::Output> {
        let s = self
            .state
            .as_ref()
            .ok_or("plain hyper prime already finished")?;
        s.environment.check()?;
        if s.trunk.remaining() != 0 {
            return Err("plain hyper prime is incomplete".into());
        }
        self.model.glm5_prime_fence(self.e)?;
        let s = self
            .state
            .take()
            .ok_or("plain hyper prime already finished")?;
        Ok((s.cache, s.trunk.logits, s.trunk.hiddens))
    }
}
