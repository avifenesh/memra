//! MTP serving adapter: preserve trunk ranges/captures, then fill the draft KV from
//! the retained predecessor hidden rows. No generated boundary is drawn per chunk.

use super::*;
use crate::prime_walker::{PrimeChunk, PrimeError, PrimeWalker};

pub(super) struct PreparedMtp {
    pub(super) prompt: Vec<u32>,
    pub(super) base: usize,
    pub(super) wall: std::time::Duration,
    pub(super) logits: Vec<f32>,
    pub(super) hiddens: Option<CudaSlice<f32>>,
    pub(super) boundary: u32,
    pub(super) init: Option<(Vec<f32>, CudaSlice<f32>)>,
    pub(super) sampling: SpecSampling,
    pub(super) k: usize,
    pub(super) constrained: bool,
    pub(super) draft_setup: Option<SpecDraftSetup>,
    pub(super) seed_buf: Option<CudaSlice<f32>>,
    pub(super) fill_prev: Option<CudaSlice<f32>>,
    pub(super) preds: Option<CudaSlice<u32>>,
}

#[derive(Clone, Copy)]
struct TrunkChunk {
    start: usize,
    end: usize,
    segment_start: usize,
    segment_end: usize,
    batched: bool,
    target_step: bool,
}

/// All mutable request state is owned here or in its SpecSession. The temporary
/// walker binds that state to an engine/model only for the duration of one advance.
pub struct MtpPrimeState {
    prompt: Vec<u32>,
    base: usize,
    chunks: Vec<TrunkChunk>,
    cursor: usize,
    fill_cursor: usize,
    fill_chunk: usize,
    short: bool,
    hiddens: Option<CudaSlice<f32>>,
    logits: Vec<f32>,
    capture_at: Option<usize>,
    ckpt_rel: Option<usize>,
    k: usize,
    sampling: SpecSampling,
    graph_draft: bool,
    prepared: Option<PreparedMtp>,
    wall: std::time::Duration,
    call_local: bool,
}

pub struct MtpPrimeWalker<'a> {
    model: &'a HybridModel,
    e: &'a Engine,
    sess: &'a mut SpecSession,
    state: &'a mut Option<MtpPrimeState>,
    constraint: Option<&'a mut dyn SpecConstraint>,
}

fn trunk_schedule(
    tp: usize,
    prime_split: Option<usize>,
    ckpt_rel: Option<usize>,
    tokenwise: bool,
    mut ranges: impl FnMut(usize) -> Vec<(usize, usize)>,
) -> Vec<TrunkChunk> {
    let mut stops: Vec<_> = [prime_split, ckpt_rel].into_iter().flatten().collect();
    stops.sort_unstable();
    stops.dedup();
    let segmented = !stops.is_empty();
    let mut chunks = Vec::new();
    let mut lo = 0;
    for hi in stops.into_iter().chain(std::iter::once(tp)) {
        let batched =
            hi - lo >= crate::hybrid_forward::PRIME_MIN_T && (segmented && hi < tp || !tokenwise);
        if batched {
            for (start, end) in ranges(hi - lo) {
                chunks.push(TrunkChunk {
                    start: lo + start,
                    end: lo + end,
                    segment_start: lo,
                    segment_end: hi,
                    batched,
                    target_step: false,
                });
            }
        } else {
            // Sub-floor segments retain their original tokenwise numerical calls.
            chunks.push(TrunkChunk {
                start: lo,
                end: hi,
                segment_start: lo,
                segment_end: hi,
                batched,
                target_step: !segmented,
            });
        }
        lo = hi;
    }
    chunks
}

impl HybridModel {
    /// Diagnostic used by run-spec: compare the saved walker with the ordinary
    /// unyielded trunk, including a real stable-boundary snapshot. A separate cache
    /// advances between chunks to exercise reuse of shared model scratch.
    pub fn check_mtp_prime_walker(
        &self,
        e: &Engine,
        prompt: &[u32],
        k: usize,
    ) -> Result<(), PrimeError> {
        let grid = Engine::gdn_chunk_size();
        let boundary = (prompt.len() / 2 / grid) * grid;
        if !self.mtp_prime_walk_supported()
            || boundary < crate::hybrid_forward::PRIME_MIN_T
            || prompt.len() - boundary < crate::hybrid_forward::PRIME_MIN_T
        {
            return Err(
                "prime walker check needs a supported GDN MTP prompt with two prime segments"
                    .into(),
            );
        }
        let cap = prompt.len() + k + 128;
        let mut oracle = Cache::new_planned(e, &self.cfg, &self.plan, cap)?;
        let (cut_logits, cut_seed, cut_h) =
            self.prime_cache(e, &prompt[..boundary], &mut oracle, prompt.len() - boundary)?;
        let cut = SpecBoundaryCapture {
            snap: oracle.snapshot(e)?,
            pos: boundary,
            logits: cut_logits,
            last_h: capture_boundary_hidden(e, &cut_h, boundary, self.cfg.n_embd as usize),
            latent_tails: Vec::new(),
        };
        drop(cut_seed);
        drop(cut_h);
        let (expected, seed, h) = self.prime_cache(e, &prompt[boundary..], &mut oracle, 0)?;
        let expected_capture = crate::prime_receipt::capture_digest(e, &[cut])?;
        drop(seed);
        drop(h);
        drop(oracle);

        let mut session = self.new_session(e, cap)?;
        session.capture_at = Some(boundary);
        session.ckpt_at = Some(boundary);
        let mut state =
            Some(self.mtp_prime_start(e, &mut session, prompt, k, None, Some(boundary))?);
        let mut peer = Cache::new_planned(e, &self.cfg, &self.plan, 1024)?;
        let (peer_logits, seed, h) = self.prime_cache(e, &prompt[..grid], &mut peer, 0)?;
        let mut peer_token = argmax(&peer_logits) as u32;
        drop(seed);
        drop(h);
        let mut walker = self.mtp_prime_walker(e, &mut session, &mut state, None);
        let mut yields = 0;
        while walker.remaining_chunks() > 0 {
            let progress = crate::prime_walker::advance_prime(
                &mut walker,
                true,
                crate::prime_walker::trace_chunk,
            )?;
            if progress.remaining_chunks > 0 {
                if peer.pos + 1 >= peer.max_ctx {
                    peer = Cache::new_planned(e, &self.cfg, &self.plan, 1024)?;
                }
                let (logits, _) = self.spec_target_step_h(e, peer_token, &mut peer)?;
                peer_token = argmax(&logits) as u32;
                yields += 1;
            }
        }
        crate::prime_walker::finish_prime(walker)?;
        let ready = session
            .prime_ready
            .as_ref()
            .ok_or("prime walker check did not finalize")?;
        if ready.logits.len() != expected.len()
            || ready
                .logits
                .iter()
                .zip(&expected)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("resumed MTP prime logits differ from unyielded trunk".into());
        }
        let actual_capture = crate::prime_receipt::capture_digest(e, &session.boundary_captures)?;
        if actual_capture != expected_capture || yields == 0 {
            return Err(format!("MTP capture/yield check failed: yields={yields} expected={expected_capture} actual={actual_capture}").into());
        }
        eprintln!(
            "[prime-walker-check] PASS rows={} yields={yields} target_logits={} boundary_state={actual_capture}",
            prompt.len(),
            crate::prime_receipt::logits_digest(&expected)
        );
        Ok(())
    }

    /// This adapter handles the single-device GDN MTP execution plan. Other
    /// placements keep their existing prime until their own adapter is qualified.
    pub fn mtp_prime_walk_supported(&self) -> bool {
        crate::plan_backend::gdn_dspark_compatible(&self.plan)
            && self.mtp.is_some()
            && self.hyper.is_none()
            && crate::pp::pp_cuts(self.layers.len()).is_none()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mtp_prime_start(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        prompt: &[u32],
        k: usize,
        sampling: Option<SpecSampling>,
        prime_split: Option<usize>,
    ) -> Result<MtpPrimeState, PrimeError> {
        if prompt.is_empty() || !self.mtp_prime_walk_supported() || k == 0 {
            return Err("MTP prime walker requires a nonempty supported prime and K>0".into());
        }
        if sess.prime_ready.is_some() {
            return Err("MTP prime already prepared".into());
        }
        if sess.scratch.plane_count() != self.mtp_head_count() {
            return Err("MTP scratch/head count mismatch".into());
        }
        if sess.pending_tok.is_some() {
            self.spec_flush_pending(e, sess, sampling)?;
        }
        sess.cache.ensure_usable("mtp_prime_start")?;
        let base = sess.cache.pos;
        if prompt.len() > sess.cache.max_ctx.saturating_sub(base) {
            return Err("MTP prime exceeds cache capacity".into());
        }
        let tp = prompt.len();
        let prime_split = prime_split.filter(|&b| b > 0 && b < tp);
        let ckpt_rel = sess
            .ckpt_at
            .take()
            .and_then(|b| b.checked_sub(base))
            .filter(|&b| b > 0 && b < tp);
        let first = prime_split.into_iter().chain(ckpt_rel).min();
        if first == prime_split && first.is_some_and(|b| b < crate::hybrid_forward::PRIME_MIN_T) {
            return Err("MTP prime split below PRIME_MIN_T".into());
        }
        let tokenwise = std::env::var("MEMRA_PRIME_TOKENWISE").is_ok()
            || e.frozen_cpu_experts_prefer_tokenwise_prime();
        let chunks = trunk_schedule(tp, prime_split, ckpt_rel, tokenwise, |len| {
            crate::hybrid_forward::prime_chunk_ranges(
                len,
                self.layers.len(),
                self.gdn_prime_grid_on(),
            )
        });
        let fill_chunk = if crate::cache::swa_ring_on() {
            crate::hybrid_forward::prime_chunk_tokens(tp, self.layers.len())
        } else {
            std::env::var("MEMRA_PRIME_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096)
        };
        let fill_chunk = if fill_chunk == 0 { tp } else { fill_chunk }
            .min(crate::hybrid_forward::CUDA_GRID_YZ_MAX);
        let short = tp <= fill_chunk;
        let graph_draft = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && self.mtp_graph_capturable()
            && k + 2 < 96
            && !crate::model::full_prec_enabled();
        Ok(MtpPrimeState {
            prompt: prompt.to_vec(),
            base,
            chunks,
            cursor: 0,
            fill_cursor: 0,
            fill_chunk,
            short,
            hiddens: Some(e.uninit(tp * self.cfg.n_embd as usize)?),
            logits: Vec::new(),
            capture_at: sess.capture_at.take(),
            ckpt_rel,
            k,
            sampling: resolve_spec_sampling(sampling),
            graph_draft,
            prepared: None,
            wall: std::time::Duration::ZERO,
            call_local: std::env::var("MEMRA_PRIME_CALLLOCAL").as_deref() == Ok("1"),
        })
    }

    pub fn mtp_prime_walker<'a>(
        &'a self,
        e: &'a Engine,
        sess: &'a mut SpecSession,
        state: &'a mut Option<MtpPrimeState>,
        constraint: Option<&'a mut dyn SpecConstraint>,
    ) -> MtpPrimeWalker<'a> {
        MtpPrimeWalker {
            model: self,
            e,
            sess,
            state,
            constraint,
        }
    }
}

impl MtpPrimeWalker<'_> {
    fn trunk(&mut self, chunk: TrunkChunk) -> Result<(), PrimeError> {
        let s = self.state.as_mut().ok_or("MTP prime already finalized")?;
        let n = self.model.cfg.n_embd as usize;
        let h_all = s.hiddens.as_mut().ok_or("MTP hidden stack missing")?;
        if chunk.batched {
            let (logits, seed, h) = self.model.prime_chunk(
                self.e,
                &s.prompt[chunk.start..chunk.end],
                &mut self.sess.cache,
                s.base
                    + if s.call_local {
                        chunk.segment_end
                    } else {
                        s.prompt.len()
                    },
                chunk.start - chunk.segment_start,
                None,
            )?;
            self.e
                .copy_into(h_all, chunk.start * n, &h, (chunk.end - chunk.start) * n)?;
            drop(seed);
            s.logits = logits;
        } else {
            for i in chunk.start..chunk.end {
                let (logits, h) = if chunk.target_step {
                    self.model
                        .spec_target_step_h(self.e, s.prompt[i], &mut self.sess.cache)?
                } else {
                    self.model
                        .decode_step_h(self.e, s.prompt[i], &mut self.sess.cache)?
                };
                self.e.copy_into(h_all, i * n, &h, n)?;
                s.logits = logits;
            }
        }
        if chunk.end == chunk.segment_end && chunk.end < s.prompt.len() {
            if (s.base == 0 || s.capture_at.is_some())
                && (s.capture_at == Some(chunk.end) || s.ckpt_rel == Some(chunk.end))
                && let Ok(snap) = self.sess.cache.snapshot(self.e)
            {
                self.sess.boundary_captures.push(SpecBoundaryCapture {
                    snap,
                    pos: s.base + chunk.end,
                    logits: s.logits.clone(),
                    last_h: capture_boundary_hidden(self.e, h_all, chunk.end, n),
                    latent_tails: Vec::new(),
                });
            }
            if s.ckpt_rel == Some(chunk.end) {
                let anchor = self.e.uninit(n).and_then(|mut a| {
                    self.e.copy_view_into(
                        &mut a,
                        0,
                        &h_all.slice((chunk.end - 1) * n..chunk.end * n),
                        n,
                    )?;
                    Ok(a)
                });
                self.sess.turn_ckpt = match (self.sess.cache.snapshot(self.e), anchor) {
                    (Ok(snap), Ok(last_h)) => Some(SpecCheckpoint {
                        snap,
                        pos: s.base + chunk.end,
                        last_h,
                    }),
                    _ => None,
                };
            }
        }
        crate::progress::note_prime_rows(chunk.end - chunk.start);
        Ok(())
    }

    fn setup(&mut self) -> Result<(), PrimeError> {
        let s = self.state.as_mut().ok_or("MTP prime already finalized")?;
        let n = self.model.cfg.n_embd as usize;
        let tp = s.prompt.len();
        let h = s.hiddens.as_ref().ok_or("MTP hidden stack missing")?;
        if s.capture_at == Some(tp)
            && self.sess.boundary_captures.is_empty()
            && let Ok(snap) = self.sess.cache.snapshot(self.e)
        {
            self.sess.boundary_captures.push(SpecBoundaryCapture {
                snap,
                pos: s.base + tp,
                logits: s.logits.clone(),
                last_h: capture_boundary_hidden(self.e, h, tp, n),
                latent_tails: Vec::new(),
            });
        }
        if s.ckpt_rel.is_none() {
            let anchor = self.e.uninit(n).and_then(|mut a| {
                self.e
                    .copy_view_into(&mut a, 0, &h.slice((tp - 1) * n..tp * n), n)?;
                Ok(a)
            });
            self.sess.turn_ckpt = match (self.sess.cache.snapshot(self.e), anchor) {
                (Ok(snap), Ok(last_h)) => Some(SpecCheckpoint {
                    snap,
                    pos: s.base + tp,
                    last_h,
                }),
                _ => None,
            };
        }
        let sp = s.sampling;
        let pen_on = sp.temp > 0.0
            && sp.penalty_last_n > 0
            && (sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0);
        let history = if spec_pen_session_on() {
            self.sess.committed.as_slice()
        } else {
            &[]
        };
        let pen_hist = if pen_on {
            pen_window_seed(history, &s.prompt, sp.penalty_last_n)
        } else {
            Vec::new()
        };
        let mut logits = s.logits.clone();
        if let Some(c) = self.constraint.as_deref_mut() {
            c.mask_logits(&mut logits)
                .map_err(|e| format!("constraint: {e}"))?;
        }
        let boundary = if sp.temp > 0.0 && self.constraint.is_none() && spec_sampled_boundary_on() {
            sample_boundary_token(
                self.e,
                &logits,
                &sp,
                &pen_hist,
                &mut self.sess.sctr,
                "cold-prime",
            )?
        } else {
            argmax(&logits) as u32
        };
        if let Some(c) = self.constraint.as_deref_mut() {
            c.consume(boundary)
                .map_err(|e| format!("constraint: {e}"))?;
        }
        let init = self
            .model
            .spec_target_step_h(self.e, boundary, &mut self.sess.cache)?;
        let seed_buf = self.e.clone_dtod(&init.1)?;
        let fill_prev = self.e.clone_dtod(&init.1)?;
        let preds = self.e.alloc_u32_zeroed(s.k + 2)?;
        let (qt, rb) = self.model.embd.qt_and_row_bytes(n);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(self.model.embd_gpu.get_or_init(|| {
                self.e
                    .upload_u8(&self.model.embd.raw)
                    .expect("embed table upload")
            }))
        };
        let p_min = std::env::var("MEMRA_SPEC_PMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let setup = self.model.prepare_spec_draft(
            self.e,
            &mut self.sess.scratch,
            self.sess.draft_ctx.take(),
            s.base,
            s.k,
            s.graph_draft,
            sp,
            pen_on,
            self.sess.capture_disabled,
            self.constraint
                .as_deref()
                .is_some_and(|c| c.draft_mask_enabled()),
            p_min,
            embd_gpu,
            qt,
            rb,
        )?;
        self.sess.scratch.set_len(self.e, s.base)?;
        s.prepared = Some(PreparedMtp {
            prompt: s.prompt.clone(),
            base: s.base,
            wall: std::time::Duration::ZERO,
            logits: s.logits.clone(),
            hiddens: None,
            boundary,
            init: Some(init),
            sampling: sp,
            k: s.k,
            constrained: self.constraint.is_some(),
            draft_setup: Some(setup),
            seed_buf: Some(seed_buf),
            fill_prev: Some(fill_prev),
            preds: Some(preds),
        });
        Ok(())
    }

    fn fill(&mut self) -> Result<usize, PrimeError> {
        let s = self.state.as_mut().ok_or("MTP prime already finalized")?;
        let start = s.fill_cursor;
        let end = (start + s.fill_chunk).min(s.prompt.len());
        let tc = end - start;
        let n = self.model.cfg.n_embd as usize;
        let ph = s.hiddens.as_ref().ok_or("MTP hidden stack missing")?;
        let mut phs = self.e.zeros(tc * n)?;
        let (src, dst, count) = if start == 0 {
            (0, n, (tc - 1) * n)
        } else {
            ((start - 1) * n, 0, tc * n)
        };
        if start == 0
            && let Some(lh) = self.sess.last_h.as_ref()
        {
            self.e.copy_into(&mut phs, 0, lh, n)?;
        }
        if count > 0 {
            self.e
                .copy_view_into(&mut phs, dst, &ph.slice(src..src + count), count)?;
        }
        let (qt, rb) = self.model.embd.qt_and_row_bytes(n);
        let embd = if spec_host_embd() {
            None
        } else {
            Some((
                self.model
                    .embd_gpu
                    .get()
                    .ok_or("MTP embedding not prepared")?,
                qt,
                rb,
            ))
        };
        self.model.mtp_kv_fill_all(
            self.e,
            &s.prompt[start..end],
            &phs,
            s.base + start,
            &mut self.sess.scratch,
            embd,
        )?;
        s.fill_cursor = end;
        Ok(tc)
    }
}

impl PrimeWalker for MtpPrimeWalker<'_> {
    type Output = ();

    fn remaining_chunks(&self) -> usize {
        self.state.as_ref().map_or(0, |s| {
            if s.short {
                usize::from(s.fill_cursor < s.prompt.len())
            } else {
                s.chunks.len() - s.cursor + (s.prompt.len() - s.fill_cursor).div_ceil(s.fill_chunk)
            }
        })
    }

    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
        let s = self.state.as_ref().ok_or("MTP prime already finalized")?;
        let tracking = s.graph_draft && self.e.ctx().is_event_tracking();
        if tracking {
            unsafe {
                self.e.ctx().disable_event_tracking();
            }
        }
        let started = std::time::Instant::now();
        let result = (|| {
            let s = self.state.as_ref().ok_or("MTP prime already finalized")?;
            let short = s.short;
            let mut rows = 0;
            if s.cursor < s.chunks.len() {
                loop {
                    let s = self.state.as_ref().unwrap();
                    let chunk = s.chunks[s.cursor];
                    self.trunk(chunk)?;
                    rows += chunk.end - chunk.start;
                    let s = self.state.as_mut().unwrap();
                    s.cursor += 1;
                    if !short || s.cursor == s.chunks.len() {
                        break;
                    }
                }
                let s = self.state.as_ref().unwrap();
                if s.cursor == s.chunks.len() {
                    self.setup()?;
                    if short {
                        self.fill()?;
                    }
                }
                self.e.stream().synchronize()?;
                Ok(PrimeChunk {
                    phase: "mtp-trunk",
                    rows,
                })
            } else {
                rows = self.fill()?;
                self.e.stream().synchronize()?;
                Ok(PrimeChunk {
                    phase: "mtp-draft-fill",
                    rows,
                })
            }
        })();
        if tracking {
            unsafe {
                self.e.ctx().enable_event_tracking();
            }
        }
        if let Some(s) = self.state.as_mut() {
            s.wall += started.elapsed();
        }
        result
    }

    fn finish(self) -> Result<(), PrimeError> {
        if self.remaining_chunks() != 0 {
            return Err("MTP prime incomplete".into());
        }
        let mut s = self.state.take().ok_or("MTP prime already finalized")?;
        let mut ready = s.prepared.take().ok_or("MTP setup incomplete")?;
        ready.hiddens = s.hiddens.take();
        ready.wall = s.wall;
        if crate::alloc_trace_on() {
            eprintln!(
                "[mtp-prime-oracle] base={} rows={} target_logits={} boundary_count={} boundary_state={}",
                ready.base,
                ready.prompt.len(),
                crate::prime_receipt::logits_digest(&ready.logits),
                self.sess.boundary_captures.len(),
                crate::prime_receipt::capture_digest(self.e, &self.sess.boundary_captures)?
            );
        }
        self.sess.prime_ready = Some(ready);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_captures_preserve_segment_local_ranges_and_tiny_tail_program() {
        let chunks = trunk_schedule(2061, Some(1024), Some(2048), false, |n| vec![(0, n)]);
        assert_eq!(
            chunks
                .iter()
                .map(|c| (c.start, c.end, c.segment_start))
                .collect::<Vec<_>>(),
            vec![(0, 1024, 0), (1024, 2048, 1024), (2048, 2061, 2048)]
        );
        assert!(chunks[0].batched && chunks[1].batched);
        assert!(!chunks[2].batched && !chunks[2].target_step);
        assert_eq!(
            chunks
                .iter()
                .flat_map(|c| c.start..c.end)
                .collect::<Vec<_>>(),
            (0..2061).collect::<Vec<_>>()
        );
    }

    #[test]
    fn tokenwise_override_retains_the_legacy_segment_program() {
        let cold = trunk_schedule(2048, None, None, true, |_| panic!("batched cold tokenwise"));
        assert_eq!(cold.len(), 1);
        assert!(!cold[0].batched && cold[0].target_step);
        let split = trunk_schedule(2048, Some(1024), Some(1024), true, |n| vec![(0, n)]);
        assert_eq!(split.len(), 2);
        assert!(split[0].batched);
        assert!(!split[1].batched && !split[1].target_step);
    }
}
