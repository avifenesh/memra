//! glm5_next T-parallel speculative verify: the acceptance/rollback machinery that turns the
//! native MTP draft head (`mtp_head_forward_mla_cached`, lane/glm5-mtp-remint) into served
//! speculative decoding on the HyperConnections trunk.
//!
//! THE WALK-REUSE DECISION, stated once and load-bearing everywhere below: the verify of K
//! drafted tokens is a t=K+1 walk over the SAME trunk, and it rides the BATCHED-DECODE walk's
//! kernel classes (`hyper_batch_range_decode`, lane/glm53-batched-decode), NOT the prefill /
//! prime walk. Why:
//!
//!   * The batched walk's row-parallel ops are per-row BIT-EXACT vs the isolated t=1 decode
//!     step at every width in `1..=hyper_batch_cap()` (= `PRIME_MIN_T - 1` = 15, the shexp
//!     decode-exact knee, measured in `batched-decode-gate/31-KNEE-b16-forced.log`): the hc
//!     glue is block-per-token, the hc-mix GEMM runs per-row m=1 (`hyper::pre_exact`, the
//!     lt_ndep law), the MoE FFN's router/experts/shexp are per-row programs below
//!     `PRIME_MIN_T`, and the lm_head is `matmul_decode_exact`. That per-row exactness is
//!     what makes spec-vs-plain BYTE IDENTITY achievable at all.
//!   * The prime walk is a different numeric class: `hyper::pre` batches the mix GEMM
//!     (cuBLASLt n-dependent reduction), the FFN takes the prefill dispatch
//!     (`moe_ffn_il_prefill` / grouped GEMMs), and KDA's prefill conv arm reads the chunk
//!     window instead of the decode ring program. None of those are per-row bit-identical
//!     to the t=1 decode chain, so a prime-walk verify could never pass a byte-identity
//!     gate against plain decode.
//!   * The knee bounds K: K+1 <= 15, i.e. K <= 14 — the DFlash2 probe's K<=7 drafter and
//!     upstream's 5/7-draft MTP configs fit with margin.
//!
//! The ONE difference from `hyper_batch_range_decode`: the batched walk runs B independent
//! sessions (each row -> its OWN cache at its own single position); the verify walk runs
//! K+1 SEQUENTIAL positions of ONE session, so the mixers chain state row -> row through the
//! one cache (KDA: the t=1 recurrent step per row, exactly the serving decode program; MLA:
//! the t=1 `mla_attn_cached` append+attend per row, so row r attends rows `0..pos0+r+1` —
//! causal within the drafted block by construction). This is the "hyper rows-walk with
//! causal verify appends" the batched-decode lane's standing refusals named as missing.
//!
//! VERIFY-ROW BATCHING (lane/glm5-verify-batch, 2026-08-30 — the flip re-battery's named
//! flip condition: "the verify walk must stop paying one plain-step per row", ~24-26 ms/row
//! measured): `MEMRA_GLM5_VERIFY_BATCH` (default ON) restructures the mixer walk PER LAYER
//! while keeping the sequential contract exactly where the math demands it. Each KDA layer
//! runs ONE t=K+1 `kda_core` call — projections/gates/conv batched through the decode-exact
//! matmul classes (`matmul_rows_exact`; the bf16 tcols twin reads each weight ONCE for all
//! t rows) with the recurrence SEQUENTIAL INSIDE one `memra_kda_scan_s128` launch (the
//! in-kernel T-loop over register state IS the chained t=1 program). Each MLA layer runs
//! ONE t=K+1 `mla_attn_cached_rows_exact` call (per-query causal kpool selection + gathered
//! attention by construction). Rollback on the batched arm: conv ring = pre-round snapshot
//! + re-roll(T=keep) over stolen raw rows; ssm = ONE scan replay at T=keep from the
//! pre-round snapshot over the stolen batched inputs (`kda::KdaRowsStash`). `0` = the
//! per-row walk below, byte-for-byte — the rollback seam. Per-row byte identity vs the
//! plain tape is held by this file's standing batteries running the batched arm, plus the
//! kernel bit-gates in `tests/glm5_verify_batch_gpu.rs`.
//!
//! ROLLBACK (the hard part, per the engine survey's upstream reading — vLLM keeps
//! num_spec+1 KDA state columns and commits the last accepted one; SGLang's ReplaySSM keeps
//! an input ring and replays the accepted prefix):
//!
//!   * KDA recurrent state: SNAPSHOT + SCAN-INPUT REPLAY (lane/glm5-loop-port port 3 —
//!     the GdnStash/ReplaySSM diet this doc used to name as the follow-up, landed). The
//!     walk clones the resident ssm state ONCE per layer per round (before row 0) and
//!     STEALS each row's scan-input buffers (`kda::KdaScanInputs`, ~160 KB/row/layer,
//!     zero copies — the step allocated them either way); accept-j REBUILDS the state by
//!     re-issuing rows 0..=j's original t=1 `memra_kda_scan_s128` launches from the
//!     snapshot (`kda::kda_scan_replay`) — byte-identical to the retired per-row clone
//!     by construction, since each replay is the very launch that produced it. Full
//!     accept keeps the resident state, no work. Memory, stated: one glm5_next KDA state
//!     is 64 heads x 128 x 128 f32 = 4 MiB, x34 KDA layers = 136 MiB per round (was 136
//!     MiB PER COLUMN — ~0.95 GiB of transient clones at K=7, retired to ~136 MiB + K x
//!     ~160 KB x 34 of stolen stash). The conv ring stays per-row cloned (288 KiB/row/
//!     layer, 1.4% of the ssm plane — not worth a replay arm).
//!   * MLA latent rows: TRUNCATE (append-only, position-addressed): `len = snap + keep`,
//!     device mirror in lock-step.
//!   * kpool index planes: the tail ring drains IN-CALL (`mla_kpool_indices`), so pool keys
//!     over drafted rows may FINALIZE during the verify walk. Rollback clamps
//!     `index_pools_ready` via `truncate_index_pool_keys` (the clamp the field's own doc
//!     demands of "the same code that shortens len"); the next call rebuilds keys for the
//!     re-appended rows from `[ready, len/pool)`. The residency tripwire in
//!     `mla_kpool_indices` fails LOUDLY if any rewind forgets this — that tripwire is the
//!     red arm of the rollback gate.
//!   * MTP draft plane (il = n_trunk): len reset by the loop (LANE.md contract: "rollback =
//!     plane len reset"), same pool-key clamp.
//!
//! Gate: `tests/glm5_tparallel_verify_gpu.rs` — accept-j-then-continue byte identity vs the
//! never-drafted sequential path for every j in 0..=K, red-proven with a stale-KDA-state
//! mutation and a pool-key-finalized-past-j mutation; plus end-to-end spec-vs-plain greedy
//! tape identity at K=1..7 with forced-rejection positions, red-proven by disabling
//! rollback. The SERVED shape (worker-sized bursts over one [`Glm5SpecSession`], state
//! carried across burst boundaries, sampled twin, EOS, receipt log red/green) is gated by
//! `tests/glm5_spec_session_gpu.rs`.
//!
//! SAMPLED ACCEPTANCE (landed, lane/glm5-spec-routing 2026-08-30): greedy
//! longest-matching-prefix below is the byte-deterministic instrument
//! (greedy-is-the-instrument law). The sampled arm applies memra's existing spec sampled
//! contract — `spec::SpecSampling` with the rejection-sampling accept walk
//! (`u_j * q_j(x_j) < p_j(x_j)`, host Philox4x32-10 stream tag 0xFFFF_FFFE via
//! `spec::host_u01`, residual resampling on the first rejection) — the same contract
//! `generate_spec_inner2`'s MEMRA_SPEC_TEMP>0 route and the dspark sampled-admission walk
//! consume. It plugs in at exactly one seam (`glm5_sampled_accept`, the accept rule over
//! the verify logit rows, with the draft chain drawn from the SAME filtered distribution
//! the q gather reads); nothing in the walk or the rollback changes. Philox counters live
//! ON the [`Glm5SpecSession`] so randomness never repeats across serve bursts (the
//! session-continuity law; burst-split invariance is pinned by `glm5_spec_session_gpu`).
//! PENALIZED sampled requests are refused — no penalty arm yet; the worker keeps them on
//! the plain path.
//!
//! FR-SPEC VOCAB MASKING (owner addition, 2026-08-30 — the house spec recipe, the q38 way):
//! the loop consumes the existing `MEMRA_FRSPEC_TRIM` contract, no new flag. The loader
//! already reaches this head: the trim match in `hybrid.rs` consumes the embedded head
//! regardless of arch, `frspec_trim_own_head_name(n_trunk)` misses (glm5_next ships no
//! private MTP lm_head) and the gather falls back to the trunk `output.weight` /
//! `token_embd.weight` — which for glm5_next is EXACT BY CONTRACT, not merely by tying:
//! the MTP block projects through the trunk lm_head (LANE.md; the draft gate pins
//! `shared_head_head.is_none()` untrimmed). With a ranks artifact loaded,
//! `mtp_head_forward_mla_cached` already projects through the gathered rows (its head is
//! `shared_head_head.unwrap_or(trunk)`), so the DRAFT logits arrive `[n_ranks]`; this
//! loop's seam is the remap: every draft argmax is a RANK id and maps through `d2t` back
//! to the true vocab BEFORE it is drafted, chained (e_tok drives the embedding gather),
//! or verified. THE VERIFY WALK STAYS FULL-VOCAB AND UNTOUCHED — a trimmed draft can only
//! change WHICH tokens get drafted, never how they verify; that invariant is the whole
//! design (q38's measured skipped-remap defect was 0/248 acceptance with every exactness
//! gate green — silent, which is why the gate below makes it loud). Rank artifacts for
//! glm5 are an INPUT DEPENDENCY: the corpus mint (SXC pools through GLM's tokenizer,
//! per traffic class, the q38 plain-text format) is the owner's CPU-only lane; the
//! self-trim d2t arm needs no external artifact and lands here first.
//!
//! PPN (lane/glm5-ppn-verify, 2026-08-30): the verify walk owns its stage split exactly as
//! the batched decode walk does — `glm5_verify_rows_ppn` mirrors
//! `decode_step_batch_hyper_ppn` (per-stage engine, per-stage pos_rows, ONE
//! `[t, streams, n_embd]` boundary payload per cut; row chaining is per-LAYER through the
//! one cache, so a straight layer-range split preserves it exactly). Rollback restores each
//! stage's layers through that stage's engine on its stream; the MTP block, its latent
//! plane and every draft-chain/accept-side op ride the LAST stage's engine
//! (`glm5_head_engine` — where the loader puts the lm head and `pp::new_cache*` puts the
//! trailing MTP plane), so the h_seed carrier never bounces devices. Gate:
//! `glm5-spec-ppn-gate` (the tparallel battery under the split, stages=2 and 3, red-proven;
//! the cross-device twin is the box arm). Worker admission bounds sharded placements to the
//! GATED stage set (`glm5_sharded_placement_admits`, worker.rs) — everything else stays
//! fail-closed by name.
//!
//! DRAFT SOURCE SEAM (lane/glm5-dflash-draft-src, 2026-08-30): the session's drafts come
//! from ONE of two sources, selected at load and pinned for the session — everything from
//! the verify walk on (accept, rollback, commit, receipts, K policy) is SHARED and
//! source-blind, which is where the exactness invariant lives (a draft source can only
//! move acceptance, never output):
//!
//!   * `NativeMtp` (existing): the embedded NextN head chains K drafts through
//!     `mtp_head_forward_mla_cached`; requires `MEMRA_GLM5_MTP=1`.
//!   * `Dflash2` (`MEMRA_GLM5_DFLASH=<dir-or-hf-spec>`): the pinned
//!     incoai/GLM-5.3-Flash-DFlash2 block-diffusion drafter (owner holds WRITTEN APPROVAL
//!     from the DFlash2 owners, 2026-08-30, for use beyond probe/eval). It REUSES the
//!     shipped q38 DFlash2 machinery verbatim (`DflashDraft`: `ctx_features` ->
//!     `ingest_ctx` -> `forward_round` -> `dflash2_propose_*`, mask-fill harvest, selector
//!     walk); the ONE glm5-specific input is the drafter's measured feature contract — the
//!     STREAM-MEAN (`hc_contract`) of the COMPLETED trunk layer output at the drafter
//!     config's `target_layer_ids` (plan layers 5,14,24,33,42 on the real artifact), the
//!     exact definition the probe's `MEMRA_TRACE_LAYER_ROWS` capture seam banked 0.73
//!     acc@1 / 3.06 tokens-per-cycle against
//!     (research/glm53-flash-bringup-20260827/dflash2-probe-20260829/RECEIPTS.md). The
//!     features flow through [`crate::cache::HcTapSink`], a HOST sink filled by the hc
//!     prime walks and this file's verify walk (host-staged so a ppN split needs no
//!     cross-device tap plumbing; the drafter itself runs on the HEAD engine, where the
//!     trunk lm_head it projects through lives). THE NATIVE MTP HEAD IS NOT LOADED for
//!     this source (the q38 pattern — a full MoE trunk layer of VRAM back); the plan's
//!     trailing MTP cache plane still allocates (plan-structural, ~`ctx * latent_width`
//!     f32 per declared block — named cost, not forked). Sampled route: the drafter's
//!     selector proposal records its true q (`DsparkDraftSample::Selector`) and the accept
//!     rides the SAME `dspark_accept_sampled` rejection walk the q38 serve route ships,
//!     with this session's Philox counters (`uctr` selector/accept draws, `sctr`
//!     bonus/residual) so randomness never repeats across bursts. K is bounded by the
//!     drafter block (K <= block_size-1 = 7): the worker clamps, the burst refuses loudly.
//!     Selection receipt: boot logs `[glm5-spec] draft source = native-mtp` or
//!     `[glm5-spec] draft source = dflash2 @ <sha8>`; both flags off = plain serving
//!     (fail-closed warn). `MEMRA_GLM5_DFLASH_GATE_RED=tap-shift` is a GATE INSTRUMENT
//!     (never a serving flag): it shifts every tap layer +1 to red-prove that a wrong
//!     feature input collapses acceptance while the tape stays byte-identical.
//!
//! SERVING EXPOSURE (lane/glm5-spec-routing, 2026-08-30): `MEMRA_GLM5_SPEC` (default OFF,
//! FLAGS.md row) is the ONE master flag — it routes `generate_spec` here for hc trunks
//! with a loaded MTP head AND arms the worker route (`glm5_spec_capable` +
//! `step_glm5_spec` driving [`Glm5SpecSession`] bursts). OFF = the named `refuse_hyper`
//! refusal and zero `[glm5-spec]` log lines, byte-identical serving. The MTP_SPEC
//! capability manifest remains deliberately UNEXTENDED — worker `mtp_spec_capable` stays
//! false for glm5_next plans; the serving capability lives in its OWN manifest
//! (`GLM5_SPEC`, execution_manifest.rs) whose table names exactly the glm5_next class, and
//! a SEALED production bundle still fails closed until it banks a `glm5-spec.v1` rewrite
//! receipt (the real-artifact qualification lane's job).

use crate::Engine;
use crate::cache::{Cache, HcTapSink};
use crate::dflash::{DflashDraft, DflashKv, DsparkDraftSample};
use crate::forward::argmax;
use crate::hybrid::{HybridModel, Mixer};
use crate::spec::SpecSampling;
use crate::spec_phase::{
    ProfClock, SPEC_PROF_ROUNDS, SpecFirstTokenProf, SpecPhaseNs, SpecRoundProf, SpecRoundsLog,
    V_SEQ_ROWS, spec_prof_on,
};
use cudarc::driver::CudaSlice;

type Res<T> = Result<T, Box<dyn std::error::Error>>;
/// Round-cadence commit hook of `glm5_spec_session_burst_streamed` (lane/b200-spec-ttft-
/// 20260902): called with every newly committed slice of a burst, in order, disjoint.
pub type CommitHook<'a> = &'a mut dyn FnMut(&[u32]);

/// `MEMRA_GLM5_SPEC=1` routes `generate_spec` to the glm5 T-parallel loop. Default OFF:
/// unset/0 keeps the standing `refuse_hyper` refusal, so serving is byte-identical to the
/// pre-lane binary. Read once (worker chunk policies read their flags the same way).
pub fn glm5_spec_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_GLM5_SPEC").as_deref() == Ok("1"))
}

// Per-burst phase attribution moved to `crate::spec_phase` (lane/glm5-extract-general):
// the draft/verify/accept/roll/maint split is spec-family-generic. This loop consumes
// `MEMRA_SPEC_TRACE` (glm5 alias `MEMRA_GLM5_SPEC_TRACE` stays honored) and passes its
// own `[glm5-phase]` / `[glm5-phase-v]` tags so every banked receipt keeps its shape.

/// `MEMRA_GLM5_VERIFY_BATCH` (default ON, lane/glm5-verify-batch): the per-LAYER batched
/// mixer walk — one t=K+1 KDA call per layer (projections/conv/gates batched through the
/// decode-exact classes, the recurrence sequential INSIDE one scan launch) and one
/// t=K+1 rows-exact MLA call per layer, replacing the per-row mixer loop. `0` restores
/// the per-row walk byte-for-byte — the rollback seam. Deliberate default (new-flags
/// law): the walk only exists behind `MEMRA_GLM5_SPEC` (default OFF in prod), per-row
/// byte identity is bit-gated on the rig (`glm5_tparallel_verify_gpu` +
/// `glm5_verify_batch_gpu`), and the box re-battery A/Bs this seam in one build. Read
/// PER CALL (the `MEMRA_KDA_FUSED_PROJ` per-call precedent) so gates drive both arms in
/// one process; one env read per verify walk.
pub fn glm5_verify_batch_on() -> bool {
    std::env::var("MEMRA_GLM5_VERIFY_BATCH").as_deref() != Ok("0")
}

/// `MEMRA_GLM5_DRAFT_PRIME_V2` (default OFF, lane/spec-route-depth-20260902): the DFlash2
/// drafter's prompt prime rides the target's own chunk schedule. OFF (the pre-lane
/// literal): the target prime fills a whole-prompt HOST tap sink (`[prompt, 5 x 4096]`
/// f32, 21 GB at 256k) through synchronous pageable DtoHs, and round 1 re-uploads it in
/// 256-row pageable HtoD chunks before the fc + k/v ingest. ON: each prime chunk
/// (`hyper_prime_ranges`, 4096 rows) arms a DEVICE-staged chunk sink, and right after that
/// chunk's trunk walk the drafter ingests it: slot DtoH into a pinned cacheable buffer,
/// host interleave, ASYNC HtoD from pinned, fc + k/v ingest at the chunk width. The trunk
/// program is byte-identical (the same per-range `prime_cache` calls with the same
/// `queued_after`); the drafter's KV rows are the same GEMM class at a different M (rig
/// gate 15 pins bit-identity at equal chunking and tape identity always). Read per session
/// creation so a gate can flip it in-process. Rollback seam: unset.
pub fn glm5_draft_prime_v2_on() -> bool {
    std::env::var("MEMRA_GLM5_DRAFT_PRIME_V2").as_deref() == Ok("1")
}

/// `MEMRA_GLM5_DRAFT_TAPS_DEVICE` (DEFAULT ON since boot D, 2026-09-03; `=0` = the
/// host-tap rollback seam; lane/spec-route-depth-20260902): the DEVICE-RESIDENT drafter
/// prime. Boot D on the 2x B200 pair (this arm) against boot C (host taps), TTFT s at
/// 4k / 42k / 128k / 256k: 4.89 / 16.84 / 56.62 / 129.42 vs 5.32 / 21.23 / 72.90 /
/// 162.26, drafter prime ms 17.9 / 59.1 / 180.7 / 362.4 vs 91.4 / 561.6 / 4430.8 /
/// 8868.4, steady rounds and decode-after unchanged (FLAGS row, receipt path there). The trunk prime stays the ONE whole-prompt call (the
/// chunked arm's per-range calls re-entered the PP-2 microchunk geometry and made the trunk
/// prime itself far slower — boot B); the tap rows never leave the device: the walk stages
/// each range's five contracted tap planes on the writing stage's device (a chunk-sized
/// ring, `HcTapSink::new_device_staged_at`), and at the range boundary the prime's own loop
/// hands the range to `glm5_taps_range_done`, which interleaves the planes into the fc
/// layout with 2D device copies (a peer copy first for planes written on another stage),
/// runs `ctx_features` + `ingest_ctx` at the range width (4096 rows: the batched GEMM
/// class), and appends to the drafter KV. No DtoH in the prime, no HtoD in the drafter
/// prime; the eager arm's 7.5 s of pageable HtoD at 256k (boot B's split: h2d 7505.8 of
/// 8870 ms) goes away outright, and its `tap_dtoh` with it. Read per session creation
/// (a gate flips it in-process). The host-tap arms (`MEMRA_GLM5_DRAFT_PRIME_V2`,
/// `MEMRA_GLM5_DRAFT_PRIME_LAZY`) are reachable only under `=0`.
pub fn glm5_draft_taps_device_on() -> bool {
    std::env::var("MEMRA_GLM5_DRAFT_TAPS_DEVICE").as_deref() != Ok("0")
}

/// The device-resident drafter prime's in-flight state (doc on
/// [`glm5_draft_taps_device_on`]), carried type-erased on the prime's tap sink
/// (`HcTapSink::ingest_state`) so the prime's range loop can hand it each range.
struct Glm5DraftPrimeInflight {
    kv: DflashKv,
    taps: Vec<usize>,
    n_embd: usize,
    /// Rows the sink ring covers (the largest range the schedule emits).
    ring: usize,
    /// Head-device staging for planes written on another stage (lazily sized `ring x h`).
    stage: Vec<Option<CudaSlice<f32>>>,
    /// The interleaved `[ring, n_taps * h]` fc input on the head device (lazy).
    rows_dev: Option<CudaSlice<f32>>,
    prof_on: bool,
    copy_ms: f64,
    feat_ms: f64,
    kv_ms: f64,
    chunks: usize,
}

/// `MEMRA_GLM5_DRAFT_PRIME_LAZY` (default OFF, lane/spec-route-depth-20260902): `=1`
/// restores the pre-lane placement of the EAGER arm's drafter ingest — inside round 1 of
/// the first burst, i.e. AFTER the prime's anchor token has been emitted under the
/// round-cadence door. Default (unset): the ingest runs at session creation, before the
/// session is returned and before any token is emitted. WHY THE FLIP: the 2x B200 pair's
/// boot A (`MEMRA_SPEC_PROF=1`, main + PR #101) measured the round-0 wall at 0.63 / 4.5 /
/// 8.9 s for 42k / 128k / 256k prompts (about 35 us per prompt token, the eager ingest)
/// while every later round sits at 55-64 ms; with the anchor already streamed, that one
/// round lands INSIDE decode, which is the bimodal "15.4 vs 32.1 tok/s" the pair saw
/// (256 tokens over 8.9 s + 255 rounds/37 tok/s = 16 tok/s; without the stall, 37). The
/// work is identical either way (same rows, same GEMMs, same KV bytes: the KV is
/// position-addressed and nothing touches it between creation and round 1); only WHEN it
/// runs moves, from the second token's latency to TTFT. Read per session creation.
pub fn glm5_draft_prime_lazy_on() -> bool {
    std::env::var("MEMRA_GLM5_DRAFT_PRIME_LAZY").as_deref() == Ok("1")
}

/// Drafter ctx KV bytes at `cap` rows (the `DflashKv::new` geometry), for the profile.
fn dflash_kv_bytes(cfg: &crate::dflash::DflashCfg, cap: usize) -> usize {
    2 * cfg.n_layer * (cap + cfg.block_size) * cfg.n_kv * cfg.head_dim * std::mem::size_of::<f32>()
}

/// Pinned host buffer viewed as `n` f32s (page-aligned by construction).
fn pinned_f32(buf: &crate::PinnedHostBuf, n: usize) -> &[f32] {
    debug_assert!(n * std::mem::size_of::<f32>() <= buf.len());
    // SAFETY: malloc_host allocations are page-aligned and the length is checked above.
    unsafe { std::slice::from_raw_parts(buf.as_slice().as_ptr() as *const f32, n) }
}

fn pinned_f32_mut(buf: &mut crate::PinnedHostBuf, n: usize) -> &mut [f32] {
    debug_assert!(n * std::mem::size_of::<f32>() <= buf.len());
    // SAFETY: as above; exclusive borrow of the buffer.
    unsafe { std::slice::from_raw_parts_mut(buf.as_mut_slice().as_mut_ptr() as *mut f32, n) }
}

/// `MEMRA_GLM5_SPEC_PREFIX` (default OFF, lane/glm5-prefix-latent2 2026-09-01): the glm5
/// spec x prefix-cache interplay — BOTH sides, one flag (the `MEMRA_DSPARK_PREFIX_RESTORE`
/// precedent): (capture) DFlash2-source sessions take a prompt-boundary capture at creation
/// for the worker's deferred prefix publication, and (restore) the worker may convert a
/// prefix hit into `glm5_spec_session_from_restored` instead of demoting it to the plain
/// route. Requires `MEMRA_PREFIX_LATENT=1` too — a capture the worker's publisher would
/// refuse (latent entries need the plane flag) is pure waste, so this predicate ANDs both
/// env reads. DEFAULT OFF BY DESIGN (new-flags law): the restored-session program is
/// unmeasured until the box battery banks restored-vs-cold byte identity on the
/// continuation; unset restores the pre-lane posture exactly (spec sessions never capture,
/// hits demote to plain). Read once per process.
pub fn glm5_spec_prefix_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let sp = std::env::var("MEMRA_GLM5_SPEC_PREFIX").as_deref() == Ok("1");
        let pl = std::env::var("MEMRA_PREFIX_LATENT").as_deref() == Ok("1");
        if sp && !pl {
            // A mis-built recipe would otherwise only surface through the battery's
            // receipt gates (PR #96 review round 2, minor) — say it at first read.
            eprintln!(
                "[glm5-spec] MEMRA_GLM5_SPEC_PREFIX=1 is INERT: it requires \
                 MEMRA_PREFIX_LATENT=1 (latent entries could never publish without it)"
            );
        }
        sp && pl
    })
}

/// `MEMRA_GLM5_SPEC_FULLCOVER` (default OFF, lane/glm5-fullcover-spec-route 2026-09-02):
/// admit the glm5 spec route on a FULL-COVER prefix hit, i.e. a hit whose restored prefix
/// already covers the whole prompt so there is no suffix left to prime.
///
/// WHY IT EXISTS (memra#74): the parent lane refused an empty suffix on the premise that
/// "the plain boundary-logits resume is faster than any prime". That is true of PREFILL and
/// says nothing about DECODE: with the drafter left un-armed the whole generation then ran
/// at plain speed. Measured on the live glm5 box 2026-09-02, same minute and vantage,
/// vendor-default sampled, 67-token prompt, 512 max_tokens: repeated (full-cover hit)
/// 29.46 s wall median / 30.8 tok/s decode vs fresh-nonce (cold, route=spec) 57.75 / 69.7.
/// A cache hit cost the customer half the decode speed on that shape.
///
/// WHY IT IS WELL-FORMED: with an empty suffix the restored session's state is exactly a
/// cold session's at the same boundary: trunk cache at `fed.len()`, drafter ctx KV at
/// `fed.len()` (`DflashKv::from_tail`, the same rows a cold prime's tap ingest would have
/// produced), no pending tap rows, and the anchor drawn from the ENTRY's boundary logits by
/// the same rule the cold burst applies to its own first token. The Dflash2 round invariant
/// `kv.len == cache.pos` therefore holds at round 1 with `pending` empty. This is the same
/// full-cover shape the MTP restore has served since lane/spec-cache
/// (`spec_restore_refusal`: a full-cover hit is admitted whenever the entry carries its
/// boundary hidden + logits).
///
/// DEFAULT OFF BY DESIGN (new-flags law): the arm has NO GPU receipt yet. Byte identity of
/// the full-cover restored tape against plain decode is GATE 13 of
/// `glm5_dflash_session_gpu` (`MEMRA_GLM5_SPEC_FULLCOVER=1`), and the serving win needs the
/// repeated-prompt cell on the glm5 box. Unset restores the pre-lane posture exactly
/// (full-cover hits serve plain, now with `reason=full-cover-hit` on the route line).
/// Read PER CALL (the `MEMRA_KDA_FUSED_PROJ` / `MEMRA_GLM5_VERIFY_BATCH` precedent) so one
/// gate process can drive both arms. Rollback seam: unset.
pub fn glm5_spec_fullcover_on() -> bool {
    std::env::var("MEMRA_GLM5_SPEC_FULLCOVER").as_deref() == Ok("1")
}

/// `MEMRA_GLM5_SPEC_TP` (default OFF, lane/glm5-composition 2026-09-01): admit glm5 spec
/// SESSIONS on a `MEMRA_GLM5_TP`-armed model. DEFAULT OFF BY DESIGN (new-flags law): the
/// composition's verify/rollback wiring is rig-gated for correctness (per-rank KDA
/// snapshot/replay, per-replica MLA latent truncation — `glm5-tp-gate` arms S*), but it has
/// ZERO real-artifact receipts and the TP serving wiring is still the named box increment;
/// an unmeasured composition does not default ON. `=1` lifts ONLY the session co-refusal —
/// every other admission law holds (draft source required, batched verify walk required:
/// the per-row rollback seam carries no TP arm and refuses by name). Read per session
/// creation. Rollback seam: unset (the co-refusal is restored verbatim).
pub fn glm5_spec_tp_on() -> bool {
    std::env::var("MEMRA_GLM5_SPEC_TP").as_deref() == Ok("1")
}

/// `MEMRA_SPEC_PMIN`, honored by the glm5 loop (loop-port 2 — the step37 shipping family,
/// `MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1` is what step37 serves; NO new flag): stop the
/// draft chain early when the drafter's confidence in its own pick drops below p_min.
/// Native chain: p = the head's softmax confidence in its pick (the spec.rs `g_p`
/// statistic, `prob_of_token_device`). DFlash2: q = the selector's recorded per-slot
/// candidate-set confidence (`q_chosen`; T=1 twin on the greedy walk) — the owner's
/// "take only high confidence offers" tau-slot form, truncated PRE-verify.
/// Unset/0 = OFF (today's rounds, byte-identical). The VALUE is a per-model measurement
/// (spec.rs bank: q27 PMIN=0.3 was -1.9% on one pack; step37 ships 0.5) — the box-B tau
/// ladder prices glm5's.
pub(crate) fn glm5_pmin() -> f32 {
    use std::sync::OnceLock;
    static P: OnceLock<f32> = OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("MEMRA_SPEC_PMIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
}

/// `MEMRA_SPEC_PMIN0=1` (llama.cpp's draft gating, vendored via spec.rs): the p-min gate
/// applies at slot 0 too, so a low-confidence round drafts NOTHING and the verify batch is
/// just the anchor row — m=1 = a plain decode step. Always legal for glm5 (the anchor row
/// exists every round). "llama's 35B win rides exactly this — draft acceptance 76% at mean
/// len 2.5 because unpredictable stretches never pay draft+verify overhead" (spec.rs).
pub(crate) fn glm5_pmin0() -> bool {
    use std::sync::OnceLock;
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| std::env::var("MEMRA_SPEC_PMIN0").as_deref() == Ok("1"))
}

/// MEMRA_SPEC_PMIN break semantics — hoisted to the shared K-policy surface
/// ([`crate::spec::spec_conf_keep`], lane/glm5-extract-general); re-exported here so the
/// glm5 gates and call sites keep their name.
pub use crate::spec::spec_conf_keep as glm5_conf_keep;

/// The loaded DFlash2 drafter (module doc, DRAFT SOURCE SEAM): the model-level half of the
/// `Dflash2` draft source — weights loaded ONCE per model on the head engine (`hybrid.rs`,
/// `MEMRA_GLM5_DFLASH`); per-session state lives in [`Glm5DraftState`].
///
/// HOISTED to the general seam ([`crate::dflash::DflashDrafter`], lane/glm5-extract2): the
/// holder is `{ drafter weights, byte-identity pin }` with nothing glm5 in it. Re-exported
/// here under its old name so glm5's call sites and gates keep the name they were written
/// against, exactly as `glm5_conf_keep` does above.
pub use crate::dflash::DflashDrafter as Glm5DflashDrafter;

/// Per-session draft-source state (module doc, DRAFT SOURCE SEAM). Selected at session
/// creation from the model's loaded sources and pinned for the session's lifetime.
pub(crate) enum Glm5DraftState {
    /// Embedded NextN head: state = the MTP latent plane + `Glm5SpecSession::pending`
    /// (token, h_seed) pairs — the pre-seam program, byte-identical.
    NativeMtp,
    /// DFlash2 block-diffusion drafter: state = the drafter's own ctx-feature KV cache
    /// plus host feature rows not yet ingested. Invariant at every round boundary:
    /// `kv.len + pending.len()/(taps.len()*n_embd) == committed.len()` — the drafter's
    /// context is exactly the committed tokens (the probe's `F_feat[new_lo:start]` walk).
    Dflash2 {
        kv: DflashKv,
        /// Committed-position feature rows awaiting ingest, `[n, n_taps*n_embd]` host
        /// (the prompt's prime taps at session start; each round's kept verify taps after).
        pending: Vec<f32>,
        /// Resolved tap layers (drafter config `target_layer_ids`, red-arm shift applied).
        taps: Vec<usize>,
    },
}

/// The retained q side of one round's draft chain — what the sampled accept walk consumes.
/// Greedy rounds carry `None` (the accept is the byte-deterministic prefix walk).
enum Glm5DraftQ {
    None,
    /// Native MTP chain: per-slot retained draft logits (rank space under a trim) + the
    /// filtered stats of the distribution each draft was drawn from.
    Mtp {
        draft_idx: Vec<u32>,
        draft_logits: Vec<CudaSlice<f32>>,
        draft_stats: Vec<(f32, f32, f32)>,
    },
    /// DFlash2 selector proposal (the recorded candidate-set q) + the retained draft-logit
    /// rows `dl` (`dspark_accept_sampled`'s buffer contract; unread on the Selector q path).
    Selector {
        prop: DsparkDraftSample,
        dl: CudaSlice<f32>,
    },
}

/// Resolve the drafter's tap layers against the trunk: the drafter config's
/// `target_layer_ids` are memra PLAN layer indices whose COMPLETED output feeds the fc
/// (the probe's capture convention: `MEMRA_TRACE_LAYER_ROWS_LAYERS=5,14,24,33,42` == the
/// drafter's own `target_layer_ids`, asserted 1:1 in `score_dflash2.py`).
/// `MEMRA_GLM5_DFLASH_GATE_RED=tap-shift` is the RED-ARM INSTRUMENT: every tap moves +1
/// layer — deliberately wrong features whose acceptance collapse the gate asserts while
/// the output tape stays byte-identical. Unknown values refuse loudly.
///
/// The resolution itself is the general seam ([`crate::dflash::resolve_tap_layers`],
/// lane/glm5-extract2); what stays here is glm5's OWN red arm — the gate instrument reads its
/// env, prints its `[glm5-spec]` tag, and hands the shift in as a parameter. Error bytes are
/// unchanged ("glm5 DFlash2" is the `what` label).
fn glm5_dflash_tap_layers(draft: &DflashDraft, n_trunk: usize) -> Res<Vec<usize>> {
    let shift = match std::env::var("MEMRA_GLM5_DFLASH_GATE_RED").ok().as_deref() {
        Some("tap-shift") => {
            eprintln!(
                "[glm5-spec] RED-ARM tap-shift: drafter tap layers shifted +1 (gate \
                 instrument, never a serving flag)"
            );
            1
        }
        Some("") | None => 0,
        Some(other) => {
            return Err(format!(
                "MEMRA_GLM5_DFLASH_GATE_RED={other:?}: unknown red arm (want tap-shift)"
            )
            .into());
        }
    };
    Ok(crate::dflash::resolve_tap_layers(
        &draft.cfg.target_layer_ids,
        n_trunk,
        shift,
        "glm5 DFlash2",
    )?)
}

/// Pre-round state checkpoint for one glm5 verify round. Captured by `glm5_verify_rows`
/// BEFORE any row runs; consumed by `glm5_verify_rollback`.
///
/// Covers exactly the state planes a glm5_next trunk mutates in a verify round:
/// - `latent_len`: per-layer MLA latent length at round start (rollback = truncate).
/// - KDA state (loop-port 3, the module doc's GdnStash/ReplaySSM diet LANDED): the old
///   per-row (conv, ssm) column clones — 4 MiB x 34 layers per COLUMN, ~0.95 GiB of
///   transient at K=7 — are replaced by
///   * `kda_ssm_snap[il]`: ONE recurrent-state clone per layer per round (the state
///     BEFORE row 0),
///   * `kda_scan_stash[il][r]`: row r's scan-input buffers, STOLEN from the step (zero
///     copies, ~160 KB/row/layer — `kda::KdaScanInputs`), rows `0..t-1` except the last
///     (`keep == t` needs no restore, so row `t-1` is never a replay target),
///   * `kda_conv_cols[il][r]`: the conv ring stays PER-ROW CLONED (288 KiB, 1.4% of the
///     ssm plane it rode beside — not worth a replay arm).
///
///   Partial-accept rollback REPLAYS rows `0..keep` from the snapshot
///   (`kda::kda_scan_replay`): each replay is the original t=1 scan launch re-issued over
///   the very buffers that row consumed, so the rebuilt state is byte-identical to the
///   clone it replaces by construction. Under a ppN split every clone/stash lives on its
///   layer's OWNING stage engine; rollback restores through the same per-stage seam.
/// - `pos`: `cache.pos` at round start.
///
/// glm5_next has no Full/Linear trunk mixers (the walk refuses them by name), so `kv`,
/// `tp_kv` and GDN stashes have no arm here — growing one is a deliberate extension with
/// its own gate, not a silent default.
pub struct Glm5VerifyCkpt {
    pos: usize,
    latent_len: Vec<Option<usize>>,
    /// Per-row conv-ring clones, rows `0..t-1` except the last (doc above). PER-ROW walk
    /// only (`MEMRA_GLM5_VERIFY_BATCH=0`); the batched walk fills `kda_rows` instead.
    kda_conv_cols: Vec<Option<Vec<CudaSlice<f32>>>>,
    /// The recurrent state BEFORE row 0, one clone per KDA layer per round (doc above).
    /// BOTH walks fill this — it is the batched replay's scan base too.
    kda_ssm_snap: Vec<Option<CudaSlice<f32>>>,
    /// Stolen per-row scan inputs, rows `0..t-1` except the last (doc above). PER-ROW
    /// walk only.
    kda_scan_stash: Vec<Option<Vec<crate::kda::KdaScanInputs>>>,
    /// BATCHED walk (lane/glm5-verify-batch): one [`crate::kda::KdaRowsStash`] per KDA
    /// layer per round — ring snapshot + stolen raw conv rows + stolen batched scan
    /// inputs; rollback re-rolls the ring and replays the scan ONCE at T=keep.
    kda_rows: Vec<Option<crate::kda::KdaRowsStash>>,
    /// glm5 TP composition (lane/glm5-composition): per-rank rollback material of each
    /// SHARDED KDA layer's batched verify call — the rank-indexed twin of
    /// (`kda_ssm_snap`, `kda_rows`), restored through each rank's own engine. `None` on
    /// every unsharded layer.
    kda_tp: Vec<Option<crate::glm5_tp::Glm5TpKdaVerifyStash>>,
    /// Row count of the walk that filled this ckpt; rollback validates `keep` against it.
    rows: usize,
}

impl Glm5VerifyCkpt {
    /// GATE RECEIPT (wiring anchor, not a serving surface): how many KDA layers filled
    /// the BATCHED rows stash vs the PER-ROW column stash — the flag A/B gate asserts
    /// the arm it set actually ran (wiring-assertions-match-prose law: anchor on the
    /// invocation's artifact, never the log prose).
    pub fn kda_stash_kinds(&self) -> (usize, usize) {
        (
            self.kda_rows.iter().filter(|s| s.is_some()).count(),
            self.kda_conv_cols.iter().filter(|s| s.is_some()).count(),
        )
    }
}

/// Position buffers for one verify walk range (per stage engine under a split — the
/// per-stage pos_d law): `all` = the `[t]` vector the BATCHED per-layer mixer calls
/// consume; `rows` = the per-row single-position buffers of the per-row arm, built only
/// when that arm can run (flag off) — the batched arm never reads them.
struct Glm5VerifyPos {
    pos0: usize,
    t: usize,
    all: CudaSlice<i32>,
    rows: Vec<CudaSlice<i32>>,
}

impl Glm5VerifyPos {
    fn new(e: &Engine, pos0: usize, t: usize) -> Res<Self> {
        let v: Vec<i32> = (0..t as i32).map(|r| pos0 as i32 + r).collect();
        let all = e.htod_i32(&v)?;
        let rows = if glm5_verify_batch_on() && t > 1 {
            Vec::new()
        } else {
            (0..t)
                .map(|r| e.htod_i32(&[(pos0 + r) as i32]))
                .collect::<Result<_, _>>()?
        };
        Ok(Self { pos0, t, all, rows })
    }
}

impl HybridModel {
    /// THE T-PARALLEL VERIFY WALK: score `tokens` (row 0 = the last committed token, rows
    /// 1..t = the K drafted tokens) in ONE forward over the hc trunk at positions
    /// `cache.pos .. cache.pos + t`, in the batched-decode kernel classes (module doc).
    ///
    /// Returns `(logits [t, n_vocab] device, collapsed [t, n_embd] device, ckpt)`:
    /// `logits` row r is bit-identical to the plain `decode_step_hyper` logits after
    /// consuming `tokens[r]` at that position (the gate's bar); `collapsed` row r is the
    /// pre-output_norm hidden — the MTP `h_seed` for position `cache.pos + r`.
    ///
    /// State effects: every trunk MLA plane appends `t` rows; every trunk KDA state
    /// advances `t` steps (per-step columns stashed in the ckpt); `cache.pos` is NOT moved
    /// (rollback owns it). The MTP block's plane (il = n_trunk) is untouched.
    pub fn glm5_verify_rows(
        &self,
        e: &Engine,
        tokens: &[u32],
        cache: &mut Cache,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>, Glm5VerifyCkpt)> {
        let topology = *self
            .hyper
            .as_ref()
            .ok_or("glm5_verify_rows on a model with no HyperConnections topology")?;
        let t = tokens.len();
        let cap = Self::hyper_batch_cap();
        if t == 0 {
            return Err("glm5_verify_rows: empty verify row set".into());
        }
        if t > cap {
            return Err(format!(
                "glm5_verify_rows: t={t} > cap {cap} — at t >= PRIME_MIN_T (16) the MoE \
                 shared-expert trio crosses off the decode-exact class (the batched-decode \
                 gate's measured B=16 knee), so per-row bit-identity vs plain decode breaks. \
                 K <= cap-1 drafts per round"
            )
            .into());
        }
        let mut any_sharded = false;
        for (il, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Kda(la) => any_sharded |= la.tp.is_some(),
                Mixer::Mla(mla) => any_sharded |= mla.tp.is_some(),
                _ => {
                    return Err(format!(
                        "glm5_verify_rows: trunk layer {il} is not a KDA or MLA mixer — the \
                         rollback contract below is built and gated for glm5_next's two state \
                         classes only; a Full/Linear arm needs its own ckpt plane and gate"
                    )
                    .into());
                }
            }
        }
        // spec x TP composition (lane/glm5-composition): the per-row walk carries no TP
        // rollback arm — a sharded trunk demands the BATCHED walk at t > 1 (t = 1 rounds
        // ride the TP decode walk below; full accept is the only legal outcome there).
        if any_sharded && t > 1 && !glm5_verify_batch_on() {
            return Err(
                "glm5_verify_rows: the trunk is glm5-TP-SHARDED and MEMRA_GLM5_VERIFY_BATCH=0 — \
                 the per-row rollback seam carries no TP arm; the spec x TP composition \
                 requires the batched verify walk (unset MEMRA_GLM5_VERIFY_BATCH or run \
                 without the TP door)"
                    .into(),
            );
        }

        let n_embd = self.cfg.n_embd as usize;
        let pos0 = cache.pos;

        // Ckpt BEFORE any state moves.
        let mut ckpt = Glm5VerifyCkpt {
            pos: pos0,
            latent_len: cache
                .latent
                .iter()
                .take(self.layers.len())
                .map(|plane| plane.as_ref().map(|plane| plane.len))
                .collect(),
            kda_conv_cols: (0..self.layers.len()).map(|_| None).collect(),
            kda_ssm_snap: (0..self.layers.len()).map(|_| None).collect(),
            kda_scan_stash: (0..self.layers.len()).map(|_| None).collect(),
            kda_rows: (0..self.layers.len()).map(|_| None).collect(),
            kda_tp: (0..self.layers.len()).map(|_| None).collect(),
            rows: t,
        };

        // ppN door — the verify walk owns its stage split exactly as the batched decode
        // walk does (`decode_step_batch_hyper_ppn`, decode_batch.rs). Loud refusal on an
        // unqualified pipeline rewrite, never a single-engine walk over stage-sharded
        // weights.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline) {
                return Err("pipeline rewrite is not qualified for this ModelPlan".into());
            }
            return self.glm5_verify_rows_ppn(e, tokens, cache, ckpt, &topology, &fence);
        }

        let pos = Glm5VerifyPos::new(e, pos0, t)?;
        let embedded = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        let x = crate::hyper::expand(e, &topology, &embedded, t, n_embd)?;
        let x = self.glm5_verify_range(
            e,
            &topology,
            x,
            0,
            self.layers.len(),
            &pos,
            cache,
            &mut ckpt,
        )?;
        let (logits, collapsed) = self.glm5_verify_head(e, &topology, &x, t)?;
        Ok((logits, collapsed, ckpt))
    }

    /// One hc layer RANGE `[lo, hi)` of the verify walk — the body `glm5_verify_rows` ran
    /// inline before the ppN twin landed, extracted so the unsplit walk and every pipeline
    /// stage run the SAME code over their own range (the `hyper_range_decode` /
    /// `decode_batch_layers` precedent: bit-identity between the arms is then structural,
    /// not a coincidence of two maintained copies). At `lo=0, hi=n_layers` the launch
    /// sequence is identical to the pre-extraction walk.
    ///
    /// KDA ckpt columns are cloned THROUGH `e` — under a split that is the owning stage's
    /// engine, so each column lives on the device (and is ordered on the stream) that owns
    /// its layer's state; `glm5_verify_rollback` restores through the same per-stage seam.
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors the range-walk call contract its siblings share
    fn glm5_verify_range(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos: &Glm5VerifyPos,
        cache: &mut Cache,
        ckpt: &mut Glm5VerifyCkpt,
    ) -> Res<CudaSlice<f32>> {
        let t = pos.t;
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        // THE BATCHED MIXER ARM (lane/glm5-verify-batch, default ON): one t=K+1 call per
        // layer per class instead of the per-row loop — KDA batches projections/conv/
        // gates through the decode-exact classes with the recurrence sequential INSIDE
        // one scan launch; MLA runs the SAME cached core at t rows on the rows-exact
        // matmul classes (per-query causal selection + attention by construction).
        // `0` = the per-row walk below, byte-for-byte (the rollback seam). Engagement is
        // a receipt, announced once per process.
        let batch = glm5_verify_batch_on() && t > 1;
        {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                if batch {
                    eprintln!(
                        "[glm5-spec] verify walk BATCHED per layer: kda=one t-call (scan \
                         sequential in-kernel), mla=rows-exact t-call, head=rows-exact, \
                         moe=pairs rows-call where qualified \
                         (MEMRA_GLM5_VERIFY_BATCH default ON)"
                    );
                } else {
                    eprintln!("[glm5-spec] verify walk PER-ROW (MEMRA_GLM5_VERIFY_BATCH=0 or t=1)");
                }
            });
        }
        let trace_v = crate::spec_phase::spec_trace_level() >= 2;
        // Sub-phase clock (trace level 2 only): drain the walking stream so the elapsed
        // ns lands in the mixer-class bucket — shares, never walls.
        let vclock = |on: bool| -> Option<std::time::Instant> {
            on.then(|| {
                let _ = e.stream().synchronize();
                std::time::Instant::now()
            })
        };
        for il in lo..hi {
            let layer = &self.layers[il];
            let hyper = layer.hyper.as_ref().ok_or_else(|| {
                format!("layer {il} carries no hyper-connection weights under an hc plan")
            })?;

            let (y, mix) = crate::hyper::pre_exact(e, topology, &hyper.attn, &x, t, n_embd)?;
            let mut h = e.uninit(t * n_embd)?;
            e.rms_norm(&y, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            // The batched arm refuses per-layer only for an MLA layer WITHOUT the DSA
            // indexer: the absorbed t>1 attention arm has no per-row bit-identity claim
            // at this seam, so it stays on the per-row loop by name (glm5_next always
            // carries the indexer, so this is a foreign-geometry guard, not a live path).
            let layer_batched = batch
                && match &layer.mixer {
                    Mixer::Kda(_) => true,
                    Mixer::Mla(mla) => mla.index.is_some(),
                    Mixer::Full(_) | Mixer::Linear(_) => unreachable!("refused at walk entry"),
                };
            let mixed = if layer_batched {
                match &layer.mixer {
                    // spec x TP composition: sharded mixers ride the TP verify walks —
                    // per-rank batched rows calls, column-parallel-over-gather joins on
                    // the rows-exact classes, per-rank rollback stash into the ckpt.
                    Mixer::Kda(la) if la.tp.is_some() => {
                        let t0 = vclock(trace_v);
                        let mut scan_ns = 0u64;
                        let (out, stash) = crate::glm5_tp::kda_tp_verify_rows(
                            e,
                            la,
                            &h,
                            t,
                            eps,
                            cache,
                            il,
                            trace_v.then_some(&mut scan_ns),
                        )?;
                        ckpt.kda_tp[il] = Some(stash);
                        if let Some(t0) = t0 {
                            let _ = e.stream().synchronize();
                            use std::sync::atomic::Ordering;
                            crate::spec_phase::V_KDA_NS
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            crate::spec_phase::V_KDA_SCAN_NS.fetch_add(scan_ns, Ordering::Relaxed);
                        }
                        out
                    }
                    Mixer::Mla(mla) if mla.tp.is_some() => {
                        let t0 = vclock(trace_v);
                        let out =
                            self.mla_tp_attn_cached(e, mla, &h, &pos.all, t, il, cache, true)?;
                        if let Some(t0) = t0 {
                            let _ = e.stream().synchronize();
                            use std::sync::atomic::Ordering;
                            crate::spec_phase::V_MLA_NS
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        out
                    }
                    Mixer::Kda(la) => {
                        // Pre-round snapshot: ONE ssm clone per layer per round, BEFORE
                        // the batched call advances the resident state (ckpt doc; also
                        // the batched rollback's scan-replay base).
                        {
                            let rl = cache.recur[il]
                                .as_ref()
                                .ok_or("glm5 verify KDA layer has no recurrent state")?;
                            ckpt.kda_ssm_snap[il] = Some(e.clone_dtod(&rl.ssm_state)?);
                        }
                        let t0 = vclock(trace_v);
                        let mut scan_ns = 0u64;
                        let (out, stash) = crate::kda::kda_verify_rows_cached(
                            e,
                            la,
                            &h,
                            t,
                            eps,
                            cache,
                            il,
                            trace_v.then_some(&mut scan_ns),
                        )?;
                        ckpt.kda_rows[il] = Some(stash);
                        if let Some(t0) = t0 {
                            let _ = e.stream().synchronize();
                            use std::sync::atomic::Ordering;
                            crate::spec_phase::V_KDA_NS
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                            crate::spec_phase::V_KDA_SCAN_NS.fetch_add(scan_ns, Ordering::Relaxed);
                        }
                        out
                    }
                    Mixer::Mla(mla) => {
                        let t0 = vclock(trace_v);
                        let out =
                            self.mla_attn_cached_rows_exact(e, mla, &h, &pos.all, t, il, cache)?;
                        if let Some(t0) = t0 {
                            let _ = e.stream().synchronize();
                            use std::sync::atomic::Ordering;
                            crate::spec_phase::V_MLA_NS
                                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        out
                    }
                    Mixer::Full(_) | Mixer::Linear(_) => unreachable!("refused at walk entry"),
                }
            } else {
                // ---- PER-ROW mixer walk (the rollback seam; also t=1 rounds and the
                // no-indexer MLA guard): row r's state input is row r-1's state output
                // (KDA) / rows 0..pos0+r (MLA latent) — each row the SAME t=1 call its
                // plain decode step makes.
                // h_row is hoisted out of the row loop (loop-port 3): the mixer consumes
                // it in stream order before the next row's overwrite, so ONE buffer per
                // layer replaces t allocations (stream-ordered pool churn is the dsv4
                // lesson).
                // Depth attribution (lane/spec-route-depth-20260902): every row that walks
                // this arm is counted; the per-round log samples the counter.
                V_SEQ_ROWS.fetch_add(t as u64, std::sync::atomic::Ordering::Relaxed);
                let mut mixed = e.uninit(t * n_embd)?;
                let mut h_row = e.uninit(n_embd)?;
                #[allow(clippy::needless_range_loop)]
                // allow: r is the sequential row cursor (slices h, offsets pos); iterating pos buffers would hide the row-chaining contract
                for r in 0..t {
                    e.dtod_copy_view(&h.slice(r * n_embd..(r + 1) * n_embd), &mut h_row)?;
                    // The per-row position buffer: prebuilt when the per-row arm owns the
                    // walk; built on demand for the rare per-layer refusal under batch.
                    let pos_row: CudaSlice<i32>;
                    let pos_r = if let Some(p) = pos.rows.get(r) {
                        p
                    } else {
                        pos_row = e.htod_i32(&[(pos.pos0 + r) as i32])?;
                        &pos_row
                    };
                    let out_row = match &layer.mixer {
                        // spec x TP composition on the per-row arm. KDA shards reach
                        // here at t == 1 ONLY, asserted locally: the walk-entry guard is
                        // a FLAG check two frames up (MEMRA_GLM5_VERIFY_BATCH=0 at t>1
                        // refuses), and under the batched flag layer_batched is
                        // unconditionally true for KDA — but neither is a structural
                        // invariant of THIS arm (#80 review's latent-trap finding). A
                        // sharded NO-INDEXER MLA layer legally lands here at any t
                        // (append + truncate rollback covers every keep; foreign
                        // geometry, never a live glm5_next path).
                        Mixer::Kda(la) if la.tp.is_some() => {
                            if t > 1 {
                                return Err(format!(
                                    "glm5 verify per-row arm reached a sharded KDA \
                                     layer {il} at t={t}: no per-rank rollback stash \
                                     exists on this arm (walk-entry guard bypassed?)"
                                )
                                .into());
                            }
                            crate::glm5_tp::kda_tp_cached(
                                e,
                                la,
                                &h_row,
                                1,
                                eps,
                                cache,
                                il,
                                crate::kda::ConvArm::Decode,
                            )?
                        }
                        Mixer::Mla(mla) if mla.tp.is_some() => {
                            self.mla_tp_attn_cached(e, mla, &h_row, pos_r, 1, il, cache, false)?
                        }
                        Mixer::Kda(la) => {
                            // Pre-round snapshot: ONE ssm clone per layer per round taken
                            // before row 0 mutates the resident state (loop-port 3).
                            if r == 0 && t > 1 {
                                let rl = cache.recur[il]
                                    .as_ref()
                                    .ok_or("glm5 verify KDA layer has no recurrent state")?;
                                ckpt.kda_ssm_snap[il] = Some(e.clone_dtod(&rl.ssm_state)?);
                            }
                            if r + 1 < t {
                                // Steal the row's scan inputs for the replay stash (zero
                                // copies); clone only the small conv ring per row.
                                let (out, inputs) = crate::kda::kda_decode_cached_stash(
                                    e, la, &h_row, eps, cache, il,
                                )?;
                                let rl = cache.recur[il]
                                    .as_ref()
                                    .ok_or("glm5 verify KDA layer has no recurrent state")?;
                                ckpt.kda_conv_cols[il]
                                    .get_or_insert_with(Vec::new)
                                    .push(e.clone_dtod(&rl.conv_state)?);
                                ckpt.kda_scan_stash[il]
                                    .get_or_insert_with(Vec::new)
                                    .push(inputs);
                                out
                            } else {
                                crate::kda::kda_decode_cached(e, la, &h_row, eps, cache, il)?
                            }
                        }
                        Mixer::Mla(mla) => {
                            self.mla_attn_cached(e, mla, &h_row, pos_r, 1, il, cache)?
                        }
                        // Refused at entry; unreachable keeps the match total without a silent arm.
                        Mixer::Full(_) | Mixer::Linear(_) => unreachable!("refused at walk entry"),
                    };
                    e.copy_into(&mut mixed, r * n_embd, &out_row, n_embd)?;
                }
                mixed
            };
            x = crate::hyper::post(e, topology, &mixed, &x, &mix, t, n_embd)?;

            let (y, mix) = crate::hyper::pre_exact(e, topology, &hyper.mlp, &x, t, n_embd)?;
            let mut z = e.uninit(t * n_embd)?;
            e.rms_norm(
                &y,
                layer.post_attn_norm.float_data(),
                &mut z,
                n_embd,
                t,
                eps,
            )?;
            // FFN branch: `batch` arms the pairs-shaped batched MoE across the t rows
            // (lane/glm5-vrest — fail-closed inside to the byte-identical sequential
            // loop); the =0 arm keeps the pre-lane per-(token,expert) class. Clocked
            // into the vffn sub-bucket at trace level 2 (batched arm only, like vkda).
            let t0 = vclock(trace_v && batch);
            let ffn_out = self.hyper_ffn_branch_batch(e, layer, &z, t, il, batch)?;
            if let Some(t0) = t0 {
                let _ = e.stream().synchronize();
                use std::sync::atomic::Ordering;
                crate::spec_phase::V_FFN_NS
                    .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            x = crate::hyper::post(e, topology, &ffn_out, &x, &mix, t, n_embd)?;
            // glm5 DFlash2 feature tap (module doc, DRAFT SOURCE SEAM): the verify rows'
            // contracted completed-layer outputs are next round's drafter context.
            self.glm5_hc_tap(e, cache, topology, il, &x, t)?;
        }
        Ok(x)
    }

    /// Write one tapped layer's CONTRACTED completed output into the armed
    /// [`HcTapSink`] — the glm5 DFlash2 drafter's measured feature contract (stream-mean
    /// over the hyper streams, the probe's `hc_contract` capture definition). Staged
    /// through the WALKING engine `e` (the owning stage engine under a ppN split), so the
    /// sink is placement-invariant. One Option check when unarmed; nothing else pays.
    ///
    /// Two staging arms (loop-port 1):
    ///   * `device_stage` (the verify-round sink): ONE async D2D into the slot's device
    ///     buffer — the walk never blocks; the round drains all slots post-walk in its
    ///     single sync point (`glm5_tap_drain`). Kills the five in-walk DtoHs the 3way
    ///     window priced into the 31.6 ms fixed round cost (map row #17).
    ///   * host-staged (prime sinks): the pre-port behavior — per-chunk DtoH, amortized
    ///     over the prime's >= 256-row chunks.
    pub(crate) fn glm5_hc_tap(
        &self,
        e: &Engine,
        cache: &mut Cache,
        topology: &crate::hyper::HyperTopology,
        il: usize,
        x: &CudaSlice<f32>,
        t: usize,
    ) -> Res<()> {
        let Some(sink) = cache.hc_taps.as_mut() else {
            return Ok(());
        };
        let base = sink.base;
        self.glm5_hc_tap_into(e, sink, base, topology, il, x, t)
    }

    /// [`Self::glm5_hc_tap`] with the sink and its row base passed EXPLICITLY instead of read
    /// from the cache.
    ///
    /// The pipelined mHC prime needs this seam. Its two stage threads run different CHUNKS at
    /// the same moment, so a single `sink.base` field cannot describe both, and
    /// [`PrimeCacheStages`] hands each stage a cache shell with `hc_taps: None` — which is why
    /// arm 2 used to refuse outright when the DFlash2 drafter had armed a sink. Passing the
    /// base per call is what makes one shared sink correct for two concurrent walks.
    #[allow(clippy::too_many_arguments)] // allow: the list is the tap contract, base included
    pub(crate) fn glm5_hc_tap_into(
        &self,
        e: &Engine,
        sink: &mut HcTapSink,
        base_abs: usize,
        topology: &crate::hyper::HyperTopology,
        il: usize,
        x: &CudaSlice<f32>,
        t: usize,
    ) -> Res<()> {
        let Some(slot) = sink.layer_ids.iter().position(|&l| l == il) else {
            return Ok(());
        };
        let saved = sink.base;
        sink.base = base_abs;
        let out = self.glm5_hc_tap_slot(e, sink, slot, topology, x, t);
        sink.base = saved;
        out
    }

    fn glm5_hc_tap_slot(
        &self,
        e: &Engine,
        sink: &mut HcTapSink,
        slot: usize,
        topology: &crate::hyper::HyperTopology,
        x: &CudaSlice<f32>,
        t: usize,
    ) -> Res<()> {
        let h = sink.hidden;
        let n_taps = sink.layer_ids.len();
        // Sink-relative row of this walk's row 0 (doc on `HcTapSink::origin`): fresh-prompt
        // sinks have origin 0 and this is exactly the pre-field arithmetic; a suffix-prime
        // sink is anchored at the restored boundary. A base below the origin is a caller
        // bug (a walk over rows the sink does not cover) — refuse, never wrap.
        let base = sink.base.checked_sub(sink.origin).ok_or_else(|| {
            format!(
                "hc tap base {} below sink origin {} (walk outside the sink's window)",
                sink.base, sink.origin,
            )
        })?;
        debug_assert!(
            base + t <= sink.t,
            "hc tap window {base}+{t} exceeds sink {}",
            sink.t
        );
        let contracted = crate::hyper::contract_mean(e, topology, x, t, h)?;
        if sink.device_stage {
            // Lazy slot buffer on the WRITING engine (this layer always walks on one
            // stage, so the buffer's device is stable for the sink's lifetime). Every
            // walk row writes every tapped layer, so the buffer is fully covered by the
            // walk that armed the sink.
            if sink.dev[slot].is_none() {
                sink.dev[slot] = Some(e.uninit(sink.t * h)?);
            }
            let buf = sink.dev[slot].as_mut().expect("just filled");
            e.copy_into(buf, base * h, &contracted, t * h)?;
            return Ok(());
        }
        let t_dtoh = std::time::Instant::now();
        let host = e.dtoh(&contracted)?;
        for r in 0..t {
            let dst = (base + r) * n_taps * h + slot * h;
            sink.rows[dst..dst + h].copy_from_slice(&host[r * h..(r + 1) * h]);
        }
        sink.dtoh_ns += t_dtoh.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Drain a device-staged tap sink into its host `rows` — the round's ONE post-walk
    /// sync point for tap features (loop-port 1). Each slot reads back through its
    /// layer's OWNING engine (the stage engine under a live split, the caller's engine
    /// otherwise); the verify walk's terminal drain has already retired every stage's
    /// writes (stream program order: the slot copy precedes its stage's TX, and the
    /// TX-wait chain covers it transitively — the pp.rs multi-stream law).
    fn glm5_tap_drain(&self, e: &Engine, sink: &mut HcTapSink) -> Res<()> {
        if !sink.device_stage {
            return Ok(());
        }
        let h = sink.hidden;
        let n_taps = sink.layer_ids.len();
        let split = match crate::pp::pp_cuts(self.layers.len()) {
            Some(fence) if !crate::pp::pp2_streams_off() => {
                Some((crate::pp::PpNRt::get(e)?, fence))
            }
            _ => None,
        };
        for slot in 0..n_taps {
            let Some(buf) = sink.dev[slot].take() else {
                continue;
            };
            let il = sink.layer_ids[slot];
            let es = match split.as_ref() {
                Some((rt, fence)) => {
                    let stage = fence
                        .windows(2)
                        .position(|w| il >= w[0] && il < w[1])
                        .ok_or_else(|| format!("tap layer {il} outside every stage range"))?;
                    rt.engine(stage, e)
                }
                None => e,
            };
            let host = es.dtoh(&buf)?;
            for r in 0..sink.t {
                let dst = r * n_taps * h + slot * h;
                sink.rows[dst..dst + h].copy_from_slice(&host[r * h..(r + 1) * h]);
            }
        }
        Ok(())
    }

    /// Trunk exit of the verify walk, the batched head's decode-exact form
    /// (`hyper_batch_head_logits`), with the collapsed pre-output_norm rows kept — they are
    /// the h_seeds the MTP head re-seeds from (LANE.md §A). Under a split this runs on the
    /// LAST stage's engine, where the loader put `output_norm` + the lm head.
    fn glm5_verify_head(
        &self,
        e: &Engine,
        topology: &crate::hyper::HyperTopology,
        x: &CudaSlice<f32>,
        t: usize,
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>)> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let collapsed =
            crate::hyper::collapse(e, topology, self.hyper_head.as_ref(), x, t, n_embd)?;
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(
            &collapsed,
            self.output_norm.float_data(),
            &mut hn,
            n_embd,
            t,
            eps,
        )?;
        // Under the batched walk the lm head rides the rows-exact classes too (the bf16
        // tcols twin reads the 1.27 GB head ONCE per round instead of once per row);
        // per-row bits unchanged by contract, the tcols bit-gate holds it.
        let logits = if glm5_verify_batch_on() && t > 1 {
            e.matmul_rows_exact(&self.output, &hn, t)?
        } else {
            e.matmul_decode_exact(&self.output, &hn, t)?
        };
        Ok((logits, collapsed))
    }

    /// ppN twin of the verify walk (lane/glm5-ppn-verify, 2026-08-30), mirroring
    /// `decode_step_batch_hyper_ppn` (decode_batch.rs): the t=K+1 rows walk as N stage
    /// subgraphs — per-stage engine, per-stage pos_rows uploads, ONE `[t, streams, n_embd]`
    /// boundary payload per fence cut. Row chaining is per-LAYER through the one cache
    /// (row r+1 at layer il depends only on row r at layer il), so a straight layer-range
    /// split preserves it exactly; no row ever crosses a boundary individually. Head +
    /// collapsed rows land on the LAST stage's engine — where the loader put the lm head
    /// and where `pp::new_cache*` places the MTP plane the re-seed feeds.
    ///
    /// DRAIN CONTRACT: this walk returns DEVICE buffers with no terminal dtoh (unlike its
    /// decode twins, whose epilogue reads back on the last stage's stream), so it owns the
    /// settle — the per-stage arm synchronizes the LAST stage's stream before returning.
    /// The TX-wait chain transitively covers every earlier stage (pp.rs multi-stream law),
    /// so the logits, the collapsed rows AND the ckpt's per-stage KDA columns are all safe
    /// for consumption from the caller's streams after this returns.
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors its decode twin's stage-walk contract
    fn glm5_verify_rows_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        cache: &mut Cache,
        mut ckpt: Glm5VerifyCkpt,
        topology: &crate::hyper::HyperTopology,
        fence: &[usize],
    ) -> Res<(CudaSlice<f32>, CudaSlice<f32>, Glm5VerifyCkpt)> {
        let t = tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let pos0 = ckpt.pos;
        let payload = t * topology.streams * n_embd;
        // Position buffers through THIS stage's engine (the per-stage pos_d law:
        // allocated, consumed and freed on one stage's stream).
        let pos_on = |eng: &Engine| -> Res<Glm5VerifyPos> { Glm5VerifyPos::new(eng, pos0, t) };

        // Same-stream seam (MEMRA_PP_STREAMS=0): one engine, boundary copies between
        // ranges — the shape every hc ppN walk uses for this knob.
        if crate::pp::pp2_streams_off() {
            let pos = pos_on(e)?;
            let embedded = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            let mut x = crate::hyper::expand(e, topology, &embedded, t, n_embd)?;
            x =
                self.glm5_verify_range(e, topology, x, fence[0], fence[1], &pos, cache, &mut ckpt)?;
            for s in 1..fence.len() - 1 {
                let boundary_tx = e.clone_dtod(&x)?;
                let boundary_rx = e.clone_dtod(&boundary_tx)?;
                x = self.glm5_verify_range(
                    e,
                    topology,
                    boundary_rx,
                    fence[s],
                    fence[s + 1],
                    &pos,
                    cache,
                    &mut ckpt,
                )?;
            }
            let (logits, collapsed) = self.glm5_verify_head(e, topology, &x, t)?;
            return Ok((logits, collapsed, ckpt));
        }

        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(),
            n_st,
            "PpNRt stage count {} != fence stages {n_st}",
            rt.n_stages()
        );
        // #87 reverse publication (see decode_step_batch_ppn): order every stage stream
        // behind the caller before this body's first stage allocation.
        rt.fence_stages_behind(&e.stream())?;

        // ---- STAGE 0: embed + expand (no weights) + layers [0, fence[1]) + TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos = pos_on(e0)?;
            let embedded = e0.htod(&self.embd.try_gather(n_embd, tokens)?)?;
            let x = crate::hyper::expand(e0, topology, &embedded, t, n_embd)?;
            let x = self
                .glm5_verify_range(e0, topology, x, fence[0], fence[1], &pos, cache, &mut ckpt)?;
            rt.tx(0, &x, payload)?
        };

        // ---- MIDDLE STAGES: RX -> range -> TX ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos = pos_on(es)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = self.glm5_verify_range(
                es,
                topology,
                x,
                fence[s],
                fence[s + 1],
                &pos,
                cache,
                &mut ckpt,
            )?;
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + collapse/head + the drain (doc above) ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos = pos_on(el)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = self.glm5_verify_range(
            el,
            topology,
            x,
            fence[n_st - 1],
            fence[n_st],
            &pos,
            cache,
            &mut ckpt,
        )?;
        let (logits, collapsed) = self.glm5_verify_head(el, topology, &x, t)?;
        // el.stream() under the enter-guard IS the stage stream (memra_runtime ambient
        // override) — this drain settles the whole walk transitively.
        el.stream().synchronize()?;
        drop(_stl);
        // EXIT PUBLICATION (lane/glm5-accrace): the drain above settles the LAST stage, and
        // the TX-wait chain covers every earlier stage's work only UP TO its `ev_tx`. Each
        // earlier stage's stream still holds the stage-scope tail its locals enqueue when
        // they drop under the override (`pos`, the boundary residual, the per-layer
        // transients, this round's ckpt clones). The caller resumes and allocates for the
        // accept walk and the MTP re-seed, so it must be ordered behind ALL stages.
        self.glm5_publish_stages(e)?;
        Ok((logits, collapsed, ckpt))
    }

    /// The engine that owns the trunk exit (collapse + output_norm + lm head), the MTP
    /// block's weights AND its latent plane under a ppN split: the LAST stage's engine —
    /// `hybrid.rs` uploads the head there (`pp::layer_engine(e, n_trunk, n_trunk - 1)`)
    /// and `pp::new_cache*` maps trailing MTP/NextN planes to the last stage. Door shut or
    /// the same-stream seam: the caller's engine, unchanged (single-device callers pay
    /// nothing — `e` is returned by identity).
    fn glm5_head_engine<'e>(&self, e: &'e Engine) -> Res<&'e Engine> {
        match crate::pp::pp_cuts(self.layers.len()) {
            Some(fence) if !crate::pp::pp2_streams_off() => {
                let rt = crate::pp::PpNRt::get(e)?;
                Ok(rt.engine(fence.len() - 2, e))
            }
            _ => Ok(e),
        }
    }

    /// Roll the trunk back to exactly `keep` accepted verify rows (1 <= keep <= t; keep =
    /// j+1: the always-committed anchor row plus j accepted drafts).
    ///
    /// - MLA latent planes: `len = snapshot + keep` (truncate; rows are position-addressed
    ///   and append-only, so the kept rows ARE what a plain decode chain would have
    ///   written — the decode-exact contract), device `len_d` in lock-step, and
    ///   `truncate_index_pool_keys(pool)` clamps pool-key finality to what the shortened
    ///   `len` still justifies (the tail-ring residency tripwire fires on the next call if
    ///   this clamp is ever skipped).
    /// - KDA state: restore column keep-1 (state after the last kept row); full accept
    ///   (keep == t) keeps the resident state — the columns are clones OF it.
    /// - `cache.pos = snapshot + keep`.
    ///
    /// Under a live ppN split each stage's layers restore THROUGH that stage's engine ON
    /// its stream: the state planes and the ckpt columns live on the owning stage's device
    /// (per-stage `KvDev` allocation; per-stage clones in the walk), and enqueuing the
    /// restores on the same stage streams the walk writes on orders them relative to the
    /// walk without any extra fence.
    ///
    /// THE EXIT PUBLICATION IS NOT OPTIONAL (lane/glm5-accrace 2026-09-01). This body used
    /// to return with the restores merely ENQUEUED on the stage streams, on the reasoning
    /// that "the next walk's own entry fence covers the primary-stream seam". It does not:
    /// `fence_stages_behind` orders the STAGE streams behind the CALLER, and everything the
    /// round does after this point — the MTP plane reset, the h_seed rows, the next round's
    /// whole draft chain, the next SESSION's cache allocation and prime — runs on the
    /// CALLER's stream and ALLOCATES. cudarc's drops carry no read guard, so the pool could
    /// hand the caller a block whose stage-stream lifetime had not retired and the caller's
    /// writes landed under queued rollback work.
    ///
    /// MEASURED CONSEQUENCE, and why a "rollback ordering" bug showed up as a PRIME bug:
    /// with per-stage streams on one device the hc ppN prime over a fixed 24-token prompt
    /// returned three distinct logit fingerprints inside one process (a third of all primes
    /// non-canonical); downstream, one glm5 spec round lost an acceptance silently
    /// (14/42 -> 13/42) and the e2e tape diverged. Publishing here took non-canonical primes
    /// from 20/110 to 2/110 in an interleaved A/B, and the walk's own exit publication
    /// closed the remainder. Receipts:
    /// `research/glm53-flash-bringup-20260827/accrace-20260901/LANE.md`.
    pub fn glm5_verify_rollback(
        &self,
        e: &Engine,
        cache: &mut Cache,
        ckpt: &Glm5VerifyCkpt,
        keep: usize,
    ) -> Res<()> {
        if keep == 0 || keep > ckpt.rows {
            return Err(format!(
                "glm5_verify_rollback: keep={keep} outside 1..={} (the anchor row is always \
                 committed; keep = accepted drafts + 1)",
                ckpt.rows
            )
            .into());
        }
        match crate::pp::pp_cuts(self.layers.len()) {
            Some(fence) if !crate::pp::pp2_streams_off() => {
                let rt = crate::pp::PpNRt::get(e)?;
                for s in 0..fence.len() - 1 {
                    let _st = rt.enter(s);
                    let es = rt.engine(s, e);
                    for il in fence[s]..fence[s + 1] {
                        self.glm5_rollback_layer(es, cache, ckpt, keep, il)?;
                    }
                }
                // EXIT PUBLICATION (doc above): every stage stream, to the caller's.
                self.glm5_publish_stages(e)?;
            }
            _ => {
                for il in 0..self.layers.len() {
                    self.glm5_rollback_layer(e, cache, ckpt, keep, il)?;
                }
            }
        }
        cache.pos = ckpt.pos + keep;
        Ok(())
    }

    /// Restore ONE trunk layer to the ckpt's `keep`-row state (the per-plane contract in
    /// [`Self::glm5_verify_rollback`]'s doc). `e` is the layer's OWNING engine — the stage
    /// engine under a split, the caller's engine otherwise.
    fn glm5_rollback_layer(
        &self,
        e: &Engine,
        cache: &mut Cache,
        ckpt: &Glm5VerifyCkpt,
        keep: usize,
        il: usize,
    ) -> Res<()> {
        match &self.layers[il].mixer {
            Mixer::Mla(mla) => {
                if keep == ckpt.rows {
                    // Full accept: the walk already advanced len AND the len_d device
                    // mirror to saved + rows on the canonical plane and every replica
                    // (append-time stores), so the restore below would rewrite unchanged
                    // values — ~11 synchronizing pageable 4-byte copies per round on the
                    // HOT outcome (the KDA arms' early-out twin; #82 review).
                    return Ok(());
                }
                let saved = ckpt.latent_len[il].ok_or_else(|| {
                    format!("glm5_verify_rollback: MLA layer {il} missing from the ckpt")
                })?;
                let plane = cache.latent[il].as_mut().ok_or_else(|| {
                    format!("glm5_verify_rollback: MLA layer {il} has no latent plane")
                })?;
                plane.len = saved + keep;
                let len_i32 =
                    i32::try_from(plane.len).map_err(|_| "latent length exceeds i32 mirror")?;
                // Door H (`MEMRA_GLM5_HTOD_DIET`): async `i32_set_k` instead of the synchronizing
                // pageable 4-byte copy — 11 of these per round, and unconditional (unlike the
                // KDA arm, which short-circuits when `keep == rows`).
                e.i32_mirror_store(&mut plane.len_d, len_i32)?;
                if let Some(indexer) = mla.index.as_ref() {
                    plane.truncate_index_pool_keys(indexer.geom.pool);
                }
                // spec x TP composition: the PEER latent replicas append in lock-step with
                // the canonical plane (the TP walk's construction), so the same truncation
                // restores each of them — through its own rank's engine for the device
                // `len_d` mirror. Full accept skips the loop (lens already read
                // saved + rows — the KDA arm's early-out twin; each skipped store is a
                // synchronizing pageable copy per rank per layer on the HOT outcome).
                // Missing replicas after a verify walk are a wiring bug and refuse by
                // name, never a silent canonical-only restore (#80 review hardening).
                if let Some(tp) = mla.tp.as_ref() {
                    let replicas = cache.glm5_tp_latent_peer[il].as_mut().ok_or_else(|| {
                        format!(
                            "glm5_verify_rollback: sharded MLA layer {il} has no peer \
                             latent replicas (the TP verify walk hydrates them; a \
                             rollback without them would silently restore the canonical \
                             plane only)"
                        )
                    })?;
                    for (i, replica) in replicas.iter_mut().enumerate() {
                        replica.len = saved + keep;
                        tp.rt.peers[i].i32_mirror_store(&mut replica.len_d, len_i32)?;
                        if let Some(indexer) = mla.index.as_ref() {
                            replica.truncate_index_pool_keys(indexer.geom.pool);
                        }
                    }
                }
            }
            Mixer::Kda(la) if la.tp.is_some() => {
                if keep == ckpt.rows {
                    return Ok(()); // resident per-rank states ARE the post-keep states
                }
                let stash = ckpt.kda_tp[il].as_ref().ok_or_else(|| {
                    format!(
                        "glm5_verify_rollback: sharded KDA layer {il} has no per-rank stash \
                         (the batched TP verify walk fills it; the per-row arm is refused \
                         at walk entry)"
                    )
                })?;
                crate::glm5_tp::kda_tp_verify_rollback(e, la, stash, keep, cache, il)?;
            }
            Mixer::Kda(la) => {
                if keep == ckpt.rows {
                    return Ok(()); // resident state IS the state after the last kept row
                }
                // BATCHED-walk stash (lane/glm5-verify-batch): ring restore + re-roll,
                // then ONE scan replay at T=keep from the pre-round snapshot.
                if let Some(stash) = ckpt.kda_rows[il].as_ref() {
                    let snap = ckpt.kda_ssm_snap[il].as_ref().ok_or_else(|| {
                        format!("glm5_verify_rollback: KDA layer {il} has no ssm snapshot")
                    })?;
                    return crate::kda::kda_verify_rollback_rows(
                        e, la, snap, stash, keep, cache, il,
                    );
                }
                // Conv ring: restore the cloned column (unchanged — 288 KiB).
                let conv_cols = ckpt.kda_conv_cols[il].as_ref().ok_or_else(|| {
                    format!("glm5_verify_rollback: KDA layer {il} has no conv columns")
                })?;
                let conv = &conv_cols[keep - 1];
                {
                    let rl = cache.recur[il].as_mut().ok_or_else(|| {
                        format!("glm5_verify_rollback: KDA layer {il} has no recurrent state")
                    })?;
                    e.copy_into(&mut rl.conv_state, 0, conv, conv.len())?;
                }
                // Recurrent state: REPLAY rows 0..keep from the pre-round snapshot
                // (loop-port 3; ckpt doc) — each replay re-issues that row's original
                // t=1 scan over its stolen inputs, so the rebuilt state is byte-identical
                // to the per-row clone this retires.
                let snap = ckpt.kda_ssm_snap[il].as_ref().ok_or_else(|| {
                    format!("glm5_verify_rollback: KDA layer {il} has no ssm snapshot")
                })?;
                let stash = ckpt.kda_scan_stash[il].as_ref().ok_or_else(|| {
                    format!("glm5_verify_rollback: KDA layer {il} has no scan stash")
                })?;
                crate::kda::kda_scan_replay(e, la, snap, &stash[..keep], cache, il)?;
            }
            Mixer::Full(_) | Mixer::Linear(_) => {
                return Err(format!(
                    "glm5_verify_rollback: layer {il} mixer class was refused at walk \
                     entry and cannot appear in a ckpt"
                )
                .into());
            }
        }
        Ok(())
    }

    /// Reset the MTP draft plane (il = n_trunk) to `len` rows — the LANE.md rollback
    /// contract ("one row per step, rollback = plane len reset") plus the same pool-key
    /// clamp every len-shortening path owes the tail ring. The plane lives on the LAST
    /// stage under a split (`pp::new_cache*` maps trailing MTP planes there), so the
    /// device mirror writes through the head engine.
    fn glm5_mtp_plane_reset(&self, e: &Engine, cache: &mut Cache, len: usize) -> Res<()> {
        let e = self.glm5_head_engine(e)?;
        let mtp = self
            .mtp
            .as_ref()
            .ok_or("glm5_mtp_plane_reset with no MTP head loaded")?;
        let il = self
            .plan
            .mtp_blocks
            .first()
            .ok_or("ModelPlan declares no MTP block")?
            .layer
            .index as usize;
        let plane = cache
            .latent
            .get_mut(il)
            .and_then(|plane| plane.as_mut())
            .ok_or_else(|| format!("MTP block layer {il} has no latent cache plane"))?;
        if len > plane.len {
            return Err(format!(
                "glm5_mtp_plane_reset: target {len} is past the plane's {} rows — a reset \
                 only ever shortens",
                plane.len
            )
            .into());
        }
        plane.len = len;
        let len_i32 = i32::try_from(len).map_err(|_| "latent length exceeds i32 mirror")?;
        e.stream().memcpy_htod(&[len_i32], &mut plane.len_d)?;
        if let Mixer::Mla(mla) = &mtp.mixer
            && let Some(indexer) = mla.index.as_ref()
        {
            plane.truncate_index_pool_keys(indexer.geom.pool);
        }
        Ok(())
    }

    /// Single-shot glm5 speculative generation: draft (MTP head, K steps) -> verify (one
    /// t=K+1 walk) -> accept j (greedy longest matching prefix) -> rollback -> re-seed.
    /// Returns `(tokens, drafted, accepted)` — `generate_spec`'s contract. One-shot form:
    /// builds a [`Glm5SpecSession`] over a fresh cache and drives it to `max_new` — the
    /// SAME round machinery the serve worker bursts, so the tparallel gate's byte-identity
    /// pins cover the served path's rounds too.
    pub fn generate_spec_glm5(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
    ) -> Res<(Vec<u32>, usize, usize)> {
        self.generate_spec_glm5_gated(e, prompt, max_new, k, Glm5SpecKnobs::default())
    }

    /// `generate_spec_glm5` with GATE INSTRUMENTS (never a serving surface): a draft
    /// override for deterministic forced-accept / forced-reject rounds, and a
    /// rollback-disable arm that red-proves the end-to-end byte-identity gate.
    pub fn generate_spec_glm5_gated(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
        mut knobs: Glm5SpecKnobs<'_>,
    ) -> Res<(Vec<u32>, usize, usize)> {
        let cap = Self::hyper_batch_cap();
        if k == 0 || k + 1 > cap {
            return Err(format!(
                "generate_spec_glm5: k={k} outside 1..={} (verify rows = k+1 must stay \
                 inside the decode-exact knee, cap {cap})",
                cap - 1
            )
            .into());
        }
        if max_new == 0 {
            return Ok((Vec::new(), 0, 0));
        }
        let max_ctx = prompt.len() + max_new + k + 8;
        let mut sess = self.glm5_spec_session_new(e, prompt, max_ctx, None)?;
        let mut out: Vec<u32> = Vec::with_capacity(max_new + k);
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        while out.len() < max_new && !sess.finished() {
            let (burst, d, a) = self.glm5_spec_session_burst_gated(
                e,
                &mut sess,
                max_new - out.len(),
                k,
                &[],
                &mut knobs,
            )?;
            if burst.is_empty() {
                break; // ctx guard tripped with nothing new — never spin
            }
            out.extend(burst);
            drafted += d;
            accepted += a;
        }
        out.truncate(max_new);
        Ok((out, drafted, accepted))
    }

    /// BATCHED MTP-PLANE WARM (loop-port fold-in; doc at the call site in
    /// `glm5_spec_session_new`): fill the NextN block's latent plane with rows for pairs
    /// `(tokens_next[i], hiddens row i)`, i in `0..t`, in chunked t-parallel passes —
    /// ops 1-7 of `mtp_head_forward_mla_cached` batched over the chunk (embed gather,
    /// enorm/hnorm, the eh_proj concat via `place_rows_strided`, attn_norm), then ONE
    /// `mla_attn_cached` append per chunk (the prime-class t>1 arm the trunk's own MLA
    /// layers warm through; its attention output is discarded — the plane rows are the
    /// product). The MoE FFN, final norm and lm-head of the per-token chain are never
    /// run: they fed nothing but the (discarded) draft logits of prompt positions.
    fn glm5_mtp_plane_fill(
        &self,
        e: &Engine,
        tokens_next: &[u32],
        hiddens: &CudaSlice<f32>,
        t: usize,
        cache: &mut Cache,
    ) -> Res<()> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or("glm5_mtp_plane_fill with no MTP head loaded")?;
        let il = self
            .plan
            .mtp_blocks
            .first()
            .ok_or("ModelPlan declares no MTP block")?
            .layer
            .index as usize;
        let Mixer::Mla(mla) = &mtp.mixer else {
            return Err("glm5_mtp_plane_fill serves MLA-mixer MTP blocks only".into());
        };
        if tokens_next.len() < t {
            return Err(format!(
                "glm5_mtp_plane_fill: {t} rows requested over {} successor tokens",
                tokens_next.len()
            )
            .into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        // Chunk bound: the trunk prime's workspace discipline — bounds the t>1 attention
        // workspace and the transient buffers below without changing the append semantics
        // (`mla_attn_cached` appends at the plane's running length either way).
        const CHUNK: usize = 512;
        let mut done = 0usize;
        while done < t {
            let tc = (t - done).min(CHUNK);
            let e_emb = e.htod(
                &self
                    .embd
                    .try_gather(n_embd, &tokens_next[done..done + tc])?,
            )?;
            let mut e_norm = e.uninit(tc * n_embd)?;
            e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, tc, eps)?;
            // hnorm over the chunk's hidden rows (one contiguous view copy — rms_norm
            // takes an owned-slice operand).
            let hv = e.view(hiddens, (done + tc) * n_embd);
            let mut h_rows = e.uninit(tc * n_embd)?;
            e.copy_view_into(
                &mut h_rows,
                0,
                &hv.slice(done * n_embd..(done + tc) * n_embd),
                tc * n_embd,
            )?;
            let mut h_norm = e.uninit(tc * n_embd)?;
            e.rms_norm(
                &h_rows,
                mtp.hnorm.float_data(),
                &mut h_norm,
                n_embd,
                tc,
                eps,
            )?;
            // concat rows [tc, 2*n_embd] = [enorm ; hnorm] — two strided placements.
            let mut concat = e.uninit(tc * 2 * n_embd)?;
            e.place_rows_strided(&e_norm, &mut concat, n_embd, tc, 2 * n_embd, 0)?;
            e.place_rows_strided(&h_norm, &mut concat, n_embd, tc, 2 * n_embd, n_embd)?;
            let inp_sa = e.matmul(&mtp.eh_proj, &concat, tc)?;
            let mut a_norm = e.uninit(tc * n_embd)?;
            e.rms_norm(
                &inp_sa,
                mtp.attn_norm.float_data(),
                &mut a_norm,
                n_embd,
                tc,
                eps,
            )?;
            let pos: Vec<i32> = (done as i32..(done + tc) as i32).collect();
            let pos_d = e.htod_i32(&pos)?;
            let _ = self.mla_attn_cached(e, mla, &a_norm, &pos_d, tc, il, cache)?;
            done += tc;
        }
        Ok(())
    }

    /// Row `row` of a `[rows, n_embd]` device stack, copied into its own `[n_embd]` buffer
    /// (the MTP `h_seed` handoff shape).
    fn glm5_seed_row(
        &self,
        e: &Engine,
        src: &CudaSlice<f32>,
        rows: usize,
        row: usize,
    ) -> Res<CudaSlice<f32>> {
        let n_embd = self.cfg.n_embd as usize;
        let stack = e.view(src, rows * n_embd);
        let view = stack.slice(row * n_embd..(row + 1) * n_embd);
        let mut seed = e.uninit(n_embd)?;
        e.copy_view_into(&mut seed, 0, &view, n_embd)?;
        Ok(seed)
    }

    /// SERVED-SESSION ENTRY (lane/glm5-spec-routing): prime the prompt, warm the MTP plane,
    /// draw the boundary token, and hand back a [`Glm5SpecSession`] the worker bursts.
    ///
    /// `sampling`: `None` / `temp <= 0` = the greedy byte-contract route (the instrument);
    /// `Some` with `temp > 0` = the sampled route — the boundary token, the draft chain and
    /// the accept walk all draw through the session's own Philox counters (`sctr` device
    /// events, `uctr` host accept-test uniforms via `spec::host_u01`, tag 0xFFFF_FFFE), so a
    /// session's randomness never repeats across bursts (the session-continuity law).
    /// PENALIZED sampled requests are refused loudly — the glm5 accept walk has no penalty
    /// arm yet; worker admission keeps them on the plain path (same split as dspark's
    /// penalized-greedy exclusion).
    pub fn glm5_spec_session_new(
        &self,
        e: &Engine,
        prompt: &[u32],
        ctx_cap: usize,
        sampling: Option<SpecSampling>,
    ) -> Res<Glm5SpecSession> {
        if self.hyper.is_none() {
            return Err("generate_spec_glm5 requires a HyperConnections trunk".into());
        }
        // Two parallel/spec programs on one model never silently coexist unless the
        // composition is EXPLICITLY armed: the spec x TP verify/rollback wiring
        // (lane/glm5-composition) exists and is rig-gated, but it has zero real-artifact
        // receipts, so sessions on a SHARDED model stay co-refused unless
        // MEMRA_GLM5_SPEC_TP=1 lifts the refusal (default OFF by design — the FLAGS row).
        // The predicate is the MODEL's own sharding (the same per-layer truth the verify
        // walk keys on), never the MEMRA_GLM5_TP env: sharding is a load-time property,
        // and an env read here is bypassable after load (set/load/unset) and spuriously
        // refuses an UNSHARDED model in a process that still carries the env (the #80
        // review's confirmed finding).
        let tp_sharded = self.layers.iter().any(|l| match &l.mixer {
            Mixer::Kda(la) => la.tp.is_some(),
            Mixer::Mla(mla) => mla.tp.is_some(),
            _ => false,
        });
        if tp_sharded {
            if !glm5_spec_tp_on() {
                return Err(
                    "glm5 spec is co-refused on a MEMRA_GLM5_TP-sharded model: set \
                     MEMRA_GLM5_SPEC_TP=1 to run the gated spec x TP composition \
                     (default OFF — zero real-artifact receipts; every other admission \
                     law still holds)"
                        .into(),
                );
            }
            if !glm5_verify_batch_on() {
                return Err("MEMRA_GLM5_SPEC_TP=1 requires the BATCHED verify walk \
                     (MEMRA_GLM5_VERIFY_BATCH must not be 0): the per-row rollback seam \
                     carries no TP arm"
                    .into());
            }
            // The ARMED announce prints AFTER the last admission law below — a session
            // refused later (draft source, ppN qualification, penalties, ctx) must never
            // log the engagement receipt (the fleet's prints-ARMED-then-serves-plain
            // trap class; rig-gates/03 caught SF3 logging it on a refused session).
        }
        // DRAFT-SOURCE SELECTION — through the GENERAL law
        // ([`crate::spec::resolve_draft_source_kind`], lane/glm5-extract2): DFlash2 when the
        // drafter is loaded (MEMRA_GLM5_DFLASH — it wins over a co-loaded MTP head, the boot
        // receipt states the selection), native MTP otherwise; neither = the loud refusal
        // below, whose bytes name the glm5 flags because the FAMILY owns the how-to-arm text.
        // The native MTP head is NOT required for the DFlash2 source (the q38 pattern).
        let dflash_src = self.glm5_dflash.as_ref();
        let source_kind = crate::spec::resolve_draft_source_kind(
            self.plan.draft_source,
            self.mtp.is_some(),
            dflash_src.is_some(),
        )
        .map_err(|why| {
            // The general law says WHY there is no usable source; the family owns the
            // how-to-arm text. Composed so the sentence is true on BOTH of the law's refusal
            // branches (nothing loaded / a head loaded under a plan that does not claim it),
            // rather than asserting "requires a draft source" at an operator whom the bracket
            // then tells a head IS loaded.
            format!(
                "generate_spec_glm5 cannot select a draft source ({why}). Arm one: the \
                 embedded MTP head (MEMRA_GLM5_MTP=1; a full MoE layer, unloaded by default) \
                 or the DFlash2 drafter (MEMRA_GLM5_DFLASH=<dir-or-hf-spec>)"
            )
        })?;
        if prompt.len() < 2 {
            return Err(
                "generate_spec_glm5 needs a prompt of >= 2 tokens (the MTP plane warms on \
                 (token[i+1], hidden[i]) pairs)"
                    .into(),
            );
        }
        if dflash_src.is_none() && crate::spec::spec_hpost() {
            // MTP-carrier-specific refusal: the DFlash2 source consumes tapped trunk
            // features, not the h_seed carrier, so the flag has nothing to flip there.
            return Err(
                "generate_spec_glm5 has no MEMRA_SPEC_HPOST arm: the flag flips the MTP \
                 carrier to the post-norm hidden, but this loop seeds every committed pair \
                 from the trunk's PRE-output_norm collapsed rows (LANE.md §A). Mixing the \
                 two silently degrades drafts; the HPOST twin needs its own gate before it \
                 may run"
                    .into(),
            );
        }
        // ppN split (lane/glm5-ppn-verify): the verify walk, the rollback and the MTP
        // chain all run under the split now — but an UNQUALIFIED pipeline rewrite still
        // refuses loudly at the session seam, before any cache is allocated over
        // stage-sharded weights (worker admission additionally bounds the stage count to
        // the gated set; see glm5_spec_capable).
        if crate::pp::pp_cuts(self.layers.len()).is_some()
            && !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline)
        {
            return Err("pipeline rewrite is not qualified for this ModelPlan".into());
        }
        let sampling = sampling.filter(|sp| sp.temp > 0.0);
        if let Some(sp) = sampling.as_ref()
            && sp.pen_on()
        {
            return Err(
                "glm5 spec has no penalty arm yet: penalized sampled requests serve on the \
                 plain path (worker admission owns the exclusion; silently dropping the \
                 request's penalties is the failure class this refusal prevents)"
                    .into(),
            );
        }
        // Room for the prompt, the anchor row and at least one verify round.
        if prompt.len() + 4 > ctx_cap {
            return Err(format!(
                "glm5 spec session needs ctx for prompt {} + anchor + one verify round, \
                 cap {ctx_cap}",
                prompt.len()
            )
            .into());
        }
        let n_vocab = self.output.out_features();
        // FR-SPEC TRIM (module doc): a loaded `MEMRA_FRSPEC_TRIM` artifact means the draft
        // head projects over gathered top-N rows and every draft pick is a RANK id that
        // must remap through d2t to the true vocabulary BEFORE it is chained or verified.
        // The verify walk stays full-vocab regardless — the invariant under gate.
        if let Some(map) = self.glm5_d2t() {
            if map.iter().any(|&t| t as usize >= n_vocab) {
                return Err(format!(
                    "glm5 FR-Spec d2t carries a token id >= n_vocab {n_vocab} — the ranks \
                     artifact was minted for a different vocabulary"
                )
                .into());
            }
            // Engagement receipt (the dspark trim receipt's shape): the server-log line
            // the trim arm's per-session engagement is verified by.
            eprintln!(
                "[glm5-spec] draft head TRIMMED to {} rows (FR-Spec d2t engaged)",
                map.len(),
            );
        }
        // Confidence-gate engagement receipt (loop-port 2; the deploy-gate greps this —
        // never-serve-greedy law's receipt discipline): armed iff MEMRA_SPEC_PMIN > 0.
        if glm5_pmin() > 0.0 {
            eprintln!(
                "[glm5-spec] draft confidence gate armed: PMIN={:.3} PMIN0={} (native \
                 chain p-of-pick; DFlash2 selector-q tau-slot truncation)",
                glm5_pmin(),
                glm5_pmin0() as u8,
            );
        }
        // The MTP plane index is a NATIVE-arm need; the DFlash2 source never touches the
        // plane (it still allocates below — plan-structural, the named cost in the module
        // doc — but nothing reads or resets it).
        let mtp_il = match dflash_src {
            Some(_) => None,
            None => Some(
                self.plan
                    .mtp_blocks
                    .first()
                    .ok_or("ModelPlan declares no MTP block")?
                    .layer
                    .index as usize,
            ),
        };
        // FIRST-TOKEN PROFILE (lane/b200-spec-ttft-20260902, `MEMRA_SPEC_PROF=1`): the
        // head engine resolves before any device work (a pure placement lookup), so the
        // clock can bound every creation phase on both the primary and head streams.
        let eh = self.glm5_head_engine(e)?;
        let mut prof = spec_prof_on().then(SpecFirstTokenProf::default);
        let mut pclk = prof.as_ref().map(|_| ProfClock::start(e, eh));
        if let Some(pf) = prof.as_mut() {
            pf.free_mb_before = self.glm5_free_mb(e);
        }
        // Stage-owned allocation under a split (each layer's planes on its stage's device,
        // trailing MTP plane on the last stage); door shut = plain `Cache::new_planned`.
        let mut cache = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, ctx_cap)?;
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.cache_alloc_ms = ck.lap(e, eh);
        }

        // ---- prime, boundary token, draft-source warm over the prompt ----
        // `prime_cache` routes to its own ppN twin under the split; `hiddens` is owned by
        // the LAST stage's engine (its published contract) — exactly where the MTP chain
        // below runs, so the warm consumes it with no device bounce. DFlash2 source: the
        // prime walk fills the armed HcTapSink with every prompt row's contracted tap
        // features (the drafter's context; round 1 ingests them into its own KV).
        let plen = prompt.len();
        let n_embd = self.cfg.n_embd as usize;
        let tap_layers = match dflash_src {
            Some(dr) => Some(glm5_dflash_tap_layers(&dr.draft, self.layers.len())?),
            None => None,
        };
        // DRAFTER PRIME ARM (lane/spec-route-depth-20260902): the chunked arm primes the
        // trunk per engine chunk and ingests each chunk's taps into the drafter KV right
        // after that chunk's walk (`glm5_draft_prime_chunked`); the eager arm (the pre-lane
        // literal) primes once over a whole-prompt host sink and leaves the ingest to
        // round 1. `hidden_rows` is the row count of the returned `hiddens` stack (the whole
        // prompt on the eager arm, the LAST chunk on the chunked arm) — the boundary
        // capture indexes its last row through it.
        let mut v2_kv: Option<DflashKv> = None;
        let (logits0, hiddens, hidden_rows) = match (dflash_src, tap_layers.as_ref()) {
            (Some(dr), Some(taps)) if glm5_draft_taps_device_on() => {
                // DEVICE-RESIDENT ARM (doc on `glm5_draft_taps_device_on`): ONE whole-prompt
                // prime; the ingest happens inside it at every range boundary.
                let kv = DflashKv::new(eh, &dr.draft.cfg, ctx_cap)?;
                if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
                    pf.draft_alloc_ms = ck.lap(e, eh);
                    pf.draft_kv_mb = dflash_kv_bytes(&dr.draft.cfg, ctx_cap) as f64 / 1e6;
                }
                let ring = crate::hybrid_forward::hyper_prime_call_rows(
                    plen,
                    self.layers.len(),
                    self.gdn_prime_grid_on(),
                );
                let mut sink = HcTapSink::new_device_staged_at(taps.clone(), n_embd, ring, 0);
                sink.ingest_state = Some(Box::new(Glm5DraftPrimeInflight {
                    kv,
                    taps: taps.clone(),
                    n_embd,
                    ring,
                    stage: (0..taps.len()).map(|_| None).collect(),
                    rows_dev: None,
                    prof_on: prof.is_some(),
                    copy_ms: 0.0,
                    feat_ms: 0.0,
                    kv_ms: 0.0,
                    chunks: 0,
                }));
                cache.hc_taps = Some(sink);
                let (l, _seed, h) = self.prime_cache(e, prompt, &mut cache, 0)?;
                let walk_ms = pclk.as_mut().map(|ck| ck.lap(e, eh));
                let mut sink = cache
                    .hc_taps
                    .take()
                    .ok_or("device-resident drafter prime: tap sink vanished")?;
                let st = sink
                    .ingest_state
                    .take()
                    .ok_or("device-resident drafter prime: ingest state vanished")?
                    .downcast::<Glm5DraftPrimeInflight>()
                    .map_err(|_| "device-resident drafter prime: ingest state of the wrong type")?;
                let st = *st;
                if st.kv.len != plen {
                    return Err(format!(
                        "device-resident drafter prime covered {} of {plen} prompt rows \
                         (the prime's range loop must hand every range to the ingest)",
                        st.kv.len
                    )
                    .into());
                }
                if let (Some(pf), Some(walk)) = (prof.as_mut(), walk_ms) {
                    // The ingest ran INSIDE the prime walk: split it back out so `prime`
                    // stays the trunk's share and `draft_prime` the drafter's.
                    let ingest = st.copy_ms + st.feat_ms + st.kv_ms;
                    pf.prime_ms = (walk - ingest).max(0.0);
                    pf.draft_prime_ms = ingest;
                    pf.draft_prime_h2d_ms = st.copy_ms;
                    pf.draft_prime_feat_ms = st.feat_ms;
                    pf.draft_prime_kv_ms = st.kv_ms;
                    pf.draft_prime_rows = plen;
                    pf.draft_prime_chunks = st.chunks;
                    pf.draft_prime_arm = "device";
                }
                v2_kv = Some(st.kv);
                (l, h, plen)
            }
            (Some(dr), Some(taps)) if glm5_draft_prime_v2_on() => {
                let mut kv = DflashKv::new(eh, &dr.draft.cfg, ctx_cap)?;
                if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
                    pf.draft_alloc_ms = ck.lap(e, eh);
                    pf.draft_kv_mb = dflash_kv_bytes(&dr.draft.cfg, ctx_cap) as f64 / 1e6;
                }
                let out = self.glm5_draft_prime_chunked(
                    e,
                    eh,
                    &dr.draft,
                    prompt,
                    &mut cache,
                    &mut kv,
                    taps,
                    prof.as_mut(),
                    pclk.as_mut(),
                )?;
                v2_kv = Some(kv);
                out
            }
            _ => {
                if let Some(taps) = tap_layers.as_ref() {
                    let t_sink = std::time::Instant::now();
                    cache.hc_taps = Some(HcTapSink::new(taps.clone(), n_embd, plen));
                    if let Some(pf) = prof.as_mut() {
                        pf.sink_alloc_ms = t_sink.elapsed().as_secs_f64() * 1e3;
                    }
                }
                let (l, _seed, h) = self.prime_cache(e, prompt, &mut cache, 0)?;
                if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
                    pf.prime_ms = ck.lap(e, eh);
                    pf.prime_tap_dtoh_ms = cache
                        .hc_taps
                        .as_ref()
                        .map(|sk| sk.dtoh_ns as f64 / 1e6)
                        .unwrap_or(0.0);
                }
                (l, h, plen)
            }
        };
        // Prompt-boundary capture (lane/glm5-prefix-latent2): taken NOW — after the prime
        // filled every plane to the boundary, before the anchor/draft machinery below and
        // before any burst mutates the conv/ssm state or laps the tail ring. DFlash2-only:
        // the native arm's plane fill moves the MTP latent layer past the boundary before a
        // capture could be taken, and restore refuses the native source anyway. A refusal
        // drops the capture loudly and the session serves regardless.
        let prefix_capture = if glm5_spec_prefix_on() && dflash_src.is_some() {
            self.glm5_prefix_boundary_capture(e, eh, &cache, &logits0, &hiddens, plen, hidden_rows)
        } else {
            None
        };
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.capture_ms = ck.lap(e, eh);
        }
        let mut sctr = 0u32;
        let anchor = match sampling.as_ref() {
            Some(sp) => {
                crate::spec::sample_boundary_token(eh, &logits0, sp, &[], &mut sctr, "glm5-prime")?
            }
            None => argmax(&logits0) as u32,
        };
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.anchor_ms = ck.lap(e, eh);
        }

        // Keyed on the KIND the general law returned, not on a second local re-derivation:
        // a seam whose answer is recomputed by its consumer is decoration. (`tap_layers` is
        // `Some` exactly when `dflash_src` is, and the law returns `Dflash2` exactly then, so
        // this is the same program the pre-seam code ran — the `_` arm's refusal is the
        // never-taken proof of that rather than a silent fallback.)
        let (mut draft, pending) = match (source_kind, dflash_src, tap_layers) {
            (crate::spec::DraftSourceKind::Dflash2, Some(dr), Some(taps)) => {
                if let Some(kv) = v2_kv.take() {
                    // Chunked arm: the KV already holds every prompt row (kv.len == plen);
                    // round 1 finds nothing pending and walks straight to its block forward.
                    debug_assert_eq!(kv.len, plen, "chunked drafter prime must cover the prompt");
                    (
                        Glm5DraftState::Dflash2 {
                            kv,
                            pending: Vec::new(),
                            taps,
                        },
                        Vec::new(),
                    )
                } else {
                    let sink = cache
                        .hc_taps
                        .take()
                        .ok_or("glm5 dflash prime tap sink vanished")?;
                    // Drafter ctx KV on the HEAD engine (where the drafter weights loaded and
                    // every round's chain runs); prompt feature rows ride `pending` so round 1
                    // ingests them through the one chunked path.
                    let kv = DflashKv::new(eh, &dr.draft.cfg, ctx_cap)?;
                    if let Some(pf) = prof.as_mut() {
                        pf.draft_kv_mb = dflash_kv_bytes(&dr.draft.cfg, ctx_cap) as f64 / 1e6;
                    }
                    (
                        Glm5DraftState::Dflash2 {
                            kv,
                            pending: sink.rows,
                            taps,
                        },
                        Vec::new(),
                    )
                }
            }
            (crate::spec::DraftSourceKind::Dflash2, _, _) => {
                return Err(
                    "the draft-source law selected DFlash2 but this session resolved no tap \
                     layers — a load-path bug, refused instead of silently drafting from the \
                     MTP plane (a VANISHED tap sink is a different failure, caught by name in \
                     the Dflash2 arm itself)"
                        .into(),
                );
            }
            (crate::spec::DraftSourceKind::NativeMtp, _, _) => {
                // BATCHED PLANE WARM (loop-port fold-in — the map's #4, the spec.rs
                // `mtp_kv_fill_all` pattern re-aimed at the MLA plane): pairs
                // (prompt[i+1], h_i) at plane pos i, i in 0..P-1, filled in CHUNKED
                // t-parallel passes instead of P-1 sequential full-block forwards. The
                // sequential warm ran ~400 tok/s — the measured +2.5 s TTFT per 1k
                // prompt tokens, spec-battery flip condition 1 by name. MTP rows are
                // INDEPENDENT given the trunk hiddens (no row-to-row recurrence — the
                // plane is the only carrier), so the fill is exact in structure; the
                // t>1 attention takes the prime-class program, which can only move
                // DRAFTS, never output (verify arbitrates; the byte-identity batteries
                // stay the proof).
                self.glm5_mtp_plane_fill(eh, &prompt[1..], &hiddens, plen - 1, &mut cache)?;
                // pending = committed (token, h_seed) pairs not yet fed to the MTP plane.
                // The LAST pair's logits are the next round's first draft — the re-warm
                // doubles as draft 1.
                let pending = vec![(anchor, self.glm5_seed_row(eh, &hiddens, plen, plen - 1)?)];
                (Glm5DraftState::NativeMtp, pending)
            }
        };
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.draft_alloc_ms += ck.lap(e, eh);
        }
        // EAGER-ARM INGEST AT CREATION (doc on `glm5_draft_prime_lazy_on`): the prompt's tap
        // rows go into the drafter KV NOW, before the session (and its anchor) is handed to
        // the worker, unless the lazy seam asks for the round-1 placement. The chunked arm
        // arrives here with nothing pending.
        if !glm5_draft_prime_lazy_on()
            && let (Glm5DraftState::Dflash2 { kv, pending, taps }, Some(dr)) =
                (&mut draft, dflash_src)
            && !pending.is_empty()
        {
            let rows = std::mem::take(pending);
            let stats = self.glm5_dflash_ingest_rows(
                eh,
                &dr.draft,
                kv,
                &rows,
                taps.len() * n_embd,
                pclk.as_mut(),
            )?;
            if let Some(pf) = prof.as_mut() {
                stats.write(pf, "eager");
            }
        }
        if let Some(pf) = prof.as_mut() {
            pf.free_mb_after = self.glm5_free_mb(e);
        }
        // Composition engagement receipt — printed immediately before the session is
        // RETURNED (after every admission law, the d2t vocabulary check, the cache
        // allocation and the prompt prime), so a grep for this line counts sessions that
        // actually opened; a refusal or a failure anywhere above never logs it (the #82
        // review moved it here after finding four fallible steps below its first home).
        if tp_sharded {
            eprintln!(
                "[glm5-spec] spec x TP composition ARMED (MEMRA_GLM5_SPEC_TP=1): verify \
                 rows ride the TP shards; rollback restores per-rank planes \
                 performance_claim=false"
            );
        }
        Ok(Glm5SpecSession {
            cache,
            committed: prompt.to_vec(),
            anchor,
            anchor_emitted: false,
            pending,
            draft,
            sampling,
            sctr,
            uctr: 0,
            rounds: 0,
            done: false,
            max_ctx: ctx_cap,
            mtp_il,
            prefix_capture,
            prof_rounds: prof.as_ref().map(|_| SpecRoundsLog::default()),
            prof,
        })
    }

    /// Range-begin hook of the device-resident drafter prime (called by the prime's range
    /// loop, `hybrid_forward.rs`): anchor the chunk ring at the range's first row so the
    /// walk's `base - origin` lands in `[0, ring)`. No-op on every other sink.
    pub(crate) fn glm5_taps_range_begin(&self, cache: &mut Cache, start: usize) {
        if let Some(sink) = cache.hc_taps.as_mut()
            && sink.ingest_state.is_some()
        {
            sink.origin = start;
        }
    }

    /// Range-done hook of the device-resident drafter prime: the range's tap planes are
    /// complete on their writing devices (the range call returned host logits, so the last
    /// stage drained, and every earlier stage's rows were consumed by it), so ingest them
    /// now. No-op on every other sink; a state of another type is put back untouched.
    pub(crate) fn glm5_taps_range_done(
        &self,
        e: &Engine,
        cache: &mut Cache,
        start: usize,
        end: usize,
    ) -> Res<()> {
        let Some(sink) = cache.hc_taps.as_mut() else {
            return Ok(());
        };
        let Some(state) = sink.ingest_state.take() else {
            return Ok(());
        };
        let mut st = match state.downcast::<Glm5DraftPrimeInflight>() {
            Ok(st) => st,
            Err(other) => {
                sink.ingest_state = Some(other);
                return Ok(());
            }
        };
        let res = self.glm5_taps_ingest_range(e, sink, &mut st, start, end);
        sink.ingest_state = Some(st);
        res
    }

    /// One range of the device-resident drafter prime: per tap slot, the plane on its
    /// writing device is interleaved into the head-device fc input with a 2D copy (after a
    /// peer copy into head-device staging when the slot's stage is another device; the
    /// source stream is drained before the plane is read), then `ctx_features` +
    /// `ingest_ctx` at the range width append the rows to the drafter KV.
    fn glm5_taps_ingest_range(
        &self,
        e: &Engine,
        sink: &mut HcTapSink,
        st: &mut Glm5DraftPrimeInflight,
        start: usize,
        end: usize,
    ) -> Res<()> {
        let t = end - start;
        if t == 0 {
            return Ok(());
        }
        if t > st.ring {
            return Err(format!(
                "device-resident tap ingest: range {start}..{end} ({t} rows) exceeds the \
                 sink ring of {} rows",
                st.ring
            )
            .into());
        }
        if st.kv.len != start {
            return Err(format!(
                "device-resident tap ingest: drafter KV holds {} rows but the range starts \
                 at {start} (a range was skipped or handed twice)",
                st.kv.len
            )
            .into());
        }
        let eh = self.glm5_head_engine(e)?;
        let dr = self
            .glm5_dflash
            .as_ref()
            .ok_or("device-resident tap ingest without a loaded drafter")?;
        let h = st.n_embd;
        let n_taps = st.taps.len();
        let mut pclk = st.prof_on.then(|| ProfClock::start(e, eh));
        if st.rows_dev.is_none() {
            st.rows_dev = Some(eh.uninit(st.ring * n_taps * h)?);
        }
        let rows_dev = st.rows_dev.as_mut().expect("just filled");
        for (slot, &il) in st.taps.iter().enumerate() {
            let buf = sink.dev[slot].as_ref().ok_or_else(|| {
                format!(
                    "device-resident tap ingest: slot {slot} (layer {il}) was never written \
                     over rows {start}..{end}"
                )
            })?;
            let es = self.glm5_tap_slot_engine(e, il)?;
            if es.ctx().ordinal() == eh.ctx().ordinal() {
                eh.copy_2d_dtod_async(rows_dev, slot * h, n_taps * h, buf, h, h, t)?;
            } else {
                if st.stage[slot].is_none() {
                    st.stage[slot] = Some(eh.uninit(st.ring * h)?);
                }
                let staging = st.stage[slot].as_mut().expect("just filled");
                eh.copy_peer_from_async(staging, es, buf, t * h)?;
                es.stream().synchronize()?;
                eh.copy_2d_dtod_async(rows_dev, slot * h, n_taps * h, staging, h, h, t)?;
            }
        }
        if let Some(ck) = pclk.as_mut() {
            st.copy_ms += ck.lap(e, eh);
        }
        let feats = dr.draft.ctx_features(eh, rows_dev, t)?;
        if let Some(ck) = pclk.as_mut() {
            st.feat_ms += ck.lap(e, eh);
        }
        let pos: Vec<i32> = ((st.kv.len as i32)..(st.kv.len + t) as i32).collect();
        dr.draft.ingest_ctx(eh, &mut st.kv, &feats, &pos, t)?;
        if let Some(ck) = pclk.as_mut() {
            st.kv_ms += ck.lap(e, eh);
        }
        st.chunks += 1;
        Ok(())
    }

    /// The EAGER drafter ingest (one implementation for session creation and round 1):
    /// host feature rows `[n, row_w]` -> 256-row pageable HtoD chunks -> `ctx_features` ->
    /// `ingest_ctx`, appended at `kv.len`. Returns the profile buckets (all 0 with the
    /// clock absent).
    fn glm5_dflash_ingest_rows(
        &self,
        eh: &Engine,
        draft: &DflashDraft,
        kv: &mut DflashKv,
        rows: &[f32],
        row_w: usize,
        mut pclk: Option<&mut ProfClock>,
    ) -> Res<DraftIngestStats> {
        debug_assert_eq!(rows.len() % row_w, 0, "ragged feature rows");
        let n_new = rows.len() / row_w;
        let mut st = DraftIngestStats {
            rows: n_new,
            ..Default::default()
        };
        let mut r0 = 0usize;
        while r0 < n_new {
            let t_c = (n_new - r0).min(256);
            let chunk = eh.htod(&rows[r0 * row_w..(r0 + t_c) * row_w])?;
            if let Some(ck) = pclk.as_deref_mut() {
                st.h2d_ms += ck.lap(eh, eh);
            }
            let feats = draft.ctx_features(eh, &chunk, t_c)?;
            if let Some(ck) = pclk.as_deref_mut() {
                st.feat_ms += ck.lap(eh, eh);
            }
            let pos_c: Vec<i32> = ((kv.len as i32)..(kv.len + t_c) as i32).collect();
            draft.ingest_ctx(eh, kv, &feats, &pos_c, t_c)?;
            if let Some(ck) = pclk.as_deref_mut() {
                st.kv_ms += ck.lap(eh, eh);
            }
            r0 += t_c;
            st.chunks += 1;
        }
        Ok(st)
    }

    /// Free device memory per device this model can hold state on (primary + ppN stages),
    /// for the depth profile's before/after samples.
    fn glm5_free_mb(&self, e: &Engine) -> Vec<(usize, u64)> {
        let mut out = vec![(e.ctx().ordinal(), e.free_mem_mb())];
        if let Ok(rt) = crate::pp::PpNRt::get(e) {
            for stage in 0..rt.n_stages() {
                let se = rt.engine(stage, e);
                let ord = se.ctx().ordinal();
                if !out.iter().any(|(o, _)| *o == ord) {
                    out.push((ord, se.free_mem_mb()));
                }
            }
        }
        out
    }

    /// The engine that OWNS tap layer `il`'s device (the stage engine under a live split,
    /// the caller's engine otherwise) — the `glm5_tap_drain` placement rule, shared.
    fn glm5_tap_slot_engine<'e>(&self, e: &'e Engine, il: usize) -> Res<&'e Engine> {
        match crate::pp::pp_cuts(self.layers.len()) {
            Some(fence) if !crate::pp::pp2_streams_off() => {
                let rt = crate::pp::PpNRt::get(e)?;
                let stage = fence
                    .windows(2)
                    .position(|w| il >= w[0] && il < w[1])
                    .ok_or_else(|| format!("tap layer {il} outside every stage range"))?;
                Ok(rt.engine(stage, e))
            }
            _ => Ok(e),
        }
    }

    /// CHUNKED DRAFTER PRIME (lane/spec-route-depth-20260902, `MEMRA_GLM5_DRAFT_PRIME_V2`):
    /// prime the trunk over the engine's own chunk schedule and ingest each chunk's tap
    /// rows into the drafter KV right after that chunk's walk. Returns the LAST chunk's
    /// (boundary logits, hidden stack, stack rows).
    ///
    /// TRUNK PROGRAM, unchanged: `prime_cache` over `[start, end)` with
    /// `queued_after = plen - end` is exactly the call the whole-prompt entry makes per
    /// range (`prime_cache_hyper`'s loop: `queued_after + (t - end)`, `seq_end` invariant),
    /// so the trunk cache after the loop is byte-identical to the eager arm's — the
    /// restored-suffix gates (11/12) already pin "continue == cold" for this call shape.
    ///
    /// DATA MOVEMENT, the whole point: per chunk, five device slot buffers (one per tap
    /// layer, on the writing stage's device) are read back into ONE pinned cacheable slot
    /// buffer (sync, `dtoh_f32_into_pinned`), CPU-interleaved into the pinned
    /// `[t, n_taps * hidden]` rows buffer (the fc input layout), uploaded ASYNC from pinned
    /// (`htod_f32_from_pinned_async`), then `ctx_features` + `ingest_ctx` at the chunk
    /// width. The host transient is two chunk-sized pinned buffers (335 MB + 67 MB at 4096
    /// rows) instead of the eager arm's `[prompt, 5 x hidden]` pageable Vec; the eager
    /// arm's 21 GB of pageable DtoH + 21 GB of pageable HtoD at 256k become pinned DMA.
    /// The ingest kernels are stream-ordered behind the upload; the chunk ends with one
    /// stream drain so the rows buffer can be refilled (the upload helper's contract).
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list is the prime contract (engines, drafter, prompt, cache,
    // KV, taps) plus the two profile instruments; bundling would hide which is which
    fn glm5_draft_prime_chunked(
        &self,
        e: &Engine,
        eh: &Engine,
        draft: &DflashDraft,
        prompt: &[u32],
        cache: &mut Cache,
        kv: &mut DflashKv,
        taps: &[usize],
        mut prof: Option<&mut SpecFirstTokenProf>,
        mut pclk: Option<&mut ProfClock>,
    ) -> Res<(Vec<f32>, CudaSlice<f32>, usize)> {
        let plen = prompt.len();
        let n_embd = self.cfg.n_embd as usize;
        let n_taps = taps.len();
        let ranges = crate::hybrid_forward::hyper_prime_ranges(
            plen,
            self.layers.len(),
            self.gdn_prime_grid_on(),
        );
        let t_max = ranges.iter().map(|&(a, b)| b - a).max().unwrap_or(0);
        let f32b = std::mem::size_of::<f32>();
        let mut slot_buf = crate::PinnedHostBuf::new(t_max * n_embd * f32b)?;
        let mut rows_buf = crate::PinnedHostBuf::new(t_max * n_taps * n_embd * f32b)?;
        let (mut prime_ms, mut h2d_ms, mut feat_ms, mut kv_ms) = (0f64, 0f64, 0f64, 0f64);
        let mut last: Option<(Vec<f32>, CudaSlice<f32>, usize)> = None;
        for &(start, end) in &ranges {
            let t = end - start;
            cache.hc_taps = Some(HcTapSink::new_device_staged_at(
                taps.to_vec(),
                n_embd,
                t,
                start,
            ));
            let (l, _seed, h) = self.prime_cache(e, &prompt[start..end], cache, plen - end)?;
            if let Some(ck) = pclk.as_deref_mut() {
                prime_ms += ck.lap(e, eh);
            }
            let mut sink = cache
                .hc_taps
                .take()
                .ok_or("chunked drafter prime: chunk tap sink vanished")?;
            // ---- drain: device slots -> pinned slot -> interleaved pinned rows ----
            for (slot, &il) in taps.iter().enumerate() {
                let buf = sink.dev[slot].take().ok_or_else(|| {
                    format!(
                        "chunked drafter prime: tap slot {slot} (layer {il}) was never \
                         written by the prime walk over rows {start}..{end}"
                    )
                })?;
                let es = self.glm5_tap_slot_engine(e, il)?;
                es.dtoh_f32_into_pinned(&buf, &mut slot_buf, t * n_embd)?;
                let src = pinned_f32(&slot_buf, t * n_embd);
                let dst = pinned_f32_mut(&mut rows_buf, t * n_taps * n_embd);
                for r in 0..t {
                    let d0 = (r * n_taps + slot) * n_embd;
                    dst[d0..d0 + n_embd].copy_from_slice(&src[r * n_embd..(r + 1) * n_embd]);
                }
            }
            let feats_in = eh.htod_f32_from_pinned_async(&rows_buf, t * n_taps * n_embd)?;
            if let Some(ck) = pclk.as_deref_mut() {
                h2d_ms += ck.lap(e, eh);
            }
            let feats = draft.ctx_features(eh, &feats_in, t)?;
            if let Some(ck) = pclk.as_deref_mut() {
                feat_ms += ck.lap(e, eh);
            }
            let pos: Vec<i32> = ((kv.len as i32)..(kv.len + t) as i32).collect();
            draft.ingest_ctx(eh, kv, &feats, &pos, t)?;
            // The rows buffer is refilled by the next chunk: drain the stream first (the
            // async-upload contract). Also bounds the kv bucket.
            eh.stream().synchronize()?;
            if let Some(ck) = pclk.as_deref_mut() {
                kv_ms += ck.lap(e, eh);
            }
            last = Some((l, h, t));
        }
        debug_assert_eq!(kv.len, plen, "chunked drafter prime must cover the prompt");
        if let Some(pf) = prof.as_mut() {
            pf.prime_ms = prime_ms;
            pf.draft_prime_ms = h2d_ms + feat_ms + kv_ms;
            pf.draft_prime_h2d_ms = h2d_ms;
            pf.draft_prime_feat_ms = feat_ms;
            pf.draft_prime_kv_ms = kv_ms;
            pf.draft_prime_rows = plen;
            pf.draft_prime_chunks = ranges.len();
            pf.draft_prime_arm = "chunked";
        }
        last.ok_or_else(|| "chunked drafter prime: empty prime schedule".into())
    }

    /// Boundary capture for the DEFERRED prefix publication (lane/glm5-prefix-latent2,
    /// 2026-09-01): the generation-destroyed state at `pos == plen` — conv/ssm via
    /// `Cache::snapshot` (D2D copies), per-layer latent tails via `snapshot_tail`, the
    /// prime's boundary logits, the pre-output_norm boundary hidden. `None` (loud) on any
    /// refusal — a capture is an optimization the session must never fail on. The
    /// append-only planes (latent rows, final pool keys, full-attn KV) are NOT copied here:
    /// the worker slices them from the live cache at publish (`snapshot_plane_at`), which is
    /// legal because the glm5 verify rollback never truncates below the prime boundary.
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list is the capture contract (two engines, the cache, the
    // boundary logits and hidden stack, the boundary and the stack's own row count);
    // bundling would hide which row index is which
    fn glm5_prefix_boundary_capture(
        &self,
        e: &Engine,
        eh: &Engine,
        cache: &Cache,
        logits0: &[f32],
        hiddens: &CudaSlice<f32>,
        plen: usize,
        // Rows in `hiddens` (its last row is the boundary hidden): the whole prompt on the
        // eager cold prime, the last chunk on the chunked arm, the suffix on a restore.
        hidden_rows: usize,
    ) -> Option<crate::spec::SpecBoundaryCapture> {
        debug_assert_eq!(
            cache.pos, plen,
            "boundary capture must sit at the prime boundary"
        );
        let snap = match cache.snapshot(e) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("[glm5-spec] prefix boundary capture SKIPPED (cache snapshot: {err})");
                return None;
            }
        };
        // The ONLY latent layer that may legitimately sit empty at the boundary is the
        // MTP/NextN plane (allocated, never executed by the trunk on the DFlash2 arm).
        // Identity-keyed, not length-keyed (PR #96 review round 2, finding 1): a TRUNK
        // plane empty at the boundary is a regression that must refuse the capture, or
        // three length-keyed layers downstream would each wave it through and reproduce
        // the parent lane's fabrication shape per-layer.
        let mtp_plane_il = self.plan.mtp_blocks.first().map(|b| b.layer.index as usize);
        let mut latent_tails = Vec::with_capacity(cache.latent.len());
        for (il, l) in cache.latent.iter().enumerate() {
            match l {
                Some(l) if l.len == 0 && cache.pos > 0 => {
                    if Some(il) != mtp_plane_il {
                        eprintln!(
                            "[glm5-spec] prefix boundary capture SKIPPED (trunk latent \
                             layer {il} is EMPTY at the boundary — not the MTP plane; a \
                             capture would publish an absent history for a live layer)"
                        );
                        return None;
                    }
                    latent_tails.push(None)
                }
                Some(l) => {
                    if l.len != plen {
                        eprintln!(
                            "[glm5-spec] prefix boundary capture SKIPPED (latent layer {il} \
                             len {} != boundary {plen})",
                            l.len,
                        );
                        return None;
                    }
                    match l.snapshot_tail(e) {
                        Ok(t) => latent_tails.push(Some(t)),
                        Err(err) => {
                            eprintln!(
                                "[glm5-spec] prefix boundary capture SKIPPED (latent layer \
                                 {il}: {err})"
                            );
                            return None;
                        }
                    }
                }
                None => latent_tails.push(None),
            }
        }
        let last_h = crate::spec::capture_boundary_hidden(
            eh,
            hiddens,
            hidden_rows,
            self.cfg.n_embd as usize,
        );
        Some(crate::spec::SpecBoundaryCapture {
            snap,
            pos: plen,
            logits: logits0.to_vec(),
            last_h,
            latent_tails,
        })
    }

    /// Re-arm a glm5 spec session from a RESTORED trunk cache plus a published DFlash2
    /// drafter tail — the glm5 twin of `dspark_spec_session_from_restored`, EXTENDED with
    /// the suffix prime the multi-turn shape needs (lane/glm5-prefix-latent2, 2026-09-01).
    ///
    /// WHY IT IS EQUIVALENT TO A COLD PRIME, field by field:
    /// * `cache` — the caller's whole-entry restored trunk cache at `fed.len()` (KDA
    ///   conv/ssm + MLA latent rows + kpool keys + tail ring, the parent lane's restore),
    ///   and the SUFFIX primes onto it through `prime_cache` — the same continuation
    ///   program every chunk after the first of a cold prime runs.
    /// * drafter — `dkv` is rebuilt from the entry's tail into the SAME absolute rows
    ///   (`DflashKv::from_tail`, caller-side while the prefix cache is borrowable), and the
    ///   suffix's tap rows ride `pending` exactly as a cold session's prompt rows do — the
    ///   drafter's context is the committed tokens either way (a truncated tail below the
    ///   window can only move ACCEPTANCE, never output: verify arbitrates).
    /// * anchor — drawn from the SUFFIX prime's boundary logits with the request's own
    ///   sampler, exactly the cold composition; Philox counters fresh (a restore is a NEW
    ///   session — the frspec continuity law, the dspark restore's own convention).
    /// * republish — with `glm5_spec_prefix_on()` the session takes a NEW boundary capture
    ///   at `fed + suffix`, so the next turn hits a DEEPER prefix (the
    ///   MEMRA_SPEC_RESTORE_REPUBLISH posture; the worker's has_key dedupe drops equals).
    ///
    /// Refuses (never asserts) whenever the restored halves disagree — a caller that gets
    /// `Err` serves the plain hit (correct, slower).
    #[allow(clippy::too_many_arguments)]
    pub fn glm5_spec_session_from_restored(
        &self,
        e: &Engine,
        mut cache: Cache,
        fed: &[u32],
        suffix: &[u32],
        // The ENTRY's boundary logits (`ReuseEntry::last_logits`), read ONLY on the
        // full-cover arm (`suffix.is_empty()`), where there is no suffix prime to draw the
        // anchor from. A suffix-bearing restore ignores it and uses the prime's own row.
        boundary_logits: &[f32],
        dkv: DflashKv,
        ctx_cap: usize,
        sampling: Option<SpecSampling>,
    ) -> Res<Glm5SpecSession> {
        if self.hyper.is_none() {
            return Err("glm5_spec_session_from_restored requires a HyperConnections trunk".into());
        }
        // The composition laws a cold session enforces hold here too, fail-closed and by
        // name — a restored session must never be the door around an admission law.
        let tp_sharded = self.layers.iter().any(|l| match &l.mixer {
            Mixer::Kda(la) => la.tp.is_some(),
            Mixer::Mla(mla) => mla.tp.is_some(),
            _ => false,
        });
        if tp_sharded {
            return Err(
                "restored glm5 spec sessions carry no TP arm (the spec x TP composition is \
                 cold-session gated only); the plain hit serves"
                    .into(),
            );
        }
        if crate::pp::pp_cuts(self.layers.len()).is_some()
            && !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline)
        {
            return Err("pipeline rewrite is not qualified for this ModelPlan".into());
        }
        // DFlash2 source ONLY: the native MTP plane fill consumes trunk hiddens the restored
        // range does not have (re-running the trunk over it would be a second prime — the
        // whole cost this restore exists to avoid).
        let dr = self.glm5_dflash.as_ref().ok_or(
            "restored glm5 spec sessions require the DFlash2 drafter (MEMRA_GLM5_DFLASH): \
             the native MTP plane cannot be re-warmed from restored KV",
        )?;
        let source = crate::spec::resolve_draft_source_kind(
            self.plan.draft_source,
            self.mtp.is_some(),
            true,
        )
        .map_err(|why| format!("restored glm5 spec session has no draft source ({why})"))?;
        if !matches!(source, crate::spec::DraftSourceKind::Dflash2) {
            return Err(
                "restored glm5 spec sessions require the DFlash2 draft source; the plan \
                 selected another"
                    .into(),
            );
        }
        let sampling = sampling.filter(|sp| sp.temp > 0.0);
        if let Some(sp) = sampling.as_ref()
            && sp.pen_on()
        {
            return Err(
                "glm5 spec has no penalty arm: penalized requests serve on the plain path".into(),
            );
        }
        if fed.is_empty() {
            return Err("restored glm5 spec session needs a non-empty restored prefix".into());
        }
        // FULL-COVER ARM (memra#74, lane/glm5-fullcover-spec-route): an empty suffix is the
        // repeated-prompt shape. It refuses unless the arm is armed AND the entry carried
        // its boundary logits: without them there is no anchor row and no way to start a
        // round, which is exactly the `spec_restore_refusal` full-cover rule for MTP.
        if suffix.is_empty() {
            if !glm5_spec_fullcover_on() {
                return Err(
                    "restored glm5 spec session needs a non-empty suffix unless \
                     MEMRA_GLM5_SPEC_FULLCOVER=1 (empty-suffix full-cover hits otherwise \
                     keep the plain boundary-logits resume)"
                        .into(),
                );
            }
            if boundary_logits.len() != self.output.out_features() {
                return Err(format!(
                    "full-cover glm5 spec restore needs the entry's boundary logits \
                     ({} rows, got {})",
                    self.output.out_features(),
                    boundary_logits.len(),
                )
                .into());
            }
        }
        if cache.pos != fed.len() {
            return Err(format!(
                "restored glm5 spec session needs a whole-entry trunk cache: cache.pos {} \
                 != restored prefix {}",
                cache.pos,
                fed.len(),
            )
            .into());
        }
        if dkv.len != fed.len() {
            return Err(format!(
                "restored draft KV len {} != restored prefix {}",
                dkv.len,
                fed.len(),
            )
            .into());
        }
        if dkv.cap != ctx_cap {
            return Err(
                format!("restored draft KV cap {} != session ctx {ctx_cap}", dkv.cap,).into(),
            );
        }
        // Room for prefix + suffix, the anchor row and at least one verify round.
        if fed.len() + suffix.len() + 4 > ctx_cap {
            return Err(format!(
                "restored glm5 spec session needs ctx for prefix {} + suffix {} + anchor + \
                 one verify round, cap {ctx_cap}",
                fed.len(),
                suffix.len(),
            )
            .into());
        }
        let n_vocab = self.output.out_features();
        if let Some(map) = self.glm5_d2t() {
            if map.iter().any(|&t| t as usize >= n_vocab) {
                return Err(format!(
                    "glm5 FR-Spec d2t carries a token id >= n_vocab {n_vocab} — the ranks \
                     artifact was minted for a different vocabulary"
                )
                .into());
            }
            eprintln!(
                "[glm5-spec] draft head TRIMMED to {} rows (FR-Spec d2t engaged)",
                map.len(),
            );
        }
        if glm5_pmin() > 0.0 {
            eprintln!(
                "[glm5-spec] draft confidence gate armed: PMIN={:.3} PMIN0={} (native \
                 chain p-of-pick; DFlash2 selector-q tau-slot truncation)",
                glm5_pmin(),
                glm5_pmin0() as u8,
            );
        }
        // ---- suffix prime over the restored planes (the continuation program), taps armed
        // for exactly the suffix rows (`HcTapSink::origin` anchors the sink at the restored
        // boundary; the chunked walk's absolute bases rebase through it).
        let n_embd = self.cfg.n_embd as usize;
        let taps = glm5_dflash_tap_layers(&dr.draft, self.layers.len())?;
        let eh = self.glm5_head_engine(e)?;
        // ---- STREAM ORDERING, UNCONDITIONAL AND BEFORE THE SUFFIX BRANCH (memra#95, the
        // fleet-fatal full-cover panic).
        //
        // Everything this session restores was written on the CALLER's stream, in the
        // worker's admission path: the trunk planes by `prefix_restore_at`, and the DFlash2
        // drafter ctx KV by `DflashKv::from_tail` (worker.rs, the "glm5 spec restore, half 1
        // of 2" block — a fresh allocation plus per-layer `memset_zeros_view` +
        // `copy_range_into`). The draft phase of round 1 then READS that KV through
        // `eh` = `glm5_head_engine`, which under a live ppN split is the last stage's OWN
        // Engine (`pp::PpNRt::build` gives every stage s>0 its own Engine even on the primary
        // device, for scratch-pool isolation) and is used OUTSIDE any `rt.enter` scope — so
        // it launches on that Engine's own stream, which is neither the caller's stream nor
        // the stage stream. Nothing ordered the two, and the drafter's first `forward_round`
        // could read the ctx rows before the import landed.
        //
        // `fence_stages_behind` is NOT the fix and was the first thing this lane got wrong:
        // it orders `StageRt::stream`, the enter-scope stream, and says nothing about a stage
        // ENGINE's own stream. `order_engine_behind` is the seam for a body that hands a
        // stage engine to a helper without entering the stage.
        //
        // Why only the FULL-COVER arm: the suffix arm calls `prime_cache` on `e`, whose
        // logits come back through `Engine::dtoh` (a `stream().synchronize()`), which drains
        // the caller's stream before round 1 ever runs. The full-cover arm has no prime and
        // no device readback at all between the import and the round — the anchor is drawn
        // from the entry's host-side boundary logits — so it is the only restored path with
        // nothing between the two. That is why production never saw this on the suffix path,
        // and it is why the ordering is done HERE, once per session and for both arms,
        // instead of relying on a host sync that belongs to the prime.
        //
        // The measured failure: a NaN drafter row -> the top-k selector's documented
        // exhausted-slot sentinel (`0xffffffff`, `cu/kernels.cu` `topk_rows_f32` and its
        // sharded twin) -> `walk: candidate 4294967295 outside codebook vocab` at
        // dflash.rs:221/:804, on the FIRST round of a restored session (the receipts put the
        // panic between the `RESTORED session` line and the round's first `[glm5-acc]`),
        // fleet-fatal after one respawn.
        crate::pp::PpNRt::order_engine_behind(e, eh)?;
        // FIRST-TOKEN PROFILE (`MEMRA_SPEC_PROF=1`): the restored shape pays no cache or
        // drafter allocation here (the caller restored both); its prime bucket is the
        // SUFFIX prime (0 on a full-cover hit), which is what makes the restore worth having.
        let mut prof = spec_prof_on().then(SpecFirstTokenProf::default);
        let mut pclk = prof.as_ref().map(|_| ProfClock::start(e, eh));
        // FULL COVER: no suffix, so no prime, no taps and no republish (the entry ALREADY
        // sits at this boundary, and a capture here would be the same key the worker's has_key
        // dedupe drops). The boundary row is the entry's own; the drafter ctx KV is already
        // at `cache.pos` from the tail, so `pending` is empty and round 1's
        // `kv.len == cache.pos` invariant holds without an ingest.
        let (logits_s, tap_rows, prefix_capture) = if suffix.is_empty() {
            (boundary_logits.to_vec(), Vec::new(), None)
        } else {
            cache.hc_taps = Some(HcTapSink::new_at(
                taps.clone(),
                n_embd,
                suffix.len(),
                fed.len(),
            ));
            let (logits_s, _seed, hiddens) = self.prime_cache(e, suffix, &mut cache, 0)?;
            if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
                pf.prime_ms = ck.lap(e, eh);
            }
            // Republish capture at the NEW (deeper) boundary — pos == fed + suffix here.
            let capture = if glm5_spec_prefix_on() {
                // `hiddens` is the SUFFIX stack: its last row is the boundary hidden
                // (indexed through `suffix.len()`, not the absolute boundary).
                self.glm5_prefix_boundary_capture(
                    e,
                    eh,
                    &cache,
                    &logits_s,
                    &hiddens,
                    cache.pos,
                    suffix.len(),
                )
            } else {
                None
            };
            let sink = cache
                .hc_taps
                .take()
                .ok_or("glm5 restored-session suffix tap sink vanished")?;
            (logits_s, sink.rows, capture)
        };
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.capture_ms = ck.lap(e, eh);
        }
        let mut sctr = 0u32;
        let anchor = match sampling.as_ref() {
            Some(sp) => crate::spec::sample_boundary_token(
                eh,
                &logits_s,
                sp,
                &[],
                &mut sctr,
                "glm5-restore",
            )?,
            None => argmax(&logits_s) as u32,
        };
        if let (Some(pf), Some(ck)) = (prof.as_mut(), pclk.as_mut()) {
            pf.anchor_ms = ck.lap(e, eh);
        }
        let mut committed = Vec::with_capacity(fed.len() + suffix.len());
        committed.extend_from_slice(fed);
        committed.extend_from_slice(suffix);
        // Engagement receipt (the dspark restore's shape — the deploy gate greps this; a
        // cached_tokens number alone cannot distinguish a spec restore from a plain hit).
        eprintln!(
            "[glm5-spec] RESTORED session: {} prefix tokens + {} suffix from cache — no \
             cold prime (drafter tail rows {}, arm {})",
            fed.len(),
            suffix.len(),
            dkv.len,
            if suffix.is_empty() {
                "full-cover"
            } else {
                "suffix-prime"
            },
        );
        Ok(Glm5SpecSession {
            cache,
            committed,
            anchor,
            anchor_emitted: false,
            pending: Vec::new(),
            draft: Glm5DraftState::Dflash2 {
                kv: dkv,
                pending: tap_rows,
                taps,
            },
            sampling,
            sctr,
            uctr: 0,
            rounds: 0,
            done: false,
            max_ctx: ctx_cap,
            mtp_il: None,
            prefix_capture,
            prof_rounds: prof.as_ref().map(|_| SpecRoundsLog::default()),
            prof,
        })
    }

    /// The loaded FR-Spec draft->target map, when a trim artifact actually landed on the
    /// embedded head (None = full-vocab head, rank id == token id).
    fn glm5_d2t(&self) -> Option<&[u32]> {
        self.mtp
            .as_ref()
            .and_then(|head| head.d2t.as_deref())
            .filter(|map| !map.is_empty())
    }

    /// ONE serve burst (the worker's per-tick call, `step_glm5_spec`): rounds of
    /// draft(K) -> `glm5_verify_rows` -> accept -> rollback/commit until `target` new
    /// tokens are out, EOS commits, or the context guard trips. Returns
    /// `(burst, drafted, accepted)`; the burst may overshoot `target` by up to K (a
    /// round commits j+1 tokens atomically — the engine surplus stays committed in the
    /// session cache and the WORKER clamps public emission to the request budget, the
    /// SpecSession overshoot contract).
    pub fn glm5_spec_session_burst(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        target: usize,
        k: usize,
        eos: &[u32],
    ) -> Res<(Vec<u32>, usize, usize)> {
        self.glm5_spec_session_burst_inner(
            e,
            sess,
            target,
            k,
            eos,
            &mut Glm5SpecKnobs::default(),
            None,
        )
    }

    /// [`glm5_spec_session_burst`] with a ROUND-CADENCE commit hook (lane/b200-spec-ttft-
    /// 20260902, the `MEMRA_SPEC_FIRST_TOKEN_EAGER` door's engine half; the spec.rs
    /// `on_commit` sse-cadence pattern): `on_commit` is called with every newly committed
    /// slice of the burst — first the prime's anchor alone, then each round's `j` accepted
    /// drafts + bonus — as DISJOINT, IN-ORDER slices whose concatenation IS the returned
    /// burst, byte for byte. The hook returns nothing and the loop's control flow never
    /// reads it, so the tokens produced are exactly `glm5_spec_session_burst`'s; only WHEN
    /// the caller learns them moves (from burst end to round end). Without a hook the
    /// first token of a cold session waits for the whole `target`-token burst.
    pub fn glm5_spec_session_burst_streamed(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        target: usize,
        k: usize,
        eos: &[u32],
        on_commit: CommitHook<'_>,
    ) -> Res<(Vec<u32>, usize, usize)> {
        self.glm5_spec_session_burst_inner(
            e,
            sess,
            target,
            k,
            eos,
            &mut Glm5SpecKnobs::default(),
            Some(on_commit),
        )
    }

    /// [`glm5_spec_session_burst`] with GATE INSTRUMENTS (`Glm5SpecKnobs` — never a serving
    /// surface; no serving path constructs a non-default value).
    pub fn glm5_spec_session_burst_gated(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        target: usize,
        k: usize,
        eos: &[u32],
        knobs: &mut Glm5SpecKnobs<'_>,
    ) -> Res<(Vec<u32>, usize, usize)> {
        self.glm5_spec_session_burst_inner(e, sess, target, k, eos, knobs, None)
    }

    /// The one burst loop behind the three public entries (plain / streamed / gated).
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list is the burst contract plus the two instruments (gate
    // knobs, commit hook); bundling would hide which inputs are serving vs instrument
    fn glm5_spec_session_burst_inner(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        target: usize,
        k: usize,
        eos: &[u32],
        knobs: &mut Glm5SpecKnobs<'_>,
        mut on_commit: Option<CommitHook<'_>>,
    ) -> Res<(Vec<u32>, usize, usize)> {
        let cap = Self::hyper_batch_cap();
        if k == 0 || k + 1 > cap {
            return Err(format!(
                "glm5_spec_session_burst: k={k} outside 1..={} (verify rows = k+1 must stay \
                 inside the decode-exact knee, cap {cap})",
                cap - 1
            )
            .into());
        }
        if let Glm5DraftState::Dflash2 { .. } = sess.draft {
            let b = self
                .glm5_dflash
                .as_ref()
                .ok_or("dflash session on a model with no loaded drafter")?
                .draft
                .cfg
                .block_size;
            if k + 1 > b {
                return Err(format!(
                    "glm5_spec_session_burst: k={k} exceeds the DFlash2 drafter's block \
                     (block_size {b} = anchor + {} drafts, the trained mask pattern) — \
                     the worker clamps operator K pins to {} for this source; refusing \
                     loudly rather than drafting an untrained shape",
                    b - 1,
                    b - 1
                )
                .into());
            }
        }
        let d2t = self.glm5_d2t();
        if d2t.is_some() && knobs.skip_d2t_remap {
            eprintln!("[glm5-spec] d2t REMAP SKIPPED — red-arm instrument, drafts are rank ids");
        }
        let sp_on: Option<SpecSampling> = sess.sampling.filter(|sp| sp.temp > 0.0);
        let mut out: Vec<u32> = Vec::with_capacity(target + k);
        let mut drafted = 0usize;
        let mut accepted = 0usize;
        let mut phase: Option<SpecPhaseNs> =
            crate::spec_phase::spec_trace_on().then(SpecPhaseNs::default);
        // FIRST-TOKEN PROFILE: the first burst's wall (its tokens are host-visible at
        // return, so no drain is needed to bound it).
        let first_burst = (!sess.anchor_emitted && sess.prof.is_some())
            .then(|| (std::time::Instant::now(), sess.rounds));
        // The hook's own share of the first burst's wall (host-only detext + sends), so
        // the profile can separate engine time from emission time under the eager door.
        let mut hook_ns: u64 = 0;
        // sse-cadence flush cursor: everything in out[..flushed] has been handed to on_commit.
        let mut flushed = 0usize;
        if !sess.anchor_emitted {
            // The prime's boundary token: emitted exactly once, by the first burst.
            out.push(sess.anchor);
            sess.anchor_emitted = true;
            if eos.contains(&sess.anchor) {
                sess.done = true;
            }
            if let Some(cb) = on_commit.as_mut() {
                let t_cb = first_burst.map(|_| std::time::Instant::now());
                cb(&out[flushed..]);
                if let Some(t_cb) = t_cb {
                    hook_ns += t_cb.elapsed().as_nanos() as u64;
                }
                flushed = out.len();
            }
        }
        while out.len() < target && !sess.done {
            // Context guard: a round appends up to k+1 trunk rows from `cache.pos` (and the
            // draft plane stays <= pos + k), so the next round must fit with one row slack.
            if sess.cache.pos + k + 2 > sess.max_ctx {
                sess.done = true;
                break;
            }
            // DEPTH LOG (lane/spec-route-depth-20260902, `MEMRA_SPEC_PROF=1`): the first
            // SPEC_PROF_ROUNDS rounds of a session get their own phase accumulator (the
            // trace's drains), a wall clock, and the per-row-arm row count — folded into
            // the per-burst trace accumulator afterwards so both instruments agree.
            let log_round = sess.prof_rounds.as_ref().is_some_and(|l| l.wants_more());
            let mut rp = log_round.then(SpecPhaseNs::default);
            let t_round = log_round.then(std::time::Instant::now);
            let seq0 = V_SEQ_ROWS.load(std::sync::atomic::Ordering::Relaxed);
            let ctx0 = sess.cache.pos;
            let (round_tokens, n_drafted) = {
                let ph: Option<&mut SpecPhaseNs> = match (rp.as_mut(), phase.as_mut()) {
                    (Some(r), _) => Some(r),
                    (None, p) => p,
                };
                self.glm5_spec_round(e, sess, k, d2t, sp_on.as_ref(), knobs, ph)?
            };
            if let (Some(r), Some(t0)) = (rp.as_ref(), t_round) {
                if let Some(p) = phase.as_mut() {
                    p.add(r);
                }
                if let Some(log) = sess.prof_rounds.as_mut() {
                    let ms = |ns: u64| ns as f32 / 1e6;
                    log.push(SpecRoundProf {
                        wall_ms: t0.elapsed().as_secs_f32() * 1e3,
                        draft_ms: ms(r.draft),
                        verify_ms: ms(r.verify),
                        accept_ms: ms(r.accept),
                        rest_ms: ms(r.roll + r.maint),
                        k: n_drafted as u16,
                        j: (round_tokens.len() - 1) as u16,
                        ctx: ctx0 as u32,
                        seq_rows: (V_SEQ_ROWS.load(std::sync::atomic::Ordering::Relaxed) - seq0)
                            as u32,
                    });
                }
            }
            drafted += n_drafted;
            accepted += round_tokens.len() - 1; // j accepted drafts + the bonus row
            for &tok in &round_tokens {
                if eos.contains(&tok) {
                    sess.done = true;
                }
            }
            out.extend_from_slice(&round_tokens);
            sess.rounds += 1;
            if let Some(cb) = on_commit.as_mut() {
                // sse-cadence: this round's accepted drafts + bonus are committed — hand
                // the caller exactly the not-yet-flushed tail (disjoint, in order).
                let t_cb = first_burst.map(|_| std::time::Instant::now());
                cb(&out[flushed..]);
                if let Some(t_cb) = t_cb {
                    hook_ns += t_cb.elapsed().as_nanos() as u64;
                }
                flushed = out.len();
            }
        }
        debug_assert!(
            on_commit.is_none() || flushed == out.len(),
            "every committed token must have been handed to on_commit"
        );
        if let Some(ph) = phase.as_ref() {
            ph.emit("glm5-phase", "glm5-phase-v", k);
        }
        if let (Some((t0, rounds0)), Some(pf)) = (first_burst, sess.prof.as_mut()) {
            pf.first_burst_ms = t0.elapsed().as_secs_f64() * 1e3;
            pf.first_burst_hook_ms = hook_ns as f64 / 1e6;
            pf.first_burst_rounds = sess.rounds - rounds0;
            pf.first_burst_tokens = out.len();
        }
        Ok((out, drafted, accepted))
    }

    /// One draft->verify->accept->rollback->re-seed round over the session state. Returns
    /// `(round_tokens, n_drafted)`: the round's committed tokens (`j` accepted drafts + the
    /// bonus token) and how many drafts actually entered the verify (== `k` today; the
    /// confidence gate may truncate it below `k`).
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors the round contract (session + policy + gate knobs +
    // the trace accumulator); bundling would hide which inputs are serving vs instrument
    fn glm5_spec_round(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        k: usize,
        d2t: Option<&[u32]>,
        sp: Option<&SpecSampling>,
        knobs: &mut Glm5SpecKnobs<'_>,
        mut phase: Option<&mut SpecPhaseNs>,
    ) -> Res<(Vec<u32>, usize)> {
        let n_vocab = self.output.out_features();
        let n_embd = self.cfg.n_embd as usize;
        // The MTP block / DFlash2 drafter, the trunk lm head and the verify walk's returned
        // rows all live on the LAST stage under a split — every draft-chain and accept-side
        // op below runs through the head engine (identity when the door is shut).
        let eh = self.glm5_head_engine(e)?;
        let mut t_mark = phase.as_ref().map(|_| SpecPhaseNs::clock(e, eh));
        // FIRST-TOKEN PROFILE (`MEMRA_SPEC_PROF=1`): round 1 only — the same drains as the
        // trace, bucketed into the session's one-shot profile instead of the per-burst
        // accumulator. `pclk` rides into the DFlash2 draft fn so the prompt ingest (the
        // drafter prime) gets its own bucket before the draft bucket starts.
        let mut pclk = (sess.rounds == 0 && sess.prof.is_some()).then(|| ProfClock::start(e, eh));
        // Phase-boundary bump: drain, bucket the elapsed ns, restart the clock. No-op with
        // the trace off (t_mark is None and no stream is ever synchronized).
        macro_rules! bump {
            ($field:ident) => {
                if let (Some(ph), Some(t0)) = (phase.as_deref_mut(), t_mark.as_mut()) {
                    let now = SpecPhaseNs::clock(e, eh);
                    ph.$field += now.duration_since(*t0).as_nanos() as u64;
                    *t0 = now;
                }
            };
        }
        // Profile lap into the named first-round bucket (no-op with the profile off).
        macro_rules! plap {
            ($field:ident) => {
                if let (Some(ck), Some(pf)) = (pclk.as_mut(), sess.prof.as_mut()) {
                    pf.$field = ck.lap(e, eh);
                }
            };
        }

        // CONFIDENCE GATE resolution (loop-port 2): the env pair is the serving surface
        // (the step37 family, no new flags); the knobs override is the gate instrument.
        let (p_min, pmin0) = knobs
            .pmin_override
            .unwrap_or_else(|| (glm5_pmin(), glm5_pmin0()));

        // ---- 1+2. produce the K drafts (+ the retained q side), SOURCE-KEYED. Everything
        // after this point is shared and source-blind — the exactness seam (module doc).
        let (drafts, qside, mtp_committed_len) = match sess.draft {
            Glm5DraftState::Dflash2 { .. } => {
                let (d, q) = self.glm5_dflash_round_drafts(
                    eh,
                    sess,
                    k,
                    sp,
                    knobs,
                    p_min,
                    pmin0,
                    pclk.as_mut(),
                )?;
                (d, q, 0)
            }
            Glm5DraftState::NativeMtp => {
                let mtp_il = sess.mtp_il.ok_or("native-mtp arm without a plane index")?;
                // ---- feed pending committed pairs; the last call yields draft 1 ----
                let mut last: Option<(CudaSlice<f32>, CudaSlice<f32>)> = None;
                for (tok, h) in sess.pending.drain(..) {
                    let plane_len = sess.cache.latent[mtp_il]
                        .as_ref()
                        .ok_or("MTP plane missing")?
                        .len;
                    last = Some(self.mtp_head_forward_mla_cached(
                        eh,
                        0,
                        tok,
                        &h,
                        &mut sess.cache,
                        plane_len,
                    )?);
                }
                let (mut d_logits, mut carrier) =
                    last.ok_or("glm5 spec round started with no pending committed pair")?;
                let mtp_committed_len = sess.cache.latent[mtp_il]
                    .as_ref()
                    .ok_or("MTP plane missing")?
                    .len;

                // ---- chain K drafts. Greedy route: argmax over the draft head. Sampled
                // route: filtered Gumbel draw through the session's device Philox stream
                // (`sctr`), with the per-step filtered stats + logits retained — they are
                // the q side of the accept walk. Trimmed heads yield RANK ids that remap
                // through d2t to true vocab before anything consumes them (chain feed,
                // verify, output); the q gather keeps the rank id.
                let d_vocab = d2t.map(|m| m.len()).unwrap_or(n_vocab);
                let mut drafts: Vec<u32> = Vec::with_capacity(k); // true-vocab tokens
                let mut draft_idx: Vec<u32> = Vec::with_capacity(k); // draft-head rank ids
                let mut draft_logits: Vec<CudaSlice<f32>> = Vec::new(); // sampled route only
                let mut draft_stats: Vec<(f32, f32, f32)> = Vec::new(); // (mx, th, z), sampled only
                for ki in 0..k {
                    let (idx, sampled_stats) = match sp {
                        Some(sp) => {
                            let (idx, stats) =
                                glm5_sampled_draft(eh, &d_logits, d_vocab, sp, &mut sess.sctr)?;
                            (idx, Some(stats))
                        }
                        None => {
                            // Device argmax + ONE 4-byte readback per draft (loop-port 1)
                            // — replaces the full d_vocab logits DtoH + host argmax the
                            // map names at this seam. Same tie-break contract
                            // (argmax_gate); drafts never decide exactness anyway, the
                            // verify arbitrates. The #87 sentinel guard mirrors the
                            // spec.rs graph chain: a device argmax may emit a sentinel
                            // on a NaN row — refuse loudly, never gather an OOB embed.
                            let td = eh.argmax_token_device(&d_logits, d_vocab)?;
                            let idx = crate::spec::guard_vocab_token(
                                eh.dtoh_u32_one(&td)?,
                                d_vocab,
                                &format!(
                                    "glm5 native draft argmax at round {} ki={ki}",
                                    sess.rounds
                                ),
                            )?;
                            (idx, None)
                        }
                    };
                    // P-MIN CONFIDENCE GATE (loop-port 2, the spec.rs chain break): p =
                    // the head's softmax confidence in its own pick (the `g_p` statistic,
                    // prob_of_token_device kernels), one 4-byte read — armed rounds only.
                    // Break BEFORE the pick is drafted or the next full-MoE-layer chain
                    // forward is paid; a discarded sampled draw's Philox advance stands
                    // (spec.rs eager parity: "counts the p-min-discarded token too").
                    if p_min > 0.0 {
                        let tok_d = eh.htod_u32_v(&[idx])?;
                        let p_d = eh.prob_of_token_device(&d_logits, &tok_d, d_vocab)?;
                        let p = eh.dtoh(&p_d)?[0];
                        if p < p_min && (ki > 0 || pmin0) {
                            break;
                        }
                    }
                    if let Some(stats) = sampled_stats {
                        draft_stats.push(stats);
                        draft_logits.push(eh.clone_dtod(&d_logits)?);
                    }
                    let mut d = match d2t {
                        Some(map) if !knobs.skip_d2t_remap => map[idx as usize],
                        _ => idx,
                    };
                    if let Some(over) = knobs.draft_override.as_mut() {
                        d = over(sess.rounds, ki, d);
                    }
                    drafts.push(d);
                    draft_idx.push(idx);
                    if ki + 1 < k {
                        let plane_len = sess.cache.latent[mtp_il]
                            .as_ref()
                            .ok_or("MTP plane missing")?
                            .len;
                        let (lg, ca) = self.mtp_head_forward_mla_cached(
                            eh,
                            0,
                            d,
                            &carrier,
                            &mut sess.cache,
                            plane_len,
                        )?;
                        d_logits = lg;
                        carrier = ca;
                    }
                }
                let q = match sp {
                    Some(_) => Glm5DraftQ::Mtp {
                        draft_idx,
                        draft_logits,
                        draft_stats,
                    },
                    None => Glm5DraftQ::None,
                };
                (drafts, q, mtp_committed_len)
            }
        };
        bump!(draft);
        plap!(first_draft_ms);

        // DFlash2 source: arm the verify tap — the walk's rows are next round's drafter
        // context features (rows 0..keep survive the accept; the sink is taken in step 7).
        // DEVICE-STAGED (loop-port 1): the walk D2Ds each tapped layer's contracted rows
        // instead of blocking on five in-walk DtoHs; step 7 drains post-walk.
        if let Glm5DraftState::Dflash2 { taps, .. } = &sess.draft {
            sess.cache.hc_taps = Some(HcTapSink::new_device_staged(
                taps.clone(),
                n_embd,
                drafts.len() + 1,
            ));
        }

        // ---- 3. verify: one t=K+1 walk over the trunk ----
        let mut rows: Vec<u32> = Vec::with_capacity(drafts.len() + 1);
        rows.push(sess.anchor);
        rows.extend_from_slice(&drafts);
        let (vlogits, collapsed, ckpt) = self.glm5_verify_rows(e, &rows, &mut sess.cache)?;
        bump!(verify);
        plap!(first_verify_ms);

        // ---- 4. accept ----
        // ZERO-DRAFT SAMPLED ROUND (PMIN0): the verify batch is just the anchor row —
        // m=1 = a plain decode step, exactly the llama.cpp gating spec.rs vendored. The
        // bonus is the full-accept filtered-Gumbel draw from that one row through the
        // session's Philox stream (identical in distribution to the plain sampled step
        // this round degenerates to). Greedy zero-draft rounds ride the general arm
        // below (j=0, bonus = the row-0 device argmax).
        let (j, bonus) = if let (true, Some(sp)) = (drafts.is_empty(), sp) {
            (
                0,
                self.glm5_sampled_bonus(eh, sess, sp, &vlogits, 0, n_vocab)?,
            )
        } else {
            match (sp, &qside) {
                (None, _) => {
                    // Greedy longest matching prefix (the DFlash2 probe's rule); bonus = the
                    // target's own argmax at the first non-accepted slot. Byte-deterministic —
                    // the instrument the spec-vs-plain identity gates pin.
                    //
                    // DEVICE ACCEPT ARGMAXES (loop-port 1, the K=1 flip): per verify row, a
                    // device argmax into one [t] slot buffer, then ONE tiny u32 readback —
                    // replacing the (K+1) x n_vocab logits DtoH + (K+1) host argmax scans
                    // (~2.4 MB + a host walk over 600k floats at K=3 on the real head; the
                    // 3way arithmetic needs 0.67 ms off the fixed round cost to flip K=1).
                    // `argmax_token_device_col` carries the host argmax's tie-break contract
                    // bit for bit (lowest index wins, argmax_gate-validated), so the accept
                    // walk commits the SAME tokens in the SAME order — the byte-identity
                    // batteries below stay the proof.
                    let t = rows.len();
                    let mut vam_d = eh.alloc_u32_zeroed(t)?;
                    for r in 0..t {
                        eh.argmax_token_device_col(&vlogits, r, n_vocab, &mut vam_d, r)?;
                    }
                    let vam = eh.dtoh_u32(&vam_d)?;
                    let mut j = 0usize;
                    while j < drafts.len() && drafts[j] == vam[j] {
                        j += 1;
                    }
                    if knobs.accept_probe {
                        self.glm5_accept_probe(eh, sess.rounds, &vlogits, &drafts, &vam, j)?;
                    }
                    (j, vam[j])
                }
                (
                    Some(sp),
                    Glm5DraftQ::Mtp {
                        draft_idx,
                        draft_logits,
                        draft_stats,
                    },
                ) => self.glm5_sampled_accept(
                    eh,
                    sess,
                    sp,
                    &vlogits,
                    &drafts,
                    draft_idx,
                    draft_logits,
                    draft_stats,
                    d2t,
                    drafts.len(),
                )?,
                (Some(sp), Glm5DraftQ::Selector { prop, dl }) => {
                    // The q38 serve route's rejection walk, VERBATIM (`dspark_accept_sampled`):
                    // `rows` = [anchor, drafts..] is its cand contract, verify row j arbitrates
                    // rows[j+1], the bonus draws from row k on full accept, and the reject-slot
                    // residual uses the selector's sparse candidate-set q. Philox counters are
                    // this session's — randomness never repeats across bursts.
                    let (m, next) = crate::dflash::dspark_accept_sampled(
                        eh,
                        &vlogits,
                        &rows,
                        rows.len(),
                        n_vocab,
                        dl,
                        prop,
                        sp,
                        &[],
                        &mut sess.sctr,
                        &mut sess.uctr,
                    )?;
                    let next = crate::spec::guard_vocab_token(
                        next,
                        n_vocab,
                        &format!(
                            "glm5 dflash2 sampled verify bonus at round {} j={m}",
                            sess.rounds
                        ),
                    )?;
                    (m, next)
                }
                (Some(_), Glm5DraftQ::None) => {
                    unreachable!("sampled round without a retained q side")
                }
            }
        };
        bump!(accept);
        plap!(first_accept_ms);

        // ---- 5. commit j drafts + the bonus token ----
        let mut round_tokens: Vec<u32> = Vec::with_capacity(j + 1);
        round_tokens.extend_from_slice(&drafts[..j]);
        round_tokens.push(bonus);

        // ---- 6. rollback the trunk to the accepted prefix ----
        let keep = j + 1;
        if knobs.disable_rollback {
            // RED-ARM INSTRUMENT: move pos, leave every state plane at post-row-K.
            sess.cache.pos = ckpt.pos + keep;
        } else {
            self.glm5_verify_rollback(e, &mut sess.cache, &ckpt, keep)?;
        }
        bump!(roll);
        plap!(first_roll_ms);

        // ---- 7+8. draft-source state maintenance, SOURCE-KEYED ----
        match &mut sess.draft {
            Glm5DraftState::NativeMtp => {
                // MTP plane: len reset to the committed boundary (chain rows out), then
                // re-seed the pending pairs (token at pos0+i, collapsed row i-1).
                self.glm5_mtp_plane_reset(e, &mut sess.cache, mtp_committed_len)?;
                for i in 1..=keep {
                    let tok = round_tokens[i - 1];
                    let h = self.glm5_seed_row(eh, &collapsed, rows.len(), i - 1)?;
                    sess.pending.push((tok, h));
                }
            }
            Glm5DraftState::Dflash2 { pending, taps, .. } => {
                // The kept verify rows' tap features (rows 0..keep = [anchor, accepted
                // drafts]) become next round's drafter context — the probe's
                // `F_feat[new_lo:start]` advance. The drafter's own KV block rows were
                // transient (forward_round never moves kv.len), so no drafter rollback
                // exists to run. The trunk-side MTP plane was never touched.
                // Device-staged rows drain HERE — the round's one post-walk sync point
                // for tap features (loop-port 1).
                let mut sink = sess
                    .cache
                    .hc_taps
                    .take()
                    .ok_or("glm5 dflash verify tap sink vanished")?;
                self.glm5_tap_drain(e, &mut sink)?;
                let row_w = taps.len() * n_embd;
                pending.extend_from_slice(&sink.rows[..keep * row_w]);
            }
        }
        // Cache-row bookkeeping: the trunk committed rows [anchor, drafts[..j]] (keep =
        // j+1), so `committed` gains exactly those tokens; the BONUS is the new live
        // anchor — emitted this round, consumed by the trunk as the NEXT round's row 0
        // (the dspark `last` convention). Invariant at every round boundary:
        // `cache.pos == committed.len()`, token-for-token.
        sess.committed.push(sess.anchor);
        sess.committed.extend_from_slice(&drafts[..j]);
        sess.anchor = bonus;
        bump!(maint);
        plap!(first_maint_ms);
        if let Some(pf) = sess.prof.as_mut()
            && pclk.is_some()
        {
            pf.first_round_tokens = round_tokens.len();
        }
        if let Some(ph) = phase {
            ph.rounds += 1;
        }
        Ok((round_tokens, drafts.len()))
    }

    /// THE ACCEPTANCE-RACE FIX (lane/glm5-accrace 2026-09-01): order the CALLER's stream
    /// behind EVERY stage stream — the exit mirror of
    /// [`crate::pp::PpNRt::fence_stages_behind`], built from
    /// [`crate::pp::PpNRt::publish_all_to`] (event waits, never a device sync, so the stage
    /// streams keep running).
    ///
    /// Call OUTSIDE any `rt.enter` scope: `e.stream()` must resolve to the caller's stream,
    /// not a stage's. Door shut or the same-stream seam (`MEMRA_PP_STREAMS=0`): a no-op by
    /// construction, so single-device and STREAMS=0 behaviour is untouched.
    fn glm5_publish_stages(&self, e: &Engine) -> Res<()> {
        if crate::pp::pp_cuts(self.layers.len()).is_some() && !crate::pp::pp2_streams_off() {
            let rt = crate::pp::PpNRt::get(e)?;
            let dst = e.stream();
            rt.publish_all_to(&dst)?;
        }
        Ok(())
    }

    /// GATE INSTRUMENT (lane/glm5-accrace; contract in [`Glm5SpecKnobs::accept_probe`]):
    /// one stderr line per greedy round pairing the DEVICE accept row against a HOST
    /// argmax over the same buffer, plus a per-row (argmax, row hash) census so two runs
    /// of the same deterministic fixture can be diffed round-for-round.
    fn glm5_accept_probe(
        &self,
        eh: &Engine,
        round: usize,
        vlogits: &CudaSlice<f32>,
        drafts: &[u32],
        vam: &[u32],
        j: usize,
    ) -> Res<()> {
        let n_vocab = self.output.out_features();
        let t = vam.len();
        let host = eh.dtoh(vlogits)?;
        let mut hvam: Vec<u32> = Vec::with_capacity(t);
        let mut rows_census: Vec<String> = Vec::with_capacity(t);
        for r in 0..t {
            let row = &host[r * n_vocab..(r + 1) * n_vocab];
            let am = argmax(row) as u32;
            hvam.push(am);
            // FNV-1a over the row's f32 BITS: a bit-level fingerprint, so a run-to-run
            // diff is exact rather than eyeballed at some print precision.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for v in row {
                for b in v.to_bits().to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
            rows_census.push(format!("{r}:{am}:{h:016x}"));
        }
        eprintln!(
            "[glm5-accrace] round={round} t={t} j={j} keep={} drafts={drafts:?} \
             dev_vam={vam:?} host_vam={hvam:?} agree={} rows=[{}]",
            j + 1,
            hvam == vam,
            rows_census.join(" ")
        );
        Ok(())
    }

    /// ONE round's drafts from the DFlash2 source (module doc, DRAFT SOURCE SEAM) — the
    /// shipped q38 selector round, re-aimed at glm5's hc-contract features:
    ///
    ///   1. ingest pending committed feature rows into the drafter's own ctx KV (chunked
    ///      at 256 rows — the qwen depth-OOM bound; round 1 carries the whole prompt);
    ///   2. block forward `[anchor, MASK x b-1]` at absolute positions over the cached ctx
    ///      (`forward_round` — block K/V transient, exactly the reference crop);
    ///   3. draft logits = trunk lm_head over rows 1..b (mask-fill harvest — the DFlash2
    ///      family census; FR-Spec trim consumed exactly as the dspark serve arm does);
    ///   4. selector walk: greedy chain, or the sampled candidate-set walk whose recorded
    ///      q (`DsparkDraftSample::Selector`) the shared rejection accept consumes.
    ///
    /// Drafts are truncated to `k` (the chain is sequential, so a prefix is well-formed),
    /// then to the CONFIDENCE prefix when `p_min` is armed (loop-port 2, the tau-slot
    /// form): the selector's recorded per-slot q — `q_chosen` on the sampled walk, its
    /// T=1 twin on the greedy walk — gates each slot through `glm5_conf_keep`, so the
    /// low-confidence tail never enters the verify batch (a truncated round rides down
    /// the `31.6 + 20.1*K` line; rejection sampling stays exact for any proposal prefix).
    /// `knobs.draft_override` applies after — the gate instrument, never serving.
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors the round contract plus the resolved gate pair
    fn glm5_dflash_round_drafts(
        &self,
        eh: &Engine,
        sess: &mut Glm5SpecSession,
        k: usize,
        sp: Option<&SpecSampling>,
        knobs: &mut Glm5SpecKnobs<'_>,
        p_min: f32,
        pmin0: bool,
        pclk: Option<&mut ProfClock>,
    ) -> Res<(Vec<u32>, Glm5DraftQ)> {
        let dr = self
            .glm5_dflash
            .as_ref()
            .ok_or("glm5 dflash draft state without a loaded drafter")?;
        let draft = &dr.draft;
        let c = &draft.cfg;
        let b = c.block_size;
        let n_embd = self.cfg.n_embd as usize;
        let n_vocab = self.output.out_features();
        let Glm5SpecSession {
            draft: state,
            cache,
            anchor,
            sctr: _,
            uctr,
            rounds,
            prof,
            ..
        } = sess;
        let Glm5DraftState::Dflash2 { kv, pending, taps } = state else {
            return Err("glm5_dflash_round_drafts on a native-mtp session".into());
        };
        let anchor = *anchor;

        // ---- 1. ingest pending committed feature rows (positions kv.len..) ----
        // Under the default placement the PROMPT rows were ingested at session creation
        // and only the kept verify rows of the previous round arrive here; under
        // `MEMRA_GLM5_DRAFT_PRIME_LAZY=1` round 1 carries the whole prompt (the pre-lane
        // literal, the round-0 wall boot A measured at depth).
        let row_w = taps.len() * n_embd;
        let mut pclk = pclk;
        let n_new = pending.len() / row_w;
        let st = if n_new > 0 {
            let rows = std::mem::take(pending);
            self.glm5_dflash_ingest_rows(eh, draft, kv, &rows, row_w, pclk.as_deref_mut())?
        } else {
            DraftIngestStats::default()
        };
        // FIRST-TOKEN PROFILE: the lazy arm's round-1 prompt ingest lands in the
        // drafter-prime buckets here (a creation-time ingest already wrote them and
        // leaves nothing prompt-sized pending); the clock is re-based either way.
        if let (Some(ck), Some(pf)) = (pclk, prof.as_mut()) {
            let tail = ck.lap(eh, eh);
            if *rounds == 0 && n_new > 0 && pf.draft_prime_arm.is_empty() {
                st.write(pf, "eager-lazy");
                pf.draft_prime_ms += tail;
            }
        }
        let start = cache.pos;
        debug_assert_eq!(
            kv.len, start,
            "drafter ctx rows must equal committed trunk rows at a round boundary"
        );

        // ---- 2. block forward over the cached ctx (decode-exact matmul scope: the m=8
        // drafter GEMMs otherwise fall into the prefill-GEMM class — the dspark round's
        // measured fix; RAII so a `?` exit never latches exact engine-wide) ----
        let exact_scope = eh.exact_scope(true);
        let mut block: Vec<u32> = vec![c.mask_token_id; b];
        block[0] = anchor;
        let noise = eh.htod(&self.embd.try_gather(n_embd, &block)?)?;
        let pos_block: Vec<i32> = ((start as i32)..(start + b) as i32).collect();
        let dh = draft.forward_round(eh, kv, &noise, &pos_block)?;

        // ---- 3. draft logits over the mask-fill harvest rows 1..b ----
        let nd = b - 1;
        let mut rows_buf = eh.uninit(nd * n_embd)?;
        {
            let dv = eh.view(&dh, b * n_embd);
            let tail = dv.slice(n_embd..b * n_embd);
            eh.copy_view_into(&mut rows_buf, 0, &tail, nd * n_embd)?;
        }
        // TRIMMED DRAFT HEAD: the dspark serve arm's resolution verbatim — the FR-Spec
        // self-trim the load path builds on the MTP struct (gathered rows of the target's
        // own head). Available only when the MTP struct loaded; the DFlash2-without-head
        // boot (the q38 VRAM pattern) runs the full target head, stated in the boot receipt.
        let trim = self
            .mtp
            .as_ref()
            .filter(|m| m.d2t_from_target_head)
            .and_then(|m| m.shared_head_head.as_ref().zip(m.d2t.as_ref()))
            .filter(|(_, d2t)| !d2t.is_empty());
        let (dl_head, dl_vocab) = match trim {
            Some((head, d2t)) => (head, d2t.len()),
            None => (&self.output, n_vocab),
        };
        // skip_d2t_remap red arm (the q38 defect made loud): candidates stay RANK ids.
        let trim_d2t = trim
            .filter(|_| !knobs.skip_d2t_remap)
            .map(|(_, d2t)| d2t.as_slice());
        let dl = eh.matmul(dl_head, &rows_buf, nd)?;

        // ---- 4. selector walk (greedy chain / sampled candidate-set walk) ----
        let (mut drafts, slot_q, qside) = match sp {
            None => {
                let (path, q) = draft
                    .dflash2_propose_greedy_q(eh, &dl, &rows_buf, nd, dl_vocab, anchor, trim_d2t)?;
                (path, q, Glm5DraftQ::None)
            }
            Some(sp) => {
                let (path, q_chosen, cand, q_rows) = draft.dflash2_propose_sampled(
                    eh, &dl, &rows_buf, nd, dl_vocab, anchor, sp.temp, sp.seed, uctr, trim_d2t,
                )?;
                let top_k = draft
                    .dflash2
                    .as_ref()
                    .ok_or("glm5 dflash drafter lost its DFlash2 head")?
                    .top_k;
                (
                    path,
                    q_chosen.clone(),
                    Glm5DraftQ::Selector {
                        prop: DsparkDraftSample::Selector {
                            cand,
                            q_rows,
                            q_chosen,
                            top_k,
                        },
                        dl,
                    },
                )
            }
        };
        drop(exact_scope);
        drafts.truncate(k);
        // TAU-SLOT CONFIDENCE TRUNCATION (loop-port 2): the low-confidence tail never
        // enters verify. Slot-indexed prefix reads keep the retained Selector q side
        // consistent (cand/q_rows/q_chosen are per-slot; the accept walk reads slots
        // 0..drafts.len()-1 only). p_min unset = today's rounds, untouched.
        if p_min > 0.0 {
            let kc = glm5_conf_keep(&slot_q[..drafts.len()], p_min, pmin0);
            drafts.truncate(kc);
        }
        if let Some(over) = knobs.draft_override.as_mut() {
            for (ki, d) in drafts.iter_mut().enumerate() {
                *d = over(*rounds, ki, *d);
            }
        }
        Ok((drafts, qside))
    }

    /// SAMPLED ACCEPT (module doc): the rejection-sampling walk `u_j * q_j(x_j) < p_j(x_j)`
    /// over the verify logit rows — memra's existing sampled spec contract (the
    /// MEMRA_SPEC_TEMP route / dspark sampled-admission walk), plugged in at exactly the
    /// accept seam; walk and rollback unchanged. p and q take the SAME filter transforms
    /// (`filter_stats` + `softmax_gather_filtered`, distribution-exact for the filtered
    /// target); the accept-test uniforms come from `spec::host_u01` on the session's `uctr`
    /// (tag 0xFFFF_FFFE) and every device draw (draft chain, full-accept bonus, residual
    /// resample) advances the session's `sctr` — counters persist on the session so
    /// randomness never repeats across bursts. Returns `(j, bonus)`. `e` is the HEAD
    /// engine (the round resolves it): the verify rows and retained draft logits live on
    /// the last stage under a split.
    #[allow(clippy::too_many_arguments)]
    // allow: the parameter list mirrors the accept seam's inputs (verify rows + the draft
    // chain's retained q side); bundling into a struct would hide the p/q pairing
    fn glm5_sampled_accept(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        sp: &SpecSampling,
        vlogits: &CudaSlice<f32>,
        drafts: &[u32],
        draft_idx: &[u32],
        draft_logits: &[CudaSlice<f32>],
        draft_stats: &[(f32, f32, f32)],
        d2t: Option<&[u32]>,
        k: usize,
    ) -> Res<(usize, u32)> {
        let n_vocab = self.output.out_features();
        let d_vocab = d2t.map(|m| m.len()).unwrap_or(n_vocab);
        // FILTERED p_j: one batched stats pass over verify rows 0..k-1 (row j is the target
        // distribution at draft j's slot), then one batched gather of the drafted tokens.
        let rows_i: Vec<i32> = (0..k as i32).collect();
        let rows_d = e.htod_i32(&rows_i)?;
        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(k)?, e.zeros(k)?, e.zeros(k)?);
        e.filter_stats(
            vlogits, n_vocab, &rows_d, &mut th_d, &mut z_d, &mut mx_d, n_vocab, k, sp.temp,
            sp.top_k, sp.top_p, sp.min_p,
        )?;
        let ids_d = e.htod_u32_v(drafts)?;
        let mut pj_d = e.zeros(k)?;
        e.softmax_gather_filtered(
            vlogits, n_vocab, &ids_d, &rows_d, &th_d, &z_d, &mut pj_d, n_vocab, k, sp.temp,
        )?;
        let pj = e.dtoh(&pj_d)?;
        let (thv, zv, mxv) = (e.dtoh(&th_d)?, e.dtoh(&z_d)?, e.dtoh(&mx_d)?);

        // The walk: FILTERED q_j from the retained draft logits (rank id for trimmed heads),
        // host Philox accept test per slot.
        let mut j = 0usize;
        while j < k {
            let (_qmx, qth, qz) = draft_stats[j];
            let idsd = e.htod_u32_v(&[draft_idx[j]])?;
            let rows0 = e.htod_i32(&[0])?;
            let thd = e.htod(&[qth])?;
            let zd = e.htod(&[qz])?;
            let mut outd = e.zeros(1)?;
            e.softmax_gather_filtered(
                &draft_logits[j],
                d_vocab,
                &idsd,
                &rows0,
                &thd,
                &zd,
                &mut outd,
                d_vocab,
                1,
                sp.temp,
            )?;
            let qj = e.dtoh(&outd)?[0];
            let u = crate::spec::host_u01(sp.seed, sess.uctr);
            sess.uctr = sess.uctr.wrapping_add(1);
            if (u as f64) * (qj as f64) < pj[j] as f64 {
                j += 1;
            } else {
                break;
            }
        }

        // Bonus: full accept draws a filtered Gumbel sample from the LAST verify row
        // (`glm5_sampled_bonus` — shared with the PMIN0 zero-draft round); rejection at j
        // resamples the residual norm(max(0, fp_j - fq_j)) — with a trimmed draft head, q
        // scatters back to full vocab first (`scatter_trim_logits`).
        if j == k {
            return Ok((
                j,
                self.glm5_sampled_bonus(e, sess, sp, vlogits, k, n_vocab)?,
            ));
        }
        let mut col = e.zeros(n_vocab)?;
        let bonus = {
            let vv = e.view(vlogits, (k + 1) * n_vocab);
            let row = vv.slice(j * n_vocab..(j + 1) * n_vocab);
            e.copy_view_into(&mut col, 0, &row, n_vocab)?;
            let p_stats = (mxv[j], thv[j], zv[j]);
            let q_stats = draft_stats[j];
            let sc = sess.sctr;
            sess.sctr = sess.sctr.wrapping_add(1);
            let mut sample_tok = e.alloc_u32_zeroed(1)?;
            match d2t {
                Some(map) => {
                    let map_d = e.htod_u32_v(map)?;
                    let mut q_full = e.zeros(n_vocab)?;
                    e.scatter_trim_logits(&draft_logits[j], &map_d, &mut q_full, d_vocab, n_vocab)?;
                    e.residual_sample_filtered(
                        &col,
                        Some(&q_full),
                        n_vocab,
                        sp.temp,
                        sp.seed,
                        sc,
                        p_stats,
                        q_stats,
                        &mut sample_tok,
                    )?;
                }
                None => {
                    e.residual_sample_filtered(
                        &col,
                        Some(&draft_logits[j]),
                        n_vocab,
                        sp.temp,
                        sp.seed,
                        sc,
                        p_stats,
                        q_stats,
                        &mut sample_tok,
                    )?;
                }
            }
            e.dtoh_u32(&sample_tok)?[0]
        };
        let bonus = crate::spec::guard_vocab_token(
            bonus,
            n_vocab,
            &format!("glm5 sampled verify bonus at round {} j={j}", sess.rounds),
        )?;
        Ok((j, bonus))
    }

    /// One filtered-Gumbel bonus draw from verify row `row` through the session's device
    /// Philox stream — the sampled FULL-ACCEPT bonus, and the entire accept of a PMIN0
    /// zero-draft round (whose verify batch is just the anchor row: m=1 = a plain sampled
    /// decode step). Advances `sctr` exactly once; byte-for-byte the pre-extraction
    /// full-accept arm of `glm5_sampled_accept`.
    fn glm5_sampled_bonus(
        &self,
        e: &Engine,
        sess: &mut Glm5SpecSession,
        sp: &SpecSampling,
        vlogits: &CudaSlice<f32>,
        row: usize,
        n_vocab: usize,
    ) -> Res<u32> {
        let mut col = e.zeros(n_vocab)?;
        let vv = e.view(vlogits, (row + 1) * n_vocab);
        let src = vv.slice(row * n_vocab..(row + 1) * n_vocab);
        e.copy_view_into(&mut col, 0, &src, n_vocab)?;
        let rows0 = e.htod_i32(&[0])?;
        let (mut bth, mut bz, mut bmx) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
        e.filter_stats(
            &col, n_vocab, &rows0, &mut bth, &mut bz, &mut bmx, n_vocab, 1, sp.temp, sp.top_k,
            sp.top_p, sp.min_p,
        )?;
        let (th, mx) = (e.dtoh(&bth)?[0], e.dtoh(&bmx)?[0]);
        let mut pb = e.zeros(n_vocab)?;
        e.gumbel_perturb_filtered(&col, &mut pb, n_vocab, sp.seed, sess.sctr, sp.temp, mx, th)?;
        sess.sctr = sess.sctr.wrapping_add(1);
        let td = e.argmax_token_device(&pb, n_vocab)?;
        crate::spec::guard_vocab_token(
            e.dtoh_u32_one(&td)?,
            n_vocab,
            &format!(
                "glm5 sampled verify bonus at round {} (row {row})",
                sess.rounds
            ),
        )
    }
}

/// One filtered Gumbel draw from a draft-head logit row through the session's device Philox
/// stream — the sampled route's PROPOSAL. Returns the drawn RANK id and the row's filtered
/// stats `(row_max, threshold_e, renorm_mass)`, which the accept walk's q gather and the
/// rejection residual both reuse (the q side must be the distribution the draft was actually
/// drawn from, or rejection sampling is not exact for the filtered target).
fn glm5_sampled_draft(
    e: &Engine,
    dl: &CudaSlice<f32>,
    d_vocab: usize,
    sp: &SpecSampling,
    sctr: &mut u32,
) -> Res<(u32, (f32, f32, f32))> {
    let rows0 = e.htod_i32(&[0])?;
    let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
    e.filter_stats(
        dl, d_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, d_vocab, 1, sp.temp, sp.top_k,
        sp.top_p, sp.min_p,
    )?;
    let (th, z, mx) = (e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0], e.dtoh(&mx_d)?[0]);
    let mut pb = e.zeros(d_vocab)?;
    e.gumbel_perturb_filtered(dl, &mut pb, d_vocab, sp.seed, *sctr, sp.temp, mx, th)?;
    *sctr = sctr.wrapping_add(1);
    let td = e.argmax_token_device(&pb, d_vocab)?;
    let idx =
        crate::spec::guard_vocab_token(e.dtoh_u32_one(&td)?, d_vocab, "glm5 sampled draft draw")?;
    Ok((idx, (mx, th, z)))
}

/// glm5_next SERVED speculative session (lane/glm5-spec-routing, 2026-08-30): the state one
/// request's spec decoding carries across worker bursts — the dspark/gemma session twins'
/// shape. The session OWNS its trunk cache (the worker's `s.cache` stays `None`); at every
/// burst boundary the invariant is `cache.pos == committed.len()` with each committed row's
/// trunk state exactly what a plain prime of that sequence would hold (the accept walk's
/// basis, pinned by the tparallel gate), plus ONE emitted-but-uncommitted `anchor` token
/// (the next round's verify row 0 — the dspark `last` convention).
pub struct Glm5SpecSession {
    cache: Cache,
    /// Every token whose trunk state the cache holds, in order (prompt + committed
    /// generation). EXCLUDES the live `anchor`.
    pub committed: Vec<u32>,
    /// The last emitted token, not yet consumed by the trunk — round anchor / verify row 0.
    anchor: u32,
    /// The prime's boundary token is emitted exactly once, by the first burst.
    anchor_emitted: bool,
    /// Committed `(token, h_seed)` pairs not yet fed to the MTP draft plane; the last
    /// feed's logits double as the next round's first draft (the re-warm contract).
    /// NATIVE-MTP arm only; the DFlash2 source keeps its own pending rows in `draft`.
    pending: Vec<(u32, CudaSlice<f32>)>,
    /// The session's pinned draft source + its state (module doc, DRAFT SOURCE SEAM).
    draft: Glm5DraftState,
    /// `None` / `temp <= 0` = greedy byte-contract route. Fixed for the session — the
    /// worker's admission owns the sampler identity.
    sampling: Option<SpecSampling>,
    /// Session-continuity Philox counters (never reset across bursts): `sctr` = device
    /// sampling events (boundary, draft chain, bonus, residual), `uctr` = host accept-test
    /// uniforms (`spec::host_u01`, tag 0xFFFF_FFFE).
    sctr: u32,
    uctr: u32,
    /// Verify rounds completed over the session lifetime (the worker's per-burst
    /// rounds-delta receipt, the dspark `rounds` convention).
    pub rounds: usize,
    done: bool,
    max_ctx: usize,
    /// MTP draft-plane layer index — `Some` on the native-MTP arm only.
    mtp_il: Option<usize>,
    /// Prompt-boundary capture for the DEFERRED prefix publication (lane/glm5-prefix-latent2,
    /// 2026-09-01; the dspark `prefix_capture` pattern): taken at session creation before any
    /// burst mutates the recurrent/tail state, drained by the worker's sweep. `None` when the
    /// worker did not request capture or when any boundary invariant refused.
    prefix_capture: Option<crate::spec::SpecBoundaryCapture>,
    /// FIRST-TOKEN PROFILE (lane/b200-spec-ttft-20260902): `Some` iff `MEMRA_SPEC_PROF=1`
    /// at creation; filled through session creation and round 1 of the first burst, then
    /// drained by the worker's one `[spec-prof]` line (`take_first_token_prof`).
    prof: Option<SpecFirstTokenProf>,
    /// DEPTH LOG (lane/spec-route-depth-20260902): the first `SPEC_PROF_ROUNDS` rounds'
    /// attribution rows, `Some` iff `MEMRA_SPEC_PROF=1`; the worker prints fresh rows after
    /// every burst and the summary once the log fills or the session ends.
    prof_rounds: Option<SpecRoundsLog>,
}

impl Glm5SpecSession {
    /// The depth log, for the worker's `[spec-prof-rounds]` / `[spec-prof-summary]` lines.
    pub fn round_log_mut(&mut self) -> Option<&mut SpecRoundsLog> {
        self.prof_rounds.as_mut()
    }
    /// Rounds the depth log keeps (the print cadence's cap).
    pub fn round_log_cap() -> usize {
        SPEC_PROF_ROUNDS
    }
    /// Drafter ctx rows currently ingested (`None` on the native arm).
    pub fn draft_kv_len(&self) -> Option<usize> {
        match &self.draft {
            Glm5DraftState::Dflash2 { kv, .. } => Some(kv.len),
            _ => None,
        }
    }
    /// GATE ACCESSOR (rig gate 15): the first `rows` rows of every drafter K and V plane,
    /// host-side, `row_floats` = n_kv * head_dim. `None` on the native arm.
    #[allow(clippy::type_complexity)]
    // allow: (k planes, v planes) is the natural shape for a bit-identity diff
    pub fn draft_kv_rows_host(
        &self,
        e: &Engine,
        rows: usize,
        row_floats: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let Glm5DraftState::Dflash2 { kv, .. } = &self.draft else {
            return None;
        };
        let n = rows * row_floats;
        let take = |planes: &[CudaSlice<f32>]| -> Option<Vec<Vec<f32>>> {
            planes
                .iter()
                .map(|p| e.dtoh_view(&p.slice(0..n)).ok())
                .collect()
        };
        Some((take(&kv.k)?, take(&kv.v)?))
    }
    /// Context capacity of the session's cache (the server's ContextFull guard).
    pub fn cache_max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// Drain the first-token profile (doc on the field). `None` with the profile off or
    /// once drained — the worker prints exactly one line per request.
    pub fn take_first_token_prof(&mut self) -> Option<SpecFirstTokenProf> {
        self.prof.take()
    }
    /// Drain the prompt-boundary capture (doc on the field; the dspark
    /// `take_prefix_capture` twin — the worker publishes it against `cache_ref`).
    pub fn take_prefix_capture(&mut self) -> Option<crate::spec::SpecBoundaryCapture> {
        self.prefix_capture.take()
    }
    /// True when the deferred prefix capture can publish NOW: a capture exists AND the
    /// DFlash2 drafter KV already covers the boundary. glm5 defers the prompt's feature
    /// ingest to round 1 (unlike dspark's at-creation ingest), so a drain that fired before
    /// the first burst would export an empty tail and waste the capture — the worker's
    /// sweep polls this instead.
    pub fn prefix_capture_ready(&self) -> bool {
        self.prefix_capture
            .as_ref()
            .is_some_and(|c| match &self.draft {
                Glm5DraftState::Dflash2 { kv, .. } => kv.len >= c.pos,
                _ => false,
            })
    }
    /// The drafter's readable KV tail at `upto` rows (the dspark `export_tail` seam) —
    /// `None` on the native arm or when the tail cannot cover the drafter window.
    pub fn export_draft_tail(
        &self,
        e: &Engine,
        upto: usize,
    ) -> Option<crate::dflash::DflashKvTail> {
        match &self.draft {
            Glm5DraftState::Dflash2 { kv, .. } => kv.export_tail(e, upto),
            _ => None,
        }
    }
    /// The session's trunk cache, for the worker's deferred prefix publication (the
    /// append-only-below-boundary slices) — never for mutation.
    pub fn cache_ref(&self) -> &Cache {
        &self.cache
    }
    /// Trunk rows currently committed (== `committed.len()` at burst boundaries).
    pub fn pos(&self) -> usize {
        self.cache.pos
    }
    /// EOS committed or the context guard tripped: the next burst would emit nothing.
    pub fn finished(&self) -> bool {
        self.done
    }
    /// True when the session is a legal demotion source (loop-port fold-in, map #8):
    /// GREEDY only — a sampled session's committed stream depends on its session-owned
    /// Philox counters, and the plain batched sampler is a different random program
    /// mid-request (the exact exclusion the MTP and dspark sweeps carry).
    pub fn demote_eligible(&self) -> bool {
        self.sampling.is_none()
    }
}

impl HybridModel {
    /// ONE-WAY DEMOTION HANDOFF for the glm5 session (loop-port fold-in — the map's #8,
    /// the `SpecSession::into_demoted` / `DsparkSpecSession::into_demoted` twin): consume
    /// the session and hand `(cache, next_pred)` to the plain batched-decode path, so a
    /// spec session admitted on a quiet box stops serializing the tick when load arrives
    /// (the spec-gate HIGH sweep's ship-safety lever; dspark receipt: "c=8 429.6 = parity
    /// (pre-lane -37%)").
    ///
    /// THE ANCHOR IS THE CARRIED-PENDING SHAPE: glm5 emits each round's bonus immediately
    /// (`round_tokens` include it) while the trunk consumes it only as the NEXT round's
    /// row 0 — so at every burst boundary the session holds ONE emitted-but-uncommitted
    /// token. Handing the cache over as-is would leave it one row short of the public
    /// stream, and `device_next` re-emitting the anchor would duplicate a served token.
    /// The flush below is `spec_flush_pending`'s exact analogue: ONE plain T=1 decode
    /// step commits the anchor (byte-identical to the never-drafted chain — the
    /// tparallel gate's accept-j-then-continue identity IS this claim), and its argmax
    /// becomes the handoff's `next_pred` — a token the batched path emits and feeds
    /// exactly as it would its own. One trunk pass, once per demotion, never per burst.
    ///
    /// ONE-WAY BY DESIGN: the draft state (MTP pending pairs / DFlash2 drafter KV and
    /// feature rows) and the Philox counters are DROPPED, freeing their VRAM; there is
    /// no cheap symmetric re-promotion (the spec.rs law, verbatim). Sampled sessions
    /// refuse loudly (`demote_eligible`; the worker's sweep excludes them first).
    pub fn glm5_spec_into_demoted(
        &self,
        e: &Engine,
        mut sess: Glm5SpecSession,
    ) -> Res<(Cache, u32)> {
        if !sess.demote_eligible() {
            return Err(
                "glm5 demote: sampled sessions stay on spec until they end (session-owned \
                 Philox vs the worker sampler is an unmeasured distributional seam — the \
                 MTP sweep's exclusion, verbatim)"
                    .into(),
            );
        }
        if sess.cache.pos + 1 > sess.max_ctx {
            return Err(format!(
                "glm5 demote: no room to flush the live anchor ({} + 1 > ctx {})",
                sess.cache.pos, sess.max_ctx
            )
            .into());
        }
        let logits = self.decode_step(e, sess.anchor, &mut sess.cache)?;
        sess.committed.push(sess.anchor);
        let next = argmax(&logits) as u32;
        Ok((sess.cache, next))
    }
}

/// Buckets of one eager drafter ingest (`glm5_dflash_ingest_rows`), written into the
/// first-token profile under the arm name that ran it.
#[derive(Default, Debug, Clone, Copy)]
struct DraftIngestStats {
    h2d_ms: f64,
    feat_ms: f64,
    kv_ms: f64,
    rows: usize,
    chunks: usize,
}

impl DraftIngestStats {
    fn write(&self, pf: &mut SpecFirstTokenProf, arm: &'static str) {
        pf.draft_prime_ms = self.h2d_ms + self.feat_ms + self.kv_ms;
        pf.draft_prime_h2d_ms = self.h2d_ms;
        pf.draft_prime_feat_ms = self.feat_ms;
        pf.draft_prime_kv_ms = self.kv_ms;
        pf.draft_prime_rows = self.rows;
        pf.draft_prime_chunks = self.chunks;
        pf.draft_prime_arm = arm;
    }
}

/// Gate instruments for `generate_spec_glm5_gated`. Documented as instruments: no serving
/// path constructs a non-default value.
#[derive(Default)]
pub struct Glm5SpecKnobs<'a> {
    /// `(round, draft_index, greedy_draft) -> draft` — deterministic forced-accept /
    /// forced-reject rounds for the end-to-end gate.
    pub draft_override: Option<&'a mut dyn FnMut(usize, usize, u32) -> u32>,
    /// RED ARM ONLY: skip the state rollback (pos still moves). A corrupted draft must then
    /// leave post-row-K KDA state and un-truncated latent rows behind — the end-to-end gate
    /// asserts the tape DIVERGES from plain decode (or the kpool residency tripwire fires).
    pub disable_rollback: bool,
    /// RED ARM ONLY: with an FR-Spec trim loaded, use the draft argmax RANK id as the vocab
    /// id (the q38 skipped-remap defect: 0/248 acceptance with every exactness gate green).
    /// The gate asserts the drafted sequence diverges from the untrimmed arm's while the
    /// output tape STAYS byte-identical to plain decode — the silent failure made loud.
    pub skip_d2t_remap: bool,
    /// GATE INSTRUMENT for the confidence gate (loop-port 2): `Some((p_min, pmin0))`
    /// overrides the `MEMRA_SPEC_PMIN`/`MEMRA_SPEC_PMIN0` env pair for this call — the
    /// env statics latch once per process, so the byte-identity gate drives its PMIN
    /// arms through here instead of the environment. `None` = the serving resolution.
    pub pmin_override: Option<(f32, bool)>,
    /// GATE INSTRUMENT (lane/glm5-accrace): trace every GREEDY round's accept decision to
    /// stderr as one `[glm5-accrace]` line — round, t, j, the drafts, the DEVICE argmax
    /// row (`argmax_token_device_col` + the one u32 readback the accept walk consumes), a
    /// HOST argmax over the same `vlogits` buffer, and a per-row (argmax, FNV-1a hash of
    /// the row's f32 bits) census.
    ///
    /// TWO THINGS IT SEPARATES, which is why it exists: (a) `dev != host` means the device
    /// accept path published a value the logits buffer does not justify (a readback/scratch
    /// race); (b) `dev == host` with a row hash that moves between two runs of the same
    /// deterministic fixture means the verify logits themselves were computed over
    /// corrupted state (an upstream walk/rollback race). The host read is issued AFTER the
    /// device path's own `dtoh_u32` has already synchronized the consuming stream, so the
    /// probe can only observe the race, never mask it.
    ///
    /// Never a serving surface: no serving path constructs a non-default value.
    pub accept_probe: bool,
}

#[cfg(test)]
mod conf_keep_tests {
    use super::glm5_conf_keep;

    /// The spec.rs chain-break semantics, pinned CPU-side (loop-port 2): break at the
    /// first sub-threshold slot; slot 0 survives a miss unless PMIN0.
    #[test]
    fn conf_keep_matches_the_spec_rs_break_semantics() {
        // Gate off: everything kept.
        assert_eq!(glm5_conf_keep(&[0.1, 0.1], 0.0, true), 2);
        // All confident: everything kept.
        assert_eq!(glm5_conf_keep(&[0.9, 0.8, 0.7], 0.5, false), 3);
        // Break mid-chain at the first miss; the confident tail after it never rides
        // (prefix truncation — the accept rule could never commit past the gap anyway).
        assert_eq!(glm5_conf_keep(&[0.9, 0.2, 0.9], 0.5, false), 1);
        assert_eq!(glm5_conf_keep(&[0.9, 0.2, 0.9], 0.5, true), 1);
        // Slot-0 miss: survives without PMIN0 (the j > 0 arm of the break condition), and
        // does NOT latch — slot 1 is judged on its own confidence (the spec.rs chain
        // evaluates each slot's p independently)...
        assert_eq!(glm5_conf_keep(&[0.2, 0.9], 0.5, false), 2);
        // ...but a sub-threshold slot past 0 still breaks.
        assert_eq!(glm5_conf_keep(&[0.2, 0.2], 0.5, false), 1);
        // PMIN0 arms the zero-draft round.
        assert_eq!(glm5_conf_keep(&[0.2, 0.9], 0.5, true), 0);
        // Boundary: q == p_min is NOT below it (strict <, the spec.rs test).
        assert_eq!(glm5_conf_keep(&[0.5, 0.5], 0.5, true), 2);
        // Empty chain: nothing to keep.
        assert_eq!(glm5_conf_keep(&[], 0.5, true), 0);
    }
}
