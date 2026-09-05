//! DFlash block-diffusion drafter (DFLASH-BRINGUP-PLAN.md, 2026-07-13).
//!
//! 5-layer qwen3-class mini-transformer that drafts a 16-token block in ONE non-causal
//! forward, conditioned on the TARGET's hidden states at 6 tapped layers (concatenated
//! through `fc` + `hidden_norm`). No embed / lm_head of its own — the round reuses the
//! target's. Reference: z-lab/dflash `dflash/model.py` (semantics frozen in the plan doc);
//! oracle: tools/dflash_oracle.py -> /data/cache/dflash-oracle.npz.
//!
//! FIRST LIGHT = f32-resident weights + fresh full-context forward (no draft KV cache) —
//! correctness vs the oracle, then the cache/quant/window arms land measurement-gated.

use crate::Engine;
use crate::model::GpuTensor;
use cudarc::driver::CudaSlice;

pub struct DflashCfg {
    pub hidden: usize,                // 5376
    pub n_head: usize,                // 64
    pub n_kv: usize,                  // 8
    pub head_dim: usize,              // 128
    pub n_ff: usize,                  // 10752
    pub n_layer: usize,               // 5
    pub eps: f32,                     // 1e-6
    pub rope_theta: f32,              // 1e6
    pub block_size: usize,            // 16
    pub mask_token_id: u32,           // 4
    pub target_layer_ids: Vec<usize>, // [1,12,23,35,46,57]
    pub sliding_window: usize,        // 2048
    /// true = sliding_attention for that layer (4x true + 1x false on the 31B draft).
    pub layer_sliding: Vec<bool>,
    /// Checkpoint training-strategy census (`dspark_strategy_census` over the raw
    /// config.json): true = a SpecForge DSPARK-strategy export (shifted labels, ALL rows
    /// supervised — the q38 arm-a family). Keys the HARVEST DEFAULT strategy-keyed,
    /// never env-keyed (owner-ratified 2026-08-20 after B1 confirmed H1 ×5;
    /// DSPARK-POSTMORTEM-20260820.md B0 default-flip plan).
    pub strategy_dspark: bool,
    /// Explicit top-level `is_causal` from config.json (z-lab reference: an explicit
    /// value OVERRIDES the per-layer-type default). The DFlash2 q38 checkpoint carries
    /// `"is_causal": false` — every sliding layer is NON-causal with a symmetric
    /// +/-2048 window (model.py `_attention_mask`). None = key absent (historical
    /// exports; the windowless-assert arm keeps handling those byte-identically).
    pub is_causal: Option<bool>,
}

pub struct DflashLayer {
    pub wq: GpuTensor,           // [nh*hd, hidden] row-major (out_f rows)
    pub wk: GpuTensor,           // [nkv*hd, hidden]
    pub wv: GpuTensor,           // [nkv*hd, hidden]
    pub wo: GpuTensor,           // [hidden, nh*hd]
    pub w_gate: GpuTensor,       // [n_ff, hidden]
    pub w_up: GpuTensor,         // [n_ff, hidden]
    pub w_down: GpuTensor,       // [hidden, n_ff]
    pub ln_in: CudaSlice<f32>,   // [hidden]
    pub ln_post: CudaSlice<f32>, // [hidden]
    pub q_norm: CudaSlice<f32>,  // [hd]
    pub k_norm: CudaSlice<f32>,  // [hd]
}

pub struct DflashDraft {
    pub cfg: DflashCfg,
    pub layers: Vec<DflashLayer>,
    pub fc: GpuTensor,               // [hidden, n_taps*hidden]
    pub hidden_norm: CudaSlice<f32>, // [hidden]
    pub norm: CudaSlice<f32>,        // [hidden]
    /// DSpark semi-AR markov head (present in the repo-root checkpoint variant):
    /// draft logits at position k get + W2(W1[prev_realized_token]) — left-to-right
    /// within the block (the patch's _markov_semiar_sample_block semantics, greedy).
    /// w1 = raw bf16 [V, rank] (row-gathered by device token id); w2 = q8_0 [rank->V].
    pub markov: Option<MarkovHead>,
    /// DSpark accept-rate head (trained with confidence loss). sglang's DSPARK planner
    /// consumes it to SIZE VERIFY WINDOWS (cumprod survival — v0.5.16 headline; the
    /// earlier "reference serving loop never consumes it" note matched SpecForge's
    /// legacy spec_generate only). memra schedules with it under
    /// `MEMRA_DSPARK_VT=confidence` (the H4 fix, DSPARK-POSTMORTEM-20260820.md:
    /// per-round verify window from cumprod survival, `dspark_confidence_vt`) and
    /// keeps it census+parity-only under the default ladder. Host-resident (5k floats).
    pub confidence: Option<ConfidenceHead>,
    /// YaRN rope (q38 arm-a inherits the target's rope_parameters: rope_type yarn,
    /// factor 32, original 8192, beta 32/1). ff = per-dim divisors for rope_neox_ff
    /// (effective inv_freq_j = base^(-2j/d)/ff[j] = the HF-yarn remapped frequency,
    /// verified vs Qwen3RotaryEmbedding to 1.6e-7), mscale = attention_scaling
    /// (0.1*ln(factor)+1) applied to q/k post-rope — cos/sin scaling distributes onto
    /// the rotated vector exactly. None = plain rope (gemma/z-lab drafters).
    pub rope_yarn: Option<(CudaSlice<f32>, f32)>,
    /// DFlash2 head (z-lab `DFlash2DraftModel`, DFLASH2-EVAL-20260820.md): grouped
    /// dynamic causal convs around EVERY sublayer + the candidate path selector that
    /// replaces the markov chain. A DISTINCT semantic program from the DSpark head
    /// (no-generic-support law): present iff config `architectures` names
    /// `DFlash2DraftModel`, and then ALL 23 family tensors are REQUIRED — loading the
    /// 58 backbone tensors alone computes an untrained model (the census trap).
    pub dflash2: Option<Dflash2Head>,
}

/// One `GroupedDynamicCausalConv` module (reference model.py): a causal 2-tap
/// depthwise conv over the BLOCK rows (block-local — row 0 zero-pads its missing
/// predecessor; stateless across rounds), with per-position dynamic per-group
/// coefficients projected from the module INPUT. `prepare` convolves the sublayer
/// input with base_kernel[0] + dyn half 0; `finish` convolves the sublayer OUTPUT
/// with base_kernel[1] + dyn half 1 (both dyn halves come from the SAME projection
/// of the pre-conv input).
pub struct Dflash2Conv {
    /// base_kernel [2, k, hidden] flattened f32 (half-major: prepare then finish).
    pub base: CudaSlice<f32>,
    /// kernel_projection.weight [2*k*groups, hidden] (row layout = view(2, k, groups)).
    pub proj: GpuTensor,
}

pub struct Dflash2Head {
    pub attn_conv: Vec<Dflash2Conv>, // per layer
    pub mlp_conv: Vec<Dflash2Conv>,  // per layer
    /// candidate_selector.hidden_projection.weight [rank, hidden].
    pub hidden_proj: GpuTensor,
    /// Codebooks [V, rank] raw bf16, HOST-resident: the walk gathers ~1+16 rows per
    /// draft slot (~70KB/round) — host math beside the round's existing chain dtoh,
    /// no device residency for 2x127MB tables. Checkpoint quirk: stored WITHOUT the
    /// `.weight` suffix (reference from_pretrained installs a key_mapping).
    pub pred_codebook: Vec<u8>,
    pub succ_codebook: Vec<u8>,
    pub rank: usize,       // selector_rank 256
    pub top_k: usize,      // selector_top_k 16
    pub conv_k: usize,     // conv_kernel_size 2
    pub group_size: usize, // conv_group_size 16
    pub vocab: usize,      // codebook rows (248320)
}

/// Resolve the named DFlash weight program. Keep this separate from loading so a typo cannot
/// silently select q8 and invalidate a performance/default receipt.
fn dflash_precision(raw: Option<&str>) -> Result<&str, String> {
    let prec = raw.unwrap_or("q4");
    match prec {
        "q4" | "q8" | "mixed" | "bf16" | "fc" => Ok(prec),
        other => Err(format!(
            "MEMRA_DFLASH_PREC={other:?}: want q4, q8, mixed, bf16, or fc \
             (q5 was measured defective and is not a serving mode)"
        )),
    }
}

/// One bf16 codebook row -> f32 (exact widening).
fn cb_row(cb: &[u8], tok: usize, rank: usize) -> Vec<f32> {
    bf16_to_f32(&cb[tok * rank * 2..(tok + 1) * rank * 2])
}

/// The per-row top-k selector's EXHAUSTED-SLOT sentinel. `topk_rows_f32` and its sharded twin
/// (`cu/kernels.cu`) fill a slot they could not fill with `0xffffffff`: a row with fewer than
/// `k` FINITE values — in practice an all-NaN logits row, since every comparison against NaN
/// is false and a NaN therefore never enters a candidate list.
pub const TOPK_EMPTY_SLOT: u32 = u32::MAX;

/// Refuse a candidate buffer the selector could not fill (memra#95).
///
/// A sentinel row reaching the walk means the DRAFT LOGITS were not a number, which is a
/// broken invariant upstream (memra#95: a restored full-cover glm5 session read its drafter
/// ctx KV on a pp stage stream before the caller's import had landed). Two reasons this is a
/// guard at the proposal seam and not a clamp:
///
/// * clamping would silently draft a wrong token, and the drafts feed the verify batch — a
///   worse outcome than any failure, per the exactness law;
/// * the walk's own `assert!(c < vocab)` is the last line of defence inside a PURE function,
///   and it kills the GPU WORKER THREAD, which is fleet-fatal (one respawn, then
///   the process exits and every in-flight session on the box dies). Returning `Err` here
///   fails ONE request instead, the same trade `crate::spec::guard_vocab_token` makes for the
///   native-MTP chain's sentinel (memra#87). The assert stays where it is.
///
/// `bound` is the selector's own column space: the `n_vocab` the caller handed `topk_rows`,
/// which on a trimmed FR-Spec head is the trim's rank count and on a full head is the target
/// vocabulary. Checked BEFORE any d2t remap, because remapping a sentinel would index the map
/// out of bounds and panic before the walk is ever reached; the map is pre-checked to cover
/// `n_vocab`, so passing this bound also proves the remap safe.
///
/// ASSUMES UNMASKED DRAFT LOGITS, which is what both proposal seams feed it today: `dl` is a
/// raw `matmul` output and constrained requests never take the spec route, so a row cannot
/// legitimately hold fewer than `top_k` finite values. A future masked-draft-logits arm would
/// have to revisit this, and would want a shorter round rather than a refusal.
///
/// NOTE THE SCOPE: both proposal seams are shared with the LIVE dspark/q38 serve arm, not
/// only the flagged glm5 restore. This changes that route's failure mode too, from a
/// fleet-fatal worker panic to one refused request. That is the intended direction, and it
/// is the only behaviour change outside the default-OFF flag.
pub fn dflash2_guard_candidates(cand: &[u32], bound: usize, ctx: &str) -> Result<(), String> {
    for (i, &c) in cand.iter().enumerate() {
        if c as usize >= bound {
            let why = if c == TOPK_EMPTY_SLOT {
                " (the top-k selector's exhausted-slot sentinel: the draft logits row carried \
                   fewer than top_k finite values, i.e. it was NaN)"
            } else {
                ""
            };
            return Err(format!(
                "{ctx}: DFlash2 candidate slot {i} is {c}, outside the selector's {bound} \
                 columns{why}"
            ));
        }
    }
    Ok(())
}

/// Greedy selector walk (reference `CandidateSelector.select` at T=0): per draft
/// slot p, score(k) = unary[p,k] + <pred_codebook[prev] .* hidden_proj_row[p],
/// succ_codebook[cand[p,k]]>, argmax over the top-k candidate set; the CHOSEN
/// candidate seeds the next slot (sequential — the chain is the semantics, not an
/// optimization). Host math (~nd*k*rank fused ops per round) over host-resident bf16
/// codebooks; ties break to the LOWEST k (torch argmax convention). Pure so the
/// selector semantics are CPU-gateable.
///
/// `unary`/`cand`: [nd, top_k] row-major; `hproj`: [nd, rank] row-major.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_walk_greedy(
    pred_codebook: &[u8],
    succ_codebook: &[u8],
    vocab: usize,
    rank: usize,
    top_k: usize,
    unary: &[f32],
    cand: &[u32],
    hproj: &[f32],
    anchor: u32,
    nd: usize,
) -> Vec<u32> {
    dflash2_walk_greedy_q(
        pred_codebook,
        succ_codebook,
        vocab,
        rank,
        top_k,
        unary,
        cand,
        hproj,
        anchor,
        nd,
    )
    .0
}

/// [`dflash2_walk_greedy`] with the per-slot CONFIDENCE recorded (lane/glm5-loop-port,
/// 2026-08-30): q[p] = softmax over the slot's candidate-set scores at T=1, of the chosen
/// candidate — the greedy twin of `dflash2_walk_sampled`'s recorded `q_chosen` (same
/// statistic family the owner's "take only high confidence offers" tau gate thresholds on
/// the dspark route). The argmax selection is UNCHANGED (q is bookkeeping over the same
/// scores, ~top_k exps per slot on host), so every existing greedy caller is byte-identical
/// through the delegating wrapper. Pure, CPU-gateable like its siblings.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_walk_greedy_q(
    pred_codebook: &[u8],
    succ_codebook: &[u8],
    vocab: usize,
    rank: usize,
    top_k: usize,
    unary: &[f32],
    cand: &[u32],
    hproj: &[f32],
    anchor: u32,
    nd: usize,
) -> (Vec<u32>, Vec<f32>) {
    let (kk, r) = (top_k, rank);
    assert_eq!(unary.len(), nd * kk, "walk: unary shape");
    assert_eq!(cand.len(), nd * kk, "walk: candidate shape");
    assert_eq!(hproj.len(), nd * r, "walk: hidden-projection shape");
    let mut path = Vec::with_capacity(nd);
    let mut q_chosen = Vec::with_capacity(nd);
    let mut prev = anchor;
    for p in 0..nd {
        assert!(
            (prev as usize) < vocab,
            "walk: predecessor token {prev} outside codebook vocab {vocab}"
        );
        let pr = cb_row(pred_codebook, prev as usize, r);
        let hp = &hproj[p * r..(p + 1) * r];
        // gate = pred_row .* hidden_proj (shared across the candidate set)
        let gate: Vec<f32> = pr.iter().zip(hp).map(|(a, b)| a * b).collect();
        let mut scores = vec![0f32; kk];
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (k, s) in scores.iter_mut().enumerate() {
            let c = cand[p * kk + k] as usize;
            assert!(c < vocab, "walk: candidate {c} outside codebook vocab");
            let sr = cb_row(succ_codebook, c, r);
            let mut acc = unary[p * kk + k];
            for j in 0..r {
                acc += gate[j] * sr[j];
            }
            *s = acc;
            if acc > best {
                best = acc;
                bi = k;
            }
        }
        // Recorded confidence: softmax at T=1 over the candidate set (f64 accumulation,
        // the sampled walk's numeric discipline), of the argmaxed candidate.
        let mut z = 0f64;
        for &s in &scores {
            z += ((s - best) as f64).exp();
        }
        q_chosen.push(if z > 0.0 { (1.0 / z) as f32 } else { 1.0 });
        prev = cand[p * kk + bi];
        path.push(prev);
    }
    (path, q_chosen)
}

impl Dflash2Head {
    /// Greedy selector walk over this head's codebooks — see `dflash2_walk_greedy`.
    pub fn walk_greedy(
        &self,
        unary: &[f32],
        cand: &[u32],
        hproj: &[f32],
        anchor: u32,
        nd: usize,
    ) -> Vec<u32> {
        dflash2_walk_greedy(
            &self.pred_codebook,
            &self.succ_codebook,
            self.vocab,
            self.rank,
            self.top_k,
            unary,
            cand,
            hproj,
            anchor,
            nd,
        )
    }

    /// Greedy walk with the per-slot confidence recorded — see `dflash2_walk_greedy_q`.
    pub fn walk_greedy_q(
        &self,
        unary: &[f32],
        cand: &[u32],
        hproj: &[f32],
        anchor: u32,
        nd: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        dflash2_walk_greedy_q(
            &self.pred_codebook,
            &self.succ_codebook,
            self.vocab,
            self.rank,
            self.top_k,
            unary,
            cand,
            hproj,
            anchor,
            nd,
        )
    }

    /// Sampled (T>0) selector walk — see `dflash2_walk_sampled`.
    #[allow(clippy::too_many_arguments)]
    pub fn walk_sampled(
        &self,
        unary: &[f32],
        cand: &[u32],
        hproj: &[f32],
        anchor: u32,
        nd: usize,
        temp: f32,
        uniforms: &mut dyn FnMut() -> f32,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        dflash2_walk_sampled(
            &self.pred_codebook,
            &self.succ_codebook,
            self.vocab,
            self.rank,
            self.top_k,
            unary,
            cand,
            hproj,
            anchor,
            nd,
            temp,
            uniforms,
        )
    }
}

/// AcceptRatePredictor: raw linear proj over [hidden ; markov_prev_embedding(rank)]
/// (with_markov=true on the q38 arm-a export) — output is the PRE-sigmoid scalar.
pub struct ConfidenceHead {
    pub w: Vec<f32>, // [in_dim]
    pub b: f32,
    pub in_dim: usize,
    pub with_markov: bool,
}

impl ConfidenceHead {
    /// Host dot: the PRE-sigmoid accept score for one draft slot. `hidden` = the
    /// drafter output row the slot is harvested from (the same row its logits use);
    /// `emb` = the markov `w1` row of the slot's PREVIOUS chain token (required iff
    /// `with_markov`) — the exact input contract the parity gate pins (prev ids =
    /// `[anchor, chain[..nd-1]]`, dspark_q38_parity.rs stage 5).
    pub fn raw_score(&self, hidden: &[f32], emb: Option<&[f32]>) -> f32 {
        let mut acc = self.b;
        for (w, x) in self.w.iter().zip(hidden) {
            acc += w * x;
        }
        if self.with_markov {
            let emb = emb.expect("with_markov confidence head scored without the markov embedding");
            debug_assert_eq!(hidden.len() + emb.len(), self.in_dim);
            for (w, x) in self.w[hidden.len()..].iter().zip(emb) {
                acc += w * x;
            }
        } else {
            debug_assert_eq!(hidden.len(), self.in_dim);
        }
        acc
    }
}

pub struct MarkovHead {
    pub w1_bf16: CudaSlice<u8>, // [V, rank] bf16 raw
    pub w2: GpuTensor,          // [rank -> V] q8_0
    pub rank: usize,
    pub vocab: usize,
}

/// Draft-row harvest convention for DFlash-family block drafters
/// (darklanes research/deepseek-flash-20260818/DSPARK-POSTMORTEM-20260820.md).
///
/// The DFlash and DSpark SpecForge training strategies supervise DIFFERENT rows of the
/// same `[anchor, MASK x b-1]` block, so the row -> trunk-position mapping is a property
/// of the CHECKPOINT's training strategy, not of the loader:
///
/// - **Dflash** (mask-fill; z-lab dflash / SpecForge `OnlineDFlashModel`): row k is
///   trained to predict the token AT position anchor+k — "Labels: same-position
///   prediction", `weight_mask *= (pos_in_block > 0)` excludes the anchor row
///   (SpecForge `specforge/algorithms/common/dflash_family_model.py:453-472`).
///   Drafts = rows 1..b-1; the anchor row's output is untrained.
/// - **Dspark** (shifted; SpecForge `OnlineDSparkModel`, `training.strategy: dspark` —
///   the q38 arm-a export): row k is trained to predict the token at anchor+k+1, ALL
///   rows supervised INCLUDING the anchor row (`label_offsets = arange(1,
///   block_size+1)`, `dflash_family_model.py:816`). sglang's DSPARK worker — the stack
///   every arm-a bank number was measured on — harvests gamma = block_size drafts with
///   the anchor row's output as draft 1 (verified on the v0.5.17 eval-pin tag:
///   `dspark_components/dspark_draft.py:248,260,318`; `dspark_config.py:269`).
///
/// Mismatching the convention verifies every slot against a position the row was never
/// trained for — the q38 accept collapse (2.9 -> 1.43) in the postmortem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DsparkHarvest {
    /// mask-fill: drafts = rows 1..b-1, row k fills position anchor+k.
    Dflash,
    /// shifted: drafts = rows 0..b-1, row k predicts position anchor+k+1.
    Dspark,
}

impl DsparkHarvest {
    /// The served resolution: explicit `MEMRA_DSPARK_HARVEST={dflash|dspark}` wins
    /// (unknown values REFUSE loudly — a typo silently reverting the convention would
    /// re-open the postmortem's misalignment); UNSET defers to the CHECKPOINT's own
    /// training-strategy census — the owner-ratified default flip (2026-08-20, after
    /// B1 confirmed H1 interleaved ×5 on serving-class hardware: accept 1.38→2.41
    /// agentic / 1.53→3.66 math, E2E ALL EXACT both arms). Strategy-keyed, not
    /// env-keyed, per the B0 plan: a DSPARK-strategy export harvests shifted
    /// (all-rows), a mask-fill export keeps the historical dflash arm byte-identical.
    pub fn resolve(cfg: &DflashCfg) -> Self {
        Self::resolve_value(
            std::env::var("MEMRA_DSPARK_HARVEST").ok().as_deref(),
            cfg.strategy_dspark,
        )
    }

    pub fn resolve_value(v: Option<&str>, strategy_dspark: bool) -> Self {
        match v {
            None | Some("") => {
                if strategy_dspark {
                    DsparkHarvest::Dspark
                } else {
                    DsparkHarvest::Dflash
                }
            }
            set => Self::from_env_value(set),
        }
    }

    /// ENV-ONLY parser (no checkpoint census): unset = `Dflash`, the historical arm.
    /// Kept for the explicit-value path of [`Self::resolve_value`] and the seam tests;
    /// round arms resolve through [`Self::resolve`] so the default stays strategy-keyed.
    pub fn from_env_value(v: Option<&str>) -> Self {
        match v {
            None | Some("") | Some("dflash") => DsparkHarvest::Dflash,
            Some("dspark") => DsparkHarvest::Dspark,
            Some(other) => panic!(
                "MEMRA_DSPARK_HARVEST={other}: unknown harvest convention (dflash|dspark); \
                 refusing — a wrong convention verifies every draft slot against a position \
                 the drafter row was not trained for (DSPARK-POSTMORTEM-20260820.md)"
            ),
        }
    }

    /// Resolve the harvest convention for a LOADED drafter — FAMILY-keyed first, then
    /// STRATEGY-keyed (v0.100 train merge of the two ratified keyings):
    /// - DFlash2 is a mask-fill-family drafter by construction (reference
    ///   `dflash_generate` harvests rows `1-verify_size:`; the card says "block size 8
    ///   (7 draft tokens per verification step)" — DFLASH2-EVAL-20260820.md §3). An env
    ///   value that CONTRADICTS the census REFUSES rather than silently re-keying the
    ///   round.
    /// - Every other checkpoint rides [`Self::resolve_value`]: explicit env wins (typos
    ///   refuse loudly), unset defers to the checkpoint's own training-strategy census
    ///   (the owner-ratified 2026-08-20 default flip).
    pub fn for_draft(draft: &DflashDraft) -> Self {
        Self::for_family_value(
            draft.dflash2.is_some(),
            std::env::var("MEMRA_DSPARK_HARVEST").ok().as_deref(),
            draft.cfg.strategy_dspark,
        )
    }

    pub fn for_family_value(is_dflash2: bool, env: Option<&str>, strategy_dspark: bool) -> Self {
        if is_dflash2 {
            if env == Some("dspark") {
                panic!(
                    "MEMRA_DSPARK_HARVEST=dspark with a DFlash2 checkpoint: DFlash2 \
                     is mask-fill (b-1 drafts, anchor row is not a draft — reference \
                     dflash_generate rows 1-verify_size:); the shifted harvest would \
                     verify every slot one position early. Refusing (census-keyed, \
                     not env-keyed)."
                );
            }
            return DsparkHarvest::Dflash;
        }
        Self::resolve_value(env, strategy_dspark)
    }

    /// Manifest/serialized name (the oracle geometry manifest's `harvest` field).
    pub fn name(self) -> &'static str {
        match self {
            DsparkHarvest::Dflash => "dflash",
            DsparkHarvest::Dspark => "dspark",
        }
    }

    pub fn from_name(v: &str) -> Option<Self> {
        match v {
            "dflash" => Some(DsparkHarvest::Dflash),
            "dspark" => Some(DsparkHarvest::Dspark),
            _ => None,
        }
    }

    /// First drafter OUTPUT row consumed as a draft candidate.
    pub fn first_row(self) -> usize {
        match self {
            DsparkHarvest::Dflash => 1,
            DsparkHarvest::Dspark => 0,
        }
    }

    /// Drafted tokens harvested per round from a `b`-row block.
    pub fn n_drafts(self, b: usize) -> usize {
        match self {
            DsparkHarvest::Dflash => b - 1,
            DsparkHarvest::Dspark => b,
        }
    }

    /// The position offset (relative to the round anchor at the block's row 0) that
    /// drafter output row `row` is TRAINED to predict under this convention.
    pub fn trained_offset_of_row(self, row: usize) -> usize {
        match self {
            DsparkHarvest::Dflash => row,
            DsparkHarvest::Dspark => row + 1,
        }
    }
}

/// Checkpoint training-strategy census over the raw config.json text (the loader's
/// minimal-extractor idiom — no json dep in-tree). TRUE iff the export declares the
/// DSPARK strategy: `architectures` naming a DSpark model class (`Qwen3DSparkModel`,
/// the SpecForge OnlineDSparkModel export form) or `dflash_config.projector_type ==
/// "dspark"`. z-lab / OnlineDFlashModel mask-fill exports carry neither signal. Pure,
/// so the census is testable against config fragments without files.
pub fn dspark_strategy_census(txt: &str) -> bool {
    let arch = txt
        .find("\"architectures\"")
        .and_then(|i| {
            let rest = &txt[i..];
            let a = rest.find('[')?;
            let b = rest.find(']')?;
            Some(rest[a..b].contains("DSpark"))
        })
        .unwrap_or(false);
    let proj = txt
        .find("\"projector_type\"")
        .map(|i| {
            let rest = &txt[i..];
            let after = rest.find(':').map(|c| &rest[c + 1..]).unwrap_or("");
            after.trim_start().starts_with("\"dspark\"")
        })
        .unwrap_or(false);
    arch || proj
}

/// Accepted-prefix length of a round's candidates against the trunk's verify argmaxes:
/// `cand[0]` = the round anchor (already decided), `cand[1..]` = the drafts;
/// `vam[j]` = the trunk's argmax prediction for position anchor+j+1. Returns m =
/// number of accepted drafts (`cand[1..=m]` committed, `vam[m]` becomes the next
/// anchor). Pure so the harvest-alignment fixture can exercise it CPU-side.
pub fn dspark_accept_prefix(cand: &[u32], vam: &[u32], vt: usize) -> usize {
    let mut m = 0usize;
    while m < vt - 1 && cand[m + 1] == vam[m] {
        m += 1;
    }
    m
}

/// Verify-window policy for the dspark round (H4, DSPARK-POSTMORTEM-20260820.md §3).
///
/// B2 measured the structural fork: the fixed full-block window (vt=8) buys 95–100%
/// of the sglang accept bank but LOSES wall speed to the reactive ladder everywhere
/// except math — at 0.2–0.5 slot rates, full-block verify pays 5–6 empty rows per
/// round. The confidence policy is the mechanism both leading engines schedule with
/// (sglang v0.5.16 `dspark_planner.py` cumprod survival; vLLM #47808): size EACH
/// round's window from the drafter's own trained accept-rate head, so windows open
/// on confident streaks (math/code) and shrink on bursty text without a 4-round
/// ladder climb.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DsparkVtPolicy {
    /// The shipped reactive ladder: `vt = (m+2).clamp(3, vt_cap)` per round
    /// (`MEMRA_DFLASH_ADAPT=0` pins vt at `vt_cap` = the fixed-window arm).
    Ladder,
    /// `MEMRA_DSPARK_VT=confidence`: per-round window from cumprod survival of the
    /// confidence head's sigmoid scores, thresholded at `tau`
    /// (`MEMRA_DSPARK_VT_TAU`, default 0.5). Raw sigmoid — no STS sidecar
    /// calibration exists for this export; the postmortem names this the starting
    /// policy.
    Confidence { tau: f32 },
    /// `MEMRA_DSPARK_VT=confidence-slot` (owner directive, 2026-08-20: "take only
    /// high confidence offers"): submit only the longest draft PREFIX whose every
    /// slot clears `tau` on its own sigmoid — the low-confidence tail never enters
    /// verify. Same tau env. vs `Confidence`: if the head's per-row score is the
    /// MARGINAL accept probability (it already sinks with depth), cumprod survival
    /// double-counts the decay and over-truncates; if it is the CONDITIONAL,
    /// per-slot under-truncates. Which statistic the q38 head emits is empirical —
    /// both arms ride the A/B.
    ConfidenceSlot { tau: f32 },
}

impl DsparkVtPolicy {
    /// The served resolution: explicit `MEMRA_DSPARK_VT={ladder|confidence|
    /// confidence-slot}` wins (unknown values REFUSE loudly — a typo silently
    /// reverting the window policy would invalidate an A/B without a trace); UNSET
    /// defaults to **`confidence-slot` at τ = `MEMRA_DSPARK_VT_TAU` (default 0.5)** —
    /// the owner-ratified H4 flip (2026-08-20; cell 2's 4-arm A/B ×5 + cell 3's tau
    /// ladder put the knee at τ=.5 for the slot arm: 94–98% of the fixed-8 accept bank
    /// at wall ≥ the reactive ladder, exactness 11/11 ALL EXACT). Census-keyed per the
    /// capacity-keyed-defaults law: a checkpoint WITHOUT an accept-rate head has no
    /// signal to schedule with, so unset-env resolves to the ladder there (loudly, at
    /// load) instead of panicking on a default; `MEMRA_DFLASH_ADAPT=0` (an explicit
    /// fixed-window request) also keeps the ladder-family arm.
    pub fn resolve(has_confidence_head: bool) -> Self {
        Self::resolve_value(
            std::env::var("MEMRA_DSPARK_VT").ok().as_deref(),
            std::env::var("MEMRA_DSPARK_VT_TAU").ok().as_deref(),
            std::env::var("MEMRA_DFLASH_ADAPT").ok().as_deref(),
            has_confidence_head,
        )
    }

    pub fn resolve_value(
        vt: Option<&str>,
        tau: Option<&str>,
        adapt: Option<&str>,
        has_confidence_head: bool,
    ) -> Self {
        match vt {
            None | Some("") => {
                if adapt == Some("0") || !has_confidence_head {
                    DsparkVtPolicy::Ladder
                } else {
                    // The ratified default rides the SAME tau parse as the explicit
                    // arm (a bad MEMRA_DSPARK_VT_TAU refuses, never silently ignored).
                    Self::from_env_value(Some("confidence-slot"), tau, adapt)
                }
            }
            set => Self::from_env_value(set, tau, adapt),
        }
    }

    /// ENV-ONLY parser (no head census): unset = `Ladder`. Kept for the explicit-value
    /// path of [`Self::resolve_value`] and the policy-gate tests; round arms resolve
    /// through [`Self::resolve`] so the default stays head-census-keyed.
    pub fn from_env_value(vt: Option<&str>, tau: Option<&str>, adapt: Option<&str>) -> Self {
        match vt {
            None | Some("") | Some("ladder") => DsparkVtPolicy::Ladder,
            Some(mode @ ("confidence" | "confidence-slot")) => {
                if adapt == Some("0") {
                    panic!(
                        "MEMRA_DSPARK_VT={mode} together with MEMRA_DFLASH_ADAPT=0 is \
                         contradictory (a pinned fixed window vs a per-round confidence \
                         window); unset one — refuse-on-ambiguity"
                    );
                }
                let tau = tau
                    .map(|t| {
                        t.parse::<f32>()
                            .unwrap_or_else(|_| panic!("MEMRA_DSPARK_VT_TAU={t}: not a float"))
                    })
                    .unwrap_or(0.5);
                assert!(
                    tau > 0.0 && tau < 1.0,
                    "MEMRA_DSPARK_VT_TAU={tau}: confidence threshold must be in (0,1)"
                );
                if mode == "confidence" {
                    DsparkVtPolicy::Confidence { tau }
                } else {
                    DsparkVtPolicy::ConfidenceSlot { tau }
                }
            }
            Some(other) => panic!(
                "MEMRA_DSPARK_VT={other}: unknown verify-window policy \
                 (ladder|confidence|confidence-slot); refusing — a wrong policy \
                 silently reverts the H4 arm (DSPARK-POSTMORTEM-20260820.md)"
            ),
        }
    }

    /// True for every head-scheduled arm (the loops gate the head requirement and
    /// the embedding stash on this).
    pub fn is_confidence(&self) -> bool {
        !matches!(self, DsparkVtPolicy::Ladder)
    }

    /// Size this round's verify window from the head's pre-sigmoid slot scores.
    /// `None` under the ladder (the caller keeps its carried vt).
    pub fn size_window(&self, raws: &[f32], vt_cap: usize) -> Option<usize> {
        match *self {
            DsparkVtPolicy::Ladder => None,
            DsparkVtPolicy::Confidence { tau } => Some(dspark_confidence_vt(raws, tau, vt_cap)),
            DsparkVtPolicy::ConfidenceSlot { tau } => {
                Some(dspark_slot_confidence_vt(raws, tau, vt_cap))
            }
        }
    }
}

/// H4 window sizing (the sglang-planner/vLLM-#47808 mechanism, thresholded): `raws[k]`
/// = the accept-rate head's PRE-sigmoid score for draft slot k+1; survival
/// `S_k = prod_{j<=k} sigmoid(raws[j])`; the window keeps leading slots while
/// `S_k >= tau`. Returns `vt` = 1 (anchor) + kept drafts, clamped to `[2, vt_cap]`:
/// the draft forward is already paid, so at least one draft rides every verify — one
/// extra verify row costs less than a guaranteed empty round. Pure, so the policy's
/// knee is testable CPU-side like `dspark_accept_prefix`.
pub fn dspark_confidence_vt(raws: &[f32], tau: f32, vt_cap: usize) -> usize {
    let mut surv = 1.0f32;
    let mut kept = 0usize;
    for &r in raws {
        surv *= 1.0 / (1.0 + (-r).exp());
        if surv < tau {
            break;
        }
        kept += 1;
    }
    (1 + kept).clamp(2, vt_cap.max(2))
}

/// Owner-directive arm (2026-08-20, "take only high confidence offers"): keep the
/// longest draft PREFIX whose EVERY slot clears `tau` on its own sigmoid — truncate
/// at the first sub-threshold slot, so the low-confidence tail (B2 measured 0.2–0.5
/// slot rates at depth) never enters verify. Prefix truncation is forced by the
/// accept rule anyway (`dspark_accept_prefix` stops at the first miss — a kept slot
/// after a dropped one could never commit); the policy fork vs `dspark_confidence_vt`
/// is only the stopping statistic (per-slot marginal vs cumulative survival). Same
/// floor/cap contract.
pub fn dspark_slot_confidence_vt(raws: &[f32], tau: f32, vt_cap: usize) -> usize {
    let mut kept = 0usize;
    for &r in raws {
        let p = 1.0 / (1.0 + (-r).exp());
        if p < tau {
            break;
        }
        kept += 1;
    }
    (1 + kept).clamp(2, vt_cap.max(2))
}

// ================= SAMPLED ADMISSION (T>0) — lane/dspark-sampled-admission-20260820 =====
// True rejection sampling for the dspark route (mystery A of DSPARK-POSTMORTEM-20260820):
// draft slot j is DRAWN from a recorded proposal distribution q_j, the trunk's verify column
// arbitrates with the Leviathan/Chen rule (accept x_j while u_j*q_j(x_j) < p_j(x_j); on
// reject resample from norm(max(0, p-q)); on full accept the bonus ~ p at the last column),
// so the committed stream's distribution equals trunk-only sampling from the FILTERED target
// p — the same contract the frspec/MTP route ships (spec.rs sampled accept walk; kernels
// oracled by sample_check). T==0/None keeps every greedy path byte-identical (the exactness
// instrument and the kill-switch are the same code).
//
// Two proposal families, each recording the TRUE distribution its drafts were drawn from:
// - Rows (dspark/dflash strategy checkpoints): per-slot FILTERED softmax of the draft-logits
//   row — markov-corrected in place when the head is present (the sglang DSPARK worker's
//   "chain rejection sampling over markov-corrected draft probs"), plain rows otherwise
//   (the z-lab reference's independent-row T>0 arm).
// - Selector (DFlash2): the candidate-path selector's per-slot softmax over its top-k
//   candidate set at temperature ONLY — the reference applies no top-k/top-p to selector
//   scores (z-lab model.py `CandidateSelector.select`: `_sampling_probs(scores, temperature)`
//   with default filters) — with the candidate-set residual (`scatter_add_` of -q, clamped).

/// Rejection-sampling prefix walk: accept draft j while `u_j * q_j < p_j` (strict, f64 —
/// byte-identical to the frspec accept test). `p`/`q` are the FILTERED target/proposal
/// probabilities of the drafted tokens; `u` the per-slot uniforms. Pure so the composition
/// gate can pin the rule on CPU.
pub fn rejection_accept_len(p: &[f32], q: &[f32], u: &[f32]) -> usize {
    assert!(
        q.len() >= p.len() && u.len() >= p.len(),
        "accept walk shape"
    );
    let mut m = 0usize;
    while m < p.len() && (u[m] as f64) * (q[m] as f64) < p[m] as f64 {
        m += 1;
    }
    m
}

/// Sampled selector walk (reference `CandidateSelector.select`, temperature>0 arm): per
/// draft slot the pair scores over the top-k candidate set become a softmax at `temp`
/// (temperature ONLY — the reference passes no top-k/top-p here), one uniform draws the
/// candidate (fixed-order CDF walk), and the CHOSEN candidate seeds the next slot exactly
/// like the greedy chain. Returns (path, q_chosen[nd], q_rows[nd*top_k]) — q_rows are the
/// recorded per-slot candidate probabilities (the residual's `scatter_add_` input), and
/// q_chosen[j] == q_rows[j*top_k + chosen_j] is the accept-test q. Pure (uniforms injected)
/// so the T->0 limit, the chain conditioning, and the recorded-q contract are CPU-gateable.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_walk_sampled(
    pred_codebook: &[u8],
    succ_codebook: &[u8],
    vocab: usize,
    rank: usize,
    top_k: usize,
    unary: &[f32],
    cand: &[u32],
    hproj: &[f32],
    anchor: u32,
    nd: usize,
    temp: f32,
    uniforms: &mut dyn FnMut() -> f32,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    assert!(
        temp > 0.0,
        "sampled walk is the T>0 arm; T=0 is walk_greedy"
    );
    let (kk, r) = (top_k, rank);
    assert_eq!(unary.len(), nd * kk, "walk: unary shape");
    assert_eq!(cand.len(), nd * kk, "walk: candidate shape");
    assert_eq!(hproj.len(), nd * r, "walk: hidden-projection shape");
    let mut path = Vec::with_capacity(nd);
    let mut q_chosen = Vec::with_capacity(nd);
    let mut q_rows = Vec::with_capacity(nd * kk);
    let mut prev = anchor;
    for p in 0..nd {
        assert!(
            (prev as usize) < vocab,
            "walk: predecessor token {prev} outside codebook vocab {vocab}"
        );
        let pr = cb_row(pred_codebook, prev as usize, r);
        let hp = &hproj[p * r..(p + 1) * r];
        let gate: Vec<f32> = pr.iter().zip(hp).map(|(a, b)| a * b).collect();
        let mut scores = vec![0f32; kk];
        for (k, s) in scores.iter_mut().enumerate() {
            let c = cand[p * kk + k] as usize;
            assert!(c < vocab, "walk: candidate {c} outside codebook vocab");
            let sr = cb_row(succ_codebook, c, r);
            let mut acc = unary[p * kk + k];
            for j in 0..r {
                acc += gate[j] * sr[j];
            }
            *s = acc;
        }
        // softmax over the candidate set at temp (f64 internals; recorded probs are the
        // f32 values the CDF walk actually samples from — recorded q IS the proposal).
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut z = 0f64;
        let ex: Vec<f64> = scores
            .iter()
            .map(|&s| {
                let e0 = (((s - mx) / temp) as f64).exp();
                z += e0;
                e0
            })
            .collect();
        let probs: Vec<f32> = ex.iter().map(|&e0| (e0 / z) as f32).collect();
        let u = uniforms() as f64;
        let mut acc = 0f64;
        // fp-residue fallback (u >= f32-accumulated mass, ~2^-24 events): the max-prob
        // candidate — never a zero-prob one (host_u01's range includes 1.0 exactly).
        let mut bi = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(k, _)| k)
            .unwrap_or(0);
        for (k, &pk) in probs.iter().enumerate() {
            acc += pk as f64;
            if u < acc {
                bi = k;
                break;
            }
        }
        prev = cand[p * kk + bi];
        path.push(prev);
        q_chosen.push(probs[bi]);
        q_rows.extend_from_slice(&probs);
    }
    (path, q_chosen, q_rows)
}

/// `dflash2_propose_sampled`'s wire: (path, q_chosen, candidate ids, q_rows).
pub(crate) type Dflash2SampledProposal = (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>);

/// Per-round proposal record for the sampled dspark round — everything the rejection
/// walk needs to evaluate the TRUE per-slot proposal distribution q.
pub(crate) enum DsparkDraftSample {
    /// q lives in the round's draft-logits buffer `dl` (markov-biased in place when the
    /// head is armed); per-slot FILTERED stats retained device-contiguous for the accept
    /// gather + host-mirrored for the reject-slot residual.
    Rows {
        th: CudaSlice<f32>,          // [nd] filter thresholds (e-units), slot-indexed
        z: CudaSlice<f32>,           // [nd] renorm masses
        stats: Vec<(f32, f32, f32)>, // host (mx, th, z) per slot
    },
    /// DFlash2 candidate-path selector: q is the recorded candidate-set distribution.
    Selector {
        cand: Vec<u32>,     // [nd*top_k] candidate ids
        q_rows: Vec<f32>,   // [nd*top_k] per-slot candidate probs
        q_chosen: Vec<f32>, // [nd] prob of the drawn candidate (accept-test q)
        top_k: usize,
    },
}

/// The sampled round's verify+accept: filtered p gathered from the trunk's verify logits
/// (row j arbitrates draft `cand[j+1]` — the position mapping the greedy prefix walk uses),
/// the rejection walk over host uniforms, then `next` = bonus (full accept: filtered-Gumbel
/// from the LAST verify row with its OWN fresh stats — the sampfix-20260805 law: that row is
/// one past the gathered set) or the residual sample at the reject slot (family-keyed q:
/// full-row logits for Rows, sparse candidate-set probs for Selector). Returns (m, next) —
/// the exact (accepted-drafts, next-anchor) contract of the greedy `dspark_accept_prefix` +
/// `vam[m]` pair, so both round bodies commit identically downstream.
///
/// PENALIZED SAMPLED (lane/dspark-penalized-sampled-20260821): when the request carries
/// non-identity penalties, the vt verify columns are materialized ONCE into a penalized
/// copy where row j's Keskar pass runs over `pen_win ++ cand[1..=j]` (window-capped) —
/// the tokens committed before position j ON EVERY PATH WHERE ROW j IS CONSULTED,
/// same-round accepts included (row j is only read when drafts 1..j were all accepted,
/// i.e. exactly when `cand[1..=j]` is the committed prefix; the bonus row vt-1 is only
/// read on full accept, when all nq drafts are committed). Every p read — the batched
/// stats+gather, the bonus draw, the reject-slot residual column — points at that buffer,
/// so p is the true penalized per-state target and the committed stream equals plain
/// penalized sampling (the composition gate's penalty fixtures, self-hit included).
/// q stays the RECORDED proposal the drafts were actually drawn from (unpenalized):
/// rejection sampling is unbiased for ANY proposal with `u·q(x) < p(x)` + residual
/// `norm(max(0, p−q))`; penalizing q would only buy acceptance overlap and would cost an
/// evolving-history pass inside the sync-free device chain. `pen_win` is the caller's
/// session window ALREADY trimmed to `min(penalty_last_n, PEN_WINDOW_MAX)` (empty when
/// penalties are off — the unpenalized path is byte-untouched).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dspark_accept_sampled(
    e: &Engine,
    tlogits: &CudaSlice<f32>,
    cand: &[u32],
    vt: usize,
    n_vocab: usize,
    dl: &CudaSlice<f32>,
    prop: &DsparkDraftSample,
    sp: &crate::spec::SpecSampling,
    pen_win: &[u32],
    sctr: &mut u32,
    uctr: &mut u32,
) -> Result<(usize, u32), Box<dyn std::error::Error>> {
    let nq = vt - 1; // drafts under this round's verify window
    debug_assert!(nq >= 1 && cand.len() > nq, "sampled accept shape");
    // --- penalized verify columns (identity penalties: no copy, no launch, raw tlogits) ---
    let pen_on = sp.pen_on();
    let ptl: Option<CudaSlice<f32>> = if pen_on {
        let win = sp.penalty_last_n.min(crate::spec::PEN_WINDOW_MAX);
        debug_assert!(pen_win.len() <= win, "pen_win must arrive pre-trimmed");
        let mut hist: Vec<u32> = Vec::with_capacity(pen_win.len() + nq);
        hist.extend_from_slice(pen_win);
        hist.extend_from_slice(&cand[1..=nq]); // drafted tokens: row j reads the first j
        let hd = e.htod_u32_v(&hist)?;
        let mut buf = e.clone_dtod(tlogits)?;
        e.penalize_logits_rows_inc(
            &mut buf,
            &hd,
            pen_win.len(),
            sp.penalty_repeat,
            sp.penalty_freq,
            sp.penalty_present,
            n_vocab,
            vt,
            win,
        )?;
        Some(buf)
    } else {
        None
    };
    let p_src: &CudaSlice<f32> = ptl.as_ref().unwrap_or(tlogits);
    // --- filtered p at the drafted tokens (one batched stats + gather over rows 0..nq-1) ---
    let rows: Vec<i32> = (0..nq as i32).collect();
    let ids: Vec<u32> = cand[1..=nq].to_vec();
    let rowsd = e.htod_i32(&rows)?;
    let idsd = e.htod_u32_v(&ids)?;
    let (mut pth, mut pz, mut pmx) = (e.zeros(nq)?, e.zeros(nq)?, e.zeros(nq)?);
    e.filter_stats(
        p_src, n_vocab, &rowsd, &mut pth, &mut pz, &mut pmx, n_vocab, nq, sp.temp, sp.top_k,
        sp.top_p, sp.min_p,
    )?;
    let mut pj_d = e.zeros(nq)?;
    e.softmax_gather_filtered(
        p_src, n_vocab, &idsd, &rowsd, &pth, &pz, &mut pj_d, n_vocab, nq, sp.temp,
    )?;
    let pj = e.dtoh(&pj_d)?;
    let (pthv, pzv, pmxv) = (e.dtoh(&pth)?, e.dtoh(&pz)?, e.dtoh(&pmx)?);
    // --- q at the drafted tokens (the recorded proposal distribution) ---
    let qj: Vec<f32> = match prop {
        DsparkDraftSample::Rows { th, z, .. } => {
            // dl row j is draft j's (bias-corrected) logits row; th/z are slot-indexed, and
            // rows 0..nq-1 index both the buffer rows and the stat pairs.
            let mut qd = e.zeros(nq)?;
            e.softmax_gather_filtered(
                dl, n_vocab, &idsd, &rowsd, th, z, &mut qd, n_vocab, nq, sp.temp,
            )?;
            e.dtoh(&qd)?
        }
        DsparkDraftSample::Selector { q_chosen, .. } => q_chosen[..nq].to_vec(),
    };
    // --- the rejection walk ---
    let mut us = Vec::with_capacity(nq);
    for _ in 0..nq {
        us.push(crate::spec::host_u01(sp.seed, *uctr));
        *uctr = uctr.wrapping_add(1);
    }
    let m = rejection_accept_len(&pj[..nq], &qj[..nq], &us);
    // --- next anchor: bonus or residual ---
    let next = if m == nq {
        // FULL ACCEPT: bonus ~ filtered p at verify row vt-1 — fresh stats for THIS row.
        // Under penalties p_src row vt-1 carries the FULL drafted block in its window
        // (all nq drafts are committed on this path — the "drafted token penalizes its
        // own successor" case the composition gate's self-hit fixture pins).
        let rows_l = e.htod_i32(&[(vt - 1) as i32])?;
        let (mut th1, mut z1, mut mx1) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
        e.filter_stats(
            p_src, n_vocab, &rows_l, &mut th1, &mut z1, &mut mx1, n_vocab, 1, sp.temp, sp.top_k,
            sp.top_p, sp.min_p,
        )?;
        let mut pb = e.zeros(n_vocab)?;
        e.gumbel_perturb_filtered_col(
            p_src,
            vt - 1,
            &mut pb,
            n_vocab,
            sp.seed,
            *sctr,
            sp.temp,
            &mx1,
            &th1,
            0,
        )?;
        *sctr = sctr.wrapping_add(1);
        let td = e.argmax_token_device(&pb, n_vocab)?;
        e.dtoh_u32_one(&td)?
    } else {
        // REJECT at slot m: token ~ norm(max(0, p_m - q_m)); p row m's stats come from the
        // gathered set (rows 0..nq-1 cover every reject slot). Under penalties the column
        // copy MUST come from p_src (the penalized buffer) — a raw-tlogits residual is the
        // composition gate's "residual reads unpenalized p" tooth.
        let mut col = e.zeros(n_vocab)?;
        e.copy_view_into(
            &mut col,
            0,
            &p_src.slice(m * n_vocab..(m + 1) * n_vocab),
            n_vocab,
        )?;
        let p_stats = (pmxv[m], pthv[m], pzv[m]);
        let mut tok_d = e.alloc_u32_zeroed(1)?;
        let sc = *sctr;
        *sctr = sctr.wrapping_add(1);
        match prop {
            DsparkDraftSample::Rows { stats, .. } => {
                let mut qbuf = e.zeros(n_vocab)?;
                e.copy_view_into(
                    &mut qbuf,
                    0,
                    &dl.slice(m * n_vocab..(m + 1) * n_vocab),
                    n_vocab,
                )?;
                e.residual_sample_filtered(
                    &col,
                    Some(&qbuf),
                    n_vocab,
                    sp.temp,
                    sp.seed,
                    sc,
                    p_stats,
                    stats[m],
                    &mut tok_d,
                )?;
            }
            DsparkDraftSample::Selector {
                cand: cids,
                q_rows,
                top_k,
                ..
            } => {
                let k = *top_k;
                let ids_m = e.htod_u32_v(&cids[m * k..(m + 1) * k])?;
                let qs_m = e.htod(&q_rows[m * k..(m + 1) * k])?;
                e.residual_sample_sparse_q(
                    &col, &ids_m, &qs_m, k, n_vocab, sp.temp, sp.seed, sc, p_stats, &mut tok_d,
                )?;
            }
        }
        e.dtoh_u32(&tok_d)?[0]
    };
    Ok((m, next))
}

/// The DFlash2 round attention over the non-causal symmetric window: one seam for both the
/// first-light (`forward_block`) and cached (`forward_round`) arms, dispatching the clipped
/// kernel unless the rollback door is thrown.
///
/// `floor` (lane/spec-exclusions-20260902): the KV's first EXISTING ctx row
/// (`DflashKv::floor`, 0 on every cold-primed or full-tail KV). A COLD-DRAFTER session
/// (`DflashKv::new_cold_at`) or a short-tail import owns rows only from `floor` up; the
/// clipped kernel raises its window floor to it and the drafter attends a shorter context,
/// exactly the program a shorter prompt runs. The legacy full-scan kernel has no floor arm
/// (it would have scored the zero rows below the floor at e^0 each and diluted the real
/// context), which is why the clipped kernel is the only round attention.
#[allow(clippy::too_many_arguments)]
fn d2_windowed_attn(
    e: &Engine,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    attn: &mut CudaSlice<f32>,
    hd: usize,
    nh: usize,
    nkv: usize,
    t: usize,
    t_kv: usize,
    scale: f32,
    c: &DflashCfg,
    floor: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    e.sdpa_naive_w_lo(
        q,
        k,
        v,
        attn,
        hd,
        nh,
        nkv,
        t,
        t_kv,
        scale,
        false,
        c.sliding_window,
        floor,
    )
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn validate_dflash_tensor(
    name: &str,
    info: &memra_gguf::safetensors::StInfo,
    expected: &[u64],
) -> Result<(), String> {
    if info.dtype != "BF16" {
        return Err(format!(
            "DFlash tensor {name} has dtype {}, expected BF16",
            info.dtype
        ));
    }
    let found = info.ne();
    if found != expected {
        return Err(format!(
            "DFlash tensor {name} has shape {found:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn validate_dflash_attention_geometry(
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
) -> Result<(), String> {
    if n_head == 0 || n_kv == 0 || head_dim == 0 || !n_head.is_multiple_of(n_kv) {
        return Err(format!(
            "DFlash attention geometry requires nonzero n_head divisible by n_kv; got n_head={n_head}, n_kv={n_kv}, head_dim={head_dim}"
        ));
    }
    n_head
        .checked_mul(head_dim)
        .ok_or("DFlash query-head geometry overflow")?;
    n_kv.checked_mul(head_dim)
        .ok_or("DFlash key/value-head geometry overflow")?;
    Ok(())
}

fn validate_selector_top_k(top_k: usize, vocab: usize) -> Result<(), String> {
    if top_k == 0 || top_k > vocab {
        return Err(format!(
            "DFlash2 selector_top_k {top_k} is outside codebook vocabulary 1..={vocab}"
        ));
    }
    Ok(())
}

fn validate_layer_layout(layer_sliding: &[bool], n_layer: usize) -> Result<(), String> {
    if layer_sliding.len() != n_layer {
        return Err(format!(
            "DFlash layer_types has {} entries, expected num_hidden_layers {n_layer}",
            layer_sliding.len()
        ));
    }
    Ok(())
}

/// Host q8_0 encode (ggml block layout: [d f16][32 x i8] = 34B/32 vals). The drafter's
/// weights ride the dp4a fast path at 1.6GB resident (bf16 3.1GB + the 31B trunk OOM'd
/// 24GB; f32 6.2GB worse). Drafter quantization moves ACCEPTANCE only — verify exactness
/// is structural.
fn encode_q8_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let dh = half_from_f32(d);
        out.extend_from_slice(&dh.to_le_bytes());
        for &v in blk {
            out.push(((v * id).round().clamp(-127.0, 127.0)) as i8 as u8);
        }
    }
    out
}

/// Host q4_0 encode (ggml: [d f16][16B packed nibbles] = 18B/32 vals; q = round(v/d)+8,
/// d = amax/-7 sign trick NOT used — plain amax/7? ggml uses d = max/-8 .. follow ggml:
/// d = amax / -8 when the max is negative-dominant; reference quantize_row_q4_0: d =
/// max(|v|)/-8 signed-max form). Implemented to match ggml quantize_row_q4_0_ref.
fn encode_q4_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 18);
    for blk in vals.chunks_exact(32) {
        // ggml ref: pick the value with the LARGEST |v| (keeping sign), d = that / -8
        let mut amax = 0f32;
        let mut mx = 0f32;
        for &v in blk {
            if v.abs() > amax {
                amax = v.abs();
                mx = v;
            }
        }
        let d = mx / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half_from_f32(d).to_le_bytes());
        for j in 0..16 {
            let x0 = (blk[j] * id + 8.5).clamp(0.0, 15.0) as u8;
            let x1 = (blk[j + 16] * id + 8.5).clamp(0.0, 15.0) as u8;
            out.push(x0 | (x1 << 4));
        }
    }
    out
}

fn half_from_f32(v: f32) -> u16 {
    // f32 -> IEEE f16 (round-to-nearest-even; range of q8_0 d values is tame)
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7fffff;
    if exp <= 0 {
        return sign;
    } // flush tiny d to zero
    if exp >= 31 {
        return sign | 0x7c00;
    } // inf (unreachable for sane d)
    let mut h = sign | ((exp as u16) << 10) | ((man >> 13) as u16);
    // round to nearest even on the truncated 13 bits
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

impl DflashDraft {
    /// Load the backbone-only checkpoint dir (config.json + model.safetensors, bf16).
    /// Config scalars ride a minimal extractor (no json dep in-tree — HfConfig precedent).
    pub fn load(e: &Engine, dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let txt = std::fs::read_to_string(dir.join("config.json"))?;
        fn num(txt: &str, key: &str) -> Option<f64> {
            let i = txt.find(&format!("\"{key}\""))?;
            let rest = &txt[i..];
            let colon = rest.find(':')?;
            let val: String = rest[colon + 1..]
                .trim_start()
                .chars()
                .take_while(|c| {
                    c.is_ascii_digit()
                        || *c == '.'
                        || *c == '-'
                        || *c == 'e'
                        || *c == 'E'
                        || *c == '+'
                })
                .collect();
            val.parse().ok()
        }
        fn num_list(txt: &str, key: &str) -> Vec<usize> {
            let Some(i) = txt.find(&format!("\"{key}\"")) else {
                return Vec::new();
            };
            let rest = &txt[i..];
            let (Some(a), Some(b)) = (rest.find('['), rest.find(']')) else {
                return Vec::new();
            };
            rest[a + 1..b]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        }
        /// Substring of the JSON OBJECT value of a top-level key (brace-balanced) —
        /// the explicit scoped parse the DFlash2 census demands: `dflash_config` and
        /// `rope_parameters` are nested objects, and finding their keys by global
        /// `txt.find` is luck, not a contract (DFLASH2-EVAL-20260820.md §5.1).
        fn scope<'a>(txt: &'a str, key: &str) -> Option<&'a str> {
            let i = txt.find(&format!("\"{key}\""))?;
            let rest = &txt[i..];
            let open = rest.find('{')?;
            let mut depth = 0usize;
            for (j, ch) in rest[open..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(&rest[open..open + j + 1]);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        // Family detection is the ARCHITECTURES string, not tensor presence: a DFlash2
        // checkpoint whose new tensors were stripped must REFUSE, not degrade into the
        // 58-tensor untrained program (DFLASH2-EVAL-20260820.md §3).
        let is_dflash2 = {
            let arch = scope_list(&txt, "architectures");
            arch.contains("DFlash2DraftModel")
        };
        fn scope_list(txt: &str, key: &str) -> String {
            let Some(i) = txt.find(&format!("\"{key}\"")) else {
                return String::new();
            };
            let rest = &txt[i..];
            match (rest.find('['), rest.find(']')) {
                (Some(a), Some(b)) if a < b => rest[a + 1..b].to_string(),
                _ => String::new(),
            }
        }
        // DFlash2 scalars parse from their OWN scopes; other families keep the
        // historical global-find behavior byte-identically.
        let d2_cfg_txt: Option<&str> = if is_dflash2 {
            Some(
                scope(&txt, "dflash_config")
                    .ok_or("DFlash2DraftModel config.json has no dflash_config object")?,
            )
        } else {
            None
        };
        let required_usize =
            |scope: &str, k: &str, label: &str| -> Result<usize, Box<dyn std::error::Error>> {
                let value = num(scope, k).ok_or_else(|| format!("{label} missing {k}"))?;
                if !value.is_finite()
                    || value < 0.0
                    || value.fract() != 0.0
                    || value > usize::MAX as f64
                {
                    return Err(format!("{label} {k}={value} is not a non-negative usize").into());
                }
                Ok(value as usize)
            };
        let g = |k: &str| required_usize(&txt, k, "config");
        let g2 = |k: &str| {
            required_usize(
                d2_cfg_txt.ok_or("DFlash2 config scope is unavailable")?,
                k,
                "dflash_config",
            )
        };
        // layer_types order: count entries, mark sliding ones
        let layer_sliding: Vec<bool> = {
            let i = txt
                .find("\"layer_types\"")
                .ok_or("config missing layer_types")?;
            let rest = &txt[i..];
            let a = rest.find('[').ok_or("layer_types is not an array")?;
            let b = rest.find(']').ok_or("layer_types array is unterminated")?;
            rest[a + 1..b]
                .split(',')
                .map(|s| s.contains("sliding_attention"))
                .collect()
        };
        // sliding_window is null on all-full-attention exports (q38 arm-a); the window
        // only constrains rounds when a sliding layer exists (reference: resolve_dflash_
        // attention_layout returns None when no layer slides).
        let sliding_window = if layer_sliding.iter().any(|&s| s) {
            g("sliding_window")?
        } else {
            num(&txt, "sliding_window")
                .map(|v| v as usize)
                .unwrap_or(usize::MAX)
        };
        // Explicit top-level is_causal (z-lab reference: overrides the layer-type
        // default). Parsed as a bare bool; absent = None (historical arms unchanged).
        let is_causal = txt
            .find("\"is_causal\"")
            .and_then(|i| txt[i..].find(':').map(|c| i + c + 1))
            .map(|v| txt[v..].trim_start().starts_with("true"));
        let cfg = DflashCfg {
            hidden: g("hidden_size")?,
            n_head: g("num_attention_heads")?,
            n_kv: g("num_key_value_heads")?,
            head_dim: g("head_dim")?,
            n_ff: g("intermediate_size")?,
            n_layer: g("num_hidden_layers")?,
            eps: num(&txt, "rms_norm_eps").ok_or("config missing rms_norm_eps")? as f32,
            // DFlash2 (transformers-5 style): rope_theta lives in the nested
            // rope_parameters object — parse it from its scope, not by global find.
            rope_theta: if is_dflash2 {
                let rp = scope(&txt, "rope_parameters")
                    .ok_or("DFlash2 config has no rope_parameters")?;
                if !rp.contains("\"default\"") {
                    return Err(format!(
                        "DFlash2 rope_parameters rope_type is not default; refusing {rp}"
                    )
                    .into());
                }
                num(rp, "rope_theta").ok_or("rope_parameters missing rope_theta")? as f32
            } else {
                num(&txt, "rope_theta").ok_or("config missing rope_theta")? as f32
            },
            block_size: if is_dflash2 {
                g2("block_size")?
            } else {
                g("block_size")?
            },
            mask_token_id: u32::try_from(if is_dflash2 {
                g2("mask_token_id")?
            } else {
                g("mask_token_id")?
            })
            .map_err(|_| "DFlash mask_token_id does not fit u32")?,
            target_layer_ids: if is_dflash2 {
                num_list(
                    d2_cfg_txt.ok_or("DFlash2 config scope is unavailable")?,
                    "target_layer_ids",
                )
            } else {
                num_list(&txt, "target_layer_ids")
            },
            sliding_window,
            layer_sliding,
            strategy_dspark: dspark_strategy_census(&txt),
            is_causal,
        };
        if cfg.n_layer == 0 || cfg.n_layer > 1_024 {
            return Err(format!(
                "DFlash num_hidden_layers {} is outside 1..=1024",
                cfg.n_layer
            )
            .into());
        }
        if cfg.hidden == 0
            || cfg.n_head == 0
            || cfg.n_kv == 0
            || cfg.head_dim == 0
            || cfg.n_ff == 0
            || !cfg.eps.is_finite()
            || cfg.eps <= 0.0
            || !cfg.rope_theta.is_finite()
            || cfg.rope_theta <= 0.0
        {
            return Err("DFlash config carries zero or non-finite model geometry".into());
        }
        validate_dflash_attention_geometry(cfg.n_head, cfg.n_kv, cfg.head_dim)?;
        validate_layer_layout(&cfg.layer_sliding, cfg.n_layer)?;
        if is_dflash2 {
            // The windowed round arm implements the reference's NON-causal symmetric
            // window only (config `is_causal: false` on the q38 DFlash2 export). A
            // causal DFlash2 variant is a different mask program — refuse it rather
            // than run the wrong one fluently.
            if cfg.is_causal != Some(false) {
                return Err(format!(
                    "DFlash2 requires explicit is_causal=false; got {:?}",
                    cfg.is_causal
                )
                .into());
            }
            if !cfg.layer_sliding.iter().all(|&sliding| sliding) {
                return Err(format!(
                    "DFlash2 expects all layers sliding_attention; got {:?}",
                    cfg.layer_sliding
                )
                .into());
            }
            if cfg.block_size > cfg.sliding_window {
                return Err(format!(
                    "DFlash2 block {} exceeds sliding window {}",
                    cfg.block_size, cfg.sliding_window
                )
                .into());
            }
        }
        let st = memra_gguf::safetensors::StModel::open(&dir.join("model.safetensors"))?;
        let validate = |name: &str,
                        info: &memra_gguf::safetensors::StInfo,
                        expected: &[u64]|
         -> Result<(), Box<dyn std::error::Error>> {
            validate_dflash_tensor(name, info, expected).map_err(Into::into)
        };
        // 1D norm weights ride raw slices; 2D matmul weights ride GpuTensor::Float
        // (cuBLASLt f32 arm — the Stage-A numeric class, right for oracle parity).
        let up =
            |name: &str, expected: &[u64]| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
                let (info, bytes) = st
                    .raw(name)
                    .ok_or_else(|| format!("missing tensor {name}"))?;
                validate(name, info, expected)?;
                e.htod(&bf16_to_f32(bytes))
            };
        // Precision policy (MEMRA_DFLASH_PREC seam): "q4" = all q4_0 (DEFAULT since
        // lane/dflash2-head-trim 2026-08-25, owner-ratified): measured on BOTH engaging
        // card classes at unchanged acceptance — RTX PRO 6000 dspark_q38_gate x3
        // interleaved 157.9 vs q8 152.4 spec tok/s (accept 0.662 vs 0.656, ALL EXACT;
        // darklanes research/dflash2-pro6000-20260824/prec-ladder + trim cells) and the
        // 5090 rig cell that shipped the arm (PR #41). "q8" = all q8_0 (1.6GB, the
        // pre-flip default = the rollback seam); "mixed" = bf16 attn+fc (the
        // ctx-conditioning path) + q8_0 ffn (~2.2GB — fits the ~2.8GB headroom beside
        // the 31B trunk); "bf16" = all bf16 (parity runs, no target). The asymmetric
        // "q5" arm was measured DEFECTIVE (acceptance 0.656 -> 0.424) and never landed.
        let prec_env = std::env::var("MEMRA_DFLASH_PREC").ok();
        let prec = dflash_precision(prec_env.as_deref())?;
        let upw = |name: &str, expected: &[u64]| -> Result<GpuTensor, Box<dyn std::error::Error>> {
            let (info, bytes) = st
                .raw(name)
                .ok_or_else(|| format!("missing tensor {name}"))?;
            validate(name, info, expected)?;
            let shape = info.ne(); // ggml order: ne[0]=in_f, ne[1]=out_f
            let in_f = shape[0] as usize;
            let is_ffn = name.contains(".mlp.");
            let bf16 = prec == "bf16"
                || (prec == "mixed" && !is_ffn)
                || (prec == "fc" && name == "fc.weight");
            if bf16 {
                return Ok(GpuTensor::FloatBf16 {
                    data: e.upload_u8(bytes)?,
                    ne: shape.to_vec(),
                });
            }
            let f32s = bf16_to_f32(bytes);
            if prec == "q4" {
                let q = encode_q4_0(&f32s);
                return Ok(GpuTensor::Quant {
                    bytes: e.upload_u8(&q)?,
                    qtype: crate::QT_Q4_0,
                    row_bytes: in_f / 32 * 18,
                    ne: shape.to_vec(),
                    scale: 1.0,
                    rp: false,
                    #[cfg(memra_cutlass)]
                    cutlass: None,
                    fp8: None,
                    blk: None,
                    rp4: None,
                    f16: None,
                });
            }
            let q = encode_q8_0(&f32s);
            Ok(GpuTensor::Quant {
                bytes: e.upload_u8(&q)?,
                qtype: crate::QT_Q8_0,
                row_bytes: in_f / 32 * 34,
                ne: shape.to_vec(),
                scale: 1.0,
                rp: false,
                #[cfg(memra_cutlass)]
                cutlass: None,
                fp8: None,
                blk: None,
                rp4: None,
                f16: None,
            })
        };
        let hidden = cfg.hidden as u64;
        let q_width = cfg
            .n_head
            .checked_mul(cfg.head_dim)
            .ok_or("DFlash q geometry overflow")? as u64;
        let kv_width = cfg
            .n_kv
            .checked_mul(cfg.head_dim)
            .ok_or("DFlash kv geometry overflow")? as u64;
        let ff = cfg.n_ff as u64;
        let head_dim = cfg.head_dim as u64;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(DflashLayer {
                wq: upw(&p("self_attn.q_proj.weight"), &[hidden, q_width])?,
                wk: upw(&p("self_attn.k_proj.weight"), &[hidden, kv_width])?,
                wv: upw(&p("self_attn.v_proj.weight"), &[hidden, kv_width])?,
                wo: upw(&p("self_attn.o_proj.weight"), &[q_width, hidden])?,
                w_gate: upw(&p("mlp.gate_proj.weight"), &[hidden, ff])?,
                w_up: upw(&p("mlp.up_proj.weight"), &[hidden, ff])?,
                w_down: upw(&p("mlp.down_proj.weight"), &[ff, hidden])?,
                ln_in: up(&p("input_layernorm.weight"), &[hidden])?,
                ln_post: up(&p("post_attention_layernorm.weight"), &[hidden])?,
                q_norm: up(&p("self_attn.q_norm.weight"), &[head_dim])?,
                k_norm: up(&p("self_attn.k_norm.weight"), &[head_dim])?,
            });
        }
        let markov = if let Some((info, bytes)) = st.raw("markov_head.markov_w1.weight") {
            let sh = info.ne(); // [rank, vocab] in ggml order (safetensors [V, rank] reversed)
            if info.dtype != "BF16" || sh.len() != 2 {
                return Err("markov_head.markov_w1.weight must be rank-2 BF16".into());
            }
            let (rank, vocab) = (sh[0] as usize, sh[1] as usize);
            let (i2, b2) = st
                .raw("markov_head.markov_w2.weight")
                .ok_or("markov_w2 missing beside markov_w1")?;
            validate("markov_head.markov_w2.weight", i2, &sh)?;
            // w2 follows the precision seam: bf16 for parity runs (the q8_0 encode is a
            // serving-size choice and would put quant error inside the markov-logits gate),
            // q8_0 otherwise (acceptance-only impact, like the trunk weights).
            let w2 = if prec == "bf16" {
                GpuTensor::FloatBf16 {
                    data: e.upload_u8(b2)?,
                    ne: i2.ne().to_vec(),
                }
            } else {
                let w2f = bf16_to_f32(b2);
                let w2q = encode_q8_0(&w2f);
                GpuTensor::Quant {
                    bytes: e.upload_u8(&w2q)?,
                    qtype: crate::QT_Q8_0,
                    row_bytes: rank / 32 * 34,
                    ne: vec![rank as u64, vocab as u64],
                    scale: 1.0,
                    rp: false,
                    #[cfg(memra_cutlass)]
                    cutlass: None,
                    fp8: None,
                    blk: None,
                    rp4: None,
                    f16: None,
                }
            };
            Some(MarkovHead {
                w1_bf16: e.upload_u8(bytes)?,
                w2,
                rank,
                vocab,
            })
        } else {
            None
        };
        let confidence = if let Some((info, bytes)) = st.raw("confidence_head.proj.weight") {
            let sh = info.ne(); // ggml order: ne[0]=in_dim, ne[1]=1
            if info.dtype != "BF16" || sh.len() != 2 || sh[1] != 1 {
                return Err("confidence_head.proj.weight must be BF16 [in_dim, 1]".into());
            }
            let in_dim = sh[0] as usize;
            let (bi, bb) = st
                .raw("confidence_head.proj.bias")
                .ok_or("confidence bias missing beside weight")?;
            validate("confidence_head.proj.bias", bi, &[1])?;
            let with_markov = markov
                .as_ref()
                .map(|m| in_dim == cfg.hidden + m.rank)
                .unwrap_or(false);
            if !with_markov && in_dim != cfg.hidden {
                return Err(format!(
                    "confidence_head in_dim {in_dim} matches neither hidden {} nor hidden+rank",
                    cfg.hidden
                )
                .into());
            }
            Some(ConfidenceHead {
                w: bf16_to_f32(bytes),
                b: bf16_to_f32(bb)[0],
                in_dim,
                with_markov,
            })
        } else {
            None
        };
        // ---- DFlash2 family tensors (DFLASH2-EVAL-20260820.md §2): 10 conv modules
        // (base_kernel + kernel_projection around attention AND mlp in EVERY layer) +
        // the candidate path selector (hidden_projection + two codebooks). REQUIRED
        // when the arch says DFlash2DraftModel: a missing tensor is a refusal (`?`),
        // never a degraded program.
        let dflash2 = if is_dflash2 {
            if markov.is_some() || confidence.is_some() {
                return Err(
                    "DFlash2 checkpoint carries unsupported markov/confidence tensors".into(),
                );
            }
            let rank = g2("selector_rank")?;
            let top_k = g2("selector_top_k")?;
            let conv_k = g2("conv_kernel_size")?;
            let group_size = g2("conv_group_size")?;
            if rank == 0
                || top_k == 0
                || conv_k == 0
                || group_size == 0
                || !cfg.hidden.is_multiple_of(group_size)
            {
                return Err(
                    "DFlash2 selector/convolution geometry is zero or not divisible".into(),
                );
            }
            let groups = cfg.hidden / group_size;
            let load_conv = |name: &str| -> Result<Dflash2Conv, Box<dyn std::error::Error>> {
                let (bi, bb) = st
                    .raw(&format!("{name}.base_kernel"))
                    .ok_or_else(|| format!("DFlash2 census: missing {name}.base_kernel"))?;
                // safetensors [2, k, hidden] -> ggml ne reversed [hidden, k, 2]
                validate(
                    &format!("{name}.base_kernel"),
                    bi,
                    &[cfg.hidden as u64, conv_k as u64, 2],
                )?;
                let pname = format!("{name}.kernel_projection.weight");
                let (pi, _pb) = st
                    .raw(&pname)
                    .ok_or_else(|| format!("DFlash2 census: missing {pname}"))?;
                let projected = 2usize
                    .checked_mul(conv_k)
                    .and_then(|value| value.checked_mul(groups))
                    .ok_or("DFlash2 convolution projection geometry overflow")?;
                let expected = [cfg.hidden as u64, projected as u64];
                validate(&pname, pi, &expected)?;
                Ok(Dflash2Conv {
                    base: e.htod(&bf16_to_f32(bb))?,
                    proj: upw(&pname, &expected)?,
                })
            };
            let mut attn_conv = Vec::with_capacity(cfg.n_layer);
            let mut mlp_conv = Vec::with_capacity(cfg.n_layer);
            for i in 0..cfg.n_layer {
                attn_conv.push(load_conv(&format!("layers.{i}.attention_conv"))?);
                mlp_conv.push(load_conv(&format!("layers.{i}.mlp_conv"))?);
            }
            // Codebooks: stored WITHOUT `.weight` (checkpoint quirk; reference
            // from_pretrained maps the keys). Host-resident raw bf16.
            let cb = |name: &str| -> Result<(Vec<u8>, usize), Box<dyn std::error::Error>> {
                let (ci, cbytes) = st
                    .raw(&format!("candidate_selector.{name}"))
                    .ok_or_else(|| format!("DFlash2 census: missing candidate_selector.{name}"))?;
                let ne = ci.ne(); // ggml: [rank, V]
                if ci.dtype != "BF16" || ne.len() != 2 || ne[0] as usize != rank {
                    return Err(format!(
                        "candidate_selector.{name} must be rank-2 BF16 with inner rank {rank}; found {:?} {}",
                        ne, ci.dtype
                    )
                    .into());
                }
                Ok((cbytes.to_vec(), ne[1] as usize))
            };
            let (pred_codebook, v1) = cb("predecessor_codebook")?;
            let (succ_codebook, v2) = cb("successor_codebook")?;
            if v1 != v2 {
                return Err(format!("DFlash2 codebook vocab mismatch: {v1} != {v2}").into());
            }
            validate_selector_top_k(top_k, v1)?;
            let hp_name = "candidate_selector.hidden_projection.weight";
            let (hi, _hb) = st
                .raw(hp_name)
                .ok_or_else(|| format!("DFlash2 census: missing {hp_name}"))?;
            let hp_expected = [cfg.hidden as u64, rank as u64];
            validate(hp_name, hi, &hp_expected)?;
            Some(Dflash2Head {
                attn_conv,
                mlp_conv,
                hidden_proj: upw(hp_name, &hp_expected)?,
                pred_codebook,
                succ_codebook,
                rank,
                top_k,
                conv_k,
                group_size,
                vocab: v1,
            })
        } else {
            None
        };
        // CENSUS GATE: every tensor in the export must be consumed by the map above.
        // DSpark-class checkpoints (markov head present) and DFlash2 checkpoints
        // REFUSE on unrecognized names — an unmapped tensor is a semantic program we
        // would silently drop (house law). Plain dflash checkpoints keep the
        // historical warn-only behavior.
        {
            let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
            for i in 0..cfg.n_layer {
                for s in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ] {
                    consumed.insert(format!("layers.{i}.{s}"));
                }
                if dflash2.is_some() {
                    for s in [
                        "attention_conv.base_kernel",
                        "attention_conv.kernel_projection.weight",
                        "mlp_conv.base_kernel",
                        "mlp_conv.kernel_projection.weight",
                    ] {
                        consumed.insert(format!("layers.{i}.{s}"));
                    }
                }
            }
            for s in [
                "fc.weight",
                "hidden_norm.weight",
                "norm.weight",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
                "confidence_head.proj.bias",
            ] {
                consumed.insert(s.into());
            }
            if dflash2.is_some() {
                for s in [
                    "candidate_selector.hidden_projection.weight",
                    "candidate_selector.predecessor_codebook",
                    "candidate_selector.successor_codebook",
                ] {
                    consumed.insert(s.into());
                }
            }
            let leftovers: Vec<&String> = st.names().filter(|n| !consumed.contains(*n)).collect();
            if !leftovers.is_empty() {
                if markov.is_some() || dflash2.is_some() {
                    return Err(format!(
                        "dspark/dflash2 census: unrecognized tensors {leftovers:?}"
                    )
                    .into());
                }
                eprintln!("[dflash census] unmapped tensors (ignored): {leftovers:?}");
            }
        }
        // YaRN rope from config rope_parameters (HF _compute_yarn_parameters, verified
        // numerically vs Qwen3RotaryEmbedding on the arm-a export).
        let rope_yarn =
            if txt.contains("\"rope_type\": \"yarn\"") || txt.contains("\"rope_type\":\"yarn\"") {
                let factor = num(&txt, "factor").ok_or("yarn missing factor")?;
                let orig = num(&txt, "original_max_position_embeddings")
                    .ok_or("yarn missing original_max_position_embeddings")?;
                let beta_fast = num(&txt, "beta_fast").ok_or("yarn missing beta_fast")?;
                let beta_slow = num(&txt, "beta_slow").ok_or("yarn missing beta_slow")?;
                if !factor.is_finite()
                    || factor <= 0.0
                    || !orig.is_finite()
                    || orig <= 0.0
                    || !beta_fast.is_finite()
                    || !beta_slow.is_finite()
                {
                    return Err("yarn parameters must be finite and positive".into());
                }
                let base = cfg.rope_theta as f64;
                let d = cfg.head_dim as f64;
                let corr =
                    |r: f64| d * (orig / (r * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
                let low = corr(beta_fast).floor().max(0.0);
                let high = corr(beta_slow).ceil().min(d - 1.0);
                let half = cfg.head_dim / 2;
                let mut ff = Vec::with_capacity(half);
                for j in 0..half {
                    let base_inv = base.powf(-2.0 * j as f64 / d);
                    let ramp = (((j as f64) - low) / (high - low)).clamp(0.0, 1.0);
                    let ex = 1.0 - ramp; // extrapolation share
                    let yarn_inv = (base_inv / factor) * (1.0 - ex) + base_inv * ex;
                    ff.push((base_inv / yarn_inv) as f32);
                }
                let mscale = (0.1 * factor.ln() + 1.0) as f32;
                Some((e.htod(&ff)?, mscale))
            } else {
                None
            };
        let fc_in = cfg
            .target_layer_ids
            .len()
            .checked_mul(cfg.hidden)
            .ok_or("DFlash fc geometry overflow")? as u64;
        let fc = upw("fc.weight", &[fc_in, hidden])?;
        // Ratified-default receipts (capacity-keyed-defaults law: the active program is
        // NAMED at load, never inferred from silence). The boot output-sample gate greps
        // these lines; a run whose log lacks them did not load this code.
        eprintln!(
            "[dspark] precision={prec} (MEMRA_DFLASH_PREC {})",
            if prec_env.is_some() { "set" } else { "unset" },
        );
        eprintln!(
            "[dspark] harvest={} (checkpoint census dflash2={} strategy_dspark={}, \
             MEMRA_DSPARK_HARVEST {})",
            DsparkHarvest::for_family_value(
                dflash2.is_some(),
                std::env::var("MEMRA_DSPARK_HARVEST").ok().as_deref(),
                cfg.strategy_dspark,
            )
            .name(),
            dflash2.is_some(),
            cfg.strategy_dspark,
            match std::env::var("MEMRA_DSPARK_HARVEST") {
                Ok(v) if !v.is_empty() => "set",
                _ => "unset",
            },
        );
        eprintln!(
            "[dspark] verify-window={:?} (accept-rate head {}, MEMRA_DSPARK_VT {})",
            DsparkVtPolicy::resolve(confidence.is_some()),
            if confidence.is_some() {
                "present"
            } else {
                "ABSENT -> ladder"
            },
            match std::env::var("MEMRA_DSPARK_VT") {
                Ok(v) if !v.is_empty() => "set",
                _ => "unset",
            },
        );
        Ok(Self {
            fc,
            hidden_norm: up("hidden_norm.weight", &[hidden])?,
            norm: up("norm.weight", &[hidden])?,
            cfg,
            layers,
            markov,
            confidence,
            rope_yarn,
            dflash2,
        })
    }

    /// Rope q or k rows in place: yarn (ff divisors + post-rope mscale) when the config
    /// carries it, plain neox otherwise. One primitive for all five drafter rope sites.
    fn rope_rows(
        &self,
        e: &Engine,
        x: &mut CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        n_heads: usize,
        n_tokens: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let c = &self.cfg;
        match &self.rope_yarn {
            Some((ff, mscale)) => {
                e.rope_neox_ff(
                    x,
                    pos_d,
                    c.head_dim,
                    c.head_dim,
                    n_heads,
                    n_tokens,
                    c.rope_theta,
                    1.0,
                    ff,
                )?;
                e.scale_inplace(x, *mscale, n_tokens * n_heads * c.head_dim)?;
            }
            None => {
                e.rope_neox(
                    x,
                    pos_d,
                    c.head_dim,
                    c.head_dim,
                    n_heads,
                    n_tokens,
                    c.rope_theta,
                    1.0,
                )?;
            }
        }
        Ok(())
    }

    /// f32 GEMM helper via the engine Float arm (cuBLASLt): y[t, out_f].
    fn mm(
        &self,
        e: &Engine,
        w: &GpuTensor,
        x: &CudaSlice<f32>,
        t: usize,
        _in_f: usize,
        _out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        e.matmul(w, x, t)
    }

    /// DFlash2 conv `prepare` (reference GroupedDynamicCausalConv.prepare): projects
    /// the pre-conv rows to BOTH dynamic kernels, convolves the rows with base half 0
    /// + dyn half 0, and returns (convolved rows, the dyn projection) — `finish`
    ///   reuses the SAME projection's half 1. Block-local causal shift (row 0 zero-pads).
    pub fn d2_conv_prepare(
        &self,
        e: &Engine,
        conv: &Dflash2Conv,
        xn: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let d2 = self
            .dflash2
            .as_ref()
            .expect("d2_conv on a non-dflash2 draft");
        let h = self.cfg.hidden;
        let groups = h / d2.group_size;
        let dyn_ = self.mm(e, &conv.proj, xn, rows, h, 2 * d2.conv_k * groups)?;
        let mut out = e.uninit(rows * h)?;
        e.dflash2_dynconv(
            xn,
            &dyn_,
            &conv.base,
            &mut out,
            rows,
            h,
            d2.group_size,
            d2.conv_k,
            0,
        )?;
        Ok((out, dyn_))
    }

    /// DFlash2 conv `finish`: convolves the sublayer OUTPUT rows with base half 1 +
    /// dyn half 1 (dyn from the matching `prepare`).
    pub fn d2_conv_finish(
        &self,
        e: &Engine,
        conv: &Dflash2Conv,
        y: &CudaSlice<f32>,
        dyn_: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let d2 = self
            .dflash2
            .as_ref()
            .expect("d2_conv on a non-dflash2 draft");
        let h = self.cfg.hidden;
        let mut out = e.uninit(rows * h)?;
        e.dflash2_dynconv(
            y,
            dyn_,
            &conv.base,
            &mut out,
            rows,
            h,
            d2.group_size,
            d2.conv_k,
            1,
        )?;
        Ok(out)
    }

    /// DFlash2 proposal (reference `DFlash2DraftModel.propose`, greedy arm): device
    /// top-k over the draft logits + the rank-`r` hidden projection, ONE small dtoh
    /// (~nd*(2k+rank) floats — the same per-round sync slot the markov chain's token
    /// readback occupies), then the host codebook walk. Returns the nd drafted tokens
    /// (mask-fill rows 1..b-1; the anchor row is not a draft).
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn dflash2_propose_greedy(
        &self,
        e: &Engine,
        dl: &CudaSlice<f32>,
        rows: &CudaSlice<f32>,
        nd: usize,
        n_vocab: usize,
        anchor: u32,
        d2t: Option<&[u32]>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        Ok(self
            .dflash2_propose_greedy_q(e, dl, rows, nd, n_vocab, anchor, d2t)?
            .0)
    }

    /// [`Self::dflash2_propose_greedy`] with the walk's per-slot confidence returned
    /// (lane/glm5-loop-port, 2026-08-30): q[p] = the chosen candidate's softmax mass over
    /// its slot's candidate set at T=1 — the statistic the glm5 loop's MEMRA_SPEC_PMIN
    /// tau-slot truncation thresholds on. Same walk, same path, same one-DtoH sync slot.
    #[allow(clippy::too_many_arguments)]
    // allow: mirrors the greedy propose contract it wraps
    pub fn dflash2_propose_greedy_q(
        &self,
        e: &Engine,
        dl: &CudaSlice<f32>,
        rows: &CudaSlice<f32>,
        nd: usize,
        n_vocab: usize,
        anchor: u32,
        d2t: Option<&[u32]>,
    ) -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        let d2 = self
            .dflash2
            .as_ref()
            .expect("dflash2_propose on a non-dflash2 draft");
        if n_vocab > d2.vocab || d2.top_k > n_vocab {
            return Err(format!(
                "DFlash2 proposal geometry invalid: target vocab {n_vocab}, selector vocab {}, top_k {}",
                d2.vocab, d2.top_k
            )
            .into());
        }
        if let Some(map) = d2t
            && map.len() < n_vocab
        {
            return Err(format!(
                "DFlash2 d2t has {} entries, fewer than proposal vocab {n_vocab}",
                map.len()
            )
            .into());
        }
        let (vals_d, idx_d) = e.topk_rows(dl, nd, n_vocab, d2.top_k)?;
        let hproj_d = e.matmul(&d2.hidden_proj, rows, nd)?;
        let unary = e.dtoh(&vals_d)?;
        let mut cand = e.dtoh_u32(&idx_d)?;
        // memra#95: refuse a selector row the top-k could not fill, BEFORE the d2t remap.
        dflash2_guard_candidates(&cand, n_vocab, "dflash2 greedy proposal")?;
        // TRIMMED draft head (lane/dflash2-head-trim, 2026-08-25): `dl` was scored over the
        // FR-Spec-gathered rows, so candidate index i names trimmed row i — remap to the true
        // token id BEFORE the selector walk (the codebooks and the verify block index the full
        // vocabulary). Same permute-the-proposal law as the MTP arm's spec.rs d2t map; verify
        // stays full-vocab, so the trim moves acceptance only, never output.
        if let Some(map) = d2t {
            for c in cand.iter_mut() {
                *c = map[*c as usize];
            }
        }
        let hproj = e.dtoh(&hproj_d)?;
        Ok(d2.walk_greedy_q(&unary, &cand, &hproj, anchor, nd))
    }

    /// DFlash2 proposal, SAMPLED arm (reference `DFlash2DraftModel.propose` at T>0): same
    /// device top-k + hidden projection + one dtoh as the greedy arm, then the host
    /// candidate-set softmax walk (`dflash2_walk_sampled`) drawing one host-Philox uniform
    /// per slot from the session's `uctr` stream. Returns (path, q_chosen, cand, q_rows).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dflash2_propose_sampled(
        &self,
        e: &Engine,
        dl: &CudaSlice<f32>,
        rows: &CudaSlice<f32>,
        nd: usize,
        n_vocab: usize,
        anchor: u32,
        temp: f32,
        seed: u64,
        uctr: &mut u32,
        d2t: Option<&[u32]>,
    ) -> Result<Dflash2SampledProposal, Box<dyn std::error::Error>> {
        let d2 = self
            .dflash2
            .as_ref()
            .expect("dflash2_propose on a non-dflash2 draft");
        if n_vocab > d2.vocab || d2.top_k > n_vocab {
            return Err(format!(
                "DFlash2 proposal geometry invalid: target vocab {n_vocab}, selector vocab {}, top_k {}",
                d2.vocab, d2.top_k
            )
            .into());
        }
        if let Some(map) = d2t
            && map.len() < n_vocab
        {
            return Err(format!(
                "DFlash2 d2t has {} entries, fewer than proposal vocab {n_vocab}",
                map.len()
            )
            .into());
        }
        let (vals_d, idx_d) = e.topk_rows(dl, nd, n_vocab, d2.top_k)?;
        let hproj_d = e.matmul(&d2.hidden_proj, rows, nd)?;
        let unary = e.dtoh(&vals_d)?;
        let mut cand = e.dtoh_u32(&idx_d)?;
        // memra#95: refuse a selector row the top-k could not fill, BEFORE the d2t remap.
        dflash2_guard_candidates(&cand, n_vocab, "dflash2 sampled proposal")?;
        // Trimmed-head remap — see the greedy arm. The q the walk reports is the softmax
        // over the candidate SET it actually proposed (ids are labels, not indices into a
        // distribution), so the rejection-verify contract is unchanged by the remap.
        if let Some(map) = d2t {
            for c in cand.iter_mut() {
                *c = map[*c as usize];
            }
        }
        let hproj = e.dtoh(&hproj_d)?;
        let mut draw = || {
            let u = crate::spec::host_u01(seed, *uctr);
            *uctr = uctr.wrapping_add(1);
            u
        };
        let (path, q_chosen, q_rows) =
            d2.walk_sampled(&unary, &cand, &hproj, anchor, nd, temp, &mut draw);
        Ok((path, q_chosen, cand, q_rows))
    }

    /// Sampled draft chain for the Rows families (T>0 twin of the greedy markov chain):
    /// slot k gets the markov bias of the PREVIOUS chain token added in place (when the
    /// head is armed — the sglang DSPARK worker's markov-corrected draft probs), then ONE
    /// draw from the row's FILTERED softmax (filter_stats -> device-stat gumbel perturb ->
    /// argmax into the chain buffer — the frspec eager-chain composition, stats kept on
    /// device so the chain stays sync-free like the greedy arm). Without a markov head the
    /// rows sample independently (the z-lab reference's T>0 arm for plain DFlash). `dl` is
    /// biased IN PLACE and retained by the caller: it is the accept walk's q source.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_chain_sampled(
        &self,
        e: &Engine,
        dl: &mut CudaSlice<f32>,
        nd: usize,
        n_vocab: usize,
        anchor: u32,
        sp: &crate::spec::SpecSampling,
        sctr: &mut u32,
        // H4 confidence-policy stash (v0.100 train merge): Some = copy each slot's
        // markov prev-token embedding (the exact `w1` row the chain gathers) into a
        // [nd, rank] buffer — the same d2d stash the greedy chain carries, so the
        // confidence window sizes identically at T>0.
        mut conf_emb: Option<&mut CudaSlice<f32>>,
    ) -> Result<(Vec<u32>, DsparkDraftSample), Box<dyn std::error::Error>> {
        let mut chain_d = e.stream().alloc_zeros::<u32>(nd + 1)?;
        e.set_u32_one(&mut chain_d, anchor)?;
        let mut th_all = e.zeros(nd)?;
        let mut z_all = e.zeros(nd)?;
        let mut mx_all = e.zeros(nd)?;
        let mut pb = e.zeros(n_vocab)?;
        for k in 0..nd {
            if let Some(mk) = &self.markov {
                let mut f = e.uninit(mk.rank)?;
                e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
                if let Some(ce) = conf_emb.as_deref_mut() {
                    let fv = e.view(&f, mk.rank);
                    e.copy_view_into(ce, k * mk.rank, &fv, mk.rank)?;
                }
                let bias = e.matmul(&mk.w2, &f, 1)?;
                e.add_row_inplace(dl, &bias, n_vocab, k * n_vocab)?;
            } else if let (Some(ce), Some(mk)) = (conf_emb.as_deref_mut(), &self.markov) {
                // MARKOV=0 arm still stashes the embedding for the confidence head —
                // the greedy chain's exact behavior.
                let mut f = e.uninit(mk.rank)?;
                e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
                let fv = e.view(&f, mk.rank);
                e.copy_view_into(ce, k * mk.rank, &fv, mk.rank)?;
            }
            let rows_k = e.htod_i32(&[k as i32])?;
            let (mut th1, mut z1, mut mx1) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
            e.filter_stats(
                dl, n_vocab, &rows_k, &mut th1, &mut z1, &mut mx1, n_vocab, 1, sp.temp, sp.top_k,
                sp.top_p, sp.min_p,
            )?;
            e.gumbel_perturb_filtered_col(
                dl, k, &mut pb, n_vocab, sp.seed, *sctr, sp.temp, &mx1, &th1, 0,
            )?;
            *sctr = sctr.wrapping_add(1);
            e.argmax_token_device_col(&pb, 0, n_vocab, &mut chain_d, k + 1)?;
            e.copy_into(&mut th_all, k, &th1, 1)?;
            e.copy_into(&mut z_all, k, &z1, 1)?;
            e.copy_into(&mut mx_all, k, &mx1, 1)?;
        }
        let chain = e.dtoh_u32(&chain_d)?;
        let (thv, zv, mxv) = (e.dtoh(&th_all)?, e.dtoh(&z_all)?, e.dtoh(&mx_all)?);
        let stats = (0..nd).map(|i| (mxv[i], thv[i], zv[i])).collect();
        Ok((
            chain[1..].to_vec(),
            DsparkDraftSample::Rows {
                th: th_all,
                z: z_all,
                stats,
            },
        ))
    }

    /// Family dispatch for the sampled proposal: Selector for DFlash2, Rows otherwise.
    /// Returns the drafted tokens (the round's `cand` tail) + the proposal record.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dspark_propose_sampled(
        &self,
        e: &Engine,
        dl: &mut CudaSlice<f32>,
        rows: &CudaSlice<f32>,
        nd: usize,
        n_vocab: usize,
        anchor: u32,
        sp: &crate::spec::SpecSampling,
        sctr: &mut u32,
        uctr: &mut u32,
        conf_emb: Option<&mut CudaSlice<f32>>,
        d2t: Option<&[u32]>,
    ) -> Result<(Vec<u32>, DsparkDraftSample), Box<dyn std::error::Error>> {
        if let Some(d2) = self.dflash2.as_ref() {
            // The confidence stash is a markov-family program; DFlash2 has no
            // accept-rate head (the policy resolver never arms it for this family).
            debug_assert!(
                conf_emb.is_none(),
                "conf_emb stash requested on a DFlash2 selector proposal"
            );
            let (path, q_chosen, cand, q_rows) = self.dflash2_propose_sampled(
                e, dl, rows, nd, n_vocab, anchor, sp.temp, sp.seed, uctr, d2t,
            )?;
            Ok((
                path,
                DsparkDraftSample::Selector {
                    cand,
                    q_rows,
                    q_chosen,
                    top_k: d2.top_k,
                },
            ))
        } else {
            self.dspark_chain_sampled(e, dl, nd, n_vocab, anchor, sp, sctr, conf_emb)
        }
    }

    /// FIRST-LIGHT forward (oracle contract): full non-causal attention over
    /// [ctx_features ; block], NO draft KV cache, NO sliding window (the oracle bypasses
    /// the reference mask machinery the same way — window/caching land in the round arm).
    ///
    /// `target_hidden`: [ctx, n_taps*hidden] (f32, device)  — raw tapped states.
    /// `noise_emb`:     [block, hidden] — target embed rows for [accepted, MASK x b-1].
    /// `pos`:           absolute positions for ctx rows THEN block rows (ctx+block i32).
    /// Returns final normed hidden [block, hidden] (feed target lm_head for draft logits).
    /// ctx features for `t` tapped rows: hidden_norm(fc(taps)) — the drafter's context
    /// representation, cacheable across rounds (append-only in committed-token order).
    pub fn ctx_features(
        &self,
        e: &Engine,
        taps: &CudaSlice<f32>,
        t: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let n_taps = c.target_layer_ids.len();
        let fc_out = self.mm(e, &self.fc, taps, t, n_taps * c.hidden, c.hidden)?;
        let mut out = e.uninit(t * c.hidden)?;
        e.rms_norm(&fc_out, &self.hidden_norm, &mut out, c.hidden, t, c.eps)?;
        Ok(out)
    }

    pub fn forward(
        &self,
        e: &Engine,
        target_hidden: &CudaSlice<f32>,
        noise_emb: &CudaSlice<f32>,
        pos: &[i32],
        ctx: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let ctx_f = self.ctx_features(e, target_hidden, ctx)?;
        if let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP") {
            let v = e.dtoh(&ctx_f)?;
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/memra-ctx_features.f32"), bytes)?;
        }
        self.forward_block(e, &ctx_f, noise_emb, pos, ctx)
    }

    /// Block forward over PRECOMPUTED ctx features (the round arm's entry: features are
    /// cached across rounds; only the block work repeats).
    pub fn forward_block(
        &self,
        e: &Engine,
        ctx_f: &CudaSlice<f32>,
        noise_emb: &CudaSlice<f32>,
        pos: &[i32],
        ctx: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv, c.head_dim);
        let b = c.block_size;
        assert_eq!(pos.len(), ctx + b, "pos covers ctx rows then block rows");

        let pos_blk = e.htod_i32(&pos[ctx..])?;

        let mut x = e.clone_dtod(noise_emb)?; // [b, hidden] residual stream
        for (li, l) in self.layers.iter().enumerate() {
            // input_layernorm on the block rows only (ctx features are norm-free per ref:
            // k/v project the SAME ctx_f every layer, un-layernormed).
            let mut xn = e.uninit(b * h)?;
            e.rms_norm(&x, &l.ln_in, &mut xn, h, b, c.eps)?;
            // DFlash2: dynamic conv WRAPS attention — q/k_noise/v_noise all project the
            // CONVOLVED block rows (reference decoder layer: prepare -> self_attn ->
            // finish, all inside the residual branch). ctx_f is never convolved.
            let mut attn_dyn: Option<CudaSlice<f32>> = None;
            if let Some(d2) = &self.dflash2 {
                let (xc, dyn_) = self.d2_conv_prepare(e, &d2.attn_conv[li], &xn, b)?;
                xn = xc;
                attn_dyn = Some(dyn_);
            }

            // q from block; k/v from [ctx_f ; block-normed]
            let q0 = self.mm(e, &l.wq, &xn, b, h, nh * hd)?;
            let k0c = self.mm(e, &l.wk, ctx_f, ctx, h, nkv * hd)?;
            let v0c = self.mm(e, &l.wv, ctx_f, ctx, h, nkv * hd)?;
            let k0b = self.mm(e, &l.wk, &xn, b, h, nkv * hd)?;
            let v0b = self.mm(e, &l.wv, &xn, b, h, nkv * hd)?;

            // per-head q/k rms norm (v passes through: ones weight trick not needed — the
            // qkv kernel norms rq+rk rows; concatenate k first).
            let mut k0 = e.uninit((ctx + b) * nkv * hd)?;
            e.copy_into(&mut k0, 0, &k0c, ctx * nkv * hd)?;
            e.copy_into(&mut k0, ctx * nkv * hd, &k0b, b * nkv * hd)?;
            let mut v = e.uninit((ctx + b) * nkv * hd)?;
            e.copy_into(&mut v, 0, &v0c, ctx * nkv * hd)?;
            e.copy_into(&mut v, ctx * nkv * hd, &v0b, b * nkv * hd)?;

            if li == 0
                && let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP")
            {
                let v = e.dtoh(&q0)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/memra-l0_q0.f32"), bytes)?;
            }
            let mut q = e.uninit(b * nh * hd)?;
            let mut k = e.uninit((ctx + b) * nkv * hd)?;
            // rms over head_dim rows: q has b*nh rows, k has (ctx+b)*nkv rows.
            e.rms_norm(&q0, &l.q_norm, &mut q, hd, b * nh, c.eps)?;
            if li == 0
                && let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP")
            {
                let v = e.dtoh(&q)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/memra-l0_qn.f32"), bytes)?;
            }
            e.rms_norm(&k0, &l.k_norm, &mut k, hd, (ctx + b) * nkv, c.eps)?;

            // rope: q at block positions, k at ctx-then-block positions (absolute).
            let norope = std::env::var("MEMRA_DFLASH_NOROPE").is_ok();
            if !norope {
                self.rope_rows(e, &mut q, &pos_blk, nh, b)?;
            }
            if li == 0
                && let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP")
            {
                let dump = |name: &str,
                            t: &cudarc::driver::CudaSlice<f32>|
                 -> Result<(), Box<dyn std::error::Error>> {
                    let v = e.dtoh(t)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/memra-l0_{name}.f32"), bytes)?;
                    Ok(())
                };
                dump("xn", &xn)?;
                dump("q_prerope", &q)?;
            }
            // k rows are laid out [row, nkv, hd] with row-major tokens — rope_neox expects
            // (n_heads, n_tokens); ctx and block ropes run as one call over ctx+b tokens.
            let pos_all = e.htod_i32(pos)?;
            if !norope {
                self.rope_rows(e, &mut k, &pos_all, nkv, ctx + b)?;
            }

            // full non-causal attention: every block query sees all ctx+b keys.
            let mut attn = e.uninit(b * nh * hd)?;
            let scale = 1.0f32 / (hd as f32).sqrt();
            // NAIVE SDPA for first light: fa_prefill's NON-CAUSAL arm with T != T_kv is
            // BROKEN (attn maxdiff 0.34 vs the torch oracle; q/k inputs bit-close — no
            // existing caller exercises that shape class, jsonl 2026-07-13). The 16 x
            // (ctx+16) block attention is tiny; the fa arm returns behind this seam once
            // its kernel is fixed + parity-gated.
            if self.dflash2.is_some() && c.layer_sliding[li] {
                // DFlash2 non-causal symmetric window (config is_causal=false, all
                // layers sliding). The kernel masks only keys OLDER than
                // q_pos-(window-1); the future side (k - q < window) never binds
                // because keys reach at most q_pos + block <= q_pos + window
                // (asserted at load). Positions must be contiguous — q_pos is derived
                // in-kernel as (T_kv - T) + qt.
                debug_assert!(pos.windows(2).all(|w| w[1] == w[0] + 1));
                d2_windowed_attn(
                    e,
                    &q,
                    &k,
                    &v,
                    &mut attn,
                    hd,
                    nh,
                    nkv,
                    b,
                    ctx + b,
                    scale,
                    c,
                    0,
                )?;
            } else if std::env::var("MEMRA_DFLASH_FA").is_ok() {
                e.fa_prefill(&q, &k, &v, &mut attn, hd, nh, nkv, b, ctx + b, scale, false)?;
            } else {
                e.sdpa_naive(&q, &k, &v, &mut attn, hd, nh, nkv, b, ctx + b, scale, false)?;
            }

            let mut o = self.mm(e, &l.wo, &attn, b, nh * hd, h)?;
            if let (Some(d2), Some(dyn_)) = (&self.dflash2, &attn_dyn) {
                o = self.d2_conv_finish(e, &d2.attn_conv[li], &o, dyn_, b)?;
            }
            let mut x1 = e.uninit(b * h)?;
            e.add(&o, &x, &mut x1, b * h)?;
            if li == 0
                && let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP")
            {
                let dump = |name: &str,
                            t: &cudarc::driver::CudaSlice<f32>|
                 -> Result<(), Box<dyn std::error::Error>> {
                    let v = e.dtoh(t)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/memra-l0_{name}.f32"), bytes)?;
                    Ok(())
                };
                dump("q", &q)?;
                dump("k", &k)?;
                dump("attn", &attn)?;
                dump("x1", &x1)?;
            }

            // mlp (DFlash2: the same conv wrap — prepare on the post-ln rows, mlp on
            // the convolved rows, finish on the mlp output, then the residual add)
            let mut x1n = e.uninit(b * h)?;
            e.rms_norm(&x1, &l.ln_post, &mut x1n, h, b, c.eps)?;
            let mut mlp_dyn: Option<CudaSlice<f32>> = None;
            if let Some(d2) = &self.dflash2 {
                let (xc, dyn_) = self.d2_conv_prepare(e, &d2.mlp_conv[li], &x1n, b)?;
                x1n = xc;
                mlp_dyn = Some(dyn_);
            }
            let gate = self.mm(e, &l.w_gate, &x1n, b, h, c.n_ff)?;
            let up_ = self.mm(e, &l.w_up, &x1n, b, h, c.n_ff)?;
            let mut act = e.uninit(b * c.n_ff)?;
            e.silu_mul(&gate, &up_, &mut act, b * c.n_ff)?;
            let mut down = self.mm(e, &l.w_down, &act, b, c.n_ff, h)?;
            if let (Some(d2), Some(dyn_)) = (&self.dflash2, &mlp_dyn) {
                down = self.d2_conv_finish(e, &d2.mlp_conv[li], &down, dyn_, b)?;
            }
            let mut x2 = e.uninit(b * h)?;
            e.add(&down, &x1, &mut x2, b * h)?;
            x = x2;
            if let Ok(dir) = std::env::var("MEMRA_DFLASH_DUMP") {
                let v = e.dtoh(&x)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/memra-layer{li}_out.f32"), bytes)?;
            }
        }
        let mut out = e.uninit(b * h)?;
        e.rms_norm(&x, &self.norm, &mut out, h, b, c.eps)?;
        Ok(out)
    }
}

/// Draft KV cache (round-cost fix, 2026-07-13): per-layer normed+roped ctx K and raw ctx V,
/// append-only in committed order. Block K/V land TRANSIENTLY at [len..len+b] each round
/// (never committed — the reference crops them identically). Kills the per-round full-ctx
/// projection recompute (first light was O(ctx)/round -> 7 tok/s).
pub struct DflashKv {
    pub k: Vec<CudaSlice<f32>>, // per layer [cap + block, nkv*hd]
    pub v: Vec<CudaSlice<f32>>,
    pub len: usize,
    pub cap: usize,
    /// Trailing rows the drafter can still observe: `sliding_window + block_size`. Carried on
    /// the KV (not recomputed at call sites) so an export and an import cannot disagree about
    /// the geometry — see `DsparkSpecSession::draft_tail_rows`.
    window_rows: usize,
    /// `n_kv * head_dim * size_of::<f32>()` — the row unit for tail copies.
    row_bytes: usize,
    /// CONTEXT FLOOR (lane/spec-exclusions-20260902): the first ctx row that EXISTS. `0` on
    /// every cold-primed KV and every full-tail import (the pre-lane shape). A COLD-DRAFTER
    /// KV (`new_cold_at`) or a short-tail import (`from_tail` of a tail whose exporter had
    /// a floor) owns rows only from here up; the round attention's window floor is raised
    /// to it (`d2_windowed_attn`), so the rows below are never read, and an export never
    /// publishes them (`export_tail` starts at the floor). The drafter then simply sees a
    /// shorter context, which can move ACCEPTANCE, never output — verify arbitrates.
    floor: usize,
}

impl DflashKv {
    pub fn new(
        e: &Engine,
        cfg: &DflashCfg,
        cap: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let rowsz = cfg.n_kv * cfg.head_dim;
        let mut k = Vec::with_capacity(cfg.n_layer);
        let mut v = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            k.push(e.uninit((cap + cfg.block_size) * rowsz)?);
            v.push(e.uninit((cap + cfg.block_size) * rowsz)?);
        }
        Ok(Self {
            k,
            v,
            len: 0,
            cap,
            window_rows: cfg.sliding_window.saturating_add(cfg.block_size),
            row_bytes: rowsz * std::mem::size_of::<f32>(),
            floor: 0,
        })
    }

    /// A COLD DRAFTER at a restored trunk boundary (lane/spec-exclusions-20260902, the
    /// `MEMRA_SPEC_WARM=1` arm): a fresh KV whose logical length is already `pos` (the
    /// restored prefix the trunk cache holds) but which owns NO ctx rows below it —
    /// `floor == len == pos`. The restored prefix's tap features do not exist (the trunk
    /// planes hold K/V latents, not the tapped residual rows), and re-running the trunk to
    /// recover them is the prime the restore exists to skip; so instead the drafter starts
    /// with an empty context at the right absolute position and fills it from the suffix
    /// prime's taps and every committed round from there, exactly as a cold session over a
    /// shorter prompt would. Row addressing (rope positions, `kv.len == cache.pos` at every
    /// round boundary) is identical to a tail import; only the attention floor differs.
    ///
    /// The `window_rows` below the floor are zero-filled so a later `export_tail` (which
    /// starts at the floor anyway) can never publish uninitialised bytes even under a
    /// future geometry mistake — finite zeros are the same belt `from_tail` wears.
    pub fn new_cold_at(
        e: &Engine,
        cfg: &DflashCfg,
        cap: usize,
        pos: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if pos > cap {
            return Err(
                format!("cold drafter position {pos} exceeds the session cap {cap}").into(),
            );
        }
        let mut kv = Self::new(e, cfg, cap)?;
        let rowsz = kv.row_bytes / std::mem::size_of::<f32>();
        let zero_from = pos.saturating_sub(kv.window_rows);
        if zero_from < pos {
            for li in 0..kv.k.len() {
                e.memset_zeros_view(&mut kv.k[li].slice_mut(zero_from * rowsz..pos * rowsz))?;
                e.memset_zeros_view(&mut kv.v[li].slice_mut(zero_from * rowsz..pos * rowsz))?;
            }
        }
        kv.len = pos;
        kv.floor = pos;
        Ok(kv)
    }

    /// The first ctx row this KV owns (doc on the field): `0` unless the KV was born as a
    /// cold drafter or imported from a short (floor-bearing) tail.
    pub fn floor(&self) -> usize {
        self.floor
    }
}

impl DflashDraft {
    /// Ingest `t` NEW ctx-feature rows (committed order, absolute positions `pos_new`) into
    /// the draft KV: per layer k/v projections + k head-norm + rope, appended at kv.len.
    pub fn ingest_ctx(
        &self,
        e: &Engine,
        kv: &mut DflashKv,
        feats: &CudaSlice<f32>,
        pos_new: &[i32],
        t: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nkv, hd) = (c.hidden, c.n_kv, c.head_dim);
        assert!(kv.len + t <= kv.cap, "draft kv overflow");
        let pos_d = e.htod_i32(pos_new)?;
        for (li, l) in self.layers.iter().enumerate() {
            let k0 = self.mm(e, &l.wk, feats, t, h, nkv * hd)?;
            let v0 = self.mm(e, &l.wv, feats, t, h, nkv * hd)?;
            let mut kn = e.uninit(t * nkv * hd)?;
            e.rms_norm(&k0, &l.k_norm, &mut kn, hd, t * nkv, c.eps)?;
            self.rope_rows(e, &mut kn, &pos_d, nkv, t)?;
            e.copy_into(&mut kv.k[li], kv.len * nkv * hd, &kn, t * nkv * hd)?;
            e.copy_into(&mut kv.v[li], kv.len * nkv * hd, &v0, t * nkv * hd)?;
        }
        kv.len += t;
        Ok(())
    }

    /// Block forward over the CACHED ctx KV: only the 16 block rows are projected per layer;
    /// block K/V land transiently at kv[len..len+b]. Bit-class-identical to forward_block
    /// (same kernels, same per-row programs; ONLY the ctx K/V recompute is cached).
    pub fn forward_round(
        &self,
        e: &Engine,
        kv: &mut DflashKv,
        noise_emb: &CudaSlice<f32>,
        pos_block: &[i32],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv, c.head_dim);
        let b = c.block_size;
        assert_eq!(pos_block.len(), b);
        let ctx = kv.len;
        let pos_blk = e.htod_i32(pos_block)?;
        let mut x = e.clone_dtod(noise_emb)?;
        for (li, l) in self.layers.iter().enumerate() {
            let mut xn = e.uninit(b * h)?;
            e.rms_norm(&x, &l.ln_in, &mut xn, h, b, c.eps)?;
            // DFlash2: dynamic conv wraps attention (see forward_block).
            let mut attn_dyn: Option<CudaSlice<f32>> = None;
            if let Some(d2) = &self.dflash2 {
                let (xc, dyn_) = self.d2_conv_prepare(e, &d2.attn_conv[li], &xn, b)?;
                xn = xc;
                attn_dyn = Some(dyn_);
            }
            let q0 = self.mm(e, &l.wq, &xn, b, h, nh * hd)?;
            let k0b = self.mm(e, &l.wk, &xn, b, h, nkv * hd)?;
            let v0b = self.mm(e, &l.wv, &xn, b, h, nkv * hd)?;
            let mut q = e.uninit(b * nh * hd)?;
            let mut kb = e.uninit(b * nkv * hd)?;
            e.rms_norm(&q0, &l.q_norm, &mut q, hd, b * nh, c.eps)?;
            e.rms_norm(&k0b, &l.k_norm, &mut kb, hd, b * nkv, c.eps)?;
            self.rope_rows(e, &mut q, &pos_blk, nh, b)?;
            self.rope_rows(e, &mut kb, &pos_blk, nkv, b)?;
            e.copy_into(&mut kv.k[li], ctx * nkv * hd, &kb, b * nkv * hd)?;
            e.copy_into(&mut kv.v[li], ctx * nkv * hd, &v0b, b * nkv * hd)?;
            let mut attn = e.uninit(b * nh * hd)?;
            let scale = 1.0f32 / (hd as f32).sqrt();
            if self.dflash2.is_some() && c.layer_sliding[li] {
                // Non-causal symmetric window (config is_causal=false): kv row index
                // == absolute position for BOTH ctx rows (committed order) and the
                // transient block rows, so the kernel's q_pos = (T_kv - T) + qt is the
                // absolute position and the old-side mask is exact. The future side
                // never binds (block <= window, asserted at load).
                d2_windowed_attn(
                    e,
                    &q,
                    &kv.k[li],
                    &kv.v[li],
                    &mut attn,
                    hd,
                    nh,
                    nkv,
                    b,
                    ctx + b,
                    scale,
                    c,
                    kv.floor,
                )?;
            } else if std::env::var("MEMRA_DFLASH_FA").is_ok() {
                e.fa_prefill(
                    &q,
                    &kv.k[li],
                    &kv.v[li],
                    &mut attn,
                    hd,
                    nh,
                    nkv,
                    b,
                    ctx + b,
                    scale,
                    false,
                )?;
            } else {
                e.sdpa_naive(
                    &q,
                    &kv.k[li],
                    &kv.v[li],
                    &mut attn,
                    hd,
                    nh,
                    nkv,
                    b,
                    ctx + b,
                    scale,
                    false,
                )?;
            }
            let mut o = self.mm(e, &l.wo, &attn, b, nh * hd, h)?;
            if let (Some(d2), Some(dyn_)) = (&self.dflash2, &attn_dyn) {
                o = self.d2_conv_finish(e, &d2.attn_conv[li], &o, dyn_, b)?;
            }
            let mut x1 = e.uninit(b * h)?;
            e.add(&o, &x, &mut x1, b * h)?;
            let mut x1n = e.uninit(b * h)?;
            e.rms_norm(&x1, &l.ln_post, &mut x1n, h, b, c.eps)?;
            let mut mlp_dyn: Option<CudaSlice<f32>> = None;
            if let Some(d2) = &self.dflash2 {
                let (xc, dyn_) = self.d2_conv_prepare(e, &d2.mlp_conv[li], &x1n, b)?;
                x1n = xc;
                mlp_dyn = Some(dyn_);
            }
            let gate = self.mm(e, &l.w_gate, &x1n, b, h, c.n_ff)?;
            let up_ = self.mm(e, &l.w_up, &x1n, b, h, c.n_ff)?;
            let mut act = e.uninit(b * c.n_ff)?;
            e.silu_mul(&gate, &up_, &mut act, b * c.n_ff)?;
            let mut down = self.mm(e, &l.w_down, &act, b, c.n_ff, h)?;
            if let (Some(d2), Some(dyn_)) = (&self.dflash2, &mlp_dyn) {
                down = self.d2_conv_finish(e, &d2.mlp_conv[li], &down, dyn_, b)?;
            }
            let mut x2 = e.uninit(b * h)?;
            e.add(&down, &x1, &mut x2, b * h)?;
            x = x2;
        }
        let mut out = e.uninit(b * h)?;
        e.rms_norm(&x, &self.norm, &mut out, h, b, c.eps)?;
        Ok(out)
    }
}

/// Emit an accepted draft run under the `max_new` budget: check BEFORE each push — at
/// real acceptance the final round often accepts a draft at the boundary, and
/// push-then-check emitted max_new+1 tokens (plain emits exactly max_new; the E2E gate
/// read it as a length divergence at index max_new with the shared prefix
/// byte-identical). f8300340cd fixed generate_spec_dspark this way; generate_spec_dflash
/// kept the buggy shape until the hermes sweep (fixed 2026-08-23) — both now share this
/// one helper. Returns true when the caller must break (budget reached or EOS emitted).
fn emit_accepted_run(out: &mut Vec<u32>, accepted: &[u32], eos: &[u32], max_new: usize) -> bool {
    for &dt in accepted {
        if out.len() >= max_new {
            return true;
        }
        out.push(dt);
        if eos.contains(&dt) {
            return true;
        }
    }
    false
}

// ===== THE DRAFT-SOURCE SEAM (lane/glm5-extract2, phase 2 of the extraction program) ======
//
// `DraftSourcePlan` (memra-gguf `model_plan.rs`) has always been general: it is the PLAN's
// statement of where a family's drafts come from. What was glm5-named was everything on the
// ENGINE side of it — the loaded-drafter holder, the flag-to-drafter load contract, and the
// tap-layer resolution. All three are family-agnostic by content, so they live here, in the
// general DFlash module, and glm5 is a CONSUMER.
//
// WHAT IS DELIBERATELY *NOT* HERE, and why (phase-1 discipline, restated):
// the PER-SESSION draft state (`glm_spec::Glm5DraftState`) and the source-keyed round /
// maintenance walks. Those are not family-agnostic today: each arm reaches into the family's
// own cache planes (glm5's MLA latent plane, `HcTapSink` hc-contract taps, the KDA rollback
// stash) and the retained-q type carries the family's rank space. A trait over them would have
// exactly ONE implementor whose associated types are all glm5 types — a decorative trait cut
// blind, on the hottest file in the lane program. The trigger for that cut is the SECOND
// hybrid spec family's session state, which is what tells us which half of the state is
// shared. The trait sketch is banked in the lane doc so the second consumer starts from it,
// not from scratch.

/// A loaded alternate draft source: the drafter weights plus the byte identity they were
/// pinned by. Model-level (loaded ONCE per model, on the head engine where the trunk lm_head
/// it projects through lives — the MTP-head placement law); per-session state is the family's.
///
/// Generalized from `glm_spec::Glm5DflashDrafter`, which stays re-exported under its old name
/// for the glm5 call sites and gates.
pub struct DflashDrafter {
    pub draft: DflashDraft,
    /// First 8 hex of sha256(model.safetensors) — the boot-receipt identity pin
    /// (`b33c0347` for the probe-pinned incoai/GLM-5.3-Flash-DFlash2 @ dc77ff1c bytes).
    pub sha8: String,
}

/// Resolve a drafter's tap layers against the trunk it will read features from.
///
/// PURE (no env, no engine): the drafter's own `target_layer_ids`, plus a caller-supplied
/// `shift` and the trunk bound. `shift` exists because the tap-shift RED ARM is a GATE
/// INSTRUMENT owned by the family that runs the gate (`MEMRA_GLM5_*_GATE_RED` is classified
/// as an instrument, never a serving flag, and never generalized) — the family reads its own
/// red-arm env, prints its own tag, and passes the shift in here.
pub fn resolve_tap_layers(
    target_layer_ids: &[usize],
    n_trunk: usize,
    shift: usize,
    what: &str,
) -> Result<Vec<usize>, String> {
    if target_layer_ids.is_empty() {
        return Err(format!("{what} drafter config carries no target_layer_ids"));
    }
    let taps: Vec<usize> = target_layer_ids.iter().map(|t| t + shift).collect();
    if let Some(&bad) = taps.iter().find(|&&t| t >= n_trunk) {
        return Err(format!(
            "{what} tap layer {bad} is outside the {n_trunk}-layer trunk"
        ));
    }
    Ok(taps)
}

/// Load a DFlash2 drafter named by `flag` from `dir`, validating every contract that binds a
/// drafter to a TARGET — family-agnostic, because each one is a property of the pair, not of
/// the family:
///
/// * the checkpoint is a `DFlash2DraftModel` (the selector family is the only draft source
///   this seam serves);
/// * `cfg.hidden == n_embd` (the drafter consumes the target's features and projects through
///   the target's embed/lm_head);
/// * `cfg.target_layer_ids` name valid trunk layers;
/// * `cfg.mask_token_id` is inside the target vocab.
///
/// A set flag that cannot load is a LOUD failure, never a silent plain fallback. Every error
/// is prefixed `{flag}={dir}` so the operator sees the flag they typed; the glm5 call site's
/// message bytes are unchanged by construction.
pub fn load_drafter(
    e: &Engine,
    dir: &std::path::Path,
    flag: &str,
    n_trunk: usize,
    n_embd: usize,
    n_vocab: usize,
) -> Result<DflashDrafter, String> {
    let dpath = dir.display();
    let draft = DflashDraft::load(e, dir)
        .map_err(|err| format!("{flag}={dpath}: drafter load failed: {err}"))?;
    if draft.dflash2.is_none() {
        return Err(format!(
            "{flag}={dpath}: checkpoint is not a DFlash2DraftModel \
             (the glm5 draft source is the selector family only)"
        ));
    }
    if draft.cfg.hidden != n_embd {
        return Err(format!(
            "{flag}={dpath}: drafter hidden {} != target n_embd {n_embd} \
             (the drafter consumes target features and the target's embed/lm_head)",
            draft.cfg.hidden
        ));
    }
    if draft.cfg.target_layer_ids.is_empty()
        || draft.cfg.target_layer_ids.iter().any(|&t| t >= n_trunk)
    {
        return Err(format!(
            "{flag}={dpath}: target_layer_ids {:?} do not name valid \
             trunk layers (n_trunk {n_trunk})",
            draft.cfg.target_layer_ids
        ));
    }
    if draft.cfg.mask_token_id as usize >= n_vocab {
        return Err(format!(
            "{flag}={dpath}: mask token {} outside the target vocab {n_vocab}",
            draft.cfg.mask_token_id
        ));
    }
    let sha8 = crate::hybrid::sha256_file_hex8(&dir.join("model.safetensors"))
        .map_err(|err| format!("{flag}={dpath}: sha256 pin: {err}"))?;
    Ok(DflashDrafter { draft, sha8 })
}

#[cfg(test)]
mod draft_source_seam_tests {
    use super::resolve_tap_layers;

    #[test]
    fn taps_resolve_and_the_shift_is_the_callers() {
        assert_eq!(
            resolve_tap_layers(&[1, 12, 23], 46, 0, "glm5 DFlash2").unwrap(),
            vec![1, 12, 23]
        );
        // the red arm's +1 rides in as a parameter, not as an env read in here
        assert_eq!(
            resolve_tap_layers(&[1, 12, 23], 46, 1, "glm5 DFlash2").unwrap(),
            vec![2, 13, 24]
        );
    }

    #[test]
    fn empty_and_out_of_trunk_taps_refuse_by_name() {
        let err = resolve_tap_layers(&[], 46, 0, "glm5 DFlash2").unwrap_err();
        assert!(err.contains("no target_layer_ids"), "{err}");
        let err = resolve_tap_layers(&[1, 46], 46, 0, "glm5 DFlash2").unwrap_err();
        assert!(
            err.contains("tap layer 46 is outside the 46-layer trunk"),
            "{err}"
        );
        // the SHIFTED tap is what gets bounds-checked — the red arm must not be able to
        // walk off the trunk silently
        let err = resolve_tap_layers(&[45], 46, 1, "glm5 DFlash2").unwrap_err();
        assert!(err.contains("tap layer 46 is outside"), "{err}");
    }
}

#[cfg(test)]
mod emit_budget_tests {
    use super::emit_accepted_run;

    #[test]
    fn accepted_run_never_exceeds_max_new() {
        // TOOTH (hermes finding, fixed 2026-08-23): the dflash accept loop pushed THEN
        // checked, emitting max_new+1 whenever the final round accepted at the boundary.
        let mut out = vec![1, 2, 3]; // 3 committed, budget 4: exactly ONE slot left
        let stop = emit_accepted_run(&mut out, &[10, 11, 12], &[], 4);
        assert!(stop, "hitting the budget must break the round loop");
        assert_eq!(
            out,
            vec![1, 2, 3, 10],
            "exactly max_new tokens, never max_new+1"
        );
        // EOS inside the run stops after emitting it (unchanged semantics).
        let mut out = vec![1];
        let stop = emit_accepted_run(&mut out, &[10, 99, 12], &[99], 8);
        assert!(stop);
        assert_eq!(out, vec![1, 10, 99]);
        // A run fitting the budget with no EOS lets the round continue.
        let mut out = vec![1];
        assert!(!emit_accepted_run(&mut out, &[10, 11], &[], 8));
        assert_eq!(out, vec![1, 10, 11]);
    }
}

// ================= DFlash spec round (greedy, first light) =================
// Exact contract: identical output stream to plain greedy decode BY CONSTRUCTION — the
// target's batched verify argmax decides every committed token; the drafter only proposes.
// (Same verify+rewind pattern as generate_spec_gemma's eager round; t=16 verify rides the
// straddle-split-safe fa_decode_rows.)
impl crate::hybrid::HybridModel {
    pub fn generate_spec_dflash(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        self.refuse_hyper("generate_spec_dflash")?;
        use crate::cache::{Cache, DflashTapSink};
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        assert!(
            draft.dflash2.is_none(),
            "DFlash2 drafters ride the qwen-hybrid dspark round (selector + windowed \
             attention); the gemma arm has no consumer for the family's ops"
        );
        assert_eq!(n_embd, c.hidden, "draft hidden must match target n_embd");
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        let max_ctx = prompt.len() + max_new + b + 8;
        // First light holds ctx <= sliding_window: the draft was trained with 4 sliding
        // layers (window 2048) and the first-light attention is windowless full — inside
        // the window the two are identical. The depth cell (1736 + 128) fits.
        assert!(
            max_ctx <= c.sliding_window,
            "first-light dflash round is windowless — ctx cap {} exceeds the draft window {}",
            max_ctx,
            c.sliding_window
        );
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // ---- prime with taps armed ----
        let tp = prompt.len();
        cache.dflash_taps = Some(DflashTapSink {
            layer_ids: c.target_layer_ids.clone(),
            buf: e.uninit(tp * n_taps * n_embd)?,
            hidden: n_embd,
            t: tp,
            base: 0,
        });
        let t_prime = std::time::Instant::now();
        let (logits, _h_seed, _hiddens) = self.prime_cache(e, prompt, &mut cache, 0)?;
        let mut last = crate::forward::argmax(&logits) as u32;
        // draft KV cache: ingest the prompt's ctx features once; per round only the kept
        // rows ingest + the block projects (round cost O(block), not O(ctx)).
        let mut dkv = DflashKv::new(e, &draft.cfg, max_ctx)?;
        {
            // CHUNKED ingest (depth OOM fix): the 1736-row prompt tap buffer is ~224MB f32;
            // running fc + 5-layer k/v projection over it in one shot stacks another
            // ~300MB of transients on the ~21.3GB trunk peak. 256-row windows bound the
            // transient set; identical values (row-independent ops).
            let taps = cache.dflash_taps.take().unwrap();
            let n_taps_h = n_taps * n_embd;
            let mut r0 = 0usize;
            while r0 < tp {
                let t_c = (tp - r0).min(256);
                let tv = e.view(&taps.buf, tp * n_taps_h);
                let win = tv.slice(r0 * n_taps_h..(r0 + t_c) * n_taps_h);
                let mut chunk = e.uninit(t_c * n_taps_h)?;
                e.copy_view_into(&mut chunk, 0, &win, t_c * n_taps_h)?;
                let f = draft.ctx_features(e, &chunk, t_c)?;
                let pos_c: Vec<i32> = ((r0 as i32)..(r0 + t_c) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_c, t_c)?;
                r0 += t_c;
            }
        }
        let mut ctx_len = tp;
        e.stream().synchronize()?;
        // published prime wall (the run-spec/gemma-gate timing contract subtracts it)
        crate::PRIME_NANOS.store(
            t_prime.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // embed-scale seam (MEMRA_DFLASH_EMB_SCALE): gemma trunks scale embeddings by
        // sqrt(n_embd) INSIDE the forward; whether the z-lab gemma4 training fed the
        // drafter scaled or raw embed rows is not visible from the reference (qwen path
        // uses raw embed_tokens). Acceptance arbitrates; default raw.
        let emb_scale = if std::env::var("MEMRA_DFLASH_EMB_SCALE").as_deref() == Ok("1") {
            (n_embd as f32).sqrt()
        } else {
            1.0
        };

        let mut out = Vec::with_capacity(max_new);
        let n_vocab = self.output.out_features();
        // VERIFY WIDTH (MEMRA_DFLASH_VERIFY_T, default 8): the drafter always drafts a full
        // block (its trained mask pattern) but only the first vt rows go through the target
        // verify — the t=16 verify rides the untuned b16 tier at ~32% of the byte wall
        // (65ms/verify) while b8 rides the tuned r2 tier; with ~2.7 committed/round the
        // deep block positions almost never survive anyway. Exactness unaffected (verify
        // still decides every committed token).
        let vt_cap: usize = std::env::var("MEMRA_DFLASH_VERIFY_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
            .clamp(2, b);
        // adaptive verify width (MEMRA_DFLASH_ADAPT!=0, MTP accepted+1 recipe): next round
        // verifies one past this round's accepted run, clamped [3, cap].
        let adapt = std::env::var("MEMRA_DFLASH_ADAPT").as_deref() != Ok("0");
        let mut vt = vt_cap;
        let mut attempted = 0usize;
        let mut accepted = 0usize;
        // The whole round runs in the decode-exact matmul scope: the m=16 draft mms were
        // otherwise falling into the prefill-GEMM class (770us/matmul, 17% of the depth
        // round). Prime (before this loop) keeps the prefill GEMM path. RAII: a `?` exit
        // anywhere in the loop restores the pre-scope value instead of latching exact ON
        // engine-wide (hermes finding, fixed 2026-08-23).
        let exact_scope = e.exact_scope(true);
        'outer: while out.len() < max_new {
            let start = cache.pos; // committed length
            // ---- draft: block = [last, MASK x b-1] ----
            let mut block: Vec<u32> = vec![c.mask_token_id; b];
            block[0] = last;
            let mut noise = e.htod(&self.embd.try_gather(n_embd, &block)?)?;
            if emb_scale != 1.0 {
                e.scale_inplace(&mut noise, emb_scale, b * n_embd)?;
            }
            if std::env::var("MEMRA_DFLASH_DEBUG").as_deref() == Ok("1") && start == cache.pos {
                let nv = e.dtoh(&noise)?;
                let r0: f32 = nv[..n_embd].iter().map(|x| x * x).sum::<f32>().sqrt();
                let r1: f32 = nv[n_embd..2 * n_embd]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                eprintln!(
                    "[dflash noise] |row0(last)|={r0:.3} |row1(MASK id {})|={r1:.3}",
                    c.mask_token_id
                );
            }
            let pos_block: Vec<i32> = ((start as i32)..(start + b) as i32).collect();
            let dh = draft.forward_round(e, &mut dkv, &noise, &pos_block)?;
            // draft tokens = argmax(lm_head(h rows 1..b))
            let mut rows = e.uninit((b - 1) * n_embd)?;
            {
                let dv = e.view(&dh, b * n_embd);
                let tail = dv.slice(n_embd..b * n_embd);
                e.copy_view_into(&mut rows, 0, &tail, (b - 1) * n_embd)?;
            }
            let mut dl = e.matmul(&self.output, &rows, b - 1)?;
            // SEMI-AR MARKOV CHAIN (DSpark head, when present):
            // left-to-right, logits_k += W2(W1[prev realized token]) — the whole chain
            // stays on-device (chain_d[0] = the pending token; argmax k writes
            // chain_d[k+1], the k+1 bias gathers from it). Greedy mirror of the patch's
            // _markov_semiar_sample_block.
            let mut chain_d = e.stream().alloc_zeros::<u32>(b)?;
            if let Some(mk) = &draft.markov {
                e.set_u32_one(&mut chain_d, last)?;
                for k in 0..(b - 1) {
                    let mut f = e.uninit(mk.rank)?;
                    e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
                    let bias = e.matmul(&mk.w2, &f, 1)?;
                    e.add_row_inplace(&mut dl, &bias, n_vocab, k * n_vocab)?;
                    e.argmax_token_device_col(&dl, k, n_vocab, &mut chain_d, k + 1)?;
                }
            } else {
                for i in 0..(b - 1) {
                    e.argmax_token_device_col(&dl, i, n_vocab, &mut chain_d, i + 1)?;
                }
            }
            let chain = e.dtoh_u32(&chain_d)?;
            let dtoks = &chain[1..];
            for (i, &dt) in dtoks.iter().enumerate() {
                block[i + 1] = dt;
            }
            let dbg = std::env::var("MEMRA_DFLASH_DEBUG").as_deref() == Ok("1");

            // ---- verify: one t=vt target forward with taps armed ----
            let vblock = &block[..vt];
            cache.dflash_taps = Some(DflashTapSink {
                layer_ids: c.target_layer_ids.clone(),
                buf: e.uninit(vt * n_taps * n_embd)?,
                hidden: n_embd,
                t: vt,
                base: 0,
            });
            let (vam, _vh) = self.gemma4_decode_step_t_am(e, vblock, start, &mut cache)?;
            let taps = cache.dflash_taps.take().unwrap();
            if dbg {
                eprintln!(
                    "[dflash r] start={start} last={last}\n  draft={:?}\n  vam  ={:?}",
                    &block[1..],
                    vam
                );
            }

            // ---- accept ----
            let mut m = 0usize;
            while m < vt - 1 && block[m + 1] as usize == vam[m] as usize {
                m += 1;
            }
            attempted += vt - 1;
            accepted += m;
            out.push(last);
            if eos.contains(&last) {
                break 'outer;
            }
            if emit_accepted_run(&mut out, &block[1..=m], eos, max_new) {
                break 'outer;
            }
            let next = vam[m];

            // ---- commit/rollback: keep m+1 of the b appended rows ----
            let keep = m + 1;
            for kvl in cache.kv.iter_mut().flatten() {
                kvl.len -= vt - keep;
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            cache.pos -= vt - keep;

            // ---- ingest the kept rows' ctx features into the draft KV ----
            {
                let tv = e.view(&taps.buf, vt * n_taps * n_embd);
                let keep_view = tv.slice(0..keep * n_taps * n_embd);
                let mut kept = e.uninit(keep * n_taps * n_embd)?;
                e.copy_view_into(&mut kept, 0, &keep_view, keep * n_taps * n_embd)?;
                let f = draft.ctx_features(e, &kept, keep)?;
                let pos_k: Vec<i32> = ((ctx_len as i32)..(ctx_len + keep) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_k, keep)?;
                ctx_len += keep;
            }
            last = next;
            if adapt {
                vt = (m + 2).clamp(3, vt_cap);
            }
        }
        drop(exact_scope);
        if std::env::var("MEMRA_SPEC_STATS").as_deref() == Ok("1") {
            eprintln!(
                "[dflash] acceptance {accepted}/{attempted} = {:.3}",
                accepted as f64 / attempted.max(1) as f64
            );
        }
        Ok(out)
    }
}

// ================= Engine-bundle slice 1: batched GDN state snapshot ====================
// DSF-ROUNDCOST-20260820 §1.1 measured the dspark round's `cache.snapshot(e)` at 0.67 ms
// native wall — 48 linear layers x {conv, ssm} x (alloc_zeros + memcpy_dtod) of pure
// dispatch serialization, zero kernels. This batcher holds ONE persistent CacheSnapshot
// (buffers allocated on round 1, reused every round — kills the per-round alloc/memset
// churn) plus device pointer tables, so a round's snap is one small H2D table refresh
// (the ssm handles ping-pong per verify row, so live pointers are re-read each round;
// conv handles are rolled in place and never move) + TWO `copy_batch_uniform_f32`
// launches. Bytes, buffers and stream order are identical to `Cache::snapshot`; only the
// dispatch count changes, so acceptance and streams stay bit-identical (E2E-gated).

pub(crate) struct DsparkSnapBatch {
    pub(crate) snap: crate::cache::CacheSnapshot,
    /// Linear-attention layer indices, in `conv_table`/`ssm_table` order.
    lin: Vec<usize>,
    /// [src_0..src_{n-1}, dst_0..dst_{n-1}] — live conv states -> snapshot conv buffers.
    conv_table: CudaSlice<u64>,
    ssm_table: CudaSlice<u64>,
    host_ssm: Vec<u64>,
    conv_words: usize,
    ssm_words: usize,
}

impl DsparkSnapBatch {
    /// Build from a fresh full snapshot (this IS round 1's snap — the caller uses
    /// `self.snap` directly after `new`). Returns None when the cache has no linear
    /// layers or their state sizes are non-uniform (a future hybrid shape) — the caller
    /// then stays on the legacy per-layer snapshot rather than copying wrong byte counts.
    pub(crate) fn new(
        e: &Engine,
        cache: &crate::cache::Cache,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let snap = cache.snapshot(e)?;
        let lin: Vec<usize> = (0..cache.recur.len())
            .filter(|&il| cache.recur[il].is_some())
            .collect();
        if lin.is_empty() {
            return Ok(None);
        }
        let first = cache.recur[lin[0]].as_ref().unwrap();
        let (conv_words, ssm_words) = (first.conv_state.len(), first.ssm_state.len());
        for &il in &lin {
            let rl = cache.recur[il].as_ref().unwrap();
            if rl.conv_state.len() != conv_words || rl.ssm_state.len() != ssm_words {
                return Ok(None);
            }
        }
        let n = lin.len();
        let mut host_conv = vec![0u64; 2 * n];
        let mut host_ssm = vec![0u64; 2 * n];
        {
            let s = &e.gpu.stream();
            for (k, &il) in lin.iter().enumerate() {
                let rl = cache.recur[il].as_ref().unwrap();
                let (pc, _g0) = rl.conv_state.device_ptr(s);
                let (ps, _g1) = rl.ssm_state.device_ptr(s);
                let (dc, _g2) = snap.conv[il].as_ref().unwrap().device_ptr(s);
                let (ds, _g3) = snap.ssm[il].as_ref().unwrap().device_ptr(s);
                host_conv[k] = pc;
                host_conv[n + k] = dc;
                host_ssm[k] = ps;
                host_ssm[n + k] = ds;
            }
        }
        let conv_table = e.htod_u64(&host_conv)?;
        let ssm_table = e.htod_u64(&host_ssm)?;
        Ok(Some(Self {
            snap,
            lin,
            conv_table,
            ssm_table,
            host_ssm,
            conv_words,
            ssm_words,
        }))
    }

    /// The per-round snap: refresh kv lens/pos host-side (as `snapshot_into` does),
    /// re-read the live ssm handles into the table (gdn ping-pong moves them; the conv
    /// handles and every snapshot dst are stable), then two batched-copy launches.
    pub(crate) fn refresh(
        &mut self,
        e: &Engine,
        cache: &crate::cache::Cache,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        for il in 0..cache.kv.len() {
            self.snap.kv_len[il] = cache.kv[il].as_ref().map(|kvl| kvl.len);
        }
        self.snap.pos = cache.pos;
        let n = self.lin.len();
        {
            let s = &e.gpu.stream();
            for (k, &il) in self.lin.iter().enumerate() {
                let rl = cache.recur[il].as_ref().unwrap();
                let (ps, _g) = rl.ssm_state.device_ptr(s);
                self.host_ssm[k] = ps;
            }
        }
        e.htod_u64_into(&self.host_ssm, &mut self.ssm_table)?;
        e.copy_batch_uniform_f32(&self.conv_table, n, self.conv_words)?;
        e.copy_batch_uniform_f32(&self.ssm_table, n, self.ssm_words)?;
        Ok(())
    }
}

// ================= DSpark spec round, QWEN-HYBRID target (lane/dspark-q38-recover) =====
// The q38 twin of generate_spec_dflash. Same drafter machinery (rounds, markov chain,
// draft KV, adaptive verify width); the TARGET side swaps gemma4's dense verify for the
// qwen serving-class verify funnel (dspark_verify_t_am) + snapshot/rollback, because the
// hybrid GDN conv/ssm state mutates in place — dense KV truncation cannot roll it back.
// Exactness contract unchanged: identical stream to plain greedy BY CONSTRUCTION (the
// target's verify argmax decides every committed token).
impl crate::hybrid::HybridModel {
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_dspark(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
        sampling: Option<&crate::spec::SpecSampling>,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        self.refuse_hyper("generate_spec_dspark")?;
        use crate::cache::{Cache, DflashTapSink};
        assert!(
            !self.uses_gemma_program(),
            "gemma4 targets use generate_spec_dflash; this is the qwen-hybrid arm"
        );
        // SAMPLED ADMISSION (T>0, lane/dspark-sampled-admission-20260820): Some+temp>0
        // routes the round's proposal/accept through the rejection-sampling arms; None or
        // temp==0 keeps every greedy path byte-identical (the exactness instrument).
        let sp_on: Option<&crate::spec::SpecSampling> = sampling.filter(|s| s.temp > 0.0);
        // PENALTIES AT T==0 ARE A LOUD REFUSAL (lane/dspark-penalized-sampled-20260821):
        // the greedy walk argmaxes RAW verify columns, so a temp==0 config carrying
        // non-identity penalties would silently serve the UNPENALIZED greedy stream —
        // exactly the H-class silent-program-switch this route refuses everywhere else.
        // Penalized greedy stays on the plain path (worker admission owns the exclusion).
        if let Some(s) = sampling
            && s.temp <= 0.0
            && s.pen_on()
        {
            return Err(
                "dspark spec at temp==0 is the greedy route and would silently drop \
                     the request's penalties; penalized greedy is served on the plain path"
                    .into(),
            );
        }
        // Penalized-sampled state: the session window (pen_window_seed — one definition
        // across both spec routes), extended with every committed token; each round's
        // accept receives the trimmed tail (min(penalty_last_n, PEN_WINDOW_MAX)).
        let pen_on = sp_on.is_some_and(|s| s.pen_on());
        let mut pen_hist: Vec<u32> = if pen_on {
            crate::spec::pen_window_seed(&[], prompt, sp_on.unwrap().penalty_last_n)
        } else {
            Vec::new()
        };
        let (mut sctr, mut uctr) = (0u32, 0u32);
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        assert_eq!(n_embd, c.hidden, "draft hidden must match target n_embd");
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        let max_ctx = prompt.len() + max_new + b + 8;
        // DFlash2 implements the reference's non-causal symmetric sliding window in
        // the round attention (sdpa_naive_w), so depth past the window is admitted;
        // other families keep the historical windowless contract.
        assert!(
            draft.dflash2.is_some() || max_ctx <= c.sliding_window,
            "dspark round is windowless — ctx cap {} exceeds the draft window {}",
            max_ctx,
            c.sliding_window
        );
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // ---- prime with taps armed (chunked prime writes at chunk offsets via sink.base) ----
        let tp = prompt.len();
        cache.dflash_taps = Some(DflashTapSink {
            layer_ids: c.target_layer_ids.clone(),
            buf: e.uninit(tp * n_taps * n_embd)?,
            hidden: n_embd,
            t: tp,
            base: 0,
        });
        let t_prime = std::time::Instant::now();
        let (logits, _h_seed, _hiddens) = self.prime_cache(e, prompt, &mut cache, 0)?;
        // Boundary token: greedy takes the argmax (byte contract); sampled draws it from
        // the request's own filtered target through the session Philox stream — the same
        // shipped composition the frspec route uses (sample_check arm 9 oracles it).
        let mut last = match sp_on {
            Some(sp) => crate::spec::sample_boundary_token(
                e,
                &logits,
                sp,
                &pen_hist,
                &mut sctr,
                "dspark-prime",
            )?,
            None => crate::forward::argmax(&logits) as u32,
        };
        let mut dkv = DflashKv::new(e, &draft.cfg, max_ctx)?;
        {
            let taps = cache.dflash_taps.take().unwrap();
            let n_taps_h = n_taps * n_embd;
            let mut r0 = 0usize;
            while r0 < tp {
                let t_c = (tp - r0).min(256);
                let tv = e.view(&taps.buf, tp * n_taps_h);
                let win = tv.slice(r0 * n_taps_h..(r0 + t_c) * n_taps_h);
                let mut chunk = e.uninit(t_c * n_taps_h)?;
                e.copy_view_into(&mut chunk, 0, &win, t_c * n_taps_h)?;
                let f = draft.ctx_features(e, &chunk, t_c)?;
                let pos_c: Vec<i32> = ((r0 as i32)..(r0 + t_c) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_c, t_c)?;
                r0 += t_c;
            }
        }
        let mut ctx_len = tp;
        e.stream().synchronize()?;
        crate::PRIME_NANOS.store(
            t_prime.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        let mut out = Vec::with_capacity(max_new);
        let n_vocab = self.output.out_features();
        // Harvest convention (DSPARK-POSTMORTEM-20260820.md): which drafter output rows
        // become draft candidates. nd = drafts/round; verify carries [anchor, drafts]
        // = up to nd+1 rows. FAMILY-keyed for DFlash2 (mask-fill by construction),
        // else default = the CHECKPOINT's own strategy census (owner-ratified flip,
        // 2026-08-20); explicit env still wins (contradiction refuses).
        let harvest = DsparkHarvest::for_draft(draft);
        let nd = harvest.n_drafts(b);
        let r0 = harvest.first_row();
        let vt_cap: usize = std::env::var("MEMRA_DFLASH_VERIFY_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(nd + 1)
            .clamp(2, nd + 1);
        let adapt = std::env::var("MEMRA_DFLASH_ADAPT").as_deref() != Ok("0");
        // Verify-window policy (H4, DSPARK-POSTMORTEM-20260820.md): default =
        // confidence-slot tau=.5 when the checkpoint carries an accept-rate head
        // (owner-ratified flip 2026-08-20; cell-3 tau ladder knee) — each round's
        // window is sized from the head's own slot scores, post-draft pre-verify.
        // Head-less checkpoints and MEMRA_DFLASH_ADAPT=0 keep the reactive ladder.
        let vt_policy = DsparkVtPolicy::resolve(draft.confidence.is_some());
        if vt_policy.is_confidence() {
            assert!(
                draft.confidence.is_some(),
                "MEMRA_DSPARK_VT={vt_policy:?} needs a checkpoint with an accept-rate \
                 head (confidence_head.* absent in this export)"
            );
        }
        let mut vt = vt_cap;
        let mut attempted = 0usize;
        let mut accepted = 0usize;
        // Engine-bundle slice 1: persistent batched snapshot (None until round 1; stays
        // None — legacy per-layer snapshot — when the batcher declines the cache shape).
        let mut snapb: Option<DsparkSnapBatch> = None;
        let mut snapb_off = false;
        // Engine-bundle slice 2: deferred chain readback needs the resident embed table
        // (verify then embeds chain_d directly). Ladder/stash arms only — the confidence
        // policies size vt from a pre-verify head readback and keep the legacy order.
        let defer_rb = !vt_policy.is_confidence();
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if !defer_rb || crate::spec::spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        // Engine-bundle slice 3: per-(segment, vt) verify graphs for the linear-layer runs
        // (rides the slice-2 deferred path only — device tokens keep the whole verify off
        // the host). PERSISTENT across generations on the model (rebuilding per call
        // re-captured ~80 graphs per prompt — measured 97.8 -> 79.1 tok/s e2e); the
        // captured bodies are cache-independent: all state reads go through per-round
        // refreshed pointer tables and ctx-owned slabs. None = eager walk, byte-identical.
        let mut vg_guard = self.dspark_vgraphs.lock().unwrap();
        if vg_guard.is_none() && embd_gpu.is_some() && crate::spec::dspark_verify_graph_on() {
            *vg_guard = crate::spec::DsparkVerifyGraphs::new(e, &cache, vt_cap, n_embd)?;
        }
        let vgraphs: &mut Option<crate::spec::DsparkVerifyGraphs> = &mut vg_guard;
        // per-phase economics counters (ns) — the verify-toll dataset
        let (mut ns_draft, mut ns_snap, mut ns_verify, mut ns_roll, mut ns_ingest) =
            (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut rounds = 0usize;
        let stats = std::env::var("MEMRA_SPEC_STATS").as_deref() == Ok("1");
        let clock = |on: bool, e: &Engine| -> std::time::Instant {
            if on {
                let _ = e.stream().synchronize();
            }
            std::time::Instant::now()
        };
        'outer: while out.len() < max_new {
            rounds += 1;
            let start = cache.pos; // committed length
            // ---- draft: block = [last, MASK x b-1] (decode-exact class for the m=b mms) ----
            let t0 = clock(stats, e);
            // RAII: a `?` exit restores the pre-scope value instead of latching exact
            // ON engine-wide (hermes finding, fixed 2026-08-23).
            let exact_scope = e.exact_scope(true);
            let mut block: Vec<u32> = vec![c.mask_token_id; b];
            block[0] = last;
            let noise = e.htod(&self.embd.try_gather(n_embd, &block)?)?;
            let pos_block: Vec<i32> = ((start as i32)..(start + b) as i32).collect();
            let dh = draft.forward_round(e, &mut dkv, &noise, &pos_block)?;
            // Harvest: logits over rows r0..r0+nd (Dflash: mask rows 1..b-1, fill
            // semantics; Dspark: ALL b rows, shifted semantics — row k predicts
            // anchor+k+1, so col k of `dl` is the draft for position start+k+1).
            let mut rows = e.uninit(nd * n_embd)?;
            {
                let dv = e.view(&dh, b * n_embd);
                let src = dv.slice(r0 * n_embd..(r0 + nd) * n_embd);
                e.copy_view_into(&mut rows, 0, &src, nd * n_embd)?;
            }
            // TRIMMED DRAFT HEAD (lane/dflash2-head-trim, 2026-08-25): DFlash2 family
            // only — the selector consumes (value, candidate-id) pairs, so a d2t remap
            // after top-k restores true ids; the markov/chain arms argmax dl columns
            // into token ids DIRECTLY and must keep the full head. Reuses the FR-Spec
            // self-trim the load path builds on the MTP struct (MEMRA_FRSPEC_TRIM):
            // gathered rows of the target's own head, zero requant. Verify stays
            // full-vocab, so the trim moves draft acceptance only, never output.
            let trim = if draft.dflash2.is_some() {
                self.mtp
                    .as_ref()
                    .filter(|m| m.d2t_from_target_head)
                    .and_then(|m| m.shared_head_head.as_ref().zip(m.d2t.as_ref()))
                    // MEMRA_MTP_SKIP stub: the same target-head trimmed rows, parked in
                    // `dflash_trim` because the embedded MTP block was skipped (hybrid.rs;
                    // rows are target-head by construction; the loader refuses otherwise).
                    .or_else(|| self.dflash_trim.as_ref().map(|t| (&t.head, &t.d2t)))
                    .filter(|(_, d2t)| !d2t.is_empty())
            } else {
                None
            };
            let (dl_head, dl_vocab) = match trim {
                Some((head, d2t)) => (head, d2t.len()),
                None => (&self.output, n_vocab),
            };
            let trim_d2t = trim.map(|(_, d2t)| d2t.as_slice());
            let mut dl = e.matmul(dl_head, &rows, nd)?;
            // Family/sampling-keyed proposal (v0.100 train merge of the port and H4/
            // engine-bundle stacks — BOTH programs preserved):
            //  - SAMPLED (sp_on): rejection-sampling proposal, records the true per-slot
            //    q (family-keyed inside: selector for DFlash2, markov-corrected rows
            //    otherwise). Host CDF/readback syncs inside — slice-2 deferral N/A.
            //  - DFlash2 greedy: the candidate path selector REPLACES the markov chain
            //    (reference DFlash2DraftModel.propose — greedy arm).
            //  - markov/plain greedy chain: the engine-bundle arm; slice-2 readback
            //    deferral decided below (needs the ckpt arm reads).
            // Confidence policy: stash each slot's markov prev-token embedding (the
            // exact `w1` row the chain gathers) into a [nd, rank] buffer — d2d async,
            // read back beside `rows` in one host sync after the chain.
            let want_conf_emb = vt_policy.is_confidence()
                && draft.confidence.as_ref().is_some_and(|ch| ch.with_markov);
            let mut conf_emb: Option<CudaSlice<f32>> = match (&draft.markov, want_conf_emb) {
                (Some(mk), true) => Some(e.uninit(nd * mk.rank)?),
                (None, true) => unreachable!(
                    "with_markov confidence head without a markov table — the loader forbids it"
                ),
                _ => None,
            };
            let mut cand: Vec<u32> = Vec::with_capacity(nd + 1);
            let mut prop: Option<DsparkDraftSample> = None;
            let mut chain_dev: Option<CudaSlice<u32>> = None;
            if let Some(sp) = sp_on {
                let (tail, ds) = draft.dspark_propose_sampled(
                    e,
                    &mut dl,
                    &rows,
                    nd,
                    dl_vocab,
                    last,
                    sp,
                    &mut sctr,
                    &mut uctr,
                    conf_emb.as_mut(),
                    trim_d2t,
                )?;
                drop(exact_scope);
                cand.push(last);
                cand.extend_from_slice(&tail);
                prop = Some(ds);
            } else if draft.dflash2.is_some() {
                let path =
                    draft.dflash2_propose_greedy(e, &dl, &rows, nd, dl_vocab, last, trim_d2t)?;
                drop(exact_scope);
                cand.push(last);
                cand.extend_from_slice(&path);
            } else {
                let mut chain_d = e.stream().alloc_zeros::<u32>(nd + 1)?;
                if let Some(mk) = &draft.markov {
                    e.set_u32_one(&mut chain_d, last)?;
                    for k in 0..nd {
                        let mut f = e.uninit(mk.rank)?;
                        e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
                        if let Some(ce) = conf_emb.as_mut() {
                            let fv = e.view(&f, mk.rank);
                            e.copy_view_into(ce, k * mk.rank, &fv, mk.rank)?;
                        }
                        let bias = e.matmul(&mk.w2, &f, 1)?;
                        e.add_row_inplace(&mut dl, &bias, n_vocab, k * n_vocab)?;
                        e.argmax_token_device_col(&dl, k, n_vocab, &mut chain_d, k + 1)?;
                    }
                } else {
                    if want_conf_emb {
                        // chain_d[0] must carry the anchor — slot 0's prev token.
                        e.set_u32_one(&mut chain_d, last)?;
                    }
                    for i in 0..nd {
                        if let (Some(ce), Some(mk)) = (conf_emb.as_mut(), &draft.markov) {
                            let mut f = e.uninit(mk.rank)?;
                            e.gather_row_bf16(&mk.w1_bf16, &chain_d, i, &mut f, mk.rank)?;
                            let fv = e.view(&f, mk.rank);
                            e.copy_view_into(ce, i * mk.rank, &fv, mk.rank)?;
                        }
                        e.argmax_token_device_col(&dl, i, n_vocab, &mut chain_d, i + 1)?;
                    }
                }
                drop(exact_scope);
                chain_dev = Some(chain_d);
            }
            // MEMRA_DSPARK_CKPT (default 1): verify with the MTP column-stash armed so a
            // partial accept restores state directly. =0 keeps the snapshot+replay arm
            // (the oracle the stash arm is gated against — MEMRA_DSPARK_CKPT_GATE=1 runs
            // BOTH per partial round and byte-compares the resulting cache state).
            // Read here (was at the verify site) — slice 2's deferral needs the arm
            // choice before deciding whether the chain readback can move past verify.
            let ckpt_on = std::env::var("MEMRA_DSPARK_CKPT").as_deref() != Ok("0");
            let ckpt_gate = std::env::var("MEMRA_DSPARK_CKPT_GATE").as_deref() == Ok("1");
            // SAMPLED x ckpt-gate refusal: the gate compares verify argmaxes across a
            // replay — a greedy-exactness instrument (port lane). Refuse loudly.
            if sp_on.is_some() && ckpt_gate {
                return Err(
                    "MEMRA_DSPARK_CKPT_GATE compares verify argmaxes across a replay \
                            — a greedy-exactness instrument; unset it for T>0 dspark rounds"
                        .into(),
                );
            }
            // Slice 2: under the stash/gate arms with a resident embed table, the GREEDY
            // chain readback is DEFERRED past verify dispatch and merged with the argmax
            // readback into one sync. The replay arm (CKPT=0) verifies host tokens and
            // keeps the legacy order; the sampled and DFlash2 proposals already synced
            // at the walk (chain_dev is None there).
            let deferred = chain_dev.is_some() && embd_gpu.is_some() && (ckpt_on || ckpt_gate);
            // ---- H4 confidence window: size THIS round's verify from the head ----
            if vt_policy.is_confidence() {
                let ch = draft.confidence.as_ref().expect("asserted at loop entry");
                let (rows_h, emb_h) = match conf_emb.as_ref() {
                    Some(ce) => {
                        let (a, b2) = e.dtoh_pair(&rows, ce)?;
                        (a, Some(b2))
                    }
                    None => (e.dtoh(&rows)?, None),
                };
                let rank = draft.markov.as_ref().map(|m| m.rank).unwrap_or(0);
                let mut raws = Vec::with_capacity(nd);
                for k in 0..nd {
                    let hrow = &rows_h[k * n_embd..(k + 1) * n_embd];
                    let emb = emb_h.as_ref().map(|eh| &eh[k * rank..(k + 1) * rank]);
                    raws.push(ch.raw_score(hrow, emb));
                }
                vt = vt_policy
                    .size_window(&raws, vt_cap)
                    .expect("confidence policies always size the window");
            }
            // Verify candidates: [anchor, draft 1..nd]. Under Dflash this is the
            // historical `block` content; under Dspark it is one longer than the
            // drafter's input block (nd = b drafts + the anchor). The sampled/DFlash2
            // proposals built `cand` at the walk; deferred greedy rounds build it after
            // the merged readback — the bytes are identical (chain_d is written before
            // either sync).
            if let Some(chain_d) = chain_dev.as_ref()
                && !deferred
            {
                let chain = e.dtoh_u32(chain_d)?;
                cand.push(last);
                cand.extend_from_slice(&chain[1..]);
            }
            ns_draft += clock(stats, e).duration_since(t0).as_nanos() as u64;

            // ---- snapshot (GDN conv/ssm state + KV lens), then verify t=vt ----
            let t1 = std::time::Instant::now();
            // Slice 1: batched snap (one table refresh + two copy launches) with the
            // legacy per-layer snapshot as the kill-switch / non-uniform fallback.
            let mut snap_legacy: Option<crate::cache::CacheSnapshot> = None;
            if !snapb_off && snapb.is_none() {
                snapb = DsparkSnapBatch::new(e, &cache)?;
                snapb_off = snapb.is_none();
            } else if let Some(sb) = snapb.as_mut() {
                sb.refresh(e, &cache)?;
            }
            let snap: &crate::cache::CacheSnapshot = match snapb.as_ref() {
                Some(sb) => &sb.snap,
                None => {
                    snap_legacy = Some(cache.snapshot(e)?);
                    snap_legacy.as_ref().unwrap()
                }
            };
            let _ = &snap_legacy;
            ns_snap += clock(stats, e).duration_since(t1).as_nanos() as u64;
            let t2 = std::time::Instant::now();
            // Slice 3: the tap-sink buffer is persistent per vt in the graphs ctx
            // (captured segments bake its address); fully rewritten by every verify.
            let tap_buf = match vgraphs.as_mut().and_then(|g| g.tap_bufs.remove(&vt)) {
                Some(buf) => buf,
                None => e.uninit(vt * n_taps * n_embd)?,
            };
            cache.dflash_taps = Some(DflashTapSink {
                layer_ids: c.target_layer_ids.clone(),
                buf: tap_buf,
                hidden: n_embd,
                t: vt,
                base: 0,
            });
            // The whole fallible verify window runs inside a closure so the Err path can
            // return the sink buffer to the ctx pool before propagating (v0.98 review
            // carry-over): five `?`s span the window, and an early return would drop
            // `cache.dflash_taps` — freeing the buffer whose ADDRESS the model-persistent
            // captured graphs bake, so the next generation's replayed tap copies would
            // write freed memory. The never-orphan invariant below now holds on EVERY
            // exit, not just the EOS/budget break.
            let verify_res = (|cache: &mut crate::cache::Cache,
                               cand: &mut Vec<u32>,
                               vgraphs: &mut Option<crate::spec::DsparkVerifyGraphs>|
             -> Result<
                (
                    Vec<u32>,
                    Option<CudaSlice<f32>>,
                    Option<crate::spec::DsparkVerifyCkpt>,
                ),
                Box<dyn std::error::Error>,
            > {
                if sp_on.is_some() {
                    // SAMPLED: keep the raw verify logits — the accept walk gathers
                    // filtered p from them (argmaxes are the greedy arm's instrument,
                    // not this one's).
                    if ckpt_on {
                        let (tl, vck) =
                            self.dspark_verify_t_logits_ckpt(e, &cand[..vt], start, cache)?;
                        Ok((Vec::new(), Some(tl), Some(vck)))
                    } else {
                        Ok((
                            Vec::new(),
                            Some(self.dspark_verify_t_logits(e, &cand[..vt], start, cache)?),
                            None,
                        ))
                    }
                } else if deferred {
                    // Slice 2: verify embeds the DEVICE chain (cand layout by construction:
                    // chain_d[0] = anchor, chain_d[1..] = drafts), then ONE host sync reads
                    // chain + verify argmaxes together — the host dispatched snap + all of
                    // verify while the draft was still executing.
                    let chain_d = chain_dev.as_ref().expect("deferred implies greedy chain");
                    let g = embd_gpu.expect("deferred implies resident embed");
                    let (am_d, vck) = self.dspark_verify_t_am_ckpt_dev(
                        e,
                        chain_d,
                        vt,
                        start,
                        cache,
                        (g, embd_qt, embd_rb),
                        vgraphs.as_mut(),
                    )?;
                    let ch = e.stream().clone_dtoh(chain_d)?;
                    let am = e.stream().clone_dtoh(&am_d)?;
                    e.stream().synchronize()?;
                    cand.push(last);
                    cand.extend_from_slice(&ch[1..]);
                    Ok((am, None, Some(vck)))
                } else if ckpt_on || ckpt_gate {
                    let (vam, vck) = self.dspark_verify_t_am_ckpt(e, &cand[..vt], start, cache)?;
                    Ok((vam, None, Some(vck)))
                } else {
                    Ok((
                        self.dspark_verify_t_am(e, &cand[..vt], start, cache)?,
                        None,
                        None,
                    ))
                }
            })(&mut cache, &mut cand, vgraphs);
            let (vam, tl, vck) = match verify_res {
                Ok(v) => v,
                Err(err) => {
                    if let (Some(g), Some(taps)) = (vgraphs.as_mut(), cache.dflash_taps.take()) {
                        g.tap_bufs.insert(vt, taps.buf);
                    }
                    return Err(err);
                }
            };
            let taps = cache.dflash_taps.take().unwrap();
            // Return the tap buffer to the ctx pool IMMEDIATELY — an EOS/budget break
            // between accept and ingest must never orphan an address the captured
            // graphs bake (the next generation would alloc a fresh buffer and the
            // replayed tap copies would write freed memory). Ingest reads it borrowed.
            let tap_local: Option<CudaSlice<f32>> = match vgraphs.as_mut() {
                Some(g) => {
                    g.tap_bufs.insert(vt, taps.buf);
                    None
                }
                None => Some(taps.buf),
            };
            let tap_ref: &CudaSlice<f32> = match &tap_local {
                Some(b) => b,
                None => &vgraphs.as_ref().expect("ctx present above").tap_bufs[&vt],
            };
            ns_verify += clock(stats, e).duration_since(t2).as_nanos() as u64;

            // ---- accept ----
            // Penalized-sampled: the anchor `last` is committed THIS round unconditionally
            // (the out.push below), so it joins the window before the accept walk — verify
            // row 0's state includes it. Accepted drafts extend the window after the walk;
            // `next` joins as the anchor of ITS round.
            if pen_on {
                pen_hist.push(last);
            }
            let (m, next) = match (sp_on, tl.as_ref()) {
                (Some(sp), Some(tl)) => {
                    let w0 = pen_hist
                        .len()
                        .saturating_sub(sp.penalty_last_n.min(crate::spec::PEN_WINDOW_MAX));
                    dspark_accept_sampled(
                        e,
                        tl,
                        &cand,
                        vt,
                        n_vocab,
                        &dl,
                        prop.as_ref()
                            .expect("sampled round without a proposal record"),
                        sp,
                        &pen_hist[w0..],
                        &mut sctr,
                        &mut uctr,
                    )?
                }
                _ => {
                    let m = dspark_accept_prefix(&cand, &vam, vt);
                    (m, vam[m])
                }
            };
            if pen_on {
                pen_hist.extend_from_slice(&cand[1..=m]);
            }
            attempted += vt - 1;
            accepted += m;
            out.push(last);
            if eos.contains(&last) {
                break 'outer;
            }
            if emit_accepted_run(&mut out, &cand[1..=m], eos, max_new) {
                break 'outer;
            }

            // ---- commit/rollback: hybrid state cannot truncate — restore + replay kept ----
            let keep = m + 1;
            let t3 = std::time::Instant::now();
            // Slice 3: rounds whose linear column stash lives in the graphs ctx's slabs
            // commit through the slab twin (same semantics, slab-addressed sources).
            let slab_commit = vgraphs.as_ref().map(|g| g.round_slab).unwrap_or(false);
            if keep < vt {
                if ckpt_gate {
                    // GATE ARM: stash-restore, snapshot S1; then the replay oracle, snapshot
                    // S2; the two cache states must match BIT-FOR-BIT (kv lens, pos, every
                    // conv/ssm buffer). Continue from the replay state (proven identical).
                    if slab_commit {
                        self.dspark_commit_prefix_slab(
                            e,
                            &mut cache,
                            snap,
                            vgraphs.as_ref().expect("slab_commit implies ctx"),
                            keep,
                        )?;
                    } else {
                        let vck = vck.as_ref().expect("gate arm always fills the ckpt");
                        self.dspark_commit_prefix(e, &mut cache, snap, vck, keep)?;
                    }
                    // host-side state capture (NO device snapshot copies — two extra
                    // device snapshots per round OOM'd beside the 15GB trunk)
                    #[allow(clippy::type_complexity)]
                    // allow: one-shot composite type; naming it would hide the shape that matters at the call site
                    let capture = |cache: &Cache| -> Result<
                        (usize, Vec<Option<usize>>, Vec<(Vec<f32>, Vec<f32>)>),
                        Box<dyn std::error::Error>,
                    > {
                        let mut lens = Vec::new();
                        let mut states = Vec::new();
                        for il in 0..cache.kv.len() {
                            lens.push(cache.kv[il].as_ref().map(|k| k.len));
                            if let Some(rl) = &cache.recur[il] {
                                states.push((e.dtoh(&rl.conv_state)?, e.dtoh(&rl.ssm_state)?));
                            }
                        }
                        Ok((cache.pos, lens, states))
                    };
                    let (p1, l1, st1) = capture(&cache)?;
                    crate::pp::restore_cache_checkpoint(e, self, None, &mut cache, snap)?;
                    let ram = self.dspark_verify_t_am(e, &cand[..keep], start, &mut cache)?;
                    assert_eq!(
                        &ram[..],
                        &vam[..keep],
                        "prefix replay must reproduce the verify argmaxes"
                    );
                    let (p2, l2, st2) = capture(&cache)?;
                    assert_eq!(p1, p2, "ckpt-gate: pos mismatch");
                    assert_eq!(l1, l2, "ckpt-gate: kv_len mismatch");
                    for (il, ((c1, s1v), (c2, s2v))) in st1.iter().zip(&st2).enumerate() {
                        let bits = |a: &[f32], b: &[f32]| {
                            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                        };
                        assert!(
                            bits(c1, c2),
                            "ckpt-gate: linear layer {il} conv state differs"
                        );
                        assert!(
                            bits(s1v, s2v),
                            "ckpt-gate: linear layer {il} ssm state differs"
                        );
                    }
                } else if slab_commit {
                    // STASH ARM, slab twin (slice 3): same restore, slab-addressed.
                    self.dspark_commit_prefix_slab(
                        e,
                        &mut cache,
                        snap,
                        vgraphs.as_ref().expect("slab_commit implies ctx"),
                        keep,
                    )?;
                } else if let Some(vck) = vck.as_ref() {
                    // STASH ARM (default): column-state restore, no replay forward.
                    self.dspark_commit_prefix(e, &mut cache, snap, vck, keep)?;
                } else {
                    // REPLAY ARM (MEMRA_DSPARK_CKPT=0): the original snapshot+replay oracle.
                    crate::pp::restore_cache_checkpoint(e, self, None, &mut cache, snap)?;
                    debug_assert_eq!(cache.pos, start, "rollback landed off the round start");
                    let ram = self.dspark_verify_t_am(e, &cand[..keep], start, &mut cache)?;
                    if sp_on.is_none() {
                        // the argmax-reproduction oracle is greedy-only; the sampled arm
                        // replays purely to rebuild the cache state.
                        debug_assert_eq!(
                            &ram[..],
                            &vam[..keep],
                            "prefix replay must reproduce the verify argmaxes"
                        );
                    }
                }
            }
            ns_roll += clock(stats, e).duration_since(t3).as_nanos() as u64;

            // ---- ingest the kept rows' ctx features into the draft KV ----
            let t4 = std::time::Instant::now();
            {
                let tv = e.view(tap_ref, vt * n_taps * n_embd);
                let keep_view = tv.slice(0..keep * n_taps * n_embd);
                let mut kept = e.uninit(keep * n_taps * n_embd)?;
                e.copy_view_into(&mut kept, 0, &keep_view, keep * n_taps * n_embd)?;
                let f = draft.ctx_features(e, &kept, keep)?;
                let pos_k: Vec<i32> = ((ctx_len as i32)..(ctx_len + keep) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_k, keep)?;
                ctx_len += keep;
            }
            ns_ingest += clock(stats, e).duration_since(t4).as_nanos() as u64;
            last = next;
            // Ladder update only — under the confidence policies vt is recomputed
            // from the head every round, post-draft pre-verify.
            if !vt_policy.is_confidence() && adapt {
                vt = (m + 2).clamp(3, vt_cap);
            }
        }
        if stats {
            let ms = |n: u64| n as f64 / 1e6;
            eprintln!(
                "[dspark-q38] acceptance {accepted}/{attempted} = {:.3} rounds={rounds} \
                 draft={:.1}ms snap={:.1}ms verify={:.1}ms rollback+replay={:.1}ms ingest={:.1}ms",
                accepted as f64 / attempted.max(1) as f64,
                ms(ns_draft),
                ms(ns_snap),
                ms(ns_verify),
                ms(ns_roll),
                ms(ns_ingest)
            );
        }
        Ok(out)
    }
}

// ================= DSpark SERVING session (lane/dspark-q38-recover serve route) =========
// Burst-scoped state for the worker's dspark spec arm — the qwen-hybrid twin of
// GemmaSpecSession. Holds the trunk cache + draft KV + the round loop's carry state
// (`last`, ctx_len, adaptive vt) so the scheduler round-robins other sessions between
// bursts. The round body is generate_spec_dspark's loop, hoisted; that bin arm stays the
// banked oracle (E2E gate), and the serve-route smoke gates this twin byte-identical to
// a spec-off boot over the real HTTP surface. Exactness contract unchanged: the target's
// verify argmax decides every committed token, so the stream equals plain greedy BY
// CONSTRUCTION on every accept path (ckpt stash, gate, replay).
fn take_dspark_prefix_capture(
    slot: &mut Option<crate::spec::SpecBoundaryCapture>,
) -> Option<crate::spec::SpecBoundaryCapture> {
    slot.take()
}

/// Deterministic preflight for the serving session's prompt-headroom requirement. Kept pure so
/// the worker can make the same decision before choosing whether to consume a prefix entry.
pub fn dspark_spec_prompt_fits(
    prompt_len: usize,
    ctx_cap: usize,
    block_size: usize,
    sliding_window: usize,
    is_dflash2: bool,
) -> bool {
    // PRIME FLOOR (incident 2026-08-25, second hit — the one that actually took prod down
    // twice). This predicate is the ONE admission gate the worker consumes for the dspark
    // route, and it only ever checked the ctx CEILING. A prompt shorter than
    // `PRIME_MIN_T` was therefore admitted and then panicked inside the cold prime, because
    // `prime_cache`'s batched arm asserts `T >= PRIME_MIN_T` and has no tokenwise twin that
    // fills the DFlash tap sink. A panic there is not a failed request: the GPU worker
    // exits 70 (poisoned-context contract) and every live session on the box dies, then the
    // guard relaunches into the same prompt — 20 panics and ~5 min of edge 502s on box10,
    // and a second loop on BOTH boxes when the route was redeployed. The trigger is
    // ordinary traffic: "Say OK." is 5 tokens, and our own watchdog sends that class.
    // Below the floor the route simply declines and the request serves on the plain path.
    if prompt_len < crate::hybrid_forward::PRIME_MIN_T {
        return false;
    }
    let max_ctx = if is_dflash2 {
        ctx_cap
    } else {
        ctx_cap.min(sliding_window)
    };
    prompt_len
        .checked_add(block_size)
        .and_then(|n| n.checked_add(8))
        .is_some_and(|need| need <= max_ctx)
}

pub struct DsparkSpecSession {
    pub cache: crate::cache::Cache,
    /// One-shot prompt-end state for the worker's cross-request prefix cache. DFlash has no
    /// restorable draft plane, so this capture deliberately carries trunk snapshot + logits
    /// only; low-load DFlash requests ignore the resulting trunk-only entry while a later
    /// shed-to-plain request can consume it.
    prefix_capture: Option<crate::spec::SpecBoundaryCapture>,
    dkv: DflashKv,
    last: u32,
    ctx_len: usize,
    vt: usize,
    pub rounds: usize,
    max_ctx: usize,
    done: bool,
    /// Engine-bundle slice 1: persistent batched snapshot (buffers + pointer tables live
    /// with the session so bursts reuse them). None until the first round; stays None —
    /// legacy per-layer snapshot — when `snapb_off`.
    snapb: Option<DsparkSnapBatch>,
    snapb_off: bool,
    /// SAMPLED ADMISSION (T>0, lane/dspark-sampled-admission-20260820): the request's
    /// sampling config (None/temp==0 = the greedy route, byte-identical). Fixed for the
    /// session — the worker's admission owns the sampler identity.
    sampling: Option<crate::spec::SpecSampling>,
    /// Philox event counters, session-owned so randomness never repeats across bursts
    /// (the frspec session-continuity law): `sctr` = device sampling events (boundary,
    /// draft chain, bonus, residual), `uctr` = host uniforms (selector walk, accept tests).
    sctr: u32,
    uctr: u32,
    /// Penalized-sampled window (lane/dspark-penalized-sampled-20260821): seeded from
    /// the prompt tail (`pen_window_seed`), extended with every committed token, carried
    /// across bursts so a burst boundary never resets the stream the client asked us to
    /// penalize. Empty (and never touched) when the request carries no penalties.
    pen_hist: Vec<u32>,
}

fn dspark_commit_limit(
    accepted_keep: usize,
    burst_out_len: usize,
    request_room: usize,
) -> (usize, bool) {
    let public_room = request_room.saturating_sub(burst_out_len);
    debug_assert!(public_room > 0);
    let keep = accepted_keep.min(public_room);
    (keep, keep < accepted_keep)
}

impl DsparkSpecSession {
    /// How many trailing draft-KV rows a restore must carry for the drafter to be
    /// indistinguishable from one that cold-primed: the sliding window plus one block.
    ///
    /// WHY A TAIL IS SUFFICIENT, and why this is a fact about THIS export rather than a hope:
    /// every DFlash2 draft layer is `sliding_attention` (the port asserts
    /// `cfg.layer_sliding.iter().all(|&s| s)` at load and refuses otherwise), so the windowed
    /// SDPA never reads a key below the current block's window floor
    /// (`sdpa_naive_w_lo`, whose bit-identity at Tkv 4104 and legacy launch failure are both
    /// pinned by kernel_check). A round at context `pos` therefore reads rows
    /// `[pos - window + 1, pos + block)` and nothing older. Storing that tail is storing
    /// everything the drafter can observe.
    ///
    /// SIZE, the reason this is affordable at all: 5 layers x (2048 + 16) rows x 8 kv x 128
    /// dim x 4 B x 2 (k+v) is ~85 MB, against ~1,057 MB for the trunk planes of a
    /// 30k-token entry. Storing the FULL draft history instead would be ~1,229 MB — more than
    /// the trunk entry itself — which is what makes the tail the only viable form.
    pub fn draft_tail_rows(&self) -> usize {
        self.dkv.cfg_window_rows()
    }

    /// The drafter's KV, for a worker publishing the tail into its cross-request prefix cache.
    pub fn draft_kv(&self) -> &DflashKv {
        &self.dkv
    }
}

/// The tail-import refusal arms, PURE so they are testable without CUDA. These are the fence
/// in front of the deliberate uninitialised-rows-below-`base` design: rows the import does not
/// copy are unreadable ONLY if the tail actually covers the drafter's window ending exactly at
/// the logical length — every arm here is what makes that "only if" hold. A refusal that
/// silently stopped firing would let a session attend garbage without crashing, which is the
/// silent-quality-loss class, so each arm names itself.
///
/// `tail_floor` (lane/spec-exclusions-20260902): the EXPORTER's context floor, `0` for
/// every tail a cold-primed drafter publishes (the pre-lane rule verbatim). A floor-bearing
/// exporter (`DflashKv::new_cold_at`, or itself a short-tail import) never owned rows below
/// it, so its tail legitimately covers only `[floor, len)`; the coverage rule then asks for
/// everything readable ABOVE the floor. A short tail whose exporter had NO floor is still the
/// refusal it always was.
#[allow(clippy::too_many_arguments)]
pub fn tail_geometry_ok(
    tail_layers: usize,
    tail_row_bytes: usize,
    tail_base: usize,
    tail_rows: usize,
    tail_len: usize,
    tail_floor: usize,
    kv_layers: usize,
    kv_row_bytes: usize,
    kv_window_rows: usize,
    cap: usize,
) -> Result<(), &'static str> {
    if tail_layers != kv_layers {
        return Err("layer count differs from the live drafter");
    }
    if tail_row_bytes != kv_row_bytes {
        return Err("row geometry differs from the live drafter");
    }
    if tail_len > cap {
        return Err("logical length exceeds the session cap");
    }
    if tail_base + tail_rows != tail_len {
        return Err("tail does not end at its own logical length");
    }
    if tail_floor > tail_base {
        return Err("tail starts below its exporter's context floor");
    }
    // The whole point of the tail: it must cover everything a round can read. A shorter
    // tail than the window is only acceptable when the tail IS the entire history above the
    // exporter's floor (floor 0: the entire history).
    if tail_rows < kv_window_rows.min(tail_len - tail_floor) {
        return Err("tail shorter than the drafter's readable window");
    }
    Ok(())
}

/// A DFlash draft-KV tail, per drafter layer, ready to ride a cross-request prefix-cache
/// entry: `(k, v)` f32 rows covering absolute positions `[base, base + rows)`.
///
/// Only the tail travels, and that is a fact about this export rather than an optimisation:
/// every DFlash2 draft layer is `sliding_attention` (the port asserts it at load), so a round
/// at context `pos` reads rows `[pos - window + 1, pos + block)` and nothing older. Storing
/// the whole history for a 30k-token prompt would be ~1,229 MB — MORE than the ~1,057 MB of
/// trunk planes it would ride with; the tail is ~85 MB.
pub struct DflashKvTail {
    pub layers: Vec<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// Absolute position of the first stored row.
    pub base: usize,
    /// Rows stored per layer.
    pub rows: usize,
    /// Logical length the KV had when exported (`= pos`), so an import can restore the same
    /// absolute row addressing the rope positions were baked against.
    pub len: usize,
    /// Bytes per row per layer, carried so an import cannot disagree about the geometry.
    pub row_bytes: usize,
    /// The exporter's context floor (`DflashKv::floor`): `0` for every tail a cold-primed
    /// drafter publishes; the first row the exporter ever owned otherwise. Travels with the
    /// tail so `tail_geometry_ok` can tell a legitimately short tail (nothing below the
    /// floor ever existed) from a truncated one, and so the import inherits the floor.
    pub floor: usize,
}

impl DflashKvTail {
    pub fn bytes(&self) -> usize {
        self.layers.len() * self.rows * self.row_bytes * 2
    }
}

impl DflashKv {
    /// Copy out the readable tail ending at `upto` (see `DflashKvTail`). `None` when there is
    /// nothing to publish or an allocation fails — publication is always optional.
    ///
    /// `upto` IS NOT `self.len`, and conflating them was the bug the first exactness-gate run
    /// caught: publication happens at the scheduler's drain sweep, by which time the session
    /// has committed generated rows, so `len` had run 35 rows past the capture boundary and
    /// every restore was refused with `draft KV len 30364 != prompt 30329`. The trunk planes
    /// are copied at the capture `pos` for the same reason; the tail must agree with them.
    pub fn export_tail(&self, e: &Engine, upto: usize) -> Option<DflashKvTail> {
        if upto == 0 || upto > self.len {
            return None;
        }
        let rowsz = self.row_bytes / std::mem::size_of::<f32>();
        // Never below the floor: a floor-bearing KV owns no rows there (doc on `floor`), and
        // a tail with nothing above the floor has nothing to publish.
        let floor = self.floor.min(upto);
        let rows = self.window_rows.min(upto - floor);
        if rows == 0 {
            return None;
        }
        let base = upto - rows;
        let mut layers = Vec::with_capacity(self.k.len());
        for li in 0..self.k.len() {
            let (Ok(mut k), Ok(mut v)) = (e.uninit(rows * rowsz), e.uninit(rows * rowsz)) else {
                return None;
            };
            if e.copy_range_into(&mut k, 0, &self.k[li], base * rowsz, rows * rowsz)
                .is_err()
                || e.copy_range_into(&mut v, 0, &self.v[li], base * rowsz, rows * rowsz)
                    .is_err()
            {
                return None;
            }
            layers.push((k, v));
        }
        Some(DflashKvTail {
            layers,
            base,
            rows,
            len: upto,
            row_bytes: self.row_bytes,
            floor,
        })
    }

    /// Rebuild a draft KV from a published tail: a fresh allocation at `cap`, the tail copied
    /// back to the SAME absolute rows it came from, and `len` restored so the next round
    /// addresses positions exactly as a cold-primed session would.
    ///
    /// Rows below `tail.base` are ZEROED, not left uninitialised. The clipped SDPA never reads
    /// below the block's window floor, but the legacy full-scan kernel (removed 2026-09-05)
    /// scanned EVERY row into the score and the
    /// output, relying on masked rows contributing exactly zero — an identity that holds only
    /// for finite data (`0.0 * NaN = NaN`, and an uninit K row can produce a NaN score that
    /// poisons the softmax sum). Zeros keep that identity on both kernel arms, so a clip
    /// rollback on a restore-armed box stays byte-exact instead of decoding silent garbage
    /// (review round 3). Rows above `tail.len` stay uninit — equally unwritten and unread in
    /// the cold path, so restored matches cold there.
    ///
    /// This function still REFUSES rather than trusts the window math — if the tail does not
    /// cover the window, the caller gets `None` and must cold-prime.
    ///
    /// FLOOR-BEARING TAILS (lane/spec-exclusions-20260902): a tail whose exporter had a
    /// context floor covers only `[floor, len)` and the import inherits `floor = tail.base`
    /// (the first row it actually owns), so the round attention never reads the zero rows
    /// below it (the clipped kernel, `d2_windowed_attn`). Full tails keep `floor = 0` and the
    /// pre-lane program exactly.
    pub fn from_tail(e: &Engine, cfg: &DflashCfg, cap: usize, tail: &DflashKvTail) -> Option<Self> {
        let mut kv = Self::new(e, cfg, cap).ok()?;
        if let Err(why) = tail_geometry_ok(
            tail.layers.len(),
            tail.row_bytes,
            tail.base,
            tail.rows,
            tail.len,
            tail.floor,
            kv.k.len(),
            kv.row_bytes,
            kv.window_rows,
            cap,
        ) {
            eprintln!("[dspark] tail import refused: {why}");
            return None;
        }
        // A short tail (rows below the window because the exporter never owned them) makes
        // this KV floor-bearing; a full tail from a floored exporter does not need the floor
        // (every readable row is present) and keeps the pre-lane program.
        let rowsz = kv.row_bytes / std::mem::size_of::<f32>();
        for li in 0..kv.k.len() {
            let (src_k, src_v) = &tail.layers[li];
            if tail.base > 0 {
                // Finite zeros below the tail: the legacy full-scan kernel reads these rows
                // (see the doc above); NaN in either K or V poisons the row's contribution.
                e.memset_zeros_view(&mut kv.k[li].slice_mut(0..tail.base * rowsz))
                    .ok()?;
                e.memset_zeros_view(&mut kv.v[li].slice_mut(0..tail.base * rowsz))
                    .ok()?;
            }
            e.copy_range_into(
                &mut kv.k[li],
                tail.base * rowsz,
                src_k,
                0,
                tail.rows * rowsz,
            )
            .ok()?;
            e.copy_range_into(
                &mut kv.v[li],
                tail.base * rowsz,
                src_v,
                0,
                tail.rows * rowsz,
            )
            .ok()?;
        }
        kv.len = tail.len;
        // A short tail (rows below the window because the exporter never owned them) makes
        // this KV floor-bearing; the clipped round attention raises its window floor to it.
        if tail.rows < kv.window_rows.min(tail.len) {
            kv.floor = tail.base;
        }
        Some(kv)
    }

    /// Rows a restore must carry (see `DsparkSpecSession::draft_tail_rows`). Stored here
    /// because `DflashKv` owns the row geometry; the value comes from the drafter cfg.
    pub fn cfg_window_rows(&self) -> usize {
        self.window_rows
    }

    /// Bytes per row per layer (`n_kv * head_dim * 4`), the unit both the export and the
    /// import address rows in.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Number of draft layers, i.e. how many per-layer planes an export produces.
    pub fn n_layer(&self) -> usize {
        self.k.len()
    }
}

impl DsparkSpecSession {
    pub fn cache_max_ctx(&self) -> usize {
        self.max_ctx
    }
    pub fn finished(&self) -> bool {
        self.done
    }
    pub fn pos(&self) -> usize {
        self.cache.pos
    }
    /// Drain the prompt-end prefix capture exactly once. Publication is worker-owned so it can
    /// apply namespace isolation, dedupe and the shared byte budget at the scheduler boundary.
    pub fn take_prefix_capture(&mut self) -> Option<crate::spec::SpecBoundaryCapture> {
        take_dspark_prefix_capture(&mut self.prefix_capture)
    }
    /// DEMOTION HANDOFF (lane/dspark-spec-gate-demote, 2026-08-24): consume this session and
    /// hand its trunk cache + next-token prediction to the plain batched-decode path — the
    /// dspark twin of [`crate::spec::SpecSession::into_demoted`].
    ///
    /// WHY THIS IS EXACT (greedy). The burst-boundary invariant is `cache.pos == prompt rows
    /// + emitted tokens`: each round commits exactly `m+1` trunk rows (anchor + accepted
    /// drafts) and emits exactly those `m+1` tokens, so every emitted token has its KV row
    /// and nothing else does. `last` is the verify argmax at the LAST committed row — and
    /// verify-column argmax equality with plain decode is the very property the dspark E2E
    /// byte-identity gate pins (`dspark_q38_gate`: ALL EXACT). Handing (cache, last) to the
    ///   batched path therefore continues the stream from a state indistinguishable from one
    ///   the batched path produced itself.
    ///
    /// Unlike the MTP twin there is no carried-pending shape: the round commits its bonus
    /// inside the burst, so a session at a burst boundary is ALWAYS in handoff shape. The
    /// caller still cross-checks `pos()` against its fed-token count (a budget-clamped
    /// overshoot leaves cache rows past the public stream — those sessions finish, never
    /// demote). The draft KV, snapshot buffers and philox counters are DROPPED here
    /// (freeing their VRAM): the batched path never drafts, and the handoff is one-way.
    ///
    /// Sampled sessions must not be demoted (the caller excludes them, mirroring the MTP
    /// gate): their committed stream depends on the session-owned philox counters, and the
    /// plain batched sampler is a different random program mid-request.
    pub fn into_demoted(self) -> (crate::cache::Cache, u32) {
        (self.cache, self.last)
    }
}

impl crate::hybrid::HybridModel {
    /// Turn-1 prime: trunk prefill with taps armed + chunked ctx ingest into the draft KV.
    /// Mirrors generate_spec_dspark's prime block exactly (chunk offsets via sink.base are
    /// handled inside prime_cache's tick loop; the 256-row ingest chunks match the bin arm).
    pub fn dspark_spec_session_new(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        prompt: &[u32],
        ctx_cap: usize,
        sampling: Option<crate::spec::SpecSampling>,
        capture_prefix: bool,
    ) -> Result<DsparkSpecSession, Box<dyn std::error::Error>> {
        use crate::cache::{Cache, DflashTapSink};
        assert!(
            !self.uses_gemma_program(),
            "gemma4 targets use the assistant-drafter route; dspark is the qwen-hybrid arm"
        );
        // Penalized SAMPLED requests are IN scope (lane/dspark-penalized-sampled-20260821:
        // p-side penalties over the true per-state window, q the recorded proposal — the
        // accept walk's penalty arm). Penalties at temp==0 stay a LOUD refusal: the greedy
        // walk argmaxes RAW columns and would silently drop them — penalized greedy is
        // served exactly on the plain path (worker admission owns that exclusion).
        if let Some(sp) = sampling.as_ref()
            && sp.temp <= 0.0
            && sp.pen_on()
        {
            return Err(
                "dspark spec at temp==0 is the greedy route and would silently drop \
                     the request's penalties; penalized greedy is served on the plain path"
                    .into(),
            );
        }
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        assert_eq!(n_embd, c.hidden, "draft hidden must match target n_embd");
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        // The dspark round is windowless: every position the session will ever hold must
        // fit the draft window. Clamp the session ctx to it and refuse prompts that
        // cannot take even one round — admission falls back to the plain path.
        // DFlash2 rounds implement the reference's symmetric sliding window
        // (sdpa_naive_w), so its sessions take the full ctx cap.
        let is_dflash2 = draft.dflash2.is_some();
        let max_ctx = if is_dflash2 {
            ctx_cap
        } else {
            ctx_cap.min(c.sliding_window)
        };
        if !dspark_spec_prompt_fits(prompt.len(), ctx_cap, b, c.sliding_window, is_dflash2) {
            let need = prompt.len().saturating_add(b).saturating_add(8);
            return Err(format!(
                "dspark session needs {need} ctx (prompt {} + block {b} + 8), cap {max_ctx}",
                prompt.len()
            )
            .into());
        }
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;
        let tp = prompt.len();
        cache.dflash_taps = Some(DflashTapSink {
            layer_ids: c.target_layer_ids.clone(),
            buf: e.uninit(tp * n_taps * n_embd)?,
            hidden: n_embd,
            t: tp,
            base: 0,
        });
        let (logits, _h_seed, _hiddens) = self.prime_cache(e, prompt, &mut cache, 0)?;
        // Boundary token: greedy argmax (byte contract) or the request's own filtered
        // draw through the session Philox stream (the frspec boundary composition) —
        // penalized over the prompt window when the request carries penalties.
        let mut sctr0 = 0u32;
        let pen_hist: Vec<u32> = match sampling.as_ref().filter(|s| s.temp > 0.0 && s.pen_on()) {
            Some(sp) => crate::spec::pen_window_seed(&[], prompt, sp.penalty_last_n),
            None => Vec::new(),
        };
        let last = match sampling.as_ref().filter(|s| s.temp > 0.0) {
            Some(sp) => crate::spec::sample_boundary_token(
                e,
                &logits,
                sp,
                &pen_hist,
                &mut sctr0,
                "dspark-prime",
            )?,
            None => crate::forward::argmax(&logits) as u32,
        };
        let mut dkv = DflashKv::new(e, &draft.cfg, max_ctx)?;
        {
            let taps = cache.dflash_taps.take().unwrap();
            let n_taps_h = n_taps * n_embd;
            let mut r0 = 0usize;
            while r0 < tp {
                let t_c = (tp - r0).min(256);
                let tv = e.view(&taps.buf, tp * n_taps_h);
                let win = tv.slice(r0 * n_taps_h..(r0 + t_c) * n_taps_h);
                let mut chunk = e.uninit(t_c * n_taps_h)?;
                e.copy_view_into(&mut chunk, 0, &win, t_c * n_taps_h)?;
                let f = draft.ctx_features(e, &chunk, t_c)?;
                let pos_c: Vec<i32> = ((r0 as i32)..(r0 + t_c) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_c, t_c)?;
                r0 += t_c;
            }
        }
        e.stream().synchronize()?;
        // FULL-PROMPT ONLY. Unlike MTP, DFlash cannot restore its draft plane from a trunk
        // prefix, so there is no LCP/message-boundary split arm here. Mandatory draft-KV
        // allocation + ingest has already succeeded; the optional snapshot can no longer turn
        // a session that would have fit into a draft-allocation failure. Capture remains before
        // any speculative burst mutates the recurrent state.
        let prefix_capture = if capture_prefix {
            cache
                .snapshot(e)
                .ok()
                .map(|snap| crate::spec::SpecBoundaryCapture {
                    snap,
                    pos: tp,
                    logits: logits.clone(),
                    last_h: Vec::new(),
                    latent_tails: Vec::new(),
                })
        } else {
            None
        };
        // Verify carries [anchor, drafts] = up to n_drafts+1 rows (harvest-dependent;
        // DSPARK-POSTMORTEM-20260820.md; family-keyed for DFlash2, else checkpoint
        // strategy census).
        let nd = DsparkHarvest::for_draft(draft).n_drafts(b);
        let vt_cap: usize = std::env::var("MEMRA_DFLASH_VERIFY_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(nd + 1)
            .clamp(2, nd + 1);
        Ok(DsparkSpecSession {
            cache,
            prefix_capture,
            dkv,
            last,
            ctx_len: tp,
            vt: vt_cap,
            rounds: 0,
            max_ctx,
            done: false,
            snapb: None,
            snapb_off: false,
            sampling,
            sctr: sctr0,
            uctr: 0,
            pen_hist,
        })
    }

    /// One scheduler burst: dspark rounds until >= `burst_target` tokens are committed,
    /// EOS lands, or the ctx cap is reached. `request_room` is the request's remaining
    /// public budget, which may be larger than the per-tick scheduler quantum. Returns
    /// (tokens, drafted, accepted) for this burst — mid-request quantum overshoot stays
    /// public, while only the true request boundary clamps the committed cache prefix.
    /// ADMISSION DEBT of this model's verify-graph pool, in bytes (lane/
    /// hermes-perf-fixes, 2026-08-23): the projected remaining growth the serve admission
    /// gate must reserve so sessions admitted while the pool is cold do not overcommit VRAM
    /// the pool will hold (it grows monotonically with no eviction by design — the pool's
    /// high-water is per-export and unknown until observed on the serving box; the 1.5 GiB
    /// SPEC_SHRINK_RESERVE never covered it). Projection contract and the self-measuring
    /// arithmetic live on [`crate::spec::dspark_vg_debt_projection`]; the observed bytes
    /// come from the device graph mem pool (`Engine::device_graph_mem_reserved`).
    ///
    /// CHARGED BY STRUCT, not by which route filled it (lane/graph-launch-guard-sweep-
    /// 20260831, fleet-peer refuted-read fix): the MTP spec route's verify-graph door
    /// (`MEMRA_SPEC_VERIFY_GRAPH`, family default for GDN+MoE) fills the SAME
    /// `dspark_vgraphs` pool with the same monotonic growth, and used to escape charging
    /// because the door check named only the dspark flags. 0 when EVERY door is closed
    /// (`MEMRA_DSPARK_VERIFY_GRAPH=0` and the MTP door off), frozen
    /// (`MEMRA_DSPARK_VG_MAX=0`), or the pool has not captured yet.
    pub fn dspark_vg_admission_debt(&self, e: &Engine) -> usize {
        let dspark_door =
            crate::spec::dspark_verify_graph_serve_on() || crate::spec::dspark_verify_graph_on();
        let mtp_door =
            crate::spec::spec_verify_graph_env().unwrap_or_else(|| self.vgraph_family_default());
        if !dspark_door && !mtp_door {
            return 0;
        }
        let reserved = e.device_graph_mem_reserved();
        self.dspark_vgraphs
            .lock()
            .unwrap()
            .as_mut()
            .map(|g| g.admission_debt(reserved))
            .unwrap_or(0)
    }

    /// MULTI-TURN RESUME (lane/dflash2-session-reuse, 2026-08-25): continue a parked
    /// dspark session with the next turn's suffix — the dspark twin of the MTP pool
    /// resume. Trunk rows for the committed stream are already resident in `cache` and
    /// their ctx features in `dkv`, so turn N+1 primes ONLY its delta instead of
    /// re-priming the whole conversation (the route previously served every turn cold —
    /// a full-prompt prime whose cost grows with the conversation).
    ///
    /// EXACTNESS. The suffix prime is the same session-continuation `prime_cache` the
    /// serve path uses for split prompts and LCP restores (chunk N+1 attends chunk N's
    /// resident KV); the tap sink collects the suffix rows prompt-relative and the dkv
    /// ingest lands them at their absolute positions, exactly as the burst's per-round
    /// keep-ingest does. The boundary token re-derives as the cold prime does: greedy
    /// argmax of the suffix's last row, or the request's filtered draw through the
    /// SESSION's own Philox stream (`sctr` continues — the frspec session-continuity
    /// law), penalized over the session+suffix window. A resumed stream is therefore
    /// byte-identical to the stream a cold prime of the full concatenation produces —
    /// the verify arbitrates every committed token either way.
    ///
    /// EOS in the committed history is fine (a finished turn parks with EOS committed;
    /// the new user turn continues past it) — `done` resets here. Callers must pass a
    /// NON-EMPTY suffix for a `done` session (an empty-suffix continuation of a finished
    /// stream would re-emit from a terminal state); the worker's probe enforces it.
    /// Re-arm a dspark session from a RESTORED trunk cache plus a published draft tail —
    /// the long-answer half of lane/dspark-draft-plane-20260827.
    ///
    /// WHY THIS EXISTS. `dspark_spec_session_new` must prime the full prompt, because the draft
    /// KV derives from trunk hidden FEATURES the prime produces as a side effect. A cache hit
    /// returns trunk K/V, not features, so before this a speculating request had to discard even
    /// a full-prompt hit and re-prefill (~10 s at 30k tokens). With the drafter's readable tail
    /// travelling on the entry, both halves are restorable and the discard is unnecessary.
    ///
    /// WHY IT IS EQUIVALENT TO A COLD PRIME, field by field:
    /// * `cache` — the caller's restored trunk cache, already at `prompt.len()` with recurrent
    ///   state, which is why only WHOLE-ENTRY hits are eligible (a GDN trunk cannot rebuild
    ///   recurrent state mid-sequence, so there is no LCP arm here — same restriction as the
    ///   cold path's full-prompt-only rule).
    /// * `dkv` — byte-copied from the tail into the SAME absolute rows, so rope positions and
    ///   every row the windowed SDPA can read are identical to what the prime produced.
    /// * `last` — drawn from the entry's boundary logits with the request's own sampler, the
    ///   same composition the cold path applies to its prime logits.
    /// * `pen_hist` / `sctr` / `uctr` — seeded exactly as a cold session's are: the penalty
    ///   window from this prompt, the Philox counters fresh, because randomness is
    ///   session-owned by the frspec continuity law and a restore is a NEW session.
    /// * `prefix_capture` — `None`: the entry this restored FROM already exists, so
    ///   republishing the same key would be dropped by the worker's dedupe anyway.
    ///
    /// Refuses (rather than asserting) whenever the rebuilt draft KV and the cache disagree, so
    /// a caller that gets `Err` simply cold-primes.
    #[allow(clippy::too_many_arguments)]
    pub fn dspark_spec_session_from_restored(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        cache: crate::cache::Cache,
        prompt: &[u32],
        // Draft KV ALREADY rebuilt from the entry's tail by the caller (`DflashKv::from_tail`)
        // while the prefix cache was borrowable. Taking the built KV rather than the tail is
        // what keeps the ~85 MB tail in the entry for other requests — `from_tail` copies OUT
        // of it, so no clone of the tail is ever needed.
        dkv: DflashKv,
        boundary_logits: &[f32],
        sampling: Option<crate::spec::SpecSampling>,
        ctx_cap: usize,
    ) -> Result<DsparkSpecSession, Box<dyn std::error::Error>> {
        assert!(
            !self.uses_gemma_program(),
            "gemma4 targets use the assistant-drafter route; dspark is the qwen-hybrid arm"
        );
        if let Some(sp) = sampling.as_ref()
            && sp.temp <= 0.0
            && sp.pen_on()
        {
            return Err("penalized greedy is served on the plain path".into());
        }
        let c = &draft.cfg;
        let b = c.block_size;
        let is_dflash2 = draft.dflash2.is_some();
        let max_ctx = if is_dflash2 {
            ctx_cap
        } else {
            ctx_cap.min(c.sliding_window)
        };
        let tp = prompt.len();
        if !dspark_spec_prompt_fits(tp, ctx_cap, b, c.sliding_window, is_dflash2) {
            return Err(format!("restored dspark session does not fit ctx {max_ctx}").into());
        }
        if cache.pos != tp {
            return Err(format!(
                "restored dspark session needs a whole-entry trunk cache: cache.pos {} !=                  prompt {tp}",
                cache.pos
            )
            .into());
        }
        if dkv.len != tp {
            return Err(format!("restored draft KV len {} != prompt {tp}", dkv.len).into());
        }
        if dkv.cap != max_ctx {
            return Err(
                format!("restored draft KV cap {} != session ctx {max_ctx}", dkv.cap).into(),
            );
        }
        if boundary_logits.is_empty() {
            return Err("restored dspark session needs the entry's boundary logits".into());
        }
        let mut sctr0 = 0u32;
        let pen_hist: Vec<u32> = match sampling.as_ref().filter(|s| s.temp > 0.0 && s.pen_on()) {
            Some(sp) => crate::spec::pen_window_seed(&[], prompt, sp.penalty_last_n),
            None => Vec::new(),
        };
        let last = match sampling.as_ref().filter(|s| s.temp > 0.0) {
            Some(sp) => crate::spec::sample_boundary_token(
                e,
                boundary_logits,
                sp,
                &pen_hist,
                &mut sctr0,
                "dspark-restore",
            )?,
            None => crate::forward::argmax(boundary_logits) as u32,
        };
        let nd = DsparkHarvest::for_draft(draft).n_drafts(b);
        let vt_cap: usize = std::env::var("MEMRA_DFLASH_VERIFY_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(nd + 1)
            .clamp(2, nd + 1);
        Ok(DsparkSpecSession {
            cache,
            prefix_capture: None,
            dkv,
            last,
            ctx_len: tp,
            vt: vt_cap,
            rounds: 0,
            max_ctx,
            done: false,
            snapb: None,
            snapb_off: false,
            sampling,
            sctr: sctr0,
            uctr: 0,
            pen_hist,
        })
    }

    pub fn dspark_spec_session_resume(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        sess: &mut DsparkSpecSession,
        suffix: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::cache::DflashTapSink;
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        let pos0 = sess.cache.pos;
        debug_assert_eq!(
            sess.ctx_len, pos0,
            "dspark resume: draft KV rows != trunk cache rows"
        );
        if suffix.is_empty() {
            return Err(
                "dspark resume needs a non-empty suffix (worker probe owns the \
                        empty-suffix exact-continuation case)"
                    .into(),
            );
        }
        // SHORT-SUFFIX FLOOR (incident 2026-08-25, box10 crash loop). The suffix prime goes
        // through `prime_cache`, which asserts `T >= PRIME_MIN_T` — the batched prefill arm
        // has no tokenwise twin that also fills the DFlash tap sink. A resumed turn shorter
        // than that floor (the watchdog's "Say OK." class, and any brief agent follow-up)
        // therefore PANICKED the GPU worker, which exits 70 and takes every session on the
        // box with it: 20 panics and ~5 minutes of 502s on box10 before MEMRA_REUSE_POOL=0
        // stopped it. The worker probe declines these before it ever gets here (its own
        // guard is the one that keeps the request on the cold path, which is exactly the
        // pre-lane behavior); this is the engine-side backstop so no future caller can
        // reintroduce the panic, and it is a refusal rather than an assert because a
        // too-short turn is ordinary traffic, not a bug.
        if suffix.len() < crate::hybrid_forward::PRIME_MIN_T {
            return Err(format!(
                "dspark resume suffix {} < PRIME_MIN_T {} (prime_cache has no tokenwise \
                 tap-filling twin); serve this turn cold",
                suffix.len(),
                crate::hybrid_forward::PRIME_MIN_T
            )
            .into());
        }
        let need = pos0
            .saturating_add(suffix.len())
            .saturating_add(b)
            .saturating_add(8);
        if need > sess.max_ctx {
            return Err(format!(
                "dspark resume needs {need} ctx (resident {pos0} + suffix {} + block {b} + 8), \
                 cap {}",
                suffix.len(),
                sess.max_ctx
            )
            .into());
        }
        let tp = suffix.len();
        sess.cache.dflash_taps = Some(DflashTapSink {
            layer_ids: c.target_layer_ids.clone(),
            buf: e.uninit(tp * n_taps * n_embd)?,
            hidden: n_embd,
            t: tp,
            base: 0,
        });
        let (logits, _h_seed, _hiddens) = self.prime_cache(e, suffix, &mut sess.cache, 0)?;
        let sp_pen = sess.sampling.filter(|s| s.temp > 0.0 && s.pen_on());
        if let Some(sp) = sp_pen.as_ref() {
            sess.pen_hist = crate::spec::pen_window_seed(&sess.pen_hist, suffix, sp.penalty_last_n);
        }
        let last = match sess.sampling.filter(|s| s.temp > 0.0) {
            Some(sp) => crate::spec::sample_boundary_token(
                e,
                &logits,
                &sp,
                &sess.pen_hist,
                &mut sess.sctr,
                "dspark-resume",
            )?,
            None => crate::forward::argmax(&logits) as u32,
        };
        {
            let taps = sess.cache.dflash_taps.take().unwrap();
            let n_taps_h = n_taps * n_embd;
            let mut r0 = 0usize;
            while r0 < tp {
                let t_c = (tp - r0).min(256);
                let tv = e.view(&taps.buf, tp * n_taps_h);
                let win = tv.slice(r0 * n_taps_h..(r0 + t_c) * n_taps_h);
                let mut chunk = e.uninit(t_c * n_taps_h)?;
                e.copy_view_into(&mut chunk, 0, &win, t_c * n_taps_h)?;
                let f = draft.ctx_features(e, &chunk, t_c)?;
                let pos_c: Vec<i32> = (((pos0 + r0) as i32)..((pos0 + r0 + t_c) as i32)).collect();
                draft.ingest_ctx(e, &mut sess.dkv, &f, &pos_c, t_c)?;
                r0 += t_c;
            }
        }
        e.stream().synchronize()?;
        sess.ctx_len += tp;
        sess.last = last;
        sess.done = false;
        Ok(())
    }

    pub fn dspark_spec_session_burst(
        &self,
        e: &Engine,
        draft: &DflashDraft,
        sess: &mut DsparkSpecSession,
        burst_target: usize,
        request_room: usize,
        eos: &[u32],
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        use crate::cache::DflashTapSink;
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        let n_vocab = self.output.out_features();
        // Harvest convention (DSPARK-POSTMORTEM-20260820.md) — identical to the bin arm
        // (family-keyed for DFlash2, else checkpoint strategy census; owner-ratified
        // flip 2026-08-20).
        let harvest = DsparkHarvest::for_draft(draft);
        let nd = harvest.n_drafts(b);
        let r0 = harvest.first_row();
        let vt_cap: usize = std::env::var("MEMRA_DFLASH_VERIFY_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(nd + 1)
            .clamp(2, nd + 1);
        let adapt = std::env::var("MEMRA_DFLASH_ADAPT").as_deref() != Ok("0");
        // Verify-window policy (H4, DSPARK-POSTMORTEM-20260820.md) — identical to the
        // bin arm: default = confidence-slot tau=.5 on a head-carrying checkpoint
        // (owner-ratified flip 2026-08-20); head-less (incl. the DFlash2 family) and
        // ADAPT=0 keep the ladder.
        let vt_policy = DsparkVtPolicy::resolve(draft.confidence.is_some());
        if vt_policy.is_confidence() {
            assert!(
                draft.confidence.is_some(),
                "MEMRA_DSPARK_VT={vt_policy:?} needs a checkpoint with an accept-rate \
                 head (confidence_head.* absent in this export)"
            );
        }
        // SAMPLED ADMISSION (T>0): session-fixed config; counters live on the session so
        // randomness never repeats across bursts. None/temp==0 = the greedy route.
        let sp_on: Option<crate::spec::SpecSampling> = sess.sampling.filter(|s| s.temp > 0.0);
        let pen_on = sp_on.as_ref().is_some_and(|s| s.pen_on());
        let mut out: Vec<u32> = Vec::with_capacity(burst_target + b);
        let mut drafted = 0usize;
        let mut accepted_n = 0usize;
        // Engine-bundle slice 2 — identical to the bin arm: deferred chain readback under
        // the stash arm with a resident embed table (ladder policy only).
        let defer_rb = !vt_policy.is_confidence();
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if !defer_rb || crate::spec::spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        // Slice 3/4c SERVE ENGAGEMENT (graphs-serve lane; DSF-ROUNDCOST §9.3 -> §10): the
        // verify-graph pool lives on the MODEL (`dspark_vgraphs`, one per process) and its
        // keys — (segment, vt) and (vt, rung, hi) — carry NOTHING session-scoped, so ANY
        // session whose round matches a key replays the same capture (this is the
        // cache-reuse-pool the old bin-arm-only note asked for). Sharing is sound because
        // every per-session-varying address the captured bodies touch is indirect:
        // conv/ssm state and the ckpt stash resolve through the per-verify refreshed
        // pointer table (refresh_tables + copy_indirect_src_f32 — the slice-3
        // parity/lifetime law; a baked address is the known 12/12-divergence class), kv
        // bases through fa_table, residual/pos/tap through ctx-owned staging rewritten
        // every round; per-row t_kv derives in-kernel from pos_seq, and the per-round
        // host bookkeeping (parity swap, len bump) runs on THIS session's cache. The
        // guard spans the burst: the slab stash is live verify -> commit inside each
        // round, and the worker drives bursts from one scheduler thread
        // (step_dspark_spec), so sessions interleave at burst boundaries only.
        // DEFAULT ON on the serve route since the v0.103 train (owner-ratified
        // 2026-08-22, §10 re-gate at flip): MEMRA_DSPARK_VERIFY_GRAPH=0 is the
        // kill-switch that keeps this None — the eager walk, byte-identical (the
        // kill-switch arm of the serve battery). The bin arm keeps its own opt-in.
        let mut vg_guard = self.dspark_vgraphs.lock().unwrap();
        if vg_guard.is_none() && embd_gpu.is_some() && crate::spec::dspark_verify_graph_serve_on() {
            *vg_guard = crate::spec::DsparkVerifyGraphs::new(e, &sess.cache, vt_cap, n_embd)?;
            if vg_guard.is_some() {
                // Engagement receipt (the §8 dead-arm lesson): prove the door is LIVE on
                // the serve surface — S6b banked the tip server carrying zero door strings.
                eprintln!("[dspark-vg] serve pool ENGAGED (vt_cap={vt_cap})");
            }
        }
        let vgraphs: &mut Option<crate::spec::DsparkVerifyGraphs> = &mut vg_guard;
        'outer: while out.len() < burst_target && !sess.done {
            let start = sess.cache.pos;
            if start + nd + 1 > sess.max_ctx {
                sess.done = true;
                break;
            }
            sess.rounds += 1;
            let mut vt = sess.vt;
            // ---- draft: block = [last, MASK x b-1] (identical to the bin arm) ----
            // RAII: a `?` exit restores the pre-scope value instead of latching exact
            // ON engine-wide across every later request (hermes finding, fixed
            // 2026-08-23 — this burst had several `?`s between the manual true/false).
            let exact_scope = e.exact_scope(true);
            let mut block: Vec<u32> = vec![c.mask_token_id; b];
            block[0] = sess.last;
            let noise = e.htod(&self.embd.try_gather(n_embd, &block)?)?;
            let pos_block: Vec<i32> = ((start as i32)..(start + b) as i32).collect();
            let dh = draft.forward_round(e, &mut sess.dkv, &noise, &pos_block)?;
            // Harvest: logits over rows r0..r0+nd (see the bin arm / the postmortem).
            let mut rows = e.uninit(nd * n_embd)?;
            {
                let dv = e.view(&dh, b * n_embd);
                let src = dv.slice(r0 * n_embd..(r0 + nd) * n_embd);
                e.copy_view_into(&mut rows, 0, &src, nd * n_embd)?;
            }
            // TRIMMED DRAFT HEAD (lane/dflash2-head-trim, 2026-08-25): DFlash2 family
            // only — the selector consumes (value, candidate-id) pairs, so a d2t remap
            // after top-k restores true ids; the markov/chain arms argmax dl columns
            // into token ids DIRECTLY and must keep the full head. Reuses the FR-Spec
            // self-trim the load path builds on the MTP struct (MEMRA_FRSPEC_TRIM):
            // gathered rows of the target's own head, zero requant. Verify stays
            // full-vocab, so the trim moves draft acceptance only, never output.
            let trim = if draft.dflash2.is_some() {
                self.mtp
                    .as_ref()
                    .filter(|m| m.d2t_from_target_head)
                    .and_then(|m| m.shared_head_head.as_ref().zip(m.d2t.as_ref()))
                    // MEMRA_MTP_SKIP stub: the same target-head trimmed rows, parked in
                    // `dflash_trim` because the embedded MTP block was skipped (hybrid.rs;
                    // rows are target-head by construction; the loader refuses otherwise).
                    .or_else(|| self.dflash_trim.as_ref().map(|t| (&t.head, &t.d2t)))
                    .filter(|(_, d2t)| !d2t.is_empty())
            } else {
                None
            };
            let (dl_head, dl_vocab) = match trim {
                Some((head, d2t)) => (head, d2t.len()),
                None => (&self.output, n_vocab),
            };
            let trim_d2t = trim.map(|(_, d2t)| d2t.as_slice());
            let mut dl = e.matmul(dl_head, &rows, nd)?;
            // Family/sampling-keyed proposal — identical to the bin arm (see there for
            // the program law: sampled records the true q, DFlash2 rides the selector,
            // the markov/plain greedy chain keeps the slice-2 deferral). Confidence
            // policy: stash markov prev-token embeddings d2d during the chain, one host
            // readback after — identical to the bin arm.
            let want_conf_emb = vt_policy.is_confidence()
                && draft.confidence.as_ref().is_some_and(|ch| ch.with_markov);
            let mut conf_emb: Option<CudaSlice<f32>> = match (&draft.markov, want_conf_emb) {
                (Some(mk), true) => Some(e.uninit(nd * mk.rank)?),
                (None, true) => unreachable!(
                    "with_markov confidence head without a markov table — the loader forbids it"
                ),
                _ => None,
            };
            let mut cand: Vec<u32> = Vec::with_capacity(nd + 1);
            let mut prop: Option<DsparkDraftSample> = None;
            let mut chain_dev: Option<CudaSlice<u32>> = None;
            // Slice 2: arm choice read before the chain readback (see the bin arm; the
            // serve arm has no CKPT_GATE oracle — the bin arm carries it).
            let ckpt_on = std::env::var("MEMRA_DSPARK_CKPT").as_deref() != Ok("0");
            let mut deferred = false;
            if let Some(sp) = sp_on.as_ref() {
                // SAMPLED proposal (family-keyed; identical to the bin arm).
                let (tail, ds) = draft.dspark_propose_sampled(
                    e,
                    &mut dl,
                    &rows,
                    nd,
                    dl_vocab,
                    sess.last,
                    sp,
                    &mut sess.sctr,
                    &mut sess.uctr,
                    conf_emb.as_mut(),
                    trim_d2t,
                )?;
                drop(exact_scope);
                cand.push(sess.last);
                cand.extend_from_slice(&tail);
                prop = Some(ds);
            } else if draft.dflash2.is_some() {
                // DFlash2: candidate path selector replaces the markov chain
                // (identical to the bin arm).
                let path = draft
                    .dflash2_propose_greedy(e, &dl, &rows, nd, dl_vocab, sess.last, trim_d2t)?;
                drop(exact_scope);
                cand.push(sess.last);
                cand.extend_from_slice(&path);
            } else {
                let mut chain_d = e.stream().alloc_zeros::<u32>(nd + 1)?;
                if let Some(mk) = &draft.markov {
                    e.set_u32_one(&mut chain_d, sess.last)?;
                    for k in 0..nd {
                        let mut f = e.uninit(mk.rank)?;
                        e.gather_row_bf16(&mk.w1_bf16, &chain_d, k, &mut f, mk.rank)?;
                        if let Some(ce) = conf_emb.as_mut() {
                            let fv = e.view(&f, mk.rank);
                            e.copy_view_into(ce, k * mk.rank, &fv, mk.rank)?;
                        }
                        let bias = e.matmul(&mk.w2, &f, 1)?;
                        e.add_row_inplace(&mut dl, &bias, n_vocab, k * n_vocab)?;
                        e.argmax_token_device_col(&dl, k, n_vocab, &mut chain_d, k + 1)?;
                    }
                } else {
                    if want_conf_emb {
                        // chain_d[0] must carry the anchor — slot 0's prev token.
                        e.set_u32_one(&mut chain_d, sess.last)?;
                    }
                    for i in 0..nd {
                        if let (Some(ce), Some(mk)) = (conf_emb.as_mut(), &draft.markov) {
                            let mut f = e.uninit(mk.rank)?;
                            e.gather_row_bf16(&mk.w1_bf16, &chain_d, i, &mut f, mk.rank)?;
                            let fv = e.view(&f, mk.rank);
                            e.copy_view_into(ce, i * mk.rank, &fv, mk.rank)?;
                        }
                        e.argmax_token_device_col(&dl, i, n_vocab, &mut chain_d, i + 1)?;
                    }
                }
                drop(exact_scope);
                deferred = embd_gpu.is_some() && ckpt_on;
                chain_dev = Some(chain_d);
            }
            // ---- H4 confidence window: size THIS round's verify from the head ----
            if vt_policy.is_confidence() {
                let ch = draft.confidence.as_ref().expect("asserted at burst entry");
                let (rows_h, emb_h) = match conf_emb.as_ref() {
                    Some(ce) => {
                        let (a, b2) = e.dtoh_pair(&rows, ce)?;
                        (a, Some(b2))
                    }
                    None => (e.dtoh(&rows)?, None),
                };
                let rank = draft.markov.as_ref().map(|m| m.rank).unwrap_or(0);
                let mut raws = Vec::with_capacity(nd);
                for k in 0..nd {
                    let hrow = &rows_h[k * n_embd..(k + 1) * n_embd];
                    let emb = emb_h.as_ref().map(|eh| &eh[k * rank..(k + 1) * rank]);
                    raws.push(ch.raw_score(hrow, emb));
                }
                vt = vt_policy
                    .size_window(&raws, vt_cap)
                    .expect("confidence policies always size the window");
            }
            // Non-deferred greedy chain readback (the sampled and DFlash2 proposals
            // built `cand` at the walk; deferred rounds build it after the merged
            // readback — bytes identical, chain_d written before either sync).
            if let Some(chain_d) = chain_dev.as_ref()
                && !deferred
            {
                let chain = e.dtoh_u32(chain_d)?;
                cand.push(sess.last);
                cand.extend_from_slice(&chain[1..]);
            }

            // ---- snapshot, then verify t=vt (ckpt stash default; oracle arms kept) ----
            // Slice 1: batched snap (see DsparkSnapBatch) with the legacy per-layer
            // snapshot as the kill-switch / non-uniform fallback.
            let mut snap_legacy: Option<crate::cache::CacheSnapshot> = None;
            if !sess.snapb_off && sess.snapb.is_none() {
                sess.snapb = DsparkSnapBatch::new(e, &sess.cache)?;
                sess.snapb_off = sess.snapb.is_none();
            } else if let Some(sb) = sess.snapb.as_mut() {
                sb.refresh(e, &sess.cache)?;
            }
            let snap: &crate::cache::CacheSnapshot = match sess.snapb.as_ref() {
                Some(sb) => &sb.snap,
                None => {
                    snap_legacy = Some(sess.cache.snapshot(e)?);
                    snap_legacy.as_ref().unwrap()
                }
            };
            let _ = &snap_legacy;
            // Slice 3: the tap-sink buffer is persistent per vt in the graphs ctx
            // (captured segments bake its address — a per-round alloc here would make
            // every session's replayed tap copies write freed memory); fully rewritten
            // by every verify, so pool ownership changes no bytes.
            let tap_buf = match vgraphs.as_mut().and_then(|g| g.tap_bufs.remove(&vt)) {
                Some(buf) => buf,
                None => e.uninit(vt * n_taps * n_embd)?,
            };
            sess.cache.dflash_taps = Some(DflashTapSink {
                layer_ids: c.target_layer_ids.clone(),
                buf: tap_buf,
                hidden: n_embd,
                t: vt,
                base: 0,
            });
            // Composition guard (sampled admission × model-owned pool, this train's
            // cross-product): the slab flag is a per-round statement, but only the
            // graphs-aware verify (`_am_ckpt_dev`) clears it. Serve sessions MIX arms
            // within one process-lifetime pool — a SAMPLED round rides the raw-logits
            // twins (no graphs param) and must not inherit `round_slab=true` from a
            // previous greedy session's captured round, or its commit is steered at
            // slabs the round never wrote. Clear at the round boundary; the deferred
            // arm re-derives it inside the verify. (The bin arm has the same shape but
            // fixes its sampling mode per process, so no mixed rounds exist there.)
            if let Some(g) = vgraphs.as_mut() {
                g.round_slab = false;
            }
            // The whole fallible verify window runs inside a closure so the Err path
            // can return the sink buffer to the ctx pool before propagating — the
            // serve-surface twin of the EOS-orphan lesson: a mid-verify error
            // propagates OUT of the burst, the request dies, the session's cache is
            // dropped — but the PROCESS (and the pool, with the tap-buffer address
            // baked into its captures) lives on. Recover the ctx-owned buffer before
            // the error escapes, or the next session's replayed tap copies write
            // freed memory. The bin arm has no such path (a gate-binary error ends
            // the process).
            #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
            let verify_out = (|| -> Result<
                (
                    Vec<u32>,
                    Option<CudaSlice<f32>>,
                    Option<crate::spec::DsparkVerifyCkpt>,
                ),
                Box<dyn std::error::Error>,
            > {
                if sp_on.is_some() {
                    // SAMPLED: raw verify logits for the rejection walk (bin-arm twin).
                    if ckpt_on {
                        let (tl, vck) = self.dspark_verify_t_logits_ckpt(
                            e,
                            &cand[..vt],
                            start,
                            &mut sess.cache,
                        )?;
                        Ok((Vec::new(), Some(tl), Some(vck)))
                    } else {
                        Ok((
                            Vec::new(),
                            Some(self.dspark_verify_t_logits(
                                e,
                                &cand[..vt],
                                start,
                                &mut sess.cache,
                            )?),
                            None,
                        ))
                    }
                } else if deferred {
                    // Slice 2: device-token verify + ONE merged readback (see the bin arm).
                    let chain_d = chain_dev.as_ref().expect("deferred implies greedy chain");
                    let g = embd_gpu.expect("deferred implies resident embed");
                    let (am_d, vck) = self.dspark_verify_t_am_ckpt_dev(
                        e,
                        chain_d,
                        vt,
                        start,
                        &mut sess.cache,
                        (g, embd_qt, embd_rb),
                        vgraphs.as_mut(),
                    )?;
                    let ch = e.stream().clone_dtoh(chain_d)?;
                    let am = e.stream().clone_dtoh(&am_d)?;
                    e.stream().synchronize()?;
                    cand.push(sess.last);
                    cand.extend_from_slice(&ch[1..]);
                    Ok((am, None, Some(vck)))
                } else if ckpt_on {
                    let (vam, vck) =
                        self.dspark_verify_t_am_ckpt(e, &cand[..vt], start, &mut sess.cache)?;
                    Ok((vam, None, Some(vck)))
                } else {
                    Ok((
                        self.dspark_verify_t_am(e, &cand[..vt], start, &mut sess.cache)?,
                        None,
                        None,
                    ))
                }
            })();
            let (vam, tl, vck) = match verify_out {
                Ok(v) => v,
                Err(err) => {
                    if let Some(taps) = sess.cache.dflash_taps.take()
                        && let Some(g) = vgraphs.as_mut()
                    {
                        g.tap_bufs.insert(vt, taps.buf);
                    }
                    return Err(err);
                }
            };
            let taps = sess.cache.dflash_taps.take().unwrap();
            // Return the tap buffer to the ctx pool IMMEDIATELY — an EOS/budget break
            // between accept and ingest must never orphan an address the captured graphs
            // bake (the bin arm's lesson, and it holds doubly here: the pool outlives
            // the SESSION, not just the round). Ingest reads it borrowed.
            let tap_local: Option<CudaSlice<f32>> = match vgraphs.as_mut() {
                Some(g) => {
                    g.tap_bufs.insert(vt, taps.buf);
                    None
                }
                None => Some(taps.buf),
            };
            let tap_ref: &CudaSlice<f32> = match &tap_local {
                Some(b) => b,
                None => &vgraphs.as_ref().expect("ctx present above").tap_bufs[&vt],
            };

            // ---- accept ----
            // Penalized-sampled: anchor joins the window before the walk (committed this
            // round via the out.push below); accepted drafts extend it after — identical
            // to the bin arm.
            if pen_on {
                sess.pen_hist.push(sess.last);
            }
            let (m, next) = match (sp_on.as_ref(), tl.as_ref()) {
                (Some(sp), Some(tl)) => {
                    let w0 = sess
                        .pen_hist
                        .len()
                        .saturating_sub(sp.penalty_last_n.min(crate::spec::PEN_WINDOW_MAX));
                    dspark_accept_sampled(
                        e,
                        tl,
                        &cand,
                        vt,
                        n_vocab,
                        &dl,
                        prop.as_ref()
                            .expect("sampled round without a proposal record"),
                        sp,
                        &sess.pen_hist[w0..],
                        &mut sess.sctr,
                        &mut sess.uctr,
                    )?
                }
                _ => {
                    let m = dspark_accept_prefix(&cand, &vam, vt);
                    (m, vam[m])
                }
            };
            drafted += vt - 1;
            accepted_n += m;
            // keep = the rows this round adds to the PUBLIC stream. Without eos that is
            // the anchor + all accepted drafts (m+1). With eos it is the anchor + drafts
            // UP TO AND INCLUDING eos: the walk may accept real tokens past eos (they are
            // the model's own continuation), but emission stops at eos, and a parked
            // session whose cache holds rows past the public stream can never resume —
            // the park gate `pos() == fed` would refuse every eos-terminated stream
            // (measured: 7/8 turns on the mtreuse gate, overshoot 1-6 rows). Truncating
            // the commit at eos uses the SAME prefix-commit machinery as a mid-round
            // rejection, so the hybrid (GDN) state is exact by the same argument.
            // Emitted bytes are untouched — this only changes post-eos cache state.
            let mut keep = m + 1;
            let mut terminal = false;
            if eos.contains(&sess.last) {
                terminal = true;
                keep = 1;
            } else {
                for (j, &dt) in cand[1..=m].iter().enumerate() {
                    if eos.contains(&dt) {
                        terminal = true;
                        keep = j + 2; // anchor + drafts through eos
                        break;
                    }
                }
            }
            // The request's max_tokens boundary is also a commit boundary, not merely an
            // output slice. It is NOT the scheduler's smaller per-tick burst quantum: accepted
            // surplus crossing that quantum stays public and the session remains live. Only at
            // the true request boundary do we keep the publishable prefix so cache.pos == fed at
            // retire and mark the session terminal until a non-empty next-turn suffix resumes
            // it. This uses the same prefix-commit machinery as EOS/rejection and makes
            // max-token sessions safe to park instead of permanently cold (Hermes
            // `f22a180d1638b95a`).
            let (bounded_keep, budget_terminal) =
                dspark_commit_limit(keep, out.len(), request_room);
            keep = bounded_keep;
            terminal |= budget_terminal;
            out.push(sess.last);
            out.extend_from_slice(&cand[1..keep]);
            sess.done = terminal;
            if pen_on {
                // Only the PUBLIC drafts feed the penalty window — tokens accepted past
                // eos never reach the stream, and a resumed session must not penalize
                // ghosts (the parked pen_hist seeds the resume's window).
                sess.pen_hist.extend_from_slice(&cand[1..keep]);
            }

            // ---- commit/rollback (stash arm default; replay oracle kept) ----
            // Slice 3: rounds whose linear column stash lives in the graphs ctx's slabs
            // commit through the slab twin (same semantics, slab-addressed sources) —
            // identical to the bin arm's dispatch.
            let slab_commit = vgraphs.as_ref().map(|g| g.round_slab).unwrap_or(false);
            if keep < vt {
                if slab_commit {
                    self.dspark_commit_prefix_slab(
                        e,
                        &mut sess.cache,
                        snap,
                        vgraphs.as_ref().expect("slab_commit implies ctx"),
                        keep,
                    )?;
                } else if let Some(vck) = vck.as_ref() {
                    self.dspark_commit_prefix(e, &mut sess.cache, snap, vck, keep)?;
                } else {
                    crate::pp::restore_cache_checkpoint(e, self, None, &mut sess.cache, snap)?;
                    debug_assert_eq!(sess.cache.pos, start, "rollback landed off the round start");
                    let ram = self.dspark_verify_t_am(e, &cand[..keep], start, &mut sess.cache)?;
                    if sp_on.is_none() {
                        // greedy-only oracle; the sampled arm replays to rebuild state.
                        debug_assert_eq!(
                            &ram[..],
                            &vam[..keep],
                            "prefix replay must reproduce the verify argmaxes"
                        );
                    }
                }
            }

            // ---- ingest the kept rows' ctx features into the draft KV ----
            {
                let tv = e.view(tap_ref, vt * n_taps * n_embd);
                let keep_view = tv.slice(0..keep * n_taps * n_embd);
                let mut kept = e.uninit(keep * n_taps * n_embd)?;
                e.copy_view_into(&mut kept, 0, &keep_view, keep * n_taps * n_embd)?;
                let f = draft.ctx_features(e, &kept, keep)?;
                let pos_k: Vec<i32> =
                    ((sess.ctx_len as i32)..(sess.ctx_len + keep) as i32).collect();
                draft.ingest_ctx(e, &mut sess.dkv, &f, &pos_k, keep)?;
                sess.ctx_len += keep;
            }
            if sess.done {
                // EOS or the public budget landed this round: cache, draft KV and ctx_len are
                // all clamped to the public stream (park shape); `next` is beyond the terminal
                // boundary and must not become the anchor of a resumed session.
                break 'outer;
            }
            sess.last = next;
            // Ladder update only — the confidence policies recompute vt from the
            // head every round, post-draft pre-verify; their carry just keeps
            // observability (sess.vt = the last confidence-sized window).
            if vt_policy.is_confidence() {
                sess.vt = vt;
            } else if adapt {
                sess.vt = (m + 2).clamp(3, vt_cap);
            }
        }
        Ok((out, drafted, accepted_n))
    }
}

// ================= Harvest-convention gate (CPU; DSPARK-POSTMORTEM-20260820.md) =========
// The parity oracle is row-count-agnostic (it reproduces the markov MODULE on whatever
// rows it is fed) and the E2E gate is harvest-independent (verify-side truth), so
// NEITHER can catch a wrong row->position mapping — that blindness is how the q38
// misalignment shipped. These tests pin the convention itself as logic the round
// consumes, so a mutation back to the mask-fill harvest under the Dspark variant fails
// HERE, naming the convention.
#[cfg(test)]
mod dflash2_tests {

    /// The tail-import refusal arms (lane/dspark-draft-plane-20260827 review finding: these
    /// were claimed tested and were not). Pure, so they run everywhere; the geometry mirrors
    /// the served DFlash2 drafter (5 layers, 8 kv x 128 dim f32 rows, window 2048 + block 8).
    #[test]
    fn tail_import_refuses_every_geometry_disagreement_and_accepts_the_exported_shape() {
        let rb = 8 * 128 * 4; // n_kv * head_dim * f32
        let win = 2048 + 8; // window_rows = sliding_window + block
        // THE EXPORTED SHAPE: window_rows ending exactly at len, same geometry — accepted.
        assert!(
            super::tail_geometry_ok(5, rb, 30_329 - win, win, 30_329, 0, 5, rb, win, 34_433)
                .is_ok()
        );
        // A short history where the tail IS the whole history — accepted.
        assert!(super::tail_geometry_ok(5, rb, 0, 100, 100, 0, 5, rb, win, 34_433).is_ok());
        // FLOOR-BEARING (lane/spec-exclusions-20260902): a cold-drafter exporter at floor
        // 30_000 owns rows [30_000, 30_329) only, so its 329-row tail is everything readable
        // above the floor — accepted; the same 329 rows from a floor-0 exporter are the
        // truncation the pre-lane rule refuses, verbatim; a tail claiming rows BELOW its
        // exporter's floor is a geometry lie and refuses by name.
        assert!(
            super::tail_geometry_ok(5, rb, 30_000, 329, 30_329, 30_000, 5, rb, win, 34_433).is_ok()
        );
        assert_eq!(
            super::tail_geometry_ok(5, rb, 30_000, 329, 30_329, 0, 5, rb, win, 34_433).unwrap_err(),
            "tail shorter than the drafter's readable window"
        );
        assert_eq!(
            super::tail_geometry_ok(5, rb, 29_990, 339, 30_329, 30_000, 5, rb, win, 34_433)
                .unwrap_err(),
            "tail starts below its exporter's context floor"
        );
        // Every refusal arm, each by name:
        let arm = |l, r, b, rows, len, cap| {
            super::tail_geometry_ok(l, r, b, rows, len, 0, 5, rb, win, cap)
        };
        assert_eq!(
            arm(4, rb, 30_329 - win, win, 30_329, 34_433).unwrap_err(),
            "layer count differs from the live drafter"
        );
        assert_eq!(
            arm(5, rb - 4, 30_329 - win, win, 30_329, 34_433).unwrap_err(),
            "row geometry differs from the live drafter"
        );
        assert_eq!(
            arm(5, rb, 30_329 - win, win, 30_329, 30_000).unwrap_err(),
            "logical length exceeds the session cap"
        );
        // THE RUN-2 BUG, pinned: a tail whose base+rows lands past its own logical length —
        // the export-at-current-length defect the gate caught on the box.
        assert_eq!(
            arm(5, rb, 30_364 - win, win, 30_329, 34_433).unwrap_err(),
            "tail does not end at its own logical length"
        );
        assert_eq!(
            arm(5, rb, 30_329 - (win - 100), win - 100, 30_329, 34_433).unwrap_err(),
            "tail shorter than the drafter's readable window"
        );
    }

    use super::{
        DsparkHarvest, dflash2_walk_greedy, dflash2_walk_sampled, dspark_commit_limit,
        rejection_accept_len,
    };

    // ================= memra#95: the full-cover restore panic =================
    //
    // `walk: candidate 4294967295 outside codebook vocab` (the greedy and sampled walk
    // asserts) on the FIRST round of a `MEMRA_GLM5_SPEC_FULLCOVER=1` restored session,
    // fleet-fatal. `4294967295` is the top-k selector's exhausted-slot sentinel, so the
    // draft logits row was NaN. The row was NaN because the drafter ctx KV is imported on
    // the CALLER's stream at admission (`DflashKv::from_tail`) while round 1 reads it
    // through `glm5_head_engine`, which under a live ppN split is the last stage's OWN
    // Engine used OUTSIDE any `rt.enter` scope, i.e. that Engine's own stream. Nothing
    // ordered the two.
    //
    // THE RETRACTED THEORY, kept because it is what made a wrong fix look right (review
    // round 1 on PR #100): "the full-cover arm never calls `prime_cache` and therefore
    // never inherits `prime_cache_hyper_ppn`'s `fence_stages_behind`". That fence has the
    // SAME blind spot — it orders `StageRt::stream`, the enter-scope stream, not a stage
    // ENGINE's own stream — so inheriting it would have fixed nothing. What actually shields
    // the suffix arm is that `prime_cache` returns its logits through `Engine::dtoh`, which
    // is `stream().synchronize()` on the caller. The full-cover arm has no prime and no
    // device readback at all between the import and the round (its anchor comes from the
    // entry's host-side boundary logits), which is why it is the only exposed path, and why
    // only round 1 dies: from round 2 on, `dflash2_propose_*`'s own `dtoh_u32` drains the
    // head engine.
    //
    // Two gates below: the ordering seam (asserted on the ENGINE-ordering helper, so it
    // rejects the `fence_stages_behind` shape too), and the blast radius.

    /// The sentinel a top-k row that could not be filled carries into the walk.
    #[test]
    fn an_unfilled_selector_slot_is_refused_by_name_not_clamped() {
        // top_k = 4, one draft slot; the selector filled two slots and gave up, which is what
        // `topk_rows_f32` writes for a partially finite row.
        let cand = [7u32, 11, super::TOPK_EMPTY_SLOT, super::TOPK_EMPTY_SLOT];
        let why = super::dflash2_guard_candidates(&cand, 1000, "gate")
            .expect_err("an exhausted slot must be refused");
        assert!(why.contains("slot 2"), "{why}");
        assert!(why.contains("4294967295"), "{why}");
        assert!(
            why.contains("exhausted-slot sentinel") && why.contains("NaN"),
            "the refusal must name the mechanism, not just the number: {why}"
        );
        // THE OBSERVED SHAPE (memra#95): a fully NaN row fills EVERY slot with the sentinel,
        // so the refusal has to bite at slot 0, before the walk's `prev` chain even starts.
        let all_nan = [super::TOPK_EMPTY_SLOT; 4];
        let why0 = super::dflash2_guard_candidates(&all_nan, 1000, "gate")
            .expect_err("an all-sentinel row must be refused");
        assert!(why0.contains("slot 0"), "{why0}");
        // An ordinary out-of-range id is refused too (the trimmed head's rank space is
        // narrower than the vocab, and a remap of a bad rank would panic in the map).
        assert!(super::dflash2_guard_candidates(&[7, 1000], 1000, "gate").is_err());
        // A well-formed row passes, and the bound is exclusive.
        super::dflash2_guard_candidates(&[0, 999, 500, 1], 1000, "gate").expect("clean row");
    }

    /// The walk's own assert is the LAST line of defence and stays exactly where it is:
    /// the guard above refuses one request, this refuses the round.
    #[test]
    #[should_panic(expected = "walk: candidate")]
    fn the_walk_assert_survives_the_guard() {
        const VOCAB: usize = 8;
        const RANK: usize = 2;
        const TOPK: usize = 2;
        let pred = vec![0u8; VOCAB * RANK * 2];
        let succ = vec![0u8; VOCAB * RANK * 2];
        let cand = [1u32, super::TOPK_EMPTY_SLOT];
        let _ = dflash2_walk_greedy(
            &pred,
            &succ,
            VOCAB,
            RANK,
            TOPK,
            &[0.0; TOPK],
            &cand,
            &[0.0; RANK],
            0,
            1,
        );
    }

    /// WIRING GATE (invocations in comment-stripped source, never prose, the
    /// wiring-assertions-match-prose law). Both halves of the memra#95 fix are LIVE code.
    ///
    /// The ordering half is asserted on the ENGINE-ordering seam, not on
    /// `fence_stages_behind`: that distinction IS the defect (a stage's enter-scope stream is
    /// not the stage Engine's own stream, and the draft phase runs on the latter), so a gate
    /// that accepted either would accept the broken shape.
    #[test]
    fn the_fullcover_restore_ordering_seam_is_live_in_comment_stripped_source() {
        let strip = |src: &str| -> String {
            src.lines()
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // 1. THE CAUSE. `glm5_spec_session_from_restored` orders the head/drafter engine
        //    behind the caller BEFORE the suffix branch, so it covers the full-cover arm
        //    (which has no prime, and therefore no incidental host sync, to hide behind).
        let glm5 = strip(include_str!("glm_spec.rs"));
        let body = glm5
            .find("pub fn glm5_spec_session_from_restored(")
            .expect("the restored-session builder exists");
        let end = glm5[body..]
            .find("fn glm5_d2t(")
            .expect("the builder's end anchor exists")
            + body;
        let scope = &glm5[body..end];
        let eh = scope
            .find("let eh = self.glm5_head_engine(e)?;")
            .expect("the builder resolves the head engine");
        let order = scope
            .find("crate::pp::PpNRt::order_engine_behind(e, eh)?;")
            .expect(
                "the restored-session builder must order the head engine's own stream behind \
                 the caller (fence_stages_behind is the WRONG seam: it orders enter-scope \
                 stage streams, and the draft phase never enters a stage)",
            );
        let branch = scope
            .find("let (logits_s, tap_rows, prefix_capture) = if suffix.is_empty()")
            .expect("the full-cover branch exists");
        assert!(
            eh < order && order < branch,
            "the ordering must sit between the head-engine binding and the full-cover/suffix \
             branch: after it so it names the right engine, before it so the prime-free arm \
             is covered"
        );

        // 1b. The helper it calls really orders the ENGINE streams, not the stage streams.
        let pp = strip(include_str!("pp.rs"));
        let helper = pp
            .find("pub fn order_engine_behind(")
            .expect("the engine-ordering helper exists");
        let rest = &pp[helper..];
        let hbody = &rest[..rest.find("\n    pub fn ").unwrap_or(rest.len().min(2000))];
        assert!(
            hbody.contains("let s = src.stream();") && hbody.contains("dst.gpu.main_stream()"),
            "the helper must order the two ENGINES' own streams, source side through the \
             ambient-aware accessor and destination side through the override-blind one"
        );
        assert!(
            hbody.contains("if src.ctx() == dst.ctx() {"),
            "the context test must be VALUE equality: CudaContext::new allocates a fresh Arc \
             per call for the same primary context, so Arc::ptr_eq would make the async event \
             path dead code and every restored session would pay a host sync"
        );

        // 2. THE BLAST RADIUS. Both proposal seams guard the candidate buffer BEFORE the
        //    d2t remap (a sentinel would index the map out of bounds) and before the walk.
        let d2 = strip(include_str!("dflash.rs"));
        // Cut at the FIRST test module so no assertion can be satisfied by this test's own
        // string literals (the self-match trap the wiring-assertions law names). Note this
        // leaves only the pre-test part of the file live: a seam added AFTER the first
        // `#[cfg(test)]` would be invisible here and would need its own anchor.
        let live = &d2[..d2.find("#[cfg(test)]").expect("this file has test modules")];
        for (seam, arm) in [
            ("pub fn dflash2_propose_greedy_q(", "greedy"),
            ("pub(crate) fn dflash2_propose_sampled(", "sampled"),
        ] {
            let at = live
                .find(seam)
                .unwrap_or_else(|| panic!("{arm} proposal seam exists"));
            let window = live.get(at..at + 4000).unwrap_or(&live[at..]);
            let guard = window
                .find("dflash2_guard_candidates(&cand, n_vocab,")
                .unwrap_or_else(|| panic!("the {arm} proposal must guard its candidates"));
            let remap = window
                .find("*c = map[*c as usize];")
                .unwrap_or_else(|| panic!("the {arm} proposal still remaps through d2t"));
            assert!(
                guard < remap,
                "the {arm} guard must run before the d2t remap"
            );
        }
    }

    #[test]
    fn max_tokens_caps_the_committed_prefix_not_only_the_visible_slice() {
        // A round crossing the scheduler's 32-token quantum is not terminal when the
        // request still has room. The whole accepted prefix stays public and committed.
        assert_eq!(dspark_commit_limit(5, 30, 100), (5, false));
        // The same round at the true request boundary is clamped and terminal so the
        // parked cache cannot contain rows the worker did not publish.
        assert_eq!(dspark_commit_limit(5, 30, 33), (3, true));
        assert_eq!(dspark_commit_limit(2, 3, 10), (2, false));
        assert_eq!(dspark_commit_limit(1, 0, 1), (1, false));
    }

    /// f32 -> bf16 bytes (truncation; test values are bf16-exact small integers).
    fn bf16(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    const V: usize = 8; // test vocab
    const R: usize = 2; // selector rank
    const K: usize = 2; // top_k

    /// Codebooks for the chain tests: pred rows are one-hot-ish, succ rows chosen so
    /// the slot-1 winner FLIPS with the slot-0 choice.
    #[allow(clippy::identity_op)] // allow: the explicit +0/*1/>>0 terms document the lane/byte symmetry of the reference layout
    fn books() -> (Vec<u8>, Vec<u8>) {
        let mut pred = vec![0f32; V * R];
        pred[0] = 1.0; // tok 0: [1, 0]  (the anchor)
        pred[1 * R + 1] = 1.0; // tok 1: [0, 1]
        pred[2 * R] = 1.0; // tok 2: [1, 0]
        let mut succ = vec![0f32; V * R];
        succ[1 * R] = 2.0; // tok 1: [2, 0]
        succ[2 * R + 1] = 5.0; // tok 2: [0, 5]
        succ[3 * R + 1] = 3.0; // tok 3: [0, 3]
        succ[4 * R] = 10.0; // tok 4: [10, 0]
        (bf16(&pred), bf16(&succ))
    }

    #[test]
    fn selector_walk_is_a_chain_not_per_slot_argmax() {
        let (pred, succ) = books();
        // slot 0 candidates {1, 2}, slot 1 candidates {3, 4}; hproj all-ones.
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        let hproj = vec![1.0f32; 2 * R];
        // Anchor 0 (pred [1,0]): slot 0 scores = <[1,0],succ> -> tok1: 2, tok2: 0
        // -> picks 1. Slot 1 must then walk from pred[1]=[0,1]: tok3 scores 3,
        // tok4 scores 0 -> picks 3. A mutation that seeds every slot from the ANCHOR
        // (pred[0]=[1,0]) scores tok3: 0 / tok4: 10 and picks 4 instead — the chain
        // IS the semantics (reference CandidateSelector.select: `predecessor` is the
        // previously CHOSEN candidate, seeded by anchor_ids).
        let path = dflash2_walk_greedy(&pred, &succ, V, R, K, &[0.0; 4], &cand, &hproj, 0, 2);
        assert_eq!(
            path,
            vec![1, 3],
            "walk must seed slot p from slot p-1's CHOSEN candidate \
             (z-lab model.py CandidateSelector.select)"
        );
    }

    #[test]
    fn selector_walk_unary_term_participates() {
        let (pred, succ) = books();
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        let hproj = vec![1.0f32; 2 * R];
        // unary +10 on slot-0 candidate 2 overrides the bilinear 2-vs-0 margin;
        // the chain then walks from pred[2]=[1,0] and slot 1 flips to tok 4.
        let path = dflash2_walk_greedy(
            &pred,
            &succ,
            V,
            R,
            K,
            &[0.0, 10.0, 0.0, 0.0],
            &cand,
            &hproj,
            0,
            2,
        );
        assert_eq!(
            path,
            vec![2, 4],
            "score = unary + bilinear (reference: `unary[:, position] + einsum(...)`); \
             dropping the unary term picks tok 1 here"
        );
    }

    #[test]
    fn selector_walk_hidden_gate_participates() {
        let (pred, succ) = books();
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        // hproj [0, .] zeroes the pred[0]=[1,0] gate for slot 0: tok1's bilinear 2
        // vanishes, and the unary tiebreak (+1 on tok2) decides. The chain from tok2
        // (pred [1,0]) with slot-1 hproj [1,1] then picks tok4 (10 vs 0).
        let hproj = vec![0.0f32, 1.0, 1.0, 1.0];
        let path = dflash2_walk_greedy(
            &pred,
            &succ,
            V,
            R,
            K,
            &[0.0, 1.0, 0.0, 0.0],
            &cand,
            &hproj,
            0,
            2,
        );
        assert_eq!(
            path,
            vec![2, 4],
            "the bilinear gate is pred_row .* HIDDEN_PROJECTION (reference: \
             `predecessor_codebook(predecessor) * hidden[:, position]`); ignoring \
             hproj leaves tok1's margin standing"
        );
    }

    #[test]
    fn dflash2_harvest_is_census_keyed() {
        // DFlash2 is mask-fill BY CONSTRUCTION (reference dflash_generate harvests
        // rows 1-verify_size:; card: "7 draft tokens per verification step").
        assert_eq!(
            DsparkHarvest::for_family_value(true, None, false),
            DsparkHarvest::Dflash
        );
        assert_eq!(
            DsparkHarvest::for_family_value(true, Some("dflash"), false),
            DsparkHarvest::Dflash
        );
        // The family key BEATS the strategy census: a (hypothetical) DFlash2 export
        // whose config also strategy-censuses dspark still harvests mask-fill.
        assert_eq!(
            DsparkHarvest::for_family_value(true, None, true),
            DsparkHarvest::Dflash
        );
        // An env override to the SHIFTED harvest contradicts the census — REFUSE,
        // never re-key (the postmortem's misalignment class in reverse).
        assert!(
            std::panic::catch_unwind(|| DsparkHarvest::for_family_value(
                true,
                Some("dspark"),
                false
            ))
            .is_err(),
            "MEMRA_DSPARK_HARVEST=dspark on a DFlash2 checkpoint must refuse"
        );
        // Non-DFlash2 checkpoints ride the strategy-keyed resolution (env wins).
        assert_eq!(
            DsparkHarvest::for_family_value(false, Some("dspark"), false),
            DsparkHarvest::Dspark
        );
        assert_eq!(
            DsparkHarvest::for_family_value(false, None, false),
            DsparkHarvest::Dflash
        );
        assert_eq!(
            DsparkHarvest::for_family_value(false, None, true),
            DsparkHarvest::Dspark,
            "unset env on a DSPARK-strategy export must keep the ratified census flip"
        );
    }

    // ============ SAMPLED ADMISSION (T>0) gates — lane/dspark-sampled-admission-20260820 =
    // The device kernels are oracled by sample_check (filter_stats/gumbel/residual arms);
    // these pin the HOST math the route ships — the selector's sampled walk, the accept
    // rule, and the round COMPOSITION (accept + residual + bonus must reproduce the target
    // distribution p exactly; a mis-composition leaves every kernel individually correct,
    // which is why the composition arm exists — sample_check arm 6's lesson).

    #[test]
    fn sampled_walk_tiny_temp_matches_greedy() {
        // T->0 continuity: at tiny temperature the candidate softmax concentrates on the
        // argmax and the sampled walk must reproduce the greedy chain token-for-token
        // (the frspec gate-(1) shape). Same fixture as the chain test.
        let (pred, succ) = books();
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        let hproj = vec![1.0f32; 2 * R];
        let greedy = dflash2_walk_greedy(&pred, &succ, V, R, K, &[0.0; 4], &cand, &hproj, 0, 2);
        let mut u = || 0.5f32;
        let (path, q_chosen, q_rows) = dflash2_walk_sampled(
            &pred, &succ, V, R, K, &[0.0; 4], &cand, &hproj, 0, 2, 1e-6, &mut u,
        );
        assert_eq!(
            path, greedy,
            "tiny-T sampled walk must equal the greedy chain"
        );
        assert_eq!(q_rows.len(), 2 * K);
        for (p, &q) in path.iter().zip(&q_chosen) {
            let _ = p;
            assert!(
                q > 0.999,
                "tiny-T chosen-candidate prob must be ~1, got {q}"
            );
        }
    }

    #[test]
    fn sampled_walk_records_the_distribution_it_samples() {
        // The recorded q IS the proposal: per slot the q_rows sum to ~1, q_chosen is the
        // row value at the drawn candidate, and the CDF walk picks the candidate whose
        // cumulative bracket contains the uniform.
        let (pred, succ) = books();
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        let hproj = vec![1.0f32; 2 * R];
        // slot-0 scores at anchor 0: tok1 = 2.0, tok2 = 0.0; at T=2.0 the softmax is
        // e^1/(e^1+e^0) ~= 0.731 for tok1.
        let q1 = (1f64.exp() / (1f64.exp() + 1.0)) as f32;
        for (u0, want0) in [(q1 - 0.01, 1u32), (q1 + 0.01, 2u32)] {
            let mut seq = vec![u0, 0.0f32].into_iter();
            let mut u = move || seq.next().unwrap();
            let (path, q_chosen, q_rows) = dflash2_walk_sampled(
                &pred, &succ, V, R, K, &[0.0; 4], &cand, &hproj, 0, 2, 2.0, &mut u,
            );
            assert_eq!(
                path[0], want0,
                "CDF walk must place u={u0} in the right candidate bracket"
            );
            let row0: f32 = q_rows[..K].iter().sum();
            assert!(
                (row0 - 1.0).abs() < 1e-5,
                "slot-0 q must sum to 1, got {row0}"
            );
            let ci = cand[..K].iter().position(|&c| c == path[0]).unwrap();
            assert_eq!(
                q_chosen[0], q_rows[ci],
                "q_chosen must be the recorded row prob of the drawn candidate"
            );
            assert!(
                (q_rows[0] - q1).abs() < 1e-4,
                "slot-0 tok1 prob must be softmax(scores/T), got {} want {q1}",
                q_rows[0]
            );
        }
    }

    #[test]
    fn sampled_walk_chains_the_drawn_candidate() {
        // The chain conditions on the DRAWN candidate, not the argmax: forcing the
        // low-prob slot-0 candidate (tok 2) flips slot 1's winner (tok 4 over tok 3),
        // exactly like the greedy chain test — a walk that seeds every slot from the
        // anchor (or the argmax) fails here.
        let (pred, succ) = books();
        let cand: Vec<u32> = vec![1, 2, 3, 4];
        let hproj = vec![1.0f32; 2 * R];
        let mut seq = vec![0.99f32, 0.01].into_iter();
        let mut u = move || seq.next().unwrap();
        let (path, _, _) = dflash2_walk_sampled(
            &pred, &succ, V, R, K, &[0.0; 4], &cand, &hproj, 0, 2, 2.0, &mut u,
        );
        assert_eq!(path[0], 2, "u=0.99 must draw the low-prob candidate");
        assert_eq!(
            path[1], 4,
            "slot 1 must walk from pred[2] (the DRAWN token), which scores tok4 at 10 \
             — chaining from the anchor or the argmax picks tok3"
        );
    }

    #[test]
    fn rejection_accept_walk_is_the_leviathan_rule() {
        // accept while u*q < p, strict, prefix-stop at the first reject.
        assert_eq!(
            rejection_accept_len(&[0.5, 0.5], &[0.5, 0.5], &[0.9, 0.9]),
            2
        );
        assert_eq!(
            rejection_accept_len(&[0.5, 0.5], &[0.5, 0.5], &[1.0, 0.0]),
            0
        );
        // u*q == p is a REJECT (strict <) — the frspec test byte-for-byte.
        assert_eq!(rejection_accept_len(&[0.25], &[0.5], &[0.5]), 0);
        // q == 0 with p > 0 accepts unconditionally (the skey exactness signature).
        assert_eq!(rejection_accept_len(&[1e-6], &[0.0], &[0.999]), 1);
        // prefix stop: slot 1 rejects, slot 2 never tested.
        assert_eq!(
            rejection_accept_len(&[0.9, 0.0, 0.9], &[0.1, 0.9, 0.1], &[0.5, 0.5, 0.5]),
            1
        );
    }

    // ---- round composition: the committed-token distribution must equal the target p ----
    // CPU mirror of the shipped rule for the FIRST post-anchor slot: draft x ~ q, accept
    // iff u*q(x) < p(x) (rejection_accept_len — the shipped fn), else commit a residual
    // sample ~ norm(max(0, p - q)). The marginal of the committed token is exactly p —
    // for ANY q — which is the whole correctness claim of the route's sampled admission.

    fn tv(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0
    }

    /// One composed trial with an injectable accept rule; returns the committed token.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // allow: NaN must take this branch; !(a > b) is not a <= b under IEEE comparisons
    fn compose_once(
        p: &[f32],
        q: &[f32],
        u_draw: f32,
        u_accept: f32,
        u_resid: f32,
        invert_accept: bool,
        skip_q_in_residual: bool,
    ) -> usize {
        let n = p.len();
        // draft ~ q (CDF walk, the walk_sampled convention)
        let mut acc = 0f64;
        let mut x = n - 1;
        for (i, &qi) in q.iter().enumerate() {
            acc += qi as f64;
            if (u_draw as f64) < acc {
                x = i;
                break;
            }
        }
        let accepted = if invert_accept {
            !((u_accept as f64) * (q[x] as f64) < p[x] as f64)
        } else {
            rejection_accept_len(&p[x..=x], &q[x..=x], &[u_accept]) == 1
        };
        if accepted {
            return x;
        }
        // residual ~ norm(max(0, p - q)) (the device kernel's fixed-order CDF walk)
        let r: Vec<f64> = p
            .iter()
            .zip(q)
            .map(|(&pi, &qi)| {
                let qq = if skip_q_in_residual { 0.0 } else { qi as f64 };
                (pi as f64 - qq).max(0.0)
            })
            .collect();
        let total: f64 = r.iter().sum();
        let mut acc = 0f64;
        let target = u_resid as f64 * total;
        for (i, &ri) in r.iter().enumerate() {
            acc += ri;
            if acc >= target && ri > 0.0 {
                return i;
            }
        }
        n - 1
    }

    fn compose_tv(q: &[f32], invert_accept: bool, skip_q_in_residual: bool) -> f64 {
        // target p: a spread-out 8-token distribution
        let p: Vec<f32> = vec![0.30, 0.22, 0.15, 0.12, 0.09, 0.06, 0.04, 0.02];
        let trials = 200_000usize;
        let mut counts = [0f64; V];
        for t in 0..trials {
            // three independent uniforms per trial off the host Philox stream
            let u_draw = crate::spec::host_u01(7, (t * 3) as u32);
            let u_accept = crate::spec::host_u01(7, (t * 3 + 1) as u32);
            let u_resid = crate::spec::host_u01(7, (t * 3 + 2) as u32);
            counts[compose_once(
                &p,
                q,
                u_draw,
                u_accept,
                u_resid,
                invert_accept,
                skip_q_in_residual,
            )] += 1.0;
        }
        let emp: Vec<f64> = counts.iter().map(|c| c / trials as f64).collect();
        let pf: Vec<f64> = p.iter().map(|&v| v as f64).collect();
        tv(&emp, &pf)
    }

    #[test]
    fn sampled_round_composition_matches_the_target() {
        // Monte-Carlo floor at 200k draws over 8 tokens ~ 0.004 TV; bound 0.01.
        // (a) full-vocab q (the Rows families' shape), far from p;
        let q_rows: Vec<f32> = vec![0.02, 0.04, 0.06, 0.09, 0.12, 0.15, 0.22, 0.30];
        // (b) SPARSE candidate-set q (the DFlash2 selector shape: support on 2 of 8).
        let q_sparse: Vec<f32> = vec![0.0, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0];
        for (name, q) in [("rows", &q_rows), ("sparse", &q_sparse)] {
            let d = compose_tv(q, false, false);
            assert!(
                d < 0.01,
                "composition[{name}]: committed-token distribution must equal p \
                 (TV {d:.4} >= 0.01)"
            );
        }
    }

    #[test]
    fn composition_teeth_inverted_accept_fails() {
        // DECISIVE teeth: the same harness with the accept inequality inverted must
        // MISS the target — otherwise the composition gate is vacuous.
        let q: Vec<f32> = vec![0.02, 0.04, 0.06, 0.09, 0.12, 0.15, 0.22, 0.30];
        let d = compose_tv(&q, true, false);
        assert!(
            d > 0.05,
            "inverted accept rule must fail the composition bound (TV {d:.4})"
        );
    }

    #[test]
    fn composition_teeth_residual_without_q_fails() {
        // Sampling the reject slot from p instead of norm(max(0, p-q)) double-counts
        // the overlap mass min(p,q) — the committed distribution leaves p.
        let q: Vec<f32> = vec![0.0, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0];
        let d = compose_tv(&q, false, true);
        assert!(
            d > 0.05,
            "residual that skips the q subtraction must fail the bound (TV {d:.4})"
        );
    }

    // ---- PENALIZED round composition (lane/dspark-penalized-sampled-20260821) ----
    // Multi-slot rounds where the penalty state EVOLVES within the round. The base trunk
    // logits are state-independent, so ALL context dependence flows through penalties —
    // the sharpest fixture for "verify row j's target is penalized by the tokens accepted
    // before j in the same round", and the proposal concentrates on ONE token, so the
    // dominant drafted block is a self-hit (a drafted token penalizing its own successor
    // — the within-round case a frozen round-start window cannot see). The reference is
    // EXACT (analytic chain p1(a)·p2(b|a), the plain sampler's semantics); the spec arm
    // is the shipped round rule — chain draw from q, `rejection_accept_len`, residual
    // norm(max(0, p−q)) at the reject slot, bonus from the one-past row on full accept —
    // with per-slot penalized p (mirroring penalize_logits_rows_inc_f32's window rule).

    const PV: usize = 6;
    const PEN_REP: f32 = 1.6;
    const PEN_FREQ: f32 = 0.8;
    const PEN_PRESENT: f32 = 1.2;

    fn pen_base() -> Vec<f32> {
        vec![1.5, 0.8, 0.3, -0.2, -0.7, -1.2]
    }

    /// The proposal: heavy on token 0 so drafted blocks repeat it (the self-hit case).
    fn pen_q() -> Vec<f32> {
        vec![0.85, 0.06, 0.04, 0.03, 0.01, 0.01]
    }

    /// CPU mirror of penalize_logits_f32 / the plain sampler's apply_penalties: first
    /// occurrence does the whole adjustment, cnt = occurrences in the window, rep
    /// divides positive logits and multiplies negative ones.
    fn pen_apply(logits: &mut [f32], window: &[u32]) {
        let mut seen: Vec<u32> = Vec::new();
        for &id in window {
            if seen.contains(&id) {
                continue;
            }
            seen.push(id);
            let cnt = window.iter().filter(|&&h| h == id).count() as f32;
            let v = &mut logits[id as usize];
            if *v > 0.0 {
                *v /= PEN_REP;
            } else {
                *v *= PEN_REP;
            }
            *v -= PEN_FREQ * cnt + PEN_PRESENT;
        }
    }

    /// Penalized target at history `window` (temp 1.0, no truncation filters — those are
    /// orthogonal and covered by the unpenalized composition tests + kernel oracles).
    fn pen_target(window: &[u32]) -> Vec<f64> {
        let mut l = pen_base();
        pen_apply(&mut l, window);
        let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let ex: Vec<f64> = l.iter().map(|&v| ((v as f64) - mx).exp()).collect();
        let z: f64 = ex.iter().sum();
        ex.iter().map(|v| v / z).collect()
    }

    /// Which penalty-arm mutation the harness runs — `None` is the shipped rule.
    #[derive(Clone, Copy, PartialEq)]
    enum PenMutation {
        None,
        /// All rows (walk + bonus) penalized with the ROUND-START window only — the
        /// within-round update dropped (the frspec per-round posture; what a flat
        /// `penalize_logits_rows` launch would ship).
        FrozenWindow,
        /// Reject-slot residual computed from the UNPENALIZED p (a raw-tlogits column
        /// copy instead of the penalized buffer).
        UnpenalizedResidual,
        /// Full-accept bonus drawn from the UNPENALIZED one-past row.
        UnpenalizedBonus,
    }

    /// Emit `want` committed tokens through spec rounds of `k` drafts (the shipped round
    /// rule, penalty-aware) and return them. `hist0` = the pre-stream window (the prompt
    /// seed); uniforms come off the injected stream.
    fn pen_round_stream(
        hist0: &[u32],
        k: usize,
        want: usize,
        mutation: PenMutation,
        next_u: &mut dyn FnMut() -> f32,
    ) -> Vec<u32> {
        let q = pen_q();
        let mut hist: Vec<u32> = hist0.to_vec();
        let mut committed: Vec<u32> = Vec::new();
        while committed.len() < want {
            // draft k tokens ~ q (fixed-order CDF walk, the walk_sampled convention)
            let drafted: Vec<u32> = (0..k)
                .map(|_| {
                    let u = next_u() as f64;
                    let mut acc = 0f64;
                    let mut bi = 0usize;
                    for (i, &qi) in q.iter().enumerate() {
                        acc += qi as f64;
                        if u < acc {
                            bi = i;
                            break;
                        }
                    }
                    bi as u32
                })
                .collect();
            // per-slot penalized p at the drafted ids (row j's window = hist ++ drafted[..j])
            let pj: Vec<f32> = (0..k)
                .map(|j| {
                    let win: Vec<u32> = if mutation == PenMutation::FrozenWindow {
                        hist.clone()
                    } else {
                        hist.iter()
                            .copied()
                            .chain(drafted[..j].iter().copied())
                            .collect()
                    };
                    pen_target(&win)[drafted[j] as usize] as f32
                })
                .collect();
            let qj: Vec<f32> = drafted.iter().map(|&d| q[d as usize]).collect();
            let us: Vec<f32> = (0..k).map(|_| next_u()).collect();
            let m = rejection_accept_len(&pj, &qj, &us);
            committed.extend_from_slice(&drafted[..m]);
            hist.extend_from_slice(&drafted[..m]);
            let next: u32 = if m == k {
                // bonus ~ p at the one-past row (window carries the WHOLE drafted block)
                let win: Vec<u32> = if matches!(
                    mutation,
                    PenMutation::FrozenWindow | PenMutation::UnpenalizedBonus
                ) {
                    if mutation == PenMutation::UnpenalizedBonus {
                        Vec::new() // raw row: no penalties at all
                    } else {
                        hist[..hist.len() - m].to_vec() // round-start window
                    }
                } else {
                    hist.clone()
                };
                let p = pen_target(&win);
                let u = next_u() as f64;
                let mut acc = 0f64;
                let mut bi = PV - 1;
                for (i, &pi) in p.iter().enumerate() {
                    acc += pi;
                    if u < acc {
                        bi = i;
                        break;
                    }
                }
                bi as u32
            } else {
                // residual ~ norm(max(0, p_m − q)) at the reject slot's state
                let win: Vec<u32> = match mutation {
                    PenMutation::UnpenalizedResidual => Vec::new(),
                    PenMutation::FrozenWindow => hist[..hist.len() - m].to_vec(),
                    _ => hist.clone(),
                };
                let p = pen_target(&win);
                let r: Vec<f64> = p
                    .iter()
                    .zip(&q)
                    .map(|(&pi, &qi)| (pi - qi as f64).max(0.0))
                    .collect();
                let total: f64 = r.iter().sum();
                let target = next_u() as f64 * total;
                let mut acc = 0f64;
                let mut bi = PV - 1;
                for (i, &ri) in r.iter().enumerate() {
                    acc += ri;
                    if acc >= target && ri > 0.0 {
                        bi = i;
                        break;
                    }
                }
                bi as u32
            };
            committed.push(next);
            hist.push(next);
        }
        committed.truncate(want);
        committed
    }

    /// Joint TV of the spec arm's first two committed tokens vs the EXACT penalized
    /// chain p1(a)·p2(b|a) — the plain sampler's distribution over the same two steps.
    fn pen_compose_tv(hist0: &[u32], k: usize, mutation: PenMutation) -> f64 {
        let trials = 300_000usize;
        let mut counts = vec![0f64; PV * PV];
        for t in 0..trials {
            // stride 64: a k<=2 round consumes <=2k+1 uniforms, <=2 rounds per trial
            let mut ctr = (t as u32) * 64;
            let mut next_u = move || {
                let u = crate::spec::host_u01(11, ctr);
                ctr = ctr.wrapping_add(1);
                u
            };
            let s = pen_round_stream(hist0, k, 2, mutation, &mut next_u);
            counts[s[0] as usize * PV + s[1] as usize] += 1.0;
        }
        let p1 = pen_target(hist0);
        let mut tv = 0f64;
        for a in 0..PV {
            let mut w: Vec<u32> = hist0.to_vec();
            w.push(a as u32);
            let p2 = pen_target(&w);
            for b in 0..PV {
                let refp = p1[a] * p2[b];
                tv += (counts[a * PV + b] / trials as f64 - refp).abs();
            }
        }
        tv / 2.0
    }

    #[test]
    fn penalized_round_composition_matches_the_penalized_chain() {
        // MC floor at 300k trials over 36 cells ~ 0.004 TV; bound 0.01. Fixture (a):
        // k=2, empty prompt window — the drafted pair (0,0) dominates, so slot 2's
        // accept is the SELF-HIT case (its own predecessor was drafted this round).
        // Fixture (b): k=1, prompt window [1,1] — the bonus is the successor of a
        // same-round accepted draft, and cnt>1 exercises the freq×count path.
        for (name, hist0, k) in [
            ("k2-selfhit", vec![], 2usize),
            ("k1-bonus-successor", vec![1u32, 1u32], 1usize),
        ] {
            let d = pen_compose_tv(&hist0, k, PenMutation::None);
            eprintln!("penalized composition[{name}]: TV {d:.4} (bound 0.01)");
            assert!(
                d < 0.01,
                "penalized composition[{name}]: committed-token distribution must equal \
                 the penalized chain (TV {d:.4} >= 0.01)"
            );
        }
    }

    #[test]
    fn penalized_composition_teeth_frozen_window_fails() {
        // DECISIVE teeth: penalizing every verify row with the ROUND-START window —
        // dropping the within-round penalty update, i.e. a flat penalize_logits_rows
        // launch where the route ships penalize_logits_rows_inc — must MISS the
        // penalized chain, or the composition gate cannot see the one thing this lane
        // adds over the frozen-window prior art.
        let d = pen_compose_tv(&[], 2, PenMutation::FrozenWindow);
        eprintln!("penalized teeth[frozen-window]: TV {d:.4} (must exceed 0.05)");
        assert!(
            d > 0.05,
            "within-round penalty update dropped (frozen round-start window) must FAIL \
             the composition bound (TV {d:.4})"
        );
    }

    #[test]
    fn penalized_composition_teeth_unpenalized_residual_fails() {
        // The reject-slot residual must read the PENALIZED column: a raw-tlogits column
        // copy (p_raw − q) commits from the wrong measure. Non-empty prompt window so
        // even round-start reject slots hit the mutation (an empty-window fixture only
        // sees it on within-round rejects and the margin thins to ~0.055).
        let d = pen_compose_tv(&[1, 1], 2, PenMutation::UnpenalizedResidual);
        eprintln!("penalized teeth[unpenalized-residual]: TV {d:.4} (must exceed 0.05)");
        assert!(
            d > 0.05,
            "residual computed from the unpenalized p must FAIL the composition bound \
             (TV {d:.4})"
        );
    }

    #[test]
    fn penalized_composition_teeth_unpenalized_bonus_fails() {
        // The full-accept bonus row must carry the whole drafted block in its window:
        // a raw one-past row draw commits the unpenalized measure right after a
        // same-round accept.
        let d = pen_compose_tv(&[1, 1], 1, PenMutation::UnpenalizedBonus);
        eprintln!("penalized teeth[unpenalized-bonus]: TV {d:.4} (must exceed 0.05)");
        assert!(
            d > 0.05,
            "bonus drawn from the unpenalized one-past row must FAIL the composition \
             bound (TV {d:.4})"
        );
    }
}

#[cfg(test)]
mod dspark_harvest_tests {
    use super::{DsparkHarvest, DsparkVtPolicy, dspark_accept_prefix, dspark_strategy_census};

    const B: usize = 7; // q38 arm-a block_size

    #[test]
    fn dspark_strategy_requires_shifted_harvest() {
        let h = DsparkHarvest::Dspark;
        assert_eq!(
            h.first_row(),
            0,
            "DSPARK-strategy checkpoints (SpecForge OnlineDSparkModel, \
             training.strategy=dspark — the q38 arm-a export) supervise ALL rows with \
             SHIFTED labels: label_offsets = arange(1, block_size+1), i.e. the ANCHOR \
             row's output is draft 1 (specforge/algorithms/common/\
             dflash_family_model.py:816; sglang v0.5.17 dspark_draft.py:248,260). \
             Harvesting from row 1 re-opens the DSPARK-POSTMORTEM-20260820 slot \
             misalignment (accept 2.9 -> 1.43)."
        );
        assert_eq!(
            h.n_drafts(B),
            B,
            "DSpark harvests gamma = block_size drafts per round (sglang \
             dspark_config.py:269, verify_num_draft_tokens = gamma+1); b-1 is the \
             DFlash mask-fill count and drops the best-trained slot \
             (DSPARK-POSTMORTEM-20260820.md §3-H1)."
        );
        for row in 0..B {
            assert_eq!(
                h.trained_offset_of_row(row),
                row + 1,
                "OnlineDSparkModel trains row k to predict anchor+k+1 \
                 (dflash_family_model.py:816); a same-position (mask-fill) mapping \
                 here verifies every slot one position early — the postmortem's \
                 collapse."
            );
        }
    }

    #[test]
    fn dflash_strategy_keeps_mask_fill_harvest() {
        // Guards the reverse mutation: z-lab dflash checkpoints (the gemma arm) are
        // mask-fill — row k FILLS anchor+k, the anchor row is loss-excluded
        // (dflash_family_model.py:453-472). Shifting THEM would break the gemma arm.
        let h = DsparkHarvest::Dflash;
        assert_eq!(h.first_row(), 1, "DFlash drafts start at mask row 1");
        assert_eq!(h.n_drafts(B), B - 1, "DFlash harvests block_size-1 drafts");
        for row in 1..B {
            assert_eq!(h.trained_offset_of_row(row), row);
        }
    }

    #[test]
    fn every_candidate_verifies_the_position_its_row_was_trained_for() {
        // The round's invariant: draft candidate i (1-based; verified against the
        // trunk's prediction for anchor+i) is filled from drafter output row
        // first_row + i - 1. Alignment == that row was TRAINED for offset i.
        for h in [DsparkHarvest::Dflash, DsparkHarvest::Dspark] {
            for i in 1..=h.n_drafts(B) {
                let row = h.first_row() + i - 1;
                assert_eq!(
                    h.trained_offset_of_row(row),
                    i,
                    "{h:?}: candidate {i} rides row {row}, which is trained for \
                     offset {} — harvest misaligned",
                    h.trained_offset_of_row(row)
                );
            }
        }
    }

    #[test]
    fn env_seam_parses_and_refuses() {
        assert_eq!(
            DsparkHarvest::from_env_value(None),
            DsparkHarvest::Dflash,
            "the ENV-ONLY parser keeps the historical arm; the ratified strategy-keyed \
             default lives in resolve_value (checkpoint census), not here"
        );
        assert_eq!(
            DsparkHarvest::from_env_value(Some("dspark")),
            DsparkHarvest::Dspark
        );
        assert_eq!(
            DsparkHarvest::from_env_value(Some("dflash")),
            DsparkHarvest::Dflash
        );
        assert!(
            std::panic::catch_unwind(|| DsparkHarvest::from_env_value(Some("shifted"))).is_err(),
            "unknown harvest values must REFUSE, not default"
        );
        assert_eq!(
            DsparkHarvest::from_name("dspark"),
            Some(DsparkHarvest::Dspark)
        );
        assert_eq!(
            DsparkHarvest::from_name("dflash"),
            Some(DsparkHarvest::Dflash)
        );
        assert_eq!(DsparkHarvest::from_name("mask-fill"), None);
    }

    /// The owner-ratified default flips (2026-08-20). Each assertion names its
    /// evidence; mutating either resolve back to the old default fails these.
    #[test]
    fn ratified_default_harvest_is_strategy_keyed() {
        // DSPARK-strategy checkpoint + unset env = the shifted harvest (B1: accept
        // 1.38->2.41 agentic / 1.53->3.66 math, E2E ALL EXACT x5, interleaved x5).
        assert_eq!(
            DsparkHarvest::resolve_value(None, true),
            DsparkHarvest::Dspark,
            "owner-ratified 2026-08-20: unset env defaults a DSPARK-strategy \
             checkpoint to the shifted harvest (DSPARK-POSTMORTEM-20260820.md B1)"
        );
        // mask-fill checkpoint + unset env = the historical arm, byte-identical.
        assert_eq!(
            DsparkHarvest::resolve_value(None, false),
            DsparkHarvest::Dflash
        );
        assert_eq!(
            DsparkHarvest::resolve_value(Some(""), false),
            DsparkHarvest::Dflash
        );
        // Explicit env overrides the census in BOTH directions (the A/B seam).
        assert_eq!(
            DsparkHarvest::resolve_value(Some("dflash"), true),
            DsparkHarvest::Dflash
        );
        assert_eq!(
            DsparkHarvest::resolve_value(Some("dspark"), false),
            DsparkHarvest::Dspark
        );
        // Unknown values still REFUSE through the resolve path.
        assert!(
            std::panic::catch_unwind(|| DsparkHarvest::resolve_value(Some("shifted"), true))
                .is_err()
        );
    }

    #[test]
    fn strategy_census_reads_the_checkpoint_not_the_env() {
        // The q38 arm-a export shape: both signals present.
        let q38 = r#"{"architectures": ["Qwen3DSparkModel"], "block_size": 7,
            "dflash_config": {"projector_type": "dspark", "markov_rank": 256}}"#;
        assert!(dspark_strategy_census(q38));
        // Either signal alone suffices.
        assert!(dspark_strategy_census(
            r#"{"architectures": ["Qwen3DSparkModel"]}"#
        ));
        assert!(dspark_strategy_census(
            r#"{"dflash_config": {"projector_type": "dspark"}}"#
        ));
        // A mask-fill DFlash export carries neither -> historical default.
        let dflash = r#"{"architectures": ["Qwen3DFlashModel"],
            "dflash_config": {"attention_mode": "gqa"}}"#;
        assert!(!dspark_strategy_census(dflash));
        assert!(!dspark_strategy_census("{}"));
    }

    #[test]
    fn ratified_default_vt_is_confidence_slot_tau_half() {
        // Head-carrying checkpoint + unset env = confidence-slot tau=.5 (H4 cell 3:
        // the tau ladder's knee; cell 2: 93.9%/97.7% of fixed-8 accept at wall >=
        // the reactive ladder, exactness 11/11 ALL EXACT).
        assert_eq!(
            DsparkVtPolicy::resolve_value(None, None, None, true),
            DsparkVtPolicy::ConfidenceSlot { tau: 0.5 },
            "owner-ratified 2026-08-20: unset MEMRA_DSPARK_VT defaults to \
             confidence-slot tau=.5 on a head-carrying checkpoint (H4 cells 2-3)"
        );
        // tau env still steers the default arm (and a bad tau still refuses).
        assert_eq!(
            DsparkVtPolicy::resolve_value(None, Some("0.35"), None, true),
            DsparkVtPolicy::ConfidenceSlot { tau: 0.35 }
        );
        assert!(
            std::panic::catch_unwind(|| DsparkVtPolicy::resolve_value(
                None,
                Some("nan-ish"),
                None,
                true
            ))
            .is_err()
        );
        // Census: no accept-rate head -> nothing to schedule with -> ladder.
        assert_eq!(
            DsparkVtPolicy::resolve_value(None, None, None, false),
            DsparkVtPolicy::Ladder
        );
        // MEMRA_DFLASH_ADAPT=0 is an explicit fixed-window request: honored.
        assert_eq!(
            DsparkVtPolicy::resolve_value(None, None, Some("0"), true),
            DsparkVtPolicy::Ladder
        );
        // Explicit values keep their exact prior semantics through resolve.
        assert_eq!(
            DsparkVtPolicy::resolve_value(Some("ladder"), None, None, true),
            DsparkVtPolicy::Ladder
        );
        assert_eq!(
            DsparkVtPolicy::resolve_value(Some("confidence"), Some("0.35"), None, true),
            DsparkVtPolicy::Confidence { tau: 0.35 }
        );
        // Explicit confidence mode with ADAPT=0 stays a refusal.
        assert!(
            std::panic::catch_unwind(|| DsparkVtPolicy::resolve_value(
                Some("confidence-slot"),
                None,
                Some("0"),
                true
            ))
            .is_err()
        );
    }

    /// End-to-end alignment fixture in miniature: a mock drafter whose row r argmaxes
    /// to token BASE + (its trained offset under the DSPARK strategy), and a mock trunk
    /// whose prediction for anchor+j is BASE + j. The DSpark harvest accepts the whole
    /// block; feeding the same drafter through the mask-fill harvest accepts ZERO —
    /// the postmortem's collapse reproduced as pure logic.
    #[test]
    fn dspark_trained_rows_through_mask_fill_harvest_accept_nothing() {
        const BASE: u32 = 1000;
        let anchor: u32 = BASE; // token at the round anchor position (offset 0)
        // trunk verify argmaxes: vam[j] = prediction for anchor offset j+1
        let vam: Vec<u32> = (1..=B as u32 + 1).map(|j| BASE + j).collect();
        // drafter rows trained under the DSPARK strategy: row r predicts offset r+1
        let dspark_trained_row_argmax =
            |r: usize| BASE + DsparkHarvest::Dspark.trained_offset_of_row(r) as u32;

        // Correct (shifted) harvest: candidate i <- row i-1.
        let h = DsparkHarvest::Dspark;
        let mut cand = vec![anchor];
        for i in 1..=h.n_drafts(B) {
            cand.push(dspark_trained_row_argmax(h.first_row() + i - 1));
        }
        let vt = h.n_drafts(B) + 1;
        assert_eq!(
            dspark_accept_prefix(&cand, &vam, vt),
            vt - 1,
            "aligned harvest must accept the full block"
        );

        // Mask-fill harvest of the SAME dspark-trained drafter: candidate i <- row i,
        // which was trained for offset i+1 — every slot one position late.
        let wrong = DsparkHarvest::Dflash;
        let mut cand_wrong = vec![anchor];
        for i in 1..=wrong.n_drafts(B) {
            cand_wrong.push(dspark_trained_row_argmax(wrong.first_row() + i - 1));
        }
        let vt_wrong = wrong.n_drafts(B) + 1;
        assert_eq!(
            dspark_accept_prefix(&cand_wrong, &vam, vt_wrong),
            0,
            "mask-fill harvest of a dspark-trained drafter verifies every slot against \
             a position the row was not trained for (DSPARK-POSTMORTEM-20260820.md)"
        );
    }
}

// ================= Verify-window policy gate (CPU; H4, DSPARK-POSTMORTEM-20260820.md) ===
// Pins the confidence-vt semantics as logic the round consumes: cumprod survival over
// sigmoid scores, thresholded, anchor + kept drafts, floor 2 / cap vt_cap — and the env
// seam's refuse-on-ambiguity. Mutating the policy (per-slot threshold instead of
// survival, off-by-one on the anchor, silent unknown-value fallback) fails HERE.
#[cfg(test)]
mod dspark_vt_tests {
    use super::{ConfidenceHead, DsparkVtPolicy, dspark_confidence_vt, dspark_slot_confidence_vt};

    /// Pre-sigmoid logit for a target probability: sigmoid(logit(p)) == p.
    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn confidence_vt_is_cumprod_survival_not_per_slot_threshold() {
        // sigmoids = [0.9, 0.8, 0.9, ...]: every PER-SLOT score clears tau=0.5, but
        // cumulative survival sinks below it at slot 6 (0.9, 0.72, 0.648, 0.583,
        // 0.525, then 0.472 < 0.5) — the window must stop where the EXPECTED
        // accepted-prefix stops paying, not where a slot looks locally fine.
        let raws: Vec<f32> = [0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
            .iter()
            .map(|&p| logit(p))
            .collect();
        assert_eq!(
            dspark_confidence_vt(&raws, 0.5, 8),
            6,
            "keeps 5 drafts + anchor"
        );
        // Tighter threshold closes the window sooner; looser opens it to the cap.
        assert_eq!(
            dspark_confidence_vt(&raws, 0.7, 8),
            3,
            "tau=0.7 keeps 2 drafts"
        );
        assert_eq!(
            dspark_confidence_vt(&raws, 0.05, 8),
            8,
            "tau→0 = full block"
        );
    }

    #[test]
    fn slot_arm_truncates_at_first_low_confidence_slot() {
        // Owner directive (2026-08-20): submit only the longest prefix whose EVERY
        // slot clears tau on its own sigmoid. On the survival test's raws
        // ([0.9, 0.8, 0.9 x5], tau=0.5) every slot clears per-slot, so the slot arm
        // opens the full block where survival stopped at 6 — the two stopping
        // statistics must stay distinct arms.
        let raws: Vec<f32> = [0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
            .iter()
            .map(|&p| logit(p))
            .collect();
        assert_eq!(dspark_slot_confidence_vt(&raws, 0.5, 8), 8);
        assert_eq!(dspark_confidence_vt(&raws, 0.5, 8), 6);
        // A low-confidence tail never enters verify: [0.9, 0.9, 0.3, 0.9, ...]
        // truncates at slot 3 REGARDLESS of the confident slots behind it — a kept
        // slot after a dropped one could never commit (prefix accept rule).
        let tail: Vec<f32> = [0.9, 0.9, 0.3, 0.9, 0.9, 0.9, 0.9]
            .iter()
            .map(|&p| logit(p))
            .collect();
        assert_eq!(
            dspark_slot_confidence_vt(&tail, 0.5, 8),
            3,
            "2 drafts + anchor"
        );
        // Tighter tau keeps less.
        assert_eq!(
            dspark_slot_confidence_vt(&tail, 0.95, 8),
            2,
            "floor at tau=0.95"
        );
    }

    #[test]
    fn confidence_vt_floor_and_cap() {
        // A hopeless round still verifies ONE draft (the draft forward is paid;
        // vt=1 would guarantee an empty round at the same cost class).
        let cold: Vec<f32> = [0.1f32, 0.1, 0.1].iter().map(|&p| logit(p)).collect();
        assert_eq!(
            dspark_confidence_vt(&cold, 0.5, 8),
            2,
            "floor = anchor + 1 draft"
        );
        assert_eq!(
            dspark_slot_confidence_vt(&cold, 0.5, 8),
            2,
            "slot arm same floor"
        );
        // The MEMRA_DFLASH_VERIFY_T cap still binds a confident round.
        let hot: Vec<f32> = vec![logit(0.99); 7];
        assert_eq!(dspark_confidence_vt(&hot, 0.5, 5), 5, "vt_cap binds");
        assert_eq!(
            dspark_confidence_vt(&hot, 0.5, 8),
            8,
            "full block when confident"
        );
        assert_eq!(
            dspark_slot_confidence_vt(&hot, 0.5, 5),
            5,
            "slot arm same cap"
        );
        // No scores (defensive): floor.
        assert_eq!(dspark_confidence_vt(&[], 0.5, 8), 2);
        assert_eq!(dspark_slot_confidence_vt(&[], 0.5, 8), 2);
    }

    #[test]
    fn vt_policy_env_seam_parses_and_refuses() {
        assert_eq!(
            DsparkVtPolicy::from_env_value(None, None, None),
            DsparkVtPolicy::Ladder,
            "default stays the shipped ladder — the H4 arm is opt-in"
        );
        assert_eq!(
            DsparkVtPolicy::from_env_value(Some(""), None, None),
            DsparkVtPolicy::Ladder
        );
        assert_eq!(
            DsparkVtPolicy::from_env_value(Some("ladder"), None, Some("0")),
            DsparkVtPolicy::Ladder,
            "ladder + ADAPT=0 = the fixed-window arm, untouched"
        );
        assert_eq!(
            DsparkVtPolicy::from_env_value(Some("confidence"), None, None),
            DsparkVtPolicy::Confidence { tau: 0.5 },
            "tau defaults to 0.5 (raw sigmoid, no STS sidecar — postmortem §3-H4)"
        );
        assert_eq!(
            DsparkVtPolicy::from_env_value(Some("confidence"), Some("0.35"), Some("1")),
            DsparkVtPolicy::Confidence { tau: 0.35 }
        );
        assert_eq!(
            DsparkVtPolicy::from_env_value(Some("confidence-slot"), Some("0.6"), None),
            DsparkVtPolicy::ConfidenceSlot { tau: 0.6 },
            "the owner-directive per-slot arm parses with the same tau env"
        );
        assert!(
            std::panic::catch_unwind(|| DsparkVtPolicy::from_env_value(
                Some("confidence-slot"),
                None,
                Some("0")
            ))
            .is_err(),
            "confidence-slot + MEMRA_DFLASH_ADAPT=0 must REFUSE like confidence"
        );
        assert!(
            std::panic::catch_unwind(|| DsparkVtPolicy::from_env_value(Some("static"), None, None))
                .is_err(),
            "unknown policy values must REFUSE, not default — a typo silently \
             reverting the window policy invalidates an A/B"
        );
        assert!(
            std::panic::catch_unwind(|| DsparkVtPolicy::from_env_value(
                Some("confidence"),
                None,
                Some("0")
            ))
            .is_err(),
            "confidence + MEMRA_DFLASH_ADAPT=0 is contradictory and must REFUSE"
        );
        for bad in ["0", "1", "1.5", "-0.1", "nan"] {
            assert!(
                std::panic::catch_unwind(|| DsparkVtPolicy::from_env_value(
                    Some("confidence"),
                    Some(bad),
                    None
                ))
                .is_err(),
                "tau={bad} must REFUSE (survival threshold lives in (0,1))"
            );
        }
    }

    #[test]
    fn raw_score_matches_the_parity_gate_dot() {
        // The head is a raw linear proj over [hidden ; markov_prev_embedding] + b —
        // the exact stage-5 contract in dspark_q38_parity.rs.
        let ch = ConfidenceHead {
            w: vec![0.5, -1.0, 2.0, 0.25, -0.5],
            b: 0.125,
            in_dim: 5,
            with_markov: true,
        };
        let hidden = [1.0f32, 2.0, 3.0];
        let emb = [4.0f32, 8.0];
        let want = 0.125 + 0.5 * 1.0 - 1.0 * 2.0 + 2.0 * 3.0 + 0.25 * 4.0 - 0.5 * 8.0;
        assert_eq!(ch.raw_score(&hidden, Some(&emb)), want);
        let ch_plain = ConfidenceHead {
            w: vec![0.5, -1.0, 2.0],
            b: -0.25,
            in_dim: 3,
            with_markov: false,
        };
        let want_plain = -0.25 + 0.5 * 1.0 - 1.0 * 2.0 + 2.0 * 3.0;
        assert_eq!(ch_plain.raw_score(&hidden, None), want_plain);
    }
}

#[cfg(test)]
mod dspark_prefix_capture_tests {
    use super::{dspark_spec_prompt_fits, take_dspark_prefix_capture};

    /// INCIDENT REGRESSION (2026-08-25). This gate is the only admission check the dspark
    /// route has, and it checked the ctx ceiling ONLY — so a short prompt was admitted and
    /// then panicked in the cold prime (`prime_cache needs T >= 16`), inside the GPU worker
    /// thread, which exits 70 and kills every live session on the box. Two crash loops and
    /// ~5 minutes of customer 502s came from a 5-token "Say OK." — the class our own
    /// watchdog sends. The floor belongs HERE, in the gate, not in each caller.
    #[test]
    fn a_prompt_below_the_prime_floor_never_enters_the_dspark_route() {
        let floor = crate::hybrid_forward::PRIME_MIN_T;
        for short in [1usize, 5, floor - 1] {
            assert!(
                !dspark_spec_prompt_fits(short, 262_144, 8, 2_048, true),
                "a {short}-token prompt must decline to the plain path, not prime"
            );
        }
        // At and above the floor the route admits exactly as before (ceiling still applies).
        assert!(dspark_spec_prompt_fits(floor, 262_144, 8, 2_048, true));
        assert!(dspark_spec_prompt_fits(512, 262_144, 8, 2_048, true));
        assert!(!dspark_spec_prompt_fits(512, 300, 8, 2_048, true));
    }

    #[test]
    fn session_prompt_preflight_matches_dflash2_and_windowed_caps() {
        // DFlash2 uses the request ctx cap: prompt + block + 8 fits exactly, one row less does not.
        assert!(dspark_spec_prompt_fits(96, 111, 7, 2_048, true));
        assert!(!dspark_spec_prompt_fits(96, 110, 7, 2_048, true));

        // Legacy/windowed drafts are additionally bounded by their own sliding window.
        assert!(dspark_spec_prompt_fits(113, 8_192, 7, 128, false));
        assert!(!dspark_spec_prompt_fits(114, 8_192, 7, 128, false));
        assert!(!dspark_spec_prompt_fits(
            usize::MAX,
            usize::MAX,
            7,
            usize::MAX,
            true,
        ));
    }

    #[test]
    fn prompt_end_capture_is_full_prompt_and_one_shot() {
        let prompt_len = 96;
        let mut slot = Some(crate::spec::SpecBoundaryCapture {
            snap: crate::cache::CacheSnapshot {
                kv_len: Vec::new(),
                tp_kv_len: Vec::new(),
                conv: Vec::new(),
                ssm: Vec::new(),
                pos: prompt_len,
            },
            pos: prompt_len,
            logits: vec![1.0, 2.0],
            last_h: Vec::new(),
            latent_tails: Vec::new(),
        });

        let capture = take_dspark_prefix_capture(&mut slot).expect("first drain gets capture");
        assert_eq!(capture.pos, prompt_len, "capture is at full prompt end");
        assert_eq!(capture.snap.pos, prompt_len);
        assert!(
            capture.last_h.is_empty(),
            "DFlash publishes no hidden anchor"
        );
        assert!(
            take_dspark_prefix_capture(&mut slot).is_none(),
            "capture drains exactly once",
        );
    }
}

#[cfg(test)]
mod dflash_precision_tests {
    use super::dflash_precision;

    #[test]
    fn default_and_supported_precision_programs_are_explicit() {
        assert_eq!(dflash_precision(None), Ok("q4"));
        for prec in ["q4", "q8", "mixed", "bf16", "fc"] {
            assert_eq!(dflash_precision(Some(prec)), Ok(prec));
        }
    }

    #[test]
    fn q5_and_typos_refuse_instead_of_silently_selecting_q8() {
        for prec in ["q5", "Q4", "", "typo"] {
            let err = dflash_precision(Some(prec)).unwrap_err();
            assert!(err.contains("want q4, q8, mixed, bf16, or fc"));
        }
    }
}

#[cfg(test)]
mod dflash_tensor_contract_tests {
    use super::{
        validate_dflash_attention_geometry, validate_dflash_tensor, validate_layer_layout,
        validate_selector_top_k,
    };
    use memra_gguf::safetensors::StInfo;

    #[test]
    fn named_tensor_contract_refuses_wrong_dtype_rank_and_shape_before_cuda() {
        let valid = StInfo {
            dtype: "BF16".into(),
            shape: vec![8, 4],
            data_offsets: [0, 64],
        };
        assert!(validate_dflash_tensor("w", &valid, &[4, 8]).is_ok());

        let mut bad = valid.clone();
        bad.dtype = "F32".into();
        assert!(
            validate_dflash_tensor("w", &bad, &[4, 8])
                .unwrap_err()
                .contains("dtype")
        );
        bad = valid.clone();
        bad.shape = vec![32];
        assert!(
            validate_dflash_tensor("w", &bad, &[4, 8])
                .unwrap_err()
                .contains("shape")
        );
        assert!(
            validate_dflash_tensor("w", &valid, &[8, 4])
                .unwrap_err()
                .contains("expected")
        );
    }

    #[test]
    fn attention_and_selector_geometry_refuse_before_cuda() {
        assert!(validate_dflash_attention_geometry(64, 8, 128).is_ok());
        assert!(validate_dflash_attention_geometry(63, 8, 128).is_err());
        assert!(validate_dflash_attention_geometry(64, 0, 128).is_err());
        assert!(validate_dflash_attention_geometry(usize::MAX, 1, 2).is_err());
        assert!(validate_selector_top_k(16, 128).is_ok());
        assert!(validate_selector_top_k(0, 128).is_err());
        assert!(validate_selector_top_k(129, 128).is_err());
        assert!(validate_layer_layout(&[true, true], 2).is_ok());
        assert!(validate_layer_layout(&[true], 2).is_err());
    }
}
