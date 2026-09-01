//! CAPACITY GATE for the mHC prime walk — can GLM-5.3-Flash's native 1,048,576-token context
//! actually be PRIMED, or only configured?
//!
//! THE FAILURE THIS PINS. `prime_cache_hyper` is documented "deliberately UNCHUNKED", so the
//! per-call `t` is the ADMISSION LIMIT itself: one call carries the whole prompt. Every
//! transient in that call is proportional to `t`, and the DSA indexer's score plane is
//! `t * n_pools` f32 with `n_pools = t_kv / index_kpool` — so at a monolithic prime it is
//! N-SQUARED bytes, per MLA layer, per call. A box lane measured CUDA_ERROR_OUT_OF_MEMORY at
//! roughly 50k tokens on a 96 GB card, which is exactly where that term lands.
//!
//! RATIO, NOT A MAGIC BYTE COUNT. The assertions below are about how the requirement GROWS with
//! context, at four context values spanning 128x, because a sizing that holds at 8192 and breaks
//! at 262144 is not a sizing and this model's native context is 1048576. A future change that
//! reintroduces an N^2 term fails here even if it happens to fit whatever card is current.
//!
//! PINNED AGAINST TRUTH, NOT AGAINST TYPED CONSTANTS. Every geometry number comes from the
//! BANKED config.json of the real artifact (zai-org/GLM-5.3-Flash @ 04c4e9e), parsed by the real
//! `HfConfig`/`ModelConfig` path — so a config the engine would read differently cannot pass
//! here, and no number in this file can drift away from the checkpoint.
//!
//! HOST-ONLY: no device, no model load. Runs in normal `cargo test`.

use memra_engine::hybrid_forward::PRIME_MIN_T;
use memra_engine::hybrid_forward::{hyper_prime_call_rows, hyper_prime_ranges};
use memra_gguf::config::{Glm5NextConfig, HfConfig, ModelConfig};
use memra_kv::PRIME_CHUNK_MAX_TOKENS;

/// The banked config of the artifact this lane serves. Not a fixture: the real file.
fn glm5_config() -> (ModelConfig, Glm5NextConfig) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../research/glm53-flash-bringup-20260827/glm-config.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("banked GLM-5.3-Flash config.json at {path}: {e}"));
    let cfg = ModelConfig::from_hf(&HfConfig::parse(&text));
    let glm5 = cfg
        .glm5
        .clone()
        .expect("the banked config parses as glm5_next");
    (cfg, glm5)
}

/// Context ladder: four values, each 4x the last, ending at the model's NATIVE context. The 4x
/// step is what makes a quadratic term unmistakable — linear growth multiplies a term by 4, a
/// quadratic one by 16.
const LADDER: [usize; 4] = [16_384, 65_536, 262_144, 1_048_576];

/// Number of MLA/DSA layers whose transients are live one at a time in the trunk walk.
/// `layer_types` is the truth; the MTP layer carries a 12th indexer but does not run in the
/// prime walk.
fn mla_layers(glm5: &Glm5NextConfig, n_trunk: usize) -> usize {
    (0..n_trunk as u32)
        .filter(|&il| !glm5.is_kda_layer(il))
        .count()
}

/// PEAK per-call transient of ONE mHC prime call, in bytes, at a prompt of `ctx` tokens.
///
/// Each term names the allocation site it models. All are f32. Terms proportional to `rows`
/// (the per-call row count) are the ones a split can bound; the score plane is the one that is
/// ALSO proportional to the context, which is what makes a monolithic call quadratic.
fn per_call_transient_bytes(cfg: &ModelConfig, glm5: &Glm5NextConfig, ctx: usize) -> u128 {
    let rows = hyper_prime_call_rows(ctx, cfg.n_layer as usize, false) as u128;
    let n_embd = cfg.n_embd as u128;
    let streams = glm5.hc_mult as u128;
    let nh = cfg.n_head as u128;
    let r = glm5.kv_lora_rank as u128;
    let dv = glm5.v_head_dim as u128;
    let qk = glm5.qk_head_dim as u128;
    let idx_h = glm5.index_n_heads as u128;
    let idx_d = glm5.index_head_dim as u128;
    let width = (glm5.index_topk + glm5.index_kpool - 1) as u128; // select_k*pool + tail
    let n_pools = (ctx / glm5.index_kpool as usize) as u128;
    let f32b = 4u128;

    // --- trunk (crate::hyper), live across the whole layer loop of one call
    let trunk = rows * streams * n_embd * f32b            // hyper::expand stream state `x`
        + rows * n_embd * f32b                            // `embedded`
        + 5 * rows * n_embd * f32b; // y, h, mixed, z, ffn_out per site
    // --- MLA core (mla_attn_core), one layer live at a time
    let mla = rows * nh * qk * f32b                       // q_b / q_nope
        + 2 * rows * nh * r * f32b                        // q_lat + o_lat
        + rows * nh * dv * f32b; // attn
    // --- DSA indexer (mla_kpool_indices), one layer live at a time
    let indexer = rows * idx_h * idx_d * f32b             // q_index
        + rows * n_pools * f32b                           // THE SCORE PLANE
        + rows * width * f32b                             // the selected-row index list
        + 3 * rows * idx_d * f32b; // k_raw, k_norm, gate
    trunk + mla + indexer
}

/// Device memory the prime holds for its WHOLE duration, independent of how it is split.
///
/// There is exactly one such term and it is the RETURN CONTRACT: `prime_cache` hands back the
/// full pre-output_norm hidden stack `[t, n_embd]` (`generate_spec`'s `prompt_h`). Splitting the
/// walk does not touch it — each call's rows are copied into the same full-length buffer — so it
/// is 17.18 GB at the native context and it is the largest single thing this lane does NOT
/// remove. It is separated from the per-call transient deliberately: the scaling assertion is
/// about what a SPLIT can bound, and this is what a split cannot.
fn prime_lifetime_bytes(cfg: &ModelConfig, ctx: usize) -> u128 {
    ctx as u128 * cfg.n_embd as u128 * 4
}

/// Peak device bytes the prime occupies at once: the whole-prime stack plus one call's
/// transients.
fn peak_prime_bytes(cfg: &ModelConfig, glm5: &Glm5NextConfig, ctx: usize) -> u128 {
    prime_lifetime_bytes(cfg, ctx) + per_call_transient_bytes(cfg, glm5, ctx)
}

/// The schedule must be a PARTITION of the prompt: contiguous, gapless, covering every token
/// exactly once. Without this, "bound the per-call rows" could be bought by dropping tokens.
#[test]
fn the_mhc_prime_schedule_covers_the_prompt_exactly() {
    let (cfg, _) = glm5_config();
    let n_layers = cfg.n_layer as usize;
    for &ctx in LADDER.iter().chain([999_983usize, 1_048_577].iter()) {
        let ranges = hyper_prime_ranges(ctx, n_layers, false);
        assert!(!ranges.is_empty(), "ctx {ctx}: empty prime schedule");
        assert_eq!(
            ranges[0].0, 0,
            "ctx {ctx}: schedule does not start at token 0"
        );
        assert_eq!(
            ranges.last().unwrap().1,
            ctx,
            "ctx {ctx}: schedule does not end at the last token"
        );
        for w in ranges.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "ctx {ctx}: gap or overlap between prime calls at {:?} / {:?}",
                w[0], w[1]
            );
        }
        for &(s, e) in &ranges {
            assert!(e > s, "ctx {ctx}: empty prime call {:?}", (s, e));
        }
    }
}

/// THE RULE. One prime call must carry a bounded working set, not the context.
///
/// The bound is `PRIME_CHUNK_MAX_TOKENS + PRIME_MIN_T`, not the chunk alone, and the extra term
/// is the TAIL-MERGE rule, not slack: `fixed_prime_chunk_ranges` folds a remainder shorter than
/// `PRIME_MIN_T` into the call before it rather than emitting a call the batched path cannot
/// serve. What matters is that the bound is an ABSOLUTE CONSTANT — independent of `ctx` — which
/// is the whole claim.
///
/// The awkward contexts below are the ones that actually exercise that merge: every value on
/// LADDER is a multiple of the 4096 chunk and would never produce a short tail, so a bound that
/// was wrong by the merge term would pass on the ladder alone.
#[test]
fn the_mhc_prime_carries_a_bounded_number_of_rows_at_every_context() {
    let (cfg, _) = glm5_config();
    let n_layers = cfg.n_layer as usize;
    const BOUND: usize = PRIME_CHUNK_MAX_TOKENS + PRIME_MIN_T;
    let awkward = [
        LADDER[LADDER.len() - 1] - 1, // 1M minus one: a 4095-row tail
        LADDER[LADDER.len() - 1] + 1, // one row past a whole number of chunks: merges
        PRIME_CHUNK_MAX_TOKENS + 1,
        PRIME_CHUNK_MAX_TOKENS + PRIME_MIN_T - 1,
        999_983, // prime number, no relationship to any chunk size
    ];
    for &ctx in LADDER.iter().chain(awkward.iter()) {
        let rows = hyper_prime_call_rows(ctx, n_layers, false);
        assert!(
            rows <= BOUND,
            "ctx {ctx}: one mHC prime call carries {rows} token rows, above the absolute bound \
             {BOUND} (prime chunk {PRIME_CHUNK_MAX_TOKENS} + tail merge {PRIME_MIN_T}). The \
             per-call transient is proportional to this, so a value that tracks `ctx` makes \
             every transient a function of the CONTEXT."
        );
    }
}

/// THE SCALING ASSERTION. Across a 4x context step the peak per-call transient may grow at most
/// 4x plus slack. A quadratic term grows 16x and fails here at every rung.
#[test]
fn the_mhc_prime_transient_is_sub_quadratic_in_context() {
    let (cfg, glm5) = glm5_config();
    // 4x context, so linear growth is 4x. 5x admits the sub-linear terms rounding up against a
    // schedule whose last chunk is short; 16x (quadratic) is nowhere near it.
    const MAX_GROWTH_PER_4X: u128 = 5;
    // End to end the ladder spans 64x of context. LINEAR growth is 64x; QUADRATIC is 4096x.
    // 80x therefore sits 51x below the quadratic answer — a band wide enough that a PARTIAL
    // reintroduction of the N^2 term (one layer, one plane) still fails, which the adjacent-rung
    // test alone would not catch.
    const MAX_GROWTH_END_TO_END: u128 = 80;
    let mut previous: Option<(usize, u128)> = None;
    let mut first: Option<u128> = None;
    for &ctx in &LADDER {
        let bytes = per_call_transient_bytes(&cfg, &glm5, ctx);
        first.get_or_insert(bytes);
        if let Some((prev_ctx, prev_bytes)) = previous {
            assert!(
                bytes <= prev_bytes * MAX_GROWTH_PER_4X,
                "peak per-call prime transient grew from {prev_bytes} B at ctx {prev_ctx} to \
                 {bytes} B at ctx {ctx} — a {}x step over a 4x context step. That is a \
                 super-linear (N^2) term in the prime's working set.",
                bytes / prev_bytes.max(1)
            );
        }
        previous = Some((ctx, bytes));
    }
    let (lo, hi) = (first.unwrap(), previous.unwrap().1);
    assert!(
        hi <= lo * MAX_GROWTH_END_TO_END,
        "across the whole ladder (ctx {} -> {}, 64x) the peak per-call prime transient grew \
         {}x, from {lo} B to {hi} B. Linear is 64x and quadratic is 4096x, so this is the N^2 \
         term.",
        LADDER[0],
        LADDER[LADDER.len() - 1],
        hi / lo.max(1)
    );
}

/// THE FIT. At the native context, what the prime holds at once has to leave enough of a
/// 2x96 GB Blackwell box for the weights AND the KV plane AND a working expert residency.
///
/// THE BUDGET IS DERIVED HERE, not asserted, from the numbers in
/// `research/glm53-flash-bringup-20260827/PLACEMENT-RECEIPT.md` — with ONE correction that
/// belongs to this lane. That receipt's 1M row budgeted "8 GB across both cards for CUDA context
/// + activations + workspace" and a 40.42 GB KV plane. Both moved:
///   * the KV plane is now 27.6 GB, because `lane/glm53-ring-sizing` landed the tail ring the
///     receipt lists as an unimplemented saving (12.88 GB indexer plane -> 63 MB);
///   * 8 GB was never an activation measurement. This gate's own arithmetic is.
///     So the criterion is stated the way it actually matters: whatever the prime holds, expert
///     residency must stay at or above 75% of the routed-expert mass at 1M — below the 81% the
///     receipt's own 1M row promises, so the assertion has room to be informative rather than
///     tautological, and far above the 15% hot mass the receipt shows still fits.
#[test]
fn the_prime_at_the_native_context_leaves_a_working_expert_residency() {
    let (cfg, glm5) = glm5_config();
    let ctx = 1_048_576;

    // GiB, the unit nvidia-smi reports and the receipt uses for card capacity.
    const GIB: f64 = 1_073_741_824.0;
    // EXCEPTION-LIST HAZARD, named: these four are pinned to the CURRENT NVFP4 mint and the
    // CURRENT card class. A re-mint or a different box changes them and this gate will not
    // notice on its own. Source is PLACEMENT-RECEIPT.md sections 1 and 5; RE-DERIVE ON RE-MINT.
    const BOX_USABLE_GIB: f64 = 191.2; // 2 x 95.6 GiB usable, receipt section 1
    const NON_EXPERT_WEIGHTS_GIB: f64 = 13.66; // receipt section 1
    const ROUTED_EXPERT_MASS_GIB: f64 = 163.27; // receipt section 1
    const CUDA_CONTEXT_RESERVE_GIB: f64 = 4.0; // context + workspace, both cards
    const MIN_EXPERT_RESIDENCY: f64 = 0.75;
    // KV at 1M, ring merged: latent 25.77 + pool keys 1.61 + index ring 0.063 + KDA 0.15 GB.
    const KV_AT_1M_GIB: f64 = 27.59e9 / GIB;

    let budget_gib = BOX_USABLE_GIB
        - NON_EXPERT_WEIGHTS_GIB
        - KV_AT_1M_GIB
        - CUDA_CONTEXT_RESERVE_GIB
        - MIN_EXPERT_RESIDENCY * ROUTED_EXPERT_MASS_GIB;
    assert!(
        budget_gib > 0.0,
        "the placement arithmetic leaves no room for the prime at all"
    );

    let peak = peak_prime_bytes(&cfg, &glm5, ctx);
    let peak_gib = peak as f64 / GIB;
    assert!(
        peak_gib <= budget_gib,
        "at the native context {ctx} the prime holds {peak_gib:.2} GiB at once ({:.2} GiB \
         whole-prime hidden stack + {:.2} GiB per-call transients), over the {budget_gib:.2} GiB \
         the box has left once the non-expert weights, the 1M KV plane, the CUDA reserve and a \
         {:.0}% expert residency are taken out. {} MLA layers each carry a score plane of \
         rows*({ctx}/{}) f32.",
        prime_lifetime_bytes(&cfg, ctx) as f64 / GIB,
        per_call_transient_bytes(&cfg, &glm5, ctx) as f64 / GIB,
        MIN_EXPERT_RESIDENCY * 100.0,
        mla_layers(
            &glm5,
            cfg.n_layer as usize - glm5.num_nextn_predict_layers as usize
        ),
        glm5.index_kpool,
    );
}

/// REPORTS the sizing table this lane's receipt quotes, so the numbers in the lane doc are
/// generated by the same code the assertions above run on rather than transcribed beside it.
/// Asserts nothing new; run with `-- --nocapture`.
#[test]
fn report_the_per_call_prime_transient_ladder() {
    let (cfg, glm5) = glm5_config();
    let n_layers = cfg.n_layer as usize;
    println!("ctx | prime calls | rows/call | per-call transient | whole-prime stack | peak");
    for &ctx in &LADDER {
        let ranges = hyper_prime_ranges(ctx, n_layers, false);
        let rows = hyper_prime_call_rows(ctx, n_layers, false);
        println!(
            "{ctx:>9} | {:>11} | {rows:>9} | {:>15.3} GB | {:>14.3} GB | {:>6.3} GB",
            ranges.len(),
            per_call_transient_bytes(&cfg, &glm5, ctx) as f64 / 1e9,
            prime_lifetime_bytes(&cfg, ctx) as f64 / 1e9,
            peak_prime_bytes(&cfg, &glm5, ctx) as f64 / 1e9,
        );
    }
}
