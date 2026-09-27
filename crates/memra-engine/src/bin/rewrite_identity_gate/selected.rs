//! Actual refusal and fallback observations for the explicit Eager/Graph-only scope.
use super::*;

fn require_prime_absent(model: &HybridModel) -> Result<()> {
    qualified(model, Scope::EagerGraph)?;
    require(
        !model.rewrite_allowed(RewriteSurface::CarriedPrime),
        "PRIME_PERMISSION_PRESENT",
    )
}

fn refusal<T>(
    result: Result<T>,
    expected: &str,
    label: &str,
    evidence: &mut Evidence,
) -> Result<()> {
    match result {
        Ok(_) => Err(format!("UNQUALIFIED_PRIME_ACCEPTED {label}").into()),
        Err(error) => {
            evidence.event(format!("PRIME_REFUSAL api={label} error={error}"))?;
            require(
                error.to_string().contains(expected),
                format!("PRIME_REFUSAL_WRONG_REASON {label}: {error}"),
            )
        }
    }
}

pub(super) fn prime_api_refusal(
    e: &Engine,
    model: &HybridModel,
    evidence: &mut Evidence,
) -> Result<()> {
    require_prime_absent(model)?;
    let mut cache = Cache::new(e, &model.cfg, PRIME_BUCKET + STEPS + 8)?;
    let n_embd = model.cfg.n_embd as usize;
    let vocab = model.cfg.n_vocab as usize;
    let x = e.zeros(PRIME_BUCKET * n_embd)?;
    let pos = e.htod_i32(&(0..PRIME_BUCKET as i32).collect::<Vec<_>>())?;
    let len = e.htod_i32(&[PRIME_BUCKET as i32])?;
    let mut logits = e.htod(&vec![1.25f32; vocab])?;
    let mut seed = e.htod(&vec![-2.5f32; n_embd])?;
    let before = cache_sha256(e, &cache)?;
    let before_logits = e.dtoh(&logits)?;
    let before_seed = e.dtoh(&seed)?;
    evidence.bytes("prime-api-cache-before.txt", before.as_bytes())?;
    evidence.logits("prime-api-logits-before", &before_logits, vocab)?;
    evidence.logits("prime-api-seed-before", &before_seed, n_embd)?;

    refusal(
        model.prime_graph_new(e, PRIME_BUCKET),
        "carried-prime rewrite is not qualified",
        "prime_graph_new",
        evidence,
    )?;
    // Structurally valid inputs; the actual entry must refuse before touching them.
    let result = model.prime_chunk_captured(
        e,
        &x,
        &pos,
        PRIME_BUCKET,
        &mut cache,
        &len,
        &mut logits,
        &mut seed,
    );
    let after = cache_sha256(e, &cache)?;
    let after_logits = e.dtoh(&logits)?;
    let after_seed = e.dtoh(&seed)?;
    evidence.bytes("prime-api-cache-after.txt", after.as_bytes())?;
    evidence.logits("prime-api-logits-after", &after_logits, vocab)?;
    evidence.logits("prime-api-seed-after", &after_seed, n_embd)?;
    refusal(
        result,
        "carried-prime rewrite is not qualified",
        "prime_chunk_captured",
        evidence,
    )?;
    require(
        before == after && before_logits == after_logits && before_seed == after_seed,
        "UNQUALIFIED_PRIME_API_MUTATED_STATE",
    )?;
    require_prime_absent(model)?;
    evidence.event("UNQUALIFIED_PRIME_API_REFUSAL_PASS constructor=true captured_entry=true cache_and_outputs_unchanged=true positive_prime_coverage=false")
}

pub(super) fn prime_production_refusal(
    e: &Engine,
    model: &mut HybridModel,
    evidence: &mut Evidence,
) -> Result<()> {
    require_prime_absent(model)?;
    let mut cache = Cache::new(e, &model.cfg, PRIME_BUCKET + STEPS + 8)?;
    // Preserve real, nonempty cache state, not merely a vacuous empty fixture.
    model.decode_step_h(e, PROMPTS[0][0], &mut cache)?;
    let before = cache_sha256(e, &cache)?;
    evidence.bytes("production-prime-cache-before.txt", before.as_bytes())?;
    let missing = evidence.directory.join("missing-bundle");
    require(!missing.exists(), "PRODUCTION_REFUSAL_REQUIRES_UNUSED_PATH")?;
    let error = model
        .install_rewrite_bundle(&missing)
        .expect_err("missing bundle must refuse");
    evidence.event(format!("PRIME_ADMISSION_REVOKED failed_install={error}"))?;
    require(
        error.to_string().contains("read artifact.lock"),
        "PRODUCTION_REFUSAL_WRONG_INSTALL_ERROR",
    )?;
    require(
        !model.rewrite_allowed(RewriteSurface::CarriedPrime)
            && !model.rewrite_allowed(RewriteSurface::DecodeEager),
        "FAILED_INSTALL_RETAINED_PERMISSION",
    )?;
    let result = model.prime_cache_batch(e, &[PROMPTS[2]], &mut [&mut cache]);
    let after = cache_sha256(e, &cache)?;
    evidence.bytes("production-prime-cache-after.txt", after.as_bytes())?;
    refusal(
        result,
        "neither batched-prime nor eager rewrite is qualified",
        "prime_cache_batch",
        evidence,
    )?;
    require(
        before == after,
        "UNQUALIFIED_PRODUCTION_PRIME_MUTATED_CACHE",
    )?;
    evidence.event("UNQUALIFIED_PRIME_PRODUCTION_REFUSAL_PASS eager_and_prime_absent=true cache_unchanged=true")
}

fn observe_comparison(
    model: &HybridModel,
    reference: &[f32],
    candidate: &[f32],
    label: &str,
    evidence: &mut Evidence,
) -> Result<bool> {
    let identity = model.rewrite_identity()?;
    let result = manifest(model, RewriteSurface::DecodeEager)?.verify_logits(
        &identity.implementation_sha256,
        reference,
        candidate,
        policy(model)?,
    )?;
    evidence.event(format!("FALLBACK_PARITY label={label} passed={} max_abs={} max_rel={} max_ref_abs={} reference_argmax={} candidate_argmax={} first_violation={:?}",
        result.passed, result.max_abs, result.max_rel, result.max_ref_abs, result.reference_argmax, result.candidate_argmax, result.first_violation))?;
    Ok(result.passed)
}

fn prime_fallback_original(e: &Engine, model: &HybridModel, evidence: &mut Evidence) -> Result<()> {
    require_prime_absent(model)?;
    let prompt = PROMPTS[2];
    require(
        prompt.len() == 16 && prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T,
        "FALLBACK_REQUIRES_ACTUAL_T16_PRIME",
    )?;
    evidence.event(format!("FALLBACK_INPUT tokens={prompt:?} prime_permission=false reference=tokenwise ordinary=prime_cache candidate=prime_cache_batch"))?;
    let capacity = prompt.len() + STEPS + 8;
    let mut reference_cache = Cache::new(e, &model.cfg, capacity)?;
    let mut ordinary_cache = Cache::new(e, &model.cfg, capacity)?;
    let mut fallback_cache = Cache::new(e, &model.cfg, capacity)?;
    let mut last = None;
    for &token in prompt {
        last = Some(model.decode_step_h(e, token, &mut reference_cache)?);
    }
    let (mut reference, reference_seed) = last.ok_or("EMPTY_FALLBACK_REFERENCE")?;
    let (mut ordinary, ordinary_seed, ordinary_hidden) =
        model.prime_cache(e, prompt, &mut ordinary_cache, 0)?;
    // This is the actual production entry. Its absent CarriedPrime permission
    // selects the existing per-prompt ordinary-prime fallback; no substitute call.
    let mut rows = model.prime_cache_batch(e, &[prompt], &mut [&mut fallback_cache])?;
    require(rows.len() == 1, "FALLBACK_RETURNED_WRONG_ROW_COUNT")?;
    let (mut candidate, candidate_seed, candidate_hidden) = rows.remove(0);
    require(
        reference_cache.pos == prompt.len()
            && ordinary_cache.pos == prompt.len()
            && fallback_cache.pos == prompt.len(),
        "FALLBACK_POSITION_MISMATCH",
    )?;
    let n_embd = model.cfg.n_embd as usize;
    let vocab = model.cfg.n_vocab as usize;
    let reference_seed = e.dtoh(&reference_seed)?;
    let ordinary_seed = e.dtoh(&ordinary_seed)?;
    let candidate_seed = e.dtoh(&candidate_seed)?;
    evidence.logits("fallback-reference-seed", &reference_seed, n_embd)?;
    evidence.logits("fallback-ordinary-seed", &ordinary_seed, n_embd)?;
    evidence.logits("fallback-candidate-seed", &candidate_seed, n_embd)?;
    evidence.logits(
        "fallback-ordinary-hidden",
        &e.dtoh(&ordinary_hidden)?,
        prompt.len() * n_embd,
    )?;
    evidence.logits(
        "fallback-candidate-hidden",
        &e.dtoh(&candidate_hidden)?,
        prompt.len() * n_embd,
    )?;
    // Keep every measured row, including failures. Each arm follows its own greedy
    // continuation; never force the reference token into a diverging candidate.
    let mut passed = observe_comparison(
        model,
        &reference_seed,
        &ordinary_seed,
        "ordinary-seed",
        evidence,
    )?;
    passed &= observe_comparison(
        model,
        &reference_seed,
        &candidate_seed,
        "candidate-seed",
        evidence,
    )?;
    let (mut reference_tokens, mut ordinary_tokens, mut candidate_tokens) =
        (Vec::new(), Vec::new(), Vec::new());
    for step in 0..=STEPS {
        evidence.logits(&format!("fallback-reference-{step}"), &reference, vocab)?;
        evidence.logits(&format!("fallback-ordinary-{step}"), &ordinary, vocab)?;
        evidence.logits(&format!("fallback-candidate-{step}"), &candidate, vocab)?;
        passed &= observe_comparison(
            model,
            &reference,
            &ordinary,
            &format!("ordinary-{step}"),
            evidence,
        )?;
        passed &= observe_comparison(
            model,
            &reference,
            &candidate,
            &format!("candidate-{step}"),
            evidence,
        )?;
        let rt = memra_engine::forward::argmax(&reference) as u32;
        let ot = memra_engine::forward::argmax(&ordinary) as u32;
        let ct = memra_engine::forward::argmax(&candidate) as u32;
        reference_tokens.push(rt);
        ordinary_tokens.push(ot);
        candidate_tokens.push(ct);
        passed &= rt == ot && rt == ct;
        if step < STEPS {
            reference = model.decode_step(e, rt, &mut reference_cache)?;
            ordinary = model.decode_step(e, ot, &mut ordinary_cache)?;
            candidate = model.decode_step(e, ct, &mut fallback_cache)?;
        }
    }
    evidence.tokens("fallback-reference", &reference_tokens)?;
    evidence.tokens("fallback-ordinary", &ordinary_tokens)?;
    evidence.tokens("fallback-candidate", &candidate_tokens)?;
    require(
        reference_cache.pos == prompt.len() + STEPS
            && ordinary_cache.pos == reference_cache.pos
            && fallback_cache.pos == reference_cache.pos,
        "FALLBACK_CONTINUATION_POSITION",
    )?;
    require_prime_absent(model)?;
    require(
        passed,
        "NATIVE_MATH_FAILED ordinary-prime-fallback; no tolerance relaxation or receipt substitution",
    )?;
    evidence.event("EAGER_PRIME_FALLBACK_ORIGINAL_CASE_PASS true_tokens=16 continuation_steps=4 carried_prime_qualified=false receipt_emitted=false")
}

#[path = "selected/fallback_matrix.rs"]
mod fallback_matrix;

pub(super) fn prime_fallback(
    e: &Engine,
    model: &HybridModel,
    evidence: &mut Evidence,
) -> Result<()> {
    // Preserve the exact original T16 reproducer before adding boundary cases.
    prime_fallback_original(e, model, evidence)?;
    fallback_matrix::run(e, model, evidence)?;
    evidence.event("EAGER_PRIME_FALLBACK_PASS lengths=15,16,17 batches=1,3 fresh_and_carried=true continuation_steps=4 carried_prime_qualified=false receipt_emitted=false")
}
