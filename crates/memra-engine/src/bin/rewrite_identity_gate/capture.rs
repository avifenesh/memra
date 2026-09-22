//! Observations are collected independently for each surface in this executable.
//! Nothing is copied from the historical eager bundle; the index is published last.
use super::*;
use memra_engine::forward::argmax;

fn compare(
    model: &HybridModel,
    surface: RewriteSurface,
    reference: &[f32],
    candidate: &[f32],
    label: &str,
    evidence: &mut Evidence,
) -> Result<()> {
    let rewrite = manifest(model, surface)?;
    let receipt = rewrite.verify_logits(
        &model.rewrite_identity()?.implementation_sha256,
        reference,
        candidate,
        policy(model)?,
    )?;
    evidence.event(format!("PARITY label={label} surface={} passed={} max_abs={} max_rel={} reference_argmax={} candidate_argmax={} first_violation={:?}", surface.as_str(), receipt.passed, receipt.max_abs, receipt.max_rel, receipt.reference_argmax, receipt.candidate_argmax, receipt.first_violation))?;
    require(
        receipt.passed,
        format!("NATIVE_MATH_FAILED {label}; no tolerance relaxation or receipt substitution"),
    )
}

pub(super) fn eager_tokens(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    label: &str,
    evidence: &mut Evidence,
) -> Result<Vec<u32>> {
    evidence.event(format!("PROMPT {label} {prompt:?}"))?;
    let mut cache = Cache::new(e, &model.cfg, prompt.len() + GRAPH_BUDGET + 8)?;
    let mut logits = Vec::new();
    for &token in prompt {
        logits = model.decode_step(e, token, &mut cache)?;
    }
    let mut tokens = Vec::new();
    for step in 0..=STEPS {
        evidence.logits(
            &format!("{label}-row-{step}"),
            &logits,
            model.cfg.n_vocab as usize,
        )?;
        let token = argmax(&logits) as u32;
        tokens.push(token);
        if step < STEPS {
            logits = model.decode_step(e, token, &mut cache)?;
        }
    }
    require(
        cache.pos == prompt.len() + STEPS,
        "REFERENCE_TOKEN_POSITION",
    )?;
    evidence.tokens(label, &tokens)?;
    Ok(tokens)
}

fn graph_tokens(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    profile: bool,
    label: &str,
    evidence: &mut Evidence,
) -> Result<Vec<u32>> {
    let (mut session, first) = model.graph_session_new(e, prompt, GRAPH_BUDGET)?;
    let mut tokens = vec![first];
    for step in 0..STEPS {
        let next = if profile {
            session.prof_apply(e, model)?;
            session.prof_launch(model)?;
            session.prof_read(e, model)?
        } else {
            session.step(e, model)?
        };
        tokens.push(next);
        require(
            session.cache.pos == prompt.len() + step + 1,
            "CANDIDATE_GRAPH_POSITION",
        )?;
    }
    require(session.gs.captures > 0, "NO_ACTUAL_GRAPH_CAPTURE")?;
    evidence.tokens(label, &tokens)?;
    Ok(tokens)
}

/// A new reference cache, new ordinary-prime cache and new graph destination for
/// every prompt. Continuations exercise copied KV, rather than only final logits.
/// The verifier uses quantized-cache rows; forward/forward_last are never borrowed.
pub(super) fn prime_rows(
    e: &Engine,
    model: &HybridModel,
    pg: &mut PrimeGraph,
    prompt_id: usize,
    label: &str,
    evidence: &mut Evidence,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let prompt = PROMPTS[prompt_id];
    evidence.event(format!(
        "PRIME_PROMPT {label} id={prompt_id} bucket={} tokens={prompt:?}",
        pg.bucket
    ))?;
    let capacity = PRIME_BUCKET + STEPS + 8;
    let vocab = model.cfg.n_vocab as usize;
    let mut reference_cache = Cache::new(e, &model.cfg, capacity)?;
    let rows = model.decode_step_t(e, prompt, 0, &mut reference_cache)?;
    require(
        rows.len() == prompt.len() * vocab,
        "PRIME_REFERENCE_VERIFY_SHAPE",
    )?;
    // Retain the complete native teacher-forced observation, not just its last row.
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-verify-all"),
        &rows,
        prompt.len() * vocab,
    )?;
    let mut reference = rows[rows.len() - vocab..].to_vec();
    let mut ordinary_cache = Cache::new(e, &model.cfg, capacity)?;
    let (mut ordinary, ordinary_seed) = if prompt.len() < memra_engine::hybrid_forward::PRIME_MIN_T
    {
        // Production primes sub-floor prompts tokenwise. Keep the same inputs
        // and last pre-output-norm seed instead of calling prime_cache illegally.
        evidence.event(format!("PRIME_ORDINARY id={prompt_id} path=tokenwise"))?;
        let mut last = None;
        for &token in prompt {
            last = Some(model.decode_step_h(e, token, &mut ordinary_cache)?);
        }
        last.ok_or("PRIME_ORDINARY_EMPTY_PROMPT")?
    } else {
        evidence.event(format!("PRIME_ORDINARY id={prompt_id} path=prime_cache"))?;
        let (logits, seed, _) = model.prime_cache(e, prompt, &mut ordinary_cache, 0)?;
        (logits, seed)
    };
    let mut graph_cache = Cache::new(e, &model.cfg, capacity)?;
    require(graph_cache.pos == 0, "PRIME_CANDIDATE_NOT_FRESH")?;
    let (mut candidate, graph_seed) = model.prime_graph_run(e, pg, prompt, &mut graph_cache)?;
    require(
        reference_cache.pos == prompt.len()
            && ordinary_cache.pos == prompt.len()
            && graph_cache.pos == prompt.len(),
        "PRIME_POSITION_MISMATCH",
    )?;
    let ordinary_seed = e.dtoh(&ordinary_seed)?;
    let graph_seed = e.dtoh(&graph_seed)?;
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-ordinary-seed"),
        &ordinary_seed,
        model.cfg.n_embd as usize,
    )?;
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-graph-seed"),
        &graph_seed,
        model.cfg.n_embd as usize,
    )?;
    // Record main outputs before any comparison can fail, including the seed check.
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-reference-0"),
        &reference,
        vocab,
    )?;
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-ordinary-0"),
        &ordinary,
        vocab,
    )?;
    evidence.logits(
        &format!("{label}-prime-{prompt_id}-graph-0"),
        &candidate,
        vocab,
    )?;
    compare(
        model,
        RewriteSurface::CarriedPrime,
        &ordinary_seed,
        &graph_seed,
        "prime-seed",
        evidence,
    )?;
    let mut references = Vec::new();
    let mut candidates = Vec::new();
    for step in 0..=STEPS {
        if step > 0 {
            evidence.logits(
                &format!("{label}-prime-{prompt_id}-reference-{step}"),
                &reference,
                vocab,
            )?;
            evidence.logits(
                &format!("{label}-prime-{prompt_id}-ordinary-{step}"),
                &ordinary,
                vocab,
            )?;
            evidence.logits(
                &format!("{label}-prime-{prompt_id}-graph-{step}"),
                &candidate,
                vocab,
            )?;
        }
        // Both comparisons must pass on every row; a global concatenated argmax
        // cannot hide a prompt or continuation mismatch.
        compare(
            model,
            RewriteSurface::CarriedPrime,
            &reference,
            &candidate,
            &format!("prime-verify-{prompt_id}-{step}"),
            evidence,
        )?;
        compare(
            model,
            RewriteSurface::CarriedPrime,
            &ordinary,
            &candidate,
            &format!("prime-ordinary-{prompt_id}-{step}"),
            evidence,
        )?;
        let token = argmax(&reference) as u32;
        evidence.event(format!(
            "PRIME_TOKEN prompt={prompt_id} step={step} token={token}"
        ))?;
        references.extend_from_slice(&reference);
        candidates.extend_from_slice(&candidate);
        if step < STEPS {
            // Parity above requires each arm's own argmax == token; no forced input
            // can hide a divergent greedy stream or graph copy-out error.
            reference = model.decode_step(e, token, &mut reference_cache)?;
            ordinary = model.decode_step(e, token, &mut ordinary_cache)?;
            candidate = model.decode_step(e, token, &mut graph_cache)?;
        }
    }
    require(
        reference_cache.pos == prompt.len() + STEPS
            && ordinary_cache.pos == reference_cache.pos
            && graph_cache.pos == reference_cache.pos,
        "PRIME_CONTINUATION_POSITION",
    )?;
    Ok((references, candidates))
}

pub(super) fn run(
    e: &Engine,
    model: &mut HybridModel,
    bundle: &Path,
    lock: &[u8],
    identity: &RewriteIdentity,
    evidence: &mut Evidence,
    scope: Scope,
) -> Result<()> {
    require(
        !model.rewrite_is_qualified(),
        "RETAINED_CAPTURE_ALREADY_QUALIFIED",
    )?;
    let eager = manifest(model, RewriteSurface::DecodeEager)?;
    let graph = manifest(model, RewriteSurface::DecodeGraph)?;
    let implementation = &identity.implementation_sha256;
    let mut receipts = Vec::new();

    let (mut eager_ref, mut eager_candidate) = (Vec::new(), Vec::new());
    // Keep the original 4/8/16 rows and straddle the prefill crossover as well.
    // The decode-exact fallback must preserve the T=1 program on both sides.
    let boundaries = [15, 17].map(|rows| {
        PROMPTS[2]
            .iter()
            .copied()
            .cycle()
            .take(rows)
            .collect::<Vec<_>>()
    });
    for (i, prompt) in PROMPTS
        .into_iter()
        .chain(boundaries.iter().map(Vec::as_slice))
        .enumerate()
    {
        evidence.event(format!("EAGER_INPUT case=eager-{i} tokens={prompt:?}"))?;
        let reference = verify_prefill(e, model, prompt)?;
        let candidate = tokenwise(e, model, prompt)?;
        evidence.logits(
            &format!("eager-reference-{i}"),
            &reference,
            model.cfg.n_vocab as usize,
        )?;
        evidence.logits(
            &format!("eager-candidate-{i}"),
            &candidate,
            model.cfg.n_vocab as usize,
        )?;
        compare(
            model,
            RewriteSurface::DecodeEager,
            &reference,
            &candidate,
            &format!("eager-{i}"),
            evidence,
        )?;
        eager_ref.extend(reference);
        eager_candidate.extend(candidate);
    }
    receipts.push(eager.verify_logits(
        implementation,
        &eager_ref,
        &eager_candidate,
        policy(model)?,
    )?);

    let (mut graph_ref, mut graph_candidate) = (Vec::new(), Vec::new());
    let update = update_prompt();
    for (i, prompt) in PROMPTS
        .into_iter()
        .chain(std::iter::once(update.as_slice()))
        .enumerate()
    {
        // Each graph observation has a fresh eager run; do not relabel an eager
        // output or duplicate a candidate stream to construct the receipt.
        for profile in [false, true] {
            let label = if profile { "profile" } else { "step" };
            let reference = eager_tokens(
                e,
                model,
                prompt,
                &format!("graph-{label}-reference-{i}"),
                evidence,
            )?;
            let candidate = graph_tokens(
                e,
                model,
                prompt,
                profile,
                &format!("graph-{label}-candidate-{i}"),
                evidence,
            )?;
            let row = graph.verify_tokens(implementation, &reference, &candidate)?;
            evidence.event(format!(
                "GRAPH_PARITY prompt={i} path={label} passed={} first_violation={:?}",
                row.passed, row.first_violation
            ))?;
            require(
                row.passed,
                format!("GRAPH_NATIVE_MATH_FAILED prompt={i} path={label}"),
            )?;
            graph_ref.extend(reference);
            graph_candidate.extend(candidate);
        }
    }
    receipts.push(graph.verify_tokens(implementation, &graph_ref, &graph_candidate)?);

    if scope == Scope::Full {
        let prime = manifest(model, RewriteSurface::CarriedPrime)?;
        let mut pg = model.prime_graph_new(e, PRIME_BUCKET)?;
        let (mut prime_ref, mut prime_candidate) = (Vec::new(), Vec::new());
        for i in 0..PROMPTS.len() {
            let (reference, candidate) = prime_rows(e, model, &mut pg, i, "capture", evidence)?;
            prime_ref.extend(reference);
            prime_candidate.extend(candidate);
        }
        receipts.push(prime.verify_logits(
            implementation,
            &prime_ref,
            &prime_candidate,
            policy(model)?,
        )?);
    } else {
        evidence.event("UNQUALIFIED_SURFACE carried-prime receipt_emitted=false positive_prime_coverage=false issue=585")?;
    }
    // Do not bind measured output to a later identity. CUDA lazy-loading drift also
    // fails this check; the runner must warm dependencies before load, never rebase it.
    require(
        model.rewrite_identity()? == identity,
        "IDENTITY_CHANGED_DURING_CAPTURE",
    )?;
    require(
        std::fs::read(bundle.join("artifact.lock"))? == lock,
        "LOCK_CHANGED_DURING_CAPTURE",
    )?;

    let mut index = INDEX_HEADER.to_string();
    let mut bound = Vec::new();
    for receipt in receipts {
        let rewrite = manifest(model, receipt.surface)?;
        receipt.validate_for(&rewrite)?;
        let receipt = bind_rewrite_artifact(model, receipt)?;
        require(
            receipt.artifact_lock_sha256.as_deref() == Some(sha256(lock).as_str()),
            "RECEIPT_LOCK_MISMATCH",
        )?;
        let text = receipt.to_tsv();
        parse_qualified_rewrite_receipt(&text)?;
        index.push_str(&format!(
            "{}\t{}\t{}\tpassed\n",
            rewrite.id,
            rewrite.plan_sha256,
            sha256(text.as_bytes())
        ));
        bound.push((rewrite.id, text));
    }
    std::fs::create_dir(bundle.join("rewrite-receipts"))?;
    for (id, text) in bound {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(bundle.join("rewrite-receipts").join(format!("{id}.tsv")))?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
    }
    let index_path = bundle.join("rewrite-receipts.tsv");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&index_path)?;
    file.write_all(index.as_bytes())?;
    file.sync_all()?;
    drop(file);
    let installed = (|| -> Result<()> {
        model.install_rewrite_bundle(bundle)?;
        qualified(model, scope)?;
        evidence.bytes(
            "PASS",
            seal(identity, index.as_bytes(), lock, scope).as_bytes(),
        )?;
        evidence.event(format!(
            "{} surfaces={:?} support_promotion=false performance=not-measured",
            scope.capture_marker(),
            scope.surfaces()
        ))
    })();
    if installed.is_err() {
        // Keep failed evidence, but never leave an installable index after failure.
        std::fs::rename(&index_path, evidence.directory.join("failed-index.tsv"))?;
    }
    installed
}
