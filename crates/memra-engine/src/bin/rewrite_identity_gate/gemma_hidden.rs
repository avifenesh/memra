//! Admitted non-PLE Gemma prime exports and the worker's real pooling consumer.
//! Each HPOST setting runs in a fresh process with its own Eager-only bundle.
//! Expected rows come from independent T1 caches and scalar CPU RMS math, never
//! from prime output or hidden_postnorm_row. No model support is promoted here.
use super::*;
use cudarc::driver::CudaSlice;
use memra_gguf::model_plan::OperationKind;

const ATOL: f32 = 0.005;
const WIDTHS: [usize; 3] = [15, 16, 17];
const CONTINUATIONS: usize = 4;

struct Outputs(std::path::PathBuf, memra_engine::plan_backend::PlanLogits);

impl Outputs {
    fn token_ids(&self, name: &str, values: &[u32]) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.0.join(format!("{name}.u32")))?;
        for token in values {
            file.write_all(&token.to_le_bytes())?;
        }
        file.sync_all()?;
        Ok(())
    }

    fn floats(&self, name: &str, values: &[f32], size: usize) -> Result<()> {
        self.raw_floats(name, values, size)?;
        require(
            values.iter().all(|x| x.is_finite()),
            format!("INVALID_OUTPUT {name}"),
        )
    }

    fn logits(&self, name: &str, values: &[f32], size: usize) -> Result<()> {
        self.raw_floats(name, values, size)?;
        self.1.validate(values)?;
        Ok(())
    }

    fn close_logits(&self, expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
        let error = self.1.max_abs_difference(expected, actual)?;
        println!("GEMMA_PARITY label={label} max_abs={error} atol={ATOL}");
        require(
            error <= ATOL,
            format!("GEMMA_PARITY_FAILED {label} max_abs={error} atol={ATOL}"),
        )
    }

    fn raw_floats(&self, name: &str, values: &[f32], size: usize) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.0.join(format!("{name}.f32")))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        println!(
            "GEMMA_RAW name={name} values={} sha256={}",
            values.len(),
            sha256(&bytes)
        );
        require(values.len() == size, format!("INVALID_OUTPUT {name}"))
    }
}

fn normalized_reference(raw: &[f32], weights: &[f32], eps: f32) -> Result<Vec<f32>> {
    require(
        !raw.is_empty() && raw.len() == weights.len(),
        "REFERENCE_NORM_SHAPE",
    )?;
    require(
        eps.is_finite() && eps > 0.0 && raw.iter().chain(weights).all(|x| x.is_finite()),
        "REFERENCE_NORM_INPUT",
    )?;
    // Independent scalar evaluation; no engine norm, prime export or pooling read.
    let inv = (raw.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / raw.len() as f64
        + f64::from(eps))
    .sqrt()
    .recip();
    Ok(raw
        .iter()
        .zip(weights)
        .map(|(&x, &w)| (f64::from(x) * inv * f64::from(w)) as f32)
        .collect())
}

fn close(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    require(
        expected.len() == actual.len() && !expected.is_empty(),
        format!("PARITY_SHAPE {label}"),
    )?;
    require(
        expected.iter().chain(actual).all(|x| x.is_finite()),
        format!("PARITY_NONFINITE {label}"),
    )?;
    let error = expected
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("GEMMA_PARITY label={label} max_abs={error} atol={ATOL}");
    require(
        error <= ATOL,
        format!("GEMMA_PARITY_FAILED {label} max_abs={error} atol={ATOL}"),
    )
}

fn tokens(n: usize, offset: usize) -> Vec<u32> {
    (0..n)
        .map(|i| PROMPTS[2][(i + offset) % PROMPTS[2].len()])
        .collect()
}

/// Establish F32 verify-row parity before any HPOST export/pooling comparison.
/// Persist every independent T1 row before running the candidate and retain both
/// full vectors on failure. These checks grant no speculative/prime admission.
fn f32_row_parity(e: &Engine, m: &HybridModel, out: &Outputs) -> Result<()> {
    let vocab = m.cfg.n_vocab as usize;
    for prefix in [0, 11] {
        for rows in WIDTHS {
            let label = format!("f32-p{prefix}-t{rows}");
            let mut reference = Cache::new(e, &m.cfg, 96)?;
            let mut candidate = Cache::new(e, &m.cfg, 96)?;
            for token in tokens(prefix, 7) {
                m.decode_step(e, token, &mut reference)?;
                m.decode_step(e, token, &mut candidate)?;
            }
            let prompt = tokens(rows, 0);
            let mut expected = Vec::new();
            for &token in &prompt {
                expected.extend(m.decode_step(e, token, &mut reference)?);
            }
            out.logits(&format!("{label}-t1-rows"), &expected, rows * vocab)?;
            let actual = m.decode_step_t(e, &prompt, prefix, &mut candidate)?;
            out.logits(&format!("{label}-verify-rows"), &actual, rows * vocab)?;
            for row in 0..rows {
                out.close_logits(
                    &expected[row * vocab..(row + 1) * vocab],
                    &actual[row * vocab..(row + 1) * vocab],
                    &format!("{label}-row{row}"),
                )?;
            }
            let mut expected = expected[(rows - 1) * vocab..].to_vec();
            let mut actual = actual[(rows - 1) * vocab..].to_vec();
            for step in 1..=CONTINUATIONS {
                let token = memra_engine::forward::argmax(&expected) as u32;
                require(
                    token == memra_engine::forward::argmax(&actual) as u32,
                    format!("F32_CONTINUATION_ARGMAX {label}-{step}"),
                )?;
                expected = m.decode_step(e, token, &mut reference)?;
                actual = m.decode_step(e, token, &mut candidate)?;
                out.logits(&format!("{label}-t1-cont{step}"), &expected, vocab)?;
                out.logits(&format!("{label}-verify-cont{step}"), &actual, vocab)?;
                out.close_logits(&expected, &actual, &format!("{label}-cont{step}"))?;
            }
            require(
                reference.pos == prefix + rows + CONTINUATIONS && candidate.pos == reference.pos,
                format!("F32_CACHE_POSITION {label}"),
            )?;
            println!("GEMMA_F32_CASE_PASS case={label}");
        }
    }
    // Actual public generators must also retain the host F32 Eager program;
    // their default greedy fast path otherwise chooses the q8 device-counter twin.
    for rows in WIDTHS {
        let prompt = tokens(rows, 0);
        let mut cache = Cache::new(e, &m.cfg, 96)?;
        let mut logits = Vec::new();
        for &token in &prompt {
            logits = m.decode_step(e, token, &mut cache)?;
        }
        let mut expected = Vec::new();
        for _ in 0..CONTINUATIONS {
            let token = memra_engine::forward::argmax(&logits) as u32;
            expected.push(token);
            logits = m.decode_step(e, token, &mut cache)?;
        }
        out.token_ids(&format!("f32-generate-t{rows}-t1"), &expected)?;
        let generated = m.generate(e, &prompt, CONTINUATIONS)?;
        out.token_ids(&format!("f32-generate-t{rows}-actual"), &generated)?;
        require(
            generated == expected,
            format!("F32_GENERATE_FAILED t{rows}"),
        )?;
        println!("GEMMA_F32_GENERATE_CASE_PASS case=generate-t{rows}");
        let params = memra_engine::decode::GenParams {
            max_new: CONTINUATIONS,
            ..Default::default()
        };
        let mut sampler = memra_engine::sampler::Sampler::new(Default::default());
        let mut streamed = Vec::new();
        let generated = m.generate_with(e, &prompt, &params, &mut sampler, |token| {
            streamed.push(token);
            true
        })?;
        out.token_ids(
            &format!("f32-generate-with-t{rows}-actual"),
            &generated.tokens,
        )?;
        out.token_ids(&format!("f32-generate-with-t{rows}-streamed"), &streamed)?;
        require(
            generated.tokens == expected && streamed == expected,
            format!("F32_GENERATE_WITH_FAILED t{rows}"),
        )?;
        println!("GEMMA_F32_GENERATE_CASE_PASS case=generate-with-t{rows}");
    }
    println!("GEMMA_F32_PARITY_PASS streams=6 atol={ATOL}");
    Ok(())
}

struct Reference {
    cache: Cache,
    raw: Vec<f32>,
    normalized: Vec<f32>,
    logits: Vec<f32>,
}

fn persist_reference(
    out: &Outputs,
    label: &str,
    reference: &Reference,
    hpost: bool,
    rows: usize,
    width: usize,
    vocab: usize,
) -> Result<()> {
    let expected = if hpost {
        &reference.normalized
    } else {
        &reference.raw
    };
    out.floats(&format!("{label}-t1-raw"), &reference.raw, rows * width)?;
    out.floats(
        &format!("{label}-expected-normalized"),
        &reference.normalized,
        rows * width,
    )?;
    out.floats(&format!("{label}-expected-export"), expected, rows * width)?;
    out.floats(
        &format!("{label}-expected-seed"),
        &expected[(rows - 1) * width..],
        width,
    )?;
    out.logits(&format!("{label}-t1-logits-0"), &reference.logits, vocab)
}

fn reference(
    e: &Engine,
    m: &HybridModel,
    prompt: &[u32],
    prefix: usize,
    weights: &[f32],
) -> Result<Reference> {
    let mut cache = Cache::new(e, &m.cfg, 96)?;
    for token in tokens(prefix, 7) {
        m.decode_step(e, token, &mut cache)?;
    }
    let mut result = Reference {
        cache,
        raw: Vec::new(),
        normalized: Vec::new(),
        logits: Vec::new(),
    };
    for &token in prompt {
        let (logits, hidden) = m.decode_step_h(e, token, &mut result.cache)?;
        let raw = e.dtoh(&hidden)?;
        require(raw.len() == weights.len(), "T1_HIDDEN_SHAPE")?;
        result
            .normalized
            .extend(normalized_reference(&raw, weights, m.cfg.rms_eps)?);
        result.raw.extend(raw);
        result.logits = logits;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)] // allow: all observations belong to this one native comparison
fn compare(
    e: &Engine,
    m: &HybridModel,
    out: &Outputs,
    label: &str,
    mut reference: Reference,
    cache: &mut Cache,
    prime: (Vec<f32>, CudaSlice<f32>, CudaSlice<f32>),
    hpost: bool,
    rows: usize,
) -> Result<()> {
    let width = m.cfg.n_embd as usize;
    let vocab = m.cfg.n_vocab as usize;
    let (mut actual_logits, seed, hidden) = prime;
    let seed = e.dtoh(&seed)?;
    let actual = e.dtoh(&hidden)?;
    let expected = if hpost {
        &reference.normalized
    } else {
        &reference.raw
    };
    let expected_seed = &expected[(rows - 1) * width..];
    // References were persisted before prime. Preserve its seed logits before any
    // hidden/pooling verdict, so an export failure cannot discard a numeric observation.
    out.logits(&format!("{label}-actual-logits-0"), &actual_logits, vocab)?;
    out.floats(&format!("{label}-actual-export"), &actual, rows * width)?;
    out.floats(&format!("{label}-actual-seed"), &seed, width)?;
    let mut pooled = Vec::new();
    for row in 0..rows {
        // The actual consumer called by worker PromptCapture, including its HPOST arm.
        pooled.extend(m.hidden_postnorm_row(e, &hidden, row)?);
    }
    out.floats(&format!("{label}-actual-pooling"), &pooled, rows * width)?;
    close(expected, &actual, &format!("{label}-export"))?;
    close(expected_seed, &seed, &format!("{label}-seed"))?;
    close(&reference.normalized, &pooled, &format!("{label}-pooling"))?;
    if !hpost {
        bitwise(&reference.raw, &actual, &format!("{label}-raw-export"))?;
    }
    // Make the normalization control non-vacuous: leaving the raw rows unchanged
    // must fail this same fixed predicate for this selected artifact and prompt.
    require(
        reference
            .raw
            .iter()
            .zip(&reference.normalized)
            .any(|(a, b)| (a - b).abs() > ATOL),
        "VACUOUS_NORM_CONTROL",
    )?;
    for step in 0..=CONTINUATIONS {
        if step > 0 {
            out.logits(
                &format!("{label}-t1-logits-{step}"),
                &reference.logits,
                vocab,
            )?;
            out.logits(
                &format!("{label}-actual-logits-{step}"),
                &actual_logits,
                vocab,
            )?;
        }
        require(
            cache.pos == reference.cache.pos,
            format!("CACHE_POSITION {label}-{step}"),
        )?;
        // This is the same qualified Eager numerical program, so logits must retain bits.
        bitwise(
            &reference.logits,
            &actual_logits,
            &format!("{label}-logits-{step}"),
        )?;
        if step < CONTINUATIONS {
            let expected_token = memra_engine::forward::argmax(&reference.logits) as u32;
            let actual_token = memra_engine::forward::argmax(&actual_logits) as u32;
            println!(
                "GEMMA_CONTINUATION case={label} step={step} reference={expected_token} actual={actual_token}"
            );
            require(expected_token == actual_token, "CONTINUATION_ARGMAX")?;
            reference.logits = m.decode_step(e, expected_token, &mut reference.cache)?;
            actual_logits = m.decode_step(e, actual_token, cache)?;
        }
    }
    println!(
        "GEMMA_CASE_PASS case={label} rows={rows} hpost={hpost} pooling_consumer=hidden_postnorm_row"
    );
    Ok(())
}

pub(super) fn run(source: &Path, bundle: &Path) -> Result<()> {
    let hpost = match std::env::var("MEMRA_SPEC_HPOST").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => return Err("GEMMA_HPOST_REQUIRES_EXPLICIT_0_OR_1_AT_EXEC".into()),
    };
    require(
        std::env::var("MEMRA_FAST").as_deref() == Ok("0"),
        "GEMMA_HPOST_REQUIRES_FAST_0",
    )?;
    require(
        source.is_file() && source.extension().is_some_and(|x| x == "gguf"),
        "GEMMA_HPOST_REQUIRES_REAL_GGUF",
    )?;
    let configured =
        std::env::var_os("MEMRA_REWRITE_BUNDLE").ok_or("GEMMA_HPOST_REQUIRES_ADMITTED_BUNDLE")?;
    require(
        std::fs::canonicalize(configured)? == std::fs::canonicalize(bundle)?,
        "BUNDLE_PATH_MISMATCH",
    )?;
    let lock_path = std::env::var_os("MEMRA_ARTIFACT_LOCK").ok_or("ARTIFACT_LOCK_REQUIRED")?;
    let lock = std::fs::read(lock_path)?;
    require(
        !lock.is_empty() && lock == std::fs::read(bundle.join("artifact.lock"))?,
        "ARTIFACT_LOCK_MISMATCH",
    )?;
    let output_dir = bundle.join("gemma-hpost");
    std::fs::create_dir(&output_dir)?; // A failure is immutable; no same-namespace retries.
    let e = Engine::new(0)?;
    let file = GgufFile::open(source)?;
    let m = HybridModel::load(&e, &file)?;
    let out = Outputs(
        output_dir,
        memra_engine::plan_backend::PlanLogits::from_plan(&m.plan)?,
    );
    require(
        m.devices() == [0]
            && m.uses_gemma_program()
            && !m.has_plan_operation(OperationKind::PleNgramEmbedding),
        "REQUIRES_SINGLE_CARD_NON_PLE_GEMMA_PLAN",
    )?;
    qualified_eager_only(&m)?;
    let _execution = m.protect_rewrite_execution()?;
    let identity = m.rewrite_identity()?;
    println!(
        "GEMMA_SCOPE hpost={hpost} artifact={} implementation={} numeric={} support_promotion=false",
        identity.artifact_sha256, identity.implementation_sha256, identity.numeric_program_sha256
    );
    require(
        PROMPTS[2].iter().all(|&x| x < m.cfg.n_vocab),
        "PROMPT_TOKEN_OUT_OF_VOCABULARY",
    )?;
    f32_row_parity(&e, &m, &out)?;
    let weights = e.dtoh(m.output_norm.float_data())?;
    out.floats("output-norm-weights", &weights, m.cfg.n_embd as usize)?;
    println!(
        "GEMMA_REFERENCE rms_epsilon={} tolerance={ATOL} math=scalar-f64-rms-to-f32",
        m.cfg.rms_eps
    );
    for prefix in [0, 11] {
        for rows in WIDTHS {
            let prompt = tokens(rows, 0);
            let reference = reference(&e, &m, &prompt, prefix, &weights)?;
            let label = format!("direct-p{prefix}-t{rows}");
            persist_reference(
                &out,
                &label,
                &reference,
                hpost,
                rows,
                m.cfg.n_embd as usize,
                m.cfg.n_vocab as usize,
            )?;
            let mut cache = Cache::new(&e, &m.cfg, 96)?;
            for token in tokens(prefix, 7) {
                m.decode_step(&e, token, &mut cache)?;
            }
            let prime = m.prime_cache(&e, &prompt, &mut cache, 0)?;
            compare(
                &e, &m, &out, &label, reference, &mut cache, prime, hpost, rows,
            )?;
        }
    }
    for order in [WIDTHS, [17, 16, 15]] {
        let prompts: Vec<_> = order
            .iter()
            .enumerate()
            .map(|(i, &n)| tokens(n, i))
            .collect();
        let mut references = Vec::new();
        let mut caches = Vec::new();
        for (i, prompt) in prompts.iter().enumerate() {
            let prefix = if i == 1 { 11 } else { 0 };
            let reference = reference(&e, &m, prompt, prefix, &weights)?;
            persist_reference(
                &out,
                &format!("batch-{}-row{i}", order[0]),
                &reference,
                hpost,
                prompt.len(),
                m.cfg.n_embd as usize,
                m.cfg.n_vocab as usize,
            )?;
            references.push(reference);
            let mut cache = Cache::new(&e, &m.cfg, 96)?;
            for token in tokens(prefix, 7) {
                m.decode_step(&e, token, &mut cache)?;
            }
            caches.push(cache);
        }
        let prime = m.prime_cache_batch(
            &e,
            &prompts.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut caches.iter_mut().collect::<Vec<_>>(),
        )?;
        require(prime.len() == 3, "BATCH_OUTPUT_SHAPE")?;
        for (i, ((reference, cache), prime)) in references
            .into_iter()
            .zip(&mut caches)
            .zip(prime)
            .enumerate()
        {
            compare(
                &e,
                &m,
                &out,
                &format!("batch-{}-row{i}", order[0]),
                reference,
                cache,
                prime,
                hpost,
                order[i],
            )?;
        }
    }
    m.rewrite_identity()?;
    qualified_eager_only(&m)?;
    require(
        lock == std::fs::read(bundle.join("artifact.lock"))?,
        "ARTIFACT_LOCK_CHANGED",
    )?;
    println!("GEMMA_HPOST_PASS hpost={hpost} streams=12 support_promotion=false");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_norm_uses_weights_and_epsilon() {
        let row = normalized_reference(&[3.0, 4.0], &[2.0, 0.5], 0.5).unwrap();
        let scale = 13.0_f64.sqrt().recip();
        assert_eq!(row, vec![(6.0 * scale) as f32, (2.0 * scale) as f32]);
        assert!(normalized_reference(&[], &[], 0.5).is_err());
        assert!(normalized_reference(&[f32::NAN], &[1.0], 0.5).is_err());
    }
    #[test]
    fn fixed_boundary_rejects_missing_and_double_normalization() {
        assert!(close(&[0.0], &[ATOL], "boundary").is_ok());
        assert!(close(&[0.0], &[f32::from_bits(ATOL.to_bits() + 1)], "outside").is_err());
        assert!(close(&[0.0], &[f32::NAN], "nan").is_err());
        let raw = [3.0, 4.0];
        let weights = [2.0, 0.5];
        let once = normalized_reference(&raw, &weights, 0.5).unwrap();
        let twice = normalized_reference(&once, &weights, 0.5).unwrap();
        assert!(close(&once, &raw, "missing").is_err());
        assert!(close(&once, &twice, "double").is_err());
    }
}
