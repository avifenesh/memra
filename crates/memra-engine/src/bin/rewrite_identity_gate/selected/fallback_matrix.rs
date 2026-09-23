//! Boundary and multi-request controls for the actual Eager-only prime fallback.
use super::*;

struct Observation {
    logits: Vec<f32>,
    seed: Vec<f32>,
    hiddens: Vec<f32>,
}

fn cache_with_prefix(
    e: &Engine,
    model: &HybridModel,
    prefix: &[u32],
    prompt_len: usize,
) -> Result<Cache> {
    let mut cache = Cache::new(e, &model.cfg, prefix.len() + prompt_len + STEPS + 8)?;
    for &token in prefix {
        model.decode_step_h(e, token, &mut cache)?;
    }
    Ok(cache)
}

fn measure(
    e: &Engine,
    model: &HybridModel,
    prompts: &[&[u32]],
    prefixes: &[&[u32]],
    evidence: &mut Evidence,
) -> Result<()> {
    require_prime_absent(model)?;
    require(
        prompts.len() == prefixes.len(),
        "FALLBACK_MATRIX_INPUT_SHAPE",
    )?;
    let n_embd = model.cfg.n_embd as usize;
    let vocab = model.cfg.n_vocab as usize;
    let mut reference_caches = Vec::new();
    let mut ordinary_caches = Vec::new();
    let mut candidate_caches = Vec::new();
    let mut references = Vec::new();
    let mut ordinary = Vec::new();
    for (stream, (&prompt, &prefix)) in prompts.iter().zip(prefixes).enumerate() {
        evidence.event(format!("FALLBACK_MATRIX_INPUT stream={stream} prompt={prompt:?} prefix={prefix:?} batch={} prime_permission=false", prompts.len()))?;
        let mut reference_cache = cache_with_prefix(e, model, prefix, prompt.len())?;
        let mut ordinary_cache = cache_with_prefix(e, model, prefix, prompt.len())?;
        let candidate_cache = cache_with_prefix(e, model, prefix, prompt.len())?;
        let mut reference = Observation {
            logits: Vec::new(),
            seed: Vec::new(),
            hiddens: Vec::new(),
        };
        for &token in prompt {
            let (logits, seed) = model.decode_step_h(e, token, &mut reference_cache)?;
            reference.logits = logits;
            reference.seed = e.dtoh(&seed)?;
            reference.hiddens.extend_from_slice(&reference.seed);
        }
        let (logits, seed, hiddens) = model.prime_cache(e, prompt, &mut ordinary_cache, 0)?;
        ordinary.push(Observation {
            logits,
            seed: e.dtoh(&seed)?,
            hiddens: e.dtoh(&hiddens)?,
        });
        references.push(reference);
        reference_caches.push(reference_cache);
        ordinary_caches.push(ordinary_cache);
        candidate_caches.push(candidate_cache);
    }
    // One real multi-request call; every returned row gets its own independent
    // tokenwise reference, seed/hidden comparison and own-greedy continuation.
    let mut refs: Vec<_> = candidate_caches.iter_mut().collect();
    let batch = model.prime_cache_batch(e, prompts, &mut refs)?;
    require(batch.len() == prompts.len(), "FALLBACK_MATRIX_RETURN_COUNT")?;
    let mut candidates = Vec::new();
    for (logits, seed, hiddens) in batch {
        candidates.push(Observation {
            logits,
            seed: e.dtoh(&seed)?,
            hiddens: e.dtoh(&hiddens)?,
        });
    }
    let mut passed = true;
    for stream in 0..prompts.len() {
        let expected_pos = prefixes[stream].len() + prompts[stream].len();
        require(
            reference_caches[stream].pos == expected_pos
                && ordinary_caches[stream].pos == expected_pos
                && candidate_caches[stream].pos == expected_pos,
            "FALLBACK_MATRIX_POSITION",
        )?;
        let r = &mut references[stream];
        let o = &mut ordinary[stream];
        let c = &mut candidates[stream];
        for (name, values, size) in [
            ("reference-seed", &r.seed, n_embd),
            ("ordinary-seed", &o.seed, n_embd),
            ("candidate-seed", &c.seed, n_embd),
            (
                "reference-hiddens",
                &r.hiddens,
                prompts[stream].len() * n_embd,
            ),
            (
                "ordinary-hiddens",
                &o.hiddens,
                prompts[stream].len() * n_embd,
            ),
            (
                "candidate-hiddens",
                &c.hiddens,
                prompts[stream].len() * n_embd,
            ),
        ] {
            evidence.logits(&format!("stream{stream}-{name}"), values, size)?;
        }
        for (name, reference, candidate) in [
            ("ordinary-seed", &r.seed, &o.seed),
            ("candidate-seed", &r.seed, &c.seed),
            ("ordinary-hiddens", &r.hiddens, &o.hiddens),
            ("candidate-hiddens", &r.hiddens, &c.hiddens),
        ] {
            passed &= observe_comparison(
                model,
                reference,
                candidate,
                &format!("stream{stream}-{name}"),
                evidence,
            )?;
        }
        let (mut rtokens, mut otokens, mut ctokens) = (Vec::new(), Vec::new(), Vec::new());
        for step in 0..=STEPS {
            for (name, row) in [
                ("reference", &r.logits),
                ("ordinary", &o.logits),
                ("candidate", &c.logits),
            ] {
                evidence.logits(&format!("stream{stream}-{name}-{step}"), row, vocab)?;
            }
            passed &= observe_comparison(
                model,
                &r.logits,
                &o.logits,
                &format!("stream{stream}-ordinary-{step}"),
                evidence,
            )?;
            passed &= observe_comparison(
                model,
                &r.logits,
                &c.logits,
                &format!("stream{stream}-candidate-{step}"),
                evidence,
            )?;
            let rt = memra_engine::forward::argmax(&r.logits) as u32;
            let ot = memra_engine::forward::argmax(&o.logits) as u32;
            let ct = memra_engine::forward::argmax(&c.logits) as u32;
            rtokens.push(rt);
            otokens.push(ot);
            ctokens.push(ct);
            passed &= rt == ot && rt == ct;
            if step < STEPS {
                r.logits = model.decode_step(e, rt, &mut reference_caches[stream])?;
                o.logits = model.decode_step(e, ot, &mut ordinary_caches[stream])?;
                c.logits = model.decode_step(e, ct, &mut candidate_caches[stream])?;
            }
        }
        for (name, tokens) in [
            ("reference", &rtokens),
            ("ordinary", &otokens),
            ("candidate", &ctokens),
        ] {
            evidence.tokens(&format!("stream{stream}-{name}"), tokens)?;
        }
        require(
            reference_caches[stream].pos == expected_pos + STEPS
                && ordinary_caches[stream].pos == expected_pos + STEPS
                && candidate_caches[stream].pos == expected_pos + STEPS,
            "FALLBACK_MATRIX_CONTINUATION_POSITION",
        )?;
    }
    require_prime_absent(model)?;
    require(
        passed,
        "NATIVE_MATH_FAILED ordinary-prime-fallback matrix; unchanged tolerance",
    )?;
    evidence.event(format!(
        "FALLBACK_MATRIX_CASE_PASS batch={} continuation_steps={STEPS}",
        prompts.len()
    ))
}

pub(super) fn run(e: &Engine, model: &HybridModel, evidence: &mut Evidence) -> Result<()> {
    let p16 = PROMPTS[2];
    let p15 = &p16[..15];
    let mut p17 = p16.to_vec();
    p17.push(5);
    let prefix = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31][..];
    type MatrixCase<'a> = (&'static str, Vec<&'a [u32]>, Vec<&'a [u32]>);
    let cases: [MatrixCase<'_>; 7] = [
        ("fresh-t15", vec![p15], vec![&[]]),
        ("fresh-t17", vec![&p17], vec![&[]]),
        ("carried-t15", vec![p15], vec![prefix]),
        ("carried-t16", vec![p16], vec![prefix]),
        ("carried-t17", vec![&p17], vec![prefix]),
        (
            "batch-15-16-17",
            vec![p15, p16, &p17],
            vec![&[], prefix, &[]],
        ),
        (
            "batch-17-16-15",
            vec![&p17, p16, p15],
            vec![&[], prefix, &[]],
        ),
    ];
    for (name, prompts, prefixes) in cases {
        let mut child = Evidence::new(evidence.directory.join(name))?;
        evidence.event(format!("FALLBACK_MATRIX_CASE_BEGIN {name}"))?;
        measure(e, model, &prompts, &prefixes, &mut child)?;
    }
    Ok(())
}
