//! Scoped native program/admission evidence on real checkpoint bytes; no support promotion.
//! Auto-discovered Cargo target and executable: `rewrite_identity_gate`.
//! Usage: rewrite_identity_gate capture|check <model-path> <bundle-dir>
//! Inspect first: artifact.lock must already exist in the bundle, and the caller must set
//! MEMRA_ARTIFACT_LOCK before either load. Capture requires MEMRA_REWRITE_BUNDLE unset;
//! check requires it set to the bundle. Keep the executable and numeric environment identical.

use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::hybrid::HybridModel;
use memra_engine::plan_backend::{
    ExecutionRewrite, RewriteParityPolicy, RewriteSurface, bind_rewrite_artifact,
    execution_rewrites, parse_qualified_rewrite_receipt,
};
use memra_gguf::GgufFile;
use memra_gguf::source::SafetensorsSource;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Fixed token-ID prompts exercise short, varied and repeated histories. These are numerical
// program probes, not tokenizer/template or model-quality qualification. Never remap OOV ids.
const PROMPTS: [&[u32]; 3] = [
    &[1, 2, 3, 4],
    &[7, 11, 19, 5, 3, 17, 2, 13],
    &[2, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 2, 3, 2, 3],
];
const INDEX_HEADER: &str = "rewrite\tplan_sha256\treceipt_sha256\tstatus\n";

fn require(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn logits_sha256(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn output(label: &str, prompt: usize, logits: &[f32], vocab: usize, bundle: &Path) -> Result<()> {
    require(
        logits.len() == vocab && !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
        format!(
            "OUTPUT_INVALID stage={label} prompt={prompt} values={} vocab={vocab}",
            logits.len()
        ),
    )?;
    let directory = bundle.join("outputs");
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!(
        "{label}-{}-prompt-{prompt}.f32",
        std::process::id()
    ));
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?
        .write_all(&bytes)?;
    println!(
        "OUTPUT stage={label} prompt={prompt} tokens={:?} values={} argmax={} file={} sha256={}",
        PROMPTS[prompt],
        logits.len(),
        memra_engine::forward::argmax(logits),
        path.display(),
        logits_sha256(logits)
    );
    Ok(())
}

fn tokenwise(engine: &Engine, model: &HybridModel, prompt: &[u32]) -> Result<Vec<f32>> {
    let mut cache = Cache::new(engine, &model.cfg, prompt.len() + 8)?;
    let mut logits = Vec::new();
    for &token in prompt {
        logits = model.decode_step(engine, token, &mut cache)?;
    }
    require(cache.pos == prompt.len(), "TOKENWISE_CACHE_POSITION")?;
    Ok(logits)
}

// Independent native reference: teacher-forced verify rows attend over the same
// quantized-cache class as tokenwise eager decode. forward_last's fresh F32 KV is a
// different program and must never be borrowed as this receipt's reference.
fn verify_prefill(engine: &Engine, model: &HybridModel, prompt: &[u32]) -> Result<Vec<f32>> {
    let mut cache = Cache::new(engine, &model.cfg, prompt.len() + 8)?;
    let logits = model.decode_step_t(engine, prompt, 0, &mut cache)?;
    let vocab = model.cfg.n_vocab as usize;
    require(logits.len() == prompt.len() * vocab, "VERIFY_PREFILL_SHAPE")?;
    require(cache.pos == prompt.len(), "VERIFY_PREFILL_CACHE_POSITION")?;
    Ok(logits[logits.len() - vocab..].to_vec())
}

fn fresh_kv_refusal(engine: &Engine, model: &HybridModel) -> Result<()> {
    for (name, result) in [
        ("forward", model.forward(engine, PROMPTS[0])),
        ("forward_last", model.forward_last(engine, PROMPTS[0])),
    ] {
        match result {
            Ok(_) => {
                return Err(
                    format!("FRESH_KV_REFUSAL_FAILED: eager receipt admitted {name}").into(),
                );
            }
            Err(error) => require(
                error
                    .to_string()
                    .contains("forward-fresh-kv rewrite is not qualified"),
                format!("FRESH_KV_REFUSAL_WRONG_REASON {name}: {error}"),
            )?,
        }
    }
    println!(
        "FRESH_KV_REFUSAL_PASS: quantized-cache eager receipt cannot authorize forward/forward_last"
    );
    Ok(())
}

fn qualified_eager_only(model: &HybridModel) -> Result<()> {
    require(model.rewrite_is_qualified(), "EAGER_NOT_QUALIFIED")?;
    for rewrite in execution_rewrites(&model.plan) {
        require(
            model.rewrite_allowed(rewrite.surface)
                == (rewrite.surface == RewriteSurface::DecodeEager),
            format!(
                "EAGER_ONLY_ADMISSION_MISMATCH surface={}",
                rewrite.surface.as_str()
            ),
        )?;
    }
    Ok(())
}

fn graph_refusal(engine: &Engine, model: &HybridModel) -> Result<()> {
    match model.graph_session_new(engine, PROMPTS[0], 2) {
        Ok(_) => Err("GRAPH_REFUSAL_FAILED: eager-only bundle admitted graph_session_new".into()),
        Err(error) => {
            let error = error.to_string();
            require(
                error.contains("decode-graph rewrite is not qualified"),
                format!("GRAPH_REFUSAL_WRONG_REASON: {error}"),
            )?;
            println!("GRAPH_REFUSAL_PASS: {error}");
            Ok(())
        }
    }
}

fn bitwise(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    require(
        expected.len() == actual.len(),
        format!("BITWISE_LENGTH stage={label}"),
    )?;
    if let Some(index) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!(
            "BITWISE_MISMATCH stage={label} index={index} expected={:08x} actual={:08x}",
            expected[index].to_bits(),
            actual[index].to_bits()
        )
        .into());
    }
    Ok(())
}

fn write_receipt(bundle: &Path, rewrite: &ExecutionRewrite, receipt: &str) -> Result<()> {
    std::fs::create_dir_all(bundle.join("rewrite-receipts"))?;
    std::fs::write(
        bundle
            .join("rewrite-receipts")
            .join(format!("{}.tsv", rewrite.id)),
        receipt,
    )?;
    let index = format!(
        "{INDEX_HEADER}{}\t{}\t{}\tpassed\n",
        rewrite.id,
        rewrite.plan_sha256,
        sha256(receipt.as_bytes())
    );
    std::fs::write(bundle.join("rewrite-receipts.tsv"), &index)?;
    println!(
        "BUNDLE path={} receipt_sha256={} index_sha256={}",
        bundle.display(),
        sha256(receipt.as_bytes()),
        sha256(index.as_bytes())
    );
    Ok(())
}

// CacheSnapshot omits KV bytes and latent state. Hash the actual short probe's full planes,
// including unused capacity and recurrent ping-pong buffers, so a write without a len advance
// cannot pass. This harness uses one device and refuses unobserved distributed/opaque state.
fn cache_sha256(engine: &Engine, cache: &Cache) -> Result<String> {
    require(
        cache.tp_kv.iter().all(Option::is_none)
            && cache.glm5_tp_recur.iter().all(Option::is_none)
            && cache.glm5_tp_latent_peer.iter().all(Option::is_none)
            && cache.qwen_prime_graph.is_none()
            && cache.glm5_decode_graph.is_none()
            && cache.glm5_tp_sym_graph.is_none()
            && cache.dflash_taps.is_none()
            && cache.hc_taps.is_none(),
        "CACHE_PROBE_UNSUPPORTED_STATE",
    )?;
    let mut hash = Sha256::new();
    hash.update(format!("{} {} {}", cache.pos, cache.max_ctx, cache.tainted));
    for layer in &cache.kv {
        hash.update([u8::from(layer.is_some())]);
        if let Some(kv) = layer {
            hash.update(format!("{} {:?}", kv.len, kv.ring));
            hash.update(engine.dtoh_u8(&kv.k)?);
            hash.update(engine.dtoh_u8(&kv.v)?);
            hash.update(format!("{:?}", engine.dtoh_i32(&kv.len_d)?));
            hash.update([u8::from(kv.base_d.is_some())]);
            if let Some(base) = &kv.base_d {
                hash.update(format!("{:?}", engine.dtoh_i32(base)?));
            }
        }
    }
    for layer in &cache.recur {
        hash.update([u8::from(layer.is_some())]);
        if let Some(recur) = layer {
            for plane in [&recur.conv_state, &recur.ssm_state, &recur.ssm_state_alt] {
                hash.update(logits_sha256(&engine.dtoh(plane)?));
            }
        }
    }
    for layer in &cache.latent {
        hash.update([u8::from(layer.is_some())]);
        if let Some(latent) = layer {
            hash.update(format!(
                "{} {} {} {:?}",
                latent.len, latent.index_pools_ready, latent.index_pool, latent.index_ring_rows
            ));
            hash.update(format!("{:?}", engine.dtoh_i32(&latent.len_d)?));
            for plane in [
                Some(&latent.rows),
                latent.index_rows.as_ref(),
                latent.index_pool_keys.as_ref(),
            ] {
                hash.update([u8::from(plane.is_some())]);
                if let Some(plane) = plane {
                    hash.update(logits_sha256(&engine.dtoh(plane)?));
                }
            }
        }
    }
    hash.update([u8::from(cache.last_logits_dev.is_some())]);
    if let Some(logits) = &cache.last_logits_dev {
        hash.update(logits_sha256(&engine.dtoh(logits)?));
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn reinstall_probe(
    engine: &Engine,
    model: &mut HybridModel,
    bundle: &Path,
    rewrite: &ExecutionRewrite,
    receipt: &str,
) -> Result<()> {
    // Preserve the valid bundle and retain the deliberately invalid control as raw evidence.
    let corrupt = bundle.join("failed-reinstall-identity");
    std::fs::create_dir(&corrupt)?;
    std::fs::copy(bundle.join("artifact.lock"), corrupt.join("artifact.lock"))?;
    let identity = model.rewrite_identity()?;
    let mut wrong_hash = identity.numeric_program_sha256.clone();
    wrong_hash.replace_range(
        ..1,
        if wrong_hash.starts_with('0') {
            "1"
        } else {
            "0"
        },
    );
    let corrupted = receipt.replace(
        &format!(
            "numeric_program_sha256\t{}\n",
            identity.numeric_program_sha256
        ),
        &format!("numeric_program_sha256\t{wrong_hash}\n"),
    );
    require(
        corrupted != receipt,
        "REINSTALL_CONTROL_DID_NOT_CHANGE_IDENTITY",
    )?;
    parse_qualified_rewrite_receipt(&corrupted)?;
    write_receipt(&corrupt, rewrite, &corrupted)?; // Rehash index: the identity check must reject it.

    let mut cache = Cache::new(engine, &model.cfg, PROMPTS[0].len() + 8)?;
    for &token in PROMPTS[0] {
        model.decode_step(engine, token, &mut cache)?;
    }
    let before = cache_sha256(engine, &cache)?;
    let probe = (|| -> Result<()> {
        match model.install_rewrite_bundle(&corrupt) {
            Ok(()) => return Err("REINSTALL_ACCEPTED_CORRUPT_IDENTITY".into()),
            Err(error) => {
                require(
                    error
                        .to_string()
                        .contains("does not bind numeric_program_sha256="),
                    format!("REINSTALL_WRONG_REASON: {error}"),
                )?;
                println!("REINSTALL_REFUSAL_PASS: {error}");
            }
        }
        require(
            !model.rewrite_is_qualified() && !model.rewrite_allowed(RewriteSurface::DecodeEager),
            "REINSTALL_DID_NOT_REVOKE_EAGER",
        )?;
        match model.decode_step(engine, 5, &mut cache) {
            Ok(_) => return Err("REVOKED_EAGER_PRODUCED_OUTPUT".into()),
            Err(error) => {
                require(
                    error
                        .to_string()
                        .contains("decode-eager rewrite is not qualified"),
                    format!("REVOKED_EAGER_WRONG_REASON: {error}"),
                )?;
                println!("REVOKED_EAGER_REFUSAL_PASS: {error}");
            }
        }
        let after = cache_sha256(engine, &cache)?;
        require(
            before == after,
            format!("REVOKED_EAGER_MUTATED_CACHE before={before} after={after}"),
        )?;
        println!(
            "REINSTALL_REVOCATION_PASS cache_before_sha256={before} cache_after_sha256={after}"
        );
        Ok(())
    })();
    // Restore even when the probe failed, without masking either failure.
    let restored = model.install_rewrite_bundle(bundle);
    if let Err(error) = restored {
        return Err(format!("VALID_REINSTALL_FAILED: {error}; probe={probe:?}").into());
    }
    qualified_eager_only(model)?;
    probe?;
    println!("VALID_REINSTALL_PASS");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    require(
        args.len() == 3,
        "USAGE: rewrite_identity_gate capture|check <model-path> <bundle-dir>",
    )?;
    let mode = args[0].to_str().ok_or("MODE_NOT_UTF8")?;
    require(
        matches!(mode, "capture" | "check"),
        "MODE: expected capture or check",
    )?;
    let source = Path::new(&args[1]);
    let bundle = Path::new(&args[2]);
    let configured_bundle = std::env::var_os("MEMRA_REWRITE_BUNDLE");
    require(
        (mode == "check") == configured_bundle.is_some(),
        "BUNDLE_ENV: capture requires MEMRA_REWRITE_BUNDLE unset; check requires it set",
    )?;
    let lock_path = std::env::var_os("MEMRA_ARTIFACT_LOCK")
        .ok_or("ARTIFACT_LOCK_REQUIRED: set MEMRA_ARTIFACT_LOCK before load")?;
    let lock = std::fs::read(&lock_path).map_err(|e| format!("READ_ARTIFACT_LOCK: {e}"))?;
    require(!lock.is_empty(), "ARTIFACT_LOCK_EMPTY")?;
    require(
        lock == std::fs::read(bundle.join("artifact.lock"))?,
        "ARTIFACT_LOCK_BUNDLE_MISMATCH",
    )?;
    if mode == "capture" {
        require(
            !bundle.join("rewrite-receipts.tsv").exists()
                && !bundle.join("rewrite-receipts").exists()
                && !bundle.join("failed-reinstall-identity").exists(),
            "CAPTURE_REQUIRES_UNUSED_RECEIPT_NAMESPACE",
        )?;
    }
    println!(
        "SCOPE native-program-admission-only support_promotion=false performance=not-measured mode={mode} artifact_lock_sha256={}",
        sha256(&lock)
    );
    let engine = Engine::new(0).map_err(|e| format!("ENGINE_INIT: {e}"))?;
    // Both modes use exactly the same ordinary loader, including its normal MTP policy.
    let mut model = if source.is_dir() || source.extension().is_some_and(|x| x == "safetensors") {
        let src = SafetensorsSource::open(source)?;
        HybridModel::load_from_source(&engine, &src)
    } else if source.extension().is_some_and(|x| x == "gguf") {
        let src = GgufFile::open(source)?;
        HybridModel::load(&engine, &src)
    } else {
        return Err("SOURCE: expected actual GGUF file or HF safetensors directory/file".into());
    }
    .map_err(|e| format!("MODEL_LOAD: {e}"))?;
    require(
        model.devices() == [0],
        "SCOPE_REQUIRES_SINGLE_DEVICE: eager-only evidence cannot admit pipeline",
    )?;
    let identity = model.rewrite_identity()?.clone();
    println!(
        "LOADED artifact_sha256={} implementation_sha256={} numeric_program_sha256={}",
        identity.artifact_sha256, identity.implementation_sha256, identity.numeric_program_sha256
    );
    let pack = memra_gguf::model_packs::for_config(&model.cfg).ok_or("PACK_UNAVAILABLE")?;
    let tolerance = pack
        .checkpoint_parity
        .ok_or("CHECKPOINT_PARITY_TOLERANCE_UNAVAILABLE")?;
    require(
        tolerance.max_abs.is_finite()
            && tolerance.max_abs >= 0.0
            && tolerance.max_rel.is_finite()
            && tolerance.max_rel >= 0.0,
        "CHECKPOINT_PARITY_TOLERANCE_INVALID",
    )?;
    let policy = RewriteParityPolicy {
        max_abs: tolerance.max_abs,
        max_rel: tolerance.max_rel,
        require_argmax: true,
    };
    println!(
        "POLICY family={} atol={} rtol={} require_argmax=true",
        pack.family, policy.max_abs, policy.max_rel
    );
    let rewrite = execution_rewrites(&model.plan)
        .into_iter()
        .find(|r| r.surface == RewriteSurface::DecodeEager)
        .ok_or("EAGER_MANIFEST_MISSING")?;
    require(
        rewrite.eligible(),
        format!("EAGER_INELIGIBLE: {:?}", rewrite.blockers),
    )?;
    for prompt in PROMPTS {
        require(
            prompt.iter().all(|&t| t < model.cfg.n_vocab),
            "PROMPT_TOKEN_OUT_OF_VOCABULARY",
        )?;
    }
    let vocab = model.cfg.n_vocab as usize;
    if mode == "check" {
        // The loader must install the caller's actual bundle; never repair admission here.
        require(
            std::fs::canonicalize(configured_bundle.unwrap())? == std::fs::canonicalize(bundle)?,
            "CHECK_BUNDLE_PATH_MISMATCH",
        )?;
        qualified_eager_only(&model)?;
        graph_refusal(&engine, &model)?;
        fresh_kv_refusal(&engine, &model)?;
        for (i, prompt) in PROMPTS.into_iter().enumerate() {
            output(
                "check-eager",
                i,
                &tokenwise(&engine, &model, prompt)?,
                vocab,
                bundle,
            )?;
        }
    } else {
        require(!model.rewrite_is_qualified(), "CAPTURE_ALREADY_QUALIFIED")?;
        let mut references = Vec::new();
        let mut candidates = Vec::new();
        for (i, prompt) in PROMPTS.into_iter().enumerate() {
            let reference = verify_prefill(&engine, &model, prompt)?;
            let candidate = tokenwise(&engine, &model, prompt)?;
            output(
                "quantized-cache-verify-prefill",
                i,
                &reference,
                vocab,
                bundle,
            )?;
            output("pre-install-tokenwise", i, &candidate, vocab, bundle)?;
            let fresh_control = model.forward_last(&engine, prompt)?;
            output("fresh-kv-diagnostic", i, &fresh_control, vocab, bundle)?;
            let class_control = rewrite.verify_logits(
                &identity.implementation_sha256,
                &fresh_control,
                &candidate,
                policy,
            )?;
            println!(
                "FRESH_KV_CLASS_CONTROL prompt={i} same_class=false max_abs={} max_rel={} within_tolerance={} used_for_qualification=false",
                class_control.max_abs, class_control.max_rel, class_control.passed
            );
            let parity = rewrite.verify_logits(
                &identity.implementation_sha256,
                &reference,
                &candidate,
                policy,
            )?;
            require(
                parity.passed,
                format!(
                    "PROMPT_PARITY_FAILED prompt={i} max_abs={} max_rel={} reference_argmax={} candidate_argmax={} first={:?}",
                    parity.max_abs,
                    parity.max_rel,
                    parity.reference_argmax,
                    parity.candidate_argmax,
                    parity.first_violation
                ),
            )?;
            println!(
                "PROMPT_PARITY_PASS prompt={i} max_abs={} max_rel={}",
                parity.max_abs, parity.max_rel
            );
            references.extend(reference);
            candidates.extend(candidate);
        }
        // Per-prompt checks above prevent a concatenated global argmax from hiding a bad row.
        let receipt = rewrite.verify_logits(
            &identity.implementation_sha256,
            &references,
            &candidates,
            policy,
        )?;
        receipt.validate_for(&rewrite)?;
        let receipt = bind_rewrite_artifact(&model, receipt)?;
        require(
            receipt.artifact_lock_sha256.as_deref() == Some(sha256(&lock).as_str())
                && std::fs::read(bundle.join("artifact.lock"))? == lock,
            "ARTIFACT_LOCK_CHANGED_DURING_CAPTURE",
        )?;
        let receipt = receipt.to_tsv();
        parse_qualified_rewrite_receipt(&receipt)?;
        write_receipt(bundle, &rewrite, &receipt)?;
        model.install_rewrite_bundle(bundle)?;
        qualified_eager_only(&model)?;
        graph_refusal(&engine, &model)?;
        fresh_kv_refusal(&engine, &model)?;
        for (i, prompt) in PROMPTS.into_iter().enumerate() {
            let actual = tokenwise(&engine, &model, prompt)?;
            output("installed-eager", i, &actual, vocab, bundle)?;
            bitwise(
                &candidates[i * vocab..(i + 1) * vocab],
                &actual,
                &format!("installed-eager prompt={i}"),
            )?;
        }
        println!("INSTALLED_EAGER_BITWISE_PASS");
        reinstall_probe(&engine, &mut model, bundle, &rewrite, &receipt)?;
        let restored = tokenwise(&engine, &model, PROMPTS[0])?;
        output("restored-eager", 0, &restored, vocab, bundle)?;
        bitwise(&candidates[..vocab], &restored, "restored-eager")?;
    }
    qualified_eager_only(&model)?;
    println!(
        "REWRITE_IDENTITY_GATE_PASS mode={mode} scope=native-program-admission-only support_promotion=false"
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("REWRITE_IDENTITY_GATE_FAIL: {error}");
        std::process::exit(1);
    }
}
