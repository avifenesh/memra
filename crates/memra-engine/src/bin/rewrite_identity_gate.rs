//! Scoped native program/admission evidence on real checkpoint bytes; no support promotion.
//! Auto-discovered Cargo target and executable: `rewrite_identity_gate`.
//! Usage: rewrite_identity_gate capture|check|fresh-control|library-drift <model-path> <bundle-dir>
//! Inspect first: artifact.lock must already exist in the bundle, and the caller must set
//! MEMRA_ARTIFACT_LOCK before either load. Capture requires MEMRA_REWRITE_BUNDLE unset;
//! check/library-drift require it set to the bundle. Keep the executable and numeric environment
//! identical. Linux library-drift is a separate retained-snapshot refusal probe; it emits no receipt.
//!
//! Separate native caller bundle (same executable, fixed Qwen/Qwen3-0.6B source pin):
//! `retained-capture <model-path> <bundle-dir>`; then one fresh process per
//! `retained-{step,prof-apply,prof-launch,prof-read,prime-run}-{library,environment}`.
//! All retained modes require MEMRA_FAST=0 at exec and MEMRA_ARTIFACT_LOCK. Capture
//! requires MEMRA_REWRITE_BUNDLE unset; each refusal case requires that new bundle set.
//! Environment cases require the external native_env_controller.py rendezvous protocol.
//! Raw evidence lives in retained-capture/ and retained-<case>-<pid>/ under that bundle.

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

#[path = "rewrite_identity_gate/retained.rs"]
mod retained;

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
    let _execution = model.protect_rewrite_execution()?;
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

// Compile the mmap probe on Unix hosts, but run it only on Linux, where strict loaded-library
// identity observes /proc/self/maps. Keeping it separate avoids changing capture/check evidence.
#[cfg(unix)]
mod library_drift {
    use super::*;
    use std::fs::{DirBuilder, File, OpenOptions};
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    use std::path::PathBuf;

    pub(super) struct PrivateFile {
        directory: PathBuf,
        pub(super) path: PathBuf,
    }

    impl PrivateFile {
        pub(super) fn new(bundle: &Path) -> Result<Self> {
            let directory = std::fs::canonicalize(bundle)?
                .join(format!("native-refusal-map-{}", std::process::id()));
            DirBuilder::new().mode(0o700).create(&directory)?;
            let file = Self {
                path: directory.join("never-executed.bin"),
                directory,
            };
            // Keep the name present until after unmapping: a deleted mapping would test a
            // different refusal. Reject names that Linux maps escapes or cannot represent.
            let path = file.path.to_str().ok_or("LIBRARY_DRIFT_PATH_NOT_UTF8")?;
            require(
                !path.contains(['\\', '\n', '\r']),
                "LIBRARY_DRIFT_PATH_NOT_VERIFIABLE",
            )?;
            let mut writer = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&file.path)?;
            writer.write_all(&[0_u8; 4096])?;
            writer.sync_all()?;
            // The only writer is closed before the read/execute mapping is created.
            drop(writer);
            Ok(file)
        }

        pub(super) fn maps(&self) -> Result<Vec<String>> {
            let suffix = format!(" {}", self.path.display());
            Ok(std::fs::read_to_string("/proc/self/maps")?
                .lines()
                .filter(|line| line.ends_with(&suffix))
                .map(str::to_owned)
                .collect())
        }
    }

    impl Drop for PrivateFile {
        fn drop(&mut self) {
            // The Mmap is declared after this owner, so even early errors unmap before unlink.
            for result in [
                std::fs::remove_file(&self.path),
                std::fs::remove_dir(&self.directory),
            ] {
                if let Err(error) = result
                    && error.kind() != std::io::ErrorKind::NotFound
                {
                    eprintln!("LIBRARY_DRIFT_CLEANUP_FAILED: {error}");
                }
            }
        }
    }

    pub(super) fn run(engine: &Engine, model: &HybridModel, bundle: &Path) -> Result<()> {
        qualified_eager_only(model)?;
        let saved = model.rewrite_execution_snapshot()?;
        let mut cache = Cache::new(engine, &model.cfg, PROMPTS[0].len() + 8)?;
        {
            let _first_scope = model.enter_rewrite_execution(&saved)?;
            for &token in PROMPTS[0] {
                model.decode_step(engine, token, &mut cache)?;
            }
            let _nested_scope = model.enter_rewrite_execution(&saved)?;
            require(
                model.rewrite_allowed(RewriteSurface::DecodeEager),
                "RETAINED_NESTED_EAGER_REFUSED",
            )?;
        }
        require(cache.pos == PROMPTS[0].len(), "RETAINED_CACHE_POSITION")?;
        let before = cache_sha256(engine, &cache)?;
        {
            // Positive control: the same origin can resume after its first scope has ended.
            let _resumed_scope = model.enter_rewrite_execution(&saved)?;
            require(
                model.rewrite_allowed(RewriteSurface::DecodeEager),
                "RETAINED_NO_DRIFT_EAGER_REFUSED",
            )?;
        }
        println!("RETAINED_NO_DRIFT_REENTRY_PASS nested_eager=true");

        let file = PrivateFile::new(bundle)?;
        let reader = File::open(&file.path)?;
        // SAFETY: this new, mode-0600 file in a mode-0700 directory has no remaining writer;
        // its owner outlives this mapping and never changes its bytes. map_exec supplies
        // PROT_READ | PROT_EXEC. No mapped bytes are ever called, executed, or modified.
        let mapping = unsafe { memmap2::MmapOptions::new().map_exec(&reader)? };
        let maps = file.maps()?;
        require(
            maps.len() == 1
                && maps[0]
                    .split_whitespace()
                    .nth(1)
                    .is_some_and(|permissions| {
                        permissions.contains('x') && !permissions.contains('w')
                    }),
            format!("LIBRARY_DRIFT_MAPPING_NOT_EXECUTABLE_READ_ONLY: {maps:?}"),
        )?;
        println!(
            "LIBRARY_DRIFT_MAPPING file={} bytes={} sha256={} maps={maps:?}",
            file.path.display(),
            mapping.len(),
            sha256(&[0_u8; 4096]),
        );

        // This MUST be the first identity/admission operation after the mapping changes.
        // No explicit validator, fresh snapshot, reinstall or model mutation may revoke the
        // saved origin on behalf of enter_rewrite_execution. The continuation witnesses that
        // refusal happens before token work, even if decode_step itself would later refuse.
        let mut token_calls = 0;
        let attempt = (|| -> Result<()> {
            let _scope = model.enter_rewrite_execution(&saved)?;
            token_calls += 1;
            model.decode_step(engine, 5, &mut cache)?;
            Ok(())
        })();
        let error = match attempt {
            Ok(()) => return Err("RETAINED_LIBRARY_DRIFT_ADMITTED_TOKEN".into()),
            Err(error) => error.to_string(),
        };
        require(
            token_calls == 0,
            "RETAINED_LIBRARY_DRIFT_REACHED_TOKEN_WORK",
        )?;
        require(
            error.contains("loaded executable mappings changed")
                && error.contains(file.path.to_str().ok_or("LIBRARY_DRIFT_PATH_NOT_UTF8")?),
            format!("RETAINED_LIBRARY_DRIFT_WRONG_REASON: {error}"),
        )?;
        require(
            model.check_rewrite_execution(&saved).is_err(),
            "RETAINED_LIBRARY_DRIFT_DID_NOT_REVOKE_ORIGIN",
        )?;
        let after = cache_sha256(engine, &cache)?;
        require(
            before == after,
            format!("RETAINED_LIBRARY_DRIFT_MUTATED_CACHE before={before} after={after}"),
        )?;
        println!("RETAINED_LIBRARY_DRIFT_REFUSAL_PASS token_calls={token_calls} reason={error}");
        println!("RETAINED_LIBRARY_DRIFT_CACHE_PASS before_sha256={before} after_sha256={after}");

        drop(mapping);
        require(file.maps()?.is_empty(), "LIBRARY_DRIFT_UNMAP_FAILED")?;
        println!("RETAINED_LIBRARY_DRIFT_UNMAP_PASS");
        // Do not refresh the origin or reinstall its bundle after removing the drift.
        match model.enter_rewrite_execution(&saved) {
            Ok(_) => return Err("RETAINED_LIBRARY_DRIFT_UNMAP_REVIVED_ORIGIN".into()),
            Err(error) => require(
                error.contains("snapshot was revoked"),
                format!("RETAINED_LIBRARY_DRIFT_UNMAP_WRONG_REASON: {error}"),
            )?,
        }
        // The external identity is back to its baseline; the original snapshot stays revoked.
        model.rewrite_identity()?;
        require(
            model.check_rewrite_execution(&saved).is_err(),
            "RETAINED_LIBRARY_DRIFT_IDENTITY_CHECK_REVIVED_ORIGIN",
        )?;
        require(
            cache_sha256(engine, &cache)? == before,
            "RETAINED_LIBRARY_DRIFT_UNMAP_MUTATED_CACHE",
        )?;
        println!("RETAINED_LIBRARY_DRIFT_REVOCATION_PASS restored_external_identity=true");
        println!(
            "NATIVE_REFUSAL_GATE_PASS mode=library-drift scope=retained-eager-snapshot receipt_emitted=false support_promotion=false"
        );
        Ok(())
    }
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
    let prior_execution = model.rewrite_execution_snapshot()?;
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
        require(
            model.enter_rewrite_execution(&prior_execution).is_err(),
            "FAILED_REINSTALL_RETAINED_SNAPSHOT",
        )?;
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
    require(
        model.enter_rewrite_execution(&prior_execution).is_err(),
        "VALID_REINSTALL_REVIVED_OLD_SNAPSHOT",
    )?;
    let renewed = model.rewrite_execution_snapshot()?;
    let _scope = model.enter_rewrite_execution(&renewed)?;
    require(
        model.rewrite_allowed(RewriteSurface::DecodeEager),
        "RENEWED_SNAPSHOT_REFUSED",
    )?;
    println!("REINSTALL_SNAPSHOT_REVOCATION_PASS");
    println!("VALID_REINSTALL_PASS");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    require(
        args.len() == 3,
        "USAGE: rewrite_identity_gate <capture|check|fresh-control|library-drift|retained-capture|retained-{step,prof-apply,prof-launch,prof-read,prime-run}-{library,environment}> <model-path> <bundle-dir>",
    )?;
    let mode = args[0].to_str().ok_or("MODE_NOT_UTF8")?;
    // Separate namespace, observations and receipts. Historical eager modes below
    // deliberately retain their existing semantics and their eager-only admission.
    if mode.starts_with("retained-") {
        return retained::run(mode, Path::new(&args[1]), Path::new(&args[2]));
    }
    require(
        matches!(
            mode,
            "capture" | "check" | "fresh-control" | "library-drift"
        ),
        "MODE: expected capture, check, fresh-control or library-drift",
    )?;
    require(
        mode != "library-drift" || cfg!(target_os = "linux"),
        "LIBRARY_DRIFT_REQUIRES_LINUX",
    )?;
    let source = Path::new(&args[1]);
    let bundle = Path::new(&args[2]);
    let configured_bundle = std::env::var_os("MEMRA_REWRITE_BUNDLE");
    require(
        matches!(mode, "check" | "library-drift") == configured_bundle.is_some(),
        "BUNDLE_ENV: capture/fresh-control require MEMRA_REWRITE_BUNDLE unset; check/library-drift require it set",
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
    if mode == "fresh-control" {
        println!(
            "DIAGNOSTIC_SOURCE artifact_sha256={} implementation_sha256={} program=unqualified-fresh-kv",
            identity.artifact_sha256, identity.implementation_sha256
        );
        for (i, prompt) in PROMPTS.into_iter().enumerate() {
            require(
                prompt.iter().all(|&token| token < model.cfg.n_vocab),
                "PROMPT_TOKEN_OUT_OF_VOCABULARY",
            )?;
            output(
                "fresh-kv-diagnostic",
                i,
                &model.forward_last(&engine, prompt)?,
                model.cfg.n_vocab as usize,
                bundle,
            )?;
        }
        println!("FRESH_CONTROL_DONE qualification=false receipt_emitted=false");
        return Ok(());
    }
    println!(
        "LOADED artifact_sha256={} implementation_sha256={} numeric_program_sha256={}",
        identity.artifact_sha256, identity.implementation_sha256, identity.numeric_program_sha256
    );
    #[cfg(unix)]
    if mode == "library-drift" {
        require(
            std::fs::canonicalize(configured_bundle.as_ref().ok_or("BUNDLE_ENV_MISSING")?)?
                == std::fs::canonicalize(bundle)?,
            "LIBRARY_DRIFT_BUNDLE_PATH_MISMATCH",
        )?;
        require(
            PROMPTS[0].iter().all(|&token| token < model.cfg.n_vocab),
            "PROMPT_TOKEN_OUT_OF_VOCABULARY",
        )?;
        return library_drift::run(&engine, &model, bundle);
    }
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
            model.rewrite_identity()?;
            let candidate = tokenwise(&engine, &model, prompt)?;
            model.rewrite_identity()?;
            output(
                "quantized-cache-verify-prefill",
                i,
                &reference,
                vocab,
                bundle,
            )?;
            output("pre-install-tokenwise", i, &candidate, vocab, bundle)?;
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
    let retained = model.rewrite_execution_snapshot()?;
    // Even same-shape mutation through the public field API revokes the byte contract.
    // Restoring metadata must not authorize replacement tensor bytes without a reload.
    let epsilon = model.cfg.rms_eps;
    model.cfg.rms_eps = epsilon * 2.0;
    model.cfg.rms_eps = epsilon;
    require(
        model.enter_rewrite_execution(&retained).is_err(),
        "MUTATION_RETAINED_SNAPSHOT",
    )?;
    require(
        model.rewrite_identity().is_err(),
        "MUTATION_RETAINED_IDENTITY",
    )?;
    require(
        !model.rewrite_allowed(RewriteSurface::DecodeEager),
        "MUTATION_RETAINED_ADMISSION",
    )?;
    println!("MODEL_MUTATION_REVOCATION_PASS");
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
