//! Native boundary witnesses in the server *lib test executable*, never serving qualification.
//! Run exactly `worker::rewrite_native_tests::native_worker_boundary --exact --ignored
//! --nocapture --test-threads=1`, once per (CASE, DRIFT) in a fresh process.
//! REWRITE_PROBE_SOURCE: the pinned Qwen3-0.6B safetensors directory below.
//! REWRITE_PROBE_BUNDLE: unused evidence directory containing artifact.lock.
//! REWRITE_PROBE_CASE: advance_sample_emit | advance_token_emit | step_session |
//! step_session_prefill | prefill_tick | step_session_async_chain.
//! REWRITE_PROBE_DRIFT: env | library. Launch with MEMRA_FAST=0, MEMRA_ARTIFACT_LOCK
//! pointing to that artifact.lock, and MEMRA_REWRITE_BUNDLE absent. The async case additionally
//! needs MEMRA_ASYNC_CHAIN=2. Env cases run under native_env_controller.py; no in-process
//! environment mutation is permitted. Missing prerequisites FAIL explicit invocation.

use super::*;
use memra_engine::plan_backend::{
    RewriteParityPolicy, RewriteSurface, bind_rewrite_artifact, execution_rewrites,
    parse_qualified_rewrite_receipt,
};
use memra_gguf::source::SafetensorsSource;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

#[path = "../../../memra-engine/tests/support/env_drift_control.rs"]
mod env_drift_control;

pub(super) mod diagnostics;

type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;
const TEST_NAME: &str = "worker::rewrite_native_tests::native_worker_boundary";
const SOURCE: &str = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca";
const PROMPTS: [&[u32]; 3] = [
    &[1, 2, 3, 4],
    &[7, 11, 19, 5, 3, 17, 2, 13],
    &[2, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 2, 3, 2, 3],
];

fn require(ok: bool, why: impl Into<String>) -> ProbeResult<()> {
    if ok { Ok(()) } else { Err(why.into().into()) }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect()
}

fn save(bundle: &Path, name: &str, bytes: &[u8]) -> ProbeResult<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(bundle.join(name))?
        .write_all(bytes)?;
    println!(
        "WORKER_PROBE_FILE name={name} bytes={} sha256={}",
        bytes.len(),
        sha(bytes)
    );
    Ok(())
}

fn pinned_source(source: &Path) -> ProbeResult<()> {
    require(
        source.is_dir(),
        "REWRITE_PROBE_SOURCE must be a real safetensors directory",
    )?;
    // Same revision as the native runner; no alternative model/encoding is accepted.
    for (name, expected) in [
        (
            "model.safetensors",
            "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        ),
        (
            "config.json",
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        ),
        (
            "generation_config.json",
            "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        ),
        (
            "tokenizer.json",
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        ),
        (
            "tokenizer_config.json",
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        ),
        (
            "merges.txt",
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        ),
        (
            "vocab.json",
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ] {
        let mut file = File::open(source.join(name))?;
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        let actual = format!("{:x}", hash.finalize());
        require(actual == expected, format!("PIN_MISMATCH {name}: {actual}"))?;
        println!("WORKER_PROBE_SOURCE file={name} sha256={actual}");
    }
    println!("WORKER_PROBE_PIN {SOURCE}");
    Ok(())
}

fn eager(engine: &Engine, model: &HybridModel, prompt: &[u32]) -> ProbeResult<Vec<f32>> {
    let _scope = model.protect_rewrite_execution()?;
    let mut cache = Cache::new(engine, &model.cfg, prompt.len() + 8)?;
    let mut logits = Vec::new();
    for &token in prompt {
        logits = model.decode_step(engine, token, &mut cache)?;
    }
    require(cache.pos == prompt.len(), "EAGER_CACHE_POSITION")?;
    Ok(logits)
}

fn qualify(
    engine: &Engine,
    model: &mut HybridModel,
    bundle: &Path,
    lock: &[u8],
) -> ProbeResult<()> {
    require(
        !model.rewrite_is_qualified(),
        "DO_NOT_BORROW_ANOTHER_EXECUTABLE_RECEIPT",
    )?;
    let identity = model.rewrite_identity()?.clone();
    let rewrite = execution_rewrites(&model.plan)
        .into_iter()
        .find(|r| r.surface == RewriteSurface::DecodeEager)
        .ok_or("EAGER_MANIFEST_MISSING")?;
    require(rewrite.eligible(), "EAGER_MANIFEST_INELIGIBLE")?;
    let pack = memra_gguf::model_packs::for_config(&model.cfg).ok_or("MODEL_PACK_MISSING")?;
    let tolerance = pack
        .checkpoint_parity
        .ok_or("CHECKPOINT_TOLERANCE_MISSING")?;
    let policy = RewriteParityPolicy {
        max_abs: tolerance.max_abs,
        max_rel: tolerance.max_rel,
        require_argmax: true,
    };
    let vocab = model.cfg.n_vocab as usize;
    let mut references = Vec::new();
    let mut candidates = Vec::new();
    for (i, prompt) in PROMPTS.into_iter().enumerate() {
        require(prompt.iter().all(|&t| t < model.cfg.n_vocab), "PROMPT_OOV")?;
        // Independent cached teacher-forced verify-prefill, NOT fresh-F32-KV forward_last
        // and NOT the eager candidate compared with itself.
        let mut cache = Cache::new(engine, &model.cfg, prompt.len() + 8)?;
        let rows = model.decode_step_t(engine, prompt, 0, &mut cache)?;
        require(
            rows.len() == prompt.len() * vocab && cache.pos == prompt.len(),
            "VERIFY_PREFILL_SHAPE",
        )?;
        let reference = &rows[rows.len() - vocab..];
        model.rewrite_identity()?;
        let candidate = eager(engine, model, prompt)?;
        model.rewrite_identity()?;
        for (label, values) in [
            ("verify-prefill", reference),
            ("eager", candidate.as_slice()),
        ] {
            require(
                values.len() == vocab && values.iter().all(|x| x.is_finite()),
                "INVALID_LOGITS",
            )?;
            save(bundle, &format!("{label}-{i}.f32"), &f32_bytes(values))?;
        }
        let parity = rewrite.verify_logits(
            &identity.implementation_sha256,
            reference,
            &candidate,
            policy,
        )?;
        require(
            parity.passed,
            format!(
                "NATIVE_EAGER_PARITY_FAILED prompt={i} abs={} rel={} first={:?}",
                parity.max_abs, parity.max_rel, parity.first_violation
            ),
        )?;
        println!(
            "WORKER_PROBE_PARITY prompt={i} abs={} rel={}",
            parity.max_abs, parity.max_rel
        );
        references.extend_from_slice(reference);
        candidates.extend(candidate);
    }
    let receipt = rewrite.verify_logits(
        &identity.implementation_sha256,
        &references,
        &candidates,
        policy,
    )?;
    receipt.validate_for(&rewrite)?;
    let receipt = bind_rewrite_artifact(model, receipt)?;
    require(
        receipt.artifact_lock_sha256.as_deref() == Some(sha(lock).as_str())
            && std::fs::read(bundle.join("artifact.lock"))? == lock,
        "ARTIFACT_LOCK_CHANGED",
    )?;
    let receipt = receipt.to_tsv();
    parse_qualified_rewrite_receipt(&receipt)?;
    std::fs::create_dir(bundle.join("rewrite-receipts"))?;
    save(
        bundle,
        &format!("rewrite-receipts/{}.tsv", rewrite.id),
        receipt.as_bytes(),
    )?;
    save(
        bundle,
        "rewrite-receipts.tsv",
        format!(
            "rewrite\tplan_sha256\treceipt_sha256\tstatus\n{}\t{}\t{}\tpassed\n",
            rewrite.id,
            rewrite.plan_sha256,
            sha(receipt.as_bytes())
        )
        .as_bytes(),
    )?;
    model.install_rewrite_bundle(bundle)?;
    require(model.rewrite_is_qualified(), "EAGER_INSTALL_FAILED")?;
    for r in execution_rewrites(&model.plan) {
        require(
            model.rewrite_allowed(r.surface) == (r.surface == RewriteSurface::DecodeEager),
            "NOT_EAGER_ONLY",
        )?;
    }
    for (i, prompt) in PROMPTS.into_iter().enumerate() {
        let actual = f32_bytes(&eager(engine, model, prompt)?);
        save(bundle, &format!("installed-eager-{i}.f32"), &actual)?;
        require(
            actual == f32_bytes(&candidates[i * vocab..(i + 1) * vocab]),
            "INSTALL_CHANGED_EAGER",
        )?;
    }
    println!(
        "WORKER_PROBE_QUALIFIED executable={} implementation={} artifact={} numeric={} scope=lib-test-eager-only serving_qualified=false",
        std::env::current_exe()?.display(),
        identity.implementation_sha256,
        identity.artifact_sha256,
        identity.numeric_program_sha256
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Caller {
    Sample,
    Token,
    Step,
    StepPrefill,
    Prefill,
    Async,
}

impl Caller {
    fn parse(s: &str) -> ProbeResult<Self> {
        Ok(match s {
            "advance_sample_emit" => Self::Sample,
            "advance_token_emit" => Self::Token,
            "step_session" => Self::Step,
            "step_session_prefill" => Self::StepPrefill,
            "prefill_tick" => Self::Prefill,
            "step_session_async_chain" => Self::Async,
            _ => return Err(format!("UNKNOWN_REWRITE_PROBE_CASE: {s}").into()),
        })
    }

    fn prefill(self) -> bool {
        matches!(self, Self::StepPrefill | Self::Prefill)
    }

    // This dispatch is the FIRST model/caller operation after external drift. In particular,
    // do not enter a guard, validate, refresh a snapshot or install a bundle on its behalf.
    fn call(
        self,
        engine: &Engine,
        loaded: &HashMap<String, LoadedModel>,
        s: &mut Session,
        px: &mut PrefixCache,
        hpx: &mut HostPrefixCache,
        token: u32,
    ) -> ProbeResult<bool> {
        match self {
            Self::Sample => {
                let (keep, next) = advance_sample_emit(loaded, s);
                require(keep == next.is_some(), "SAMPLE_RETURN_SHAPE")?;
                Ok(keep)
            }
            Self::Token => Ok(advance_token_emit(loaded, s, token).0),
            Self::Step | Self::StepPrefill => {
                step_session(engine, loaded, s, &mut SpecMetricState::new(60.0))
            }
            Self::Prefill => Ok(prefill_tick(
                engine,
                loaded,
                px,
                hpx,
                s,
                1,
                None,
                None,
                None,
                None,
                memra_engine::vision::OverlayPublish::Auto,
            )? == 1),
            Self::Async => step_session_async_chain(engine, loaded, s)?
                .ok_or_else(|| "ASYNC_PATH_DECLINED".into()),
        }
    }
}

fn session(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    px: &mut PrefixCache,
    hpx: &mut HostPrefixCache,
) -> ProbeResult<(Session, EventReceiver)> {
    let mut req = super::tests::bare_request();
    let (tx, rx) = event_channel();
    req.tx = tx;
    req.prompt_ids = PROMPTS[0].to_vec();
    req.params = GenParams {
        max_new: 16,
        max_ctx: Some(64),
        eos: Vec::new(),
    };
    req.sampler_cfg = SamplerConfig {
        temperature: 0.0,
        ..Default::default()
    };
    req.spec_k_replay = Some(0);
    let shape = prepare_request(loaded, &mut req, None).map_err(|e| e.message)?;
    let s = admit(
        engine,
        loaded,
        &mut HashMap::new(),
        &mut HashMap::new(),
        &mut HashMap::new(),
        &mut SpecSizing::default(),
        &mut ReuseMetrics::default(),
        px,
        hpx,
        0,
        0,
        false,
        false,
        req,
        shape,
        None,
        false,
        false,
        None,
        false,
        None,
        None,
        None,
    )
    .map_err(|(_, e)| e.message)?;
    require(
        s.cache.is_some()
            && s.spec.is_none()
            && !s.dspark_on
            && !s.glm5_on
            && s.gspec_k == 0
            && s.vision.is_none()
            && s.constraint.is_none()
            && !s.prefill_done,
        "NOT_PLAIN_NATIVE_SESSION",
    )?;
    Ok((s, rx))
}

fn events(rx: &mut EventReceiver) -> Vec<Event> {
    let mut out = Vec::new();
    while let Ok(event) = rx.try_recv() {
        out.push(event);
    }
    out
}

#[derive(PartialEq)]
struct Snapshot {
    cache: Vec<u8>,
    state: serde_json::Value,
}

impl Snapshot {
    fn capture(
        engine: &Engine,
        s: &Session,
        px: &PrefixCache,
        hpx: &HostPrefixCache,
    ) -> ProbeResult<Self> {
        engine.ctx().synchronize()?;
        let cache = s.cache.as_ref().ok_or("CACHE_MISSING")?;
        // Qwen3-0.6B is a dense KV model. Refuse new/unobserved planes instead of claiming
        // their bytes were checked. Hash FULL allocations, including unwritten capacity.
        require(
            cache.recur.iter().all(Option::is_none)
                && cache.latent.iter().all(Option::is_none)
                && cache.tp_kv.iter().all(Option::is_none)
                && cache.glm5_tp_recur.iter().all(Option::is_none)
                && cache.glm5_tp_latent_peer.iter().all(Option::is_none)
                && cache.qwen_prime_graph.is_none()
                && cache.glm5_decode_graph.is_none()
                && cache.glm5_tp_sym_graph.is_none()
                && cache.dflash_taps.is_none()
                && cache.hc_taps.is_none(),
            "UNOBSERVED_CACHE_STATE",
        )?;
        require(
            s.ckpt_snap.is_none()
                && s.mask_dev.is_none()
                && s.glm5_device_sampler.is_none()
                && px.entries.is_empty()
                && hpx.entries.is_empty(),
            "UNOBSERVED_WORKER_STATE",
        )?;
        let mut bytes = Vec::new();
        let mut layers = Vec::new();
        for kv in &cache.kv {
            let kv = kv.as_ref().ok_or("PINNED_QWEN_KV_LAYER_MISSING")?;
            for plane in [&kv.k, &kv.v] {
                let plane = engine.dtoh_u8(plane)?;
                bytes.extend_from_slice(&(plane.len() as u64).to_le_bytes());
                bytes.extend(plane);
            }
            layers.push(serde_json::json!({"len": kv.len, "len_d": engine.dtoh_i32(&kv.len_d)?,
                "ring": format!("{:?}", kv.ring), "base_d": kv.base_d.as_ref().map(|b| engine.dtoh_i32(b)).transpose()?}));
        }
        let device_logits = cache
            .last_logits_dev
            .as_ref()
            .map(|row| engine.dtoh(row))
            .transpose()?;
        let state = serde_json::json!({
            "pos": cache.pos, "max_ctx": cache.max_ctx, "tainted": cache.tainted, "layers": layers,
            "generated": s.generated, "fed": s.fed, "last_logits_bits": s.last_logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "device_logits_bits": device_logits.map(|v| v.into_iter().map(f32::to_bits).collect::<Vec<_>>()),
            "device_next": s.device_next, "prefill_queue": s.prefill_queue, "prefill_done": s.prefill_done,
            "decoded_bytes": s.decoded_bytes, "emitted_bytes": s.emitted_bytes, "tokens_emitted": s.tokens_emitted,
            "snapshot_at": s.snapshot_at, "ckpt_at": s.ckpt_at, "seed_prefix": s.seed_prefix,
            "prefix_pin": format!("{:?}", s.prefix_pin), "prompt_tok_decode_step": s.prompt_tok_decode_step,
            "prefix": [px.next_id, px.inserts, px.evictions, px.total_bytes as u64],
            "host_prefix": [hpx.demotions, hpx.promotions, hpx.evictions, hpx.total_bytes as u64],
        });
        Ok(Self {
            cache: bytes,
            state,
        })
    }

    fn persist(&self, bundle: &Path, label: &str) -> ProbeResult<()> {
        save(bundle, &format!("{label}.cache.bin"), &self.cache)?;
        save(
            bundle,
            &format!("{label}.state.json"),
            &serde_json::to_vec(&self.state)?,
        )
    }
}

// The file is private and immutable while mapped RX, and its bytes are NEVER executed.
// libc is already a server dependency; this adds no production mapping or dependency.
#[cfg(unix)]
struct LibraryDrift {
    address: *mut libc::c_void,
    directory: PathBuf,
    path: PathBuf,
}

#[cfg(unix)]
impl LibraryDrift {
    fn new(bundle: &Path) -> ProbeResult<Self> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
        let directory =
            std::fs::canonicalize(bundle)?.join(format!("private-rx-{}", std::process::id()));
        std::fs::DirBuilder::new().mode(0o700).create(&directory)?;
        let mut owner = Self {
            address: libc::MAP_FAILED,
            path: directory.join("never-executed.bin"),
            directory,
        };
        let path = owner.path.to_str().ok_or("RX_PATH_NOT_UTF8")?;
        require(!path.contains(['\\', '\n', '\r']), "RX_PATH_UNVERIFIABLE")?;
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&owner.path)?;
        writer.write_all(&[0; 4096])?;
        writer.sync_all()?;
        drop(writer);
        let reader = File::open(&owner.path)?;
        // SAFETY: a new private file, fixed at 4096 bytes, with no writer; owner unmaps
        // before unlinking. No reference/function pointer is formed from this address.
        owner.address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_EXEC,
                libc::MAP_PRIVATE,
                reader.as_raw_fd(),
                0,
            )
        };
        require(
            owner.address != libc::MAP_FAILED,
            format!("RX_MAP_FAILED: {}", std::io::Error::last_os_error()),
        )?;
        let maps = owner.maps()?;
        require(
            maps.len() == 1 && maps[0].split_whitespace().nth(1) == Some("r-xp"),
            "RX_MAP_NOT_PRIVATE_READ_EXECUTE",
        )?;
        println!(
            "WORKER_PROBE_RX path={} sha256={} maps={maps:?}",
            owner.path.display(),
            sha(&[0; 4096])
        );
        Ok(owner)
    }

    fn maps(&self) -> ProbeResult<Vec<String>> {
        let suffix = format!(" {}", self.path.display());
        Ok(std::fs::read_to_string("/proc/self/maps")?
            .lines()
            .filter(|l| l.ends_with(&suffix))
            .map(str::to_owned)
            .collect())
    }

    fn restore(&mut self) -> ProbeResult<()> {
        if self.address != libc::MAP_FAILED {
            // SAFETY: this is exactly the address and length returned by mmap above.
            require(
                unsafe { libc::munmap(self.address, 4096) } == 0,
                "RX_UNMAP_FAILED",
            )?;
            self.address = libc::MAP_FAILED;
        }
        require(self.maps()?.is_empty(), "RX_MAPPING_RETAINED")
    }
}

#[cfg(unix)]
impl Drop for LibraryDrift {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            eprintln!("RX_CLEANUP_FAILED: {error}");
            return;
        }
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

#[test]
#[ignore = "native CUDA; pinned artifact, fresh in-executable qualification and external drift controller required"]
fn native_worker_boundary() -> ProbeResult<()> {
    require(cfg!(target_os = "linux"), "NATIVE_PROBE_REQUIRES_LINUX")?;
    require(
        option_env!("DOCS_RS").is_none() && std::env::var_os("DOCS_RS").is_none(),
        "DOCUMENTATION_BUILD_IS_NOT_NATIVE",
    )?;
    let args: Vec<_> = std::env::args().collect();
    require(
        [
            TEST_NAME,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ]
        .iter()
        .all(|arg| args.iter().any(|a| a == arg)),
        "INVOKE_EXACTLY_ONE_NATIVE_TEST_PER_FRESH_PROCESS",
    )?;
    let case = std::env::var("REWRITE_PROBE_CASE")?;
    let caller = Caller::parse(&case)?;
    let drift = std::env::var("REWRITE_PROBE_DRIFT")?;
    require(
        matches!(drift.as_str(), "env" | "library"),
        "DRIFT_MUST_BE_ENV_OR_LIBRARY",
    )?;
    require(
        std::env::var("MEMRA_FAST").as_deref() == Ok("0"),
        "LAUNCH_WITH_MEMRA_FAST_0",
    )?;
    require(
        std::env::var_os("MEMRA_REWRITE_BUNDLE").is_none(),
        "MEMRA_REWRITE_BUNDLE_MUST_BE_UNSET",
    )?;
    if caller == Caller::Async {
        require(
            std::env::var("MEMRA_ASYNC_CHAIN").as_deref() == Ok("2"),
            "ASYNC_REQUIRES_MEMRA_ASYNC_CHAIN_2_AT_LAUNCH",
        )?;
    }
    let source = PathBuf::from(
        std::env::var_os("REWRITE_PROBE_SOURCE").ok_or("REWRITE_PROBE_SOURCE_REQUIRED")?,
    );
    let bundle = PathBuf::from(
        std::env::var_os("REWRITE_PROBE_BUNDLE").ok_or("REWRITE_PROBE_BUNDLE_REQUIRED")?,
    );
    let lock_path = PathBuf::from(
        std::env::var_os("MEMRA_ARTIFACT_LOCK").ok_or("ARTIFACT_LOCK_REQUIRED_AT_LAUNCH")?,
    );
    let lock = std::fs::read(&lock_path)?;
    require(
        !lock.is_empty()
            && std::fs::canonicalize(&lock_path)?
                == std::fs::canonicalize(bundle.join("artifact.lock"))?,
        "BUNDLE_LOCK_MISMATCH",
    )?;
    require(
        !bundle.join("rewrite-receipts").exists() && !bundle.join("rewrite-receipts.tsv").exists(),
        "FRESH_BUNDLE_REQUIRED",
    )?;
    save(
        &bundle,
        "worker-probe.json",
        &serde_json::to_vec(&serde_json::json!({"test": TEST_NAME,
        "case": case, "drift": drift, "source": SOURCE, "pid": std::process::id(),
        "executable": std::env::current_exe()?, "artifact_lock_sha256": sha(&lock), "serving_qualified": false}))?,
    )?;
    pinned_source(&source)?;
    let engine = Engine::new(0)?;
    let src = SafetensorsSource::open(&source)?;
    let mut model = HybridModel::load_from_source(&engine, &src)?;
    require(model.devices() == [0], "SINGLE_DEVICE_REQUIRED")?;
    qualify(&engine, &mut model, &bundle, &lock)?;
    let tok = Arc::new(Tokenizer::from_hf_dir(&source)?);
    let mut lm = LoadedModel::new_for_test(model, tok);
    lm.eos_id = lm.tok.eos_id();
    lm.from_dir = true;
    let loaded = HashMap::from([("m".to_string(), lm)]);
    let model = &loaded["m"].model;
    let mut px = PrefixCache::default();
    let mut hpx = HostPrefixCache::new(0);
    let (mut s, mut rx) = session(&engine, &loaded, &mut px, &mut hpx)?;
    let admitted = events(&mut rx);
    // This fixture calls admit directly. PromptUsage is published by the outer
    // worker loop after admit returns; it is not an event of this entry point.
    require(
        s.n_prompt == PROMPTS[0].len() && s.n_cached == 0 && admitted.is_empty(),
        format!(
            "DIRECT_ADMISSION_MISMATCH n_prompt={} n_cached={} events={admitted:?}",
            s.n_prompt, s.n_cached
        ),
    )?;
    println!(
        "WORKER_PROBE_ADMITTED entry=admit n_prompt={} n_cached={} events=0",
        s.n_prompt, s.n_cached
    );
    // Complete the omitted outer scheduler publication using its actual producer
    // and the returned session's fields, before observing any target caller.
    publish_admitted_prompt_usage(&s.tx, s.n_prompt, s.n_cached);
    let admitted = events(&mut rx);
    require(
        admitted.len() == 1
            && matches!(
                admitted[0],
                Event::PromptUsage {
                    n_prompt: 4,
                    n_cached: 0
                }
            ),
        "ADMISSION_USAGE_MISSING",
    )?;
    // Keep this very origin throughout. Guards from admission and every prefill call have
    // closed before the positive re-entry, and that caller's guard closes before injection.
    let original = s.rewrite_execution.clone();
    if !caller.prefill() {
        for _ in 0..PROMPTS[0].len() {
            require(
                step_session(&engine, &loaded, &mut s, &mut SpecMetricState::new(60.0))?,
                "PREFILL_STOPPED",
            )?;
        }
        require(
            s.prefill_done && s.fed == PROMPTS[0] && events(&mut rx).is_empty(),
            "PREFILL_NOT_READY",
        )?;
    }
    let token = if caller.prefill() {
        5
    } else {
        memra_engine::forward::argmax(&s.last_logits) as u32
    };
    require(
        !s.params.eos.contains(&token),
        "POSITIVE_CONTROL_WOULD_STOP_ON_EOS",
    )?;
    let initial = Snapshot::capture(&engine, &s, &px, &hpx)?;
    initial.persist(&bundle, "control-before")?;
    require(
        caller.call(&engine, &loaded, &mut s, &mut px, &mut hpx, token)?,
        "POSITIVE_CALL_REFUSED",
    )?;
    let control_events = events(&mut rx);
    if caller.prefill() {
        require(
            s.fed == PROMPTS[0][..1]
                && s.prefill_queue
                    .iter()
                    .copied()
                    .eq(PROMPTS[0][1..].iter().copied())
                && !s.prefill_done
                && control_events.is_empty(),
            "PREFILL_CONTROL_DID_NOT_ADVANCE",
        )?;
    } else {
        let ids: Vec<_> = control_events
            .iter()
            .filter_map(|e| match e {
                Event::Token { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        require(
            ids.first() == Some(&token) && ids == s.generated && ids.len() == control_events.len(),
            "TOKEN_CONTROL_NOT_SUCCESSFUL",
        )?;
        if caller == Caller::Step {
            require(
                s.fed.len() == PROMPTS[0].len() + 1 && s.fed.last() == Some(&token),
                "STEP_CONTROL_DID_NOT_FEED_TOKEN",
            )?;
        }
        if caller == Caller::Async {
            require(
                s.generated.len() == 2
                    && s.fed.len() == PROMPTS[0].len() + 2
                    && s.device_next.is_some(),
                "ASYNC_CONTROL_DID_NOT_ENGAGE_CHAIN",
            )?;
        }
    }
    require(
        s.cache
            .as_ref()
            .is_some_and(|cache| cache.pos == s.fed.len())
            && s.last_logits.len() == model.cfg.n_vocab as usize
            && s.last_logits.iter().all(|value| value.is_finite()),
        "CONTROL_CACHE_OR_LOGITS_INVALID",
    )?;
    save(
        &bundle,
        "control-events.txt",
        format!("{control_events:?}\n").as_bytes(),
    )?;
    let before = Snapshot::capture(&engine, &s, &px, &hpx)?;
    require(before != initial, "VACUOUS_POSITIVE_CONTROL")?;
    before.persist(&bundle, "before-drift")?;
    println!("WORKER_PROBE_CONTROL_PASS case={case}");
    // Snapshot capture fenced all device work. No guard is held during the external stop/map.
    #[cfg(unix)]
    let mut mapping = if drift == "library" {
        Some(LibraryDrift::new(&bundle)?)
    } else {
        None
    };
    if drift == "env" {
        engine.ctx().synchronize()?;
        env_drift_control::rendezvous("mutate")?;
    }
    let (attempt, diagnostics) =
        diagnostics::capture(|| caller.call(&engine, &loaded, &mut s, &mut px, &mut hpx, token))?;
    let expected = if drift == "env" {
        "numerical environment changed since load"
    } else {
        "loaded executable mappings changed"
    };
    // Capture real bytes before restoration, even when the refusal assertion will fail.
    let after = Snapshot::capture(&engine, &s, &px, &hpx);
    let refused = diagnostics::refusal(attempt, &mut rx, expected, diagnostics);
    if drift == "env" {
        engine.ctx().synchronize()?;
        env_drift_control::rendezvous("restore")?;
    }
    #[cfg(unix)]
    if let Some(mapping) = mapping.as_mut() {
        mapping.restore()?;
    }
    let after = after?;
    after.persist(&bundle, "after-drift")?;
    let reason = refused?;
    require(before == after, "DRIFT_MUTATED_CACHE_OR_TOKEN_STATE")?;
    require(
        reason.contains(if drift == "env" {
            "MEMRA_FAST"
        } else {
            "never-executed.bin"
        }),
        "REFUSAL_DID_NOT_NAME_FAULT",
    )?;
    save(&bundle, "drift-error.txt", reason.as_bytes())?;
    // Call the ACTUAL caller again before asking an explicit validator to prove revocation.
    let (restored, diagnostics) =
        diagnostics::capture(|| caller.call(&engine, &loaded, &mut s, &mut px, &mut hpx, token))?;
    let reason = diagnostics::refusal(restored, &mut rx, "snapshot was revoked", diagnostics)?;
    let final_state = Snapshot::capture(&engine, &s, &px, &hpx)?;
    final_state.persist(&bundle, "after-restore")?;
    require(
        before == final_state,
        "RESTORE_MUTATED_CACHE_OR_TOKEN_STATE",
    )?;
    model.rewrite_identity()?; // external identity is restored, not a still-present mismatch
    require(
        model.enter_rewrite_execution(&original).is_err(),
        "OLD_ADMISSION_ORIGIN_REVIVED",
    )?;
    save(&bundle, "restored-error.txt", reason.as_bytes())?;
    println!(
        "NATIVE_WORKER_BOUNDARY_PASS case={case} drift={drift} executed_cases=1 scope=lib-test-only serving_qualified=false"
    );
    Ok(())
}
