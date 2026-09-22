//! Bounded, real native caller evidence. Each refusal invocation owns one process,
//! one loaded model and one retained caller. Never run these from a unit-test harness.
use super::*;
use memra_engine::decode::GraphSession;
use memra_engine::plan_backend::RewriteIdentity;
use memra_engine::prime_graph::PrimeGraph;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;

#[path = "capture.rs"]
mod capture;
#[path = "../../../tests/support/env_drift_control.rs"]
mod env_drift_control;
#[path = "selected.rs"]
mod selected;

const SOURCE_PIN: &str = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca";
const GRAPH_BUDGET: usize = 512; // Allocation/capture budget, never a generation benchmark.
const STEPS: usize = 4;
const PRIME_BUCKET: usize = 16;
const SURFACES: [RewriteSurface; 3] = [
    RewriteSurface::DecodeEager,
    RewriteSurface::DecodeGraph,
    RewriteSurface::CarriedPrime,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Full,
    EagerGraph,
}

impl Scope {
    fn surfaces(self) -> &'static [RewriteSurface] {
        match self {
            Self::Full => &SURFACES,
            Self::EagerGraph => &[RewriteSurface::DecodeEager, RewriteSurface::DecodeGraph],
        }
    }

    fn capture_directory(self) -> &'static str {
        match self {
            Self::Full => "retained-capture",
            Self::EagerGraph => "retained-eg-capture",
        }
    }

    fn capture_marker(self) -> &'static str {
        match self {
            Self::Full => "RETAINED_CAPTURE_PASS",
            Self::EagerGraph => "RETAINED_EAGER_GRAPH_CAPTURE_PASS",
        }
    }

    fn caller_marker(self) -> &'static str {
        match self {
            Self::Full => "RETAINED_CALLER_PASS",
            Self::EagerGraph => "RETAINED_EAGER_GRAPH_CALLER_PASS",
        }
    }
}

/// Cross a real split-count boundary inside the captured bucket. Short prompts
/// below the scalar split size make prof_apply a no-op and cannot witness its guard.
fn update_prompt() -> Vec<u32> {
    PROMPTS[2].iter().copied().cycle().take(256).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Api {
    Step,
    Apply,
    Launch,
    Read,
    Prime,
}

impl Api {
    fn name(self) -> &'static str {
        match self {
            Self::Step => "GraphSession::step",
            Self::Apply => "GraphSession::prof_apply",
            Self::Launch => "GraphSession::prof_launch",
            Self::Read => "GraphSession::prof_read",
            Self::Prime => "HybridModel::prime_graph_run",
        }
    }
}

struct Evidence {
    directory: PathBuf,
    log: File,
}

impl Evidence {
    fn new(directory: PathBuf) -> Result<Self> {
        std::fs::create_dir(&directory)?; // Refuse reuse, including an earlier failed capture.
        let log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("events.log"))?;
        Ok(Self { directory, log })
    }

    fn event(&mut self, message: impl AsRef<str>) -> Result<()> {
        println!("{}", message.as_ref());
        writeln!(self.log, "{}", message.as_ref())?;
        self.log.flush()?;
        std::io::stdout().flush()?;
        Ok(())
    }

    fn bytes(&mut self, label: &str, bytes: &[u8]) -> Result<()> {
        let path = self.directory.join(label);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        self.event(format!(
            "RAW file={} bytes={} sha256={}",
            path.display(),
            bytes.len(),
            sha256(bytes)
        ))
    }

    fn logits(&mut self, label: &str, values: &[f32], size: usize) -> Result<()> {
        // Retain even nonfinite or wrong-shape output before refusing qualification.
        self.bytes(
            &format!("{label}.f32"),
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        require(
            values.len() == size && values.iter().all(|x| x.is_finite()),
            format!("INVALID_OUTPUT {label}"),
        )
    }

    fn tokens(&mut self, label: &str, values: &[u32]) -> Result<()> {
        self.bytes(
            &format!("{label}.u32"),
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.event(format!("TOKENS {label} {values:?}"))
    }
}

fn source_pin(source: &Path, evidence: &mut Evidence) -> Result<()> {
    require(
        source.is_dir(),
        "RETAINED_REQUIRES_OFFICIAL_SAFETENSORS_DIRECTORY",
    )?;
    // Independent pins from the existing native runner's immutable source manifest.
    // Hash the real bytes, not a caller-supplied identity or a model's marketing name.
    let files = [
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
        (
            "LICENSE",
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        ),
    ];
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_str().ok_or("SOURCE_FILENAME_NOT_UTF8")?;
        require(
            files.iter().any(|(known, _)| *known == name)
                || matches!(name, "qualification-source.json" | ".cache"),
            format!("UNPINNED_SOURCE_ENTRY {name}"),
        )?;
    }
    evidence.event(format!("SOURCE pin={SOURCE_PIN}"))?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    for (name, expected) in files {
        let mut file = File::open(source.join(name))?;
        let mut hash = Sha256::new();
        let mut size = 0_u64;
        loop {
            let len = file.read(&mut buffer)?;
            if len == 0 {
                break;
            }
            hash.update(&buffer[..len]);
            size += len as u64;
        }
        let actual = format!("{:x}", hash.finalize());
        evidence.event(format!(
            "SOURCE_FILE name={name} bytes={size} sha256={actual}"
        ))?;
        require(actual == expected, format!("PINNED_SOURCE_MISMATCH {name}"))?;
    }
    Ok(())
}

fn qualified(model: &HybridModel, scope: Scope) -> Result<()> {
    require(model.rewrite_is_qualified(), "RETAINED_NOT_QUALIFIED")?;
    for rewrite in execution_rewrites(&model.plan) {
        require(
            model.rewrite_allowed(rewrite.surface) == scope.surfaces().contains(&rewrite.surface),
            format!("RETAINED_ADMISSION_MASK {}", rewrite.id),
        )?;
    }
    Ok(())
}

fn policy(model: &HybridModel) -> Result<RewriteParityPolicy> {
    let pack = memra_gguf::model_packs::for_config(&model.cfg).ok_or("PACK_UNAVAILABLE")?;
    require(pack.family == "qwen3", "RETAINED_REQUIRES_QWEN3")?;
    let tolerance = pack
        .checkpoint_parity
        .ok_or("CHECKPOINT_PARITY_UNAVAILABLE")?;
    require(
        tolerance.max_abs.is_finite()
            && tolerance.max_abs >= 0.0
            && tolerance.max_rel.is_finite()
            && tolerance.max_rel >= 0.0,
        "INVALID_PACK_TOLERANCE",
    )?;
    Ok(RewriteParityPolicy {
        max_abs: tolerance.max_abs,
        max_rel: tolerance.max_rel,
        require_argmax: true,
    })
}

fn manifest(model: &HybridModel, surface: RewriteSurface) -> Result<ExecutionRewrite> {
    let rewrite = execution_rewrites(&model.plan)
        .into_iter()
        .find(|r| r.surface == surface)
        .ok_or("REWRITE_MISSING")?;
    require(
        rewrite.eligible(),
        format!("INELIGIBLE {} {:?}", rewrite.id, rewrite.blockers),
    )?;
    Ok(rewrite)
}

fn seal(identity: &RewriteIdentity, index: &[u8], lock: &[u8], scope: Scope) -> String {
    format!(
        "{}-v1\nsource={SOURCE_PIN}\nartifact_sha256={}\nimplementation_sha256={}\nnumeric_program_sha256={}\nindex_sha256={}\nartifact_lock_sha256={}\n",
        scope.capture_directory(),
        identity.artifact_sha256,
        identity.implementation_sha256,
        identity.numeric_program_sha256,
        sha256(index),
        sha256(lock)
    )
}

pub(super) fn run(mode: &str, source: &Path, bundle: &Path) -> Result<()> {
    let (scope, action) = if let Some(action) = mode.strip_prefix("retained-eg-") {
        (Scope::EagerGraph, action)
    } else {
        (
            Scope::Full,
            mode.strip_prefix("retained-")
                .ok_or("RETAINED_MODE_INVALID")?,
        )
    };
    let capture_mode = action == "capture";
    let selected_case = scope == Scope::EagerGraph
        && matches!(
            action,
            "prime-api-refusal" | "prime-production-refusal" | "prime-fallback"
        );
    let case = if capture_mode || selected_case {
        None
    } else {
        let (api, drift) = action.rsplit_once('-').ok_or("RETAINED_MODE_INVALID")?;
        let api = match api {
            "step" => Api::Step,
            "prof-apply" => Api::Apply,
            "prof-launch" => Api::Launch,
            "prof-read" => Api::Read,
            "prime-run" => Api::Prime,
            _ => return Err("RETAINED_API_INVALID".into()),
        };
        require(
            scope == Scope::Full || api != Api::Prime,
            "EAGER_GRAPH_SCOPE_EXCLUDES_POSITIVE_PRIME",
        )?;
        require(
            matches!(drift, "library" | "environment"),
            "RETAINED_DRIFT_INVALID",
        )?;
        Some((api, drift))
    };
    require(cfg!(target_os = "linux"), "RETAINED_REQUIRES_LINUX")?;
    require(
        std::env::var_os("DOCS_RS").is_none(),
        "DOCS_RS_IS_NOT_NATIVE_QUALIFICATION",
    )?;
    require(
        std::env::var("MEMRA_FAST").as_deref() == Ok("0"),
        "RETAINED_REQUIRES_MEMRA_FAST_0_AT_EXEC",
    )?;
    let configured_bundle = std::env::var_os("MEMRA_REWRITE_BUNDLE");
    require(
        capture_mode == configured_bundle.is_none(),
        "RETAINED_BUNDLE_ENV_MISMATCH",
    )?;
    if let Some(path) = configured_bundle {
        require(
            std::fs::canonicalize(path)? == std::fs::canonicalize(bundle)?,
            "RETAINED_BUNDLE_PATH_MISMATCH",
        )?;
    }
    let lock_path = std::env::var_os("MEMRA_ARTIFACT_LOCK").ok_or("ARTIFACT_LOCK_REQUIRED")?;
    let lock = std::fs::read(lock_path)?;
    require(
        !lock.is_empty() && lock == std::fs::read(bundle.join("artifact.lock"))?,
        "ARTIFACT_LOCK_MISMATCH",
    )?;
    if capture_mode {
        require(
            !bundle.join("rewrite-receipts.tsv").exists()
                && !bundle.join("rewrite-receipts").exists()
                && !bundle.join("outputs").exists(),
            "RETAINED_CAPTURE_REQUIRES_SEPARATE_UNUSED_BUNDLE",
        )?;
    }
    let directory = if capture_mode {
        bundle.join(scope.capture_directory())
    } else {
        bundle.join(format!("{mode}-{}", std::process::id()))
    };
    let mut evidence = Evidence::new(directory)?;
    let result = (|| -> Result<()> {
        evidence.event(format!("SCOPE mode={mode} native-caller-only support_promotion=false performance=not-measured pid={}", std::process::id()))?;
        evidence.bytes("artifact.lock", &lock)?;
        source_pin(source, &mut evidence)?;
        let engine = Engine::new(0)?;
        let src = SafetensorsSource::open(source)?;
        let mut model = HybridModel::load_from_source(&engine, &src)?;
        require(model.devices() == [0], "RETAINED_SINGLE_DEVICE_ONLY")?;
        for prompt in PROMPTS {
            require(
                prompt.iter().all(|&token| token < model.cfg.n_vocab),
                "PROMPT_OUT_OF_VOCABULARY",
            )?;
        }
        let identity = model.rewrite_identity()?.clone();
        evidence.bytes("identity.txt", format!("{identity:?}\n").as_bytes())?;
        evidence.event(format!("POLICY {:?}", policy(&model)?))?;
        evidence.event(format!(
            "QUALIFICATION_SCOPE {scope:?} surfaces={:?} carried_prime_selected={}",
            scope.surfaces(),
            scope == Scope::Full
        ))?;
        if capture_mode {
            return capture::run(
                &engine,
                &mut model,
                bundle,
                &lock,
                &identity,
                &mut evidence,
                scope,
            );
        }
        qualified(&model, scope)?;
        let index = std::fs::read(bundle.join("rewrite-receipts.tsv"))?;
        require(
            std::fs::read_to_string(bundle.join(scope.capture_directory()).join("PASS"))?
                == seal(&identity, &index, &lock, scope),
            "RETAINED_CAPTURE_SEAL_MISMATCH",
        )?;
        if selected_case {
            return match action {
                "prime-api-refusal" => selected::prime_api_refusal(&engine, &model, &mut evidence),
                "prime-production-refusal" => {
                    selected::prime_production_refusal(&engine, &mut model, &mut evidence)
                }
                "prime-fallback" => selected::prime_fallback(&engine, &model, &mut evidence),
                _ => unreachable!("validated selected case"),
            };
        }
        let (api, drift) = case.ok_or("RETAINED_CASE_MISSING")?;
        probe(&engine, &model, bundle, api, drift, &mut evidence)?;
        evidence.event(format!(
            "{} mode={mode} api={} receipt_emitted=false support_promotion=false",
            scope.caller_marker(),
            api.name()
        ))
    })();
    if let Err(error) = &result {
        evidence.event(format!("RETAINED_FAIL mode={mode} error={error}"))?;
    }
    result
}

enum Caller {
    Graph(Box<GraphSession>),
    Prime(Box<PrimeGraph>, Box<Cache>),
}

impl Caller {
    // This is the only call between drift installation and the first observation.
    // Never put an outer guard, validator, snapshot refresh or reinstall here.
    fn call(&mut self, e: &Engine, model: &HybridModel, api: Api) -> Result<Option<u32>> {
        match (self, api) {
            (Self::Graph(s), Api::Step) => s.step(e, model).map(Some),
            (Self::Graph(s), Api::Apply) => s.prof_apply(e, model).map(|()| None),
            (Self::Graph(s), Api::Launch) => s.prof_launch(model).map(|()| None),
            (Self::Graph(s), Api::Read) => s.prof_read(e, model).map(Some),
            (Self::Prime(pg, cache), Api::Prime) => model
                .prime_graph_run(e, pg, PROMPTS[1], cache)
                .map(|(logits, _)| Some(memra_engine::forward::argmax(&logits) as u32)),
            _ => Err("CALLER_API_MISMATCH".into()),
        }
    }

    fn state(&self, e: &Engine, label: &str, evidence: &mut Evidence) -> Result<String> {
        e.stream().synchronize()?; // A launch submitted before SIGSTOP must finish first.
        let state = match self {
            Self::Graph(s) => format!(
                "cache={} pos={} token={} device_pos={:?} captures={} bucket={} split_arguments={:?}",
                cache_sha256(e, &s.cache)?,
                s.cache.pos,
                e.dtoh_u32_one(&s.gs.token_d)?,
                e.dtoh_i32(&s.gs.pos_d)?,
                s.gs.captures,
                s.bucket_max,
                s.diagnostic_split_arguments()?
            ),
            Self::Prime(pg, cache) => {
                let io = pg.diagnostic_io_bytes(e)?;
                evidence.bytes(&format!("{label}-prime-io.bin"), &io)?;
                format!(
                    "session={} scratch={} bucket={} io={}",
                    cache_sha256(e, cache)?,
                    cache_sha256(e, pg.scratch())?,
                    pg.bucket,
                    sha256(&io)
                )
            }
        };
        evidence.bytes(&format!("{label}-state.txt"), state.as_bytes())?;
        Ok(state)
    }
}

fn positive(e: &Engine, model: &HybridModel, api: Api, evidence: &mut Evidence) -> Result<Caller> {
    if api == Api::Prime {
        let mut pg = model.prime_graph_new(e, PRIME_BUCKET)?;
        // Independent observations on actual loaded weights; same path as qualification.
        capture::prime_rows(e, model, &mut pg, 0, "positive", evidence)?;
        let fresh = Cache::new(e, &model.cfg, PRIME_BUCKET + STEPS + 8)?;
        require(fresh.pos == 0, "PRIME_PROBE_CACHE_NOT_FRESH")?;
        evidence.event("POSITIVE_PASS api=HybridModel::prime_graph_run fresh_probe_cache=true")?;
        return Ok(Caller::Prime(Box::new(pg), Box::new(fresh)));
    }
    let prompt = update_prompt();
    let reference = capture::eager_tokens(e, model, &prompt, "positive-reference", evidence)?;
    let (mut session, first) = model.graph_session_new(e, &prompt, GRAPH_BUDGET)?;
    require(first == reference[0], "POSITIVE_GRAPH_FIRST_TOKEN_MISMATCH")?;
    let next = if api == Api::Step {
        session.step(e, model)?
    } else {
        let before = session.diagnostic_split_arguments()?;
        let cache_before = cache_sha256(e, &session.cache)?;
        session.prof_apply(e, model)?;
        let after = session.diagnostic_split_arguments()?;
        evidence.event(format!("POSITIVE_APPLY before={before:?} after={after:?}"))?;
        require(
            cache_before == cache_sha256(e, &session.cache)?,
            "POSITIVE_APPLY_MUTATED_CACHE",
        )?;
        if api == Api::Apply {
            require(
                !before.is_empty() && before != after,
                "PROF_APPLY_NO_ACTUAL_UPDATE_WITNESS",
            )?;
        }
        session.prof_launch(model)?;
        session.prof_read(e, model)?
    };
    evidence.tokens("positive-candidate", &[first, next])?;
    require(
        next == reference[1] && session.cache.pos == prompt.len() + 1,
        "POSITIVE_GRAPH_TOKEN_OR_POSITION_MISMATCH",
    )?;
    // Prepare a valid next phase, while the external identity is still unchanged.
    match api {
        Api::Apply => {
            let applied = session.diagnostic_split_arguments()?;
            // Real recapture, not a fabricated plan or counter. The origin is retained.
            model.graph_session_recapture_pub(e, &mut session)?;
            require(
                session.diagnostic_split_arguments()? != applied,
                "PROF_APPLY_RECAPTURE_HAS_NO_PENDING_UPDATE",
            )?;
        }
        Api::Launch => session.prof_apply(e, model)?,
        Api::Read => {
            session.prof_apply(e, model)?;
            session.prof_launch(model)?;
            e.stream().synchronize()?;
            require(
                e.dtoh_u32_one(&session.gs.token_d)? == reference[2],
                "POSITIVE_PENDING_READ_TOKEN_MISMATCH",
            )?;
        }
        Api::Step => (),
        Api::Prime => unreachable!(),
    }
    e.stream().synchronize()?;
    evidence.event(format!(
        "POSITIVE_PASS api={} all_prior_guards_ended=true",
        api.name()
    ))?;
    Ok(Caller::Graph(Box::new(session)))
}

struct Drift {
    #[cfg(unix)]
    mapping: Option<memmap2::Mmap>,
    #[cfg(unix)]
    file: Option<super::library_drift::PrivateFile>,
    environment: bool,
}

impl Drift {
    fn install(kind: &str, bundle: &Path, evidence: &mut Evidence) -> Result<Self> {
        let mut drift = Self {
            #[cfg(unix)]
            mapping: None,
            #[cfg(unix)]
            file: None,
            environment: kind == "environment",
        };
        if drift.environment {
            env_drift_control::rendezvous("mutate")?;
        } else {
            #[cfg(unix)]
            {
                let file = super::library_drift::PrivateFile::new(bundle)?;
                let reader = File::open(&file.path)?;
                // SAFETY: the private file's writer is closed; the owner keeps its name
                // and bytes stable until after unmapping. These bytes are NEVER executed.
                let mapping = unsafe { memmap2::MmapOptions::new().map_exec(&reader)? };
                let maps = file.maps()?;
                require(
                    maps.len() == 1
                        && maps[0]
                            .split_whitespace()
                            .nth(1)
                            .is_some_and(|p| p.contains('x') && !p.contains('w')),
                    "DRIFT_MAP_NOT_READ_EXECUTE",
                )?;
                evidence.event(format!(
                    "LIBRARY_DRIFT_MAPPING path={} sha256={} maps={maps:?}",
                    file.path.display(),
                    sha256(&mapping)
                ))?;
                drift.mapping = Some(mapping);
                drift.file = Some(file);
            }
            #[cfg(not(unix))]
            return Err("LIBRARY_DRIFT_REQUIRES_LINUX".into());
        }
        Ok(drift)
    }

    fn expected_reason(&self, error: &str) -> bool {
        if self.environment {
            error.contains("numerical environment changed since load")
                && error.contains("MEMRA_FAST")
        } else {
            #[cfg(unix)]
            {
                error.contains("loaded executable mappings changed")
                    && self
                        .file
                        .as_ref()
                        .is_some_and(|f| error.contains(&f.path.to_string_lossy().to_string()))
            }
            #[cfg(not(unix))]
            {
                false
            }
        }
    }

    fn restore(&mut self) -> Result<()> {
        if self.environment {
            env_drift_control::rendezvous("restore")?;
        } else {
            #[cfg(unix)]
            {
                self.mapping.take();
                require(
                    self.file
                        .as_ref()
                        .ok_or("DRIFT_FILE_MISSING")?
                        .maps()?
                        .is_empty(),
                    "DRIFT_UNMAP_FAILED",
                )?;
            }
        }
        Ok(())
    }
}

fn probe(
    e: &Engine,
    model: &HybridModel,
    bundle: &Path,
    api: Api,
    kind: &str,
    evidence: &mut Evidence,
) -> Result<()> {
    let origin = model.rewrite_execution_snapshot()?;
    let identity = model.rewrite_identity()?.clone();
    let mut caller = positive(e, model, api, evidence)?;
    // No guard survives this boundary. The diagnostic copies also synchronize the stream.
    let before = caller.state(e, "before", evidence)?;
    e.ctx().synchronize()?;
    let mut drift = Drift::install(kind, bundle, evidence)?;
    let attempt = caller.call(e, model, api); // FIRST identity/admission operation after drift.
    let observed = (|| -> Result<String> {
        evidence.event(format!("DRIFT_CALL api={} result={attempt:?}", api.name()))?;
        caller.state(e, "drift", evidence)
    })();
    // Restore external state even for an unexpected success or a diagnostic failure.
    e.ctx().synchronize()?;
    let restored = drift.restore();
    require(
        restored.is_ok(),
        format!("EXTERNAL_RESTORE_FAILED {restored:?}; attempt={attempt:?}"),
    )?;
    let after = observed?;
    let error = attempt.err().ok_or("DRIFT_API_SUCCEEDED")?.to_string();
    require(
        drift.expected_reason(&error),
        format!("DRIFT_WRONG_REASON api={} error={error}", api.name()),
    )?;
    require(
        before == after,
        format!("DRIFT_MUTATED_STATE api={}", api.name()),
    )?;
    // The same named caller must still refuse after restore, without help from a
    // standalone validator, fresh snapshot or bundle reinstall.
    let retry = caller.call(e, model, api);
    evidence.event(format!("RESTORED_CALL api={} result={retry:?}", api.name()))?;
    require(
        retry
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string().contains("snapshot was revoked")),
        "RESTORE_REVIVED_CALLER_OR_WRONG_REASON",
    )?;
    require(
        caller.state(e, "restored", evidence)? == before,
        "RESTORE_MUTATED_STATE",
    )?;
    match model.enter_rewrite_execution(&origin) {
        Ok(_) => return Err("RESTORE_REVIVED_OLD_ORIGIN".into()),
        Err(error) => require(
            error.contains("snapshot was revoked"),
            format!("OLD_ORIGIN_WRONG_REASON {error}"),
        )?,
    }
    require(
        model.rewrite_identity()? == &identity,
        "EXTERNAL_IDENTITY_NOT_RESTORED",
    )?;
    evidence.event(format!("REFUSAL_PASS api={} drift={kind} state_sha256={} restored_external_identity=true old_origin_revoked=true", api.name(), sha256(before.as_bytes())))
}

#[cfg(test)]
mod selected_scope_tests {
    use super::*;

    #[test]
    fn selected_scope_withholds_prime_and_cannot_borrow_full_scope_seal() {
        assert_eq!(
            Scope::EagerGraph.surfaces(),
            &[RewriteSurface::DecodeEager, RewriteSurface::DecodeGraph]
        );
        assert!(
            !Scope::EagerGraph
                .surfaces()
                .contains(&RewriteSurface::CarriedPrime)
        );
        assert!(
            Scope::Full
                .surfaces()
                .contains(&RewriteSurface::CarriedPrime)
        );
        let identity = RewriteIdentity {
            artifact_sha256: "a".repeat(64),
            implementation_sha256: "b".repeat(64),
            numeric_program_sha256: "c".repeat(64),
        };
        let old = seal(&identity, b"same index", b"same lock", Scope::Full);
        let selected = seal(&identity, b"same index", b"same lock", Scope::EagerGraph);
        assert!(old.starts_with("retained-capture-v1\n"));
        assert!(selected.starts_with("retained-eg-capture-v1\n"));
        assert_ne!(old, selected);
    }

    #[test]
    fn selected_scope_rejects_positive_prime_modes_before_source_or_gpu_work() {
        for drift in ["library", "environment"] {
            let error = run(
                &format!("retained-eg-prime-run-{drift}"),
                Path::new("missing-source"),
                Path::new("missing-bundle"),
            )
            .expect_err("selected scope cannot dispatch positive prime");
            assert!(
                error
                    .to_string()
                    .contains("EAGER_GRAPH_SCOPE_EXCLUDES_POSITIVE_PRIME")
            );
        }
    }
}
