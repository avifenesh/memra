//! Re-entry regressions adapted from the independent cbafa2eb drift reproduction.
//! Synthetic receipt fixtures test admission only, never model/GPU qualification.
use super::super::runtime_identity;
use super::*;
use memra_gguf::execution_manifest::*;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::path::PathBuf;

struct Fixture {
    directory: PathBuf,
    admission: RewriteAdmission,
    generation: Arc<ProgramGeneration>,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "rewrite-reentry-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        std::fs::create_dir(directory.join("rewrite-receipts")).unwrap();
        let plan = memra_gguf::model_packs::by_alias("qwen3")
            .unwrap()
            .compile_tiny_plan()
            .unwrap();
        let identity = RewriteIdentity {
            artifact_sha256: "a".repeat(64),
            implementation_sha256: "b".repeat(64),
            numeric_program_sha256: "c".repeat(64),
        };
        let lock = b"CPU-only fixture; no model or GPU qualification";
        std::fs::write(directory.join("artifact.lock"), lock).unwrap();
        let mut index = "rewrite\tplan_sha256\treceipt_sha256\tstatus\n".to_string();
        for rewrite in execution_rewrites(&plan).into_iter().filter(|rewrite| {
            matches!(
                rewrite.surface,
                RewriteSurface::DecodeEager
                    | RewriteSurface::DecodeGraph
                    | RewriteSurface::CarriedPrime
            )
        }) {
            let text = rewrite
                .verify_tokens(&identity.implementation_sha256, &[1], &[1])
                .unwrap()
                .bind_runtime_identity(&identity)
                .unwrap()
                .bind_artifact_lock(lock)
                .to_tsv();
            std::fs::write(
                directory
                    .join("rewrite-receipts")
                    .join(format!("{}.tsv", rewrite.id)),
                &text,
            )
            .unwrap();
            index.push_str(&format!(
                "{}\t{}\t{:x}\tpassed\n",
                rewrite.id,
                rewrite.plan_sha256,
                Sha256::digest(text.as_bytes())
            ));
        }
        std::fs::write(directory.join("rewrite-receipts.tsv"), index).unwrap();
        let admission = RewriteAdmission::Qualified(
            RewriteQualifications::load(&directory, &plan, &identity).unwrap(),
        );
        Self {
            directory,
            admission,
            generation: Arc::default(),
        }
    }

    fn snapshot(&self) -> RewriteExecutionSnapshot {
        RewriteExecutionSnapshot::validated(&self.generation, &self.admission, false, || Ok(()))
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}

fn isolated_child(test: &str) -> bool {
    const CHILD: &str = "REWRITE_REENTRY_TEST_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        return true;
    }
    let module = module_path!().split_once("::").unwrap().1;
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("{module}::{test}"),
            "--test-threads=1",
            "--nocapture",
        ])
        .env(CHILD, test)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Only the parent belongs to the unfiltered suite census. Still require one
    // actual passing child test; zero matches or a child failure can never pass.
    assert!(
        output.status.success()
            && stdout
                .lines()
                .any(|line| line.starts_with("test result: ok. 1 passed; 0 failed; 0 ignored;")),
        "isolated re-entry regression failed or ran no test:\n{stdout}\n{stderr}"
    );
    for line in stdout.lines() {
        if let Some(at) = line.find("REENTRY_") {
            println!("{}", &line[at..]);
        }
    }
    false
}

#[test]
fn retained_qualified_scope_must_refuse_later_environment_drift() {
    if !isolated_child("retained_qualified_scope_must_refuse_later_environment_drift") {
        return;
    }
    // This test runs alone in a disposable process, with no CUDA/background workers.
    unsafe {
        std::env::set_var("MEMRA_FAST", "0");
    }
    let expected = runtime_identity::numeric_environment(std::env::vars_os());
    let boundary_calls = Cell::new(0);
    let boundary = || {
        boundary_calls.set(boundary_calls.get() + 1);
        if runtime_identity::numeric_environment(std::env::vars_os()) == expected {
            Ok(())
        } else {
            Err("numerical environment changed since load".into())
        }
    };
    for surface in [
        RewriteSurface::DecodeGraph,
        RewriteSurface::CarriedPrime,
        RewriteSurface::DecodeEager,
    ] {
        let fixture = Fixture::new();
        let retained = RewriteExecutionSnapshot::validated(
            &fixture.generation,
            &fixture.admission,
            false,
            boundary,
        )
        .unwrap();
        {
            let _scope = retained.enter(&fixture.generation, &(), boundary).unwrap();
            assert!(
                active_execution(&fixture.generation)
                    .unwrap()
                    .allows(surface)
            );
        }
        let previous_calls = boundary_calls.get();
        unsafe {
            std::env::set_var("MEMRA_FAST", "1");
        }
        let mut emitted = false;
        let error = match retained.enter(&fixture.generation, &(), boundary) {
            Ok(_scope) => {
                emitted = true;
                None
            }
            Err(error) => Some(error),
        };
        assert!(
            !emitted,
            "retained qualified token path admitted after environment drift"
        );
        assert_eq!(
            error.as_deref(),
            Some("numerical environment changed since load")
        );
        assert_eq!(boundary_calls.get(), previous_calls + 1);
        assert!(active_execution(&fixture.generation).is_none());
        unsafe {
            std::env::set_var("MEMRA_FAST", "0");
        }
        assert!(
            retained.enter(&fixture.generation, &(), boundary).is_err(),
            "restoring environment revived old origin"
        );
        println!(
            "REENTRY_ENV_REFUSAL_PASS surface={} emitted={emitted}",
            surface.as_str()
        );
    }
}

#[test]
#[cfg(unix)]
fn retained_qualified_scope_must_refuse_later_library_inventory_drift() {
    for surface in [
        RewriteSurface::DecodeGraph,
        RewriteSurface::CarriedPrime,
        RewriteSurface::DecodeEager,
    ] {
        let fixture = Fixture::new();
        let library = fixture.directory.join("mapped-library");
        std::fs::write(&library, b"original").unwrap();
        let inventory = runtime_identity::file_inventory_for_reentry_test(&library);
        let boundary_calls = Cell::new(0);
        let boundary = || {
            boundary_calls.set(boundary_calls.get() + 1);
            inventory()
        };
        let retained = RewriteExecutionSnapshot::validated(
            &fixture.generation,
            &fixture.admission,
            false,
            boundary,
        )
        .unwrap();
        {
            let _scope = retained.enter(&fixture.generation, &(), boundary).unwrap();
            assert!(
                active_execution(&fixture.generation)
                    .unwrap()
                    .allows(surface)
            );
        }
        {
            let _unchanged_resume = retained.enter(&fixture.generation, &(), boundary).unwrap();
        }
        let previous_calls = boundary_calls.get();
        let original = fixture.directory.join("original-library");
        std::fs::rename(&library, &original).unwrap();
        std::fs::write(&library, b"modified").unwrap();
        let error = retained
            .enter(&fixture.generation, &(), boundary)
            .err()
            .expect("library drift admitted retained output");
        assert!(
            error.contains("loaded executable mappings changed"),
            "{error}"
        );
        assert_eq!(boundary_calls.get(), previous_calls + 1);
        assert!(active_execution(&fixture.generation).is_none());
        std::fs::remove_file(&library).unwrap();
        std::fs::rename(&original, &library).unwrap();
        assert!(retained.enter(&fixture.generation, &(), boundary).is_err());
        println!("REENTRY_LIBRARY_REFUSAL_PASS surface={}", surface.as_str());
    }
}

#[test]
fn validator_revocation_cannot_rebase_the_retained_origin() {
    let fixture = Fixture::new();
    let retained = fixture.snapshot();
    let error = retained
        .enter(&fixture.generation, &(), || {
            fixture.generation.revoke();
            Ok(())
        })
        .err()
        .expect("validator silently refreshed the retained origin");
    assert!(error.contains("revoked during boundary validation"));
    assert!(active_execution(&fixture.generation).is_none());
}
