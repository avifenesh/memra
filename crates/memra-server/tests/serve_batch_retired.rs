//! memra #1158: `MEMRA_SERVE_BATCH` is retired. The stock `memra-server` binary refuses the name
//! at boot, set to anything, before the environment audit and before any model or device work.
//! CPU only: every arm stops at a startup FATAL.
//!
//! The sentinels: each child also carries an empty `MEMRA_METRICS_TOKEN`, a startup FATAL a few
//! lines after the refusal, and a `MEMRA_MODELS` entry whose path does not exist, which the
//! model list refuses before the worker starts. A boot that walks past the retired name dies at
//! the first, so a missing refusal fails this test by its message; the second keeps a boot that
//! somehow passed the first from loading the default models or opening a device.

use std::process::{Command, Output};

fn server() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_memra-server"));
    // The caller's own MEMRA_* names must not steer the child: the arms below set what they test.
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"MEMRA_") {
            cmd.env_remove(&name);
        }
    }
    cmd.env("MEMRA_METRICS_TOKEN", "");
    cmd.env(
        "MEMRA_MODELS",
        "retired-seam=/nonexistent/memra-1158/model.gguf",
    );
    cmd
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_refused(out: &Output, arm: &str) {
    let err = stderr(out);
    assert_eq!(out.status.code(), Some(1), "{arm}: exit code: {err}");
    assert!(
        err.contains("[server] FATAL: MEMRA_SERVE_BATCH is retired"),
        "{arm}: the retired name is not refused by name: {err}"
    );
    assert!(
        err.contains("Removed doors, 2026-11-10"),
        "{arm}: the refusal does not name its ledger: {err}"
    );
    assert!(
        !err.contains("MEMRA_METRICS_TOKEN") && !err.contains("[server] starting"),
        "{arm}: the boot went past the retired name: {err}"
    );
}

#[test]
fn serve_batch_set_to_anything_is_refused_at_boot() {
    // `0` chose the removed scheduler, `1` was the default: both name a seam that is gone.
    for value in ["0", "1", "", "true", "off"] {
        let out = server()
            .env("MEMRA_SERVE_BATCH", value)
            .output()
            .expect("spawn memra-server");
        assert_refused(&out, &format!("MEMRA_SERVE_BATCH={value:?}"));
    }
}

#[test]
fn the_audit_switch_cannot_wave_the_retired_name_through() {
    for audit in ["warn", "0", "off"] {
        let out = server()
            .env("MEMRA_SERVE_BATCH", "0")
            .env("MEMRA_ENV_AUDIT", audit)
            .output()
            .expect("spawn memra-server");
        assert_refused(&out, &format!("MEMRA_ENV_AUDIT={audit}"));
    }
}

/// Non-vacuity: without the name the same boot reaches the sentinel, so the refusal above is
/// keyed on `MEMRA_SERVE_BATCH` and not on something else in the child's environment.
#[test]
fn a_boot_without_the_retired_name_is_not_refused_for_it() {
    let out = server().output().expect("spawn memra-server");
    let err = stderr(&out);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(!err.contains("MEMRA_SERVE_BATCH"), "{err}");
    assert!(
        err.contains("[server] FATAL: MEMRA_METRICS_TOKEN must not be empty"),
        "the sentinel did not stop the boot: {err}"
    );
}
