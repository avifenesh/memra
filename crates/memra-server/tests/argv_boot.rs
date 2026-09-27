//! memra#617: the stock `memra-server` binary refuses an unknown command-line argument at boot,
//! naming the token, with a non-zero exit and before any device work; every documented
//! argument shape still works. CPU only: nothing here loads a model or opens a device.
//!
//! The shapes come from the parser (`memra_server::argv`, `auth::run_cli`), not from memory:
//! `--version`, `-V`, `--gen-key <tenant> [--lane interactive|batch] [--rate-limit N]
//! [--keys <path>]`, `--revoke-key <prefix> [--keys <path>]`.

use std::path::PathBuf;
use std::process::{Command, Output};

fn server() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_memra-server"));
    // A keyring named by the environment would let `--gen-key` succeed without `--keys`; the
    // tests below pass the path explicitly so the shape under test is the one exercised.
    cmd.env_remove("MEMRA_API_KEYS");
    cmd
}

fn run(args: &[&str]) -> Output {
    server().args(args).output().expect("spawn memra-server")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A refusal is exit 2, names the token, lists the accepted shapes, prints nothing on stdout
/// and never reaches the boot's first line (the build identity), so no device work followed.
fn assert_refused(out: &Output, token: &str) {
    let err = stderr(out);
    assert_eq!(out.status.code(), Some(2), "exit code for {token}: {err}");
    assert!(err.contains("[server] FATAL:"), "{err}");
    assert!(
        err.contains(&format!("{token:?}")),
        "token not named: {err}"
    );
    assert!(
        err.contains("accepted arguments: --version | -V | --gen-key"),
        "{err}"
    );
    assert!(
        !err.contains("[server] build:"),
        "boot proceeded past argv: {err}"
    );
    assert!(stdout(out).is_empty(), "stdout not empty: {}", stdout(out));
}

fn keys_path(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("memra_argv_boot_{}_{tag}.toml", std::process::id()));
    let _ = std::fs::remove_file(&p);
    p
}

#[test]
fn bogus_flag_is_refused_before_boot() {
    let out = run(&["--experts-via-tier"]);
    assert_refused(&out, "--experts-via-tier");
    assert!(
        stderr(&out).contains("unknown argument"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn bogus_flag_beside_a_valid_one_is_refused() {
    let out = run(&["--version", "--bogus"]);
    assert_refused(&out, "--bogus");
}

#[test]
fn positional_token_is_refused() {
    assert_refused(&run(&["serve"]), "serve");
}

#[test]
fn key_modifier_without_a_command_is_refused() {
    let out = run(&["--keys", "/nonexistent/keys.toml"]);
    assert_refused(&out, "--keys");
    assert!(
        stderr(&out).contains("only beside --gen-key or --revoke-key"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn version_flags_are_accepted() {
    for flag in ["--version", "-V"] {
        let out = run(&[flag]);
        assert_eq!(out.status.code(), Some(0), "{flag}: {}", stderr(&out));
        let text = stdout(&out);
        assert!(text.starts_with("memra-server "), "{flag}: {text}");
        assert!(text.contains("system_fingerprint "), "{flag}: {text}");
        assert!(
            !stderr(&out).contains("unknown argument"),
            "{flag}: {}",
            stderr(&out)
        );
    }
}

#[test]
fn key_lifecycle_shapes_are_accepted() {
    let keys = keys_path("lifecycle");
    let keys_s = keys.to_str().unwrap();

    let out = run(&["--gen-key", "acme", "--keys", keys_s]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(stdout(&out).starts_with("mk-acme-"), "{}", stdout(&out));

    let out = run(&[
        "--gen-key",
        "bulk",
        "--lane",
        "batch",
        "--rate-limit",
        "2",
        "--keys",
        keys_s,
    ]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(stdout(&out).starts_with("mk-bulk-"), "{}", stdout(&out));

    let out = run(&["--revoke-key", "mk-bulk-", "--keys", keys_s]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("[revoke-key] mk-bulk-"),
        "{}",
        stderr(&out)
    );

    let _ = std::fs::remove_file(&keys);
}

#[test]
fn gen_key_without_a_keyring_is_run_cli_usage_not_an_argv_refusal() {
    // Admitted by argv; `auth::run_cli` owns this error and its exit code.
    let out = run(&["--gen-key", "acme"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("no keys file"), "{err}");
    assert!(!err.contains("unknown argument"), "{err}");
}
