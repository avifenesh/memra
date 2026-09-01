//! Bake the build's git SHA into the binary for the OpenAI `system_fingerprint` field
//! (gap-scan F1): the fingerprint identifies the backend configuration a response was
//! produced by, so determinism claims (`seed`) are checkable across deploys.

fn main() {
    // A checkout moves HEAD without touching this crate, so without these lines cargo
    // caches the build script and serves a STALE fingerprint on freshly rebuilt code
    // (observed 2026-08-29: a fresh dd7f1d11d build answering the previous head).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    if let Ok(head) = std::fs::read_to_string("../../.git/HEAD") {
        if let Some(r) = head.strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=../../.git/{}", r.trim());
        }
    }
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=MEMRA_BUILD_SHA={sha}");
}
