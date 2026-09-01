//! Bake the build's git SHA into the binary for the OpenAI `system_fingerprint` field
//! (gap-scan F1): the fingerprint identifies the backend configuration a response was
//! produced by, so determinism claims (`seed`) are checkable across deploys.

fn git(args: &[&str]) -> Option<String> {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    // Without these lines cargo caches this script's output and skips it whenever a
    // commit touches nothing under crates/memra-server/, so the binary reports the
    // PREVIOUS build's sha — two in-range commits of the 2026-08-31 perf-chain lane
    // shipped fingerprints naming the wrong commit (research/perf-chain-20260831/).
    // HEAD changes on checkout/detach; the ref it points to changes on commit. If the
    // ref is packed the loose file is absent and cargo re-runs every build, which is
    // the safe direction.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(refname) = git(&["symbolic-ref", "-q", "HEAD"])
        && let Some(refpath) = git(&["rev-parse", "--git-path", &refname])
    {
        println!("cargo:rerun-if-changed={refpath}");
    }

    let sha = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=MEMRA_BUILD_SHA={sha}");
}
