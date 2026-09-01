//! Bake a REAL build identity into the binary for the OpenAI `system_fingerprint` field
//! (gap-scan F1): the fingerprint identifies the backend configuration a response was
//! produced by, so determinism claims (`seed`) are checkable across deploys.
//!
//! WHAT THIS FILE USED TO DO, AND WHY IT WAS WRONG. The identity was
//! `git rev-parse --short=12 HEAD`, with any git failure swallowed into the literal
//! `"unknown"`. Reading a git repo is a thing that fails, and it failed in exactly the
//! place that matters: darklanes' release container (`serving/build-artifact.sh`) compiles
//! as root over a uid-1000 read-only mount, so before the `safe.directory` line landed
//! there on 2026-08-30 21:33 git aborted with "detected dubious ownership" and this script
//! baked `unknown`. The 03:30 build-script output of that same day
//! (`serving/dist/target/release/build/memra-server-*/output` =
//! `MEMRA_BUILD_SHA=unknown`) is the binary the fleet was serving, and prod answered
//! `system_fingerprint: memra-unknown` to every customer request for a whole deploy
//! generation. Nothing warned, at build time or at boot, because a successful build with a
//! meaningless label looks exactly like a successful build.
//!
//! WHAT IT DOES NOW. The identity is `<crate version>-<content id>`, where the content id
//! is a digest of the workspace's compiled inputs computed by this script itself
//! (`src/build_id.rs`). Two properties the git sha did not have:
//!
//! - It CANNOT degrade to a label. Hashing files the compiler is about to read cannot fail
//!   in an environment where the compile succeeds; and if the tree somehow is not there,
//!   the fallback is still a shaped id, plus a `cargo:warning` here and a WARN at boot.
//! - It survives a HISTORY REWRITE. Rewriting commits changes every SHA while the bytes of
//!   the tree stay put, so a fingerprint published before the rewrite still names the same
//!   build after it. A git sha baked into a shipped binary becomes a dangling reference the
//!   moment history moves.
//!
//! The git sha is still baked, as an EXTRA provenance field (`MEMRA_BUILD_SHA`), never as
//! the identity. It is allowed to read `unknown` there, because there it is honest.

#[allow(dead_code)] // one implementation, two callers: each uses a subset.
mod build_id {
    include!("src/build_id.rs");
}
use build_id::{
    BUILD_ID_SRC_DEGRADED, BUILD_ID_SRC_TREE, content_id, degraded_build_id, workspace_root,
};

fn git(args: &[&str]) -> Option<String> {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Watch the refs behind `HEAD` so the EXTRA git-sha field cannot go stale. The content id
/// is kept fresh by the per-file watches emitted in `main`; these two lines only cover the
/// sha. HEAD changes on checkout/detach; the ref it points to changes on commit. In a
/// submodule checkout (how darklanes builds) HEAD is always detached, so `symbolic-ref`
/// fails and only the first watch arms - which is the safe direction.
fn watch_git_head() {
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(refname) = git(&["symbolic-ref", "-q", "HEAD"])
        && let Some(refpath) = git(&["rev-parse", "--git-path", &refname])
    {
        println!("cargo:rerun-if-changed={refpath}");
    }
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();

    watch_git_head();

    let scan = workspace_root(&manifest_dir)
        .as_deref()
        .and_then(content_id);
    let (build_id, id_src, note) = match scan {
        Some(scan) => {
            // Files AND directories: a file's mtime covers edits, a directory's covers a
            // source file being added or removed.
            for dir in &scan.dirs {
                println!("cargo:rerun-if-changed={}", dir.display());
            }
            for file in &scan.files {
                println!("cargo:rerun-if-changed={}", file.display());
            }
            (scan.id, BUILD_ID_SRC_TREE, String::new())
        }
        None => {
            let reason = format!(
                "workspace source tree not readable from CARGO_MANIFEST_DIR={manifest_dir} \
                 (expected <root>/crates alongside <root>/Cargo.toml)"
            );
            // Loud at BUILD time as well as at boot. The whole defect this replaces was a
            // silent failure, and a diagnostic that is itself silent is not a diagnostic.
            println!(
                "cargo:warning=memra build identity DEGRADED: {reason}. system_fingerprint \
                 will carry a version-only id that does not identify this build's source; \
                 published performance pins cannot be verified against it."
            );
            (
                degraded_build_id(&pkg_name, &pkg_version),
                BUILD_ID_SRC_DEGRADED,
                reason,
            )
        }
    };

    // The identity. `MEMRA_BUILD_ID` is never empty and never the string `unknown`.
    println!("cargo:rustc-env=MEMRA_BUILD_ID={build_id}");
    println!("cargo:rustc-env=MEMRA_BUILD_ID_SRC={id_src}");
    println!("cargo:rustc-env=MEMRA_BUILD_ID_NOTE={note}");

    // Extra provenance only. `unknown` here is a fact about the build environment, not a
    // hole in the customer-visible fingerprint.
    let sha = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=MEMRA_BUILD_SHA={sha}");
}
