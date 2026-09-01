// The build identity, computed by ONE implementation with TWO callers.
//
// `build.rs` pulls this file in with `include!` to BAKE the id at compile time; the crate
// compiles it as a module (`mod build_id;` in lib.rs) so the fingerprint tests can
// RE-DERIVE the id from the working tree and prove the baked value is a function of the
// SOURCE, not of the build environment. A second copy of the algorithm is how a "the gate
// is wired" test starts pinning a copy of the gate instead of the gate itself (see the
// `validate_bind_security` note in `lib.rs`).
//
// Plain `//` and not `//!` on purpose: an inner doc comment is not legal at an `include!`
// expansion site, so the module-level documentation lives on the `mod build_id;`
// declaration in lib.rs instead.
//
// Why content and not git history: the id has to survive a history rewrite. Rewriting
// commits changes every SHA while the bytes of the tree stay put, so an id derived from
// file content still names the same build afterwards, and a `system_fingerprint`
// published before the rewrite keeps meaning what it meant. It also cannot degrade into a
// label: reading a git repo is a thing that can FAIL (it did, silently, inside the
// release container), and hashing files the compiler is already reading cannot.

/// Extensions that define the compiled server: Rust sources, cargo manifests, and the CUDA
/// sources the engine's build script turns into the fatbins linked by `include_bytes!`.
pub(crate) const BUILD_ID_EXTS: &[&str] = &["rs", "toml", "cu", "cuh", "h"];

/// Workspace-root files hashed when present. `Cargo.lock` is TRACKED in this repo, so the
/// exact resolved dependency graph is part of the identity: a dep bump that changes the
/// binary changes the fingerprint even when no first-party line moves.
pub(crate) const BUILD_ID_ROOT_FILES: &[&str] = &["Cargo.toml", "Cargo.lock"];

/// Directory names never walked. `target` is build output (not input) and `.git` is the
/// history the id deliberately does not depend on.
pub(crate) const BUILD_ID_SKIP_DIRS: &[&str] = &["target", ".git"];

/// The rendered id's width in hex characters. 12 matches the `--short=12` git sha the old
/// fingerprint carried, so the field's length is unchanged for anything parsing it.
pub(crate) const BUILD_ID_HEX: usize = 12;

/// Marker for an id derived from the real source tree.
pub(crate) const BUILD_ID_SRC_TREE: &str = "source-tree";
/// Marker for an id that could NOT be derived from source. Always paired with a build
/// `cargo:warning` and a boot WARN naming the reason.
pub(crate) const BUILD_ID_SRC_DEGRADED: &str = "degraded";

const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;

/// FNV-1a 128, folded to `BUILD_ID_HEX` hex digits by `render_build_id`.
///
/// Uniqueness class (build identity), not crypto - the same class as `gen_hex128` in
/// `lib.rs`. It is not a tamper seal and is not meant to be one: anybody who can edit the
/// tree can edit this file too. What it must do is change whenever the compiled bytes
/// change, and be identical for identical input on any machine, which it does.
pub(crate) fn fnv1a128(bytes: &[u8], mut h: u128) -> u128 {
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013B;
    for b in bytes {
        h ^= *b as u128;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Fold a digest to the published id: the TOP `BUILD_ID_HEX * 4` bits, lowercase hex,
/// zero-padded so the field is fixed-width even when the leading nibbles are zero.
pub(crate) fn render_build_id(h: u128) -> String {
    let bits = (BUILD_ID_HEX * 4) as u32;
    format!("{:0width$x}", h >> (128 - bits), width = BUILD_ID_HEX)
}

/// What one scan found: the id, plus the inputs so `build.rs` can tell cargo to re-run the
/// script whenever any of them moves. Without those `rerun-if-changed` lines cargo caches
/// the script's output and the binary reports a STALE identity - two in-range commits of
/// the 2026-08-31 perf-chain lane shipped fingerprints naming the wrong commit exactly
/// that way (`research/perf-chain-20260831/`).
pub(crate) struct BuildIdScan {
    pub(crate) id: String,
    /// Every file hashed, in the order it was hashed.
    pub(crate) files: Vec<std::path::PathBuf>,
    /// Every directory walked. Watched as well as the files, because a directory's mtime
    /// is what changes when a source file is ADDED or REMOVED.
    pub(crate) dirs: Vec<std::path::PathBuf>,
}

/// The workspace root, derived from a crate manifest dir: `<root>/crates/memra-server`.
/// `None` when the layout is not there (a vendored or packaged crate), which is a degraded
/// build, not a panic.
pub(crate) fn workspace_root(manifest_dir: &str) -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(manifest_dir).parent()?.parent()?;
    (root.join("crates").is_dir() && root.join("Cargo.toml").is_file()).then(|| root.to_path_buf())
}

/// Workspace-relative path with forward slashes, so the digest does not depend on WHERE
/// the checkout lives (the release container mounts it at `/src/engine/memra`, a worktree
/// puts it somewhere else again, and both must produce the same id).
fn rel_key(root: &std::path::Path, path: &std::path::Path) -> Option<String> {
    Some(
        path.strip_prefix(root)
            .ok()?
            .to_str()?
            .replace(std::path::MAIN_SEPARATOR, "/"),
    )
}

fn collect(
    dir: &std::path::Path,
    files: &mut Vec<std::path::PathBuf>,
    dirs: &mut Vec<std::path::PathBuf>,
) -> std::io::Result<()> {
    dirs.push(dir.to_path_buf());
    let mut entries: Vec<std::fs::DirEntry> =
        std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        // `DirEntry::file_type` does not follow symlinks, so a symlinked directory is
        // neither dir nor file here and is skipped. Deliberate: following links would make
        // the id depend on what lives OUTSIDE the checkout.
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let name = entry.file_name();
            if BUILD_ID_SKIP_DIRS.contains(&name.to_string_lossy().as_ref()) {
                continue;
            }
            collect(&path, files, dirs)?;
        } else if file_type.is_file()
            && let Some(ext) = path.extension().and_then(|e| e.to_str())
            && BUILD_ID_EXTS.contains(&ext)
        {
            files.push(path);
        }
    }
    Ok(())
}

/// Hash the workspace's compiled inputs into a stable id. `None` when nothing could be
/// read, which the caller must report LOUDLY rather than paper over.
pub(crate) fn content_id(root: &std::path::Path) -> Option<BuildIdScan> {
    let crates = root.join("crates");
    if !crates.is_dir() {
        return None;
    }
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    for name in BUILD_ID_ROOT_FILES {
        let path = root.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    collect(&crates, &mut files, &mut dirs).ok()?;

    // Sort by the RELATIVE path: `read_dir` order is filesystem-dependent, and a digest
    // that depends on it would differ between two clones of the same commit.
    let mut keyed: Vec<(String, std::path::PathBuf)> = files
        .into_iter()
        .filter_map(|p| rel_key(root, &p).map(|k| (k, p)))
        .collect();
    keyed.sort();
    keyed.dedup_by(|a, b| a.0 == b.0);
    if keyed.is_empty() {
        return None;
    }
    dirs.sort();

    let mut h = FNV_OFFSET;
    let mut hashed = Vec::with_capacity(keyed.len());
    for (key, path) in keyed {
        let bytes = std::fs::read(&path).ok()?;
        // Path, then LENGTH, then content: without the length a rename plus a matching
        // content shift could collide two different trees.
        h = fnv1a128(key.as_bytes(), h);
        h = fnv1a128(&(bytes.len() as u64).to_le_bytes(), h);
        h = fnv1a128(&bytes, h);
        hashed.push(path);
    }
    Some(BuildIdScan {
        id: render_build_id(h),
        files: hashed,
        dirs,
    })
}

/// The id every build falls back to when the source tree is unreadable: a digest of the
/// package's own identity. WEAK on purpose - it is the same for every build of a version,
/// which is why it is always accompanied by `BUILD_ID_SRC_DEGRADED` and a WARN. It exists
/// so that "no identity" is representable as a real, shaped value instead of the string
/// `unknown`, which is what the field used to degrade to and what nobody noticed.
pub(crate) fn degraded_build_id(pkg_name: &str, pkg_version: &str) -> String {
    let mut h = FNV_OFFSET;
    h = fnv1a128(pkg_name.as_bytes(), h);
    h = fnv1a128(b"\0", h);
    h = fnv1a128(pkg_version.as_bytes(), h);
    render_build_id(h)
}

/// The documented `system_fingerprint` shape: `memra-<version>-<12 lowercase hex>`.
///
/// This is the assertion the old envelope tests were missing. They checked
/// `starts_with("memra-")`, which `memra-unknown` passes, so the defect that served prod a
/// meaningless fingerprint was inside the tested surface the whole time.
pub(crate) fn fingerprint_is_well_formed(fp: &str) -> bool {
    // Belt and braces: the literal that shipped must never validate again, whatever the
    // rest of the shape check does.
    if fp.contains("unknown") {
        return false;
    }
    let Some((head, id)) = fp.rsplit_once('-') else {
        return false;
    };
    let Some(version) = head.strip_prefix("memra-") else {
        return false;
    };
    // `-` and `+` are allowed so a pre-release or build-metadata version (`0.123.0-rc1`)
    // still validates: the id is taken from the LAST `-`, so the version keeps its own.
    let version_ok = version.starts_with(|c: char| c.is_ascii_digit())
        && version.split('.').count() >= 3
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '+');
    let id_ok = id.len() == BUILD_ID_HEX
        && id
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c));
    version_ok && id_ok
}
