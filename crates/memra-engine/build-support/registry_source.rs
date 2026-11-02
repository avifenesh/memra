//! Adapter for the shared prepared-package registry input receiver.
//! The shared Python implementation owns source/corpus/physical-mode validation.
//! Rust checks the response and derives roles from its producer-validated member map;
//! it does not parse the original TOML manifests or inventory source trees again.

use base64::{Engine, engine::general_purpose::STANDARD};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const CAPSULE: &str = ".memra-package-source.json";
const FRESHNESS: &str = ".memra-source-freshness-required";
const MAX_JSON: usize = 64 * 1024 * 1024;
const MAX_INPUTS: usize = 10_000;
// A request capsule can inventory many dependency files. The registry response
// has its separate, tighter input-owner limit below.
const MAX_METADATA_ENTRIES: usize = 100_000;
const MAX_DIAGNOSTIC: usize = 64 * 1024;
const SCHEMA: &str = "memra-registry-source-inputs-v2";
// Independent shared implementation pin. Update only with its source/control
// review; capsule declarations cannot replace the code that verifies them.
const SHARED_RECEIVER_SHA256: &str =
    "bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39";

type Result<T> = std::result::Result<T, String>;
type DirectoryChain = Vec<(u64, u64, u32)>;
type DecodedReply = (BTreeMap<String, Vec<u8>>, Vec<String>, Value);

fn fail(reason: &str) -> String {
    format!("env audit registry provenance: {reason}")
}

fn relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && !name.chars().any(char::is_control)
        && name.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

fn sha256_shape(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

struct Unique(Value);

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("bounded JSON without duplicate keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| Unique(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Unique, E> {
                Ok(Unique(v.into()))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut values = Vec::new();
                while let Some(Unique(value)) = seq.next_element()? {
                    if values.len() == MAX_METADATA_ENTRIES {
                        return Err(de::Error::custom("JSON sequence exceeds bound"));
                    }
                    values.push(value);
                }
                Ok(Unique(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.len() == MAX_METADATA_ENTRIES {
                        return Err(de::Error::custom("JSON object exceeds bound"));
                    }
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                    let Unique(value) = map.next_value()?;
                    values.insert(key, value);
                }
                Ok(Unique(Value::Object(values)))
            }
        }
        d.deserialize_any(UniqueVisitor)
    }
}

fn json(raw: &[u8]) -> Result<Value> {
    if raw.len() > MAX_JSON {
        return Err(fail("JSON exceeds response bound"));
    }
    let mut parser = serde_json::Deserializer::from_slice(raw);
    let Unique(value) =
        Unique::deserialize(&mut parser).map_err(|e| fail(&format!("invalid JSON: {e}")))?;
    parser
        .end()
        .map_err(|_| fail("extra data after JSON response"))?;
    Ok(value)
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| fail(&format!("missing or non-text {name}")))
}

#[derive(PartialEq, Eq)]
struct MetadataSnapshot {
    bytes: Vec<u8>,
    identity: (u64, u64, u32, u64, i64, i64, i64, i64),
    directories: DirectoryChain,
}

#[cfg(unix)]
fn read_metadata(path: &Path) -> Result<MetadataSnapshot> {
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    fn parent(path: &Path) -> Result<(File, DirectoryChain)> {
        if !path.is_absolute() {
            return Err(fail("metadata path is not absolute"));
        }
        let mut directory = File::open("/").map_err(|_| fail("cannot open metadata root"))?;
        let mut chain = Vec::new();
        for component in path.components() {
            let name = match component {
                Component::RootDir => continue,
                Component::Normal(name) => name,
                _ => return Err(fail("noncanonical metadata ancestor")),
            };
            let name = CString::new(name.as_bytes()).map_err(|_| fail("NUL metadata ancestor"))?;
            // Each descriptor owns one no-follow directory; File closes it on every path.
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err(fail("metadata ancestor is unavailable or an alias"));
            }
            directory = unsafe { File::from_raw_fd(fd) };
            let m = directory
                .metadata()
                .map_err(|_| fail("metadata ancestor stat failed"))?;
            chain.push((m.dev(), m.ino(), m.mode()));
        }
        Ok((directory, chain))
    }
    let parent_path = path
        .parent()
        .ok_or_else(|| fail("metadata parent missing"))?;
    let name = CString::new(
        path.file_name()
            .ok_or_else(|| fail("metadata leaf missing"))?
            .as_bytes(),
    )
    .map_err(|_| fail("NUL metadata leaf"))?;
    let (directory, directories) = parent(parent_path)?;
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(fail("present metadata is unreadable or an alias"));
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    let before = file.metadata().map_err(|_| fail("metadata stat failed"))?;
    if !before.is_file() || before.len() > MAX_JSON as u64 {
        return Err(fail("metadata type or length is invalid"));
    }
    let identity = |m: &std::fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.mode(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    let mut bytes = Vec::new();
    (&mut file)
        .take(MAX_JSON as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| fail("metadata read failed"))?;
    if bytes.len() as u64 != before.len()
        || bytes.len() > MAX_JSON
        || identity(
            &file
                .metadata()
                .map_err(|_| fail("metadata restat failed"))?,
        ) != identity(&before)
    {
        return Err(fail("metadata changed during read"));
    }
    let (fresh, fresh_directories) = parent(parent_path)?;
    let fresh_fd = unsafe {
        libc::openat(
            fresh.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fresh_fd < 0 {
        return Err(fail("metadata disappeared or became an alias"));
    }
    let current = unsafe { File::from_raw_fd(fresh_fd) };
    if fresh_directories != directories
        || identity(
            &current
                .metadata()
                .map_err(|_| fail("fresh metadata stat failed"))?,
        ) != identity(&before)
    {
        return Err(fail("metadata pathname or ancestry changed"));
    }
    Ok(MetadataSnapshot {
        bytes,
        identity: identity(&before),
        directories,
    })
}

#[cfg(not(unix))]
fn read_metadata(_: &Path) -> Result<MetadataSnapshot> {
    Err(fail(
        "prepared source receiver requires its supported Unix profile",
    ))
}

fn present(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(_) => Err(fail("cannot inspect prepared source metadata")),
    }
}

/// Recognize only the original engine member in the fixed workspace layout.
/// A normalized Cargo package has a literal version, so even a package placed
/// under an unrelated `crates/memra-engine` cannot inherit that parent's docs.
/// This reads two ownership manifests, not a source inventory or another parent.
pub(super) fn owns_workspace(manifest_dir: &Path) -> Result<bool> {
    if manifest_dir
        .file_name()
        .is_none_or(|name| name != "memra-engine")
    {
        return Ok(false);
    }
    let Some(crates) = manifest_dir.parent() else {
        return Ok(false);
    };
    if crates.file_name().is_none_or(|name| name != "crates") {
        return Ok(false);
    }
    let Some(root) = crates.parent() else {
        return Ok(false);
    };
    let engine_path = manifest_dir.join("Cargo.toml");
    let root_path = root.join("Cargo.toml");
    if !present(&engine_path)? || !present(&root_path)? {
        return Ok(false);
    }
    let engine_source = read_metadata(&engine_path)?;
    let root_source = read_metadata(&root_path)?;
    let parse = |source: &[u8]| -> Result<toml::Value> {
        let text = std::str::from_utf8(source)
            .map_err(|_| fail("workspace ownership manifest is not UTF-8"))?;
        toml::from_str(text).map_err(|_| fail("workspace ownership manifest is not valid TOML"))
    };
    let engine = parse(&engine_source.bytes)?;
    let workspace_root = parse(&root_source.bytes)?;
    let Some(package) = engine.get("package").and_then(toml::Value::as_table) else {
        return Ok(false);
    };
    if package.get("name").and_then(toml::Value::as_str) != Some("memra-engine")
        || package.contains_key("workspace")
        || engine.get("workspace").is_some()
    {
        return Ok(false);
    }
    let Some(version) = package.get("version").and_then(toml::Value::as_table) else {
        return Ok(false);
    };
    if version.len() != 1 || version.get("workspace").and_then(toml::Value::as_bool) != Some(true) {
        return Ok(false);
    }
    let Some(workspace) = workspace_root
        .get("workspace")
        .and_then(toml::Value::as_table)
    else {
        return Ok(false);
    };
    let Some(members) = workspace.get("members").and_then(toml::Value::as_array) else {
        return Ok(false);
    };
    if !members.iter().all(|m| m.as_str().is_some())
        || !members
            .iter()
            .any(|m| m.as_str() == Some("crates/memra-engine"))
    {
        return Ok(false);
    }
    // The admitted workspace uses explicit members and no exclusions. Do not
    // guess the effect of a different exclusion/glob policy on this owner.
    if workspace
        .get("exclude")
        .is_some_and(|v| v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Ok(false);
    }
    if workspace
        .get("package")
        .and_then(toml::Value::as_table)
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Ok(false);
    }
    if read_metadata(&engine_path)? != engine_source || read_metadata(&root_path)? != root_source {
        return Err(fail("workspace ownership changed during selection"));
    }
    Ok(true)
}

fn bounded_output(mut stream: impl Read, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    stream
        .by_ref()
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| fail("receiver pipe read failed"))?;
    if bytes.len() > limit {
        return Err(fail("receiver output exceeds bound"));
    }
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn receive(helper: &[u8], manifest: &Path) -> Result<Vec<u8>> {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::process::CommandExt;
    // Execute the checked immutable bytes, not a pathname that can be swapped
    // after the hash check. Only this sealed descriptor is inherited explicitly.
    let name = CString::new("memra-registry-receiver").unwrap();
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(fail("cannot create immutable receiver descriptor"));
    }
    let mut sealed = unsafe { File::from_raw_fd(fd) };
    sealed
        .write_all(helper)
        .map_err(|_| fail("cannot copy checked receiver bytes"))?;
    sealed
        .seek(SeekFrom::Start(0))
        .map_err(|_| fail("cannot rewind checked receiver bytes"))?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;
    if unsafe { libc::fcntl(sealed.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
        return Err(fail("cannot seal checked receiver bytes"));
    }
    let receiver_fd = sealed.as_raw_fd();
    // GNU timeout bounds the reviewed Linux receiver, including a helper that
    // closes its pipes without exiting. No owner process or environment is changed.
    let mut command = Command::new("timeout");
    command
        .args(["--kill-after=1s", "60s", "python3", "-I"])
        .arg(format!("/proc/self/fd/{receiver_fd}"))
        .args(["registry-inputs", "--manifest"])
        .arg(manifest)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    unsafe {
        command.pre_exec(move || {
            let flags = libc::fcntl(receiver_fd, libc::F_GETFD);
            if flags < 0 || libc::fcntl(receiver_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|_| fail("present provenance requires the bounded shared receiver"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| fail("receiver stdout missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| fail("receiver stderr missing"))?;
    let out = std::thread::spawn(move || bounded_output(stdout, MAX_JSON));
    let err = std::thread::spawn(move || bounded_output(stderr, MAX_DIAGNOSTIC));
    let status = child.wait().map_err(|_| fail("receiver wait failed"))?;
    let output = out
        .join()
        .map_err(|_| fail("receiver stdout reader failed"))??;
    let diagnostic = err
        .join()
        .map_err(|_| fail("receiver stderr reader failed"))??;
    if !status.success() {
        return Err(fail(&format!(
            "shared receiver refused: {}",
            String::from_utf8_lossy(&diagnostic)
        )));
    }
    Ok(output)
}

#[cfg(not(target_os = "linux"))]
fn receive(_: &[u8], _: &Path) -> Result<Vec<u8>> {
    Err(fail(
        "prepared receiver requires the reviewed Linux descriptor profile",
    ))
}

pub(super) struct PreparedInputs {
    capsule_path: PathBuf,
    snapshot: MetadataSnapshot,
    helper_path: PathBuf,
    helper_snapshot: MetadataSnapshot,
    bodies: BTreeMap<String, Vec<u8>>,
    rust_owners: Vec<String>,
    pub(super) provenance: Value,
}

impl PreparedInputs {
    pub(super) fn text_inputs(&self) -> (&str, Vec<&str>) {
        // Decode validates selected text before this view is exposed.
        (
            std::str::from_utf8(&self.bodies["docs/FLAGS.md"]).unwrap(),
            self.rust_owners
                .iter()
                .map(|name| std::str::from_utf8(&self.bodies[name]).unwrap())
                .collect(),
        )
    }
    pub(super) fn recheck(&self) -> Result<()> {
        if read_metadata(&self.capsule_path)? != self.snapshot {
            return Err(fail("request capsule changed before registry publication"));
        }
        if read_metadata(&self.helper_path)? != self.helper_snapshot {
            return Err(fail(
                "shared receiver file changed before registry publication",
            ));
        }
        Ok(())
    }
}

pub(super) fn acquire(manifest_dir: &Path, package_name: &str) -> Result<Option<PreparedInputs>> {
    let manifest = manifest_dir.join("Cargo.toml");
    let mut candidates = Vec::new();
    if present(&manifest_dir.join(CAPSULE))? {
        candidates.push(manifest_dir.to_path_buf());
    }
    if let Some(vendor) = manifest_dir.parent()
        && vendor.file_name().is_some_and(|n| n == "vendor")
        && let Some(entry) = vendor.parent()
        && present(&entry.join(CAPSULE))?
    {
        candidates.push(entry.to_path_buf());
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    if candidates.len() != 1 {
        return Err(fail("ambiguous entry and vendor dependency provenance"));
    }
    let root = &candidates[0];
    let capsule_path = root.join(CAPSULE);
    let snapshot = read_metadata(&capsule_path)?;
    let cap = json(&snapshot.bytes)?;
    let source_seal = text(&cap, "source_seal")?;
    if !sha256_shape(source_seal) {
        return Err(fail("request source seal is malformed"));
    }
    let roots = cap
        .get("roots")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("request roots missing"))?;
    let packages = cap
        .pointer("/snapshot/payload/packages")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("request packages missing"))?;
    let owners: Vec<&str> = roots
        .iter()
        .filter_map(|(key, path)| {
            let path = path.as_str()?;
            (root.join(path).join("Cargo.toml") == manifest
                && packages.get(key)?.get("name")?.as_str()? == package_name)
                .then_some(key.as_str())
        })
        .collect();
    if owners.len() != 1 || package_name != "memra-engine" {
        return Err(fail("request is not the unique actual engine owner"));
    }
    let helper = root.join("build-support/package_source_identity.py");
    let helper_snapshot = read_metadata(&helper)?;
    if format!("{:x}", Sha256::digest(&helper_snapshot.bytes)) != SHARED_RECEIVER_SHA256 {
        return Err(fail(
            "shared receiver bytes differ from the independently admitted implementation",
        ));
    }
    println!("cargo:rerun-if-changed={}", capsule_path.display());
    println!("cargo:rerun-if-changed={}", root.join(FRESHNESS).display());
    println!("cargo:rerun-if-changed={}", helper.display());
    let raw = receive(&helper_snapshot.bytes, &manifest)?;
    let decoded = decode(&raw, owners[0], source_seal, package_name)?;
    let result = PreparedInputs {
        capsule_path,
        snapshot,
        helper_path: helper,
        helper_snapshot,
        bodies: decoded.0,
        rust_owners: decoded.1,
        provenance: decoded.2,
    };
    result.recheck()?;
    Ok(Some(result))
}

fn decode(
    raw: &[u8],
    expected_owner: &str,
    expected_source_seal: &str,
    engine_name: &str,
) -> Result<DecodedReply> {
    let reply = json(raw)?;
    let object = reply
        .as_object()
        .ok_or_else(|| fail("response is not an object"))?;
    let fields = [
        "schema",
        "owner",
        "source_seal",
        "corpus_sha256",
        "workspace_members",
        "inputs",
        "engine_manifest",
        "qualification",
    ];
    if object.len() != fields.len() || !fields.iter().all(|name| object.contains_key(*name)) {
        return Err(fail("unknown or missing response field"));
    }
    if text(&reply, "schema")? != SCHEMA || reply["qualification"] != Value::Bool(false) {
        return Err(fail("unsupported or qualified response"));
    }
    if text(&reply, "owner")? != expected_owner
        || text(&reply, "source_seal")? != expected_source_seal
    {
        return Err(fail("response differs from request owner/source seal"));
    }
    let corpus = text(&reply, "corpus_sha256")?;
    if !sha256_shape(corpus) {
        return Err(fail("corpus commitment is malformed"));
    }
    let member_values = reply["workspace_members"]
        .as_object()
        .ok_or_else(|| fail("original member map missing"))?;
    let mut members = BTreeSet::new();
    for path in member_values.values() {
        let path = path
            .as_str()
            .ok_or_else(|| fail("member role is not text"))?;
        if !relative_name(path) || !members.insert(path.to_owned()) {
            return Err(fail("invalid or duplicate member role"));
        }
    }
    if members.is_empty() {
        return Err(fail("empty original member map"));
    }
    let engine_member = member_values
        .get(engine_name)
        .and_then(Value::as_str)
        .ok_or_else(|| fail("engine member role missing"))?;
    fn body(encoded: &Value) -> Result<Vec<u8>> {
        let encoded = encoded
            .as_str()
            .ok_or_else(|| fail("input body is not base64 text"))?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| fail("invalid base64 input"))?;
        if STANDARD.encode(&bytes) != encoded {
            return Err(fail("noncanonical base64 input"));
        }
        Ok(bytes)
    }
    let engine_manifest = body(&reply["engine_manifest"])?;
    if engine_manifest.is_empty() || std::str::from_utf8(&engine_manifest).is_err() {
        return Err(fail("actual PACKAGE manifest is not UTF-8 text"));
    }
    let rows = reply["inputs"]
        .as_object()
        .ok_or_else(|| fail("input owner map missing"))?;
    if rows.is_empty() || rows.len() > MAX_INPUTS {
        return Err(fail("input owner count is invalid"));
    }
    let mut bodies = BTreeMap::new();
    let mut rust_owners = Vec::new();
    let mut reached_members = BTreeSet::new();
    for (owner, encoded) in rows {
        if !relative_name(owner) {
            return Err(fail("noncanonical input owner"));
        }
        let bytes = body(encoded)?;
        let selected = if owner.ends_with(".rs") {
            let (member, _) = owner
                .split_once("/src/")
                .ok_or_else(|| fail("Rust input is outside original source role"))?;
            if !members.contains(member) {
                return Err(fail("Rust input belongs to an undeclared member"));
            }
            reached_members.insert(member.to_owned());
            rust_owners.push(owner.clone());
            true
        } else {
            owner == "docs/FLAGS.md" || owner == "Cargo.toml" || owner.ends_with("/Cargo.toml")
        };
        if selected && std::str::from_utf8(&bytes).is_err() {
            return Err(fail("selected Rust/FLAGS/manifest input is not UTF-8"));
        }
        bodies.insert(owner.clone(), bytes);
    }
    if reached_members != members
        || !bodies.contains_key("docs/FLAGS.md")
        || !bodies.contains_key("Cargo.toml")
    {
        return Err(fail("original member/FLAGS/workspace roles are incomplete"));
    }
    if bodies.contains_key(&format!("{engine_member}/Cargo.toml")) {
        return Err(fail(
            "actual engine manifest cannot be substituted by SUPPLEMENTARY",
        ));
    }
    for member in &members {
        if member != engine_member && !bodies.contains_key(&format!("{member}/Cargo.toml")) {
            return Err(fail("original member manifest missing"));
        }
    }
    let provenance = serde_json::json!({"schema": "memra-env-registry-provenance-v1", "owner": expected_owner,
        "shared_receiver_sha256": SHARED_RECEIVER_SHA256,
        "source_seal": expected_source_seal, "corpus_sha256": corpus, "workspace_members": member_values,
        "rust_source_owners": rust_owners, "qualification": false,
        "authority": "shared receiver verifies physical source/corpus; Rust validates response and producer-derived parser/member roles"});
    Ok((bodies, rust_owners, provenance))
}
