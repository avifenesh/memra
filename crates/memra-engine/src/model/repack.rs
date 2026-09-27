//! CPU-only native NVFP4 disk repacking. Strict identity loads derive their bytes from the
//! opened source, never from a size-only cache hit. HostBuf retains the returned map AND file.
use memra_gguf::source::{Nvfp4StackedNative, TensorSource};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

#[cfg(all(test, unix))]
#[path = "repack/tests.rs"]
mod tests;

pub(super) struct MappedRepack {
    pub map: Arc<memmap2::Mmap>,
    pub file: Arc<File>,
}

/// Bound producers own both input selection and output authority. Never downgrade an error
/// into a legacy named-cache lookup, and check the engine's expected extent before installation.
pub(super) fn load_bound(
    source: &dyn TensorSource,
    name: &str,
    expected_bytes: usize,
) -> io::Result<Option<memra_gguf::bound_disk::BoundDiskView>> {
    let Some(view) = source
        .try_canonical_nvfp4_bank(name)
        .map_err(io::Error::other)?
    else {
        return Ok(None);
    };
    if view.len() != expected_bytes {
        return Err(invalid(format!(
            "bound canonical bank {name} has {} bytes, expected {expected_bytes}",
            view.len()
        )));
    }
    Ok(Some(view))
}

fn strict_identity_requested() -> bool {
    // Match runtime_identity::identity_requested, including empty/non-Unicode values.
    // Keep this loader independent of the CUDA/admission modules for host qualification.
    std::env::var_os("MEMRA_ARTIFACT_LOCK").is_some()
        || std::env::var_os("MEMRA_REWRITE_BUNDLE").is_some()
}

pub(super) fn load_stacked(
    cache: &Path,
    bank: &Nvfp4StackedNative<'_>,
) -> io::Result<MappedRepack> {
    let (n, out_f, in_f) = (bank.n_expert, bank.out_f, bank.in_f);
    let total = repack_len(n, out_f, in_f)?;
    let codes = out_f * (in_f / 2);
    let scales = out_f * (in_f / 16);
    if bank.codes.len() != n * codes || bank.scales.len() != n * scales {
        return Err(invalid("stacked NVFP4 source extent mismatch"));
    }
    load(cache, total, |out| {
        for expert in 0..n {
            out.write_all(&memra_gguf::nvfp4_repack::repack_modelopt_to_gguf(
                &bank.codes[expert * codes..(expert + 1) * codes],
                &bank.scales[expert * scales..(expert + 1) * scales],
                out_f,
                in_f,
            ))?;
        }
        Ok(())
    })
}

pub(super) fn load_experts(
    cache: &Path,
    src: &dyn TensorSource,
    layer: u32,
    projection: &str,
    n: usize,
    out_f: usize,
    in_f: usize,
) -> io::Result<MappedRepack> {
    let total = repack_len(n, out_f, in_f)?;
    load(cache, total, |out| {
        // One expert at a time, including owned unswizzled scales. Never collect a bank in RAM.
        for expert in 0..n {
            let name = format!("blk.{layer}.ffn_{projection}_exps.{expert}.weight");
            let nv = src
                .try_find_nvfp4_native(&name)
                .map_err(io::Error::other)?
                .ok_or_else(|| invalid(format!("expert {name} lost NVFP4-native mid-gather")))?;
            if (nv.out_f, nv.in_f) != (out_f, in_f)
                || nv.wbytes.len() != out_f * (in_f / 2)
                || nv.wscale.len() != out_f * (in_f / 16)
            {
                return Err(invalid(format!(
                    "expert {name} NVFP4 source extent mismatch"
                )));
            }
            out.write_all(&memra_gguf::nvfp4_repack::repack_modelopt_to_gguf(
                nv.wbytes, &nv.wscale, out_f, in_f,
            ))?;
        }
        Ok(())
    })
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn repack_len(n: usize, out_f: usize, in_f: usize) -> io::Result<usize> {
    if n == 0 || out_f == 0 || in_f == 0 || !in_f.is_multiple_of(64) {
        return Err(invalid("NVFP4 bank must be nonempty and 64-aligned"));
    }
    n.checked_mul(out_f)
        .and_then(|rows| rows.checked_mul(in_f / 64))
        .and_then(|blocks| blocks.checked_mul(36))
        .ok_or_else(|| invalid("NVFP4 repack size overflow"))
}

fn load<F>(cache: &Path, total: usize, write: F) -> io::Result<MappedRepack>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let file = if strict_identity_requested() {
        strict_file(cache, total, write)?
    } else {
        // Compatibility: legacy loads retain the existing size-only cache and shared mapping.
        if !repack_cache_is_fresh(cache, total) {
            write_repack_cache(cache, write)?;
        }
        open_repack_cache(cache, false)?
    };
    let file = Arc::new(file);
    // Strict: the writer is closed and the inode is unlinked before mapping; no cache pathname
    // can mutate it. Legacy: preserve the existing model-file mmap ownership contract.
    let map = unsafe { memmap2::Mmap::map(file.as_ref())? };
    if map.len() != total {
        return Err(invalid(format!("repack cache {cache:?} size mismatch")));
    }
    Ok(MappedRepack {
        map: Arc::new(map),
        file,
    })
}

#[cfg(not(unix))]
fn strict_file<F>(_cache: &Path, _total: usize, _write: F) -> io::Result<File>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "strict native NVFP4 repacking requires Unix private unlinked file backing",
    ))
}

#[cfg(unix)]
use private::strict_file;

#[cfg(unix)]
mod private {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{FileExt, MetadataExt};

    fn open_at(dir: &File, name: &CStr, flags: i32) -> io::Result<File> {
        let fd = unsafe {
            libc::openat(
                dir.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat returned a new owned fd. NONBLOCK prevents hostile FIFOs from hanging.
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    fn unlink_at(dir: &File, name: &CStr) -> io::Result<()> {
        if unsafe { libc::unlinkat(dir.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    // The cache directory may be model-controlled. Never put a canonical or publication temp
    // directly in it: another directory writer could replace/link/open that temp before use.
    // All temp operations use a held fd to a fresh 0700 directory owned by the serving uid.
    struct PrivateDir {
        parent: File,
        dir: File,
        name: CString,
    }

    impl PrivateDir {
        fn new(parent: File) -> io::Result<Self> {
            let mut entropy = File::open("/dev/urandom")?;
            for _ in 0..32 {
                let mut nonce = [0u8; 16];
                entropy.read_exact(&mut nonce)?;
                let name =
                    CString::new(format!(".private-{:032x}", u128::from_ne_bytes(nonce))).unwrap();
                if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
                    let error = io::Error::last_os_error();
                    if error.kind() == io::ErrorKind::AlreadyExists {
                        continue;
                    }
                    return Err(error);
                }
                let opened = open_at(&parent, &name, libc::O_RDONLY | libc::O_DIRECTORY);
                let dir = match opened {
                    Ok(dir) => dir,
                    Err(error) => {
                        unsafe {
                            libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR);
                        }
                        return Err(error);
                    }
                };
                let check = dir.metadata().and_then(|metadata| {
                    if metadata.uid() != unsafe { libc::geteuid() }
                        || metadata.mode() & 0o777 != 0o700
                    {
                        return Err(invalid(
                            "NVFP4 temporary directory is not private to the loader",
                        ));
                    }
                    Ok(())
                });
                if let Err(error) = check {
                    // Do not let the normal cleanup inspect entries in an unvalidated directory.
                    unsafe {
                        libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR);
                    }
                    return Err(error);
                }
                return Ok(Self { parent, dir, name });
            }
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a private NVFP4 repack directory",
            ))
        }

        fn create(&self, name: &CStr) -> io::Result<File> {
            open_at(
                &self.dir,
                name,
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
            )
        }
    }

    impl Drop for PrivateDir {
        fn drop(&mut self) {
            let _ = unlink_at(&self.dir, c"canonical");
            let _ = unlink_at(&self.dir, c"publish");
            unsafe {
                libc::unlinkat(
                    self.parent.as_raw_fd(),
                    self.name.as_ptr(),
                    libc::AT_REMOVEDIR,
                );
            }
        }
    }

    fn regular_file(file: &File) -> io::Result<std::fs::Metadata> {
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.nlink() != 1 {
            return Err(invalid("repack cache is not a singly-linked regular file"));
        }
        Ok(metadata)
    }

    fn cache_matches(dir: &File, name: &CStr, canonical: &File, total: usize) -> io::Result<bool> {
        let cache = match open_at(dir, name, libc::O_RDONLY) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        };
        if regular_file(&cache)?.len() != total as u64 {
            return Ok(false);
        }
        // Never mmap untrusted cache bytes even while verifying: concurrent truncation must
        // yield a mismatch, not SIGBUS. Memory use is bounded independently of bank size.
        let mut expected = [0u8; 64 * 1024];
        let mut actual = [0u8; 64 * 1024];
        let mut offset = 0;
        while offset < total {
            let len = (total - offset).min(expected.len());
            canonical.read_exact_at(&mut expected[..len], offset as u64)?;
            match cache.read_exact_at(&mut actual[..len], offset as u64) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(false),
                Err(error) => return Err(error),
            }
            if actual[..len] != expected[..len] {
                return Ok(false);
            }
            offset += len;
        }
        Ok(true)
    }

    pub(super) fn strict_file<F>(cache: &Path, total: usize, write: F) -> io::Result<File>
    where
        F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
    {
        let parent = cache
            .parent()
            .ok_or_else(|| invalid("repack cache has no parent"))?;
        let name = cache
            .file_name()
            .ok_or_else(|| invalid("repack cache has no filename"))?;
        let name =
            CString::new(name.as_bytes()).map_err(|_| invalid("repack cache filename has NUL"))?;
        let private = PrivateDir::new(open_repack_cache_dir(parent)?)?;
        let mut out = BufWriter::new(private.create(c"canonical")?);
        let written = regular_file(out.get_ref())?;
        // Acquire the independent read-only fd inside our validated private directory, never
        // through the model/cache path. Unlink before repacking so even a failed/crashed load
        // cannot leave a named bank-sized canonical temp behind.
        let canonical = open_at(&private.dir, c"canonical", libc::O_RDONLY)?;
        let opened = regular_file(&canonical)?;
        if (opened.dev(), opened.ino()) != (written.dev(), written.ino()) {
            return Err(invalid("canonical NVFP4 repack inode changed"));
        }
        unlink_at(&private.dir, c"canonical")?;
        if canonical.metadata()?.nlink() != 0 {
            return Err(invalid("canonical NVFP4 repack must be unlinked"));
        }
        write(&mut out)?;
        out.flush()?;
        if canonical.metadata()?.len() != total as u64 {
            return Err(invalid("canonical NVFP4 repack length mismatch"));
        }
        if unsafe { libc::fchmod(out.get_ref().as_raw_fd(), 0o400) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // No writer survives into cache comparison, publication, or the returned mmap backing.
        drop(out);

        if !cache_matches(&private.parent, &name, &canonical, total)? {
            // Publish a distinct inode. Linking or renaming canonical into the cache would
            // reintroduce mutation of live weights by a later writer of the named cache.
            let mut publish = BufWriter::new(private.create(c"publish")?);
            io::copy(&mut &canonical, &mut publish)?;
            publish.flush()?;
            publish.get_ref().sync_all()?;
            drop(publish);
            if unsafe {
                libc::renameat(
                    private.dir.as_raw_fd(),
                    c"publish".as_ptr(),
                    private.parent.as_raw_fd(),
                    name.as_ptr(),
                )
            } != 0
            {
                return Err(io::Error::last_os_error());
            }
            private.parent.sync_all()?;
        }
        // PrivateDir removes its empty directory now. The model retains only this O_RDONLY
        // unlinked inode; both mmap and future positioned reads use it for the model lifetime.
        Ok(canonical)
    }
}

/// Refuse attacker-controlled filesystem objects in the model-local repack cache.
///
/// Repack artifacts are derived data, but they are opened by the serving process and therefore
/// must not be allowed to follow a model-provided symlink into an arbitrary path. `create_dir_all`
/// and ordinary `File::create` both follow links; use `symlink_metadata` for the directory and
/// `O_NOFOLLOW` for the final file component on Unix. The non-Unix fallback still rejects existing
/// symlinks and keeps the same behavior on platforms without that flag.
pub(super) fn ensure_repack_cache_dir(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_symlink() || !meta.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("repack cache directory is not a real directory: {path:?}"),
                ));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::create_dir(path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    ensure_repack_cache_dir(path)
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    }
}

fn repack_cache_is_fresh(path: &Path, expected_len: usize) -> bool {
    std::fs::symlink_metadata(path)
        .is_ok_and(|meta| meta.file_type().is_file() && meta.len() == expected_len as u64)
}

#[cfg(unix)]
fn open_repack_cache_dir(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = std::fs::OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW);
    options.open(path)
}

#[cfg(not(unix))]
fn open_repack_cache_dir(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new().read(true).open(path)
}

#[cfg(unix)]
fn open_repack_cache(path: &Path, write: bool) -> std::io::Result<std::fs::File> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "repack cache has no parent",
        )
    })?;
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "repack cache has no filename",
        )
    })?;
    let name = CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "repack cache filename has NUL",
        )
    })?;
    let dir = open_repack_cache_dir(parent)?;
    let flags = if write {
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW
    } else {
        libc::O_RDONLY | libc::O_NOFOLLOW
    };
    let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags, 0o600) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: openat returned a fresh, owned descriptor.
    let file = unsafe { std::fs::File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("repack cache is not a regular file: {path:?}"),
        ));
    }
    if std::os::unix::fs::MetadataExt::nlink(&metadata) > 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("repack cache refuses a multiply-linked file: {path:?}"),
        ));
    }
    Ok(file)
}

#[cfg(not(unix))]
fn open_repack_cache(path: &Path, write: bool) -> std::io::Result<std::fs::File> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("repack cache is not a regular file: {path:?}"),
        ));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(!write).write(write);
    options.open(path)
}

/// Write one repack artifact through a descriptor for its real parent directory. The payload is
/// first written to an O_EXCL temporary sibling, fsynced, and atomically renamed into place; a
/// pre-existing symlink, non-regular file, or hard link is rejected before the rename. Thus a
/// malformed model cannot truncate a service-owned inode, and a crash cannot leave a fresh-sized
/// partial cache that a later load would mistake for valid data.
fn write_repack_cache<F>(path: &Path, write: F) -> std::io::Result<()>
where
    F: FnOnce(&mut std::io::BufWriter<std::fs::File>) -> std::io::Result<()>,
{
    use std::io::Write;

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "repack cache has no parent",
        )
    })?;
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "repack cache has no filename",
        )
    })?;
    let dir = open_repack_cache_dir(parent)?;

    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::io::{AsRawFd, FromRawFd};
        static TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name = CString::new(name.as_bytes()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "repack cache filename has NUL",
            )
        })?;
        let mut temp_name = None;
        let mut temp_file = None;
        for _ in 0..32 {
            let suffix = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let candidate = CString::new(format!(
                ".{}.tmp-{}-{suffix}",
                name.to_string_lossy(),
                std::process::id()
            ))
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "temporary filename has NUL",
                )
            })?;
            let fd = unsafe {
                libc::openat(
                    dir.as_raw_fd(),
                    candidate.as_ptr(),
                    libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW,
                    0o600,
                )
            };
            if fd >= 0 {
                temp_name = Some(candidate);
                // SAFETY: openat returned a fresh, owned descriptor.
                temp_file = Some(unsafe { std::fs::File::from_raw_fd(fd) });
                break;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(error);
            }
        }
        let temp_name = temp_name.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique repack cache temporary",
            )
        })?;
        let mut out = std::io::BufWriter::new(temp_file.expect("temporary file accompanies name"));
        let result = write(&mut out).and_then(|()| {
            out.flush()?;
            out.get_ref().sync_all()?;
            Ok(())
        });
        drop(out);
        if let Err(error) = result {
            unsafe {
                libc::unlinkat(dir.as_raw_fd(), temp_name.as_ptr(), 0);
            }
            return Err(error);
        }

        // Never replace a caller-provided link or a hard-linked service inode. If a race swaps the
        // final entry after this check, renameat only replaces that directory entry; it cannot
        // write through the swapped inode, and the temporary remains private to this directory.
        if let Ok(metadata) = std::fs::symlink_metadata(path)
            && (metadata.file_type().is_symlink()
                || !metadata.is_file()
                || std::os::unix::fs::MetadataExt::nlink(&metadata) > 1)
        {
            unsafe {
                libc::unlinkat(dir.as_raw_fd(), temp_name.as_ptr(), 0);
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("repack cache target is not a private regular file: {path:?}"),
            ));
        }
        let status = unsafe {
            libc::renameat(
                dir.as_raw_fd(),
                temp_name.as_ptr(),
                dir.as_raw_fd(),
                name.as_ptr(),
            )
        };
        if status != 0 {
            unsafe {
                libc::unlinkat(dir.as_raw_fd(), temp_name.as_ptr(), 0);
            }
            return Err(std::io::Error::last_os_error());
        }
        dir.sync_all()
    }

    #[cfg(not(unix))]
    {
        let temp = parent.join(format!(
            ".{}.tmp-{}",
            name.to_string_lossy(),
            std::process::id()
        ));
        let mut out = std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?,
        );
        write(&mut out)?;
        out.flush()?;
        out.get_ref().sync_all()?;
        drop(out);
        if let Ok(metadata) = std::fs::symlink_metadata(path) {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                std::fs::remove_file(&temp).ok();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("repack cache target is not a private regular file: {path:?}"),
                ));
            }
        }
        std::fs::rename(temp, path)
    }
}
