//! Canonical derived output under a retained source-directory capability.
//! No checkpoint path is reopened and no named cache file is trusted as model bytes.
use crate::bound_disk::{BoundDiskCache, BoundDiskView};
use crate::source::{DiskExtent, Nvfp4Native};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub(crate) struct RetainedOutputRoot {
    directory: Result<Arc<File>, String>,
    cache: Mutex<Option<Result<Arc<File>, String>>>,
}

impl RetainedOutputRoot {
    /// Capturing failure does not prevent read-only source loading. A later requested output
    /// fails with this original error; it never reopens a possibly replaced directory path.
    pub(crate) fn capture(path: &Path) -> Self {
        let path = if path.as_os_str().is_empty() {
            Path::new(".")
        } else {
            path
        };
        #[cfg(unix)]
        let directory = {
            use std::os::unix::fs::OpenOptionsExt;
            File::options()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC)
                .open(path)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        };
        #[cfg(not(unix))]
        let directory = Err("canonical bound output requires Unix directory capabilities".into());
        #[cfg(unix)]
        let cache = directory.as_ref().ok().and_then(|root| {
            match unix::open_at(root, c".memra-repack", libc::O_RDONLY | libc::O_DIRECTORY) {
                Ok(file) => Some(Ok(Arc::new(file))),
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => Some(Err(error.to_string())),
            }
        });
        #[cfg(not(unix))]
        let cache = None;
        let _ = path;
        Self {
            directory,
            cache: Mutex::new(cache),
        }
    }

    #[cfg(unix)]
    fn cache_directory(&self) -> Result<Arc<File>, String> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "canonical output directory lock poisoned")?;
        if let Some(result) = cache.as_ref() {
            return result.clone();
        }
        let result = self
            .directory
            .as_ref()
            .map_err(Clone::clone)
            .and_then(|root| {
                use std::os::fd::AsRawFd;
                if unsafe { libc::mkdirat(root.as_raw_fd(), c".memra-repack".as_ptr(), 0o700) } != 0
                {
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::AlreadyExists {
                        return Err(error.to_string());
                    }
                }
                unix::open_at(root, c".memra-repack", libc::O_RDONLY | libc::O_DIRECTORY)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            });
        *cache = Some(result.clone());
        result
    }

    pub(crate) fn write<F>(
        &self,
        bytes: usize,
        authority: Arc<str>,
        produce: F,
    ) -> Result<BoundDiskView, String>
    where
        F: FnOnce(&mut dyn Write) -> io::Result<()>,
    {
        #[cfg(unix)]
        {
            let root = self.cache_directory()?;
            let file = unix::canonical(&root, bytes, produce).map_err(|e| e.to_string())?;
            let file = Arc::new(file);
            // SAFETY: the only writer is closed, the inode is unlinked, and the retained fd is
            // read-only. Named cache replacement cannot modify this backing.
            let map =
                Arc::new(unsafe { memmap2::Mmap::map(file.as_ref()) }.map_err(|e| e.to_string())?);
            BoundDiskCache::default().view(
                DiskExtent {
                    file,
                    map,
                    offset: 0,
                    len: bytes,
                },
                authority,
            )
        }
        #[cfg(not(unix))]
        {
            let _ = (bytes, authority, produce);
            Err("canonical bound output requires Unix private backing".into())
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::MetadataExt;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "memra-canonical-output-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn canonical_output_is_read_only_unlinked_and_length_checked() {
        let scratch = Scratch::new();
        let root = RetainedOutputRoot::capture(&scratch.0);
        let cache = root.cache_directory().unwrap();
        let file = unix::canonical(&cache, 8, |writer| writer.write_all(&[7; 8])).unwrap();
        assert_eq!(file.metadata().unwrap().nlink(), 0);
        assert_eq!(file.metadata().unwrap().mode() & 0o777, 0o400);
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) } & libc::O_ACCMODE,
            libc::O_RDONLY
        );
        for bytes in [4usize, 12] {
            assert!(
                root.write(8, Arc::from("test"), |writer| writer
                    .write_all(&vec![1; bytes]))
                    .is_err()
            );
        }
        assert!(
            root.write(8, Arc::from("test"), |writer| {
                writer.write_all(&[1; 4])?;
                Err(io::Error::other("injected producer failure"))
            })
            .is_err()
        );
        assert_eq!(
            std::fs::read_dir(scratch.0.join(".memra-repack"))
                .unwrap()
                .count(),
            0
        );
    }

    #[test]
    fn source_and_existing_cache_directory_replacements_do_not_redirect_output() {
        for existing_cache in [false, true] {
            let scratch = Scratch::new();
            let source = scratch.0.join("source");
            std::fs::create_dir(&source).unwrap();
            let outside = scratch.0.join("outside");
            std::fs::create_dir(&outside).unwrap();
            std::fs::write(outside.join("sentinel"), b"unchanged").unwrap();
            if existing_cache {
                std::fs::create_dir(source.join(".memra-repack")).unwrap();
            }
            let root = RetainedOutputRoot::capture(&source);
            let retained = scratch.0.join("retained");
            std::fs::rename(&source, &retained).unwrap();
            std::fs::create_dir(&source).unwrap();
            std::os::unix::fs::symlink(&outside, source.join(".memra-repack")).unwrap();
            if existing_cache {
                std::fs::rename(retained.join(".memra-repack"), retained.join("old-cache"))
                    .unwrap();
                std::os::unix::fs::symlink(&outside, retained.join(".memra-repack")).unwrap();
            }
            let view = root
                .write(8, Arc::from("test"), |writer| writer.write_all(&[9; 8]))
                .unwrap();
            assert_eq!(view.bytes(), [9; 8]);
            assert_eq!(
                std::fs::read(outside.join("sentinel")).unwrap(),
                b"unchanged"
            );
            assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 1);
            assert_eq!(
                std::fs::read_dir(retained.join(if existing_cache {
                    "old-cache"
                } else {
                    ".memra-repack"
                }))
                .unwrap()
                .count(),
                0
            );
        }
    }

    #[test]
    fn unsafe_cache_entries_refuse_without_following_or_waiting() {
        use std::os::unix::ffi::OsStrExt;
        for fifo in [false, true] {
            let scratch = Scratch::new();
            let cache = scratch.0.join(".memra-repack");
            if fifo {
                let path = std::ffi::CString::new(cache.as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            } else {
                std::os::unix::fs::symlink(std::env::temp_dir(), &cache).unwrap();
            }
            let root = RetainedOutputRoot::capture(&scratch.0);
            assert!(
                root.write(4, Arc::from("test"), |writer| writer.write_all(&[1; 4]))
                    .is_err()
            );
        }
    }

    #[test]
    fn concurrent_private_outputs_keep_independent_bytes_and_no_named_temps() {
        let scratch = Scratch::new();
        let root = RetainedOutputRoot::capture(&scratch.0);
        let views = std::thread::scope(|scope| {
            let handles: Vec<_> = (0u8..8)
                .map(|byte| {
                    let root = &root;
                    scope.spawn(move || {
                        root.write(128, Arc::from("concurrent"), |writer| {
                            writer.write_all(&[byte; 128])
                        })
                        .unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        drop(root);
        for (byte, view) in views.iter().enumerate() {
            let mut read = [0; 128];
            view.read_at(&mut read, 0).unwrap();
            assert_eq!(read, [byte as u8; 128]);
            assert!(view.read_at(&mut [0; 2], 127).is_err());
        }
        assert_eq!(
            std::fs::read_dir(scratch.0.join(".memra-repack"))
                .unwrap()
                .count(),
            0
        );
    }
}

/// One expert at a time, including any source-owned scale unswizzle. The producer cannot
/// replace the requested directory, input operands, or canonical codec.
pub(crate) fn repack<'a>(
    root: &RetainedOutputRoot,
    n: usize,
    out_f: usize,
    in_f: usize,
    authority: Arc<str>,
    mut member: impl FnMut(usize) -> Result<Nvfp4Native<'a>, String>,
) -> Result<BoundDiskView, String> {
    if n == 0 || out_f == 0 || in_f == 0 || !in_f.is_multiple_of(64) {
        return Err("canonical NVFP4 output requires nonempty, 64-aligned geometry".into());
    }
    let codes = out_f
        .checked_mul(in_f / 2)
        .ok_or("NVFP4 source size overflow")?;
    let scales = out_f
        .checked_mul(in_f / 16)
        .ok_or("NVFP4 scale size overflow")?;
    let total = n
        .checked_mul(out_f)
        .and_then(|v| v.checked_mul(in_f / 64))
        .and_then(|v| v.checked_mul(36))
        .ok_or("NVFP4 canonical size overflow")?;
    root.write(total, authority, |writer| {
        for index in 0..n {
            let native = member(index).map_err(io::Error::other)?;
            if (native.out_f, native.in_f) != (out_f, in_f)
                || native.wbytes.len() != codes
                || native.wscale.len() != scales
            {
                return Err(io::Error::other("bound NVFP4 member geometry changed"));
            }
            writer.write_all(&crate::nvfp4_repack::repack_modelopt_to_gguf(
                native.wbytes,
                &native.wscale,
                out_f,
                in_f,
            ))?;
        }
        Ok(())
    })
}

pub(crate) fn repack_stacked(
    root: &RetainedOutputRoot,
    bank: &crate::source::Nvfp4StackedNative<'_>,
    authority: Arc<str>,
) -> Result<BoundDiskView, String> {
    let code_stride = bank
        .out_f
        .checked_mul(bank.in_f / 2)
        .ok_or("NVFP4 code extent overflow")?;
    let scale_stride = bank
        .out_f
        .checked_mul(bank.in_f / 16)
        .ok_or("NVFP4 scale extent overflow")?;
    if bank.n_expert.checked_mul(code_stride) != Some(bank.codes.len())
        || bank.n_expert.checked_mul(scale_stride) != Some(bank.scales.len())
        || bank.macros.len() != bank.n_expert
    {
        return Err("native NVFP4 bank extents disagree with their geometry".into());
    }
    repack(
        root,
        bank.n_expert,
        bank.out_f,
        bank.in_f,
        authority,
        |index| {
            Ok(Nvfp4Native {
                wbytes: &bank.codes[index * code_stride..(index + 1) * code_stride],
                wscale: std::borrow::Cow::Borrowed(
                    &bank.scales[index * scale_stride..(index + 1) * scale_stride],
                ),
                out_f: bank.out_f,
                in_f: bank.in_f,
            })
        },
    )
}

#[cfg(unix)]
mod unix {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::fs::MetadataExt;

    pub(super) fn open_at(dir: &File, name: &CStr, flags: i32) -> io::Result<File> {
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
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    struct PrivateDirectory<'a> {
        root: &'a File,
        directory: File,
        name: CString,
    }
    impl<'a> PrivateDirectory<'a> {
        fn create(root: &'a File) -> io::Result<Self> {
            let mut entropy = File::open("/dev/urandom")?;
            for _ in 0..32 {
                let mut nonce = [0u8; 16];
                entropy.read_exact(&mut nonce)?;
                let name =
                    CString::new(format!(".bound-{:032x}", u128::from_ne_bytes(nonce))).unwrap();
                if unsafe { libc::mkdirat(root.as_raw_fd(), name.as_ptr(), 0o700) } != 0 {
                    let error = io::Error::last_os_error();
                    if error.kind() == io::ErrorKind::AlreadyExists {
                        continue;
                    }
                    return Err(error);
                }
                let opened = open_at(root, &name, libc::O_RDONLY | libc::O_DIRECTORY);
                let directory = match opened {
                    Ok(directory) => directory,
                    Err(error) => {
                        unsafe {
                            libc::unlinkat(root.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR);
                        }
                        return Err(error);
                    }
                };
                let check = directory.metadata().and_then(|metadata| {
                    if metadata.uid() != unsafe { libc::geteuid() }
                        || metadata.mode() & 0o777 != 0o700
                    {
                        Err(io::Error::other(
                            "canonical output directory is not loader-private",
                        ))
                    } else {
                        Ok(())
                    }
                });
                if let Err(error) = check {
                    // An unvalidated replacement directory must never have entries removed.
                    unsafe {
                        libc::unlinkat(root.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR);
                    }
                    return Err(error);
                }
                return Ok(Self {
                    root,
                    directory,
                    name,
                });
            }
            Err(io::Error::other(
                "unable to create private canonical output directory",
            ))
        }
    }
    impl Drop for PrivateDirectory<'_> {
        fn drop(&mut self) {
            unsafe {
                libc::unlinkat(self.directory.as_raw_fd(), c"canonical".as_ptr(), 0);
                libc::unlinkat(
                    self.root.as_raw_fd(),
                    self.name.as_ptr(),
                    libc::AT_REMOVEDIR,
                );
            }
        }
    }

    struct LimitedWriter<'a> {
        writer: &'a mut BufWriter<File>,
        remaining: usize,
    }
    impl Write for LimitedWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.remaining {
                return Err(io::Error::other("canonical output exceeds declared extent"));
            }
            let n = self.writer.write(bytes)?;
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.writer.flush()
        }
    }

    pub(super) fn canonical<F>(root: &File, bytes: usize, produce: F) -> io::Result<File>
    where
        F: FnOnce(&mut dyn Write) -> io::Result<()>,
    {
        if bytes == 0 {
            return Err(io::Error::other("canonical output cannot be empty"));
        }
        let directory = PrivateDirectory::create(root)?;
        let mut writer = BufWriter::new(open_at(
            &directory.directory,
            c"canonical",
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?);
        let written = writer.get_ref().metadata()?;
        let reader = open_at(&directory.directory, c"canonical", libc::O_RDONLY)?;
        let read = reader.metadata()?;
        if !read.is_file()
            || read.nlink() != 1
            || (read.dev(), read.ino()) != (written.dev(), written.ino())
        {
            return Err(io::Error::other(
                "canonical output inode changed before unlink",
            ));
        }
        if unsafe { libc::unlinkat(directory.directory.as_raw_fd(), c"canonical".as_ptr(), 0) } != 0
        {
            return Err(io::Error::last_os_error());
        }
        if reader.metadata()?.nlink() != 0 {
            return Err(io::Error::other("canonical output must be unlinked"));
        }
        let mut bounded = LimitedWriter {
            writer: &mut writer,
            remaining: bytes,
        };
        produce(&mut bounded)?;
        if bounded.remaining != 0 {
            return Err(io::Error::other(
                "canonical output is shorter than declared extent",
            ));
        }
        writer.flush()?;
        if reader.metadata()?.len() != bytes as u64 {
            return Err(io::Error::other("canonical output length mismatch"));
        }
        if unsafe { libc::fchmod(writer.get_ref().as_raw_fd(), 0o400) } != 0 {
            return Err(io::Error::last_os_error());
        }
        drop(writer);
        Ok(reader)
    }
}
