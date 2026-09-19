//! Exact aligned extent I/O. macOS F_NOCACHE is a development fallback, NOT O_DIRECT.
use super::ReadAt;
use crate::contracts::{Error, Result};
use std::{fs::File, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    Buffered,
    Uncached,
}
pub struct AlignedFile {
    file: File,
}
impl AlignedFile {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_direct(path, false)
    }
    /// Exclusive creation prevents partial writes from clobbering an immutable extent.
    pub fn create_new(path: &Path) -> Result<Self> {
        Self::open_direct(path, true)
    }
    fn open_direct(path: &Path, create: bool) -> Result<Self> {
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ))]
        {
            use std::os::unix::fs::OpenOptionsExt;
            #[cfg(target_arch = "x86_64")]
            const O_DIRECT: i32 = 0o40000;
            #[cfg(target_arch = "aarch64")]
            const O_DIRECT: i32 = 0o200000;
            let file = std::fs::OpenOptions::new()
                .read(!create)
                .write(create)
                .create_new(create)
                .custom_flags(O_DIRECT)
                .open(path)?;
            Ok(Self { file })
        }
        #[cfg(not(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )))]
        {
            let _ = (path, create);
            Err(Error::Unsupported)
        }
    }
    /// The native adapter configures uncached flags before handing over ownership.
    /// This does not assert that an arbitrary caller-provided descriptor is O_DIRECT.
    pub fn from_configured_file(file: File) -> Self {
        Self { file }
    }
    pub fn backend_label() -> &'static str {
        if cfg!(target_os = "macos") {
            "macos-f-nocache-development-fallback"
        } else {
            "linux-o-direct"
        }
    }
    pub fn read_aligned(&self, offset: u64, dst: &mut [u8]) -> Result<u64> {
        if !offset.is_multiple_of(4096)
            || !dst.len().is_multiple_of(4096)
            || !(dst.as_ptr() as usize).is_multiple_of(4096)
        {
            return Err(Error::InvalidLayout);
        }
        offset
            .checked_add(dst.len() as u64)
            .ok_or(Error::Overflow)?;
        let mut done = 0;
        while done < dst.len() {
            match self.read_at(&mut dst[done..], offset + done as u64) {
                Ok(n) => {
                    done += n;
                    if n == 0 || !n.is_multiple_of(4096) {
                        return Err(Error::ShortIo {
                            expected: dst.len() as u64,
                            actual: done as u64,
                        });
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(done as u64)
    }
    /// A short unaligned write cannot be retried at a now-unaligned offset.
    /// The caller must discard the private temporary extent, never publish it.
    pub fn write_aligned(&self, offset: u64, src: &[u8]) -> Result<u64> {
        if !offset.is_multiple_of(4096)
            || !src.len().is_multiple_of(4096)
            || !(src.as_ptr() as usize).is_multiple_of(4096)
        {
            return Err(Error::InvalidLayout);
        }
        offset
            .checked_add(src.len() as u64)
            .ok_or(Error::Overflow)?;
        let mut done = 0;
        while done < src.len() {
            match std::os::unix::fs::FileExt::write_at(
                &self.file,
                &src[done..],
                offset + done as u64,
            ) {
                Ok(n) => {
                    done += n;
                    if n == 0 || !n.is_multiple_of(4096) {
                        return Err(Error::ShortIo {
                            expected: src.len() as u64,
                            actual: done as u64,
                        });
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(done as u64)
    }
    pub fn sync_all(&self) -> Result<()> {
        self.file.sync_all().map_err(Error::from)
    }
}
impl ReadAt for AlignedFile {
    fn read_at(&self, dst: &mut [u8], offset: u64) -> std::io::Result<usize> {
        if !offset.is_multiple_of(4096)
            || !dst.len().is_multiple_of(4096)
            || !(dst.as_ptr() as usize).is_multiple_of(4096)
        {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
        }
        std::os::unix::fs::FileExt::read_at(&self.file, dst, offset)
    }
}
/// Bounded pageable bounce backing, not CUDA-pinned. Handles an unaligned caller
/// buffer without relaxing direct offset/length constraints or changing valid bytes.
pub struct AlignedBuffer {
    backing: Vec<u8>,
    start: usize,
    len: usize,
}
impl AlignedBuffer {
    pub fn new(size: usize) -> Result<Self> {
        if size > crate::object_store::MAX_CHUNK + 4096 {
            return Err(Error::Capacity);
        }
        if size == 0 || !size.is_multiple_of(4096) {
            return Err(Error::InvalidLayout);
        }
        let capacity = size.checked_add(4095).ok_or(Error::Overflow)?;
        let mut backing = Vec::new();
        backing
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Capacity)?;
        backing.resize(capacity, 0);
        let start = (4096 - backing.as_ptr() as usize % 4096) % 4096;
        Ok(Self {
            backing,
            start,
            len: size,
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.backing[self.start..self.start + self.len]
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.backing[self.start..self.start + self.len]
    }
}
pub type FileOpener = fn(&Path) -> Result<File>;
pub fn read_file(path: &Path, size: usize, opener: Option<FileOpener>) -> Result<Vec<u8>> {
    let mut buffer = AlignedBuffer::new(size)?;
    let file = match opener {
        Some(open) => AlignedFile::from_configured_file(open(path)?),
        None => AlignedFile::open(path)?,
    };
    file.read_aligned(0, buffer.bytes_mut())?;
    Ok(buffer.bytes().to_vec())
}
