//! Exact aligned reads. macOS F_NOCACHE is a development fallback, NOT O_DIRECT.
use super::{ReadAt, read_exact_at};
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
                .read(true)
                .custom_flags(O_DIRECT)
                .open(path)?;
            Ok(Self { file })
        }
        #[cfg(not(all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )))]
        {
            let _ = path;
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
        read_exact_at(self, dst, offset)
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
/// Bounded readback buffer; this is pageable development memory, never pinned/GPU memory.
pub type FileOpener = fn(&Path) -> Result<File>;
pub fn read_file(path: &Path, size: usize, opener: Option<FileOpener>) -> Result<Vec<u8>> {
    if size > crate::object_store::MAX_CHUNK + 4096 {
        return Err(Error::Capacity);
    }
    if !size.is_multiple_of(4096) {
        return Err(Error::InvalidLayout);
    }
    let capacity = size.checked_add(4095).ok_or(Error::Overflow)?;
    let mut backing = vec![0; capacity];
    let start = (4096 - backing.as_ptr() as usize % 4096) % 4096;
    let file = match opener {
        Some(open) => AlignedFile::from_configured_file(open(path)?),
        None => AlignedFile::open(path)?,
    };
    file.read_aligned(0, &mut backing[start..start + size])?;
    Ok(backing[start..start + size].to_vec())
}
