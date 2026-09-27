//! Authorized tensor extents and positioned readers, without raw file or mmap access.
#[cfg(unix)]
pub mod ffi;

use crate::source::DiskExtent;
use std::{
    collections::HashMap,
    fs::File,
    io,
    ops::Range,
    sync::{Arc, Mutex, Weak},
};

/// Migration boundary: raw sources keep their retained extent; compiler-bound sources can only
/// return the opaque arm. Consumers must not widen an opaque view to its containing shard.
#[derive(Clone)]
pub enum ExpertDiskView {
    Unbound(DiskExtent),
    Bound(BoundDiskView),
}
impl ExpertDiskView {
    pub fn len(&self) -> usize {
        match self {
            Self::Unbound(extent) => extent.len,
            Self::Bound(view) => view.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Bound(view) => view.bytes(),
            Self::Unbound(extent) => {
                &extent.map[extent.offset as usize..extent.offset as usize + extent.len]
            }
        }
    }
    pub fn subrange(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len() {
            return Err("requested subrange leaves the expert extent".into());
        }
        match self {
            Self::Bound(view) => view.subrange(range).map(Self::Bound),
            Self::Unbound(extent) => Ok(Self::Unbound(DiskExtent {
                map: extent.map.clone(),
                file: extent.file.clone(),
                offset: extent
                    .offset
                    .checked_add(range.start as u64)
                    .ok_or("expert subrange offset overflow")?,
                len: range.end - range.start,
            })),
        }
    }
    /// Preserve the existing spill mapping advice without exporting the whole mapping.
    pub fn advise_expert_access(&self) {
        let map = match self {
            Self::Bound(view) => &view.backing.map,
            Self::Unbound(extent) => &extent.map,
        };
        let _ = crate::source::apply_expert_mmap_advice(map);
    }
    /// Preserve whole-slab population policy without exporting the file. A partial tensor
    /// window is never widened to the containing file for this optional loader hint.
    pub fn populate_whole_slab(&self, label: &str) -> bool {
        let (file, offset, len, total) = match self {
            Self::Bound(view) => (
                &view.backing.file,
                view.offset,
                view.len,
                view.backing.map.len(),
            ),
            Self::Unbound(extent) => (&extent.file, extent.offset, extent.len, extent.map.len()),
        };
        offset == 0 && len == total && crate::source::populate_expert_slab(file, len, label)
    }
    /// Failed adjacency is an optimization miss: both individual windows remain usable.
    pub fn join_adjacent(&self, next: &Self) -> Result<Self, String> {
        match (self, next) {
            (Self::Bound(a), Self::Bound(b)) => a.join_adjacent(b).map(Self::Bound),
            (Self::Unbound(a), Self::Unbound(b))
                if Arc::ptr_eq(&a.map, &b.map)
                    && Arc::ptr_eq(&a.file, &b.file)
                    && a.offset.checked_add(a.len as u64) == Some(b.offset) =>
            {
                let len = a
                    .len
                    .checked_add(b.len)
                    .ok_or("joined disk extent overflows")?;
                let start =
                    usize::try_from(a.offset).map_err(|_| "disk offset exceeds address space")?;
                if start.checked_add(len).is_none_or(|end| end > a.map.len()) {
                    return Err("joined disk extent exceeds its mapping".into());
                }
                Ok(Self::Unbound(DiskExtent {
                    map: a.map.clone(),
                    file: a.file.clone(),
                    offset: a.offset,
                    len,
                }))
            }
            _ => Err("disk extents do not have adjacent matching backing and authority".into()),
        }
    }
}

/// Opened-file generation used by the CPU mirror map. This is metadata, not byte identity.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileGeneration {
    pub device: u64,
    pub inode: u64,
    pub file_bytes: u64,
    pub ctime_seconds: i64,
    pub ctime_nanoseconds: i64,
}
#[cfg(unix)]
impl FileGeneration {
    fn read(file: &File) -> io::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let m = file.metadata()?;
        if !m.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mirror backing must be a regular file",
            ));
        }
        Ok(Self {
            device: m.dev(),
            inode: m.ino(),
            file_bytes: m.len(),
            ctime_seconds: m.ctime(),
            ctime_nanoseconds: m.ctime_nsec(),
        })
    }
}
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MirrorKey {
    source: FileGeneration,
    alternate: FileGeneration,
    offset: u64,
    len: usize,
}

#[cfg(target_os = "linux")]
fn retained_direct_file(file: &File) -> io::Result<Arc<File>> {
    use std::os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    };
    let before = file.metadata()?;
    let direct = File::options()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
    let flags = unsafe { libc::fcntl(direct.as_raw_fd(), libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::O_DIRECT == 0 {
        return Err(io::Error::other(
            "bound direct reader did not retain O_DIRECT",
        ));
    }
    let after = direct.metadata()?;
    if (before.dev(), before.ino()) != (after.dev(), after.ino()) {
        return Err(io::Error::other("direct reopen changed the bound inode"));
    }
    Ok(Arc::new(direct))
}

struct Backing {
    map: Arc<memmap2::Mmap>,
    file: Arc<File>,
    #[cfg(target_os = "linux")]
    direct: Mutex<Option<Arc<File>>>,
    #[cfg(target_os = "linux")]
    random_advised: Mutex<bool>,
    #[cfg(target_os = "linux")]
    mirrors: Mutex<HashMap<MirrorKey, BoundDiskReader>>,
    #[cfg(target_os = "linux")]
    mirror_files: Mutex<HashMap<FileGeneration, Arc<File>>>,
}

/// Load-local cache shares the read descriptor across all tensor windows of one opened mapping.
/// Only weak references live here; loaded views retain the exact mapping and inode themselves.
#[derive(Default)]
pub(crate) struct BoundDiskCache(Mutex<HashMap<(usize, usize), Weak<Backing>>>);
impl BoundDiskCache {
    pub(crate) fn view(
        &self,
        extent: DiskExtent,
        authority: Arc<str>,
    ) -> Result<BoundDiskView, String> {
        let key = (
            Arc::as_ptr(&extent.map) as usize,
            Arc::as_ptr(&extent.file) as usize,
        );
        let mut cache = self
            .0
            .lock()
            .map_err(|_| "bound disk backing cache poisoned")?;
        let backing = match cache.get(&key).and_then(Weak::upgrade) {
            Some(backing) => backing,
            None => {
                let backing = Arc::new(Backing {
                    map: extent.map,
                    file: extent.file,
                    #[cfg(target_os = "linux")]
                    direct: Mutex::new(None),
                    #[cfg(target_os = "linux")]
                    random_advised: Mutex::new(false),
                    #[cfg(target_os = "linux")]
                    mirrors: Mutex::new(HashMap::new()),
                    #[cfg(target_os = "linux")]
                    mirror_files: Mutex::new(HashMap::new()),
                });
                cache.insert(key, Arc::downgrade(&backing));
                backing
            }
        };
        BoundDiskView::from_backing(backing, extent.offset, extent.len, authority)
    }
}

/// Process-local equality token for asynchronous lifetime deduplication. Not artifact identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundBackingId(usize);
impl std::fmt::Debug for BoundBackingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BoundBackingId(..)")
    }
}

#[derive(Clone)]
pub struct BoundDiskView {
    backing: Arc<Backing>,
    offset: u64,
    len: usize,
    authority: Arc<str>,
}
impl std::fmt::Debug for BoundDiskView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundDiskView")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundReadMode {
    Buffered,
    Random,
    Direct,
}

/// A worker-owned capability for one tensor range. Offsets passed to it are relative to that range.
#[derive(Clone)]
pub struct BoundDiskReader {
    file: Arc<File>,
    offset: u64,
    len: usize,
    direct: bool,
}
impl BoundDiskReader {
    pub fn read_exact_at(&self, mut dst: &mut [u8], mut relative: u64) -> io::Result<()> {
        if relative
            .checked_add(dst.len() as u64)
            .is_none_or(|end| end > self.len as u64)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read leaves the authorized tensor extent",
            ));
        }
        while !dst.is_empty() {
            match self.read_at(dst, relative) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "short bounded read",
                    ));
                }
                Ok(n) => {
                    relative = relative.checked_add(n as u64).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "bounded read offset overflow")
                    })?;
                    dst = &mut dst[n..];
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    pub fn read_at(&self, dst: &mut [u8], relative: u64) -> io::Result<usize> {
        if relative
            .checked_add(dst.len() as u64)
            .is_none_or(|end| end > self.len as u64)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read leaves the authorized tensor extent",
            ));
        }
        let offset = self
            .offset
            .checked_add(relative)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        if self.direct
            && (!offset.is_multiple_of(4096)
                || !dst.len().is_multiple_of(4096)
                || !(dst.as_ptr() as usize).is_multiple_of(4096))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct tensor read requires aligned offset, length and destination",
            ));
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::FileExt::read_at(self.file.as_ref(), dst, offset)
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::FileExt::seek_read(self.file.as_ref(), dst, offset)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = offset;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "positioned reads unavailable",
            ))
        }
    }
}
impl BoundDiskView {
    #[cfg(test)]
    fn new(extent: DiskExtent, authority: Arc<str>) -> Result<Self, String> {
        BoundDiskCache::default().view(extent, authority)
    }
    fn from_backing(
        backing: Arc<Backing>,
        offset: u64,
        len: usize,
        authority: Arc<str>,
    ) -> Result<Self, String> {
        let start =
            usize::try_from(offset).map_err(|_| "bound disk offset exceeds address space")?;
        if start
            .checked_add(len)
            .is_none_or(|end| end > backing.map.len())
        {
            return Err("bound disk extent exceeds its opened mapping".into());
        }
        Ok(Self {
            backing,
            offset,
            len,
            authority,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn backing_id(&self) -> BoundBackingId {
        BoundBackingId(Arc::as_ptr(&self.backing) as usize)
    }
    pub fn bytes(&self) -> &[u8] {
        let start = self.offset as usize;
        &self.backing.map[start..start + self.len]
    }
    pub fn subrange(&self, range: Range<usize>) -> Result<Self, String> {
        if range.start > range.end || range.end > self.len {
            return Err("requested subrange leaves the bound tensor".into());
        }
        Self::from_backing(
            self.backing.clone(),
            self.offset
                .checked_add(range.start as u64)
                .ok_or("bound subrange offset overflows")?,
            range.end - range.start,
            self.authority.clone(),
        )
    }
    pub fn join_adjacent(&self, next: &Self) -> Result<Self, String> {
        if self.authority != next.authority
            || !Arc::ptr_eq(&self.backing.map, &next.backing.map)
            || !Arc::ptr_eq(&self.backing.file, &next.backing.file)
            || self.offset.checked_add(self.len as u64) != Some(next.offset)
        {
            return Err(
                "only adjacent authorized extents of the same opened backing may be joined".into(),
            );
        }
        Self::from_backing(
            self.backing.clone(),
            self.offset,
            self.len
                .checked_add(next.len)
                .ok_or("joined bound extent overflows")?,
            self.authority.clone(),
        )
    }
    pub fn advise_willneed(&self, relative: usize, len: usize) -> bool {
        if len == 0 || relative.checked_add(len).is_none_or(|end| end > self.len) {
            return false;
        }
        #[cfg(unix)]
        {
            self.backing
                .map
                .advise_range(
                    memmap2::Advice::WillNeed,
                    self.offset as usize + relative,
                    len,
                )
                .is_ok()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }
    pub fn direct_aligned(&self) -> bool {
        self.offset.is_multiple_of(4096) && self.len.is_multiple_of(4096)
    }
    pub fn reader(&self, mode: BoundReadMode) -> io::Result<BoundDiskReader> {
        let file = match mode {
            BoundReadMode::Buffered => self.backing.file.clone(),
            BoundReadMode::Random => {
                #[cfg(target_os = "linux")]
                {
                    use std::os::fd::AsRawFd;
                    let mut advised = self
                        .backing
                        .random_advised
                        .lock()
                        .map_err(|_| io::Error::other("random advice cache poisoned"))?;
                    if !*advised {
                        let code = unsafe {
                            libc::posix_fadvise(
                                self.backing.file.as_raw_fd(),
                                0,
                                0,
                                libc::POSIX_FADV_RANDOM,
                            )
                        };
                        if code != 0 {
                            return Err(io::Error::from_raw_os_error(code));
                        }
                        *advised = true;
                    }
                }
                self.backing.file.clone()
            }
            BoundReadMode::Direct => {
                if !self.offset.is_multiple_of(4096) || !self.len.is_multiple_of(4096) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "direct tensor extent requires 4096-byte aligned offset and length",
                    ));
                }
                #[cfg(target_os = "linux")]
                {
                    let mut cache = self
                        .backing
                        .direct
                        .lock()
                        .map_err(|_| io::Error::other("direct descriptor cache poisoned"))?;
                    if cache.is_none() {
                        *cache = Some(retained_direct_file(&self.backing.file)?);
                    }
                    cache.as_ref().unwrap().clone()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "direct tensor reads require Linux",
                    ));
                }
            }
        };
        Ok(BoundDiskReader {
            file,
            offset: self.offset,
            len: self.len,
            direct: mode == BoundReadMode::Direct,
        })
    }
    #[cfg(unix)]
    pub fn file_generation(&self) -> io::Result<FileGeneration> {
        FileGeneration::read(&self.backing.file)
    }

    /// Admit a mirror only for this exact aligned window. Both declared file generations,
    /// distinct filesystems and every selected byte must match before a reader is published.
    /// Cached readers remain tied to those generations and retain their opened inode.
    #[cfg(target_os = "linux")]
    pub fn verified_mirror_reader(
        &self,
        path: &std::path::Path,
        source: FileGeneration,
        alternate: FileGeneration,
    ) -> io::Result<BoundDiskReader> {
        if !self.direct_aligned()
            || source.device == alternate.device
            || source.file_bytes != alternate.file_bytes
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mirror requires aligned windows on distinct filesystems with equal file sizes",
            ));
        }
        if self.file_generation()? != source {
            return Err(io::Error::other(
                "source generation differs from the mirror declaration",
            ));
        }
        let key = MirrorKey {
            source,
            alternate,
            offset: self.offset,
            len: self.len,
        };
        if let Some(reader) = self
            .backing
            .mirrors
            .lock()
            .map_err(|_| io::Error::other("mirror cache poisoned"))?
            .get(&key)
            .cloned()
        {
            if FileGeneration::read(&reader.file)? != alternate {
                return Err(io::Error::other(
                    "mirror generation changed after verification",
                ));
            }
            return Ok(reader);
        }
        let cached_file = self
            .backing
            .mirror_files
            .lock()
            .map_err(|_| io::Error::other("mirror file cache poisoned"))?
            .get(&alternate)
            .cloned();
        let file = if let Some(file) = &cached_file {
            use std::os::fd::AsRawFd;
            if FileGeneration::read(file)? != alternate {
                return Err(io::Error::other("mirror generation changed after opening"));
            }
            // A temporary buffered descriptor verifies another window of the same retained inode.
            // The persistent direct descriptor is shared by every admitted expert window.
            Arc::new(File::open(format!("/proc/self/fd/{}", file.as_raw_fd()))?)
        } else {
            Arc::new(File::open(path)?)
        };
        if FileGeneration::read(&file)? != alternate {
            return Err(io::Error::other(
                "opened mirror generation differs from its declaration",
            ));
        }
        let primary = self.reader(BoundReadMode::Buffered)?;
        let candidate = BoundDiskReader {
            file: file.clone(),
            offset: self.offset,
            len: self.len,
            direct: false,
        };
        let mut left = vec![0; 65536.min(self.len)];
        let mut right = vec![0; left.len()];
        let mut offset = 0;
        while offset < self.len {
            let len = left.len().min(self.len - offset);
            primary.read_exact_at(&mut left[..len], offset as u64)?;
            candidate.read_exact_at(&mut right[..len], offset as u64)?;
            if left[..len] != right[..len] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "mirror bytes differ from the authorized tensor window",
                ));
            }
            offset += len;
        }
        let direct = match cached_file {
            Some(file) => file,
            None => retained_direct_file(&file)?,
        };
        if self.file_generation()? != source || FileGeneration::read(&direct)? != alternate {
            return Err(io::Error::other(
                "source or mirror changed during byte verification",
            ));
        }
        let direct = self
            .backing
            .mirror_files
            .lock()
            .map_err(|_| io::Error::other("mirror file cache poisoned"))?
            .entry(alternate)
            .or_insert(direct)
            .clone();
        let reader = BoundDiskReader {
            file: direct,
            offset: self.offset,
            len: self.len,
            direct: true,
        };
        if self.file_generation()? != source || FileGeneration::read(&reader.file)? != alternate {
            return Err(io::Error::other(
                "source or mirror changed during byte verification",
            ));
        }
        let mut cache = self
            .backing
            .mirrors
            .lock()
            .map_err(|_| io::Error::other("mirror cache poisoned"))?;
        Ok(cache.entry(key).or_insert(reader).clone())
    }

    pub fn read_at(&self, dst: &mut [u8], relative: u64) -> io::Result<usize> {
        self.reader(BoundReadMode::Buffered)?.read_at(dst, relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_disk_extent_cannot_read_or_join_outside_its_authorized_range() {
        let path = std::env::temp_dir().join(format!("memra-bound-disk-{}", std::process::id()));
        std::fs::write(&path, b"unselectedSELECTEDunselected").unwrap();
        let file = Arc::new(std::fs::File::open(&path).unwrap());
        let map = Arc::new(unsafe { memmap2::Mmap::map(file.as_ref()).unwrap() });
        let view = BoundDiskView::new(
            DiskExtent {
                map,
                file,
                offset: 10,
                len: 8,
            },
            Arc::from("test-scope"),
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(view.bytes(), b"SELECTED");
        let mut bytes = [0; 8];
        assert_eq!(view.read_at(&mut bytes, 0).unwrap(), 8);
        assert_eq!(&bytes, b"SELECTED");
        assert!(view.read_at(&mut bytes, 1).is_err());
        assert!(view.read_at(&mut bytes, u64::MAX).is_err());
        assert!(view.subrange(0..9).is_err());
        let a = view.subrange(0..3).unwrap();
        let b = view.subrange(3..8).unwrap();
        assert_eq!(a.join_adjacent(&b).unwrap().bytes(), b"SELECTED");
        let bounded = ExpertDiskView::Bound(a.clone());
        let next = ExpertDiskView::Bound(b.clone());
        let combined = bounded.join_adjacent(&next).unwrap();
        assert_eq!(combined.len(), 8);
        assert!(matches!(combined, ExpertDiskView::Bound(_)));
        let raw_next = ExpertDiskView::Unbound(DiskExtent {
            map: b.backing.map.clone(),
            file: b.backing.file.clone(),
            offset: b.offset,
            len: b.len,
        });
        assert!(bounded.join_adjacent(&raw_next).is_err());
        assert_eq!(raw_next.bytes(), b"ECTED");
        assert_eq!(raw_next.subrange(1..4).unwrap().bytes(), b"CTE");
        assert!(raw_next.subrange(0..6).is_err());
        assert_eq!(bounded.subrange(1..3).unwrap().bytes(), b"EL");
        assert!(bounded.subrange(0..4).is_err());

        let mut other_scope = b.clone();
        other_scope.authority = Arc::from("different-scope");
        assert!(a.join_adjacent(&other_scope).is_err());
        assert!(a.join_adjacent(&view.subrange(4..8).unwrap()).is_err());
    }

    #[test]
    fn worker_readers_retain_the_opened_inode_and_reject_out_of_range_reads() {
        let path = std::env::temp_dir().join(format!("memra-bound-worker-{}", std::process::id()));
        std::fs::write(&path, b"leftSELECTEDright").unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let map = Arc::new(unsafe { memmap2::Mmap::map(file.as_ref()).unwrap() });
        let cache = BoundDiskCache::default();
        let extent = DiskExtent {
            map,
            file,
            offset: 4,
            len: 8,
        };
        let view = cache
            .view(extent.clone(), Arc::from("worker-test"))
            .unwrap();
        let other = cache.view(extent, Arc::from("worker-test")).unwrap();
        assert!(Arc::ptr_eq(&view.backing, &other.backing));
        assert_eq!(view.backing_id(), other.backing_id());
        assert_eq!(view.backing_id(), view.subrange(2..4).unwrap().backing_id());
        let readers =
            [BoundReadMode::Buffered, BoundReadMode::Random].map(|mode| view.reader(mode).unwrap());
        assert!(view.reader(BoundReadMode::Direct).is_err());
        assert!(!view.advise_willneed(7, 2));
        assert!(!view.advise_willneed(usize::MAX, 2));
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"leftREPLACEDright").unwrap();
        drop((view, other, cache));
        for reader in readers {
            std::thread::spawn(move || {
                let mut dst = [0; 8];
                assert_eq!(reader.read_at(&mut dst, 0).unwrap(), 8);
                assert_eq!(&dst, b"SELECTED");
                assert_eq!(
                    reader.read_at(&mut dst, 1).unwrap_err().kind(),
                    io::ErrorKind::InvalidInput
                );
                assert_eq!(
                    reader.read_at(&mut dst, u64::MAX).unwrap_err().kind(),
                    io::ErrorKind::InvalidInput
                );
            })
            .join()
            .unwrap();
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn direct_reader_checks_destination_alignment_without_reading() {
        let path = std::env::temp_dir().join(format!("memra-bound-direct-{}", std::process::id()));
        std::fs::write(&path, vec![1; 8192]).unwrap();
        // The check is independent of O_DIRECT availability and runs on all host platforms.
        let reader = BoundDiskReader {
            file: Arc::new(File::open(&path).unwrap()),
            offset: 0,
            len: 8192,
            direct: true,
        };
        #[repr(align(4096))]
        struct Aligned([u8; 8192]);
        let mut buffer = Aligned([0; 8192]);
        for (start, len, relative) in [(1, 4096, 0), (0, 4095, 0), (0, 4096, 1), (0, 4096, 8192)] {
            assert_eq!(
                reader
                    .read_at(&mut buffer.0[start..start + len], relative)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::InvalidInput
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
