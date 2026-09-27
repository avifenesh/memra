//! Retained C-ABI reader for a single authorized extent. No descriptor or widening operation.
//! Native consumers must retain before detached use and release exactly once after draining it.
use super::BoundDiskReader;
use std::{
    ffi::c_void,
    io,
    sync::{Arc, Weak},
};

/// Process-native cache-key metadata, matching the existing CPU file-key inputs. Offset is
/// informational; all reads still take checked offsets relative to the authorized window.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopedDiskInfoV1 {
    pub device: u64,
    pub inode: u64,
    pub file_bytes: u64,
    pub ctime_seconds: i64,
    pub ctime_nanoseconds: i64,
    pub offset: u64,
    pub len: u64,
    pub direct: u32,
}

/// Borrowed ABI descriptor. Copying this value does not retain its context. Every callback is
/// unsafe because the foreign caller must uphold pointer validity and balanced ownership.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScopedDiskReaderV1 {
    pub version: u32,
    pub context: *const c_void,
    pub retain: unsafe extern "C" fn(*const c_void),
    pub release: unsafe extern "C" fn(*const c_void),
    pub read_at: unsafe extern "C" fn(*const c_void, *mut u8, usize, u64, *mut usize) -> i32,
    pub info: unsafe extern "C" fn(*const c_void, *mut ScopedDiskInfoV1) -> i32,
}

/// Rust owner of one prepared reader. The ABI borrow remains valid while this owner lives; an
/// asynchronous native consumer must create its own reference with `retain` before returning.
pub struct ScopedDiskReaderHandle {
    reader: Arc<BoundDiskReader>,
}
/// Non-owning lifetime diagnostic. It cannot read bytes or acquire a strong reference.
pub struct ScopedDiskReaderLifetime(Weak<BoundDiskReader>);
impl ScopedDiskReaderLifetime {
    pub fn is_alive(&self) -> bool {
        self.0.strong_count() != 0
    }
}
impl ScopedDiskReaderHandle {
    pub fn lifetime(&self) -> ScopedDiskReaderLifetime {
        ScopedDiskReaderLifetime(Arc::downgrade(&self.reader))
    }
    pub fn as_abi(&self) -> ScopedDiskReaderV1 {
        ScopedDiskReaderV1 {
            version: 1,
            context: Arc::as_ptr(&self.reader).cast(),
            retain,
            release,
            read_at,
            info,
        }
    }
}
impl BoundDiskReader {
    pub fn into_ffi_handle(self) -> ScopedDiskReaderHandle {
        ScopedDiskReaderHandle {
            reader: Arc::new(self),
        }
    }
}

unsafe extern "C" fn retain(context: *const c_void) {
    // SAFETY: ABI requires the live context obtained from as_abi, and one release per retain.
    unsafe { Arc::increment_strong_count(context.cast::<BoundDiskReader>()) };
}
unsafe extern "C" fn release(context: *const c_void) {
    // SAFETY: ABI requires one live reference acquired by retain, consumed exactly once.
    unsafe { Arc::decrement_strong_count(context.cast::<BoundDiskReader>()) };
}
fn error_code(error: io::Error) -> i32 {
    error.raw_os_error().unwrap_or(match error.kind() {
        io::ErrorKind::InvalidInput => libc::EINVAL,
        io::ErrorKind::Unsupported => libc::ENOTSUP,
        io::ErrorKind::Interrupted => libc::EINTR,
        _ => libc::EIO,
    })
}
unsafe extern "C" fn read_at(
    context: *const c_void,
    destination: *mut u8,
    len: usize,
    relative_offset: u64,
    bytes_read: *mut usize,
) -> i32 {
    if bytes_read.is_null() {
        return libc::EINVAL;
    }
    // SAFETY: non-null out pointer must point to writable usize storage, per the ABI.
    unsafe { *bytes_read = 0 };
    if context.is_null() || (destination.is_null() && len != 0) || len > isize::MAX as usize {
        return libc::EINVAL;
    }
    // SAFETY: the caller retains this context throughout the callback.
    let reader = unsafe { &*context.cast::<BoundDiskReader>() };
    // Refuse out-of-range access before constructing a slice over foreign memory.
    if relative_offset
        .checked_add(len as u64)
        .is_none_or(|end| end > reader.len as u64)
    {
        return libc::EINVAL;
    }
    let dst = if len == 0 {
        &mut []
    } else {
        // SAFETY: ABI requires unique writable destination storage for len bytes.
        unsafe { std::slice::from_raw_parts_mut(destination, len) }
    };
    match reader.read_at(dst, relative_offset) {
        Ok(count) => {
            unsafe { *bytes_read = count };
            0
        }
        Err(error) => error_code(error),
    }
}
unsafe extern "C" fn info(context: *const c_void, out: *mut ScopedDiskInfoV1) -> i32 {
    if context.is_null() || out.is_null() {
        return libc::EINVAL;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: ABI requires a live retained context and a writable output object.
        let reader = unsafe { &*context.cast::<BoundDiskReader>() };
        match reader.file.metadata() {
            Ok(metadata) => {
                unsafe {
                    *out = ScopedDiskInfoV1 {
                        device: metadata.dev(),
                        inode: metadata.ino(),
                        file_bytes: metadata.len(),
                        ctime_seconds: metadata.ctime(),
                        ctime_nanoseconds: metadata.ctime_nsec(),
                        offset: reader.offset,
                        len: reader.len as u64,
                        direct: u32::from(reader.direct),
                    }
                };
                0
            }
            Err(error) => error_code(error),
        }
    }
    #[cfg(not(unix))]
    {
        libc::ENOTSUP
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{
        bound_disk::{BoundDiskCache, BoundReadMode},
        source::DiskExtent,
    };
    use std::{
        fs::File,
        sync::atomic::{AtomicUsize, Ordering},
    };

    // Test consumer follows the same explicit retain/drain/release contract required of C++.
    struct Lease(ScopedDiskReaderV1);
    unsafe impl Send for Lease {} // SAFETY: one retained Send+Sync reader; callbacks are range checked.
    impl Drop for Lease {
        fn drop(&mut self) {
            unsafe { (self.0.release)(self.0.context) };
        }
    }
    impl Lease {
        fn retained(abi: ScopedDiskReaderV1) -> Self {
            unsafe { (abi.retain)(abi.context) };
            Self(abi)
        }
        fn read(&self, dst: &mut [u8], offset: u64) -> (i32, usize) {
            let mut n = 999;
            let code = unsafe {
                (self.0.read_at)(self.0.context, dst.as_mut_ptr(), dst.len(), offset, &mut n)
            };
            (code, n)
        }
    }
    fn fixture() -> (std::path::PathBuf, ScopedDiskReaderHandle) {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "memra-scoped-ffi-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"outsideSELECTEDoutside").unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let map = Arc::new(unsafe { memmap2::Mmap::map(file.as_ref()).unwrap() });
        let view = BoundDiskCache::default()
            .view(
                DiskExtent {
                    map,
                    file,
                    offset: 7,
                    len: 8,
                },
                Arc::from("ffi-test"),
            )
            .unwrap();
        (
            path,
            view.reader(BoundReadMode::Buffered)
                .unwrap()
                .into_ffi_handle(),
        )
    }
    #[test]
    fn retained_foreign_reader_survives_owner_drop_and_path_replacement() {
        let (path, owner) = fixture();
        let weak = Arc::downgrade(&owner.reader);
        let abi = owner.as_abi();
        assert_eq!(abi.version, 1);
        let first = Lease::retained(abi);
        let second = Lease::retained(abi);
        drop(owner);
        std::fs::remove_file(&path).unwrap();
        std::fs::write(&path, b"changed").unwrap();
        for lease in [first, second] {
            std::thread::spawn(move || {
                let mut out = [0; 8];
                assert_eq!(lease.read(&mut out, 0), (0, 8));
                assert_eq!(&out, b"SELECTED");
                assert_eq!(lease.read(&mut out, 1), (libc::EINVAL, 0));
                assert_eq!(lease.read(&mut out, u64::MAX), (libc::EINVAL, 0));
            })
            .join()
            .unwrap();
        }
        assert!(weak.upgrade().is_none());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn metadata_and_foreign_pointer_errors_do_not_expose_a_descriptor() {
        let (path, owner) = fixture();
        let abi = owner.as_abi();
        let mut metadata = ScopedDiskInfoV1::default();
        assert_eq!(unsafe { (abi.info)(abi.context, &mut metadata) }, 0);
        use std::os::unix::fs::MetadataExt;
        let raw = std::fs::metadata(&path).unwrap();
        assert_eq!(
            (metadata.device, metadata.inode, metadata.file_bytes),
            (raw.dev(), raw.ino(), raw.len())
        );
        assert_eq!((metadata.offset, metadata.len, metadata.direct), (7, 8, 0));
        let mut n = 999;
        assert_eq!(
            unsafe { (abi.read_at)(abi.context, std::ptr::null_mut(), 1, 0, &mut n) },
            libc::EINVAL
        );
        assert_eq!(n, 0);
        assert_eq!(
            unsafe { (abi.read_at)(abi.context, std::ptr::null_mut(), 0, 8, &mut n) },
            0
        );
        assert_eq!(n, 0);
        assert_eq!(
            unsafe { (abi.read_at)(abi.context, std::ptr::null_mut(), 0, 9, &mut n) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { (abi.info)(std::ptr::null(), &mut metadata) },
            libc::EINVAL
        );
        std::fs::remove_file(path).unwrap();
    }
}
