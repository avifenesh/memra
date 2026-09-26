//! Host-tier handoff file I/O (lane/spill-f-20260919, OWED 18, `M1-PREREG.md` section E).
//!
//! `MEMRA_KV_HOST_HANDOFF_IO` selects how the handoff file meets storage:
//! - `buffered` (default): a 4 MiB `BufWriter` / `BufReader` through the page cache, then
//!   `fsync`. The program every earlier handoff receipt measured.
//! - `direct`: `O_DIRECT` through one 4096-aligned 4 MiB buffer. Full buffers go straight to the
//!   device; the last block is zero-padded, the file is truncated to its logical length, then
//!   `fdatasync`. Reads use the same aligned buffer shape.
//!
//! The wire format does not depend on the mode: both writers produce the same bytes and both
//! readers accept either file. A filesystem that refuses `O_DIRECT` fails the call loudly; the
//! direct arm never falls back to buffered I/O.

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};

/// Alignment for the direct arm's buffer, offsets and lengths. 4096 covers 512-byte and
/// 4 KiB logical-block devices.
pub(crate) const DIRECT_ALIGN: usize = 4096;
/// Buffer size for both arms (the buffered arm's historical 4 MiB).
pub(crate) const HANDOFF_BUF: usize = 4 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HandoffIo {
    Buffered,
    Direct,
}

impl HandoffIo {
    pub(crate) fn name(self) -> &'static str {
        match self {
            HandoffIo::Buffered => "buffered",
            HandoffIo::Direct => "direct",
        }
    }

    pub(crate) fn parse(v: &str) -> Result<HandoffIo, String> {
        match v {
            "" | "buffered" => Ok(HandoffIo::Buffered),
            "direct" => Ok(HandoffIo::Direct),
            other => Err(format!(
                "MEMRA_KV_HOST_HANDOFF_IO={other:?}: expected buffered or direct"
            )),
        }
    }
}

/// MEMRA_KV_HOST_HANDOFF_IO (default `buffered`; default-OFF door, see docs/FLAGS.md). An
/// unparseable value is an error at the first handoff, never a silent default.
pub(crate) fn handoff_io_mode() -> Result<HandoffIo, String> {
    static M: std::sync::OnceLock<Result<HandoffIo, String>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        HandoffIo::parse(&std::env::var("MEMRA_KV_HOST_HANDOFF_IO").unwrap_or_default())
    })
    .clone()
}

/// One heap block aligned to `DIRECT_ALIGN`.
struct AlignedBuf {
    ptr: std::ptr::NonNull<u8>,
    cap: usize,
}

// SAFETY: the buffer is uniquely owned heap memory with no interior sharing.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(cap: usize) -> AlignedBuf {
        assert!(cap > 0 && cap.is_multiple_of(DIRECT_ALIGN));
        let layout = std::alloc::Layout::from_size_align(cap, DIRECT_ALIGN).expect("layout");
        // SAFETY: the layout has non-zero size.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr =
            std::ptr::NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        AlignedBuf { ptr, cap }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `cap` initialized bytes (alloc_zeroed) owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.cap, DIRECT_ALIGN).expect("layout");
        // SAFETY: allocated in `new` with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) }
    }
}

fn open_direct(path: &str, write: bool) -> io::Result<File> {
    let mut o = std::fs::OpenOptions::new();
    if write {
        o.write(true).create(true).truncate(true);
    } else {
        o.read(true);
    }
    o.custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("O_DIRECT open refused for {path}: {e}")))
}

/// `O_DIRECT` writer: every device write is a whole aligned buffer at an aligned offset.
pub(crate) struct DirectWriter {
    file: File,
    buf: AlignedBuf,
    fill: usize,
    offset: u64,
}

impl DirectWriter {
    pub(crate) fn create(path: &str) -> io::Result<DirectWriter> {
        Ok(DirectWriter {
            file: open_direct(path, true)?,
            buf: AlignedBuf::new(HANDOFF_BUF),
            fill: 0,
            offset: 0,
        })
    }

    fn write_block(&mut self, len: usize) -> io::Result<()> {
        debug_assert!(len.is_multiple_of(DIRECT_ALIGN));
        self.file
            .write_all_at(&self.buf.as_slice()[..len], self.offset)?;
        self.offset += len as u64;
        Ok(())
    }
}

impl Write for DirectWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let n = data.len().min(self.buf.cap - self.fill);
        let fill = self.fill;
        self.buf.as_mut_slice()[fill..fill + n].copy_from_slice(&data[..n]);
        self.fill += n;
        if self.fill == self.buf.cap {
            self.write_block(self.buf.cap)?;
            self.fill = 0;
        }
        Ok(n)
    }

    /// Buffered bytes stay buffered: a partial block cannot be written with `O_DIRECT`
    /// before `finish` pads it.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `O_DIRECT` reader: every device read is one whole aligned buffer at an aligned offset;
/// the final read returns the short tail.
pub(crate) struct DirectReader {
    file: File,
    buf: AlignedBuf,
    pos: usize,
    len: usize,
    offset: u64,
    eof: bool,
}

impl DirectReader {
    pub(crate) fn open(path: &str) -> io::Result<DirectReader> {
        Ok(DirectReader {
            file: open_direct(path, false)?,
            buf: AlignedBuf::new(HANDOFF_BUF),
            pos: 0,
            len: 0,
            offset: 0,
            eof: false,
        })
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    fn refill(&mut self) -> io::Result<()> {
        let mut got = 0usize;
        while got < self.buf.cap {
            let off = self.offset + got as u64;
            match self.file.read_at(&mut self.buf.as_mut_slice()[got..], off) {
                Ok(0) => {
                    self.eof = true;
                    break;
                }
                Ok(n) => {
                    got += n;
                    // A short read that leaves an unaligned position can only be EOF.
                    if !got.is_multiple_of(DIRECT_ALIGN) {
                        self.eof = true;
                        break;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        self.offset += got as u64;
        self.pos = 0;
        self.len = got;
        Ok(())
    }
}

impl Read for DirectReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos == self.len {
            if self.eof {
                return Ok(0);
            }
            self.refill()?;
            if self.len == 0 {
                return Ok(0);
            }
        }
        let n = out.len().min(self.len - self.pos);
        out[..n].copy_from_slice(&self.buf.as_slice()[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// The export side of either arm.
pub(crate) enum HandoffWriter {
    Buffered(io::BufWriter<File>),
    Direct(DirectWriter),
}

impl HandoffWriter {
    pub(crate) fn create(path: &str, mode: HandoffIo) -> Result<HandoffWriter, String> {
        match mode {
            HandoffIo::Buffered => File::create(path)
                .map(|f| HandoffWriter::Buffered(io::BufWriter::with_capacity(HANDOFF_BUF, f)))
                .map_err(|e| format!("create {path}: {e}")),
            HandoffIo::Direct => DirectWriter::create(path)
                .map(HandoffWriter::Direct)
                .map_err(|e| format!("create {path}: {e}")),
        }
    }

    /// Flush every byte to the file: the buffered arm's `into_inner`, the direct arm's padded
    /// tail plus truncate. Durability is `sync`, measured apart by the caller.
    pub(crate) fn finish(self) -> Result<FinishedHandoff, String> {
        match self {
            HandoffWriter::Buffered(w) => w
                .into_inner()
                .map(FinishedHandoff::Buffered)
                .map_err(|e| format!("handoff flush failed: {e}")),
            HandoffWriter::Direct(w) => {
                let DirectWriter {
                    file,
                    buf,
                    fill,
                    offset,
                } = w;
                let logical = offset + fill as u64;
                Ok(FinishedHandoff::Direct {
                    pending: Some(DirectWriter {
                        file,
                        buf,
                        fill,
                        offset,
                    }),
                    logical,
                })
            }
        }
    }
}

impl Write for HandoffWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            HandoffWriter::Buffered(w) => w.write(data),
            HandoffWriter::Direct(w) => w.write(data),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            HandoffWriter::Buffered(w) => w.flush(),
            HandoffWriter::Direct(w) => w.flush(),
        }
    }
}

/// A written handoff before its durability step.
pub(crate) enum FinishedHandoff {
    Buffered(File),
    /// The direct arm's tail block is written in `sync` so the caller's write clock covers the
    /// same work in both arms: bytes handed to the kernel (buffered) or the device (direct).
    Direct {
        pending: Option<DirectWriter>,
        logical: u64,
    },
}

impl FinishedHandoff {
    /// Write the direct arm's tail now (part of the write stage in both arms).
    pub(crate) fn complete_writes(&mut self) -> Result<(), String> {
        if let FinishedHandoff::Direct {
            pending: Some(w),
            logical,
        } = self
        {
            let fill = w.fill;
            if fill > 0 {
                let padded = fill.div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
                w.buf.as_mut_slice()[fill..padded].fill(0);
                w.write_block(padded)
                    .map_err(|e| format!("handoff direct tail write failed: {e}"))?;
                w.fill = 0;
            }
            w.file
                .set_len(*logical)
                .map_err(|e| format!("handoff truncate to {logical} failed: {e}"))?;
        }
        Ok(())
    }

    /// Durability: `fsync` for buffered (writeback of the page cache), `fdatasync` for direct
    /// (the data is already on the device; this commits the size and the device cache).
    pub(crate) fn sync(&mut self) -> Result<(), String> {
        match self {
            FinishedHandoff::Buffered(f) => f.sync_all(),
            FinishedHandoff::Direct {
                pending: Some(w), ..
            } => w.file.sync_data(),
            FinishedHandoff::Direct { pending: None, .. } => Ok(()),
        }
        .map_err(|e| format!("handoff fsync failed: {e}"))
    }
}

/// The import side of either arm.
pub(crate) enum HandoffReader {
    Buffered(io::BufReader<File>),
    Direct(DirectReader),
}

impl HandoffReader {
    /// Open `path` and return the reader plus the file's length.
    pub(crate) fn open(path: &str, mode: HandoffIo) -> Result<(HandoffReader, u64), String> {
        match mode {
            HandoffIo::Buffered => {
                let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
                let len = f.metadata().map_or(u64::MAX, |m| m.len());
                Ok((
                    HandoffReader::Buffered(io::BufReader::with_capacity(HANDOFF_BUF, f)),
                    len,
                ))
            }
            HandoffIo::Direct => {
                let r = DirectReader::open(path).map_err(|e| format!("open {path}: {e}"))?;
                let len = r.file().metadata().map_or(u64::MAX, |m| m.len());
                Ok((HandoffReader::Direct(r), len))
            }
        }
    }
}

impl Read for HandoffReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        match self {
            HandoffReader::Buffered(r) => r.read(out),
            HandoffReader::Direct(r) => r.read(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> String {
        // The crate's target dir sits on a real filesystem; tmpfs refuses O_DIRECT.
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/handoff-io-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}-{}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn payload(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 7)) as u8)
            .collect()
    }

    fn write_with(path: &str, mode: HandoffIo, data: &[u8], chunk: usize) -> u64 {
        let mut w = HandoffWriter::create(path, mode).unwrap();
        for c in data.chunks(chunk.max(1)) {
            w.write_all(c).unwrap();
        }
        let mut f = w.finish().unwrap();
        f.complete_writes().unwrap();
        f.sync().unwrap();
        std::fs::metadata(path).unwrap().len()
    }

    fn read_with(path: &str, mode: HandoffIo) -> Vec<u8> {
        let (mut r, _) = HandoffReader::open(path, mode).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        out
    }

    fn direct_supported() -> bool {
        let p = scratch("probe");
        let ok = DirectWriter::create(&p).is_ok();
        let _ = std::fs::remove_file(&p);
        ok
    }

    #[test]
    fn both_writers_produce_identical_bytes_and_both_readers_read_both() {
        if !direct_supported() {
            eprintln!("SKIP: the test filesystem refuses O_DIRECT");
            return;
        }
        for len in [
            0,
            1,
            DIRECT_ALIGN - 1,
            DIRECT_ALIGN,
            DIRECT_ALIGN + 1,
            HANDOFF_BUF - 1,
            HANDOFF_BUF,
            HANDOFF_BUF + 1,
            2 * HANDOFF_BUF + 12345,
        ] {
            let data = payload(len);
            for chunk in [7, 4096, 1 << 20, HANDOFF_BUF + 3] {
                let (pb, pd) = (scratch("b"), scratch("d"));
                assert_eq!(
                    write_with(&pb, HandoffIo::Buffered, &data, chunk),
                    len as u64
                );
                assert_eq!(write_with(&pd, HandoffIo::Direct, &data, chunk), len as u64);
                assert_eq!(
                    std::fs::read(&pb).unwrap(),
                    std::fs::read(&pd).unwrap(),
                    "len {len} chunk {chunk}"
                );
                for p in [&pb, &pd] {
                    assert_eq!(read_with(p, HandoffIo::Buffered), data, "len {len}");
                    assert_eq!(read_with(p, HandoffIo::Direct), data, "len {len}");
                }
                let _ = std::fs::remove_file(&pb);
                let _ = std::fs::remove_file(&pd);
            }
        }
    }

    #[test]
    fn mode_parse_refuses_unknown_values() {
        assert_eq!(HandoffIo::parse("").unwrap(), HandoffIo::Buffered);
        assert_eq!(HandoffIo::parse("buffered").unwrap(), HandoffIo::Buffered);
        assert_eq!(HandoffIo::parse("direct").unwrap(), HandoffIo::Direct);
        assert!(
            HandoffIo::parse("odirect")
                .unwrap_err()
                .contains("expected buffered or direct")
        );
    }

    #[test]
    fn direct_open_on_a_refusing_filesystem_fails_loudly() {
        // /dev/shm is tmpfs, which refuses O_DIRECT on Linux: the arm must not fall back.
        let p = format!("/dev/shm/memra-handoff-io-refuse-{}", std::process::id());
        match HandoffWriter::create(&p, HandoffIo::Direct) {
            Err(e) => assert!(e.contains("O_DIRECT open refused"), "{e}"),
            Ok(_) => eprintln!("NOTE: this kernel's tmpfs accepts O_DIRECT; refusal not exercised"),
        }
        let _ = std::fs::remove_file(&p);
    }
}
