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

/// Buffers in the direct writer's ring: one being filled, up to `DIRECT_RING - 1` in flight.
pub(crate) const DIRECT_RING: usize = 4;

type Block = (AlignedBuf, usize, u64);

/// `O_DIRECT` writer (section E v2, the pipelined arm): the serializer fills one 4096-aligned
/// buffer while a dedicated writer thread writes the previous ones, each a whole aligned block at an
/// ascending aligned offset. A write error stops the thread and surfaces on the next call; the
/// caller's export then fails and removes its `.tmp` file.
pub(crate) struct DirectWriter {
    file: std::sync::Arc<File>,
    buf: Option<AlignedBuf>,
    fill: usize,
    offset: u64,
    blocks: Option<std::sync::mpsc::SyncSender<Block>>,
    free: std::sync::mpsc::Receiver<AlignedBuf>,
    writer: Option<std::thread::JoinHandle<io::Result<()>>>,
}

impl DirectWriter {
    pub(crate) fn create(path: &str) -> io::Result<DirectWriter> {
        Self::create_with_fault(path, None)
    }

    /// `fail_at_block`: a test seam; the writer thread fails that block (0-based) with EIO.
    fn create_with_fault(path: &str, fail_at_block: Option<u64>) -> io::Result<DirectWriter> {
        let file = std::sync::Arc::new(open_direct(path, true)?);
        let (blocks, pending) = std::sync::mpsc::sync_channel::<Block>(DIRECT_RING);
        let (give_back, free) = std::sync::mpsc::sync_channel::<AlignedBuf>(DIRECT_RING);
        for _ in 1..DIRECT_RING {
            give_back
                .send(AlignedBuf::new(HANDOFF_BUF))
                .expect("fresh ring has room");
        }
        let target = file.clone();
        let writer = std::thread::Builder::new()
            .name("memra-handoff-direct".into())
            .spawn(move || -> io::Result<()> {
                let mut n = 0u64;
                for (buf, len, offset) in pending {
                    debug_assert!(len.is_multiple_of(DIRECT_ALIGN));
                    if fail_at_block == Some(n) {
                        return Err(io::Error::other("injected direct write fault"));
                    }
                    target.write_all_at(&buf.as_slice()[..len], offset)?;
                    n += 1;
                    // The owner may already be finishing; a closed ring is not an error.
                    let _ = give_back.send(buf);
                }
                Ok(())
            })?;
        Ok(DirectWriter {
            file,
            buf: Some(AlignedBuf::new(HANDOFF_BUF)),
            fill: 0,
            offset: 0,
            blocks: Some(blocks),
            free,
            writer: Some(writer),
        })
    }

    /// The writer thread's outcome once it has stopped (the ring closed or it failed).
    fn writer_error(&mut self) -> io::Error {
        self.blocks = None;
        match self.writer.take().map(|h| h.join()) {
            Some(Ok(Err(e))) => {
                io::Error::new(e.kind(), format!("handoff direct write failed: {e}"))
            }
            Some(Err(_)) => io::Error::other("handoff direct writer thread panicked"),
            _ => io::Error::other("handoff direct writer stopped"),
        }
    }

    /// Hand the current buffer (its first `len` bytes) to the writer thread at the next offset.
    fn submit(&mut self, len: usize) -> io::Result<()> {
        let buf = self.buf.take().expect("a current buffer");
        let offset = self.offset;
        let sent = self
            .blocks
            .as_ref()
            .map(|tx| tx.send((buf, len, offset)).is_ok())
            .unwrap_or(false);
        if !sent {
            return Err(self.writer_error());
        }
        self.offset += len as u64;
        self.fill = 0;
        Ok(())
    }

    fn next_buffer(&mut self) -> io::Result<()> {
        match self.free.recv() {
            Ok(buf) => {
                self.buf = Some(buf);
                Ok(())
            }
            Err(_) => Err(self.writer_error()),
        }
    }

    /// Pad and submit the tail, wait for every write, truncate to the logical length. Returns
    /// the file for the durability step.
    fn complete(&mut self) -> io::Result<()> {
        let logical = self.offset + self.fill as u64;
        if self.fill > 0 {
            let padded = self.fill.div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
            let fill = self.fill;
            self.buf.as_mut().expect("a current buffer").as_mut_slice()[fill..padded].fill(0);
            self.submit(padded)?;
        }
        self.blocks = None; // close the ring: the writer drains it and returns
        match self.writer.take().map(|h| h.join()) {
            Some(Ok(Ok(()))) | None => {}
            Some(Ok(Err(e))) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("handoff direct write failed: {e}"),
                ));
            }
            Some(Err(_)) => return Err(io::Error::other("handoff direct writer thread panicked")),
        }
        self.file.set_len(logical)
    }
}

impl Write for DirectWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.buf.is_none() {
            self.next_buffer()?;
        }
        let cap = HANDOFF_BUF;
        let n = data.len().min(cap - self.fill);
        let fill = self.fill;
        self.buf.as_mut().expect("a current buffer").as_mut_slice()[fill..fill + n]
            .copy_from_slice(&data[..n]);
        self.fill += n;
        if self.fill == cap {
            self.submit(cap)?;
        }
        Ok(n)
    }

    /// Buffered bytes stay buffered: a partial block cannot be written with `O_DIRECT`
    /// before `complete` pads it.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for DirectWriter {
    fn drop(&mut self) {
        // An abandoned export (an error upstream): stop the writer before the file goes.
        self.blocks = None;
        if let Some(h) = self.writer.take() {
            let _ = h.join();
        }
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
                let logical = w.offset + w.fill as u64;
                Ok(FinishedHandoff::Direct {
                    pending: Some(w),
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
    /// Write the direct arm's tail and wait for every in-flight write (part of the write stage in
    /// both arms), then truncate to the logical length.
    pub(crate) fn complete_writes(&mut self) -> Result<(), String> {
        if let FinishedHandoff::Direct {
            pending: Some(w),
            logical,
        } = self
        {
            w.complete()
                .map_err(|e| format!("handoff direct tail or truncate to {logical} failed: {e}"))?;
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
            eprintln!("SKIP[O_DIRECT filesystem]: direct writer and reader parity not run");
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
    fn a_direct_write_fault_fails_the_export() {
        if !direct_supported() {
            eprintln!("SKIP: the test filesystem refuses O_DIRECT");
            return;
        }
        // Section E v2: the writer thread fails block 1 of 3; the error must surface, never a
        // silently short file.
        let path = scratch("fault");
        let mut w = DirectWriter::create_with_fault(&path, Some(1)).unwrap();
        let data = payload(3 * HANDOFF_BUF);
        let mut failed = None;
        for chunk in data.chunks(1 << 20) {
            if let Err(e) = w.write_all(chunk) {
                failed = Some(e);
                break;
            }
        }
        let err = match failed {
            Some(e) => e,
            None => w.complete().unwrap_err(),
        };
        assert!(
            err.to_string().contains("injected direct write fault"),
            "{err}"
        );
        drop(w);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_ring_keeps_writes_in_order_across_many_blocks() {
        if !direct_supported() {
            eprintln!("SKIP: the test filesystem refuses O_DIRECT");
            return;
        }
        // More blocks than the ring holds, written in small pieces: offsets must stay ascending
        // and the file byte-identical to the buffered arm.
        let data = payload(9 * HANDOFF_BUF + 777);
        let (pb, pd) = (scratch("ring-b"), scratch("ring-d"));
        assert_eq!(
            write_with(&pb, HandoffIo::Buffered, &data, 4093),
            data.len() as u64
        );
        assert_eq!(
            write_with(&pd, HandoffIo::Direct, &data, 4093),
            data.len() as u64
        );
        assert_eq!(std::fs::read(&pb).unwrap(), std::fs::read(&pd).unwrap());
        let _ = std::fs::remove_file(&pb);
        let _ = std::fs::remove_file(&pd);
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
