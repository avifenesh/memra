//! Host-tier handoff file I/O (lane/spill-f-20260919, OWED 18, `M1-PREREG.md` section E;
//! `docs/decisions/HANDOFF-DIRECT-EXPORT.md`).
//!
//! The export writes the handoff file one of two ways, chosen by the worker's card:
//! - `direct` on the full-power RTX PRO 6000 Blackwell Workstation Edition, where the registered
//!   cell read the export 8.3 to 8.4% shorter: `O_DIRECT` through one 4096-aligned 4 MiB buffer.
//!   Full buffers go straight to the device; the last block is zero-padded, the file is
//!   truncated to its logical length, then `fdatasync`. A filesystem that refuses `O_DIRECT` at
//!   open (`EINVAL`: it has no direct path) gets the buffered writer, and the export line says so.
//! - `buffered` everywhere else: a 4 MiB `BufWriter` through the page cache, then `fsync`.
//!
//! The import reads through a 4 MiB `BufReader` on every card: the direct reader read flat on
//! both rigs and is deleted. The wire format does not depend on the writer: both writers produce
//! the same bytes.

use std::fs::File;
use std::io::{self, Write};
use std::os::unix::fs::{FileExt, OpenOptionsExt};

/// Alignment for the direct writer's buffer, offsets and lengths. 4096 covers 512-byte and
/// 4 KiB logical-block devices.
pub(crate) const DIRECT_ALIGN: usize = 4096;
/// Buffer size for both writers and the reader (the buffered program's historical 4 MiB).
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

    /// The export writer for a device: `direct` on the RTX PRO 6000 Blackwell Workstation
    /// Edition, full power and Max-Q, each from its own cell (10 pairs per size; export of
    /// buffered: BOX36 at 600 W 0.917 at 1 GiB and 0.916 at 8 GiB, the Max-Q at a 300 W limit
    /// 0.860 and 0.916, `research/spill-f-20260919/pro-maxq-owed18/`), `buffered` on every other
    /// card (the RTX 5090's 1 GiB cell read flat; the Server Edition has no cell).
    pub(crate) fn for_device(device_name: &str) -> HandoffIo {
        use memra_engine::parallel::HardwareTarget;
        if matches!(
            HardwareTarget::from_device_name(device_name),
            Ok(HardwareTarget::RtxPro6000Blackwell)
        ) && device_name.contains("Workstation")
        {
            HandoffIo::Direct
        } else {
            HandoffIo::Buffered
        }
    }
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

/// `O_DIRECT` writer: every device write is a whole aligned buffer at an aligned offset.
pub(crate) struct DirectWriter {
    file: File,
    buf: AlignedBuf,
    fill: usize,
    offset: u64,
}

impl DirectWriter {
    /// Opens `path` with `O_DIRECT`. The error keeps the open's kind, so a caller can tell a
    /// filesystem without a direct path (`InvalidInput`, `EINVAL`) from every other failure.
    pub(crate) fn create(path: &str) -> io::Result<DirectWriter> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| {
                io::Error::new(e.kind(), format!("O_DIRECT open refused for {path}: {e}"))
            })?;
        Ok(DirectWriter {
            file,
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

/// The export side of either writer.
pub(crate) enum HandoffWriter {
    Buffered(io::BufWriter<File>),
    Direct(DirectWriter),
}

impl HandoffWriter {
    /// Exactly `mode`: the direct writer fails loudly where `O_DIRECT` is refused.
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

    /// The export's writer for the card's `mode`. A filesystem that refuses `O_DIRECT` at open
    /// (`EINVAL`) gets the buffered writer, the program every earlier handoff ran, and the
    /// second value carries the refusal for the export line; any other open failure fails.
    pub(crate) fn create_for_export(
        path: &str,
        mode: HandoffIo,
    ) -> Result<(HandoffWriter, Option<String>), String> {
        if mode == HandoffIo::Direct {
            return match DirectWriter::create(path) {
                Ok(w) => Ok((HandoffWriter::Direct(w), None)),
                Err(e) if e.kind() == io::ErrorKind::InvalidInput => {
                    HandoffWriter::create(path, HandoffIo::Buffered)
                        .map(|w| (w, Some(e.to_string())))
                }
                Err(e) => Err(format!("create {path}: {e}")),
            };
        }
        HandoffWriter::create(path, mode).map(|w| (w, None))
    }

    /// The writer this export actually runs.
    pub(crate) fn mode(&self) -> HandoffIo {
        match self {
            HandoffWriter::Buffered(_) => HandoffIo::Buffered,
            HandoffWriter::Direct(_) => HandoffIo::Direct,
        }
    }

    /// Flush every byte to the file: the buffered writer's `into_inner`, the direct writer's
    /// padded tail plus truncate. Durability is `sync`, measured apart by the caller.
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
    /// The direct writer's tail block is written in `complete_writes` so the caller's write
    /// clock covers the same work for both writers: bytes handed to the kernel (buffered) or
    /// the device (direct).
    Direct {
        pending: Option<DirectWriter>,
        logical: u64,
    },
}

impl FinishedHandoff {
    /// Write the direct writer's tail now (part of the write stage for both writers).
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

/// The import side: open `path` for a buffered read and return the reader plus the file's
/// length (the frame-size bound). Either writer's file reads here.
pub(crate) fn open_import_reader(path: &str) -> Result<(io::BufReader<File>, u64), String> {
    let f = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let len = f.metadata().map_or(u64::MAX, |m| m.len());
    Ok((io::BufReader::with_capacity(HANDOFF_BUF, f), len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

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

    fn read_back(path: &str) -> Vec<u8> {
        let (mut r, _) = open_import_reader(path).unwrap();
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
    fn both_writers_produce_identical_bytes_and_the_import_reads_both() {
        if !direct_supported() {
            eprintln!("SKIP[O_DIRECT filesystem]: direct writer parity not run");
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
                    assert_eq!(read_back(p), data, "len {len}");
                }
                let _ = std::fs::remove_file(&pb);
                let _ = std::fs::remove_file(&pd);
            }
        }
    }

    /// The direct export is the card default of the full-power Workstation Edition only.
    #[test]
    fn the_export_writer_follows_the_card() {
        for name in [
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        ] {
            assert_eq!(HandoffIo::for_device(name), HandoffIo::Direct, "{name}");
        }
        for name in [
            "NVIDIA GeForce RTX 5090",
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA B200",
            "",
        ] {
            assert_eq!(HandoffIo::for_device(name), HandoffIo::Buffered, "{name}");
        }
    }

    #[test]
    fn a_refusing_filesystem_fails_the_strict_writer_and_buffers_the_export() {
        // /dev/shm is tmpfs, which refuses O_DIRECT on older Linux kernels.
        let p = format!("/dev/shm/memra-handoff-io-refuse-{}", std::process::id());
        match HandoffWriter::create(&p, HandoffIo::Direct) {
            Err(e) => {
                assert!(e.contains("O_DIRECT open refused"), "{e}");
                let (w, refused) = HandoffWriter::create_for_export(&p, HandoffIo::Direct).unwrap();
                assert_eq!(w.mode(), HandoffIo::Buffered);
                assert!(
                    refused.is_some_and(|r| r.contains("O_DIRECT open refused")),
                    "the export line must carry the refusal"
                );
            }
            Ok(_) => eprintln!("NOTE: this kernel's tmpfs accepts O_DIRECT; refusal not exercised"),
        }
        let _ = std::fs::remove_file(&p);
    }
}
