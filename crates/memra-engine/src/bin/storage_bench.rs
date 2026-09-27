//! Development-Mac I/O characterization, NOT memra spill speed or GPU qualification.
//! This CPU executable can be linked to the workspace memra-tier rlib without nvcc.
use memra_tier::contracts::*;
use memra_tier::io::direct::{AlignedFile, ReadMode};
use memra_tier::object_store::{ExtentStore, FileBackend, MAX_CHUNK};
use memra_tier::tier::Governor;
use std::{cell::RefCell, path::Path, rc::Rc};

/// Native syscall stays outside the unsafe-free metadata/storage crate.
/// macOS F_NOCACHE is explicitly a development fallback, not Linux O_DIRECT.
pub fn open_uncached(path: &Path) -> Result<std::fs::File> {
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        let file = std::fs::File::open(path)?;
        // SAFETY: valid owned descriptor; Darwin F_NOCACHE takes an integer boolean.
        if unsafe { fcntl(file.as_raw_fd(), 48, 1i32) } == -1 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(file)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = path;
        Err(Error::Unsupported)
    }
}

fn ns_since(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Stage clocks (lane/spill-f-20260919 OWED 9). `read_ns` is the summed wall time of
/// `ExtentStore::read` calls and is published as the frozen sample's `io_ns`; the rest go to one
/// stderr line, which the collector's StorageSample join does not consume.
#[derive(Default)]
struct Stages {
    put_ns: u64,
    commit_ns: u64,
    lease_ns: u64,
    read_ns: u64,
    verify_ns: u64,
}

fn fixture(offset: usize, len: usize) -> Vec<u8> {
    (offset..offset + len)
        .map(|i| (i.wrapping_mul(17).wrapping_add(3) % 251) as u8)
        .collect()
}
pub fn run(args: &[String]) -> std::result::Result<StorageSample, Box<dyn std::error::Error>> {
    if !(3..=5).contains(&args.len()) || !matches!(args[1].as_str(), "roundtrip" | "restore") {
        return Err("usage: storage-bench roundtrip|restore <owned-directory> [bytes<=1073741824] [buffered|uncached|direct]".into());
    }
    let restore = args[1] == "restore";
    let directory = Path::new(&args[2]);
    let len: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(264);
    if len == 0 || len > 1 << 30 {
        return Err("bytes must be 1..=1073741824".into());
    }
    let direct_writes = args.get(4).is_some_and(|s| s == "direct");
    let mode = match args.get(4).map(String::as_str).unwrap_or("buffered") {
        "buffered" => ReadMode::Buffered,
        "uncached" | "direct" => ReadMode::Uncached,
        _ => return Err("backend must be buffered, uncached or direct".into()),
    };
    if restore {
        if !directory.is_dir() {
            return Err("restore requires an existing object directory".into());
        }
    } else if directory.exists() && directory.read_dir()?.next().is_some() {
        return Err("roundtrip requires an empty owned directory".into());
    }
    let key = ObjectKey {
        version: 1,
        artifact: digest("benchmark-artifact", b"affine-mod251-v1"),
        semantic_id: digest("benchmark-size", &(len as u64).to_le_bytes()),
        layout: digest("benchmark-layout", b"opaque-bytes-v1"),
        generation: 0,
    };
    let mut backend = FileBackend::open_mode(directory, Durability::Persistent)?;
    backend.set_read_mode(mode);
    backend.set_direct_writes(direct_writes)?;
    #[cfg(target_os = "macos")]
    if mode == ReadMode::Uncached {
        backend.set_uncached_opener(open_uncached);
    }
    let mut cap = TierBudget::zero(0);
    cap.nvme = 2 * (len as u64) + (1 << 24);
    let clock = std::time::Instant::now();
    let gov = Rc::new(RefCell::new(Governor::new(
        cap.clone(),
        TierBudget::zero(0),
        1,
        0,
        std::sync::Arc::new(move || clock.elapsed().as_nanos().try_into().unwrap_or(u64::MAX)),
    )?));
    let mut store = ExtentStore::new(backend, gov.clone());
    let mut stages = Stages::default();
    let start = std::time::Instant::now();
    if !restore {
        let mut txn = store.begin(key.clone(), len as u64, Durability::Persistent)?;
        for offset in (0..len).step_by(MAX_CHUNK) {
            let chunk = fixture(offset, MAX_CHUNK.min(len - offset));
            let t = std::time::Instant::now();
            store.put(&mut txn, &chunk)?;
            stages.put_ns += ns_since(t);
        }
        let t = std::time::Instant::now();
        store.commit(&mut txn)?;
        stages.commit_ns = ns_since(t);
    }
    let manifest = store.lookup(&key)?.ok_or("committed root missing")?;
    let request = BudgetRequest {
        bytes: cap,
        priority: Priority::MandatoryActive,
        deadline: Deadline(u64::MAX),
        tenant: [0; 32],
    };
    let t = std::time::Instant::now();
    let lease = store.lease(&manifest, &request)?;
    stages.lease_ns = ns_since(t);
    use sha2::{Digest as _, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"memra-tier\0v1\0");
    hash.update(11u64.to_le_bytes());
    hash.update(b"valid-bytes");
    hash.update((len as u64).to_le_bytes());
    let mut offset = 0;
    // Bounded readback + byte-by-byte verification; no full-object RAM allocation.
    for (i, c) in manifest.chunks.iter().enumerate() {
        let mut got = vec![0; c.valid_bytes as usize];
        let t = std::time::Instant::now();
        store.read(&lease, i as u32, &mut got)?;
        stages.read_ns += ns_since(t);
        let t = std::time::Instant::now();
        if got != fixture(offset, got.len()) {
            return Err("byte mismatch".into());
        }
        hash.update(&got);
        stages.verify_ns += ns_since(t);
        offset += got.len();
    }
    if offset != len {
        return Err("incomplete restore".into());
    }
    store.release(&lease)?;
    if gov.borrow().used().nvme != 0 {
        return Err("lease charge leaked".into());
    }
    let total_ns = start.elapsed().as_nanos().try_into()?;
    let payload_checksum = hash.finalize().into();
    eprintln!(
        "[storage-bench] stages put_ns={} commit_ns={} lease_ns={} read_ns={} verify_ns={} total_ns={total_ns}",
        stages.put_ns, stages.commit_ns, stages.lease_ns, stages.read_ns, stages.verify_ns
    );
    Ok(StorageSample {
        version: 1,
        fixture: format!(
            "opaque-affine-mod251-v1-{}-development-{}-io-not-spill-speed",
            args[1],
            std::env::consts::OS
        ),
        backend_requested: if direct_writes {
            "direct"
        } else if mode == ReadMode::Buffered {
            "buffered"
        } else {
            "uncached"
        }
        .into(),
        backend_actual: if direct_writes {
            "linux-o-direct-read-write"
        } else if mode == ReadMode::Buffered {
            "buffered-filesystem-persistent"
        } else if cfg!(target_os = "linux") {
            "linux-o-direct-read-buffered-write"
        } else {
            AlignedFile::backend_label()
        }
        .into(),
        status: "byte-exact".into(),
        valid_bytes: len as u64,
        padded_bytes: manifest.chunks.iter().map(|c| c.storage_bytes).sum(),
        io_bytes: store.backend().io_bytes(),
        physical_bytes: None,
        queue_ns: None,
        io_ns: Some(stages.read_ns),
        h2d_ns: None,
        d2h_ns: None,
        p2p_ns: None,
        total_ns,
        inflight: 1,
        pinned_bytes: 0,
        pageable_bytes: None,
        fallbacks: u64::from(mode == ReadMode::Uncached && cfg!(target_os = "macos")),
        payload_checksum,
    })
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--help" {
        println!(
            "storage-bench roundtrip|restore <owned-directory> [bytes<=1073741824] [buffered|uncached|direct]\nCPU filesystem only. Development-Mac I/O characterization, not memra spill speed. No GPU modes."
        );
        return Ok(());
    }
    let sample = run(&args)?;
    println!("{}", String::from_utf8(sample.encode()?)?);
    Ok(())
}
