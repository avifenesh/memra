//! Day-1 storage-bench skeleton. CPU byte-store fixture only, never GPU/NVMe qualification.
//! Built by the standalone A scaffold until lead workspace/bin wiring lands.
use memra_tier::contracts::ObjectKey;
use memra_tier::object_store::{
    ALIGNMENT, ExtentStore, FileBackend, ObjectStore, digest, hex, padded_len,
};
use memra_tier::telemetry::StorageSample;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--help" {
        println!(
            "storage-bench cpu-fixture <empty-owned-directory> [valid-bytes<=1048576]\nCPU ephemeral buffered filesystem only; other backends fail closed."
        );
        return Ok(());
    }
    if !(args.len() == 3 || args.len() == 4) || args[1] != "cpu-fixture" {
        return Err(
            "usage: storage-bench cpu-fixture <empty-owned-directory> [valid-bytes]".into(),
        );
    }
    let directory = Path::new(&args[2]);
    // Refuse reusing arbitrary data. Caller owns cleanup, CLI only creates objects in this dir.
    if directory.exists() && directory.read_dir()?.next().is_some() {
        return Err("fixture directory must be empty".into());
    }
    let len: usize = if args.len() == 4 {
        args[3].parse()?
    } else {
        264
    };
    if len == 0 || len > memra_tier::object_store::MAX_CHUNK {
        return Err("valid bytes must be 1..=1048576".into());
    }
    let payload: Vec<u8> = (0..len)
        .map(|i| (i.wrapping_mul(17).wrapping_add(3) % 251) as u8)
        .collect();
    let key = ObjectKey {
        artifact: digest(b"spill-a-byte-fixture-v1"),
        semantic_id: digest(&payload),
        layout: digest(b"opaque-bytes-v1"),
        generation: 0,
    };
    let mut store = ExtentStore::new(FileBackend::open(directory)?);
    let start = std::time::Instant::now();
    let mut txn = store.begin(key.clone(), len as u64)?;
    store.put(&mut txn, &payload)?;
    store.commit(txn)?;
    let lease = store.lookup(&key)?.ok_or("published object missing")?;
    let got = store.read(&lease, 0)?;
    if got != payload {
        return Err("byte mismatch".into());
    }
    let extent_bytes = ALIGNMENT + padded_len(len)?;
    let root_bytes = ALIGNMENT + padded_len(160)?;
    // put: write+verify, commit: verify+root write+root readback, lookup: root read, read: chunk read.
    let sample = StorageSample {
        schema_version: 1,
        fixture: "opaque-affine-mod251-v1".into(),
        backend_requested: "cpu-fixture".into(),
        backend_actual: "buffered-filesystem-ephemeral".into(),
        status: "byte-exact".into(),
        valid_bytes: len as u64,
        padded_bytes: padded_len(len)? as u64,
        io_bytes: (4 * extent_bytes + 3 * root_bytes) as u64,
        physical_bytes: None,
        queue_ns: None,
        io_ns: None,
        h2d_ns: None,
        d2h_ns: None,
        total_ns: start.elapsed().as_nanos().try_into()?,
        inflight: 1,
        pinned_bytes: 0,
        pageable_bytes: None,
        fallbacks: 0,
        payload_sha256: hex(&digest(&got)),
    };
    println!("{}", sample.json_line()?);
    Ok(())
}
