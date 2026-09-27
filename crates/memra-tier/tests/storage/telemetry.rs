use super::*;

#[test]
fn telemetry_join_schema_cumulative_bytes_unknowns_and_bounded_intervals() {
    use memra_tier::telemetry::{
        StorageSample,
        join::{DeviceSample, StorageDirection, StorageTelemetry},
    };
    use std::io::Write;
    use std::process::{Command, Stdio};
    // Spawns python3: hold the process fence exclusively (see `OwnedDirectory`).
    let _fence = Fence::exclusive();
    let mut join = StorageTelemetry::new(3).unwrap();
    let mut sample = StorageSample {
        version: 1,
        fixture: "cpu-exact".into(),
        backend_requested: "worker".into(),
        backend_actual: "worker".into(),
        status: "ok".into(),
        valid_bytes: 264,
        padded_bytes: 4096,
        io_bytes: 8192,
        physical_bytes: None,
        queue_ns: Some(17),
        io_ns: Some(100),
        h2d_ns: None,
        d2h_ns: None,
        p2p_ns: None,
        total_ns: 120,
        inflight: 0,
        pinned_bytes: 0,
        pageable_bytes: Some(4096),
        fallbacks: 0,
        payload_checksum: digest(&payload(264)),
    };
    join.observe(&sample, StorageDirection::Read).unwrap();
    sample.io_ns = Some(200);
    join.observe(&sample, StorageDirection::Read).unwrap();
    sample.io_ns = Some(300);
    join.observe(&sample, StorageDirection::Write).unwrap();
    assert_eq!(
        join.observe(&sample, StorageDirection::Read),
        Err(Error::Capacity)
    );
    let device = DeviceSample::CpuFixture { device: 0 };
    let first = join
        .json_line(250_000_000, &device, 0, 0, Some(4096))
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&first).unwrap();
    assert_eq!(parsed["nvme"]["read_bytes"], 16384);
    assert_eq!(parsed["nvme"]["write_bytes"], 8192);
    assert!(parsed["nvme"]["physical_bytes"].is_null());
    assert_eq!(parsed["wait_ns"]["io"]["p50"], 200);
    assert_eq!(parsed["wait_ns"]["io"]["p99"], 300);
    sample.physical_bytes = Some(8192); // a later measurement cannot fill an earlier unknown
    join.observe(&sample, StorageDirection::Read).unwrap();
    assert_eq!(
        join.json_line(250_000_000, &device, 0, 0, None),
        Err(Error::InvalidLayout)
    );
    let second = join.json_line(500_000_000, &device, 0, 0, None).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&second).unwrap();
    assert_eq!(parsed["nvme"]["read_bytes"], 24576);
    assert!(parsed["nvme"]["physical_bytes"].is_null());
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/storage/check_telemetry.py");
    let mut child = Command::new("python3")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    write!(child.stdin.take().unwrap(), "{first}\n{second}\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("TELEMETRY_SCHEMA_PASS: 2 rows"));
    println!("TELEMETRY_ROW {first}\nTELEMETRY_ROW {second}");
}
