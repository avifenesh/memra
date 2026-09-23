#![cfg(test)]
#[path = "../../../../../crates/memra-engine/src/trim_ranks.rs"]
mod trim_ranks;
use memra_gguf::{GgmlType, GgufFile, micro_gguf::GgufWriter};
use std::{
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "memra-rank-consumer-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn rows() -> Vec<u8> {
    (0..4u8).flat_map(|id| [id, id + 10, id + 20]).collect()
}

#[test]
fn actual_text_consumer_reuses_captured_ids_and_stamp_across_file_changes() {
    for replacement in [false, true] {
        let f = Fixture::new();
        let path = f.0.join("ranks.txt");
        std::fs::write(&path, "3\n1\n0\n").unwrap();
        let spec = path.to_str().unwrap();
        let mut input = trim_ranks::RankInput::default();
        let capture = input.capture(spec).unwrap();
        let identity = capture.artifact().identity().clone();
        let stamp = capture.sha16().to_owned();
        capture.artifact().validate_for_head(4, "head").unwrap();
        assert_eq!(
            trim_ranks::gather_rows(&rows(), 3, capture.artifact().ids()),
            [3, 13, 23, 1, 11, 21, 0, 10, 20]
        );
        if replacement {
            std::fs::rename(&path, f.0.join("old.txt")).unwrap();
        }
        std::fs::write(&path, "2\n1\n0\n").unwrap();
        let same = input.capture(spec).unwrap();
        assert_eq!(same.artifact().ids(), [3, 1, 0]);
        assert_eq!(same.artifact().identity(), &identity);
        assert_eq!(same.sha16(), stamp);
        assert_eq!(same.path(), spec);
        assert_eq!(stamp, &identity.source_sha256()[..16]);
        let mut fresh = trim_ranks::RankInput::default();
        let fresh = fresh.capture(spec).unwrap();
        assert_eq!(fresh.artifact().ids(), [2, 1, 0]);
        assert_ne!(fresh.artifact().identity(), &identity);
    }
}

#[test]
fn actual_gguf_consumer_never_reopens_for_a_later_identity_stamp() {
    let f = Fixture::new();
    let path = f.0.join("ranks.gguf");
    let mut writer = GgufWriter::new();
    writer.tensor_raw(
        "d2t",
        &[3],
        GgmlType::I64,
        [3i64, 1, 0]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
    );
    writer.write(&path).unwrap();
    let spec = path.to_str().unwrap();
    let mut input = trim_ranks::RankInput::default();
    let capture = input.capture(spec).unwrap();
    let identity = capture.artifact().identity().clone();
    let offset = {
        let g = GgufFile::open(&path).unwrap();
        g.data_start + g.find("d2t").unwrap().offset
    };
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&2i64.to_le_bytes()).unwrap();
    drop(file);
    let same = input.capture(spec).unwrap();
    assert_eq!(same.artifact().identity(), &identity);
    assert_eq!(same.sha16(), &identity.source_sha256()[..16]);
    same.artifact().validate_for_head(4, "head").unwrap();
    assert_eq!(
        trim_ranks::gather_rows(&rows(), 3, same.artifact().ids()),
        [3, 13, 23, 1, 11, 21, 0, 10, 20]
    );
    let mut fresh = trim_ranks::RankInput::default();
    assert_eq!(fresh.capture(spec).unwrap().artifact().ids(), [2, 1, 0]);
}

#[test]
fn actual_consumer_keeps_raw_identity_distinct_and_refuses_source_switches() {
    let f = Fixture::new();
    let path = f.0.join(".txt");
    let other = f.0.join("other.txt");
    std::fs::write(&path, "3\n1\n0\n").unwrap();
    std::fs::write(&other, " 3\n\n1\n0\n").unwrap();
    let mut a = trim_ranks::RankInput::default();
    let first = a
        .capture(path.to_str().unwrap())
        .unwrap()
        .artifact()
        .identity()
        .clone();
    assert!(a.capture(other.to_str().unwrap()).is_err());
    let mut b = trim_ranks::RankInput::default();
    let second = b.capture(other.to_str().unwrap()).unwrap();
    assert_eq!(
        first.order_sha256(),
        second.artifact().identity().order_sha256()
    );
    assert_ne!(
        first.source_sha256(),
        second.artifact().identity().source_sha256()
    );
    assert!(second.artifact().validate_for_head(3, "too small").is_err());
    std::fs::write(&other, "3\n3\n0\n").unwrap();
    assert!(
        trim_ranks::RankInput::default()
            .capture(other.to_str().unwrap())
            .is_err()
    );
}
