use super::*;
use crate::micro_gguf::{GgufWriter, MetaW};
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
            "memra-rank-input-{}-{}",
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
fn write(path: &Path, dtype: GgmlType, ids: &[i64], shape: &[u64], unused: f32) {
    let mut w = GgufWriter::new();
    let bytes: Vec<u8> = ids
        .iter()
        .flat_map(|id| {
            if dtype == GgmlType::I64 {
                id.to_le_bytes().to_vec()
            } else {
                (*id as i32).to_le_bytes().to_vec()
            }
        })
        .collect();
    w.tensor_raw("d2t", shape, dtype, bytes);
    w.tensor_f32("unused.weight", &[20000], &vec![unused; 20000]);
    w.write(path).unwrap();
}
fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
fn rank_input_preserves_order_encoding_and_raw_source_identity() {
    let f = Fixture::new();
    let text = f.0.join("ranks.txt");
    std::fs::write(&text, " 11\n\n2\n9\n").unwrap();
    let t = RankArtifact::open(&text).unwrap();
    assert_eq!(t.ids(), [11, 2, 9]);
    assert_eq!(t.identity().encoding(), RankEncoding::Text);
    assert_eq!(t.identity().source_sha256(), hash(&text));
    for (dtype, encoding) in [
        (GgmlType::I32, RankEncoding::GgufI32),
        (GgmlType::I64, RankEncoding::GgufI64),
    ] {
        let p = f.0.join(format!("{dtype:?}.gguf"));
        write(&p, dtype, &[11, 2, 9], &[3], 0.25);
        let ranks = RankArtifact::open(&p).unwrap();
        assert_eq!(ranks.ids(), t.ids());
        assert_eq!(ranks.identity().encoding(), encoding);
        assert_eq!(ranks.identity().order_sha256(), t.identity().order_sha256());
        assert_eq!(ranks.identity().source_sha256(), hash(&p));
        assert_eq!(
            ranks.identity().source_bytes(),
            std::fs::metadata(&p).unwrap().len()
        );
        assert_ne!(
            ranks.identity().artifact_sha256(),
            t.identity().artifact_sha256()
        );
        ranks.validate_for_head(12, "head").unwrap();
        assert!(ranks.validate_for_head(9, "head").is_err());
    }
    std::fs::write(&text, "2\n11\n9\n").unwrap();
    assert_ne!(
        RankArtifact::open(&text).unwrap().identity().order_sha256(),
        t.identity().order_sha256()
    );
}

#[test]
fn rank_input_text_snapshot_does_not_hash_a_second_read() {
    let f = Fixture::new();
    let path = f.0.join("ranks.txt");
    std::fs::write(&path, "11\n2\n9\n").unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let original = hash(&path);
    std::fs::write(&path, "7\n2\n9\n").unwrap();
    let old = RankArtifact::from_text(bytes).unwrap();
    assert_eq!(old.ids(), [11, 2, 9]);
    assert_eq!(old.identity().source_sha256(), original);
    let fresh = RankArtifact::open(&path).unwrap();
    assert_eq!(fresh.ids(), [7, 2, 9]);
    assert_ne!(fresh.identity(), old.identity());
}

#[test]
fn rank_input_gguf_snapshot_survives_in_place_payload_edit() {
    let f = Fixture::new();
    let path = f.0.join("ranks.gguf");
    write(&path, GgmlType::I64, &[11, 2, 9], &[3], 0.25);
    let original = hash(&path);
    let gguf = GgufFile::open(&path).unwrap();
    let offset = gguf.data_start + gguf.find("d2t").unwrap().offset;
    let captured = capture_gguf(&gguf).unwrap();
    drop(gguf);
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&7i64.to_le_bytes()).unwrap();
    drop(file);
    let old = RankArtifact::from_gguf_capture(captured).unwrap();
    assert_eq!(old.ids(), [11, 2, 9]);
    assert_eq!(old.identity().source_sha256(), original);
    let fresh = RankArtifact::open(&path).unwrap();
    assert_eq!(fresh.ids(), [7, 2, 9]);
    assert_eq!(fresh.identity().source_sha256(), hash(&path));
    assert_ne!(
        fresh.identity().order_sha256(),
        old.identity().order_sha256()
    );
}

#[test]
fn rank_input_opened_handles_survive_path_replacement() {
    let f = Fixture::new();
    let path = f.0.join("ranks.gguf");
    write(&path, GgmlType::I32, &[11, 2, 9], &[3], 0.25);
    let original = hash(&path);
    let gguf = GgufFile::open(&path).unwrap();
    std::fs::rename(&path, f.0.join("original.gguf")).unwrap();
    write(&path, GgmlType::I32, &[7, 2, 9], &[3], 0.75);
    let old = RankArtifact::from_gguf(&gguf).unwrap();
    assert_eq!(old.ids(), [11, 2, 9]);
    assert_eq!(old.identity().source_sha256(), original);
    assert_eq!(RankArtifact::open(&path).unwrap().ids(), [7, 2, 9]);
}

#[test]
fn rank_input_rejects_header_edits_and_truncation_before_capture() {
    for truncate in [false, true] {
        let f = Fixture::new();
        let path = f.0.join("ranks.gguf");
        write(&path, GgmlType::I32, &[11, 2, 9], &[3], 0.25);
        let bytes = std::fs::read(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        if truncate {
            file.set_len(gguf.data_start).unwrap();
        } else {
            let offset = bytes.windows(3).position(|w| w == b"d2t").unwrap();
            file.seek(SeekFrom::Start(offset as u64)).unwrap();
            file.write_all(b"x2t").unwrap();
        }
        assert!(RankArtifact::from_gguf(&gguf).is_err());
    }
}

#[test]
fn rank_input_full_source_hash_covers_unconsumed_bytes() {
    let f = Fixture::new();
    let a = f.0.join("a.gguf");
    let b = f.0.join("b.gguf");
    write(&a, GgmlType::I32, &[11, 2, 9], &[3], 0.25);
    write(&b, GgmlType::I32, &[11, 2, 9], &[3], 0.75);
    let a = RankArtifact::open(&a).unwrap();
    let b = RankArtifact::open(&b).unwrap();
    assert_eq!(a.ids(), b.ids());
    assert_eq!(a.identity().order_sha256(), b.identity().order_sha256());
    assert_ne!(a.identity().source_sha256(), b.identity().source_sha256());
    assert_ne!(
        a.identity().artifact_sha256(),
        b.identity().artifact_sha256()
    );
}

#[test]
fn rank_input_split_container_covers_every_opened_shard() {
    let f = Fixture::new();
    let mut paths = Vec::new();
    for index in 0..2 {
        let path = f.0.join(format!("ranks-{:05}-of-00002.gguf", index + 1));
        let mut w = GgufWriter::new();
        w.kv("split.no", MetaW::U32(index));
        w.kv("split.count", MetaW::U32(2));
        w.kv("split.tensors.count", MetaW::U32(2));
        if index == 0 {
            w.tensor_raw(
                "d2t",
                &[3],
                GgmlType::I32,
                [11i32, 2, 9]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            );
        } else {
            w.tensor_f32("unused.weight", &[1], &[0.25]);
        }
        w.write(&path).unwrap();
        paths.push(path);
    }
    let a = RankArtifact::open(&paths[0]).unwrap();
    let b = RankArtifact::open(&paths[1]).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a.identity().source_count(), 2);
    assert_eq!(a.ids(), [11, 2, 9]);
    let mut expected = Sha256::new();
    expected.update(b"memra-rank-shards-v1\0");
    expected.update(2u64.to_le_bytes());
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        expected.update((bytes.len() as u64).to_le_bytes());
        expected.update(Sha256::digest(bytes));
    }
    assert_eq!(
        a.identity().source_sha256(),
        format!("{:x}", expected.finalize())
    );
}

#[test]
fn rank_input_refuses_invalid_order_payloads_and_preserves_text_parser_rules() {
    let f = Fixture::new();
    let text = f.0.join("ranks.txt");
    assert_eq!(parse_text_ranks("5\n\n 7\n0\n", "t").unwrap(), [5, 7, 0]);
    for bad in [
        "",
        "\n",
        "1\n1\n",
        "-1\n",
        "4294967296\n",
        "1.0\n",
        "#rank\n",
    ] {
        std::fs::write(&text, bad).unwrap();
        assert!(RankArtifact::open(&text).is_err(), "{bad:?}");
    }
    for (dtype, ids, shape) in [
        (GgmlType::I32, vec![-1, 2, 3], vec![3]),
        (GgmlType::I64, vec![4294967297, 2, 3], vec![3]),
        (GgmlType::I64, vec![1, 1, 3], vec![3]),
        (GgmlType::I32, vec![1, 2, 3], vec![1, 3]),
        (GgmlType::F32, vec![1, 2, 3], vec![3]),
    ] {
        let path = f.0.join("invalid.gguf");
        write(&path, dtype, &ids, &shape, 0.25);
        assert!(RankArtifact::open(&path).is_err());
    }
    assert!(validate_ranks(&[3, 1, 0], 4, "head").is_ok());
    assert!(validate_ranks(&[3, 1, 3], 4, "head").is_err());
    assert!(validate_ranks(&[3, 4], 4, "head").is_err());
    assert!(validate_ranks(&[0, 1, 2, 3, 0], 4, "head").is_err());
}
