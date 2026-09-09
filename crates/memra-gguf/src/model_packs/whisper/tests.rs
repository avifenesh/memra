use super::*;
use crate::execution_manifest::execution_rewrites;
use sha2::{Digest, Sha256};

const CONFIG: &str = include_str!("fixtures/config.json");
const FRONTEND: &str = include_str!("fixtures/preprocessor_config.json");
const INDEX: &str = include_str!("fixtures/model.safetensors.index.json");
const HEADER1: &[u8] = include_bytes!("fixtures/model-00001-of-00002.safetensors.header");
const HEADER2: &[u8] = include_bytes!("fixtures/model-00002-of-00002.safetensors.header");

fn shards() -> [ShardMetadata<'static>; 2] {
    [
        ShardMetadata {
            name: "model-00001-of-00002.safetensors",
            header: HEADER1,
            file_size: 4_993_448_880,
        },
        ShardMetadata {
            name: "model-00002-of-00002.safetensors",
            header: HEADER2,
            file_size: 1_180_663_192,
        },
    ]
}

#[test]
fn real_ivrit_metadata_binds_every_tensor_and_stays_unsupported() {
    let (plan, bound) = inspect_metadata(CONFIG, FRONTEND, INDEX, &shards()).unwrap();
    assert_eq!(bound.tensors.len(), 1259);
    assert_eq!(
        bound
            .tensors
            .values()
            .map(|t| t.physical_bytes)
            .sum::<u64>(),
        6_173_962_240
    );
    assert!(plan.layers.is_empty());
    let speech = plan.speech.as_ref().unwrap();
    assert_eq!(speech.encoder_layers, 32);
    assert_eq!(speech.decoder_layers, 32);
    assert_eq!(speech.norm.kind, NormKind::LayerNorm);
    assert_eq!(speech.activation, SpeechActivation::GeluErf);
    assert!(!speech.attention.key_bias);
    assert_eq!(speech.conv_stem[1].stride, 2);
    assert!(speech.state.encoder_cross_kv_per_decoder_layer);
    assert!(WhisperPack::SUPPORT.is_none());
    for rewrite in execution_rewrites(&plan) {
        assert!(
            !rewrite.eligible(),
            "{} must refuse unimplemented speech operations",
            rewrite.id
        );
    }
    let tokens = JsonObj::parse(include_str!("fixtures/added_tokens.json"));
    for (token, id) in [
        ("<|he|>", 50279),
        ("<|transcribe|>", 50360),
        ("<|notimestamps|>", 50364),
    ] {
        assert_eq!(tokens.raw(token).unwrap().parse::<u32>().unwrap(), id);
    }
}

#[test]
fn fixture_bytes_are_the_pinned_source_metadata() {
    for (bytes, expected) in [
        (
            CONFIG.as_bytes(),
            "fd9bf36939c1e7ebb34a525b68fe1b8ac72e7a2814d5be91ab84473662f7d75b",
        ),
        (
            FRONTEND.as_bytes(),
            "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711",
        ),
        (
            INDEX.as_bytes(),
            "42cd1e689ab096987f668b944b9440547114e00ecfe49304e3bc70167234d5b3",
        ),
        (
            HEADER1,
            "0fc771c0cfd67c897106bc099c58fec3bbede2fdf65afc0de71b9a7041c946e6",
        ),
        (
            HEADER2,
            "495108435fbb7b094360741ed0d2f2225012997fad1d8a7ba345127df38d7988",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    }
}

#[test]
fn missing_duplicate_wrong_shard_and_truncated_payload_are_rejected() {
    assert!(inspect_metadata(CONFIG, FRONTEND, INDEX, &shards()[..1]).is_err());
    let original = shards();
    let duplicated = [
        ShardMetadata {
            name: original[0].name,
            header: HEADER1,
            file_size: original[0].file_size,
        },
        ShardMetadata {
            name: original[0].name,
            header: HEADER1,
            file_size: original[0].file_size,
        },
    ];
    assert!(
        inspect_metadata(CONFIG, FRONTEND, INDEX, &duplicated)
            .unwrap_err()
            .contains("duplicate")
    );
    let mut wrong = shards();
    wrong[0].name = "wrong.safetensors";
    assert!(
        inspect_metadata(CONFIG, FRONTEND, INDEX, &wrong)
            .unwrap_err()
            .contains("index/shard")
    );
    let mut short = shards();
    short[0].file_size -= 4;
    assert!(inspect_metadata(CONFIG, FRONTEND, INDEX, &short).is_err());
}

#[test]
fn same_size_wrong_shape_and_unexpected_bias_fail_the_tensor_contract() {
    let plan = PACK.compile_plan(CONFIG, FRONTEND).unwrap();
    let infos = safetensors::parse_header_json_checked(std::str::from_utf8(&HEADER1[8..]).unwrap())
        .unwrap();
    let infos2 =
        safetensors::parse_header_json_checked(std::str::from_utf8(&HEADER2[8..]).unwrap())
            .unwrap();
    let mut census: Vec<_> = infos
        .iter()
        .chain(infos2.iter())
        .map(|(n, i)| census_entry(n, i).unwrap())
        .collect();
    let row = census
        .iter_mut()
        .find(|t| t.name == "model.encoder.conv1.weight")
        .unwrap();
    row.shape.swap(1, 2);
    assert!(bind(&plan, &census).is_err());
    row_restore(&mut census);
    let mut extra = census[0].clone();
    extra.name = "model.encoder.layers.0.self_attn.k_proj.bias".into();
    census.push(extra);
    assert!(bind(&plan, &census).is_err());
}

fn row_restore(census: &mut [TensorCensusEntry]) {
    census
        .iter_mut()
        .find(|t| t.name == "model.encoder.conv1.weight")
        .unwrap()
        .shape
        .swap(1, 2);
}

#[test]
fn changed_graph_frontend_and_storage_are_refused() {
    for changed in [
        CONFIG.replace("\"gelu\"", "\"gelu_new\""),
        CONFIG.replace("\"decoder_layers\": 32", "\"decoder_layers\": 4"),
        CONFIG.replace("\"float32\"", "\"float16\""),
        CONFIG.replace("\"scale_embedding\": false", "\"scale_embedding\": true"),
    ] {
        assert!(PACK.compile_plan(&changed, FRONTEND).is_err());
    }
    assert!(
        PACK.compile_plan(CONFIG, &FRONTEND.replace("400", "512"))
            .is_err()
    );
    let mut info =
        safetensors::parse_header_json_checked(std::str::from_utf8(&HEADER1[8..]).unwrap())
            .unwrap()
            .into_values()
            .next()
            .unwrap();
    info.dtype = "F16".into();
    assert!(census_entry("wrong_dtype", &info).is_err());
}

/// Sparse payload holes exercise the actual mmap loader using real headers without real
/// weights. This is metadata coverage only, never checkpoint numerical parity.
#[test]
fn mmap_loader_validates_real_checkpoint_metadata_without_weight_download() {
    use std::io::Write;
    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Ok(entries) = std::fs::read_dir(&self.0) {
                for entry in entries.flatten() {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
            let _ = std::fs::remove_dir(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!(
        "memra-whisper-metadata-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    let fixture = Fixture(path);
    for (name, text) in [
        ("config.json", CONFIG),
        ("preprocessor_config.json", FRONTEND),
        ("model.safetensors.index.json", INDEX),
    ] {
        std::fs::write(fixture.0.join(name), text).unwrap();
    }
    for shard in shards() {
        let mut f = std::fs::File::create(fixture.0.join(shard.name)).unwrap();
        f.write_all(shard.header).unwrap();
        f.set_len(shard.file_size).unwrap();
    }
    let checkpoint = WhisperCheckpoint::open(&fixture.0).unwrap();
    assert_eq!(checkpoint.weights.n_tensors(), 1259);
    assert_eq!(checkpoint.tensors.tensors.len(), 1259);
    assert_eq!(
        checkpoint
            .weights
            .info("model.encoder.conv1.weight")
            .unwrap()
            .shape,
        vec![1280, 128, 3]
    );
    drop(checkpoint);
}
