#![cfg(test)]
use memra_gguf::{
    GgmlType, GgufFile,
    bound_source::{
        PreparedModelSource,
        head_trim::{HeadChoice, PreparedHeadTrim, TrimPolicy, TrimProgram},
        ranks::RankArtifact,
    },
    source::{GgufSource, TensorSource},
};
mod fixture;
#[path = "../../../../../crates/memra-engine/src/head_trim.rs"]
mod head_trim;
#[path = "../../../../../crates/memra-engine/src/trim_ranks.rs"]
mod trim_ranks;

struct Engine;
impl Engine {
    fn htod_bytes(&self, b: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(b.to_vec())
    }
    fn htod(&self, b: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(b.to_vec())
    }
}
mod model {
    use super::*;
    #[derive(Debug)]
    pub enum GpuTensor {
        FloatBf16 {
            data: Vec<u8>,
            ne: Vec<u64>,
        },
        Float {
            data: Vec<f32>,
            ne: Vec<u64>,
        },
        Quant {
            bytes: Vec<u8>,
            dtype: GgmlType,
            ne: Vec<u64>,
            scale: f32,
        },
    }
    impl GpuTensor {
        pub fn from_quant_bytes(
            _: &Engine,
            b: &[u8],
            dtype: GgmlType,
            n: u64,
            m: u64,
            scale: f32,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self::Quant {
                bytes: b.to_vec(),
                dtype,
                ne: vec![n, m],
                scale,
            })
        }
    }
}
fn check_upload(trim: &PreparedHeadTrim) {
    let (gpu, sizes) = head_trim::load(&Engine, trim).unwrap();
    assert_eq!(sizes, trim.requant_sizes());
    match gpu {
        model::GpuTensor::FloatBf16 { data, ne } => {
            assert_eq!(trim.dtype(), GgmlType::BF16);
            assert_eq!(data, trim.bytes());
            assert_eq!(ne, trim.shape());
        }
        model::GpuTensor::Float { data, ne } => {
            assert_eq!(trim.dtype(), GgmlType::F32);
            assert_eq!(
                data.into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
                trim.bytes()
            );
            assert_eq!(ne, trim.shape());
        }
        model::GpuTensor::Quant {
            bytes,
            dtype,
            ne,
            scale,
        } => {
            assert_eq!(bytes, trim.bytes());
            assert_eq!(dtype, trim.dtype());
            assert_eq!(ne, trim.shape());
            assert_eq!(scale.to_bits(), trim.macro_scale().to_bits());
        }
    }
}
fn gathered(bytes: &[u8], width: usize) -> Vec<u8> {
    [
        bytes[3 * width..4 * width].to_vec(),
        bytes[width..2 * width].to_vec(),
    ]
    .concat()
}

#[test]
fn actual_private_and_extra_head_uploads_keep_their_own_macros() {
    let f = fixture::make(GgmlType::NVFP4, 64, false, None, GgmlType::F32);
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let raw = GgufSource(&g);
    let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    assert!(
        PreparedHeadTrim::prepare(&raw, &ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
            .is_err()
    );
    PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            for (choice, name, scale) in [
                (HeadChoice::ModelOutput, "output.weight", 2.0),
                (
                    HeadChoice::FirstMtpOrModel,
                    "blk.1.nextn.shared_head_head.weight",
                    3.0,
                ),
                (
                    HeadChoice::MtpBlock { index: 2 },
                    "blk.2.nextn.shared_head_head.weight",
                    5.0,
                ),
            ] {
                let trim = PreparedHeadTrim::prepare(s, &ranks, choice, TrimPolicy::Preserve)
                    .unwrap()
                    .unwrap();
                assert_eq!(trim.runtime_name(), name);
                assert_eq!(trim.macro_scale(), scale);
                assert_eq!(trim.bytes(), gathered(&f.heads[name], 36));
                check_upload(&trim);
            }
        })
        .unwrap();
}

#[test]
fn missing_private_heads_follow_only_the_declared_selection_rule() {
    let f = fixture::make(GgmlType::NVFP4, 64, true, Some(1), GgmlType::F32);
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let raw = GgufSource(&g);
    let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            let trim = PreparedHeadTrim::prepare(
                s,
                &ranks,
                HeadChoice::FirstMtpOrModel,
                TrimPolicy::Preserve,
            )
            .unwrap()
            .unwrap();
            assert!(trim.from_model_output());
            assert_eq!(trim.runtime_name(), "token_embd.weight");
            assert_eq!(trim.macro_scale(), 2.0);
            check_upload(&trim);
            assert!(
                PreparedHeadTrim::prepare(
                    s,
                    &ranks,
                    HeadChoice::MtpBlock { index: 1 },
                    TrimPolicy::Preserve
                )
                .unwrap()
                .is_none()
            );
            assert!(
                PreparedHeadTrim::prepare(
                    s,
                    &ranks,
                    HeadChoice::MtpBlock { index: 99 },
                    TrimPolicy::Preserve
                )
                .is_err()
            );
        })
        .unwrap();
}

#[test]
fn bf16_requantization_matches_known_nvfp4_blocks_and_preserve_stays_byte_exact() {
    for dtype in [GgmlType::BF16, GgmlType::F32, GgmlType::Q8_0] {
        let f = fixture::make(dtype, 64, false, None, GgmlType::F32);
        let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
        let raw = GgufSource(&g);
        let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
        PreparedModelSource::text(&raw)
            .unwrap()
            .with_runtime(|s| {
                let raw = PreparedHeadTrim::prepare(
                    s,
                    &ranks,
                    HeadChoice::ModelOutput,
                    TrimPolicy::Preserve,
                )
                .unwrap()
                .unwrap();
                assert_eq!(
                    raw.bytes(),
                    gathered(
                        &f.heads["output.weight"],
                        f.heads["output.weight"].len() / 4
                    )
                );
                check_upload(&raw);
                let converted = PreparedHeadTrim::prepare(
                    s,
                    &ranks,
                    HeadChoice::ModelOutput,
                    TrimPolicy::Nvfp4ForEligibleBf16,
                )
                .unwrap()
                .unwrap();
                check_upload(&converted);
                if dtype == GgmlType::BF16 {
                    let expected =
                        [vec![0x50; 4], vec![0x77; 32], vec![0x40; 4], vec![0x77; 32]].concat();
                    assert_eq!(converted.bytes(), expected);
                    assert_eq!(converted.dtype(), GgmlType::NVFP4);
                    assert_eq!(converted.macro_scale(), 1.0);
                    assert_eq!(
                        converted.identity().program(),
                        TrimProgram::Bf16RowsToNvfp4V1
                    );
                    assert_eq!(converted.requant_sizes(), Some((72, 256)));
                } else {
                    assert_eq!(converted.bytes(), raw.bytes());
                    assert_eq!(converted.identity().program(), TrimProgram::RowGather);
                }
            })
            .unwrap();
    }
}

#[test]
fn unsupported_row_formats_short_macros_and_bad_ranks_refuse_before_upload() {
    for (dtype, width, scale_dtype) in [
        (GgmlType::F16, 64, GgmlType::F32),
        (GgmlType::Q4_K, 128, GgmlType::F32),
        (GgmlType::NVFP4, 64, GgmlType::F16),
        (GgmlType::NVFP4, 64, GgmlType::BF16),
    ] {
        let f = fixture::make(dtype, width, false, None, scale_dtype);
        let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
        let raw = GgufSource(&g);
        let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
        PreparedModelSource::text(&raw)
            .unwrap()
            .with_runtime(|s| {
                assert!(
                    PreparedHeadTrim::prepare(
                        s,
                        &ranks,
                        HeadChoice::ModelOutput,
                        TrimPolicy::Preserve
                    )
                    .is_err()
                )
            })
            .unwrap();
    }
    let f = fixture::make(GgmlType::F32, 64, false, None, GgmlType::F32);
    std::fs::write(f.dir.join("ranks.txt"), "4\n1\n").unwrap();
    let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let raw = GgufSource(&g);
    PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            assert!(
                PreparedHeadTrim::prepare(
                    s,
                    &ranks,
                    HeadChoice::FirstMtpOrModel,
                    TrimPolicy::Preserve
                )
                .is_err()
            )
        })
        .unwrap();
}

#[test]
fn prepared_payload_and_identity_survive_source_and_rank_mutation() {
    use std::io::{Seek, SeekFrom, Write};
    let f = fixture::make(GgmlType::NVFP4, 64, false, None, GgmlType::F32);
    let path = f.dir.join("model.gguf");
    let g = GgufFile::open(&path).unwrap();
    let raw = GgufSource(&g);
    let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    let trim = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &ranks, HeadChoice::FirstMtpOrModel, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    let identity = trim.identity().clone();
    let bytes = trim.bytes().to_vec();
    let offset = g.data_start
        + g.find("blk.1.nextn.shared_head_head.weight")
            .unwrap()
            .offset;
    let macro_offset = g.data_start + g.find("blk.1.nextn.shared_head_head.scale").unwrap().offset;
    drop(g);
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&[0x40; 4]).unwrap();
    drop(file);
    std::fs::write(f.dir.join("ranks.txt"), "1\n3\n").unwrap();
    assert_eq!(trim.bytes(), bytes);
    assert_eq!(trim.identity(), &identity);
    check_upload(&trim);
    let g = GgufFile::open(&path).unwrap();
    let raw = GgufSource(&g);
    let fresh = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &ranks, HeadChoice::FirstMtpOrModel, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(fresh.bytes(), bytes);
    assert_ne!(
        fresh.identity().source_head_sha256(),
        identity.source_head_sha256()
    );
    assert_ne!(
        fresh.identity().materialization_sha256(),
        identity.materialization_sha256()
    );
    drop(g);
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(macro_offset)).unwrap();
    file.write_all(&7.0f32.to_le_bytes()).unwrap();
    drop(file);
    let g = GgufFile::open(&path).unwrap();
    let raw = GgufSource(&g);
    let scaled = PreparedModelSource::text(&raw)
        .unwrap()
        .with_runtime(|s| {
            PreparedHeadTrim::prepare(s, &ranks, HeadChoice::FirstMtpOrModel, TrimPolicy::Preserve)
        })
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(scaled.bytes(), fresh.bytes());
    assert_eq!(
        scaled.identity().source_head_sha256(),
        fresh.identity().source_head_sha256()
    );
    assert_eq!(scaled.macro_scale(), 7.0);
    assert_eq!(trim.macro_scale(), 3.0);
    assert_ne!(
        scaled.identity().materialization_sha256(),
        fresh.identity().materialization_sha256()
    );
    check_upload(&scaled);
}

#[test]
fn external_draft_authority_cannot_be_reused_as_the_target_head_source() {
    let f = fixture::make(GgmlType::F32, 64, false, None, GgmlType::F32);
    let g = GgufFile::open(f.dir.join("model.gguf")).unwrap();
    let raw = GgufSource(&g);
    let ranks = RankArtifact::open(&f.dir.join("ranks.txt")).unwrap();
    let external =
        memra_gguf::bound_source::draft::PreparedExternalDraftSource::compile(&raw, &raw.config())
            .unwrap();
    external
        .with_runtime(|s| {
            assert!(
                PreparedHeadTrim::prepare(s, &ranks, HeadChoice::ModelOutput, TrimPolicy::Preserve)
                    .is_err()
            )
        })
        .unwrap();
}

#[test]
fn actual_chain_gate_preserves_external_trimmed_heads_and_checks_captured_order() {
    let f = fixture::make(GgmlType::F32, 64, false, None, GgmlType::F32);
    let mut input = trim_ranks::RankInput::default();
    // A standalone already-trimmed external draft has d2t but zero extra heads and no ranks file.
    assert!(
        head_trim::extra_head_ranks(&input, 0, Some(&[3, 1]))
            .unwrap()
            .is_none()
    );
    assert!(
        head_trim::extra_head_ranks(&input, 2, None)
            .unwrap()
            .is_none()
    );
    assert!(head_trim::extra_head_ranks(&input, 2, Some(&[3, 1])).is_err());
    let captured = input
        .capture(f.dir.join("ranks.txt").to_str().unwrap())
        .unwrap();
    assert!(!captured.path().is_empty());
    assert_eq!(captured.sha16().len(), 16);
    assert!(head_trim::extra_head_ranks(&input, 2, Some(&[1, 3])).is_err());
    let ranks = head_trim::extra_head_ranks(&input, 2, Some(&[3, 1]))
        .unwrap()
        .unwrap();
    assert_eq!(ranks.ids(), [3, 1]);
    assert_eq!(
        trim_ranks::gather_rows(&[0, 1, 2, 3], 1, ranks.ids()),
        [3, 1]
    );
}
