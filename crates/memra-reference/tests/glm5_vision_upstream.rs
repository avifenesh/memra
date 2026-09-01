//! GLM-5.3-Flash vision tower: reference-vs-upstream pin (lane/glm5-vision).
//!
//! Loads the real tower weights (the artifact's visual shard, BF16 -> f32) through the
//! glm5_next pack's OWN tensor census — so a census/shard disagreement fails here too —
//! and runs `memra_reference::execute_vision` against the banked transformers fixture
//! (research/glm5-vision-20260830: pinned rev, shard sha256, transformers 5.16.1, torch
//! CPU f32; pixels + per-stage dumps + merger output).
//!
//! Ignored by default (needs the 1.26 GB shard). Invocation (the cert line):
//!   MEMRA_GLM5_VISION_SHARD=~/models/glm53-vision/model-00062-of-00062.safetensors \
//!   cargo test -p memra-reference --test glm5_vision_upstream -- --ignored --nocapture
//!
//! Bands (f32 reference vs f32 upstream, same weights-cast class; measured 2026-08-30):
//! - det112 (64 patches): fully deterministic upstream (fresh-vs-banked 0.0) -> tight pin.
//!   Measured: post 5.8e-5, downsample 2.5e-5, merger 1.1e-6.
//! - det448x224 (512 patches): upstream differs from ITSELF fresh-vs-banked (torch CPU
//!   kernel/reduction-order variation, chaotic growth through 24 blocks): post 7.2e-2,
//!   merger 6.8e-4. Bands sized ~7x that; the merger (what the trunk consumes) is the
//!   meaningful pin.
//! - text448: numerically ill-conditioned (mostly-white canvas -> ~1k near-identical
//!   tokens -> softmax-tie amplification; upstream self-delta post ~1.0, merger 4.8e-3).
//!   Stages NOT gated; merger gated loosely so a gross defect still fails. Its real job
//!   is the can't-hallucinate transcription probe.
//!
//! bf16-activation class (artifact dtype) is banked in the fixture meta for the
//! engine-dtype decision, not gated here.

use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::safetensors::StShard;
use memra_gguf::tensor_contract::{ContractOptions, TensorOwner};
use memra_reference::{ReferenceTensor, ReferenceVisionInput, ReferenceWeights};
use std::path::{Path, PathBuf};

fn lane_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../research/glm5-vision-20260830")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Load every census-declared vision tensor from the shard as f32 reference weights.
fn load_tower(shard_path: &Path, config: &ModelConfig, plan: &ModelPlan) -> ReferenceWeights {
    let pack = memra_gguf::model_packs::by_alias("glm5_next").expect("glm5_next pack");
    let contract = pack
        .compile_tensor_contract(
            config,
            plan,
            memra_gguf::tensor_contract::CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .expect("glm5_next contract");
    let shard = StShard::open(shard_path).expect("open visual shard");
    let mut weights = ReferenceWeights::new();
    let mut loaded = 0usize;
    for requirement in &contract.requirements {
        let TensorOwner::Vision(_) = requirement.owner else {
            continue;
        };
        let name = &requirement.names[0];
        let (info, raw) = shard
            .raw(name)
            .unwrap_or_else(|| panic!("census tensor missing from shard: {name}"));
        assert_eq!(
            info.shape.to_vec(),
            requirement.shape,
            "census shape mismatch for {name}"
        );
        assert_eq!(
            info.dtype, "BF16",
            "{name}: unexpected dtype {}",
            info.dtype
        );
        let data: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let shape: Vec<usize> = requirement.shape.iter().map(|&d| d as usize).collect();
        weights.insert(
            requirement.id.clone(),
            ReferenceTensor::new(shape, data).expect("tensor shape"),
        );
        loaded += 1;
    }
    assert_eq!(
        loaded, 347,
        "hand-census expects exactly 347 visual tensors"
    );
    weights
}

#[test]
#[ignore = "needs the 1.26 GB visual shard; see the module doc for the invocation"]
fn reference_matches_upstream_tower_on_banked_fixture() {
    let shard_path = std::env::var("MEMRA_GLM5_VISION_SHARD")
        .expect("MEMRA_GLM5_VISION_SHARD must point at model-00062-of-00062.safetensors");
    let shard_path = PathBuf::from(shellexpand_home(&shard_path));
    let config_json =
        std::fs::read_to_string(lane_dir().join("../glm53-flash-bringup-20260827/glm-config.json"))
            .expect("banked glm-config.json");
    let config = ModelConfig::from_hf(&HfConfig::parse(&config_json));
    let plan = ModelPlan::compile(&config).expect("glm5 plan");
    let weights = load_tower(&shard_path, &config, &plan);

    // (fixture, stage band or None = ungated, merger band) — class per the module doc.
    for (fixture, band_stage, band_out) in [
        ("det112", Some(2e-4f32), 1e-5f32),
        ("det448x224", Some(0.3), 5e-3),
        ("text448", None, 2e-2),
    ] {
        let dir = lane_dir().join("fixtures").join(fixture);
        if !dir.join("pixels.bin").exists() {
            // det112 is committed; the larger fixtures' binaries are regenerated by
            // gen_upstream_fixture.py. Absence of a REGENERABLE fixture is reported,
            // never silently passed.
            assert_ne!(
                fixture, "det112",
                "committed fixture {fixture} is missing its binaries"
            );
            eprintln!(
                "SKIP {fixture} (regenerable fixture not present; run gen_upstream_fixture.py)"
            );
            continue;
        }
        let meta: serde_lite::Meta =
            serde_lite::parse(&std::fs::read_to_string(dir.join("meta.json")).expect("meta.json"));
        let n_patches = meta.n_patches;
        let n_tokens = meta.n_tokens;
        let pixels = read_f32(&dir.join("pixels.bin"));
        assert_eq!(pixels.len(), n_patches * 1176);
        let pos_raw = std::fs::read(dir.join("pos_ids.bin")).expect("pos_ids.bin");
        let positions: Vec<[u32; 2]> = pos_raw
            .chunks_exact(8)
            .map(|c| {
                [
                    u32::from_le_bytes(c[0..4].try_into().unwrap()),
                    u32::from_le_bytes(c[4..8].try_into().unwrap()),
                ]
            })
            .collect();
        assert_eq!(positions.len(), n_patches);
        let input = ReferenceVisionInput {
            patches: ReferenceTensor::new(vec![n_patches, 1176], pixels).unwrap(),
            positions,
            output_tokens: n_tokens,
        };
        let output =
            memra_reference::execute_vision(&plan, &weights, &input).expect("vision forward");

        // Stage pins (bisect anchors), then the merger output pin.
        let post = read_f32(&dir.join("stage_post.bin"));
        let post_diff = max_abs_diff(&output.encoder_hidden, &post);
        let down = read_f32(&dir.join("stage_down.bin"));
        let down_diff = max_abs_diff(&output.pooled_hidden, &down);
        let out = read_f32(&dir.join("out_f32.bin"));
        let out_diff = max_abs_diff(&output.projected_hidden, &out);
        println!(
            "{fixture}: post_blocks max_abs {post_diff:.2e} | downsample max_abs {down_diff:.2e} | merger max_abs {out_diff:.2e}"
        );
        if let Some(band_stage) = band_stage {
            assert!(
                post_diff <= band_stage,
                "{fixture}: post-blocks diverges from upstream ({post_diff:.3e} > {band_stage:.1e})"
            );
            assert!(
                down_diff <= band_stage,
                "{fixture}: downsample diverges from upstream ({down_diff:.3e} > {band_stage:.1e})"
            );
        }
        assert!(
            out_diff <= band_out,
            "{fixture}: merger output diverges from upstream ({out_diff:.3e} > {band_out:.1e})"
        );
    }
}

fn shellexpand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").expect("HOME")),
        None => path.to_string(),
    }
}

/// Minimal JSON field extraction for the fixture meta (no serde dependency in this crate).
mod serde_lite {
    pub struct Meta {
        pub n_patches: usize,
        pub n_tokens: usize,
    }
    pub fn parse(json: &str) -> Meta {
        let field = |key: &str| -> usize {
            let marker = format!("\"{key}\":");
            let start = json
                .find(&marker)
                .unwrap_or_else(|| panic!("meta missing {key}"))
                + marker.len();
            json[start..]
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or_else(|_| panic!("meta field {key} is not an integer"))
        };
        Meta {
            n_patches: field("n_patches"),
            n_tokens: field("n_tokens"),
        }
    }
}
