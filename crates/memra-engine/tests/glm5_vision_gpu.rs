//! GPU gate for the glm5_next vision tower: engine (f32 GEMMs, TF32 OFF) vs the banked
//! upstream transformers f32 fixture (research/glm5-vision-20260830 — the same truth the
//! memra_reference twin is pinned against, so engine and reference are pinned to ONE
//! oracle). Loader under gate too: the tower loads `model.visual.*` through StModel from
//! the real shard, so a name/shape/dtype drift fails here, not in serving.
//!
//! Invocation (rig 5090, correctness-only — no timing on this box):
//!   flock /tmp/memra-5090.lock env NVIDIA_TF32_OVERRIDE=0 \
//!   MEMRA_GLM5_VISION_SHARD=~/models/glm53-vision/model-00062-of-00062.safetensors \
//!   cargo test --release -p memra-engine --test glm5_vision_gpu -- --ignored --nocapture
//!
//! Bands: det112 is the tight pin (upstream deterministic on it; measured engine deltas
//! recorded in the lane doc). det448x224/text448 carry the upstream self-reproducibility
//! caveat (see the reference pin's module doc) and gate on the merger output only.

use memra_engine::Engine;
use memra_engine::vision_glm5::Glm5VisionTower;
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

fn grid_of(meta: &str) -> (usize, usize, usize) {
    // "grid_thw": [1, gh, gw]
    let marker = "\"grid_thw\": [";
    let start = meta.find(marker).expect("grid_thw in meta") + marker.len();
    let nums: Vec<usize> = meta[start..]
        .split(']')
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().parse().expect("grid dim"))
        .collect();
    assert_eq!(nums[0], 1, "image fixtures are single-frame");
    (nums[0], nums[1], nums[2])
}

/// The default-ON detection seam, absent case (lane/glm5-vision-default-on): a safetensors
/// artifact with NO `model.visual.*` tensors probes false — text-only, vision off, no flag
/// needed. Hermetic (a minimal one-tensor safetensors file); the PRESENT case is asserted
/// against the real shard inside the GPU gate below.
#[test]
fn visual_tensor_probe_is_false_on_a_textonly_artifact() {
    let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    let path = std::env::temp_dir().join(format!("glm5v-probe-{}.safetensors", std::process::id()));
    std::fs::write(&path, bytes).expect("write probe fixture");
    let present = memra_engine::vision_glm5::glm5_visual_tensors_present(&path)
        .expect("probe must read a valid safetensors file");
    let _ = std::fs::remove_file(&path);
    assert!(!present, "a text-only artifact must probe ABSENT");
}

#[test]
#[ignore = "needs a CUDA device + the 1.26 GB visual shard — run under flock /tmp/memra-5090.lock with NVIDIA_TF32_OVERRIDE=0"]
fn engine_tower_matches_banked_upstream_fixture() {
    let shard = std::env::var("MEMRA_GLM5_VISION_SHARD")
        .expect("MEMRA_GLM5_VISION_SHARD must point at model-00062-of-00062.safetensors");
    let shard = shard.replace("~/", &format!("{}/", std::env::var("HOME").expect("HOME")));
    // Default-ON detection seam, present case: the artifact that carries the tower probes
    // PRESENT (the same signal the worker keys the auto-load on).
    assert!(
        memra_engine::vision_glm5::glm5_visual_tensors_present(Path::new(&shard))
            .expect("probe reads the shard"),
        "the visual shard must probe PRESENT"
    );
    let e = Engine::new(
        std::env::var("MEMRA_PROBE_DEVICE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    )
    .expect("engine");
    let tower = Glm5VisionTower::load(&e, Path::new(&shard)).expect("tower load");

    // (fixture, per-stage band or None, merger band)
    for (fixture, band_stage, band_out) in [
        ("det112", Some(1e-3f32), 1e-4f32),
        ("det448x224", None, 5e-3),
        ("text448", None, 2e-2),
    ] {
        let dir = lane_dir().join("fixtures").join(fixture);
        if !dir.join("pixels.bin").exists() {
            assert_ne!(
                fixture, "det112",
                "committed fixture {fixture} is missing its binaries"
            );
            eprintln!(
                "SKIP {fixture} (regenerable fixture not present; run gen_upstream_fixture.py)"
            );
            continue;
        }
        let meta = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");
        let (_, gh, gw) = grid_of(&meta);
        let n = gh * gw;
        let pixels = read_f32(&dir.join("pixels.bin"));
        assert_eq!(pixels.len(), n * memra_engine::vision_glm5::G5V_PATCH_IN);

        // Stage dumps via the house MEMRA_VISION_DEBUG seam, for the bisect anchors.
        let dump_dir = std::env::temp_dir().join(format!("glm5v-gate-{fixture}"));
        std::fs::create_dir_all(&dump_dir).expect("dump dir");
        // SAFETY: single-threaded test binary section; the env var is read once inside
        // forward() on this same thread.
        unsafe { std::env::set_var("MEMRA_VISION_DEBUG", &dump_dir) };
        let out_d = tower.forward(&e, &pixels, gh, gw).expect("tower forward");
        unsafe { std::env::remove_var("MEMRA_VISION_DEBUG") };
        let out = e.dtoh(&out_d).expect("dtoh");

        let banked_out = read_f32(&dir.join("out_f32.bin"));
        let out_diff = max_abs_diff(&out, &banked_out);
        let stage = |ours: &str, banked: &str| -> f32 {
            max_abs_diff(
                &read_f32(&dump_dir.join(ours)),
                &read_f32(&dir.join(banked)),
            )
        };
        let patch_diff = stage("rust_pre_blocks.bin", "stage_patch.bin");
        let blk0_diff = stage("rust_blk0.bin", "stage_blk0.bin");
        let post_diff = stage("rust_post_blocks.bin", "stage_post.bin");
        let down_diff = stage("rust_downsample.bin", "stage_down.bin");
        println!(
            "{fixture}: patch {patch_diff:.2e} | blk0 {blk0_diff:.2e} | post {post_diff:.2e} | down {down_diff:.2e} | merger {out_diff:.2e}"
        );
        let _ = std::fs::remove_dir_all(&dump_dir);
        if let Some(band) = band_stage {
            for (name, diff) in [
                ("patch-embed", patch_diff),
                ("block-0", blk0_diff),
                ("post-blocks", post_diff),
                ("downsample", down_diff),
            ] {
                assert!(
                    diff <= band,
                    "{fixture}: {name} diverges from upstream ({diff:.3e} > {band:.1e})"
                );
            }
        }
        assert!(
            out_diff <= band_out,
            "{fixture}: merger output diverges from upstream ({out_diff:.3e} > {band_out:.1e})"
        );
    }
}
