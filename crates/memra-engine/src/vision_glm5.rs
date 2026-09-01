//! Vision tower for GLM-5.3-Flash / glm5_next (lane/glm5-vision, 2026-08-30).
//!
//! glm5_next is its OWN semantic program — nothing here is inherited from the qwen3_5 or
//! gemma-4 towers by analogy (no-generic-support law). Every rule below was pinned against
//! upstream transformers 5.16.1 `Glm5NextVisionModel` (vision classes diffed byte-identical
//! to transformers main; the memra_reference twin is gated against a banked upstream f32
//! fixture in research/glm5-vision-20260830):
//!
//! - ViT: 24 blocks, hidden 1024, 16 heads (head_dim 64), ffn 4096. RMS norms (eps 1e-5,
//!   weight-only), FUSED qkv `[3072, 1024]` WITH bias, per-head q/k RMS norms (64) applied
//!   BEFORE the rope, scaled attention (1/sqrt(64)), non-causal, one segment per image.
//! - Positions: ROPE-ONLY. No learned position table (upstream `GlmOcrVisionModel.__init__`
//!   deletes `self.embeddings`). 2D rope theta 10000 (hardcoded upstream, not a config
//!   key): head-dim half 32, first 16 angle dims rotate by the token's grid row (h), the
//!   next 16 by its column (w), NeoX pairs `(d, d + 32)`, `inv_freq[i] = theta^(-2i/32)`.
//! - Block MLP: gate/up/down `[4096, 1024]` WITH biases, clamped SwiGLU:
//!   `silu(min(gate, 10)) * clamp(up, -10, 10)` (swiglu_limit 10.0 from vision_config).
//! - Head: post-encoder RMS norm, then a conv 2x2/stride-2 downsample `[4096, 1024, 2, 2]`
//!   plus bias over each spatial-merge block (token order is block-major, so a block is 4
//!   consecutive rows in (in_row, in_col) order), then the gated merger:
//!   proj 4096->4096 (no bias) -> LayerNorm(w, b; torch default eps 1e-5) -> EXACT-erf
//!   GELU -> gate/up 4096->10240 (no bias, clamped like the block MLP) -> down -> 4096.
//!   Merger output width == trunk n_embd (4096): rows splice directly over `<|image|>`
//!   token embeddings (ids: start 154830, image 154854, end 154831).
//! - Input: the processor's pixel contract — rescale 1/255, CLIP mean/std normalize,
//!   patch rows `[n, 1176]` in (c, t=2 duplicated frames, 14, 14) flat order, sequence in
//!   spatial-merge block-major order. Grids come from the vendor smart_resize
//!   (factor 28, budget 16..8000 merged tokens; serving caps lower — see the intake).
//!
//! v1 posture matches the house towers: correctness-first — f32 GEMMs (`Engine::linear`),
//! `sdpa_naive(causal=false)`, host-side split/per-head-norm/rope/activations; parity gate
//! (tests/glm5_vision_gpu.rs) against the banked upstream fixture before any serving path.

use crate::Engine;
use cudarc::driver::CudaSlice;
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::safetensors::StModel;
use std::path::Path;

pub const G5V_HIDDEN: usize = 1024;
pub const G5V_HEADS: usize = 16;
pub const G5V_HEAD_DIM: usize = G5V_HIDDEN / G5V_HEADS; // 64
pub const G5V_INTER: usize = 4096;
pub const G5V_DEPTH: usize = 24;
pub const G5V_PATCH: usize = 14;
pub const G5V_TEMPORAL: usize = 2;
pub const G5V_MERGE: usize = 2;
pub const G5V_OUT: usize = 4096; // == GLM-5.3-Flash trunk n_embd
pub const G5V_PROJ_INTER: usize = 10_240;
pub const G5V_LIMIT: f32 = 10.0;
pub const G5V_PATCH_IN: usize = 3 * G5V_TEMPORAL * G5V_PATCH * G5V_PATCH; // 1176
const RMS_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5; // torch nn.LayerNorm default (merger post_projection_norm)
const ROPE_THETA: f32 = 10_000.0;

/// Splice ids (config.json truth, research/glm53-flash-bringup-20260827/glm-config.json).
pub const G5V_TOK_IMAGE_START: u32 = 154_830; // <|begin_of_image|>
pub const G5V_TOK_IMAGE_END: u32 = 154_831; // <|end_of_image|>
pub const G5V_TOK_IMAGE: u32 = 154_854; // <|image|> (placeholder, expanded per grid)
pub const G5V_TOK_VIDEO_START: u32 = 154_832; // <|begin_of_video|>
pub const G5V_TOK_VIDEO_END: u32 = 154_833; // <|end_of_video|>
pub const G5V_TOK_VIDEO: u32 = 154_855; // <|video|>

// ---- preprocessing (the vendor pixel contract, serving-capped) ----
/// smart_resize alignment: patch * merge * patch_expand_factor = 28.
pub const G5V_ALIGN: usize = G5V_PATCH * G5V_MERGE;
/// Vendor minimum (Glm5NextImageProcessor.min_image_tokens).
pub const G5V_MIN_TOKENS: usize = 16;
/// SERVING token ceiling per image (merged tokens). The vendor default is 8000, but a
/// grid of `4 * tokens` patches must clear the v1 tower's sdpa shared-memory ceiling
/// (12288 patches -> 3072 merged tokens). smart_resize downsizes larger images INTO the
/// budget (the vendor algorithm's own budget arm), so this is a fidelity knob for very
/// large images, not a refusal — stated in the lane doc.
pub const G5V_SERVE_MAX_TOKENS: usize = 3072;
/// CLIP normalization constants (OPENAI_CLIP_MEAN/STD; processor_config.json truth).
pub const G5V_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const G5V_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// One preprocessed glm5 image, server-carried from the HTTP layer to the GPU worker.
/// `patches` is the tower input (block-major `(c, t, ph, pw)` rows, dropped after the
/// tower forward); the prompt must carry `n_merged()` `<|image|>` placeholders for it.
pub struct Glm5VisionUnit {
    pub patches: Vec<f32>,
    pub gh: usize,
    pub gw: usize,
}

impl Glm5VisionUnit {
    pub fn n_merged(&self) -> usize {
        n_merged_for_grid(self.gh, self.gw)
    }
}

/// Merged tokens a `(gh, gw)` PATCH grid produces — usable from `glm5_plan_image`
/// BEFORE any decode (the placeholder run and the request's token price).
pub fn n_merged_for_grid(gh: usize, gw: usize) -> usize {
    gh * gw / (G5V_MERGE * G5V_MERGE)
}

/// The vendor smart_resize (transformers 5.16.1 image_processing_glm5_next.smart_resize),
/// image shape: num_frames == temporal_factor == 2, factor 28, exact integer arithmetic.
/// Returns the TARGET canvas (aligned_height, aligned_width) in pixels.
pub fn glm5_smart_resize(height: u32, width: u32) -> (u32, u32) {
    let factor = G5V_ALIGN as u64; // 28
    let frames = G5V_TEMPORAL as u64; // aligned_frames == temporal_factor for images
    let pixels_per_token = frames * factor * factor; // 1568
    let min_pixels = G5V_MIN_TOKENS as u64 * pixels_per_token;
    let max_pixels = G5V_SERVE_MAX_TOKENS as u64 * pixels_per_token;
    let (h, w) = (height as u64, width as u64);
    let align = |value: u64| value.div_ceil(factor) * factor;
    let mut aligned_h = align(h.max(1));
    let mut aligned_w = align(w.max(1));
    if frames * aligned_h * aligned_w < min_pixels {
        // Upscale into the minimum budget (vendor: sqrt scale then ceil-align).
        let scale = ((min_pixels as f64) / ((frames * h * w) as f64)).sqrt();
        aligned_h = align(((h as f64 * scale).ceil() as u64).max(1));
        aligned_w = align(((w as f64 * scale).ceil() as u64).max(1));
    }
    if frames * aligned_h * aligned_w > max_pixels {
        // Vendor fit_within_budget: binary search over content_height.
        let (mut low, mut high) = (1u64, h);
        let (mut best_h, mut best_w) = (factor, factor);
        while low <= high {
            let content_h = (low + high) / 2;
            let content_w = ((w * content_h) / h).max(1);
            let (cand_h, cand_w) = (align(content_h), align(content_w));
            if frames * cand_h * cand_w <= max_pixels {
                best_h = cand_h;
                best_w = cand_w;
                low = content_h + 1;
            } else {
                high = content_h - 1;
            }
        }
        aligned_h = best_h;
        aligned_w = best_w;
    }
    (aligned_h as u32, aligned_w as u32)
}

/// Content placement inside the target canvas (vendor resize: aspect-preserving scale,
/// capped at 1.0 unless the image is under the minimum budget; the remainder is
/// right/bottom ZERO padding — black pixels, normalized like any pixel).
pub fn glm5_content_size(height: u32, width: u32, target_h: u32, target_w: u32) -> (u32, u32) {
    let (h, w) = (height as f64, width as f64);
    let mut scale = (target_h as f64 / h).min(target_w as f64 / w);
    let pixels_per_token = (G5V_TEMPORAL * G5V_ALIGN * G5V_ALIGN) as f64;
    if (G5V_TEMPORAL as f64) * h * w >= pixels_per_token * G5V_MIN_TOKENS as f64 {
        scale = scale.min(1.0);
    }
    let content_h = ((h * scale).floor() as u32).clamp(1, target_h);
    let content_w = ((w * scale).floor() as u32).clamp(1, target_w);
    (content_h, content_w)
}

/// PRE-DECODE admission (the hermes decode-bomb law, same as vision_pre/vision_gemma):
/// header dims -> decode-budget check -> target PATCH grid `(gh, gw)`. The placeholder
/// run (and the request's token price) is known before any canvas expands.
pub fn glm5_plan_image(bytes: &[u8]) -> Result<(usize, usize), String> {
    let (w, h) = crate::vision_pre::image_header_dims(bytes)?;
    if w.saturating_mul(h) > crate::vision_pre::IMG_MAX_DECODE_PIXELS {
        return Err(format!(
            "image {w}x{h} exceeds the decode budget ({} px) — refused before decode",
            crate::vision_pre::IMG_MAX_DECODE_PIXELS
        ));
    }
    let (th, tw) = glm5_smart_resize(h as u32, w as u32);
    Ok(((th as usize) / G5V_PATCH, (tw as usize) / G5V_PATCH))
}

/// Decode + resize + pad + normalize + patchify one image: bytes -> (block-major patch
/// rows `[gh*gw, 1176]` in `(c, t, ph, pw)` order with CLIP normalization baked in,
/// gh, gw). Admission runs FIRST (header-only); the decoder is capped to the admitted
/// dimensions. Resample: CatmullRom (the bicubic-class kernel; the vendor path is
/// torchvision bicubic — kernels may differ by a hair, which is why the parity oracle
/// feeds FIXED pixels and the two committed fixtures are identity-resize).
pub fn glm5_prep_image(
    bytes: &[u8],
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    glm5_plan_image(bytes)?;
    let (hw, hh) = crate::vision_pre::image_header_dims(bytes)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(hw as u32);
    limits.max_image_height = Some(hh as u32);
    reader.limits(limits);
    let img = reader.decode()?.to_rgb8();
    let (w0, h0) = img.dimensions();
    let (th, tw) = glm5_smart_resize(h0, w0);
    let (ch, cw) = glm5_content_size(h0, w0, th, tw);
    let content = if (ch, cw) == (h0, w0) {
        img
    } else {
        image::imageops::resize(&img, cw, ch, image::imageops::FilterType::CatmullRom)
    };
    let (gh, gw) = ((th as usize) / G5V_PATCH, (tw as usize) / G5V_PATCH);
    let bw = gw / G5V_MERGE;
    let mut patches = vec![0f32; gh * gw * G5V_PATCH_IN];
    // Block-major token order: (block_row, block_col, in_row, in_col); each row is the
    // (c, t, ph, pw) flat patch with the single frame duplicated across t.
    for t_idx in 0..gh * gw {
        let block = t_idx / (G5V_MERGE * G5V_MERGE);
        let within = t_idx % (G5V_MERGE * G5V_MERGE);
        let py = (block / bw) * G5V_MERGE + within / G5V_MERGE;
        let px = (block % bw) * G5V_MERGE + within % G5V_MERGE;
        let dst = &mut patches[t_idx * G5V_PATCH_IN..(t_idx + 1) * G5V_PATCH_IN];
        for c in 0..3 {
            for ky in 0..G5V_PATCH {
                for kx in 0..G5V_PATCH {
                    let (y, x) = ((py * G5V_PATCH + ky) as u32, (px * G5V_PATCH + kx) as u32);
                    // Right/bottom canvas padding is zero pixels (pre-normalize black).
                    let raw = if y < ch && x < cw {
                        content.get_pixel(x, y)[c] as f32
                    } else {
                        0.0
                    };
                    let v = (raw / 255.0 - G5V_MEAN[c]) / G5V_STD[c];
                    for frame in 0..G5V_TEMPORAL {
                        dst[((c * G5V_TEMPORAL + frame) * G5V_PATCH + ky) * G5V_PATCH + kx] = v;
                    }
                }
            }
        }
    }
    Ok((patches, gh, gw))
}

/// Prep a data-URI image into a Glm5VisionUnit (decode + smart-resize + patchify).
pub fn glm5_prep_data_uri(uri: &str) -> Result<Glm5VisionUnit, String> {
    let bytes = crate::vision_pre::decode_data_uri(uri)?;
    let (patches, gh, gw) = glm5_prep_image(&bytes).map_err(|e| e.to_string())?;
    Ok(Glm5VisionUnit { patches, gh, gw })
}

struct Lin {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    in_f: usize,
    out_f: usize,
}

struct Blk {
    norm1: CudaSlice<f32>,
    norm2: CudaSlice<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    qkv: Lin,
    proj: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
}

pub struct Glm5VisionTower {
    patch: Lin,
    blocks: Vec<Blk>,
    post_ln: CudaSlice<f32>,
    downsample: Lin, // conv [4096, 1024, 2, 2] flattened over (c, ky, kx)
    merger_proj: Lin,
    merger_ln_w: Vec<f32>,
    merger_ln_b: Vec<f32>,
    merger_gate: Lin,
    merger_up: Lin,
    merger_down: Lin,
}

fn read_f32(m: &StModel, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (info, raw) = m
        .raw(name)
        .ok_or_else(|| format!("glm5 vision tensor missing: {name}"))?;
    match info.dtype.as_str() {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        other => Err(format!("glm5 vision tensor {name}: unsupported dtype {other}").into()),
    }
}

fn load_lin(
    e: &Engine,
    m: &StModel,
    stem: &str,
    in_f: usize,
    out_f: usize,
    bias: bool,
) -> Result<Lin, Box<dyn std::error::Error>> {
    let w = read_f32(m, &format!("{stem}.weight"))?;
    if w.len() != in_f * out_f {
        return Err(format!(
            "{stem}.weight has {} elements, expected {in_f}x{out_f}",
            w.len()
        )
        .into());
    }
    let b = if bias {
        let b = read_f32(m, &format!("{stem}.bias"))?;
        if b.len() != out_f {
            return Err(format!("{stem}.bias has {} elements, expected {out_f}", b.len()).into());
        }
        Some(e.htod(&b)?)
    } else {
        None
    };
    Ok(Lin {
        w: e.htod(&w)?,
        b,
        in_f,
        out_f,
    })
}

/// Exact-erf GELU (torch `nn.GELU()` default; erf via Abramowitz & Stegun 7.1.26 in f64,
/// max abs error 1.5e-7 — the same helper the reference oracle uses).
fn gelu_erf(value: f32) -> f32 {
    let x = value as f64 / std::f64::consts::SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = sign * (1.0 - poly * (-x * x).exp());
    (0.5 * value as f64 * (1.0 + erf)) as f32
}

/// The tensor whose PRESENCE decides whether an artifact carries the glm5 vision tower at
/// all (the default-ON detection seam, lane/glm5-vision-default-on): absent = a text-only
/// glm5_next artifact, vision stays off with no flag needed; present = the tower loads and
/// any FURTHER missing/malformed tensor is a boot panic (a drifted artifact, never a
/// silent text-only fallback).
pub const G5V_PROBE_TENSOR: &str = "model.visual.patch_embed.proj.weight";

/// Does this safetensors model (directory with an index, or a single shard file) carry the
/// glm5 vision tower? Presence of `G5V_PROBE_TENSOR` in the index — a metadata read, no
/// tensor bytes touched.
pub fn glm5_visual_tensors_present(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let m = StModel::open(path)?;
    Ok(m.raw(G5V_PROBE_TENSOR).is_some())
}

impl Glm5VisionTower {
    /// Load the tower from a safetensors model (directory with an index, or a single
    /// shard file carrying the `model.visual.*` tensors). f32-resident: ~2.3 GB for the
    /// ~0.58 B-param tower — the correctness-first house posture (BF16 residency is a
    /// measured later optimization, not a v1 decision). The load line reports the measured
    /// device-memory delta (the FLAGS.md VRAM figure's source).
    pub fn load(e: &Engine, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let free_before = e.ctx().mem_get_info().map(|(free, _)| free).unwrap_or(0);
        let m = StModel::open(path)?;
        let p = "model.visual";
        let patch = {
            // conv3d [1024, 3, 2, 14, 14] row-major == Linear rows over the processor's
            // (c, t, ph, pw) flat patch order.
            let w = read_f32(&m, &format!("{p}.patch_embed.proj.weight"))?;
            let b = read_f32(&m, &format!("{p}.patch_embed.proj.bias"))?;
            if w.len() != G5V_HIDDEN * G5V_PATCH_IN || b.len() != G5V_HIDDEN {
                return Err("patch_embed.proj shape mismatch".into());
            }
            Lin {
                w: e.htod(&w)?,
                b: Some(e.htod(&b)?),
                in_f: G5V_PATCH_IN,
                out_f: G5V_HIDDEN,
            }
        };
        let mut blocks = Vec::with_capacity(G5V_DEPTH);
        for il in 0..G5V_DEPTH {
            let bp = format!("{p}.blocks.{il}");
            let q_norm = read_f32(&m, &format!("{bp}.attn.q_norm.weight"))?;
            let k_norm = read_f32(&m, &format!("{bp}.attn.k_norm.weight"))?;
            if q_norm.len() != G5V_HEAD_DIM || k_norm.len() != G5V_HEAD_DIM {
                return Err(format!("block {il} q/k norm width is not {G5V_HEAD_DIM}").into());
            }
            blocks.push(Blk {
                norm1: e.htod(&read_f32(&m, &format!("{bp}.norm1.weight"))?)?,
                norm2: e.htod(&read_f32(&m, &format!("{bp}.norm2.weight"))?)?,
                q_norm,
                k_norm,
                qkv: load_lin(
                    e,
                    &m,
                    &format!("{bp}.attn.qkv"),
                    G5V_HIDDEN,
                    3 * G5V_HIDDEN,
                    true,
                )?,
                proj: load_lin(
                    e,
                    &m,
                    &format!("{bp}.attn.proj"),
                    G5V_HIDDEN,
                    G5V_HIDDEN,
                    true,
                )?,
                gate: load_lin(
                    e,
                    &m,
                    &format!("{bp}.mlp.gate_proj"),
                    G5V_HIDDEN,
                    G5V_INTER,
                    true,
                )?,
                up: load_lin(
                    e,
                    &m,
                    &format!("{bp}.mlp.up_proj"),
                    G5V_HIDDEN,
                    G5V_INTER,
                    true,
                )?,
                down: load_lin(
                    e,
                    &m,
                    &format!("{bp}.mlp.down_proj"),
                    G5V_INTER,
                    G5V_HIDDEN,
                    true,
                )?,
            });
        }
        let post_ln = e.htod(&read_f32(&m, &format!("{p}.post_layernorm.weight"))?)?;
        let downsample = {
            let w = read_f32(&m, &format!("{p}.downsample.weight"))?;
            let b = read_f32(&m, &format!("{p}.downsample.bias"))?;
            if w.len() != G5V_OUT * G5V_HIDDEN * G5V_MERGE * G5V_MERGE || b.len() != G5V_OUT {
                return Err("downsample shape mismatch".into());
            }
            Lin {
                w: e.htod(&w)?,
                b: Some(e.htod(&b)?),
                in_f: G5V_HIDDEN * G5V_MERGE * G5V_MERGE,
                out_f: G5V_OUT,
            }
        };
        let merger_proj = load_lin(e, &m, &format!("{p}.merger.proj"), G5V_OUT, G5V_OUT, false)?;
        let merger_ln_w = read_f32(&m, &format!("{p}.merger.post_projection_norm.weight"))?;
        let merger_ln_b = read_f32(&m, &format!("{p}.merger.post_projection_norm.bias"))?;
        if merger_ln_w.len() != G5V_OUT || merger_ln_b.len() != G5V_OUT {
            return Err("merger.post_projection_norm width is not 4096".into());
        }
        let merger_gate = load_lin(
            e,
            &m,
            &format!("{p}.merger.gate_proj"),
            G5V_OUT,
            G5V_PROJ_INTER,
            false,
        )?;
        let merger_up = load_lin(
            e,
            &m,
            &format!("{p}.merger.up_proj"),
            G5V_OUT,
            G5V_PROJ_INTER,
            false,
        )?;
        let merger_down = load_lin(
            e,
            &m,
            &format!("{p}.merger.down_proj"),
            G5V_PROJ_INTER,
            G5V_OUT,
            false,
        )?;
        let free_after = e
            .ctx()
            .mem_get_info()
            .map(|(free, _)| free)
            .unwrap_or(free_before);
        eprintln!(
            "[glm5-vision] tower loaded from {} ({G5V_DEPTH} blocks, out_width {G5V_OUT}, \
             f32-resident, {:.2} GiB device delta at load)",
            path.display(),
            free_before.saturating_sub(free_after) as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Ok(Self {
            patch,
            blocks,
            post_ln,
            downsample,
            merger_proj,
            merger_ln_w,
            merger_ln_b,
            merger_gate,
            merger_up,
            merger_down,
        })
    }

    /// Embedding width per merged token (== trunk n_embd; admission compares).
    pub fn out_width(&self) -> usize {
        G5V_OUT
    }

    fn linear(
        &self,
        e: &Engine,
        x: &CudaSlice<f32>,
        l: &Lin,
        m: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = e.linear(x, &l.w, m, l.in_f, l.out_f)?;
        if let Some(b) = l.b.as_ref() {
            for r in 0..m {
                e.add_row_inplace(&mut y, b, l.out_f, r * l.out_f)?;
            }
        }
        Ok(y)
    }

    /// Forward one image: host patch rows `[gh*gw, 1176]` in the processor's block-major
    /// (c, t, ph, pw) order -> `[gh*gw/4, 4096]` merged embeddings (device). One attention
    /// segment (grid_thw t == 1 for images; video groups run this per frame pair).
    pub fn forward(
        &self,
        e: &Engine,
        patches: &[f32],
        gh: usize,
        gw: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n = gh * gw;
        if patches.len() != n * G5V_PATCH_IN {
            return Err(format!(
                "patch buffer is {} floats, grid {gh}x{gw} needs {}",
                patches.len(),
                n * G5V_PATCH_IN
            )
            .into());
        }
        if !gh.is_multiple_of(G5V_MERGE) || !gw.is_multiple_of(G5V_MERGE) {
            return Err(format!("grid {gh}x{gw} is not {G5V_MERGE}-aligned").into());
        }
        if n > 12_288 {
            return Err(format!(
                "vision segment {n} patches exceeds the sdpa shared-memory ceiling (12288); \
                 lower the image token budget"
            )
            .into());
        }
        let dbg = std::env::var("MEMRA_VISION_DEBUG").ok();
        let dump = |tag: &str, buf: &[f32]| {
            if let Some(dir) = dbg.as_deref() {
                let raw: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/rust_{tag}.bin"), raw);
            }
        };
        let xd = e.htod(patches)?;
        let mut x = self.linear(e, &xd, &self.patch, n)?;
        if dbg.is_some() {
            dump("pre_blocks", &e.dtoh(&x)?);
        }
        // Block-major position ids ((block_row, block_col, in_row, in_col) — the
        // processor's patchify order) and the 2D rope tables built from them.
        let half = G5V_HEAD_DIM / 2; // 32
        let quarter = half / 2; // 16
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| ROPE_THETA.powf(-((2 * i) as f32) / half as f32))
            .collect();
        let (bh, bw) = (gh / G5V_MERGE, gw / G5V_MERGE);
        let mut rope_cos = vec![0f32; n * half];
        let mut rope_sin = vec![0f32; n * half];
        for t in 0..n {
            let block = t / (G5V_MERGE * G5V_MERGE);
            let within = t % (G5V_MERGE * G5V_MERGE);
            debug_assert!(block < bh * bw);
            let (h, w) = (
                (block / bw) * G5V_MERGE + within / G5V_MERGE,
                (block % bw) * G5V_MERGE + within % G5V_MERGE,
            );
            for d in 0..half {
                let angle = if d < quarter {
                    h as f32 * inv_freq[d]
                } else {
                    w as f32 * inv_freq[d - quarter]
                };
                rope_cos[t * half + d] = angle.cos();
                rope_sin[t * half + d] = angle.sin();
            }
        }
        let scale = 1.0 / (G5V_HEAD_DIM as f32).sqrt();
        let head_rms = |row: &mut [f32], weight: &[f32]| {
            let mut ss = 0f32;
            for v in row.iter() {
                ss += v * v;
            }
            let inv = 1.0 / (ss / G5V_HEAD_DIM as f32 + RMS_EPS).sqrt();
            for (d, v) in row.iter_mut().enumerate() {
                *v *= inv * weight[d];
            }
        };
        for (ib, blk) in self.blocks.iter().enumerate() {
            // attn: rms(norm1) -> fused qkv+bias -> per-head q/k RMS -> rope ->
            //       sdpa(1/sqrt(64), non-causal) -> proj+bias -> +residual
            let mut h = e.zeros(n * G5V_HIDDEN)?;
            e.rms_norm(&x, &blk.norm1, &mut h, G5V_HIDDEN, n, RMS_EPS)?;
            let qkv = self.linear(e, &h, &blk.qkv, n)?;
            let qkv_h = e.dtoh(&qkv)?;
            let mut qh = vec![0f32; n * G5V_HIDDEN];
            let mut kh = vec![0f32; n * G5V_HIDDEN];
            let mut vh = vec![0f32; n * G5V_HIDDEN];
            for t in 0..n {
                let row = &qkv_h[t * 3 * G5V_HIDDEN..(t + 1) * 3 * G5V_HIDDEN];
                let dst = t * G5V_HIDDEN;
                vh[dst..dst + G5V_HIDDEN].copy_from_slice(&row[2 * G5V_HIDDEN..3 * G5V_HIDDEN]);
                for hd in 0..G5V_HEADS {
                    let o = hd * G5V_HEAD_DIM;
                    let mut q = row[o..o + G5V_HEAD_DIM].to_vec();
                    let mut k = row[G5V_HIDDEN + o..G5V_HIDDEN + o + G5V_HEAD_DIM].to_vec();
                    head_rms(&mut q, &blk.q_norm);
                    head_rms(&mut k, &blk.k_norm);
                    for d in 0..half {
                        let (c, s) = (rope_cos[t * half + d], rope_sin[t * half + d]);
                        let (qa, qb) = (q[d], q[d + half]);
                        qh[dst + o + d] = qa * c - qb * s;
                        qh[dst + o + d + half] = qb * c + qa * s;
                        let (ka, kb) = (k[d], k[d + half]);
                        kh[dst + o + d] = ka * c - kb * s;
                        kh[dst + o + d + half] = kb * c + ka * s;
                    }
                }
            }
            let (qd, kd, vd) = (e.htod(&qh)?, e.htod(&kh)?, e.htod(&vh)?);
            let mut od = e.zeros(n * G5V_HIDDEN)?;
            e.sdpa_naive(
                &qd,
                &kd,
                &vd,
                &mut od,
                G5V_HEAD_DIM,
                G5V_HEADS,
                G5V_HEADS,
                n,
                n,
                scale,
                false,
            )?;
            let attn = self.linear(e, &od, &blk.proj, n)?;
            let mut xr = e.zeros(n * G5V_HIDDEN)?;
            e.add(&x, &attn, &mut xr, n * G5V_HIDDEN)?;
            // mlp: rms(norm2) -> clamped SwiGLU (gate max-clamp, up +/- clamp) -> down -> +res
            let mut h2 = e.zeros(n * G5V_HIDDEN)?;
            e.rms_norm(&xr, &blk.norm2, &mut h2, G5V_HIDDEN, n, RMS_EPS)?;
            let gate = self.linear(e, &h2, &blk.gate, n)?;
            let up = self.linear(e, &h2, &blk.up, n)?;
            let (gh_, uh) = (e.dtoh(&gate)?, e.dtoh(&up)?);
            let mut act = vec![0f32; n * G5V_INTER];
            for i in 0..n * G5V_INTER {
                let g = gh_[i].min(G5V_LIMIT);
                let u = uh[i].clamp(-G5V_LIMIT, G5V_LIMIT);
                act[i] = g / (1.0 + (-g).exp()) * u; // silu(g) * u
            }
            let ad = e.htod(&act)?;
            let down = self.linear(e, &ad, &blk.down, n)?;
            let mut xn = e.zeros(n * G5V_HIDDEN)?;
            e.add(&xr, &down, &mut xn, n * G5V_HIDDEN)?;
            x = xn;
            if dbg.is_some() && ib == 0 {
                dump("blk0", &e.dtoh(&x)?);
            }
        }
        // post-encoder RMS, then the conv 2x2 downsample over block-major token groups.
        let mut post = e.zeros(n * G5V_HIDDEN)?;
        e.rms_norm(&x, &self.post_ln, &mut post, G5V_HIDDEN, n, RMS_EPS)?;
        let post_h = e.dtoh(&post)?;
        if dbg.is_some() {
            dump("post_blocks", &post_h);
        }
        let groups = n / (G5V_MERGE * G5V_MERGE);
        // Conv input layout per group: (c, ky, kx) — matches the conv weight's row-major
        // [out, c, ky, kx] flattening, so the downsample runs as one Linear.
        let mut gathered = vec![0f32; groups * self.downsample.in_f];
        for g in 0..groups {
            let dst = &mut gathered[g * self.downsample.in_f..(g + 1) * self.downsample.in_f];
            for c in 0..G5V_HIDDEN {
                for ky in 0..G5V_MERGE {
                    for kx in 0..G5V_MERGE {
                        let t = g * G5V_MERGE * G5V_MERGE + ky * G5V_MERGE + kx;
                        dst[(c * G5V_MERGE + ky) * G5V_MERGE + kx] = post_h[t * G5V_HIDDEN + c];
                    }
                }
            }
        }
        let gd = e.htod(&gathered)?;
        let pooled = self.linear(e, &gd, &self.downsample, groups)?;
        if dbg.is_some() {
            dump("downsample", &e.dtoh(&pooled)?);
        }
        // merger: proj (no bias) -> LayerNorm(w, b) -> exact GELU -> clamped SwiGLU -> down
        let proj = self.linear(e, &pooled, &self.merger_proj, groups)?;
        let mut proj_h = e.dtoh(&proj)?;
        for row in proj_h.chunks_exact_mut(G5V_OUT) {
            let mean = row.iter().sum::<f32>() / G5V_OUT as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / G5V_OUT as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for (c, v) in row.iter_mut().enumerate() {
                *v = gelu_erf((*v - mean) * inv * self.merger_ln_w[c] + self.merger_ln_b[c]);
            }
        }
        let md = e.htod(&proj_h)?;
        let gate = self.linear(e, &md, &self.merger_gate, groups)?;
        let up = self.linear(e, &md, &self.merger_up, groups)?;
        let (gh_, uh) = (e.dtoh(&gate)?, e.dtoh(&up)?);
        let mut act = vec![0f32; groups * G5V_PROJ_INTER];
        for i in 0..groups * G5V_PROJ_INTER {
            let g = gh_[i].min(G5V_LIMIT);
            let u = uh[i].clamp(-G5V_LIMIT, G5V_LIMIT);
            act[i] = g / (1.0 + (-g).exp()) * u;
        }
        let ad = e.htod(&act)?;
        let out = self.linear(e, &ad, &self.merger_down, groups)?;
        if dbg.is_some() {
            dump("projected", &e.dtoh(&out)?);
        }
        Ok(out)
    }
}
