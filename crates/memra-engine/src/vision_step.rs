//! Vision tower for StepFun Step-3.7-Flash (arch step35), lane/step37-vision 2026-08-30.
//!
//! step37 is its OWN semantic program. Every law here was derived from the pinned
//! artifact (HF stepfun-ai/Step-3.7-Flash-NVFP4 @ 4275532f): config.json
//! `vision_config` (model_type perception_encoder) plus the vendor reference code the
//! checkpoint ships (vision_encoder.py / processing_step3.py / modeling_step3p7.py)
//! and the shard-header tensor census. Full census + plan:
//! research/step37-vision-20260830/CENSUS.md. Nothing is inherited from the qwen3_5
//! or gemma4 towers by analogy; the census decided this is a third program:
//!
//! - ViT: 47 blocks, hidden 1536, 16 heads (head_dim 96), mlp 8960, patch 14,
//!   CLIP lineage: LayerNorm WITH biases (eps 1e-5), fused in_proj qkv (+bias),
//!   out_proj (+bias), quick_gelu MLP (x*sigmoid(1.702x); NOT gelu_tanh, NOT GEGLU),
//!   and LayerScale gammas (ls_1/ls_2) on both residual branches, which neither
//!   shipped tower has. ln_pre only (use_ln_post false, use_cls_token false).
//! - Positions enter twice: a learned 52x52 absolute table added to the patch
//!   embeddings (other grids: bilinear interpolation, align_corners=FALSE — the
//!   qwen3_5 table interpolates align_corners=TRUE, flagged so nobody unifies them),
//!   and a per-layer 2D rope over the FULL head_dim 96: first 48 dims rotate by the
//!   COLUMN, last 48 by the ROW, theta 10000, INTERLEAVED pairing ((2i, 2i+1) share
//!   one angle — GPT-J style; qwen and gemma both pair NeoX-style (d, d+half/2)).
//! - Attention: sdpa scale 1/sqrt(96), non-causal, one image or one 504-crop per
//!   segment (the reference batches tiles on the batch dim; attention never crosses
//!   tiles). No video path exists for this family.
//! - Head: NOT a merger. [n,1536] reshapes to [1536,g,g], runs two OVERLAPPING 3x3
//!   stride-2 pad-1 convs (1536->3072->6144, biases), row-major flatten, then
//!   vit_large_projector 6144->4096 (no bias). 52-grid -> 169 rows, 36-grid -> 81.
//! - Preprocessing (processing_step3.py): CLIP mean/std, every ViT input a SQUARE
//!   bilinear resize (728 main view, 504 crop tiles); ImagePatcher tiling for large /
//!   extreme-aspect images (window law in `determine_window_size`). The vendor
//!   Compose normalizes BEFORE resizing; per-channel affine commutes with the linear
//!   resample, memra resizes first (parity arbitrated by the fixed-pixel oracle).
//! - Token layout per image (ids from the tokenizer, hardcoded nowhere): crops FIRST
//!   (<patch_start> + 81 pads + <patch_end>, <patch_newline> per full tile row except
//!   a trailing one), then <im_start> + 169 pads + <im_end>. Embedding rows replace
//!   pad positions in order; delimiters keep their text embeddings. Image spans are
//!   CAUSAL in the LM (standard create_causal_mask in the reference — unlike gemma4's
//!   bidirectional islands), so the existing overlay prime path is the correct one.
//!
//! v1 posture matches the shipped towers: correctness-first — f32 GEMMs
//! (Engine::linear + add_row_inplace bias), sdpa_naive, host-side rope / LayerScale /
//! quick_gelu / im2col; parity gate before any serving path (bin/step_vision_oracle).

use crate::Engine;
use cudarc::driver::CudaSlice;
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::safetensors::StModel;
use std::path::Path;

pub const SV_HIDDEN: usize = 1536;
pub const SV_HEADS: usize = 16;
pub const SV_HEAD_DIM: usize = SV_HIDDEN / SV_HEADS; // 96
pub const SV_INTER: usize = 8960; // int(1536 * 5.8333...)
pub const SV_DEPTH: usize = 47;
pub const SV_PATCH: usize = 14;
pub const SV_POS_GRID: usize = 52; // 728 / 14, the learned-table grid
pub const SV_PATCH_IN: usize = 3 * SV_PATCH * SV_PATCH; // 588
/// Main-view edge (px) and its patch grid; crop-tile edge and grid.
pub const SV_IMAGE_SIZE: usize = 728;
pub const SV_TILE_SIZE: usize = 504;
pub const SV_GRID_MAIN: usize = SV_IMAGE_SIZE / SV_PATCH; // 52
pub const SV_GRID_TILE: usize = SV_TILE_SIZE / SV_PATCH; // 36
/// Trunk rows per view after the two stride-2 downsamplers (52->26->13, 36->18->9).
pub const SV_MAIN_ROWS: usize = 169;
pub const SV_TILE_ROWS: usize = 81;
/// ImagePatcher long-side cap before tiling (MAX_IMAGE_SIZE in processing_step3.py).
pub const SV_MAX_IMAGE_SIZE: usize = 3024;
const LN_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10000.0;
/// CLIP normalization (processing_step3.py Step3VisionProcessor).
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

struct Lin {
    w: CudaSlice<f32>,
    b: Option<CudaSlice<f32>>,
    in_f: usize,
    out_f: usize,
}

struct SBlock {
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    ln2_w: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
    ls1: Vec<f32>,
    ls2: Vec<f32>,
    qkv: Lin,
    proj: Lin,
    fc: Lin,
    cproj: Lin,
}

/// 3x3 stride-2 pad-1 conv as im2col + GEMM: weight rows are the PyTorch
/// [C_out, C_in, 3, 3] flatten, i.e. (c, ky, kx) inner order.
struct Conv3x3s2 {
    w: CudaSlice<f32>, // [C_out, C_in*9]
    b: CudaSlice<f32>,
    c_in: usize,
    c_out: usize,
}

pub struct StepVisionTower {
    patch: Lin, // conv1 14x14 stride-14 (no bias) as Linear 588 -> 1536
    /// Host copy of the learned pos table [2704, 1536].
    pos: Vec<f32>,
    ln_pre_w: CudaSlice<f32>,
    ln_pre_b: CudaSlice<f32>,
    blocks: Vec<SBlock>,
    down1: Conv3x3s2, // 1536 -> 3072
    down2: Conv3x3s2, // 3072 -> 6144
    proj: Lin,        // vit_large_projector 6144 -> 4096, no bias
}

fn read_f32(m: &StModel, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (info, raw) = m
        .raw(name)
        .ok_or_else(|| format!("step vision tensor missing: {name}"))?;
    match info.dtype.as_str() {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        other => Err(format!("step vision tensor {name}: unsupported dtype {other}").into()),
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
    assert_eq!(w.len(), in_f * out_f, "{stem}.weight shape");
    let b = if bias {
        let b = read_f32(m, &format!("{stem}.bias"))?;
        assert_eq!(b.len(), out_f, "{stem}.bias shape");
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

// ---------------- preprocessing (processing_step3.py, pinned rev) ----------------

/// `determine_window_size`: the crop-tiling decision. 0 = no tiles.
fn window_size(long: usize, short: usize) -> usize {
    if long <= SV_IMAGE_SIZE {
        if long as f64 / short as f64 > 1.5 {
            short
        } else {
            0
        }
    } else if long as f64 / short as f64 > 4.0 {
        short.min(SV_TILE_SIZE)
    } else {
        SV_TILE_SIZE
    }
}

/// `get_image_size_for_padding`: tiny extreme-aspect images pad to a black square.
fn pad_rule(w: usize, h: usize) -> (usize, usize) {
    let ratio = w as f64 / h as f64;
    if w.min(h) < 32 && !(0.25..=4.0).contains(&ratio) {
        let s = w.max(h);
        (s, s)
    } else {
        (w, h)
    }
}

/// `get_image_size_for_preprocess`: cap the long side at 3024 (int() truncation).
fn cap_rule(w: usize, h: usize) -> (usize, usize) {
    if w.max(h) > SV_MAX_IMAGE_SIZE {
        let s = SV_MAX_IMAGE_SIZE as f64 / w.max(h) as f64;
        ((w as f64 * s) as usize, (h as f64 * s) as usize)
    } else {
        (w, h)
    }
}

/// `get_image_size_for_crop`: snap a side to whole windows with the 0.2 overflow rule.
fn crop_snap(side: usize, win: usize) -> usize {
    let ratio = side as f64 / win as f64;
    if ratio < 1.0 {
        return side;
    }
    let whole = side / win;
    let n = if ratio - whole as f64 > 0.2 {
        whole + 1
    } else {
        whole
    };
    win * n
}

/// Tiling plan for an image of (w, h): tile count, tiles per row (x_num), and the
/// per-tile newline mask (vendor law: a newline after each full tile row, except a
/// trailing one on the final tile). Derivable from HEADER dims alone, so the pad run
/// and the request's token price are known before any canvas expands.
pub struct StepImagePlan {
    pub n_tiles: usize,
    pub newline_mask: Vec<bool>,
}

impl StepImagePlan {
    /// Trunk embedding rows this image occupies (pads only, not delimiters).
    pub fn n_rows(&self) -> usize {
        self.n_tiles * SV_TILE_ROWS + SV_MAIN_ROWS
    }
    /// Prompt TOKENS the placeholder expansion renders (pads + delimiters + newlines).
    pub fn n_prompt_tokens(&self) -> usize {
        let newlines = self.newline_mask.iter().filter(|&&b| b).count();
        self.n_tiles * (SV_TILE_ROWS + 2) + newlines + SV_MAIN_ROWS + 2
    }
}

fn plan_for_dims(w0: usize, h0: usize) -> StepImagePlan {
    let (w, h) = pad_rule(w0, h0);
    let (w, h) = cap_rule(w, h);
    let win = window_size(w.max(h), w.min(h));
    if win == 0 {
        return StepImagePlan {
            n_tiles: 0,
            newline_mask: Vec::new(),
        };
    }
    let (cw, ch) = (crop_snap(w, win), crop_snap(h, win));
    // slide_window with size == step: whole non-overlapping tiles (cw, ch are snapped
    // to multiples of win when >= win; a side < win yields one column/row).
    let x_num = (cw / win).max(1);
    let y_num = (ch / win).max(1);
    let n = x_num * y_num;
    let mut mask = vec![false; n];
    let mut newlines: Vec<usize> = (0..n).filter(|i| (i + 1) % x_num == 0).collect();
    if newlines.last() == Some(&(n - 1)) {
        newlines.pop(); // the vendor pops a trailing row-final newline
    }
    for i in newlines {
        mask[i] = true;
    }
    StepImagePlan {
        n_tiles: n,
        newline_mask: mask,
    }
}

/// PRE-DECODE admission (hermes decode-bomb law, same as the qwen/gemma planners):
/// header dims -> decode-budget check -> tiling plan. No canvas expands here.
pub fn step_plan_image(bytes: &[u8]) -> Result<StepImagePlan, String> {
    let (w, h) = crate::vision_pre::image_header_dims(bytes)?;
    if w.saturating_mul(h) > crate::vision_pre::IMG_MAX_DECODE_PIXELS {
        return Err(format!(
            "image {w}x{h} exceeds the decode budget ({} px) — refused before decode",
            crate::vision_pre::IMG_MAX_DECODE_PIXELS
        ));
    }
    if w < 2 || h < 2 {
        return Err(format!("image too small: {w}x{h}"));
    }
    Ok(plan_for_dims(w, h))
}

/// One preprocessed step37 image: the 728 main view plus its 504 crop tiles, each as
/// patch rows the tower consumes directly. Carried from the HTTP layer to the GPU
/// worker; the patch buffers drop after the tower forward.
pub struct StepVisionUnit {
    /// [52*52, 588] rows, (c, ky, kx) inner order.
    pub main: Vec<f32>,
    /// Each [36*36, 588]; slide-window order (row-major over the tile grid).
    pub tiles: Vec<Vec<f32>>,
    pub newline_mask: Vec<bool>,
}

impl StepVisionUnit {
    pub fn n_rows(&self) -> usize {
        self.tiles.len() * SV_TILE_ROWS + SV_MAIN_ROWS
    }
}

/// Normalize + patchify one square RGB view into [g*g, 588] rows.
fn patchify(img: &image::RgbImage, g: usize) -> Vec<f32> {
    let mut rows = vec![0f32; g * g * SV_PATCH_IN];
    for py in 0..g {
        for px in 0..g {
            let dst = &mut rows[(py * g + px) * SV_PATCH_IN..(py * g + px + 1) * SV_PATCH_IN];
            for c in 0..3 {
                for ky in 0..SV_PATCH {
                    for kx in 0..SV_PATCH {
                        let p =
                            img.get_pixel((px * SV_PATCH + kx) as u32, (py * SV_PATCH + ky) as u32);
                        dst[(c * SV_PATCH + ky) * SV_PATCH + kx] =
                            ((p[c] as f32) / 255.0 - MEAN[c]) / STD[c];
                    }
                }
            }
        }
    }
    rows
}

/// Decode + preprocess one image: bytes -> main view + crop tiles per the vendor
/// pipeline. Admission runs FIRST (header-only); the decoder is capped to the admitted
/// dimensions. The returned unit's tile count MUST match the header plan — the caller
/// refuses on drift (pad runs are already rendered from the plan).
pub fn step_prep_image(bytes: &[u8]) -> Result<StepVisionUnit, Box<dyn std::error::Error>> {
    step_plan_image(bytes)?;
    let (hw, hh) = crate::vision_pre::image_header_dims(bytes)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(hw as u32);
    limits.max_image_height = Some(hh as u32);
    reader.limits(limits);
    let mut img = reader.decode()?.to_rgb8();
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    // 1. tiny extreme-aspect pad-to-square (paste at (0,0), black fill)
    let (pw, ph) = pad_rule(w0, h0);
    if (pw, ph) != (w0, h0) {
        let mut padded = image::RgbImage::new(pw as u32, ph as u32);
        image::imageops::replace(&mut padded, &img, 0, 0);
        img = padded;
    }
    // 2. long-side cap at 3024 (aspect kept; bilinear, like the vendor's PIL resize)
    let (cw, ch) = cap_rule(img.width() as usize, img.height() as usize);
    if (cw, ch) != (img.width() as usize, img.height() as usize) {
        img = image::imageops::resize(
            &img,
            cw as u32,
            ch as u32,
            image::imageops::FilterType::Triangle,
        );
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    // 3. main view: square 728 resize of the (padded/capped) image
    let main_img = image::imageops::resize(
        &img,
        SV_IMAGE_SIZE as u32,
        SV_IMAGE_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let main = patchify(&main_img, SV_GRID_MAIN);
    // 4. crop tiles
    let win = window_size(w.max(h), w.min(h));
    let (mut tiles, mut newline_mask) = (Vec::new(), Vec::new());
    if win > 0 {
        let (sw, sh) = (crop_snap(w, win), crop_snap(h, win));
        let snapped = if (sw, sh) != (w, h) {
            image::imageops::resize(
                &img,
                sw as u32,
                sh as u32,
                image::imageops::FilterType::Triangle,
            )
        } else {
            img
        };
        let x_num = (sw / win).max(1);
        let y_num = (sh / win).max(1);
        let n = x_num * y_num;
        for ty in 0..y_num {
            for tx in 0..x_num {
                let crop = image::imageops::crop_imm(
                    &snapped,
                    (tx * win) as u32,
                    (ty * win) as u32,
                    win as u32,
                    win as u32,
                )
                .to_image();
                let tile = image::imageops::resize(
                    &crop,
                    SV_TILE_SIZE as u32,
                    SV_TILE_SIZE as u32,
                    image::imageops::FilterType::Triangle,
                );
                tiles.push(patchify(&tile, SV_GRID_TILE));
            }
        }
        let mut newlines: Vec<usize> = (0..n).filter(|i| (i + 1) % x_num == 0).collect();
        if newlines.last() == Some(&(n - 1)) {
            newlines.pop();
        }
        newline_mask = vec![false; n];
        for i in newlines {
            newline_mask[i] = true;
        }
    }
    Ok(StepVisionUnit {
        main,
        tiles,
        newline_mask,
    })
}

// ---------------- tower ----------------

impl StepVisionTower {
    /// Load the tower from the serving artifact's own directory (the vision tensors
    /// live unquantized, BF16, inside the NVFP4 checkpoint: `model.vision_model.*` +
    /// `model.vit_large_projector.weight`, routed by model.safetensors.index.json).
    /// Refuses any directory whose tensors do not census as this exact program.
    pub fn load(e: &Engine, dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let m = StModel::open(dir)?;
        let p = "model.vision_model";
        let patch = {
            // conv1 [1536, 3, 14, 14] stride 14, no bias == Linear over (c, ky, kx)
            // 588-float patch rows (the patchify order above).
            let w = read_f32(&m, &format!("{p}.conv1.weight"))?;
            assert_eq!(w.len(), SV_HIDDEN * SV_PATCH_IN, "conv1.weight shape");
            Lin {
                w: e.htod(&w)?,
                b: None,
                in_f: SV_PATCH_IN,
                out_f: SV_HIDDEN,
            }
        };
        let pos = read_f32(&m, &format!("{p}.positional_embedding"))?;
        assert_eq!(
            pos.len(),
            SV_POS_GRID * SV_POS_GRID * SV_HIDDEN,
            "positional_embedding shape"
        );
        let ln_pre_w = e.htod(&read_f32(&m, &format!("{p}.ln_pre.weight"))?)?;
        let ln_pre_b = e.htod(&read_f32(&m, &format!("{p}.ln_pre.bias"))?)?;
        let mut blocks = Vec::with_capacity(SV_DEPTH);
        for il in 0..SV_DEPTH {
            let bp = format!("{p}.transformer.resblocks.{il}");
            let ls1 = read_f32(&m, &format!("{bp}.ls_1.gamma"))?;
            let ls2 = read_f32(&m, &format!("{bp}.ls_2.gamma"))?;
            assert_eq!(ls1.len(), SV_HIDDEN, "ls_1.gamma shape");
            assert_eq!(ls2.len(), SV_HIDDEN, "ls_2.gamma shape");
            blocks.push(SBlock {
                ln1_w: e.htod(&read_f32(&m, &format!("{bp}.ln_1.weight"))?)?,
                ln1_b: e.htod(&read_f32(&m, &format!("{bp}.ln_1.bias"))?)?,
                ln2_w: e.htod(&read_f32(&m, &format!("{bp}.ln_2.weight"))?)?,
                ln2_b: e.htod(&read_f32(&m, &format!("{bp}.ln_2.bias"))?)?,
                ls1,
                ls2,
                qkv: {
                    // fused in_proj: weight [4608, 1536] + bias [4608], chunk order q,k,v
                    let w = read_f32(&m, &format!("{bp}.attn.in_proj_weight"))?;
                    let b = read_f32(&m, &format!("{bp}.attn.in_proj_bias"))?;
                    assert_eq!(w.len(), 3 * SV_HIDDEN * SV_HIDDEN, "in_proj_weight shape");
                    assert_eq!(b.len(), 3 * SV_HIDDEN, "in_proj_bias shape");
                    Lin {
                        w: e.htod(&w)?,
                        b: Some(e.htod(&b)?),
                        in_f: SV_HIDDEN,
                        out_f: 3 * SV_HIDDEN,
                    }
                },
                proj: load_lin(
                    e,
                    &m,
                    &format!("{bp}.attn.out_proj"),
                    SV_HIDDEN,
                    SV_HIDDEN,
                    true,
                )?,
                fc: load_lin(e, &m, &format!("{bp}.mlp.c_fc"), SV_HIDDEN, SV_INTER, true)?,
                cproj: load_lin(
                    e,
                    &m,
                    &format!("{bp}.mlp.c_proj"),
                    SV_INTER,
                    SV_HIDDEN,
                    true,
                )?,
            });
        }
        let load_conv = |stem: &str,
                         c_in: usize,
                         c_out: usize|
         -> Result<Conv3x3s2, Box<dyn std::error::Error>> {
            let w = read_f32(&m, &format!("{stem}.weight"))?;
            let b = read_f32(&m, &format!("{stem}.bias"))?;
            assert_eq!(w.len(), c_out * c_in * 9, "{stem}.weight shape");
            assert_eq!(b.len(), c_out, "{stem}.bias shape");
            Ok(Conv3x3s2 {
                w: e.htod(&w)?,
                b: e.htod(&b)?,
                c_in,
                c_out,
            })
        };
        let down1 = load_conv(&format!("{p}.vit_downsampler1"), SV_HIDDEN, 2 * SV_HIDDEN)?;
        let down2 = load_conv(
            &format!("{p}.vit_downsampler2"),
            2 * SV_HIDDEN,
            4 * SV_HIDDEN,
        )?;
        let proj = {
            // vit_large_projector [n_embd, 6144], projector_bias false. Output width is
            // the TRUNK's n_embd, derived from the tensor, never assumed — admission
            // compares the serving trunk's n_embd against `out_width()`.
            let w = read_f32(&m, "model.vit_large_projector.weight")?;
            assert_eq!(w.len() % (4 * SV_HIDDEN), 0, "vit_large_projector shape");
            let out_f = w.len() / (4 * SV_HIDDEN);
            Lin {
                w: e.htod(&w)?,
                b: None,
                in_f: 4 * SV_HIDDEN,
                out_f,
            }
        };
        eprintln!(
            "[step-vision] tower loaded from {} ({SV_DEPTH} blocks, out_width {}, f32-resident)",
            dir.display(),
            proj.out_f
        );
        Ok(Self {
            patch,
            pos,
            ln_pre_w,
            ln_pre_b,
            blocks,
            down1,
            down2,
            proj,
        })
    }

    /// Embedding width this tower emits per row (the projector out_features == the
    /// trunk n_embd of the checkpoint it loaded from; 4096 on Step-3.7-Flash).
    pub fn out_width(&self) -> usize {
        self.proj.out_f
    }

    fn linear_bias(
        &self,
        e: &Engine,
        x: &CudaSlice<f32>,
        l: &Lin,
        m: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = e.linear(x, &l.w, m, l.in_f, l.out_f)?;
        if let Some(b) = &l.b {
            for r in 0..m {
                e.add_row_inplace(&mut y, b, l.out_f, r * l.out_f)?;
            }
        }
        Ok(y)
    }

    /// Bilinear-interpolate the 52x52 learned pos table to [g, g] on host,
    /// align_corners=FALSE (torch F.interpolate default — the vendor's
    /// sample_abs_posemb; NOT the qwen table's align_corners=true law).
    fn pos_for_grid(&self, g: usize) -> Vec<f32> {
        if g == SV_POS_GRID {
            return self.pos.clone();
        }
        let scale = SV_POS_GRID as f32 / g as f32;
        let mut out = vec![0f32; g * g * SV_HIDDEN];
        for y in 0..g {
            for x in 0..g {
                let sy = ((y as f32 + 0.5) * scale - 0.5).clamp(0.0, (SV_POS_GRID - 1) as f32);
                let sx = ((x as f32 + 0.5) * scale - 0.5).clamp(0.0, (SV_POS_GRID - 1) as f32);
                let (y0, x0) = (sy.floor() as usize, sx.floor() as usize);
                let (y1, x1) = ((y0 + 1).min(SV_POS_GRID - 1), (x0 + 1).min(SV_POS_GRID - 1));
                let (fy, fx) = (sy - y0 as f32, sx - x0 as f32);
                let dst = &mut out[(y * g + x) * SV_HIDDEN..(y * g + x + 1) * SV_HIDDEN];
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit channel index keeps the four-corner offset arithmetic
                // visible and aligned across the four `self.pos` reads and the `dst` write
                for c in 0..SV_HIDDEN {
                    let p00 = self.pos[(y0 * SV_POS_GRID + x0) * SV_HIDDEN + c];
                    let p01 = self.pos[(y0 * SV_POS_GRID + x1) * SV_HIDDEN + c];
                    let p10 = self.pos[(y1 * SV_POS_GRID + x0) * SV_HIDDEN + c];
                    let p11 = self.pos[(y1 * SV_POS_GRID + x1) * SV_HIDDEN + c];
                    dst[c] = p00 * (1.0 - fy) * (1.0 - fx)
                        + p01 * (1.0 - fy) * fx
                        + p10 * fy * (1.0 - fx)
                        + p11 * fy * fx;
                }
            }
        }
        out
    }

    /// im2col for a 3x3 stride-2 pad-1 conv over a [c_in, g, g] feature map held as
    /// token-major [g*g, c_in] rows: emits [og*og, c_in*9] rows in (c, ky, kx) inner
    /// order (the PyTorch conv-weight flatten), og = floor((g - 1) / 2) + 1.
    fn im2col(x: &[f32], g: usize, c_in: usize) -> (Vec<f32>, usize) {
        let og = (g - 1) / 2 + 1;
        let mut out = vec![0f32; og * og * c_in * 9];
        for oy in 0..og {
            for ox in 0..og {
                let dst = &mut out[(oy * og + ox) * c_in * 9..(oy * og + ox + 1) * c_in * 9];
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let iy = (2 * oy + ky) as isize - 1;
                        let ix = (2 * ox + kx) as isize - 1;
                        if iy < 0 || ix < 0 || iy >= g as isize || ix >= g as isize {
                            continue; // zero padding
                        }
                        let src = &x[((iy as usize) * g + ix as usize) * c_in..];
                        for c in 0..c_in {
                            dst[c * 9 + ky * 3 + kx] = src[c];
                        }
                    }
                }
            }
        }
        (out, og)
    }

    /// Forward ONE view (the 728 main image at g=52, or one 504 crop tile at g=36):
    /// host patch rows [g*g, 588] -> device [rows, out_width] projector output,
    /// rows = (g/4 rounded per the two stride-2 convs)^2 (169 or 81).
    pub fn forward(
        &self,
        e: &Engine,
        patches: &[f32],
        g: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n = g * g;
        assert_eq!(patches.len(), n * SV_PATCH_IN, "patch buffer shape");
        if n > 12288 {
            return Err(format!(
                "step vision segment {n} patches exceeds the sdpa shared-memory ceiling (12288)"
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
        // patch embed (no bias) + learned abs posemb + ln_pre
        let xd = e.htod(patches)?;
        let embedded = self.linear_bias(e, &xd, &self.patch, n)?;
        let pos = self.pos_for_grid(g);
        let pos_d = e.htod(&pos)?;
        let mut summed = e.zeros(n * SV_HIDDEN)?;
        e.add(&embedded, &pos_d, &mut summed, n * SV_HIDDEN)?;
        let mut x = e.zeros(n * SV_HIDDEN)?;
        e.layer_norm_bias(
            &summed,
            &self.ln_pre_w,
            &self.ln_pre_b,
            &mut x,
            SV_HIDDEN,
            n,
            LN_EPS,
        )?;
        if dbg.is_some() {
            dump("pre_blocks", &e.dtoh(&x)?);
        }
        // 2D rope tables (EncoderRope2D): dim = head_dim = 96; inv_freq[i] =
        // theta^(-2i/48), i in 0..24; per token (row, col) the FIRST 48 dims carry
        // col-angles, the LAST 48 row-angles, each angle repeated for the INTERLEAVED
        // pair (2i, 2i+1) inside its half. Same table every block/head.
        let half = SV_HEAD_DIM / 2; // 48
        let quarter = half / 2; // 24 distinct angles per half
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| ROPE_THETA.powf(-2.0 * (i as f32) / half as f32))
            .collect();
        let mut cos_t = vec![0f32; n * half]; // [token, 48]: 24 col-angles then 24 row-angles
        let mut sin_t = vec![0f32; n * half];
        for t in 0..n {
            let (row, col) = (t / g, t % g);
            for i in 0..quarter {
                let ac = col as f32 * inv_freq[i];
                let ar = row as f32 * inv_freq[i];
                cos_t[t * half + i] = ac.cos();
                sin_t[t * half + i] = ac.sin();
                cos_t[t * half + quarter + i] = ar.cos();
                sin_t[t * half + quarter + i] = ar.sin();
            }
        }
        let scale = 1.0 / (SV_HEAD_DIM as f32).sqrt();
        for (ib, blk) in self.blocks.iter().enumerate() {
            // attn: ln1 -> fused qkv -> rope2d(q,k) -> sdpa(non-causal, 1/sqrt(96))
            //       -> out_proj -> ls_1 -> +res
            let mut h = e.zeros(n * SV_HIDDEN)?;
            e.layer_norm_bias(&x, &blk.ln1_w, &blk.ln1_b, &mut h, SV_HIDDEN, n, LN_EPS)?;
            let qkv = self.linear_bias(e, &h, &blk.qkv, n)?;
            let qkv_h = e.dtoh(&qkv)?;
            let mut qh = vec![0f32; n * SV_HIDDEN];
            let mut kh = vec![0f32; n * SV_HIDDEN];
            let mut vh = vec![0f32; n * SV_HIDDEN];
            for t in 0..n {
                let row = &qkv_h[t * 3 * SV_HIDDEN..(t + 1) * 3 * SV_HIDDEN];
                let dst = t * SV_HIDDEN;
                vh[dst..dst + SV_HIDDEN].copy_from_slice(&row[2 * SV_HIDDEN..3 * SV_HIDDEN]);
                for hd in 0..SV_HEADS {
                    let o = hd * SV_HEAD_DIM;
                    // interleaved pairs (2i, 2i+1) per half; angle index i within the
                    // half's 24-entry table (col-half at 0, row-half at +quarter... the
                    // cos_t row is [col 0..24, row 0..24], halves at dim 0..48 / 48..96)
                    for hf in 0..2usize {
                        for i in 0..quarter {
                            let (c, s) = (
                                cos_t[t * half + hf * quarter + i],
                                sin_t[t * half + hf * quarter + i],
                            );
                            let d = hf * half + 2 * i;
                            let (qa, qb) = (row[o + d], row[o + d + 1]);
                            qh[dst + o + d] = qa * c - qb * s;
                            qh[dst + o + d + 1] = qb * c + qa * s;
                            let (ka, kb) = (row[SV_HIDDEN + o + d], row[SV_HIDDEN + o + d + 1]);
                            kh[dst + o + d] = ka * c - kb * s;
                            kh[dst + o + d + 1] = kb * c + ka * s;
                        }
                    }
                }
            }
            let (qd, kd, vd) = (e.htod(&qh)?, e.htod(&kh)?, e.htod(&vh)?);
            let mut od = e.zeros(n * SV_HIDDEN)?;
            e.sdpa_naive(
                &qd,
                &kd,
                &vd,
                &mut od,
                SV_HEAD_DIM,
                SV_HEADS,
                SV_HEADS,
                n,
                n,
                scale,
                false,
            )?;
            let attn = self.linear_bias(e, &od, &blk.proj, n)?;
            // LayerScale then residual (host: per-channel gamma multiply)
            let mut ah = e.dtoh(&attn)?;
            for t in 0..n {
                for c in 0..SV_HIDDEN {
                    ah[t * SV_HIDDEN + c] *= blk.ls1[c];
                }
            }
            let ad = e.htod(&ah)?;
            let mut xr = e.zeros(n * SV_HIDDEN)?;
            e.add(&x, &ad, &mut xr, n * SV_HIDDEN)?;
            // mlp: ln2 -> c_fc -> quick_gelu -> c_proj -> ls_2 -> +res
            let mut h2 = e.zeros(n * SV_HIDDEN)?;
            e.layer_norm_bias(&xr, &blk.ln2_w, &blk.ln2_b, &mut h2, SV_HIDDEN, n, LN_EPS)?;
            let f1 = self.linear_bias(e, &h2, &blk.fc, n)?;
            let mut fh = e.dtoh(&f1)?;
            for v in fh.iter_mut() {
                // quick_gelu(x) = x * sigmoid(1.702 x) — NOT the tanh approximation.
                *v = *v / (1.0 + (-1.702 * *v).exp());
            }
            let fd = e.htod(&fh)?;
            let f2 = self.linear_bias(e, &fd, &blk.cproj, n)?;
            let mut mh = e.dtoh(&f2)?;
            for t in 0..n {
                for c in 0..SV_HIDDEN {
                    mh[t * SV_HIDDEN + c] *= blk.ls2[c];
                }
            }
            let md = e.htod(&mh)?;
            let mut xn = e.zeros(n * SV_HIDDEN)?;
            e.add(&xr, &md, &mut xn, n * SV_HIDDEN)?;
            x = xn;
            if dbg.is_some() && ib == 0 {
                dump("blk0", &e.dtoh(&x)?);
            }
        }
        // NO ln_post (vision_config.use_ln_post = false)
        if dbg.is_some() {
            dump("post_blocks", &e.dtoh(&x)?);
        }
        // head: [n, 1536] as [1536, g, g] -> downsampler1 -> downsampler2 (im2col +
        // GEMM each) -> [og*og, 6144] rows -> vit_large_projector
        let xh = e.dtoh(&x)?;
        let (col1, g1) = Self::im2col(&xh, g, SV_HIDDEN);
        let c1 = e.htod(&col1)?;
        let mut y1 = e.linear(
            &c1,
            &self.down1.w,
            g1 * g1,
            self.down1.c_in * 9,
            self.down1.c_out,
        )?;
        for r in 0..g1 * g1 {
            e.add_row_inplace(
                &mut y1,
                &self.down1.b,
                self.down1.c_out,
                r * self.down1.c_out,
            )?;
        }
        let y1h = e.dtoh(&y1)?;
        let (col2, g2) = Self::im2col(&y1h, g1, self.down2.c_in);
        let c2 = e.htod(&col2)?;
        let mut y2 = e.linear(
            &c2,
            &self.down2.w,
            g2 * g2,
            self.down2.c_in * 9,
            self.down2.c_out,
        )?;
        for r in 0..g2 * g2 {
            e.add_row_inplace(
                &mut y2,
                &self.down2.b,
                self.down2.c_out,
                r * self.down2.c_out,
            )?;
        }
        if dbg.is_some() {
            dump("downsampled", &e.dtoh(&y2)?);
        }
        let out = self.linear_bias(e, &y2, &self.proj, g2 * g2)?;
        if dbg.is_some() {
            dump("projected", &e.dtoh(&out)?);
        }
        Ok(out)
    }

    /// Forward one whole unit (tiles first, then the main view — the vendor merge
    /// order) into `rows` at `row_off * out_width`. Returns rows written.
    pub fn forward_unit(
        &self,
        e: &Engine,
        unit: &StepVisionUnit,
        rows: &mut CudaSlice<f32>,
        row_off: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let w = self.out_width();
        let mut off = row_off;
        for tile in &unit.tiles {
            let emb = self.forward(e, tile, SV_GRID_TILE)?;
            e.dtod_copy_into(&emb, rows, off * w)?;
            off += SV_TILE_ROWS;
        }
        let emb = self.forward(e, &unit.main, SV_GRID_MAIN)?;
        e.dtod_copy_into(&emb, rows, off * w)?;
        off += SV_MAIN_ROWS;
        Ok(off - row_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiling plan law, pinned to processing_step3.py arithmetic by hand-checked
    /// cells (each row worked from the vendor code at the pinned rev).
    #[test]
    fn tiling_plan_cells() {
        // small square-ish: no tiles (window 0), 169 + 2 prompt tokens
        let p = plan_for_dims(600, 400);
        assert_eq!((p.n_tiles, p.n_prompt_tokens()), (0, 171));
        let p = plan_for_dims(728, 728);
        assert_eq!(p.n_tiles, 0);
        // small extreme-aspect (long <= 728, ratio > 1.5): window = short
        // 700x300: snap 700 -> 2.33 ratio -> decimal 0.33 > 0.2 -> 3 windows = 900;
        // 300 -> 1 window; tiles 3x1, newline after id 2 popped (trailing) -> 0 newlines
        let p = plan_for_dims(700, 300);
        assert_eq!(p.n_tiles, 3);
        assert_eq!(p.newline_mask, vec![false, false, false]);
        // 1600x900: window 504 (long > 728, ratio 1.78 <= 4); 1600/504 = 3.17 ->
        // 3 cols (0.17 <= 0.2); 900/504 = 1.79 -> 2 rows (0.79 > 0.2);
        // 6 tiles, newlines after ids 2 and 5, 5 popped -> mask true only at 2
        let p = plan_for_dims(1600, 900);
        assert_eq!(p.n_tiles, 6);
        assert_eq!(
            p.newline_mask,
            vec![false, false, true, false, false, false]
        );
        // prompt tokens: 6*(81+2) + 1 newline + 169 + 2 = 670
        assert_eq!(p.n_prompt_tokens(), 670);
        // tiny extreme-aspect pads to square then window 0
        let p = plan_for_dims(200, 20);
        assert_eq!(p.n_tiles, 0);
        // huge: capped to 3024 first; 4000x1000 -> 3024x756, ratio 4.0 (NOT > 4) ->
        // window 504; 3024/504 = 6 cols; 756/504 = 1.5 -> decimal 0.5 > 0.2 -> 2 rows
        let p = plan_for_dims(4000, 1000);
        assert_eq!(p.n_tiles, 12);
        assert_eq!(p.n_rows(), 12 * 81 + 169);
    }

    /// im2col output geometry: 52 -> 26 -> 13 and 36 -> 18 -> 9 (the 169/81 law).
    #[test]
    fn downsampler_geometry() {
        let x = vec![0f32; 52 * 52 * 4];
        let (_, og) = StepVisionTower::im2col(&x, 52, 4);
        assert_eq!(og, 26);
        let x = vec![0f32; 26 * 26 * 4];
        let (_, og) = StepVisionTower::im2col(&x, 26, 4);
        assert_eq!(og, 13);
        let x = vec![0f32; 36 * 36 * 4];
        let (_, og) = StepVisionTower::im2col(&x, 36, 4);
        assert_eq!(og, 18);
        let x = vec![0f32; 18 * 18 * 4];
        let (_, og) = StepVisionTower::im2col(&x, 18, 4);
        assert_eq!(og, 9);
    }

    /// im2col values: a 3x3 map with c_in 1, identity-checkable by hand.
    #[test]
    fn im2col_values() {
        // map [1,3,3] = [[1,2,3],[4,5,6],[7,8,9]]; og = 2; output position (0,0)
        // covers input rows -1..2 x -1..2 (zero pad): window [[0,0,0],[0,1,2],[0,4,5]]
        let x: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let (col, og) = StepVisionTower::im2col(&x, 3, 1);
        assert_eq!(og, 2);
        assert_eq!(&col[0..9], &[0., 0., 0., 0., 1., 2., 0., 4., 5.]);
        // position (1,1): rows 1..4 x 1..4 -> [[5,6,0],[8,9,0],[0,0,0]]
        assert_eq!(&col[27..36], &[5., 6., 0., 8., 9., 0., 0., 0., 0.]);
    }

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 253) as u8])
        });
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// The decoded prep must land on the header plan exactly (pad runs render from the
    /// plan; decode_pending_vision refuses on drift, so drift here is a shipped bug).
    #[test]
    fn prep_matches_plan() {
        for (w, h) in [(64u32, 64u32), (1600, 900), (700, 300), (900, 3000)] {
            let bytes = png_bytes(w, h);
            let plan = step_plan_image(&bytes).unwrap();
            let prep = step_prep_image(&bytes).unwrap();
            assert_eq!(prep.tiles.len(), plan.n_tiles, "{w}x{h} tile count");
            assert_eq!(prep.newline_mask, plan.newline_mask, "{w}x{h} newline mask");
            assert_eq!(prep.main.len(), SV_GRID_MAIN * SV_GRID_MAIN * SV_PATCH_IN);
            for t in &prep.tiles {
                assert_eq!(t.len(), SV_GRID_TILE * SV_GRID_TILE * SV_PATCH_IN);
            }
            assert_eq!(prep.n_rows(), plan.n_rows());
        }
    }

    /// Patch rows carry the CLIP normalization: a flat mid-gray image lands every
    /// channel plane on its exact normalized constant.
    #[test]
    fn patchify_normalization() {
        let img = image::RgbImage::from_pixel(
            SV_IMAGE_SIZE as u32,
            SV_IMAGE_SIZE as u32,
            image::Rgb([128, 128, 128]),
        );
        let rows = patchify(&img, SV_GRID_MAIN);
        let want: Vec<f32> = (0..3).map(|c| (128.0 / 255.0 - MEAN[c]) / STD[c]).collect();
        let r0 = &rows[..SV_PATCH_IN];
        for c in 0..3 {
            for i in 0..SV_PATCH * SV_PATCH {
                assert!((r0[c * SV_PATCH * SV_PATCH + i] - want[c]).abs() < 1e-6);
            }
        }
    }
}
