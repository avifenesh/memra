//! Vision tower for the gemma-4 family (lane/gemma-vision, 2026-08-16).
//!
//! Gemma-4 is its OWN semantic program — nothing here is inherited from the qwen3_5
//! tower by analogy; every law below was derived from the family's reference
//! implementation (llama.cpp `clip_graph_gemma4v`, the vendor-blessed consumer of the
//! official `google/gemma-4-*-qat-q4_0-gguf` mmproj packaging) and the mmproj tensor
//! census (`research/gemma-vision-20260816/REPORT.md` carries the receipts):
//!
//! - ViT: 27 blocks, hidden 1152, 16 heads (head_dim 72), ffn 4304 — but RMS norms
//!   (not LayerNorm), NO biases anywhere, per-head RMS q/k norms (72), a WEIGHTLESS
//!   RMS on V before attention, and sandwich post-norms (attn_post / ffn_post) applied
//!   BEFORE each residual add. Attention runs UNSCALED: kq_scale = 1.0, not 1/sqrt(d).
//! - FFN is GEGLU-quick: gelu_quick(gate(x)) * up(x), gelu_quick(x) = x*sigmoid(1.702x)
//!   (llama.cpp default when the mmproj carries neither use_gelu nor use_silu).
//! - Positions: FACTORED ADDITIVE tables (one x-table, one y-table, 10240 rows each,
//!   `v.position_embd.weight` logical [2, 10240, 1152]) added to the patch embeddings,
//!   PLUS a per-layer 2D rope on q/k: first 36 dims rotate by pos_x, last 36 by pos_y,
//!   neox pairing (d, d+18) inside each half, theta = 100.0 (hardcoded in the
//!   reference, not a metadata key).
//! - Input: pixels in [0,1], NO mean/std normalization (image_mean 0 / std 1 in the
//!   census); the graph applies 2x-1. Patch embed is a bias-less conv16, flattened
//!   here to a Linear over (c, ky, kx)-ordered 768-float patch rows.
//! - Head: 3x3 avg-pool over the patch grid (n_merge 3), scale by sqrt(1152),
//!   (x - std_bias) * std_scale, WEIGHTLESS RMS, then a single 1152 -> 5376 projection
//!   (`mm.input_projection`; this file ships no ClippableLinear clamp scalars).
//! - Preprocessing: native resolution — smart-resize to a 48-aligned grid with the
//!   token budget 40..280 (token = 48x48 px block), bilinear.
//!
//! Serving-law note (derived, NOT wired here): gemma-4 image spans decode with
//! NON-CAUSAL attention inside the LM (`mtmd_decode_use_non_causal` = true for this
//! family). memra's prime path is causal, so serving gemma vision through it would be
//! silently wrong — the serving gate must refuse until a masked-prefill arm exists.
//!
//! v1 posture matches the qwen tower: correctness-first — f32 GEMMs, `sdpa_naive`,
//! host-side permutes/rope/geglu; parity gate before any serving path.

use crate::Engine;
use cudarc::driver::CudaSlice;
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::{GgmlType, GgufFile};
use std::path::Path;

pub const GV_HIDDEN: usize = 1152;
pub const GV_HEADS: usize = 16;
pub const GV_HEAD_DIM: usize = GV_HIDDEN / GV_HEADS; // 72
pub const GV_INTER: usize = 4304;
pub const GV_DEPTH: usize = 27;
pub const GV_PATCH: usize = 16;
pub const GV_MERGE: usize = 3; // pooling kernel (n_merge)
pub const GV_POS_ROWS: usize = 10240; // per-axis position table rows
pub const GV_OUT: usize = 5376; // gemma-4-31B n_embd
pub const GV_PATCH_IN: usize = 3 * GV_PATCH * GV_PATCH; // 768
pub const GV_ALIGN: usize = GV_PATCH * GV_MERGE; // 48
/// Output-token budget (llama.cpp gemma4v: set_limit_image_tokens(40, 280)); a token
/// is one pooled 48x48-pixel block, so the pixel budget is tokens * 48*48.
pub const GV_MIN_TOKENS: usize = 40;
pub const GV_MAX_TOKENS: usize = 280;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 100.0;

/// Begin/end delimiters + the soft token that occupies image positions in the token
/// stream (gemma-4-31B vocab; the soft token embedding is REPLACED by tower rows).
pub const GV_TOK_BEGIN: u32 = 255999; // <|image>
pub const GV_TOK_SOFT: u32 = 258880; // <|image|>
pub const GV_TOK_END: u32 = 258882; // <image|>

struct GLin {
    w: CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
}

struct GBlock {
    ln1: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    attn_post: CudaSlice<f32>,
    ffn_post: CudaSlice<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wq: GLin,
    wk: GLin,
    wv: GLin,
    wo: GLin,
    gate: GLin,
    up: GLin,
    down: GLin,
}

pub struct GemmaVisionTower {
    patch_w: GLin, // conv16 flattened: 768 -> 1152, no bias
    /// Host copies of the factored position tables, [10240, 1152] each.
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    blocks: Vec<GBlock>,
    std_bias: Vec<f32>,
    std_scale: Vec<f32>,
    proj: GLin, // 1152 -> 5376
}

fn read_f32(g: &GgufFile, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let t = g
        .find(name)
        .ok_or_else(|| format!("gemma vision tensor missing: {name}"))?;
    let raw = g.tensor_data(t);
    match t.ggml_type {
        GgmlType::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        GgmlType::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        other => Err(format!("gemma vision tensor {name}: unsupported type {other:?}").into()),
    }
}

fn load_lin(
    e: &Engine,
    g: &GgufFile,
    name: &str,
    in_f: usize,
    out_f: usize,
) -> Result<GLin, Box<dyn std::error::Error>> {
    let w = read_f32(g, name)?;
    assert_eq!(w.len(), in_f * out_f, "{name} shape");
    Ok(GLin {
        w: e.htod(&w)?,
        in_f,
        out_f,
    })
}

/// The dyn-size resize law (llama.cpp `calc_size_preserved_ratio`, "smart_resize"):
/// round each side to the 48 grid, then rescale into the [min,max] pixel budget with
/// floor/ceil-to-48 — identical arithmetic, so the grid memra feeds the tower is the
/// grid the reference implementation would feed it.
pub fn gemma_target_size(w: u32, h: u32) -> (u32, u32) {
    let align = GV_ALIGN as f32;
    let min_px = (GV_MIN_TOKENS * GV_ALIGN * GV_ALIGN) as f32;
    let max_px = (GV_MAX_TOKENS * GV_ALIGN * GV_ALIGN) as f32;
    let round = |x: f32| ((x / align).round() * align).max(align) as u32;
    let ceilf = |x: f32| ((x / align).ceil() * align).max(align) as u32;
    let floorf = |x: f32| ((x / align).floor() * align).max(align) as u32;
    let (wf, hf) = (w as f32, h as f32);
    let mut w_bar = round(wf);
    let mut h_bar = round(hf);
    if (w_bar * h_bar) as f32 > max_px {
        let beta = (wf * hf / max_px).sqrt();
        w_bar = floorf(wf / beta);
        h_bar = floorf(hf / beta);
    } else if ((w_bar * h_bar) as f32) < min_px {
        let beta = (min_px / (wf * hf)).sqrt();
        w_bar = ceilf(wf * beta);
        h_bar = ceilf(hf * beta);
    }
    (w_bar, h_bar)
}

/// One preprocessed gemma image, server-carried from the HTTP layer to the GPU worker.
/// `patches` is the tower input (dropped after the tower forward); `n_soft` is the pooled
/// token count = (gw/3)(gh/3), i.e. the number of `<|image|>` soft tokens the prompt run
/// must carry for this unit.
pub struct GemmaVisionUnit {
    pub patches: Vec<f32>,
    pub gw: usize,
    pub gh: usize,
}

impl GemmaVisionUnit {
    pub fn n_soft(&self) -> usize {
        n_soft_for_grid(self.gw, self.gh)
    }
}

/// Soft tokens a `(gw, gh)` grid pools to — the planned twin of
/// `GemmaVisionUnit::n_soft`, usable from `gemma_plan_image` BEFORE any decode.
pub fn n_soft_for_grid(gw: usize, gh: usize) -> usize {
    (gw / GV_MERGE) * (gh / GV_MERGE)
}

/// Decode a base64 `data:` URI to raw image bytes (mirrors vision_pre::decode_data_uri;
/// http(s) fetch stays off for SSRF). Carries the same per-image raw cap
/// (`vision_pre::IMG_MAX_RAW_BYTES`), refused by encoded length before any allocation.
pub fn gemma_decode_data_uri(uri: &str) -> Result<Vec<u8>, String> {
    let comma = uri.find(',').ok_or("data URI has no comma")?;
    let meta = &uri[..comma];
    let body = &uri[comma + 1..];
    if !meta.contains(";base64") {
        return Err("only base64 data URIs are supported".into());
    }
    if let Some(err) = crate::vision_pre::data_uri_payload_over_cap(body) {
        return Err(err);
    }
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(body.as_bytes())
        .map_err(|e| format!("base64 decode: {e}"))
}

/// Prep a data-URI image into a GemmaVisionUnit (decode + smart-resize + patchify).
pub fn gemma_prep_data_uri(uri: &str) -> Result<GemmaVisionUnit, String> {
    let bytes = gemma_decode_data_uri(uri)?;
    let (patches, gw, gh) = gemma_prep_image(&bytes).map_err(|e| e.to_string())?;
    Ok(GemmaVisionUnit { patches, gw, gh })
}

/// PRE-DECODE admission (hermes decode-bomb finding, fixed 2026-08-23 — same law as
/// `vision_pre::plan_image_bytes`): header dims -> decode-budget check -> target grid.
/// Returns `(gw, gh)` so `n_soft` (thus the pad run and the request's token price) is
/// known before any canvas expands.
pub fn gemma_plan_image(bytes: &[u8]) -> Result<(usize, usize), String> {
    let (w, h) = crate::vision_pre::image_header_dims(bytes)?;
    if w.saturating_mul(h) > crate::vision_pre::IMG_MAX_DECODE_PIXELS {
        return Err(format!(
            "image {w}x{h} exceeds the decode budget ({} px) — refused before decode",
            crate::vision_pre::IMG_MAX_DECODE_PIXELS
        ));
    }
    let (tw, th) = gemma_target_size(w as u32, h as u32);
    Ok(((tw as usize) / GV_PATCH, (th as usize) / GV_PATCH))
}

/// Decode + resize + patchify one image: bytes -> (patch rows [n, 768] in the conv's
/// (c, ky, kx) flat order with the graph's 2x-1 scaling baked in, grid_w, grid_h).
/// Admission runs FIRST (`gemma_plan_image`, header-only); the decoder is capped to the
/// admitted dimensions so a header lying small cannot expand past them.
pub fn gemma_prep_image(
    bytes: &[u8],
) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    gemma_plan_image(bytes)?;
    let (hw, hh) = crate::vision_pre::image_header_dims(bytes)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(hw as u32);
    limits.max_image_height = Some(hh as u32);
    reader.limits(limits);
    let img = reader.decode()?.to_rgb8();
    let (w0, h0) = img.dimensions();
    let (tw, th) = gemma_target_size(w0, h0);
    // Bilinear per the reference (RESIZE_ALGO_BILINEAR); image's Triangle filter is the
    // bilinear kernel. Resize kernels are allowed to differ by a hair from llama.cpp's
    // own bilinear — the parity oracle feeds FIXED pixels so the tower is gated
    // independently of resampling.
    let resized = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);
    let (gw, gh) = ((tw as usize) / GV_PATCH, (th as usize) / GV_PATCH);
    let mut patches = vec![0f32; gw * gh * GV_PATCH_IN];
    for py in 0..gh {
        for px in 0..gw {
            let dst = &mut patches[(py * gw + px) * GV_PATCH_IN..(py * gw + px + 1) * GV_PATCH_IN];
            for c in 0..3 {
                for ky in 0..GV_PATCH {
                    for kx in 0..GV_PATCH {
                        let p = resized
                            .get_pixel((px * GV_PATCH + kx) as u32, (py * GV_PATCH + ky) as u32);
                        // [0,1] then the graph's 2x-1
                        dst[(c * GV_PATCH + ky) * GV_PATCH + kx] =
                            (p[c] as f32) / 255.0 * 2.0 - 1.0;
                    }
                }
            }
        }
    }
    Ok((patches, gw, gh))
}

impl GemmaVisionTower {
    /// Load the tower from a gemma-4 mmproj GGUF (`general.type = mmproj`,
    /// `clip.vision.projector_type = gemma4v`).
    pub fn load(e: &Engine, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let g = GgufFile::open(path)?;
        let proj_type = g
            .metadata
            .get("clip.vision.projector_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if proj_type != "gemma4v" {
            return Err(format!(
                "mmproj {path:?} projector_type {proj_type:?} is not gemma4v — this loader \
                 refuses other families by design (no generic support claims)",
                path = path
            )
            .into());
        }
        let patch_w = {
            // conv weight logical [1152, 3, 16, 16] row-major == Linear rows over the
            // (c, ky, kx) patch order gemma_prep_image emits.
            let w = read_f32(&g, "v.patch_embd.weight")?;
            assert_eq!(
                w.len(),
                GV_HIDDEN * GV_PATCH_IN,
                "v.patch_embd.weight shape"
            );
            GLin {
                w: e.htod(&w)?,
                in_f: GV_PATCH_IN,
                out_f: GV_HIDDEN,
            }
        };
        let pos = read_f32(&g, "v.position_embd.weight")?;
        assert_eq!(
            pos.len(),
            2 * GV_POS_ROWS * GV_HIDDEN,
            "position table shape"
        );
        let (pos_x, pos_y) = {
            let half = GV_POS_ROWS * GV_HIDDEN;
            (pos[..half].to_vec(), pos[half..].to_vec())
        };
        let mut blocks = Vec::with_capacity(GV_DEPTH);
        for il in 0..GV_DEPTH {
            let bp = format!("v.blk.{il}");
            blocks.push(GBlock {
                ln1: e.htod(&read_f32(&g, &format!("{bp}.ln1.weight"))?)?,
                ln2: e.htod(&read_f32(&g, &format!("{bp}.ln2.weight"))?)?,
                attn_post: e.htod(&read_f32(&g, &format!("{bp}.attn_post_norm.weight"))?)?,
                ffn_post: e.htod(&read_f32(&g, &format!("{bp}.ffn_post_norm.weight"))?)?,
                q_norm: read_f32(&g, &format!("{bp}.attn_q_norm.weight"))?,
                k_norm: read_f32(&g, &format!("{bp}.attn_k_norm.weight"))?,
                wq: load_lin(e, &g, &format!("{bp}.attn_q.weight"), GV_HIDDEN, GV_HIDDEN)?,
                wk: load_lin(e, &g, &format!("{bp}.attn_k.weight"), GV_HIDDEN, GV_HIDDEN)?,
                wv: load_lin(e, &g, &format!("{bp}.attn_v.weight"), GV_HIDDEN, GV_HIDDEN)?,
                wo: load_lin(
                    e,
                    &g,
                    &format!("{bp}.attn_out.weight"),
                    GV_HIDDEN,
                    GV_HIDDEN,
                )?,
                gate: load_lin(e, &g, &format!("{bp}.ffn_gate.weight"), GV_HIDDEN, GV_INTER)?,
                up: load_lin(e, &g, &format!("{bp}.ffn_up.weight"), GV_HIDDEN, GV_INTER)?,
                down: load_lin(e, &g, &format!("{bp}.ffn_down.weight"), GV_INTER, GV_HIDDEN)?,
            });
        }
        let std_bias = read_f32(&g, "v.std_bias")?;
        let std_scale = read_f32(&g, "v.std_scale")?;
        assert_eq!(std_bias.len(), GV_HIDDEN);
        assert_eq!(std_scale.len(), GV_HIDDEN);
        let proj = load_lin(e, &g, "mm.input_projection.weight", GV_HIDDEN, GV_OUT)?;
        eprintln!(
            "[gemma-vision] tower loaded from {} ({GV_DEPTH} blocks, f32-resident)",
            path.display()
        );
        Ok(Self {
            patch_w,
            pos_x,
            pos_y,
            blocks,
            std_bias,
            std_scale,
            proj,
        })
    }

    fn linear(
        &self,
        e: &Engine,
        x: &CudaSlice<f32>,
        l: &GLin,
        m: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        e.linear(x, &l.w, m, l.in_f, l.out_f)
    }

    /// Forward one image's patch rows -> [gh*gw/9, 5376] embeddings (device).
    pub fn forward(
        &self,
        e: &Engine,
        patches: &[f32],
        gw: usize,
        gh: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n = gw * gh;
        assert_eq!(patches.len(), n * GV_PATCH_IN, "patch buffer shape");
        assert_eq!(gw % GV_MERGE, 0, "grid width must be 3-aligned");
        assert_eq!(gh % GV_MERGE, 0, "grid height must be 3-aligned");
        if n > 12288 {
            return Err(format!(
                "gemma vision segment {n} patches exceeds the sdpa shared-memory ceiling (12288)"
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

        // patch embed + factored additive position tables (row-major grid: x = col, y = row)
        let xd = e.htod(patches)?;
        let embedded = self.linear(e, &xd, &self.patch_w, n)?;
        let mut pos = vec![0f32; n * GV_HIDDEN];
        for t in 0..n {
            let (py, px) = (t / gw, t % gw);
            let dst = &mut pos[t * GV_HIDDEN..(t + 1) * GV_HIDDEN];
            let tx = &self.pos_x[px * GV_HIDDEN..(px + 1) * GV_HIDDEN];
            let ty = &self.pos_y[py * GV_HIDDEN..(py + 1) * GV_HIDDEN];
            for c in 0..GV_HIDDEN {
                dst[c] = tx[c] + ty[c];
            }
        }
        let pos_d = e.htod(&pos)?;
        let mut x = e.zeros(n * GV_HIDDEN)?;
        e.add(&embedded, &pos_d, &mut x, n * GV_HIDDEN)?;
        if dbg.is_some() {
            dump("pre_blocks", &e.dtoh(&x)?);
        }

        // 2D rope tables: FIRST 36 dims rotate by pos_x, LAST 36 by pos_y (the reference
        // order — opposite of qwen3_5's y-first); neox pairing (d, d+18) inside each
        // half; inv_freq[i] = theta^(-2i/36), theta 100.
        let half = GV_HEAD_DIM / 2; // 36
        let quarter = half / 2; // 18
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| ROPE_THETA.powf(-2.0 * (i as f32) / half as f32))
            .collect();
        let mut cos_x = vec![0f32; n * quarter];
        let mut sin_x = vec![0f32; n * quarter];
        let mut cos_y = vec![0f32; n * quarter];
        let mut sin_y = vec![0f32; n * quarter];
        for t in 0..n {
            let (py, px) = (t / gw, t % gw);
            for i in 0..quarter {
                let ax = px as f32 * inv_freq[i];
                let ay = py as f32 * inv_freq[i];
                cos_x[t * quarter + i] = ax.cos();
                sin_x[t * quarter + i] = ax.sin();
                cos_y[t * quarter + i] = ay.cos();
                sin_y[t * quarter + i] = ay.sin();
            }
        }
        // per-head RMS over head_dim with weight (q/k) or weightless (v)
        let head_rms = |row: &mut [f32], w: Option<&[f32]>| {
            let mut ss = 0f32;
            for v in row.iter() {
                ss += v * v;
            }
            let inv = 1.0 / (ss / GV_HEAD_DIM as f32 + RMS_EPS).sqrt();
            for (d, v) in row.iter_mut().enumerate() {
                *v *= inv * w.map_or(1.0, |w| w[d]);
            }
        };

        for (ib, blk) in self.blocks.iter().enumerate() {
            // attn: rms(ln1) -> q/k/v -> per-head norms -> 2D rope -> sdpa(scale=1) ->
            //       o_proj -> rms(attn_post) -> +residual
            let mut h = e.zeros(n * GV_HIDDEN)?;
            e.rms_norm(&x, &blk.ln1, &mut h, GV_HIDDEN, n, RMS_EPS)?;
            let q = self.linear(e, &h, &blk.wq, n)?;
            let k = self.linear(e, &h, &blk.wk, n)?;
            let v = self.linear(e, &h, &blk.wv, n)?;
            let (mut qh, mut kh, mut vh) = (e.dtoh(&q)?, e.dtoh(&k)?, e.dtoh(&v)?);
            for t in 0..n {
                for hd in 0..GV_HEADS {
                    let o = t * GV_HIDDEN + hd * GV_HEAD_DIM;
                    head_rms(&mut qh[o..o + GV_HEAD_DIM], Some(&blk.q_norm));
                    head_rms(&mut kh[o..o + GV_HEAD_DIM], Some(&blk.k_norm));
                    head_rms(&mut vh[o..o + GV_HEAD_DIM], None);
                    // rope: first half by x, second half by y, pairs (d, d+18) per half
                    for (base, cos, sin) in [(0, &cos_x, &sin_x), (half, &cos_y, &sin_y)] {
                        for i in 0..quarter {
                            let (c, s) = (cos[t * quarter + i], sin[t * quarter + i]);
                            for buf in [&mut qh, &mut kh] {
                                let a = buf[o + base + i];
                                let b = buf[o + base + i + quarter];
                                buf[o + base + i] = a * c - b * s;
                                buf[o + base + i + quarter] = b * c + a * s;
                            }
                        }
                    }
                }
            }
            let (qd, kd, vd) = (e.htod(&qh)?, e.htod(&kh)?, e.htod(&vh)?);
            let mut od = e.zeros(n * GV_HIDDEN)?;
            // UNSCALED attention (kq_scale = 1.0 in the reference graph), full/non-causal.
            e.sdpa_naive(
                &qd,
                &kd,
                &vd,
                &mut od,
                GV_HEAD_DIM,
                GV_HEADS,
                GV_HEADS,
                n,
                n,
                1.0,
                false,
            )?;
            let attn = self.linear(e, &od, &blk.wo, n)?;
            let mut post = e.zeros(n * GV_HIDDEN)?;
            e.rms_norm(&attn, &blk.attn_post, &mut post, GV_HIDDEN, n, RMS_EPS)?;
            let mut xr = e.zeros(n * GV_HIDDEN)?;
            e.add(&x, &post, &mut xr, n * GV_HIDDEN)?;

            // ffn: rms(ln2) -> gelu_quick(gate) * up -> down -> rms(ffn_post) -> +residual
            let mut h2 = e.zeros(n * GV_HIDDEN)?;
            e.rms_norm(&xr, &blk.ln2, &mut h2, GV_HIDDEN, n, RMS_EPS)?;
            let gate = self.linear(e, &h2, &blk.gate, n)?;
            let up = self.linear(e, &h2, &blk.up, n)?;
            let (gh_, uh) = (e.dtoh(&gate)?, e.dtoh(&up)?);
            let mut act = vec![0f32; n * GV_INTER];
            for i in 0..n * GV_INTER {
                let g = gh_[i];
                // gelu_quick(x) = x * sigmoid(1.702 x) — NOT the tanh approximation.
                act[i] = g / (1.0 + (-1.702 * g).exp()) * uh[i];
            }
            let ad = e.htod(&act)?;
            let down = self.linear(e, &ad, &blk.down, n)?;
            let mut fpost = e.zeros(n * GV_HIDDEN)?;
            e.rms_norm(&down, &blk.ffn_post, &mut fpost, GV_HIDDEN, n, RMS_EPS)?;
            let mut xn = e.zeros(n * GV_HIDDEN)?;
            e.add(&xr, &fpost, &mut xn, n * GV_HIDDEN)?;
            x = xn;
            if dbg.is_some() && ib == 0 {
                dump("blk0", &e.dtoh(&x)?);
            }
        }
        if dbg.is_some() {
            dump("post_blocks", &e.dtoh(&x)?);
        }

        // head: 3x3 avg-pool over the grid -> *sqrt(1152) -> (x - std_bias)*std_scale
        //       -> weightless RMS -> project 1152 -> 5376
        let xh = e.dtoh(&x)?;
        let (mw, mh) = (gw / GV_MERGE, gh / GV_MERGE);
        let nm = mw * mh;
        let scale = (GV_HIDDEN as f32).sqrt();
        let mut pooled = vec![0f32; nm * GV_HIDDEN];
        for my in 0..mh {
            for mx in 0..mw {
                let dst = &mut pooled[(my * mw + mx) * GV_HIDDEN..(my * mw + mx + 1) * GV_HIDDEN];
                for sy in 0..GV_MERGE {
                    for sx in 0..GV_MERGE {
                        let t = (my * GV_MERGE + sy) * gw + (mx * GV_MERGE + sx);
                        for c in 0..GV_HIDDEN {
                            dst[c] += xh[t * GV_HIDDEN + c];
                        }
                    }
                }
                for (c, d) in dst.iter_mut().enumerate() {
                    *d = (*d / (GV_MERGE * GV_MERGE) as f32 * scale - self.std_bias[c])
                        * self.std_scale[c];
                }
            }
        }
        // weightless RMS then projection
        for row in pooled.chunks_exact_mut(GV_HIDDEN) {
            let mut ss = 0f32;
            for v in row.iter() {
                ss += v * v;
            }
            let inv = 1.0 / (ss / GV_HIDDEN as f32 + RMS_EPS).sqrt();
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        if dbg.is_some() {
            dump("pre_proj", &pooled);
        }
        let pd = e.htod(&pooled)?;
        let out = self.linear(e, &pd, &self.proj, nm)?;
        if dbg.is_some() {
            dump("projected", &e.dtoh(&out)?);
        }
        Ok(out)
    }
}
