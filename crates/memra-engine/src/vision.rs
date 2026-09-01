//! Vision tower for Qwen3.8-27B multimodal input (lane/vision, 2026-08-15).
//!
//! The qwen3_5_vision ViT (depth 27, hidden 1152, heads 16, gelu_pytorch_tanh, patch 16,
//! spatial_merge 2, temporal_patch 2, LEARNED pos embeddings on a 48x48 grid) lives in the
//! official checkpoint's `outside.safetensors` (the unquantized shard) — the quantized
//! trunks (ct-NVFP4 etc.) strip it. `MEMRA_VISION_DIR` points at any directory carrying
//! that shard; the tower output is plain [n_tokens, out_width] embeddings (out_width =
//! merger fc2 out_features, derived from the shard — 5120 on q38-27B, 2048 on
//! Ornith-1.5-35B-A3B), so vision requests serve on any trunk whose n_embd matches the
//! shard's. Text side uses standard sequential rope (rope_scaling is null on these
//! models — no M-RoPE), so spliced image tokens take ordinary positions.
//!
//! v1 posture: correctness-first — cuBLASLt f32 GEMMs (`Engine::linear` + bias epilogue),
//! `sdpa_naive(causal=false)` for the bidirectional attention, host-side permutes between
//! stages (the tower is a small fraction of a vision request; optimize later). Parity gate:
//! merger-output cosine vs the HF reference per VISION-LANE.md.

use crate::Engine;
use cudarc::driver::CudaSlice;
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::safetensors::StShard;
use std::path::Path;

pub const V_HIDDEN: usize = 1152;
pub const V_HEADS: usize = 16;
pub const V_HEAD_DIM: usize = V_HIDDEN / V_HEADS; // 72
pub const V_INTER: usize = 4304;
pub const V_DEPTH: usize = 27;
pub const V_PATCH: usize = 16;
pub const V_MERGE: usize = 2;
pub const V_TEMPORAL: usize = 2;
pub const V_POS_GRID: usize = 48; // 2304 learned positions = 48x48
pub const V_PATCH_IN: usize = 3 * V_TEMPORAL * V_PATCH * V_PATCH; // 1536
pub const V_MERGED_IN: usize = V_HIDDEN * V_MERGE * V_MERGE; // 4608
const LN_EPS: f32 = 1e-6;

/// Mixed-embedding prime overlay: image embeddings that replace `<|image_pad|>` token
/// embeddings at prompt-relative positions during `prime_cache_overlaid`. `rows` holds all
/// images' merger outputs concatenated ([total_rows, n_embd]); each span is
/// `(prompt_pos, row_off, n_rows)` — rows `[row_off, row_off+n_rows)` land at prompt
/// positions `[prompt_pos, prompt_pos+n_rows)`. Spans must not overlap.
pub struct EmbedOverlay {
    pub rows: CudaSlice<f32>,
    pub spans: Vec<(usize, usize, usize)>,
}

impl EmbedOverlay {
    /// Sub-window for a prime call covering prompt-relative `[off, off+len)`: spans clipped
    /// and rebased so the callee sees call-relative positions (the serve prefill tick primes
    /// a prompt across multiple `prime_cache_overlaid` calls). `rows` is an Arc clone, not a
    /// copy. None = no image rows in this window (caller may prime plain).
    pub fn window(&self, off: usize, len: usize) -> Option<EmbedOverlay> {
        let spans: Vec<(usize, usize, usize)> = self
            .spans
            .iter()
            .filter_map(|&(pos, row_off, n_rows)| {
                let lo = pos.max(off);
                let hi = (pos + n_rows).min(off + len);
                (lo < hi).then(|| (lo - off, row_off + (lo - pos), hi - lo))
            })
            .collect();
        (!spans.is_empty()).then(|| EmbedOverlay {
            rows: self.rows.clone(),
            spans,
        })
    }

    /// The mixed-embedding splice itself: image rows overwrite placeholder-token embeddings
    /// inside a prime call's prompt-relative window `[chunk_off, chunk_off + t)`, BEFORE any
    /// downstream transform (stream expansion, trunk walk). ONE implementation shared by the
    /// single-engine hyper walk and the ppN stage-0 intake (lane/glm5-vision-default-on) so
    /// the splice point cannot drift between arms. `embedded` is the `[t, n_embd]` token
    /// embedding buffer of this call; `rows` must live on `e`'s device.
    pub fn splice_into(
        &self,
        e: &Engine,
        embedded: &mut CudaSlice<f32>,
        chunk_off: usize,
        t: usize,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for &(pos, row_off, n_rows) in &self.spans {
            let lo = pos.max(chunk_off);
            let hi = (pos + n_rows).min(chunk_off + t);
            if lo < hi {
                let src_row = row_off + (lo - pos);
                let view = self
                    .rows
                    .slice(src_row * n_embd..(src_row + (hi - lo)) * n_embd);
                e.copy_view_into(
                    embedded,
                    (lo - chunk_off) * n_embd,
                    &view,
                    (hi - lo) * n_embd,
                )?;
            }
        }
        Ok(())
    }
}

struct Lin {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    in_f: usize,
    out_f: usize,
}

struct VisBlock {
    norm1_w: CudaSlice<f32>,
    norm1_b: CudaSlice<f32>,
    norm2_w: CudaSlice<f32>,
    norm2_b: CudaSlice<f32>,
    qkv: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

pub struct VisionTower {
    patch: Lin,
    /// Host copy of the learned pos table [2304, 1152] — bilinear-interpolated per grid.
    pos: Vec<f32>,
    blocks: Vec<VisBlock>,
    merger_norm_w: CudaSlice<f32>,
    merger_norm_b: CudaSlice<f32>,
    merger_fc1: Lin,
    merger_fc2: Lin,
}

fn read_f32(sh: &StShard, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (info, raw) = sh
        .raw(name)
        .ok_or_else(|| format!("vision tensor missing: {name}"))?;
    match info.dtype.as_str() {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        other => Err(format!("vision tensor {name}: unsupported dtype {other}").into()),
    }
}

fn load_lin(
    e: &Engine,
    sh: &StShard,
    stem: &str,
    in_f: usize,
    out_f: usize,
) -> Result<Lin, Box<dyn std::error::Error>> {
    let w = read_f32(sh, &format!("{stem}.weight"))?;
    let b = read_f32(sh, &format!("{stem}.bias"))?;
    assert_eq!(w.len(), in_f * out_f, "{stem}.weight shape");
    assert_eq!(b.len(), out_f, "{stem}.bias shape");
    Ok(Lin {
        w: e.htod(&w)?,
        b: e.htod(&b)?,
        in_f,
        out_f,
    })
}

impl VisionTower {
    /// Load the tower from a directory containing `outside.safetensors`.
    pub fn load(e: &Engine, dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let sh = StShard::open(dir.join("outside.safetensors"))?;
        let p = "model.visual";
        let patch = {
            // conv [1152, 3, 2, 16, 16] flattens to Linear 1536 -> 1152 (HF patchify order:
            // channel-major within the (c, t, h, w) patch — the preprocessor emits the
            // matching flat order).
            let w = read_f32(&sh, &format!("{p}.patch_embed.proj.weight"))?;
            let b = read_f32(&sh, &format!("{p}.patch_embed.proj.bias"))?;
            assert_eq!(w.len(), V_HIDDEN * V_PATCH_IN);
            Lin {
                w: e.htod(&w)?,
                b: e.htod(&b)?,
                in_f: V_PATCH_IN,
                out_f: V_HIDDEN,
            }
        };
        let pos = read_f32(&sh, &format!("{p}.pos_embed.weight"))?;
        assert_eq!(pos.len(), V_POS_GRID * V_POS_GRID * V_HIDDEN);
        let mut blocks = Vec::with_capacity(V_DEPTH);
        for il in 0..V_DEPTH {
            let bp = format!("{p}.blocks.{il}");
            blocks.push(VisBlock {
                norm1_w: e.htod(&read_f32(&sh, &format!("{bp}.norm1.weight"))?)?,
                norm1_b: e.htod(&read_f32(&sh, &format!("{bp}.norm1.bias"))?)?,
                norm2_w: e.htod(&read_f32(&sh, &format!("{bp}.norm2.weight"))?)?,
                norm2_b: e.htod(&read_f32(&sh, &format!("{bp}.norm2.bias"))?)?,
                qkv: load_lin(e, &sh, &format!("{bp}.attn.qkv"), V_HIDDEN, 3 * V_HIDDEN)?,
                proj: load_lin(e, &sh, &format!("{bp}.attn.proj"), V_HIDDEN, V_HIDDEN)?,
                fc1: load_lin(e, &sh, &format!("{bp}.mlp.linear_fc1"), V_HIDDEN, V_INTER)?,
                fc2: load_lin(e, &sh, &format!("{bp}.mlp.linear_fc2"), V_INTER, V_HIDDEN)?,
            });
        }
        let merger_norm_w = e.htod(&read_f32(&sh, &format!("{p}.merger.norm.weight"))?)?;
        let merger_norm_b = e.htod(&read_f32(&sh, &format!("{p}.merger.norm.bias"))?)?;
        let merger_fc1 = load_lin(
            e,
            &sh,
            &format!("{p}.merger.linear_fc1"),
            V_MERGED_IN,
            V_MERGED_IN,
        )?;
        let merger_fc2 = {
            // Output width is the TRUNK's embedding width (5120 on q38, 2048 on ornith15) —
            // derived from the shard's merger shape, never assumed. Admission compares the
            // serving trunk's n_embd against `out_width()`.
            let w = read_f32(&sh, &format!("{p}.merger.linear_fc2.weight"))?;
            let b = read_f32(&sh, &format!("{p}.merger.linear_fc2.bias"))?;
            assert_eq!(w.len() % V_MERGED_IN, 0, "merger.linear_fc2.weight shape");
            let out_f = w.len() / V_MERGED_IN;
            assert_eq!(b.len(), out_f, "merger.linear_fc2.bias shape");
            Lin {
                w: e.htod(&w)?,
                b: e.htod(&b)?,
                in_f: V_MERGED_IN,
                out_f,
            }
        };
        eprintln!(
            "[vision] tower loaded from {} ({} blocks, out_width {}, f32-resident)",
            dir.display(),
            V_DEPTH,
            merger_fc2.out_f
        );
        Ok(Self {
            patch,
            pos,
            blocks,
            merger_norm_w,
            merger_norm_b,
            merger_fc1,
            merger_fc2,
        })
    }

    /// Embedding width this tower emits per merged token (the merger fc2 out_features,
    /// i.e. the trunk n_embd of the checkpoint the shard came from).
    pub fn out_width(&self) -> usize {
        self.merger_fc2.out_f
    }

    fn linear_bias(
        &self,
        e: &Engine,
        x: &CudaSlice<f32>,
        l: &Lin,
        m: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = e.linear(x, &l.w, m, l.in_f, l.out_f)?;
        // Row-broadcast bias via add_row_inplace (per-row launch; the tower is small and
        // v1 is correctness-first — the cuBLASLt bias epilogue is the later optimization).
        for r in 0..m {
            e.add_row_inplace(&mut y, &l.b, l.out_f, r * l.out_f)?;
        }
        Ok(y)
    }

    /// Bilinear-interpolate the 48x48 learned pos table to [gh, gw] and return host
    /// [gh*gw, 1152] (added to the patch embeddings).
    fn pos_for_grid(&self, gh: usize, gw: usize) -> Vec<f32> {
        let g = V_POS_GRID as f32;
        let mut out = vec![0f32; gh * gw * V_HIDDEN];
        for y in 0..gh {
            for x in 0..gw {
                // HF fast_pos_embed_interpolate: linspace(0, 47, g) == align_corners=TRUE
                let sy = if gh > 1 {
                    y as f32 * (g - 1.0) / (gh as f32 - 1.0)
                } else {
                    0.0
                };
                let sx = if gw > 1 {
                    x as f32 * (g - 1.0) / (gw as f32 - 1.0)
                } else {
                    0.0
                };
                let (y0, x0) = (sy.floor() as usize, sx.floor() as usize);
                let (y1, x1) = ((y0 + 1).min(V_POS_GRID - 1), (x0 + 1).min(V_POS_GRID - 1));
                let (fy, fx) = (sy - y0 as f32, sx - x0 as f32);
                let dst = &mut out[(y * gw + x) * V_HIDDEN..(y * gw + x + 1) * V_HIDDEN];
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for c in 0..V_HIDDEN {
                    let p00 = self.pos[(y0 * V_POS_GRID + x0) * V_HIDDEN + c];
                    let p01 = self.pos[(y0 * V_POS_GRID + x1) * V_HIDDEN + c];
                    let p10 = self.pos[(y1 * V_POS_GRID + x0) * V_HIDDEN + c];
                    let p11 = self.pos[(y1 * V_POS_GRID + x1) * V_HIDDEN + c];
                    dst[c] = p00 * (1.0 - fy) * (1.0 - fx)
                        + p01 * (1.0 - fy) * fx
                        + p10 * fy * (1.0 - fx)
                        + p11 * fy * fx;
                }
            }
        }
        out
    }

    /// Forward one image's patches -> [gh*gw/4, 5120] merged embeddings (device).
    /// `patches` is host [gh*gw, 1536] in the preprocessor's (c, t, ph, pw) flat order.
    pub fn forward(
        &self,
        e: &Engine,
        patches: &[f32],
        gh: usize,
        gw: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.forward_seq(e, patches, 1, gh, gw)
    }

    /// Forward `groups` temporal groups of one video (or a single image at groups=1):
    /// host patches [groups*gh*gw, 1536], frame-major -> [groups*gh*gw/4, 5120] merged
    /// embeddings, frame-major. HF cu_seqlens law (vision_utils.get_vision_cu_seqlens,
    /// merge_temporal=False — the qwen2_vl/qwen3_vl/qwen3_5 convention): EACH temporal
    /// group is its own attention segment, and pos table / rope / merger are all
    /// frame-local too — so a video is exactly its groups run through the single-image
    /// forward, concatenated. (Joint clip attention is the kimi_k25 convention only;
    /// parity receipt: joint span scored mean_cos 0.92 vs the HF oracle, per-group 1.0.)
    pub fn forward_seq(
        &self,
        e: &Engine,
        patches: &[f32],
        groups: usize,
        gh: usize,
        gw: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n = groups * gh * gw;
        assert_eq!(patches.len(), n * V_PATCH_IN, "patch buffer shape");
        if groups > 1 {
            let frame = gh * gw;
            let out_per = frame / (V_MERGE * V_MERGE) * self.merger_fc2.out_f;
            let mut out = e.uninit(groups * out_per)?;
            for g in 0..groups {
                let emb = self.forward_one(
                    e,
                    &patches[g * frame * V_PATCH_IN..(g + 1) * frame * V_PATCH_IN],
                    gh,
                    gw,
                )?;
                e.dtod_copy_into(&emb, &mut out, g * out_per)?;
            }
            return Ok(out);
        }
        self.forward_one(e, patches, gh, gw)
    }

    /// One attention segment (a single image, or one temporal group of a video).
    fn forward_one(
        &self,
        e: &Engine,
        patches: &[f32],
        gh: usize,
        gw: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let groups = 1usize;
        let n = gh * gw;
        assert_eq!(patches.len(), n * V_PATCH_IN, "patch buffer shape");
        if n > 12288 {
            return Err(format!(
                "vision segment {n} patches exceeds the sdpa shared-memory ceiling (12288); \
                 lower the pixel budget"
            )
            .into());
        }
        let xd = e.htod(patches)?;
        let mut x = self.linear_bias(e, &xd, &self.patch, n)?;
        // + interpolated pos embed (same table every temporal group)
        let pos_one = self.pos_for_grid(gh, gw);
        let mut pos = Vec::with_capacity(n * V_HIDDEN);
        for _ in 0..groups {
            pos.extend_from_slice(&pos_one);
        }
        let pos_d = e.htod(&pos)?;
        let mut x2 = e.zeros(n * V_HIDDEN)?;
        e.add(&x, &pos_d, &mut x2, n * V_HIDDEN)?;
        x = x2;
        // dev-only stage dumps for the HF parity bisect (row-major grid order, f32 LE)
        let dbg = std::env::var("MEMRA_VISION_DEBUG").ok();
        let dump = |tag: &str, buf: &[f32]| {
            if let Some(dir) = dbg.as_deref() {
                let raw: Vec<u8> = buf.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/rust_{tag}.bin"), raw);
            }
        };
        if dbg.is_some() {
            dump("pre_blocks", &e.dtoh(&x)?);
        }
        let scale = 1.0 / (V_HEAD_DIM as f32).sqrt();
        // 2D vision rope (Qwen3_5VisionRotaryEmbedding, theta 10000): per token (y, x) the
        // head_dim/2 = 36 rotation angles are [y * inv_freq[0..18], x * inv_freq[0..18]],
        // GPT-NeoX pairing (d, d+36). Same table for every block/head — precompute cos/sin.
        let half = V_HEAD_DIM / 2; // 36
        let quarter = half / 2; // 18
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| 10000f32.powf(-(i as f32) / quarter as f32))
            .collect();
        let mut rope_cos = vec![0f32; n * half];
        let mut rope_sin = vec![0f32; n * half];
        for t in 0..n {
            let f = t % (gh * gw); // frame-local index (rope has no temporal axis here)
            let (y, x) = (f / gw, f % gw);
            for d in 0..half {
                let f = if d < quarter {
                    y as f32 * inv_freq[d]
                } else {
                    x as f32 * inv_freq[d - quarter]
                };
                rope_cos[t * half + d] = f.cos();
                rope_sin[t * half + d] = f.sin();
            }
        }
        for (ib, blk) in self.blocks.iter().enumerate() {
            // attn: ln1 -> qkv -> sdpa(causal=false) -> proj -> +res
            let mut h = e.zeros(n * V_HIDDEN)?;
            e.layer_norm_bias(&x, &blk.norm1_w, &blk.norm1_b, &mut h, V_HIDDEN, n, LN_EPS)?;
            let qkv = self.linear_bias(e, &h, &blk.qkv, n)?;
            // sdpa_naive consumes token-major [T, n_head, head_dim] — exactly the qkv GEMM
            // row layout, so q/k/v are column splits of each row (no permute). Host pass
            // applies the vision rope to q/k on the way (v untouched).
            let qkv_h = e.dtoh(&qkv)?;
            let mut qh = vec![0f32; n * V_HIDDEN];
            let mut kh = vec![0f32; n * V_HIDDEN];
            let mut vh = vec![0f32; n * V_HIDDEN];
            for t in 0..n {
                let row = &qkv_h[t * 3 * V_HIDDEN..(t + 1) * 3 * V_HIDDEN];
                let dst = t * V_HIDDEN;
                vh[dst..dst + V_HIDDEN].copy_from_slice(&row[2 * V_HIDDEN..3 * V_HIDDEN]);
                for hd in 0..V_HEADS {
                    let o = hd * V_HEAD_DIM;
                    // rotate-half pairs (d, d+36), angles shared across heads
                    for d in 0..half {
                        let (c, sn) = (rope_cos[t * half + d], rope_sin[t * half + d]);
                        let (qa, qb) = (row[o + d], row[o + d + half]);
                        qh[dst + o + d] = qa * c - qb * sn;
                        qh[dst + o + d + half] = qb * c + qa * sn;
                        let (ka, kb) = (row[V_HIDDEN + o + d], row[V_HIDDEN + o + d + half]);
                        kh[dst + o + d] = ka * c - kb * sn;
                        kh[dst + o + d + half] = kb * c + ka * sn;
                    }
                }
            }
            let (qd, kd, vd) = (e.htod(&qh)?, e.htod(&kh)?, e.htod(&vh)?);
            let mut od = e.zeros(n * V_HIDDEN)?;
            e.sdpa_naive(
                &qd, &kd, &vd, &mut od, V_HEAD_DIM, V_HEADS, V_HEADS, n, n, scale, false,
            )?;
            let attn = self.linear_bias(e, &od, &blk.proj, n)?;
            let mut xr = e.zeros(n * V_HIDDEN)?;
            e.add(&x, &attn, &mut xr, n * V_HIDDEN)?;
            // mlp: ln2 -> fc1 -> gelu_tanh -> fc2 -> +res
            let mut h2 = e.zeros(n * V_HIDDEN)?;
            e.layer_norm_bias(
                &xr,
                &blk.norm2_w,
                &blk.norm2_b,
                &mut h2,
                V_HIDDEN,
                n,
                LN_EPS,
            )?;
            let f1 = self.linear_bias(e, &h2, &blk.fc1, n)?;
            let mut g = e.zeros(n * V_INTER)?;
            e.gelu_tanh(&f1, &mut g, n * V_INTER)?;
            let f2 = self.linear_bias(e, &g, &blk.fc2, n)?;
            let mut xn = e.zeros(n * V_HIDDEN)?;
            e.add(&xr, &f2, &mut xn, n * V_HIDDEN)?;
            x = xn;
            if dbg.is_some() && ib == 0 {
                dump("blk0", &e.dtoh(&x)?);
            }
        }
        if dbg.is_some() {
            dump("post_blocks", &e.dtoh(&x)?);
        }
        // merger: LN over [n, 1152], then 2x2 spatial concat -> [n/4, 4608] -> fc1 -> gelu -> fc2
        let mut ln = e.zeros(n * V_HIDDEN)?;
        e.layer_norm_bias(
            &x,
            &self.merger_norm_w,
            &self.merger_norm_b,
            &mut ln,
            V_HIDDEN,
            n,
            LN_EPS,
        )?;
        let lh = e.dtoh(&ln)?;
        let (mh, mw) = (gh / V_MERGE, gw / V_MERGE);
        let nm = groups * mh * mw;
        let mut merged = vec![0f32; nm * V_MERGED_IN];
        for g in 0..groups {
            for my in 0..mh {
                for mx in 0..mw {
                    let out_t = (g * mh + my) * mw + mx;
                    let dst = &mut merged[out_t * V_MERGED_IN..(out_t + 1) * V_MERGED_IN];
                    for sy in 0..V_MERGE {
                        for sx in 0..V_MERGE {
                            let t = g * gh * gw + (my * V_MERGE + sy) * gw + (mx * V_MERGE + sx);
                            let seg = (sy * V_MERGE + sx) * V_HIDDEN;
                            dst[seg..seg + V_HIDDEN]
                                .copy_from_slice(&lh[t * V_HIDDEN..(t + 1) * V_HIDDEN]);
                        }
                    }
                }
            }
        }
        let md = e.htod(&merged)?;
        let f1 = self.linear_bias(e, &md, &self.merger_fc1, nm)?;
        let mut g = e.zeros(nm * V_MERGED_IN)?;
        e.gelu_tanh(&f1, &mut g, nm * V_MERGED_IN)?;
        self.linear_bias(e, &g, &self.merger_fc2, nm)
    }
}
