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
use cudarc::driver::{CudaContext, CudaSlice};
use memra_gguf::dequant::bf16_to_f32;
use memra_gguf::safetensors::StShard;
use std::path::Path;
use std::sync::Arc;

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

/// Overlay publication policy (`MEMRA_VISION_OVERLAY_PUBLISH`, lane/glm53-vision-ppn):
/// whether the tower's rows are republished into the engine that owns embedding intake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayPublish {
    /// DEFAULT: publish iff the intake engine's CUDA context differs from the tower's.
    Auto,
    /// Always take the publication path, even when the contexts already match — the rig arm
    /// that exercises the cross-context code on a one-card box (byte identity is the bar).
    Force,
    /// Never publish: the pre-lane program. A cross-context intake is then refused by the
    /// residency law in `splice_into` / the ppN prime, never silently peer-read.
    Never,
}

/// Pure resolver for `MEMRA_VISION_OVERLAY_PUBLISH` so unit tests pin the whole arm matrix
/// without mutating process-global environment (the `pp::peer_probe_startup_policy` pattern).
/// `None` = unset = Auto; `auto`/`1` = Auto; `force` = Force; `0`/`off` = Never. An
/// unrecognized value is a REFUSAL, not a silent default — a mistyped door must never decide
/// a correctness path quietly.
pub fn overlay_publish_resolve(v: Option<&str>) -> Result<OverlayPublish, String> {
    match v.map(str::trim) {
        None | Some("") | Some("auto") | Some("1") => Ok(OverlayPublish::Auto),
        Some("force") => Ok(OverlayPublish::Force),
        Some("0") | Some("off") => Ok(OverlayPublish::Never),
        Some(other) => Err(format!(
            "MEMRA_VISION_OVERLAY_PUBLISH={other:?} unrecognized (want auto|force|0)"
        )),
    }
}

/// `overlay_publish_resolve` over the live environment.
pub fn overlay_publish_mode() -> Result<OverlayPublish, Box<dyn std::error::Error>> {
    let raw = std::env::var("MEMRA_VISION_OVERLAY_PUBLISH").ok();
    overlay_publish_resolve(raw.as_deref()).map_err(|e| -> Box<dyn std::error::Error> { e.into() })
}

static OVERLAY_PUBLICATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Overlay publications this process has performed. A gate arm that means to exercise the
/// publication path asserts this counter MOVED — a same-context run would otherwise take the
/// zero-copy branch and print PASS having tested nothing (the non-vacuity law).
pub fn overlay_publications() -> u64 {
    OVERLAY_PUBLICATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Mixed-embedding prime overlay: image embeddings that replace `<|image_pad|>` token
/// embeddings at prompt-relative positions during `prime_cache_overlaid`. `rows` holds all
/// images' merger outputs concatenated ([total_rows, n_embd]); each span is
/// `(prompt_pos, row_off, n_rows)` — rows `[row_off, row_off+n_rows)` land at prompt
/// positions `[prompt_pos, prompt_pos+n_rows)`. Spans must not overlap.
///
/// ROW RESIDENCY IS CARRIED, NOT ASSUMED (lane/glm53-vision-ppn, 2026-09-01). `rows` is a
/// device pointer, and a device pointer is only meaningful inside ONE CUDA context. The
/// overlay therefore records the context its rows live in, and every consumer checks it. The
/// pre-lane code proxied that invariant with "the overlay was built on the primary engine AND
/// stage 0 IS the primary engine" (`std::ptr::eq` on `&Engine`), which is both too weak (two
/// Engines can share one context — `CudaContext::new` retains the device's PRIMARY context,
/// so per-stage Engines on one device are one address space) and too strong (it refused the
/// real 3-card serving shape, where the worker's primary engine follows the LAST pp stage and
/// stage 0's intake engine is a different device). Context identity is the exact condition.
pub struct EmbedOverlay {
    pub rows: CudaSlice<f32>,
    pub spans: Vec<(usize, usize, usize)>,
    /// The CUDA context `rows` live in. Held as an `Arc` so the context cannot outlive the
    /// pointer it owns, and so the struct stays `Send` (a raw `CUcontext` would not).
    ctx: Arc<CudaContext>,
}

impl EmbedOverlay {
    /// Wrap rows that were built on `e`, recording `e`'s context as their residency.
    pub fn new(e: &Engine, rows: CudaSlice<f32>, spans: Vec<(usize, usize, usize)>) -> Self {
        Self {
            rows,
            spans,
            ctx: e.ctx().clone(),
        }
    }

    /// The context `rows` live in.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// true iff `e` can dereference `rows` — i.e. `e` runs in the same CUDA context.
    pub fn resident_in(&self, e: &Engine) -> bool {
        e.ctx().cu_ctx() == self.ctx.cu_ctx()
    }

    /// The residency law as a REFUSAL, shared by every consumer of `rows` (the common splice,
    /// and gemma4's masked-prefill arm, whose splice arithmetic differs deliberately). One
    /// message, one law: a site that reads `rows` calls this first.
    pub fn require_resident(&self, e: &Engine) -> Result<(), Box<dyn std::error::Error>> {
        if self.resident_in(e) {
            return Ok(());
        }
        Err(format!(
            "vision embedding overlay refused: overlay rows are resident in the CUDA context \
             of dev{} but the splice runs on dev{} — a cross-context splice would read a \
             pointer from another address space. Publish the overlay into the consuming \
             engine (EmbedOverlay::new_published / MEMRA_VISION_OVERLAY_PUBLISH)",
            self.ctx.ordinal(),
            e.ctx().ordinal(),
        )
        .into())
    }

    /// Build an overlay whose rows are resident in the engine that will CONSUME them.
    ///
    /// `tower` is the engine the vision tower ran on (where `rows` currently live); `intake`
    /// is the engine that owns embedding intake for the serving placement
    /// (`HybridModel::vision_intake_engine` — the primary engine on a single-device or
    /// streams-off shape, pp stage 0's engine under a per-stage-stream ppN split).
    ///
    /// PUBLICATION IS A HOST BOUNCE, deliberately. `tower.dtoh` drains the tower's stream at
    /// a HOST boundary and `intake.htod` + one `synchronize` place the bytes on the intake
    /// stream, so the ordering argument needs no event plumbing and no P2P capability: it
    /// holds on every placement, including `MEMRA_PP_HOST_BOUNCE=1` boxes with no peer
    /// access. The cost is ONE round trip of `[total_rows, n_embd]` f32 per session (a few
    /// MiB; ~5 MiB for a 256-row 5120-wide image) against a tower that already host-bounces
    /// q/k/v and the merger input in EVERY block — a peer D2D twin is a named follow-up, not
    /// a correctness question. Nothing here is per token: decode never touches an overlay.
    ///
    /// The bytes are moved, never transformed: f32 through the host is bit-exact, which is
    /// what makes `MEMRA_VISION_OVERLAY_PUBLISH=force` a byte-identity gate arm.
    ///
    /// CALL OUTSIDE ANY pp STAGE SCOPE. The ambient stream override is THREAD-LOCAL and applies
    /// to every engine on the thread, so inside `PpNRt::enter(s)` this upload would bind to
    /// stage `s`'s stream — which belongs to another context whenever the stages differ. The
    /// serving caller (`build_vision_overlay`, at the first prefill tick) and the gate arms both
    /// run outside stage scopes.
    ///
    /// THE FREE IS ORDERED TOO, and by the same host boundary the reads are. `rows` is dropped
    /// when the session's overlay is dropped, and cudarc enqueues that free on the stream the
    /// buffer was allocated on. Every prime call that could have read it ends with a
    /// host-synchronizing `dtoh` of its logits (`hyper_prime_tail`) plus `publish_all_to` over
    /// every stage stream, so all stage-side reads have retired before any later host code —
    /// including the drop — runs. This is the same argument the pre-lane path relied on for
    /// rows allocated on the caller's stream; it is written here because the accrace lane
    /// (`MEMRA_PP_EXIT_PUBLISH`) is what happens when it is only assumed.
    pub fn new_published(
        tower: &Engine,
        intake: &Engine,
        rows: CudaSlice<f32>,
        spans: Vec<(usize, usize, usize)>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mode = overlay_publish_mode()?;
        let same_ctx = tower.ctx().cu_ctx() == intake.ctx().cu_ctx();
        if mode == OverlayPublish::Never || (mode == OverlayPublish::Auto && same_ctx) {
            return Ok(Self::new(tower, rows, spans));
        }
        let host = tower.dtoh(&rows)?;
        drop(rows);
        let published = intake.htod(&host)?;
        // The consumer may run on a DIFFERENT stream of the intake context (the ppN stage-0
        // stage stream, while this upload lands on the intake engine's ambient stream). One
        // host boundary orders every later stream against this upload; an event would order
        // only the stream we recorded it on.
        intake.stream().synchronize()?;
        OVERLAY_PUBLICATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[vision] overlay published to the intake engine: dev{} -> dev{} rows={} \
             elems={} MiB={:.2} mode={mode:?}",
            tower.ctx().ordinal(),
            intake.ctx().ordinal(),
            spans.iter().map(|&(_, _, n)| n).sum::<usize>(),
            host.len(),
            (host.len() * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0),
        );
        Ok(Self::new(intake, published, spans))
    }

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
            // A window aliases the SAME device rows, so it carries the same residency.
            ctx: self.ctx.clone(),
        })
    }

    /// The mixed-embedding splice itself: image rows overwrite placeholder-token embeddings
    /// inside a prime call's prompt-relative window `[chunk_off, chunk_off + t)`, BEFORE any
    /// downstream transform (stream expansion, trunk walk). ONE implementation shared by the
    /// single-engine hyper walk and the ppN stage-0 intake (lane/glm5-vision-default-on) so
    /// the splice point cannot drift between arms. `embedded` is the `[t, n_embd]` token
    /// embedding buffer of this call.
    ///
    /// RESIDENCY LAW, checked HERE — at the copy, for every arm (serial chunk walk, hyper
    /// walk, ppN stage-0 intake), rather than at one call site's placement assumption: `rows`
    /// must live in `e`'s CUDA context. A mismatch is a REFUSAL. Publish the overlay into the
    /// consuming engine with `EmbedOverlay::new_published` instead of relaxing this.
    pub fn splice_into(
        &self,
        e: &Engine,
        embedded: &mut CudaSlice<f32>,
        chunk_off: usize,
        t: usize,
        n_embd: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.require_resident(e)?;
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

#[cfg(test)]
mod tests {
    use super::{EmbedOverlay, OverlayPublish, overlay_publish_resolve};
    use crate::Engine;
    use cudarc::driver::CudaContext;

    #[test]
    fn overlay_publish_arm_matrix_is_pinned() {
        // Absent and empty mean the DEFAULT, and the default is Auto by decision (FLAGS row):
        // byte-identical to the pre-lane program wherever the contexts already match, and the
        // only way a per-stage-stream ppN placement serves an image at all.
        assert_eq!(overlay_publish_resolve(None), Ok(OverlayPublish::Auto));
        assert_eq!(overlay_publish_resolve(Some("")), Ok(OverlayPublish::Auto));
        assert_eq!(
            overlay_publish_resolve(Some("auto")),
            Ok(OverlayPublish::Auto)
        );
        assert_eq!(overlay_publish_resolve(Some("1")), Ok(OverlayPublish::Auto));
        assert_eq!(
            overlay_publish_resolve(Some(" force ")),
            Ok(OverlayPublish::Force)
        );
        assert_eq!(
            overlay_publish_resolve(Some("0")),
            Ok(OverlayPublish::Never)
        );
        assert_eq!(
            overlay_publish_resolve(Some("off")),
            Ok(OverlayPublish::Never)
        );
    }

    #[test]
    fn an_unrecognized_overlay_publish_value_refuses_rather_than_defaulting() {
        // The failure mode this closes: `MEMRA_VISION_OVERLAY_PUBLISH=yes` silently reading as
        // the default would make an operator believe a door was thrown that never was.
        for bad in ["yes", "true", "2", "Force", "no"] {
            let err = overlay_publish_resolve(Some(bad)).expect_err(bad);
            assert!(err.contains("unrecognized"), "{err}");
            assert!(err.contains(bad), "{err}");
        }
    }
    /// THE RESIDENCY REFUSAL, EXECUTED — on one card.
    ///
    /// The law's whole point is that it bites when the overlay's rows are in another CUDA
    /// context, and the serving shape that provokes it needs two devices. But a context is not
    /// a device: `CudaContext::new_non_primary` gives a genuinely independent context on the
    /// SAME card (`CudaContext::new` retains the device's PRIMARY context, which is why every
    /// per-stage Engine on one device shares an address space). So the refusal path can be run
    /// here rather than only reasoned about — the "loud failures fail quietly" law: execute
    /// every failure path and assert the outcome.
    ///
    /// What this does NOT claim: nothing about whether a PUBLISHED pointer works across two
    /// contexts. That is the multi-card box battery's job (`research/glm53-vision-ppn-20260901/box/`).
    ///
    /// Needs a CUDA device and FAILS (never skips) without one — a skipping test is how a gate
    /// reports green in perpetuity while running nothing. Not reachable from CI, which runs lib
    /// suites for the CUDA-FREE crates only; its receipt is the banked rig run.
    #[test]
    #[ignore = "needs a CUDA device; run on the rig under flock /tmp/memra-5090.lock"]
    fn a_foreign_context_overlay_is_refused_not_dereferenced() {
        let e = Engine::new(0).expect("CUDA engine on device 0");
        // An independent context on the same card, and rows allocated inside it. No Engine is
        // built on it deliberately: nothing in this file may make a foreign overlay reachable
        // from production code, so the fixture goes through the struct's private field.
        let foreign = CudaContext::new_non_primary(0, 0).expect("non-primary context on device 0");
        assert_ne!(
            foreign.cu_ctx(),
            e.ctx().cu_ctx(),
            "new_non_primary must NOT hand back the primary context, or this test is vacuous"
        );
        let stream = foreign.new_stream().expect("stream in the foreign context");
        let rows = stream
            .alloc_zeros::<f32>(8 * 4)
            .expect("rows in the foreign context");
        let ov = EmbedOverlay {
            rows,
            spans: vec![(0, 0, 2)],
            ctx: foreign.clone(),
        };

        assert!(
            !ov.resident_in(&e),
            "an overlay built in another context must not read as resident"
        );
        let err = ov
            .require_resident(&e)
            .expect_err("the residency law must REFUSE a foreign-context overlay")
            .to_string();
        assert!(err.contains("another address space"), "{err}");

        // And the refusal happens BEFORE any copy is attempted: the splice must return the same
        // named error rather than handing a foreign pointer to a memcpy.
        let mut embedded = e
            .zeros(8 * 4)
            .expect("destination rows on the primary engine");
        let err = ov
            .splice_into(&e, &mut embedded, 0, 8, 4)
            .expect_err("splice_into must refuse a foreign-context overlay");
        assert!(err.to_string().contains("another address space"), "{err}");

        // The primary-context twin of the same shape is accepted, so the assertion above is
        // about RESIDENCY and not about the arguments.
        let ok = EmbedOverlay::new(&e, e.zeros(8 * 4).unwrap(), vec![(0, 0, 2)]);
        assert!(ok.resident_in(&e));
        ok.splice_into(&e, &mut embedded, 0, 8, 4)
            .expect("a same-context overlay splices");
    }
}
