//! Host preprocessor for vision input (lane/vision): bytes -> ViT patch rows.
//!
//! Qwen2VLImageProcessorFast semantics: smart_resize to multiples of
//! factor = patch(16) * merge(2) = 32 with the pixel-area budget, rescale 1/255,
//! normalize mean/std 0.5 -> [-1, 1], patchify to [gh*gw, 3*2*16*16 = 1536] rows in
//! row-major grid order with (c, t, ph, pw) inner order — the flatten of the conv
//! weight [1152, 3, 2, 16, 16], so `VisionTower::forward` consumes rows directly.
//! Images duplicate their frame across temporal_patch 2; videos fill the pair with
//! consecutive sampled frames.
//!
//! Resize filter: CatmullRom (Keys bicubic a=-0.5, PIL-BICUBIC family). The HF fast
//! processor runs torch bicubic (a=-0.75) antialias — close but not bit-equal; the
//! merger-cosine parity gate arbitrates whether the difference matters.

use crate::vision::{V_MERGE, V_PATCH, V_PATCH_IN, V_TEMPORAL};
use base64::Engine as _;
use image::RgbImage;
use image::imageops::FilterType;

/// Area budget (pixels) from preprocessor_config: shortest_edge / longest_edge.
pub const MIN_PIXELS: usize = 65536;
pub const MAX_PIXELS: usize = 16_777_216;
const FACTOR: usize = V_PATCH * V_MERGE; // 32

pub struct PreppedImage {
    /// [gh*gw, 1536] row-major grid order.
    pub patches: Vec<f32>,
    pub gh: usize,
    pub gw: usize,
}

impl PreppedImage {
    /// Trunk tokens this image occupies (after 2x2 merge).
    pub fn n_tokens(&self) -> usize {
        n_tokens_for_grid(self.gh, self.gw)
    }
}

/// Trunk tokens a `(gh, gw)` patch grid occupies after the 2x2 merge — the planned twin
/// of `PreppedImage::n_tokens`, usable from `plan_image_bytes` BEFORE any decode.
pub fn n_tokens_for_grid(gh: usize, gw: usize) -> usize {
    gh * gw / (V_MERGE * V_MERGE)
}

/// HF's smart_resize uses Python round(), which is round-half-EVEN: a side landing
/// exactly on .5 factors (e.g. 336/32 = 10.5) rounds to the even multiple (320, not
/// 352). Rust's f64::round is half-away-from-zero and diverged there — caught by the
/// ornith15 parity gate on a 448x336 probe (grid 22x28 vs HF 20x28).
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// smart_resize (HF): round each side to a multiple of 32 preserving aspect ratio,
/// then scale into the [MIN_PIXELS, MAX_PIXELS] area budget.
pub fn smart_resize(h: usize, w: usize) -> Result<(usize, usize), String> {
    if h < 2 || w < 2 {
        return Err(format!("image too small: {w}x{h}"));
    }
    let ar = h.max(w) as f64 / h.min(w) as f64;
    if ar > 200.0 {
        return Err(format!("aspect ratio {ar:.0} exceeds 200"));
    }
    let f = FACTOR as f64;
    let (hf, wf) = (h as f64, w as f64);
    let mut h_bar = (round_half_even(hf / f) * f).max(f);
    let mut w_bar = (round_half_even(wf / f) * f).max(f);
    if h_bar * w_bar > MAX_PIXELS as f64 {
        let beta = (hf * wf / MAX_PIXELS as f64).sqrt();
        h_bar = ((hf / beta / f).floor() * f).max(f);
        w_bar = ((wf / beta / f).floor() * f).max(f);
    } else if h_bar * w_bar < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (hf * wf)).sqrt();
        h_bar = (hf * beta / f).ceil() * f;
        w_bar = (wf * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Still-image DECODE ceiling (hermes finding, fixed 2026-08-23 — the GIF bomb's
/// sibling): `load_from_memory` + `to_rgb8` expanded the FULL canvas in host RAM before
/// smart_resize's MAX_PIXELS check ever ran, so a small crafted file claiming huge
/// dimensions allocated GBs per request, pre-admission. The budget is now admitted from
/// the HEADER, before any pixel decodes — same ceiling family as `GIF_MAX_TOTAL_PIXELS`
/// (67.1M px = 192 MiB retained RGB), 4x the resize budget so every legitimately sized
/// image still decodes.
pub const IMG_MAX_DECODE_PIXELS: usize = 1 << 26;

/// Image dimensions from the container HEADER — no pixel decode, no canvas allocation.
pub fn image_header_dims(bytes: &[u8]) -> Result<(usize, usize), String> {
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image container: {e}"))?
        .into_dimensions()
        .map_err(|e| format!("image header: {e}"))?;
    Ok((w as usize, h as usize))
}

/// PRE-DECODE admission for one still image: header dims -> decode-budget check ->
/// smart_resize (min-size / aspect-ratio / area budget). Returns the patch grid
/// `(gh, gw)` the decoded image WILL produce — `n_tokens` and pad runs derive from it,
/// so budget admission can price a vision request before any canvas expands.
pub fn plan_image_bytes(bytes: &[u8]) -> Result<(usize, usize), String> {
    let (w, h) = image_header_dims(bytes)?;
    if w.saturating_mul(h) > IMG_MAX_DECODE_PIXELS {
        return Err(format!(
            "image {w}x{h} exceeds the decode budget ({IMG_MAX_DECODE_PIXELS} px) — \
             refused before decode"
        ));
    }
    let (rh, rw) = smart_resize(h, w)?;
    Ok((rh / V_PATCH, rw / V_PATCH))
}

/// Decode + resize one image to its target grid. Returns the resized RGB frame and
/// the patch grid (gh, gw) in 16px patches (both even — factor 32 guarantees it).
/// Admission runs FIRST (`plan_image_bytes`, header-only), and the decoder itself is
/// capped to the admitted dimensions so a header lying small cannot expand past them.
fn decode_frame(bytes: &[u8]) -> Result<(RgbImage, usize, usize), String> {
    plan_image_bytes(bytes)?;
    let (hw, hh) = image_header_dims(bytes)?;
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("image container: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(hw as u32);
    limits.max_image_height = Some(hh as u32);
    reader.limits(limits);
    let img = reader.decode().map_err(|e| format!("image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (rh, rw) = smart_resize(h, w)?;
    let resized = image::imageops::resize(&rgb, rw as u32, rh as u32, FilterType::CatmullRom);
    Ok((resized, rh / V_PATCH, rw / V_PATCH))
}

/// Fill patch rows for one temporal slot `t` from a frame. Rows are row-major over
/// the (gh, gw) grid; inner order (c, t, ph, pw).
fn fill_slot(rows: &mut [f32], frame: &RgbImage, gh: usize, gw: usize, t: usize) {
    let inv = 1.0f32 / 127.5;
    for py in 0..gh {
        for px in 0..gw {
            let row = &mut rows[(py * gw + px) * V_PATCH_IN..(py * gw + px + 1) * V_PATCH_IN];
            for c in 0..3 {
                let base = c * V_TEMPORAL * V_PATCH * V_PATCH + t * V_PATCH * V_PATCH;
                for ph in 0..V_PATCH {
                    for pw in 0..V_PATCH {
                        let p =
                            frame.get_pixel((px * V_PATCH + pw) as u32, (py * V_PATCH + ph) as u32);
                        row[base + ph * V_PATCH + pw] = p.0[c] as f32 * inv - 1.0;
                    }
                }
            }
        }
    }
}

/// Image bytes (png/jpeg/webp/gif/bmp) -> patch rows. The single frame fills both
/// temporal slots (HF: images are tiled to temporal_patch_size).
pub fn prep_image_bytes(bytes: &[u8]) -> Result<PreppedImage, String> {
    let (frame, gh, gw) = decode_frame(bytes)?;
    let mut patches = vec![0f32; gh * gw * V_PATCH_IN];
    for t in 0..V_TEMPORAL {
        fill_slot(&mut patches, &frame, gh, gw, t);
    }
    Ok(PreppedImage { patches, gh, gw })
}

/// `data:image/...;base64,<payload>` -> patch rows.
pub fn prep_data_uri(uri: &str) -> Result<PreppedImage, String> {
    let bytes = decode_data_uri(uri)?;
    prep_image_bytes(&bytes)
}

/// One pad-run unit crossing the API boundary: a standalone image, or one temporal
/// group of a video. Units with the same `video` index are consecutive and forward
/// TOGETHER through `forward_seq` (one attention span per video).
pub struct VisionUnit {
    pub prep: PreppedImage,
    /// Some(video_idx) for video groups; None for standalone images.
    pub video: Option<usize>,
}

/// One prepared VIDEO: temporal groups as PreppedImage units (each = one pad run of
/// `gh*gw/4` tokens) + per-group timestamps for the HF placeholder format
/// (`<t.t seconds>` before each group's pad run). Groups forward TOGETHER through
/// `VisionTower::forward_seq` — one attention span per video, the HF cu_seqlens law.
pub struct PreppedVideo {
    pub groups: Vec<PreppedImage>,
    pub timestamps: Vec<f32>,
}

/// Serving cap on total video patches (groups*gh*gw): sdpa_naive keys the whole span in
/// shared memory, so the pixel budget stays well under the HF default. Env-tunable.
pub fn video_max_pixels() -> usize {
    std::env::var("MEMRA_VIDEO_MAX_PIXELS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2_097_152)
}
pub const VID_MIN_PIXELS: usize = 4096;
/// Sampled frame cap (2 frames per temporal group).
pub const VID_MAX_FRAMES: usize = 32;

/// Decode ceilings for `prep_video_gif` (hermes finding, fixed 2026-08-19): the loop used
/// to expand EVERY frame to full-canvas RGB in host RAM before the `VID_MAX_FRAMES`
/// sample — and it runs in the HTTP handler pre-admission, so a small crafted GIF (big
/// canvas x many frames; LZW expands ~1000x) allocated GBs per request. The canvas
/// dimensions come from the GIF header, so `frames x canvas pixels` is checked against
/// the pixel ceiling AS DECODE PROCEEDS and the request is refused (clean 4xx at the
/// handler) the moment the budget would cross — retained RAM is bounded by
/// `GIF_MAX_TOTAL_PIXELS` RGB (192 MiB) plus at most one transient canvas, no matter
/// what the stream claims. 512 frames / 67.1M px comfortably cover legitimate clips
/// (a 480p GIF may run ~370 frames, ~12 s at 30 fps) — the serve path samples down to
/// 32 frames and ~2M px right after this anyway.
pub const GIF_MAX_FRAMES: usize = 512;
pub const GIF_MAX_TOTAL_PIXELS: usize = 1 << 26; // 67.1M px = 192 MiB retained RGB

/// Header-only video plan used by the HTTP admission path. It carries exactly the information
/// needed to price/render the pad runs; frame pixels are not materialized until the request has
/// passed budget and concurrency admission.
#[derive(Debug, Clone)]
pub struct PlannedVideoGroup {
    pub gh: usize,
    pub gw: usize,
    pub timestamp: f32,
}

#[derive(Debug, Clone)]
pub struct PlannedVideo {
    pub groups: Vec<PlannedVideoGroup>,
}

fn gif_need(bytes: &[u8], pos: usize, len: usize, what: &str) -> Result<(), String> {
    if pos.checked_add(len).is_none_or(|end| end > bytes.len()) {
        return Err(format!("truncated GIF {what}"));
    }
    Ok(())
}

fn gif_skip_subblocks(bytes: &[u8], pos: &mut usize) -> Result<(), String> {
    loop {
        gif_need(bytes, *pos, 1, "sub-block length")?;
        let len = bytes[*pos] as usize;
        *pos += 1;
        if len == 0 {
            return Ok(());
        }
        gif_need(bytes, *pos, len, "sub-block payload")?;
        *pos += len;
    }
}

/// Parse GIF structure, frame count, delays, and canvas dimensions without decoding a single
/// pixel. This mirrors the sampling arithmetic in `prep_video_gif`, but keeps the expensive
/// composited frame buffers behind the HTTP admission gates.
pub fn plan_video_gif(bytes: &[u8]) -> Result<PlannedVideo, String> {
    if bytes.len() < 13 || !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Err("invalid GIF header".into());
    }
    let cw = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
    let ch = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let canvas_px = cw
        .checked_mul(ch)
        .ok_or_else(|| "gif canvas dimensions overflow".to_string())?;
    if canvas_px == 0 {
        return Err("gif has an empty canvas".into());
    }
    let max_frames = GIF_MAX_FRAMES.min(GIF_MAX_TOTAL_PIXELS / canvas_px);
    if max_frames == 0 {
        return Err(format!(
            "gif canvas {cw}x{ch} exceeds the decode budget ({GIF_MAX_TOTAL_PIXELS} px)"
        ));
    }

    let mut pos = 13usize;
    let packed = bytes[10];
    if packed & 0x80 != 0 {
        let table_len = 3usize
            .checked_mul(1usize << ((packed & 0x07) as usize + 1))
            .ok_or_else(|| "GIF color table length overflow".to_string())?;
        gif_need(bytes, pos, table_len, "global color table")?;
        pos += table_len;
    }

    let mut timestamps = Vec::new();
    let mut elapsed = 0f32;
    let mut next_delay = 0.01f32;
    let mut trailer_seen = false;
    while pos < bytes.len() {
        match bytes[pos] {
            0x3B => {
                trailer_seen = true;
                break;
            }
            0x21 => {
                gif_need(bytes, pos, 2, "extension label")?;
                let label = bytes[pos + 1];
                pos += 2;
                if label == 0xF9 {
                    gif_need(bytes, pos, 1, "graphic-control block size")?;
                    let block_len = bytes[pos] as usize;
                    pos += 1;
                    if block_len != 4 {
                        return Err(format!(
                            "unsupported GIF graphic-control block length {block_len}"
                        ));
                    }
                    gif_need(bytes, pos, block_len + 1, "graphic-control block")?;
                    let delay_cs = u16::from_le_bytes([bytes[pos + 1], bytes[pos + 2]]);
                    // image::Delay exposes the same centiseconds as milliseconds
                    // (delay_cs * 10) before dividing by 1000. Mirror that exact
                    // operation order so the metadata-only planner and decoder
                    // accumulate identical f32 timestamps over long GIFs.
                    next_delay = ((delay_cs as f32 * 10.0) / 1000.0).max(0.01);
                    pos += block_len;
                    if bytes[pos] != 0 {
                        return Err("GIF graphic-control block is not terminated".into());
                    }
                    pos += 1;
                } else {
                    gif_skip_subblocks(bytes, &mut pos)?;
                }
            }
            0x2C => {
                gif_need(bytes, pos, 10, "image descriptor")?;
                let fw = u16::from_le_bytes([bytes[pos + 5], bytes[pos + 6]]) as usize;
                let fh = u16::from_le_bytes([bytes[pos + 7], bytes[pos + 8]]) as usize;
                if fw == 0 || fh == 0 {
                    return Err("GIF frame has an empty rectangle".into());
                }
                if timestamps.len() >= max_frames {
                    return Err(format!(
                        "gif exceeds the decode budget: more than {max_frames} frames at {cw}x{ch} \
                         (ceiling {GIF_MAX_FRAMES} frames / {GIF_MAX_TOTAL_PIXELS} total px)"
                    ));
                }
                let frame_packed = bytes[pos + 9];
                pos += 10;
                if frame_packed & 0x80 != 0 {
                    let table_len = 3usize
                        .checked_mul(1usize << ((frame_packed & 0x07) as usize + 1))
                        .ok_or_else(|| "GIF local color table length overflow".to_string())?;
                    gif_need(bytes, pos, table_len, "local color table")?;
                    pos += table_len;
                }
                gif_need(bytes, pos, 1, "LZW minimum code size")?;
                pos += 1;
                gif_skip_subblocks(bytes, &mut pos)?;
                timestamps.push(elapsed);
                elapsed += next_delay;
                next_delay = 0.01;
            }
            other => return Err(format!("unsupported GIF block 0x{other:02x}")),
        }
    }
    if !trailer_seen {
        return Err("GIF is missing its trailer".into());
    }
    if timestamps.is_empty() {
        return Err("gif has no frames".into());
    }
    if timestamps.len() == 1 {
        timestamps.push(timestamps[0]);
    }
    let total = timestamps.len();
    let take = total.min(VID_MAX_FRAMES) & !1;
    let picked: Vec<usize> = (0..take).map(|i| i * total / take).collect();
    let (rh, rw) = smart_resize_video(take, ch, cw)?;
    let (gh, gw) = (rh / V_PATCH, rw / V_PATCH);
    let groups = (0..take / 2)
        .map(|g| PlannedVideoGroup {
            gh,
            gw,
            timestamp: timestamps[picked[2 * g]],
        })
        .collect();
    Ok(PlannedVideo { groups })
}

/// HF Qwen3VL video smart_resize: the pixel budget covers t_bar*h*w — ALL frames.
fn smart_resize_video(frames: usize, h: usize, w: usize) -> Result<(usize, usize), String> {
    if h < 2 || w < 2 {
        return Err(format!("frame too small: {w}x{h}"));
    }
    let ar = h.max(w) as f64 / h.min(w) as f64;
    if ar > 200.0 {
        return Err(format!("aspect ratio {ar:.0} exceeds 200"));
    }
    let f = FACTOR as f64;
    let (hf, wf) = (h as f64, w as f64);
    let t_bar = ((frames as f64 / V_TEMPORAL as f64).round() * V_TEMPORAL as f64).max(2.0);
    let mut h_bar = (round_half_even(hf / f) * f).max(f);
    let mut w_bar = (round_half_even(wf / f) * f).max(f);
    let (min_px, max_px) = (VID_MIN_PIXELS as f64, video_max_pixels() as f64);
    if t_bar * h_bar * w_bar > max_px {
        let beta = (frames as f64 * hf * wf / max_px).sqrt();
        h_bar = ((hf / beta / f).floor() * f).max(f);
        w_bar = ((wf / beta / f).floor() * f).max(f);
    } else if t_bar * h_bar * w_bar < min_px {
        let beta = (min_px / (frames as f64 * hf * wf)).sqrt();
        h_bar = (hf * beta / f).ceil() * f;
        w_bar = (wf * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// Animated GIF -> prepared video: decode frames + delays, uniform-sample to an even
/// count <= VID_MAX_FRAMES, resize on the total-pixel budget, patchify CONSECUTIVE
/// frame pairs into temporal groups (frame 2g fills t=0, 2g+1 fills t=1). Timestamps
/// come from the GIF's own delays at the sampled indices (HF `_calculate_timestamps`).
pub fn prep_video_gif(bytes: &[u8]) -> Result<PreppedVideo, String> {
    use image::AnimationDecoder;
    use image::ImageDecoder as _;
    let dec = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|e| format!("gif decode: {e}"))?;
    // Decode budget from the HEADER, before any frame expands: every decoded frame
    // composites to the full canvas, so canvas pixels bound the per-frame cost and
    // frames x canvas is checked against the ceiling as decode proceeds (at most one
    // transient frame past the cap ever exists). Over-limit refuses cleanly — the
    // handler surfaces it as a 4xx — instead of expanding the whole stream in host RAM.
    let (cw, ch) = dec.dimensions();
    let canvas_px = (cw as usize) * (ch as usize);
    if canvas_px == 0 {
        return Err("gif has an empty canvas".into());
    }
    let max_frames = GIF_MAX_FRAMES.min(GIF_MAX_TOTAL_PIXELS / canvas_px);
    if max_frames == 0 {
        return Err(format!(
            "gif canvas {cw}x{ch} exceeds the decode budget ({GIF_MAX_TOTAL_PIXELS} px)"
        ));
    }
    let mut frames: Vec<(RgbImage, f32)> = Vec::new(); // (frame, start_seconds)
    let mut t = 0f32;
    for fr in dec.into_frames() {
        if frames.len() >= max_frames {
            return Err(format!(
                "gif exceeds the decode budget: more than {max_frames} frames at {cw}x{ch} \
                 (ceiling {GIF_MAX_FRAMES} frames / {GIF_MAX_TOTAL_PIXELS} total px)"
            ));
        }
        let fr = fr.map_err(|e| format!("gif frame: {e}"))?;
        let (num, den) = fr.delay().numer_denom_ms();
        let dt = if den == 0 {
            100.0
        } else {
            num as f32 / den as f32
        } / 1000.0;
        frames.push((
            image::DynamicImage::ImageRgba8(fr.into_buffer()).to_rgb8(),
            t,
        ));
        t += dt.max(0.01);
    }
    if frames.is_empty() {
        return Err("gif has no frames".into());
    }
    // still gif: duplicate the frame so one temporal group forms
    if frames.len() == 1 {
        let f0 = frames[0].clone();
        frames.push((f0.0, f0.1));
    }
    // uniform sample to an even count <= VID_MAX_FRAMES
    let total = frames.len();
    let take = total.min(VID_MAX_FRAMES) & !1;
    let picked: Vec<usize> = (0..take)
        .map(|i| i * total / take) // floor spacing, strictly increasing for take <= total
        .collect();
    let (h, w) = (frames[0].0.height() as usize, frames[0].0.width() as usize);
    let (rh, rw) = smart_resize_video(take, h, w)?;
    let (gh, gw) = (rh / V_PATCH, rw / V_PATCH);
    let mut groups = Vec::with_capacity(take / 2);
    let mut timestamps = Vec::with_capacity(take / 2);
    for g in 0..take / 2 {
        let (a, b) = (picked[2 * g], picked[2 * g + 1]);
        let mut patches = vec![0f32; gh * gw * V_PATCH_IN];
        for (slot, idx) in [(0usize, a), (1usize, b)] {
            let resized = image::imageops::resize(
                &frames[idx].0,
                rw as u32,
                rh as u32,
                FilterType::CatmullRom,
            );
            fill_slot(&mut patches, &resized, gh, gw, slot);
        }
        groups.push(PreppedImage { patches, gh, gw });
        timestamps.push(frames[a].1);
    }
    Ok(PreppedVideo { groups, timestamps })
}

/// Parse a base64 data URI into raw bytes (any `data:*;base64,` media type).
pub fn decode_data_uri(uri: &str) -> Result<Vec<u8>, String> {
    let rest = uri
        .strip_prefix("data:")
        .ok_or_else(|| "expected data: URI (http fetch requires MEMRA_FETCH_URLS=1)".to_string())?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| "malformed data URI: no comma".to_string())?;
    if !meta.ends_with(";base64") {
        return Err("data URI must be base64-encoded".into());
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("base64 decode: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_multiples_and_budget() {
        // typical photo
        let (h, w) = smart_resize(1080, 1920).unwrap();
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h * w >= MIN_PIXELS && h * w <= MAX_PIXELS);
        // tiny icon scales UP to the floor
        let (h, w) = smart_resize(64, 64).unwrap();
        assert!(h * w >= MIN_PIXELS);
        // huge pano scales DOWN under the cap
        let (h, w) = smart_resize(8000, 12000).unwrap();
        assert!(h * w <= MAX_PIXELS);
        assert!(smart_resize(10, 4000).is_err()); // ar > 200
    }

    /// Minimal BMP whose HEADER claims `w x h` — the pixel payload is absent, so any
    /// path that survives past the header check would fail loudly at decode, and any
    /// path that ALLOCATES the claimed canvas before checking would try to expand
    /// w*h*3 bytes. The tooth's decode-bomb stand-in.
    fn bmp_header_claiming(w: u32, h: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"BM"); // signature
        b.extend_from_slice(&54u32.to_le_bytes()); // file size (lie, irrelevant)
        b.extend_from_slice(&0u32.to_le_bytes()); // reserved
        b.extend_from_slice(&54u32.to_le_bytes()); // pixel data offset
        b.extend_from_slice(&40u32.to_le_bytes()); // BITMAPINFOHEADER size
        b.extend_from_slice(&(w as i32).to_le_bytes());
        b.extend_from_slice(&(h as i32).to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes()); // planes
        b.extend_from_slice(&24u16.to_le_bytes()); // bpp
        b.extend_from_slice(&[0u8; 24]); // compression..colors_important
        b
    }

    #[test]
    fn decode_bomb_refuses_pre_decode() {
        // TOOTH (hermes findings: still-image decode bomb + full-canvas expansion
        // before the pixel budget; fixed 2026-08-23): a tiny request whose header
        // claims a 768-megapixel canvas must refuse at ADMISSION — named decode-budget
        // error from the header dims, before load/to_rgb8 can expand anything.
        let bomb = bmp_header_claiming(16_000, 16_000);
        let err = plan_image_bytes(&bomb).unwrap_err();
        assert!(
            err.contains("exceeds the decode budget"),
            "want the named pre-decode refusal, got: {err}"
        );
        // The full prep path refuses with the same admission error (it must not reach
        // the decoder at all — an absent pixel payload would produce a decode error
        // instead, which would mean the canvas was attempted).
        let err = match prep_image_bytes(&bomb) {
            Ok(_) => panic!("bomb must not prep"),
            Err(e) => e,
        };
        assert!(
            err.contains("exceeds the decode budget"),
            "prep must refuse at admission, not at decode: {err}"
        );
        // Header dims really are read without pixel decode.
        assert_eq!(image_header_dims(&bomb).unwrap(), (16_000, 16_000));
        // Positive control: an in-budget image plans to the same grid decode produces.
        let img = RgbImage::new(64, 64);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let planned = plan_image_bytes(buf.get_ref()).unwrap();
        let prep = prep_image_bytes(buf.get_ref()).unwrap();
        assert_eq!(planned, (prep.gh, prep.gw), "planned grid == decoded grid");
    }

    #[test]
    fn patchify_shape_and_order() {
        // 2x2-patch (32x32 px) synthetic image, distinct channel values
        let mut img = RgbImage::new(64, 64);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([x as u8, y as u8, 200]);
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let prep = prep_image_bytes(buf.get_ref()).unwrap();
        assert_eq!(prep.patches.len(), prep.gh * prep.gw * V_PATCH_IN);
        assert_eq!(prep.gh % V_MERGE, 0);
        assert_eq!(prep.gw % V_MERGE, 0);
        // temporal slots identical for still images
        let row = &prep.patches[0..V_PATCH_IN];
        let slot = V_PATCH * V_PATCH;
        for c in 0..3 {
            let b = c * V_TEMPORAL * slot;
            assert_eq!(row[b..b + slot], row[b + slot..b + 2 * slot]);
        }
        // values in [-1, 1]
        assert!(prep.patches.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    /// Hand-crafted minimal GIF: `frames` one-black-pixel frames on a `w x h` canvas.
    /// Each frame is the canonical 35-byte smallest-GIF image block (LZW: clear, index 0,
    /// end -> bytes 0x44 0x01), so a high frame count costs ~18 bytes/frame on the wire
    /// while every DECODED frame composites to the full canvas — the decode-bomb shape.
    fn crafted_gif(w: u16, h: u16, frames: usize) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GIF89a");
        b.extend_from_slice(&w.to_le_bytes());
        b.extend_from_slice(&h.to_le_bytes());
        b.push(0x80); // global color table, 2 entries
        b.push(0); // background color index
        b.push(0); // aspect ratio
        b.extend_from_slice(&[0, 0, 0, 0xFF, 0xFF, 0xFF]); // GCT: black, white
        for _ in 0..frames {
            b.push(0x2C); // image descriptor
            b.extend_from_slice(&0u16.to_le_bytes()); // left
            b.extend_from_slice(&0u16.to_le_bytes()); // top
            b.extend_from_slice(&1u16.to_le_bytes()); // width 1
            b.extend_from_slice(&1u16.to_le_bytes()); // height 1
            b.push(0); // no local color table
            b.push(0x02); // LZW min code size
            b.extend_from_slice(&[0x02, 0x44, 0x01]); // sub-block: clear, idx 0, end
            b.push(0x00); // block terminator
        }
        b.push(0x3B); // trailer
        b
    }

    #[test]
    fn gif_decode_bomb_is_refused_before_full_expansion() {
        // 2000x2000 canvas = 4M px/frame -> the 67.1M px budget admits 16 frames; a
        // 64-frame stream (~1.3 KB on the wire, ~1 GiB decoded) must refuse at the
        // budget, not expand: pre-fix this test allocated 64 x 16 MB RGBA canvases.
        fn expect_err(bytes: &[u8]) -> String {
            match prep_video_gif(bytes) {
                Err(e) => e,
                Ok(_) => panic!("decode-bomb GIF was accepted"),
            }
        }
        let bomb = crafted_gif(2000, 2000, 64);
        assert!(bomb.len() < 2048, "the bomb itself is tiny on the wire");
        let err = expect_err(&bomb);
        assert!(err.contains("decode budget"), "{err}");

        // same canvas, frame count within budget: decodes fine.
        let ok = crafted_gif(2000, 2000, 4);
        let vid = prep_video_gif(&ok).unwrap();
        assert_eq!(vid.groups.len(), 2); // 4 frames -> 2 temporal groups

        // frame-count bomb on a tiny canvas: trips the flat frame ceiling.
        let err = expect_err(&crafted_gif(8, 8, GIF_MAX_FRAMES + 8));
        assert!(err.contains("decode budget"), "{err}");

        // canvas alone past the pixel budget: refused straight from the header.
        let err = expect_err(&crafted_gif(0xFFFF, 0xFFFF, 1));
        assert!(err.contains("exceeds the decode budget"), "{err}");
    }

    #[test]
    fn gif_plan_reads_metadata_without_materializing_frames() {
        let bytes = crafted_gif(64, 64, 4);
        let plan = plan_video_gif(&bytes).unwrap();
        assert_eq!(plan.groups.len(), 2);
        assert!(plan.groups.iter().all(|group| group.gh > 0 && group.gw > 0));
        let prepared = prep_video_gif(&bytes).unwrap();
        assert_eq!(
            plan.groups
                .iter()
                .map(|group| (group.gh, group.gw))
                .collect::<Vec<_>>(),
            prepared
                .groups
                .iter()
                .map(|group| (group.gh, group.gw))
                .collect::<Vec<_>>()
        );
        assert!(plan_video_gif(&crafted_gif(2000, 2000, 64)).is_err());
    }

    #[test]
    fn data_uri_roundtrip() {
        let png = {
            let img = RgbImage::new(32, 32);
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
            buf.into_inner()
        };
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        );
        let prep = prep_data_uri(&uri).unwrap();
        assert_eq!(prep.n_tokens(), prep.gh * prep.gw / 4);
        assert!(decode_data_uri("http://x/y.png").is_err());
    }
}
