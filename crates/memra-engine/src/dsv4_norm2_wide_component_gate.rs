//! Component gate for `MEMRA_DSV4_NORM2_WIDE`: bit equality and per-launch timing
//! for the wide norm2 pack against the single-CTA original, on the operands the
//! live forward actually presents.
//!
//! The door is a SAME-CLASS rewrite, so the bit check here is not a tolerance
//! check. Every CTA of the wide kernel repeats the identical 128-thread
//! eight-load accumulation and the identical `dsv4_block_sum_f32` tree over the
//! whole row, and the written value is a pure function of (column, rsq). A
//! single differing bit is therefore a defect, not a numeric-class question, and
//! this gate refuses on the first one.
//!
//! Operands come from the norm2 capture directory produced by
//! `dsv4-norm-fuse2-gate --capture-components`: sites 0 (attention) and 1 (FFN)
//! are the two pack sites, 43 layers on each of 2 ranks, 172 real operands. The
//! pack's live-count domain is a single point by construction: `memra_dsv4_norm2_pack`
//! refuses anything but one row of 4096, so 172 operands cover it exhaustively.
use super::norm2_component_gate::read_words;
use super::*;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Tiles the sweep walks. 1 reproduces the original geometry through the wide
/// symbol, which is the sweep's own red arm: it must land on the control's time.
pub const WIDE_TILE_SWEEP: [i32; 6] = [1, 2, 4, 8, 16, 32];
const GUARD: usize = 32;
const CANARY: u32 = 0x4b123456;
const COLS: usize = 4096;
/// Repeats per (site, cold) cell. `repeat % 4` in {1,2} is the arm, so each cell
/// is five ABBA blocks and the first block is discarded as warm-up.
const REPEATS: usize = 40;
const WARMUP: usize = 4;

/// Unique working set for one pack launch, matching the SHAPES.md model:
/// 4n input + 4n weight + 4n retained f32 + 2n BF16 pack.
fn unique_bytes(n: usize) -> usize {
    14 * n
}
/// What the wide arm actually issues: every extra tile re-reads the row for its
/// own copy of the reduction. Reported next to the unique model so the redundancy
/// is visible rather than hidden inside a flattering GB/s.
fn issued_bytes(n: usize, tiles: i32) -> usize {
    unique_bytes(n) + (tiles as usize - 1) * 4 * n
}
fn gb_s(bytes: usize, us: f32) -> f32 {
    bytes as f32 / (us * 1e3)
}

struct Site {
    rank: usize,
    layer: usize,
    kind: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    eps: f32,
}

fn load_sites(dir: &Path, rank: usize) -> Res<Vec<Site>> {
    let mut sites = Vec::new();
    for layer in 0..43 {
        for kind in 0..2 {
            let path = dir.join(format!("rank{rank}-layer{layer:02}-site{kind}"));
            let x = read_words(&path.join("x.bin"))?;
            let w = read_words(&path.join("w.bin"))?;
            let meta: Vec<usize> = std::fs::read_to_string(path.join("meta.txt"))
                .map_err(|e| e.to_string())?
                .split_whitespace()
                .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
                .collect::<Res<_>>()?;
            if meta.len() != 3 {
                return Err("bad norm2 metadata".into());
            }
            let (cols, rows, eps) = (meta[0], meta[1], f32::from_bits(meta[2] as u32));
            // The pack domain is exactly one row of 4096. Anything else means the
            // capture directory is not the one this door dispatches on.
            if cols != COLS
                || rows != 1
                || x.len() != COLS
                || w.len() != COLS
                || !(eps > 0.0 && eps.is_finite())
                || x.iter().chain(&w).any(|v| !v.is_finite())
            {
                return Err(format!("norm2 pack operand contract at {path:?}"));
            }
            sites.push(Site {
                rank,
                layer,
                kind,
                x,
                w,
                eps,
            });
        }
    }
    if sites.len() != 86 {
        return Err("expected 86 pack operands per rank".into());
    }
    Ok(sites)
}

impl Dsv4Gpu {
    /// Bit equality plus warm and cold per-launch timing, ABBA within each cell,
    /// for every captured pack operand on both ranks. `sweep` additionally walks
    /// the tile domain on the first site of each rank so the pinned constant is
    /// chosen from a measurement rather than a guess.
    pub fn run_norm2_wide_components_for_gate(dir: &Path, sweep: bool) -> Res<()> {
        let tiles = Self::norm2_wide_tiles_for_gate();
        println!(
            "COMPONENT_PROTOCOL door=MEMRA_DSV4_NORM2_WIDE class=same tiles={tiles} block=128 sites=172 repeats_per_cell={REPEATS} warmup={WARMUP} cells=warm,cold"
        );
        let mut comparisons = 0usize;
        for rank in 0..2 {
            let ctx = cudarc::driver::CudaContext::new(rank).map_err(e("norm2 wide context"))?;
            let stream = ctx.new_stream().map_err(e("norm2 wide stream"))?;
            let mut scrub = stream
                .alloc_zeros::<u8>(128 * 1024 * 1024)
                .map_err(e("norm2 wide scrub"))?;
            let sites = load_sites(dir, rank)?;
            for site in &sites {
                let xd = upload_f32(&stream, &site.x)?;
                let wd = upload_f32(&stream, &site.w)?;
                let mut yd = upload_f32(&stream, &vec![f32::from_bits(CANARY); COLS + 2 * GUARD])?;
                let mut pack = stream
                    .clone_htod(&vec![0xa5u8; 2 * COLS + 2 * GUARD])
                    .map_err(e("norm2 wide packed"))?;
                let xp = xd.device_ptr(&stream).0 as *const f32;
                let wp = wd.device_ptr(&stream).0 as *const f32;
                let mut reference: Option<Vec<u8>> = None;
                for cold in [false, true] {
                    let mut times = [Vec::new(), Vec::new()];
                    for repeat in 0..REPEATS {
                        let on = matches!(repeat % 4, 1 | 2);
                        let us = Self::norm2_wide_one_launch(
                            &ctx,
                            &stream,
                            &mut yd,
                            &mut pack,
                            &mut scrub,
                            cold,
                            xp,
                            wp,
                            site.eps,
                            if on { Some(tiles) } else { None },
                        )?;
                        let bytes = Self::norm2_wide_read_outputs(&stream, &yd, &pack)?;
                        comparisons += 1;
                        if let Some(expected) = &reference {
                            if expected != &bytes {
                                return Err(format!(
                                    "norm2 wide raw-bit mismatch rank={} layer={} site={} cold={cold} on={on} repeat={repeat}",
                                    site.rank, site.layer, site.kind
                                ));
                            }
                        } else {
                            reference = Some(bytes);
                        }
                        if repeat >= WARMUP {
                            times[usize::from(on)].push(us);
                        }
                    }
                    let avg = times.map(|v| v.iter().sum::<f32>() / v.len() as f32);
                    println!(
                        "COMPONENT rank={} layer={} site={} cols={COLS} rows=1 cold={cold} tiles={tiles} control_us={} wide_us={} speedup={} control_gb_s={} wide_gb_s_unique={} wide_gb_s_issued={} bits_equal=true output_sha256={:x}",
                        site.rank,
                        site.layer,
                        site.kind,
                        avg[0],
                        avg[1],
                        avg[0] / avg[1],
                        gb_s(unique_bytes(COLS), avg[0]),
                        gb_s(unique_bytes(COLS), avg[1]),
                        gb_s(issued_bytes(COLS, tiles), avg[1]),
                        Sha256::digest(reference.as_ref().unwrap())
                    );
                }
                if sweep && site.layer == 0 && site.kind == 0 {
                    Self::norm2_wide_sweep(
                        &ctx,
                        &stream,
                        &mut yd,
                        &mut pack,
                        &mut scrub,
                        xp,
                        wp,
                        site,
                        reference.as_ref().unwrap(),
                    )?;
                }
            }
        }
        println!(
            "COMPONENT_PASS door=MEMRA_DSV4_NORM2_WIDE sites=172 comparisons={comparisons} bits_equal=true class=same"
        );
        Ok(())
    }

    /// One poisoned, optionally cold, event-bracketed launch. `tiles = None` is
    /// the control symbol, so the control's launch overhead is measured through
    /// the same instrument as the arm.
    #[allow(clippy::too_many_arguments)]
    fn norm2_wide_one_launch(
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        stream: &std::sync::Arc<CudaStream>,
        yd: &mut CudaSlice<f32>,
        pack: &mut CudaSlice<u8>,
        scrub: &mut CudaSlice<u8>,
        cold: bool,
        xp: *const f32,
        wp: *const f32,
        eps: f32,
        tiles: Option<i32>,
    ) -> Res<f32> {
        // Poison both outputs on every arm so a missing write cannot inherit the
        // previous arm's valid bytes and pass the bit check.
        stream
            .memcpy_htod(&vec![f32::from_bits(CANARY); COLS + 2 * GUARD], yd)
            .map_err(e("norm2 wide reset y"))?;
        stream
            .memcpy_htod(&vec![0xa5u8; 2 * COLS + 2 * GUARD], pack)
            .map_err(e("norm2 wide reset pack"))?;
        if cold {
            stream
                .memset_zeros(scrub)
                .map_err(e("norm2 wide cold scrub"))?;
        }
        let yp = unsafe { (yd.device_ptr_mut(stream).0 as *mut f32).add(GUARD) };
        let bp = (pack.device_ptr_mut(stream).0 as usize + GUARD) as *mut c_void;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = ctx.new_event(flags).map_err(e("norm2 wide start"))?;
        let end = ctx.new_event(flags).map_err(e("norm2 wide end"))?;
        // Standalone stream, no graph capture: events bracket actual work.
        start.record(stream).map_err(e("norm2 wide start record"))?;
        unsafe {
            match tiles {
                Some(tiles) => ck(
                    "norm2 wide pack",
                    k::memra_dsv4_norm2_pack_wide(
                        xp,
                        wp,
                        yp,
                        bp,
                        COLS as i32,
                        eps,
                        tiles,
                        sp(stream),
                    ),
                )?,
                None => ck(
                    "norm2 pack",
                    k::memra_dsv4_norm2_pack(xp, wp, yp, bp, COLS as i32, eps, sp(stream)),
                )?,
            }
        }
        end.record(stream).map_err(e("norm2 wide end record"))?;
        stream.synchronize().map_err(e("norm2 wide drain"))?;
        Ok(1000.0 * start.elapsed_ms(&end).map_err(e("norm2 wide elapsed"))?)
    }

    /// Guarded readback of both outputs as one byte string, refusing on canary
    /// damage before the bytes are ever compared.
    fn norm2_wide_read_outputs(
        stream: &std::sync::Arc<CudaStream>,
        yd: &CudaSlice<f32>,
        pack: &CudaSlice<u8>,
    ) -> Res<Vec<u8>> {
        let y: Vec<u32> = dtoh_f32(stream, yd)?.iter().map(|v| v.to_bits()).collect();
        let b = stream.clone_dtoh(pack).map_err(e("norm2 wide read pack"))?;
        if y[..GUARD]
            .iter()
            .chain(&y[GUARD + COLS..])
            .any(|&v| v != CANARY)
        {
            return Err("norm2 wide f32 canary".into());
        }
        if b[..GUARD]
            .iter()
            .chain(&b[GUARD + 2 * COLS..])
            .any(|&v| v != 0xa5)
        {
            return Err("norm2 wide pack canary".into());
        }
        let mut bytes = b;
        for v in &y {
            bytes.extend(v.to_le_bytes());
        }
        Ok(bytes)
    }

    /// Tile sweep on one real operand. Every tile count must reproduce the same
    /// bytes, and `tiles=1` must land on the control: the reduction is redundant
    /// per CTA, so the sweep's asymptote is the reduction floor and the gap
    /// between tiles=1 and that floor is the epilogue this door can actually buy.
    #[allow(clippy::too_many_arguments)]
    fn norm2_wide_sweep(
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        stream: &std::sync::Arc<CudaStream>,
        yd: &mut CudaSlice<f32>,
        pack: &mut CudaSlice<u8>,
        scrub: &mut CudaSlice<u8>,
        xp: *const f32,
        wp: *const f32,
        site: &Site,
        reference: &[u8],
    ) -> Res<()> {
        for cold in [false, true] {
            for tiles in WIDE_TILE_SWEEP {
                let mut times = Vec::new();
                for repeat in 0..REPEATS {
                    let us = Self::norm2_wide_one_launch(
                        ctx,
                        stream,
                        yd,
                        pack,
                        scrub,
                        cold,
                        xp,
                        wp,
                        site.eps,
                        Some(tiles),
                    )?;
                    if Self::norm2_wide_read_outputs(stream, yd, pack)? != reference {
                        return Err(format!("norm2 wide sweep bit mismatch tiles={tiles}"));
                    }
                    if repeat >= WARMUP {
                        times.push(us);
                    }
                }
                let avg = times.iter().sum::<f32>() / times.len() as f32;
                println!(
                    "SWEEP rank={} layer={} site={} cold={cold} tiles={tiles} ctas={tiles} us={avg} gb_s_unique={} gb_s_issued={} bits_equal=true",
                    site.rank,
                    site.layer,
                    site.kind,
                    gb_s(unique_bytes(COLS), avg),
                    gb_s(issued_bytes(COLS, tiles), avg)
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_models_and_sweep_domain() {
        assert_eq!(unique_bytes(COLS), 57344);
        assert_eq!(issued_bytes(COLS, 1), 57344);
        assert_eq!(issued_bytes(COLS, 32), 57344 + 31 * 4 * 4096);
        // Every swept tile count must satisfy the launcher's own contract.
        for tiles in WIDE_TILE_SWEEP {
            assert!((1..=COLS as i32 / 128).contains(&tiles));
            assert_eq!(COLS as i32 % (128 * tiles), 0);
        }
        assert!(WIDE_TILE_SWEEP.contains(&Dsv4Gpu::norm2_wide_tiles_for_gate()));
    }
}
