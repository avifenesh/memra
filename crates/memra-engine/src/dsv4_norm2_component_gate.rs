//! Live operands for every norm2 site, with standalone bit and timing gates.
use super::*;
use sha2::{Digest, Sha256};
use std::path::Path;

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_words(
    dir: &Path,
    rank: usize,
    layer: usize,
    kind: usize,
    x: &[f32],
    w: &[f32],
    cols: usize,
    rows: usize,
    param: f32,
) -> Res<()> {
    if rank >= 2
        || layer >= 43
        || kind >= 4
        || x.len() != cols * rows
        || (kind < 3 && w.len() != cols)
        || (kind == 3 && !w.is_empty())
    {
        return Err("invalid norm2 capture geometry".into());
    }
    let site = dir.join(format!("rank{rank}-layer{layer:02}-site{kind}"));
    if site.exists() {
        return Ok(());
    }
    std::fs::create_dir(&site).map_err(|e| e.to_string())?;
    for (name, words) in [("x.bin", x), ("w.bin", w)] {
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        std::fs::write(site.join(name), &bytes).map_err(|e| e.to_string())?;
        println!(
            "OPERAND rank={rank} layer={layer} site={kind} name={name} bytes={} sha256={:x}",
            bytes.len(),
            Sha256::digest(&bytes)
        );
    }
    std::fs::write(
        site.join("meta.txt"),
        format!("{cols} {rows} {}\n", param.to_bits()),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
fn read_words(path: &Path) -> Res<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| e.to_string())?;
    if b.len() % 4 != 0 {
        return Err("unaligned operand".into());
    }
    Ok(b.chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect())
}
impl Dsv4Gpu {
    pub fn enable_norm2_components_for_gate(&self, dir: &Path) -> Res<()> {
        if !self.topology.is_tp_ep() || !self.chains_f32 {
            return Err("norm2 capture requires TP/EP f32x".into());
        }
        *self.norm2_component_dir.lock().map_err(|e| e.to_string())? = Some(dir.to_path_buf());
        // Latch last: the forward path checks this before it touches the directory
        // mutex, so an un-armed walk keeps main's dispatch exactly.
        self.norm2_component_capture.store(true, Ordering::Relaxed);
        Ok(())
    }
    pub fn finish_norm2_components_for_gate(&self) -> Res<()> {
        self.norm2_component_capture.store(false, Ordering::Relaxed);
        let dir = self
            .norm2_component_dir
            .lock()
            .map_err(|e| e.to_string())?
            .take()
            .ok_or("norm2 capture not enabled")?;
        for rank in 0..2 {
            for layer in 0..43 {
                for kind in 0..4 {
                    let site = dir.join(format!("rank{rank}-layer{layer:02}-site{kind}"));
                    for file in ["x.bin", "w.bin", "meta.txt"] {
                        if !site.join(file).is_file() {
                            return Err(format!("missing norm2 operand: {site:?}/{file}"));
                        }
                    }
                }
            }
        }
        println!("CAPTURE_PASS sites=344 ranks=2 layers=43 site_types=4");
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_norm2_component(
        &self,
        st: &Stage,
        layer: &LayerDev,
        kind: usize,
        x: &CudaSlice<f32>,
        w: Option<&CudaSlice<f32>>,
        cols: usize,
        param: f32,
    ) -> Res<()> {
        let dir = self
            .norm2_component_dir
            .lock()
            .map_err(|e| e.to_string())?
            .clone();
        if let Some(dir) = dir {
            if st.gpu.ctx.ordinal() != st.dev {
                return Err("norm2 stream-owner mismatch".into());
            }
            let stream = st.gpu.stream();
            let x = stream
                .clone_dtoh(&x.slice(..cols))
                .map_err(e("norm2 input capture"))?;
            let w = stream
                .clone_dtoh(&w.ok_or("norm2 weight missing")?.slice(..cols))
                .map_err(e("norm2 weight capture"))?;
            capture_words(
                &dir,
                st.dev,
                layer.il as usize,
                kind,
                &x,
                &w,
                cols,
                1,
                param,
            )?;
        }
        Ok(())
    }
    pub fn run_norm2_components_for_gate(dir: &Path) -> Res<()> {
        const GUARD: usize = 32;
        const CANARY: u32 = 0x4b123456;
        for rank in 0..2 {
            let ctx = cudarc::driver::CudaContext::new(rank).map_err(e("norm2 context"))?;
            let stream = ctx.new_stream().map_err(e("norm2 stream"))?;
            let mut scrub = stream
                .alloc_zeros::<u8>(128 * 1024 * 1024)
                .map_err(e("norm2 scrub"))?;
            for layer in 0..43 {
                for kind in 0..4 {
                    let site = dir.join(format!("rank{rank}-layer{layer:02}-site{kind}"));
                    let x = read_words(&site.join("x.bin"))?;
                    let w = read_words(&site.join("w.bin"))?;
                    let meta: Vec<usize> = std::fs::read_to_string(site.join("meta.txt"))
                        .map_err(|e| e.to_string())?
                        .split_whitespace()
                        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
                        .collect::<Res<_>>()?;
                    if meta.len() != 3 {
                        return Err("bad norm2 metadata".into());
                    }
                    let (cols, rows, param) = (meta[0], meta[1], f32::from_bits(meta[2] as u32));
                    let n = rows * cols;
                    if x.len() != n
                        || x.iter().chain(&w).any(|v| !v.is_finite())
                        || (kind < 2 && (cols != 4096 || rows != 1 || w.len() != cols))
                        || (kind == 2 && (cols != 2048 || rows != 1 || w.len() != cols))
                        || (kind == 3
                            && (cols != 2048 || !(1..=6).contains(&rows) || !w.is_empty()))
                    {
                        return Err("norm2 operand contract".into());
                    }
                    let xd = upload_f32(&stream, &x)?;
                    let wd = upload_f32(&stream, if w.is_empty() { &[0.0] } else { &w })?;
                    let mut yd = upload_f32(&stream, &vec![f32::from_bits(CANARY); n + 2 * GUARD])?;
                    let mut pack = stream
                        .clone_htod(&vec![0xa5u8; 2 * n + 2 * GUARD])
                        .map_err(e("norm2 packed"))?;
                    let mut codes = stream.alloc_zeros::<u8>(n).map_err(e("norm2 codes"))?;
                    let mut scales = stream
                        .alloc_zeros::<f32>(n / 128)
                        .map_err(e("norm2 scales"))?;
                    let mut rs =
                        upload_f32(&stream, &vec![f32::from_bits(CANARY); rows + 2 * GUARD])?;
                    let mut status = stream
                        .clone_htod(&vec![CANARY as i32; rows + 2 * GUARD])
                        .map_err(e("norm2 status"))?;
                    let xp = xd.device_ptr(&stream).0 as *const f32;
                    let wp = wd.device_ptr(&stream).0 as *const f32;
                    let yp = unsafe { (yd.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
                    let bp = (pack.device_ptr_mut(&stream).0 as usize + GUARD) as *mut c_void;
                    let rsp = unsafe { (rs.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
                    let stp = unsafe { (status.device_ptr_mut(&stream).0 as *mut i32).add(GUARD) };
                    let mut reference: Option<Vec<u8>> = None;
                    for cold in [false, true] {
                        let mut times = [Vec::new(), Vec::new()];
                        for repeat in 0..40 {
                            let on = matches!(repeat % 4, 1 | 2);
                            // Poison each output on every arm so a missing write cannot inherit
                            // the previous arm's valid bytes. Reset and scrub precede timing.
                            stream
                                .memcpy_htod(&vec![f32::from_bits(CANARY); n + 2 * GUARD], &mut yd)
                                .map_err(e("norm2 reset y"))?;
                            stream
                                .memcpy_htod(&vec![0xa5u8; 2 * n + 2 * GUARD], &mut pack)
                                .map_err(e("norm2 reset pack"))?;
                            stream
                                .memcpy_htod(
                                    &vec![f32::from_bits(CANARY); rows + 2 * GUARD],
                                    &mut rs,
                                )
                                .map_err(e("norm2 reset scale"))?;
                            stream
                                .memcpy_htod(&vec![CANARY as i32; rows + 2 * GUARD], &mut status)
                                .map_err(e("norm2 reset status"))?;
                            if cold {
                                stream
                                    .memset_zeros(&mut scrub)
                                    .map_err(e("norm2 cold scrub"))?;
                            }
                            let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                            let start = ctx.new_event(flags).map_err(e("norm2 start"))?;
                            let end = ctx.new_event(flags).map_err(e("norm2 end"))?;
                            // Standalone stream, no graph capture: events bracket actual work.
                            start.record(&stream).map_err(e("norm2 start record"))?;
                            unsafe {
                                match (kind, on) {
                                    (0 | 1, true) => ck(
                                        "norm2 pack",
                                        k::memra_dsv4_norm2_pack(
                                            xp,
                                            wp,
                                            yp,
                                            bp,
                                            cols as i32,
                                            param,
                                            sp(&stream),
                                        ),
                                    )?,
                                    (0 | 1, false) => {
                                        ck(
                                            "norm2 reference norm",
                                            k::memra_dsv4_rmsnorm_f32acc(
                                                xp,
                                                wp,
                                                yp,
                                                1,
                                                cols as i32,
                                                param,
                                                sp(&stream),
                                            ),
                                        )?;
                                        ck(
                                            "norm2 reference pack",
                                            k::memra_dsv4_cvt_bf16(yp, bp, n as i64, sp(&stream)),
                                        )?;
                                        if kind == 0 {
                                            ck(
                                                "norm2 reference KV repack",
                                                k::memra_dsv4_cvt_bf16(
                                                    yp,
                                                    bp,
                                                    n as i64,
                                                    sp(&stream),
                                                ),
                                            )?;
                                        }
                                    }
                                    (2, true) => ck(
                                        "norm2 swiglu pack",
                                        k::memra_dsv4_norm2_swiglu_pack(
                                            xp,
                                            wp,
                                            bp,
                                            cols as i32,
                                            param,
                                            sp(&stream),
                                        ),
                                    )?,
                                    (2, false) => {
                                        ck(
                                            "norm2 reference swiglu",
                                            k::memra_dsv4_swiglu(
                                                xp,
                                                wp,
                                                yp,
                                                1,
                                                cols as i32,
                                                param,
                                                std::ptr::null(),
                                                sp(&stream),
                                            ),
                                        )?;
                                        ck(
                                            "norm2 reference pack",
                                            k::memra_dsv4_cvt_bf16(yp, bp, n as i64, sp(&stream)),
                                        )?;
                                    }
                                    (3, true) => ck(
                                        "norm2 quant half",
                                        k::memra_dsv4_norm2_quant_half(
                                            xp,
                                            bp,
                                            rsp,
                                            stp,
                                            rows as i32,
                                            cols as i32,
                                            sp(&stream),
                                        ),
                                    )?,
                                    (3, false) => {
                                        ck(
                                            "norm2 reference quant",
                                            k::memra_dsv4_act_quant_fp8(
                                                xp,
                                                codes.device_ptr_mut(&stream).0 as *mut c_void,
                                                scales.device_ptr_mut(&stream).0 as *mut f32,
                                                rows as i32,
                                                cols as i32,
                                                sp(&stream),
                                            ),
                                        )?;
                                        ck(
                                            "norm2 reference gather",
                                            k::memra_dsv4_fp8_gather_half(
                                                codes.device_ptr(&stream).0 as *const c_void,
                                                scales.device_ptr(&stream).0 as *const f32,
                                                std::ptr::null(),
                                                bp,
                                                rsp,
                                                stp,
                                                rows as i32,
                                                cols as i32,
                                                sp(&stream),
                                            ),
                                        )?;
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            end.record(&stream).map_err(e("norm2 end record"))?;
                            stream.synchronize().map_err(e("norm2 drain"))?;
                            let us = 1000.0 * start.elapsed_ms(&end).map_err(e("norm2 elapsed"))?;
                            let y: Vec<u32> = dtoh_f32(&stream, &yd)?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect();
                            let b = stream.clone_dtoh(&pack).map_err(e("norm2 read pack"))?;
                            let r: Vec<u32> = dtoh_f32(&stream, &rs)?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect();
                            let status =
                                stream.clone_dtoh(&status).map_err(e("norm2 read status"))?;
                            for (v, len) in [(&y, n), (&r, rows)] {
                                if v[..GUARD]
                                    .iter()
                                    .chain(&v[GUARD + len..])
                                    .any(|&v| v != CANARY)
                                {
                                    return Err("norm2 f32 canary".into());
                                }
                            }
                            if b[..GUARD]
                                .iter()
                                .chain(&b[GUARD + 2 * n..])
                                .any(|&v| v != 0xa5)
                                || status[..GUARD]
                                    .iter()
                                    .chain(&status[GUARD + rows..])
                                    .any(|&v| v != CANARY as i32)
                            {
                                return Err("norm2 pack/status canary".into());
                            }
                            let mut bytes = b;
                            if kind < 2 {
                                for v in &y {
                                    bytes.extend(v.to_le_bytes());
                                }
                            }
                            if kind == 3 {
                                if status[GUARD..GUARD + rows].iter().any(|&v| v != 0) {
                                    return Err("norm2 transport refusal on real operand".into());
                                }
                                for v in &r {
                                    bytes.extend(v.to_le_bytes());
                                }
                                for v in &status {
                                    bytes.extend(v.to_le_bytes());
                                }
                            }
                            if let Some(expected) = &reference {
                                if expected != &bytes {
                                    return Err(format!(
                                        "norm2 raw-bit mismatch rank={rank} layer={layer} site={kind} cold={cold} on={on} repeat={repeat}"
                                    ));
                                }
                            } else {
                                reference = Some(bytes);
                            }
                            if repeat >= 4 {
                                times[usize::from(on)].push(us);
                            }
                        }
                        let avg = times.map(|v| v.iter().sum::<f32>() / v.len() as f32);
                        println!(
                            "COMPONENT rank={rank} layer={layer} site={kind} rows={rows} cols={cols} cold={cold} control_us={} fused_us={} samples_per_arm=18 launches={}/1 bits_equal=true output_sha256={:x}",
                            avg[0],
                            avg[1],
                            if kind == 0 { 3 } else { 2 },
                            Sha256::digest(reference.as_ref().unwrap())
                        );
                    }
                }
            }
        }
        println!(
            "COMPONENT_PASS sites=344 calls=27520 comparisons=27176 launches_removed_per_rank_step=215"
        );
        Ok(())
    }
}
