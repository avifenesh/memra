//! Capture each live KV producer once per layer/rank, then replay standalone.
use super::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn write_words(path: PathBuf, row: &[f32]) -> Res<()> {
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}
fn read_words(path: PathBuf) -> Res<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if !bytes.len().is_multiple_of(4) {
        return Err("unaligned operand".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect())
}
impl Dsv4Gpu {
    pub fn enable_norm_components_for_gate(&self, dir: &Path) -> Res<()> {
        if !self.topology.is_tp_ep() || !self.chains_f32 {
            return Err("component requires TP/EP f32x".into());
        }
        *self.norm_component_dir.lock().map_err(|e| e.to_string())? = Some(dir.to_path_buf());
        for seen in &self.norm_component_seen {
            seen.store(0, Ordering::Relaxed);
        }
        self.norm_component_capture.store(true, Ordering::Relaxed);
        Ok(())
    }
    pub fn finish_norm_components_for_gate(&self) -> Res<()> {
        self.norm_component_capture.store(false, Ordering::Relaxed);
        for seen in &self.norm_component_seen {
            if seen.load(Ordering::Relaxed) != (1u64 << 43) - 1 {
                return Err("missing real norm site".into());
            }
        }
        println!("CAPTURE_PASS sites=86 ranks=2 layers=43");
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn capture_norm_component(
        &self,
        st: &Stage,
        layer: &LayerDev,
        kv: &CudaSlice<f32>,
        hd: usize,
        rd: usize,
        eps: f32,
        fc: *const f32,
        positions: &CudaSlice<i32>,
    ) -> Res<()> {
        if st.dev >= 2 || layer.il >= 43 || hd != 512 || rd != 64 {
            return Err("unexpected norm component geometry".into());
        }
        let bit = 1u64 << layer.il;
        if self.norm_component_seen[st.dev].load(Ordering::Relaxed) & bit != 0 {
            return Ok(());
        }
        let stream = st.gpu.stream();
        let input = dtoh_f32(&stream, kv)?;
        let weight = dtoh_f32(&stream, &layer.kv_norm)?;
        let pos = stream.clone_dtoh(positions).map_err(e("norm positions"))?[0];
        if pos <= 0 {
            return Err("norm capture requires nonzero rotary position".into());
        }
        let mut cs = upload_f32(&stream, &[0f32; 64])?;
        // Copy the exact selected rotary row without reconstructing trig values.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                cs.device_ptr_mut(&stream).0,
                fc.add(pos as usize * rd) as u64,
                rd * std::mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .map_err(e("norm rotary capture"))?;
        }
        let cs = dtoh_f32(&stream, &cs)?;
        let dir = self
            .norm_component_dir
            .lock()
            .map_err(|e| e.to_string())?
            .clone()
            .ok_or("missing component directory")?
            .join(format!("rank{}-layer{:02}", st.dev, layer.il));
        std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        write_words(dir.join("x.bin"), &input[..hd])?;
        write_words(dir.join("w.bin"), &weight[..hd])?;
        write_words(dir.join("cs.bin"), &cs)?;
        std::fs::write(
            dir.join("meta.txt"),
            format!(
                "rank={} layer={} ratio={} position={} dim={hd} rd={rd} eps_bits={}\n",
                st.dev,
                layer.il,
                layer.ratio,
                pos,
                eps.to_bits()
            ),
        )
        .map_err(|e| e.to_string())?;
        self.norm_component_seen[st.dev].fetch_or(bit, Ordering::Relaxed);
        Ok(())
    }
    /// Standalone raw-bit and warm/cache-scrubbed timing on all captured sites.
    pub fn run_norm_components_for_gate(dir: &Path) -> Res<()> {
        const GUARD: usize = 32;
        const CANARY: u32 = 0x4b123456;
        for rank in 0..2 {
            let ctx = cudarc::driver::CudaContext::new(rank).map_err(e("component context"))?;
            let stream = ctx.default_stream();
            // 128 MiB written before each cold sample, outside event timing.
            let mut scrub = stream
                .alloc_zeros::<u8>(128 * 1024 * 1024)
                .map_err(e("cold scrub"))?;
            for layer in 0..43 {
                let site = dir.join(format!("rank{rank}-layer{layer:02}"));
                let x = read_words(site.join("x.bin"))?;
                let w = read_words(site.join("w.bin"))?;
                let cs = read_words(site.join("cs.bin"))?;
                if x.len() != 512 || w.len() != 512 || cs.len() != 64 {
                    return Err("bad operand lengths".into());
                }
                if !cs.chunks_exact(2).any(|pair| pair[1] != 0.0)
                    || x.iter().chain(&w).chain(&cs).any(|v| !v.is_finite())
                {
                    return Err("vacuous rotary tape or nonfinite operand".into());
                }
                let meta =
                    std::fs::read_to_string(site.join("meta.txt")).map_err(|e| e.to_string())?;
                let eps = meta
                    .split_whitespace()
                    .find_map(|s| s.strip_prefix("eps_bits="))
                    .ok_or("eps missing")?
                    .parse::<u32>()
                    .map_err(|e| e.to_string())?;
                let wd = upload_f32(&stream, &w)?;
                let cd = upload_f32(&stream, &cs)?;
                let pd = stream
                    .clone_htod(&[0i32])
                    .map_err(e("component position"))?;
                let mut xd = upload_f32(&stream, &vec![f32::from_bits(CANARY); 512 + 2 * GUARD])?;
                let mut reference = None;
                for cold in [false, true] {
                    let mut times = [Vec::new(), Vec::new()];
                    for repeat in 0..40 {
                        let on = matches!(repeat % 4, 1 | 2);
                        stream
                            .memcpy_htod(&x, &mut xd.slice_mut(GUARD..GUARD + 512))
                            .map_err(e("component reset"))?;
                        if cold {
                            stream
                                .memset_zeros(&mut scrub)
                                .map_err(e("component cache scrub"))?;
                        }
                        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
                        let start = ctx.new_event(flags).map_err(e("component start"))?;
                        let end = ctx.new_event(flags).map_err(e("component end"))?;
                        let xp = unsafe { (xd.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
                        start.record(&stream).map_err(e("component start record"))?;
                        unsafe {
                            if on {
                                ck(
                                    "component fused",
                                    k::memra_dsv4_norm_rope_f32_fixed_order(
                                        xp,
                                        wd.device_ptr(&stream).0 as *const f32,
                                        512,
                                        f32::from_bits(eps),
                                        64,
                                        cd.device_ptr(&stream).0 as *const f32,
                                        pd.device_ptr(&stream).0 as *const i32,
                                        sp(&stream),
                                    ),
                                )?;
                            } else {
                                ck(
                                    "component norm",
                                    k::memra_dsv4_rmsnorm_f32acc(
                                        xp,
                                        wd.device_ptr(&stream).0 as *const f32,
                                        xp,
                                        1,
                                        512,
                                        f32::from_bits(eps),
                                        sp(&stream),
                                    ),
                                )?;
                                ck(
                                    "component rope",
                                    k::memra_dsv4_rope(
                                        xp,
                                        1,
                                        1,
                                        512,
                                        64,
                                        cd.device_ptr(&stream).0 as *const f32,
                                        pd.device_ptr(&stream).0 as *const i32,
                                        0,
                                        sp(&stream),
                                    ),
                                )?;
                            }
                        }
                        end.record(&stream).map_err(e("component end record"))?;
                        stream.synchronize().map_err(e("component drain"))?;
                        let us = 1000.0 * start.elapsed_ms(&end).map_err(e("component elapsed"))?;
                        let words: Vec<u32> = dtoh_f32(&stream, &xd)?
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        if words[..GUARD]
                            .iter()
                            .chain(&words[GUARD + 512..])
                            .any(|v| *v != CANARY)
                        {
                            return Err("norm canary overwritten".into());
                        }
                        if words[GUARD..GUARD + 512]
                            .iter()
                            .any(|v| !f32::from_bits(*v).is_finite())
                        {
                            return Err("nonfinite norm output".into());
                        }
                        if let Some(ref expected) = reference {
                            if expected != &words {
                                return Err(format!(
                                    "norm bit mismatch rank={rank} layer={layer} cold={cold} on={on}"
                                ));
                            }
                        } else {
                            reference = Some(words);
                        }
                        // First ABBA quartet is untimed warm-up; every call is bit checked.
                        if repeat >= 4 {
                            times[usize::from(on)].push(us);
                        }
                    }
                    let avg = times.map(|v| v.iter().sum::<f32>() / v.len() as f32);
                    let mut h = Sha256::new();
                    for w in reference.as_ref().unwrap() {
                        h.update(w.to_le_bytes());
                    }
                    println!(
                        "COMPONENT rank={rank} layer={layer} cold={cold} control_us={} fused_us={} samples_per_arm=18 launches=2/1 bits_equal=true output_sha256={:x}",
                        avg[0],
                        avg[1],
                        h.finalize()
                    );
                }
            }
        }
        println!(
            "COMPONENT_PASS sites=86 calls=6880 comparisons=6794 launches_removed_per_rank_step=43"
        );
        Ok(())
    }
}
