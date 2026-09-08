//! Gate-only cells on the first live checkpoint HC and Q projection per rank.
//! Scratch allocations and output reads are outside CUDA event timing.
use super::*;

const GUARD: usize = 32;
const CANARY: u32 = 0x4b123456;

fn guarded(stream: &std::sync::Arc<CudaStream>, n: usize) -> Res<CudaSlice<f32>> {
    upload_f32(stream, &vec![f32::from_bits(CANARY); n + 2 * GUARD])
}

fn read_guarded(stream: &std::sync::Arc<CudaStream>, x: &CudaSlice<f32>) -> Res<Vec<u32>> {
    let words: Vec<_> = dtoh_f32(stream, x)?.iter().map(|x| x.to_bits()).collect();
    if words[..GUARD]
        .iter()
        .chain(&words[words.len() - GUARD..])
        .any(|&x| x != CANARY)
    {
        return Err("small-kernel component guard overwritten".into());
    }
    let payload = &words[GUARD..words.len() - GUARD];
    if payload
        .iter()
        .any(|&v| v == CANARY || !f32::from_bits(v).is_finite())
    {
        return Err("small-kernel component unwritten or nonfinite output".into());
    }
    Ok(words)
}

fn events(st: &Stage) -> Res<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)> {
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    Ok((
        st.gpu.ctx.new_event(flags).map_err(e("component event"))?,
        st.gpu.ctx.new_event(flags).map_err(e("component event"))?,
    ))
}

impl Dsv4Gpu {
    pub fn enable_small_kernel_components_for_gate(&self) {
        self.small_kernel_component_mask.store(0, Ordering::Relaxed);
    }

    pub fn small_kernel_components_complete_for_gate(&self) -> bool {
        self.small_kernel_component_mask.load(Ordering::Relaxed) == 15
    }

    pub(super) fn small_component_claim(&self, dev: usize, kind: usize) -> bool {
        if dev >= 2 {
            return false;
        }
        let bit = 1 << (dev * 2 + kind);
        if self.small_kernel_component_mask.load(Ordering::Relaxed) & bit != 0 {
            return false;
        }
        self.small_kernel_component_mask
            .fetch_or(bit, Ordering::Relaxed)
            & bit
            == 0
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn small_component_hc(
        &self,
        st: &Stage,
        x: *const f32,
        mixes: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        d: usize,
        iters: u32,
        eps: f32,
    ) -> Res<()> {
        if d != 4096 || !self.chains_f32 {
            return Err("component requires HC4 f32x D4096".into());
        }
        let stream = st.gpu.stream();
        let original = dtoh_f32(&stream, mixes)?;
        let mut reference = None;
        for repeat in 0..32 {
            let diet = matches!(repeat % 4, 1 | 2);
            let mut m = guarded(&stream, 24)?;
            stream
                .memcpy_htod(&original[..24], &mut m.slice_mut(GUARD..GUARD + 24))
                .map_err(e("component mixes"))?;
            let mut pre = guarded(&stream, 4)?;
            let mut post = guarded(&stream, 4)?;
            let mut comb = guarded(&stream, 16)?;
            let mut y = guarded(&stream, d)?;
            let mp = unsafe { (m.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let pp = unsafe { (pre.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let po = unsafe { (post.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let cp = unsafe { (comb.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let yp = unsafe { (y.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let sc = scale.device_ptr(&stream).0 as *const f32;
            let ba = base.device_ptr(&stream).0 as *const f32;
            let (start, end) = events(st)?;
            start.record(&stream).map_err(e("component start"))?;
            unsafe {
                if diet {
                    ck(
                        "component HC diet",
                        k::memra_dsv4_small_hc_f32_fixed_order(
                            x,
                            mp,
                            sc,
                            ba,
                            pp,
                            po,
                            cp,
                            yp,
                            1,
                            4,
                            d as i32,
                            iters as i32,
                            eps,
                            sp(&stream),
                        ),
                    )?;
                } else {
                    ck(
                        "component rowsq",
                        k::memra_dsv4_rowsq_scale_f32acc(
                            x,
                            mp,
                            1,
                            (4 * d) as i32,
                            24,
                            eps,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "component Sinkhorn",
                        k::memra_dsv4_hc_sinkhorn_m(
                            mp,
                            sc,
                            ba,
                            pp,
                            po,
                            cp,
                            1,
                            4,
                            iters as i32,
                            eps,
                            sp(&stream),
                        ),
                    )?;
                    ck(
                        "component collapse",
                        k::memra_dsv4_hc_collapse(x, pp, yp, 1, 4, d as i32, sp(&stream)),
                    )?;
                }
            }
            end.record(&stream).map_err(e("component end"))?;
            stream.synchronize().map_err(e("component drain"))?;
            let us = 1000.0 * start.elapsed_ms(&end).map_err(e("component timing"))?;
            let mut words = Vec::new();
            for output in [&m, &pre, &post, &comb, &y] {
                words.extend(read_guarded(&stream, output)?);
            }
            if let Some(ref expected) = reference {
                if expected != &words {
                    return Err(format!(
                        "HC component differs rank={} repeat={repeat}",
                        st.dev
                    ));
                }
            } else {
                reference = Some(words);
            }
            println!(
                "COMPONENT kind=hc numeric_class=dsv4_hc_f32_fixed_order rank={} repeat={repeat} diet={diet} us={us:.6} launches={} bits_equal=true canaries=true timing_scope=cuda_event_chain",
                st.dev,
                if diet { 1 } else { 3 }
            );
        }
        Ok(())
    }

    pub(super) fn small_component_norm(
        &self,
        st: &Stage,
        x: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Res<()> {
        if !self.chains_f32 {
            return Err("component requires f32x".into());
        }
        let stream = st.gpu.stream();
        let original = dtoh_f32(&stream, x)?;
        let mut reference = None;
        for repeat in 0..32 {
            let diet = matches!(repeat % 4, 1 | 2);
            let mut xd = guarded(&stream, n)?;
            stream
                .memcpy_htod(&original[..n], &mut xd.slice_mut(GUARD..GUARD + n))
                .map_err(e("component Q input"))?;
            let mut packed = upload_u8(&stream, &vec![0xa5; 2 * n + 2 * GUARD])?;
            let xp = unsafe { (xd.device_ptr_mut(&stream).0 as *mut f32).add(GUARD) };
            let bp = unsafe { (packed.device_ptr_mut(&stream).0 as *mut u8).add(GUARD) };
            let wp = weight.device_ptr(&stream).0 as *const f32;
            let (start, end) = events(st)?;
            start.record(&stream).map_err(e("component start"))?;
            unsafe {
                if diet {
                    ck(
                        "component Q diet",
                        k::memra_dsv4_small_norm_pack_f32_fixed_order(
                            xp,
                            wp,
                            bp.cast(),
                            1,
                            n as i32,
                            eps,
                            sp(&stream),
                        ),
                    )?;
                } else {
                    ck(
                        "component Q norm",
                        k::memra_dsv4_rmsnorm_f32acc(xp, wp, xp, 1, n as i32, eps, sp(&stream)),
                    )?;
                    ck(
                        "component Q pack",
                        k::memra_dsv4_cvt_bf16(xp, bp.cast(), n as i64, sp(&stream)),
                    )?;
                }
            }
            end.record(&stream).map_err(e("component end"))?;
            stream.synchronize().map_err(e("component drain"))?;
            let us = 1000.0 * start.elapsed_ms(&end).map_err(e("component timing"))?;
            let mut words = read_guarded(&stream, &xd)?;
            let mut bytes = vec![0u8; packed.len()];
            stream
                .memcpy_dtoh(&packed, &mut bytes)
                .map_err(e("component packed read"))?;
            stream.synchronize().map_err(e("component packed drain"))?;
            if bytes[..GUARD]
                .iter()
                .chain(&bytes[bytes.len() - GUARD..])
                .any(|&v| v != 0xa5)
            {
                return Err("Q pack guard overwritten".into());
            }
            words.extend(bytes.into_iter().map(u32::from));
            if let Some(ref expected) = reference {
                if expected != &words {
                    return Err(format!(
                        "Q component differs rank={} repeat={repeat}",
                        st.dev
                    ));
                }
            } else {
                reference = Some(words);
            }
            println!(
                "COMPONENT kind=norm_pack numeric_class=dsv4_norm_pack_f32_fixed_order rank={} repeat={repeat} diet={diet} us={us:.6} launches={} bits_equal=true canaries=true timing_scope=cuda_event_chain",
                st.dev,
                if diet { 1 } else { 2 }
            );
        }
        Ok(())
    }
}
