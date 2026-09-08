//! Only the compressor snapshot and pending-row KV/score pairs use this helper.
use super::*;
use std::sync::Arc;

/// Returns true only when one paired kernel was enqueued. Otherwise preserve the
/// two original ordered driver copies, including cross-stream dependencies.
pub(super) fn copy_pair<
    A: DevicePtr<f32>,
    B: DevicePtr<f32>,
    C: DevicePtrMut<f32>,
    D: DevicePtrMut<f32>,
>(
    stream: &Arc<CudaStream>,
    a: &A,
    b: &B,
    c: &mut C,
    d: &mut D,
    enabled: bool,
) -> Res<bool> {
    if enabled
        && !a.is_empty()
        && a.len() == b.len()
        && c.len() >= a.len()
        && d.len() >= b.len()
        && [a.stream(), b.stream(), c.stream(), d.stream()]
            .iter()
            .all(|s| Arc::ptr_eq(s, stream))
    {
        stream
            .context()
            .bind_to_thread()
            .map_err(e("copy-pair context"))?;
        let result = {
            // Retain cudarc dependency/lifetime guards until AFTER the enqueue.
            let (ap, _ar) = a.device_ptr(stream);
            let (bp, _br) = b.device_ptr(stream);
            let (cp, _cr) = c.device_ptr_mut(stream);
            let (dp, _dr) = d.device_ptr_mut(stream);
            unsafe {
                k::memra_dsv4_compressor_copy_pair(
                    ap as *const f32,
                    bp as *const f32,
                    cp as *mut f32,
                    dp as *mut f32,
                    a.len(),
                    sp(stream),
                )
            }
        };
        if result != 40071 {
            ck("compressor copy pair", result)?;
            return Ok(true);
        }
        // 40071 is a pre-enqueue span/alias refusal. Keep the original semantics.
    }
    stream.memcpy_dtod(a, c).map_err(e("compressor copy kv"))?;
    stream
        .memcpy_dtod(b, d)
        .map_err(e("compressor copy score"))?;
    Ok(false)
}

impl Dsv4Gpu {
    pub fn compressor_paired_copy_enabled(&self) -> bool {
        self.compressor_paired_copy
    }

    /// Actual successful targeted driver calls and paired kernel enqueues.
    pub fn compressor_copy_counts(&self) -> [u64; 2] {
        std::array::from_fn(|i| self.compressor_copy_counts[i].load(Ordering::Relaxed))
    }

    pub fn compressor_copy_pairs_per_step(&self) -> u64 {
        // Every resident compressor performs one snapshot pair and one append pair
        // per single-token step. Indexer compressors use the same body.
        2 * self
            .stages
            .iter()
            .flat_map(|s| &s.layers)
            .map(|l| u64::from(l.cmp.is_some()) + u64::from(l.idx.is_some()))
            .sum::<u64>()
    }

    /// Model-free gate for the exact observed span sizes plus tail/offset cases.
    /// Uses the production helper; full-model enqueues/digests are checked separately.
    pub fn compressor_copy_components_for_gate() -> Res<()> {
        const GUARD: usize = 17;
        const CANARY: u32 = 0x7fa12345;
        let mut cases = 0;
        for device in 0..2 {
            let ctx = cudarc::driver::CudaContext::new(device).map_err(e("copy gate context"))?;
            let stream = ctx.new_stream().map_err(e("copy gate stream"))?;
            for n in [
                1usize, 3, 255, 256, 257, 512, 1024, 2048, 8192, 65535, 65536, 65537,
            ] {
                let input: Vec<f32> = (0..n + 2 * GUARD)
                    .map(|i| {
                        let special = [0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7fa54321];
                        f32::from_bits(if i < 5 {
                            special[i]
                        } else {
                            (i as u32).wrapping_mul(0x9e3779b9)
                        })
                    })
                    .collect();
                let other: Vec<f32> = input
                    .iter()
                    .map(|x| f32::from_bits(x.to_bits() ^ 0xa5a5a5a5))
                    .collect();
                let src0 = upload_f32(&stream, &input)?;
                let src1 = upload_f32(&stream, &other)?;
                for shared_source in [false, true] {
                    for on in [false, true] {
                        let mut dst0 =
                            upload_f32(&stream, &vec![f32::from_bits(CANARY); n + 2 * GUARD])?;
                        let mut dst1 =
                            upload_f32(&stream, &vec![f32::from_bits(CANARY); n + 2 * GUARD])?;
                        let a = src0.slice(0..n);
                        let b = if shared_source {
                            src0.slice(0..n)
                        } else {
                            src1.slice(0..n)
                        };
                        let used = copy_pair(
                            &stream,
                            &a,
                            &b,
                            &mut dst0.slice_mut(GUARD..GUARD + n),
                            &mut dst1.slice_mut(GUARD..GUARD + n),
                            on,
                        )?;
                        assert_eq!(used, on);
                        // A queued driver consumer must observe the copy without a host drain.
                        let mut consumer = stream
                            .alloc_zeros::<f32>(n)
                            .map_err(e("copy gate consumer"))?;
                        stream
                            .memcpy_dtod(&dst0.slice(GUARD..GUARD + n), &mut consumer)
                            .map_err(e("copy gate consume"))?;
                        for (out, want) in [
                            (&dst0, &input),
                            (&dst1, if shared_source { &input } else { &other }),
                        ] {
                            let words = dtoh_f32(&stream, out)?;
                            assert!(
                                words[..GUARD]
                                    .iter()
                                    .chain(&words[GUARD + n..])
                                    .all(|x| x.to_bits() == CANARY)
                            );
                            assert!(
                                words[GUARD..GUARD + n]
                                    .iter()
                                    .zip(&want[..n])
                                    .all(|(x, y)| x.to_bits() == y.to_bits())
                            );
                        }
                        assert!(
                            dtoh_f32(&stream, &consumer)?
                                .iter()
                                .zip(&input[..n])
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                        cases += 1;
                    }
                }
                assert!(
                    dtoh_f32(&stream, &src0)?
                        .iter()
                        .zip(&input)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                assert!(
                    dtoh_f32(&stream, &src1)?
                        .iter()
                        .zip(&other)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
            }
            let mut out = stream
                .alloc_zeros::<f32>(128)
                .map_err(e("copy gate refusal"))?;
            let inp = upload_f32(&stream, &vec![1.0; 128])?;
            let (ip, _ir) = inp.device_ptr(&stream);
            let (op, _or) = out.device_ptr_mut(&stream);
            // Raw launcher refuses write/read, write/write and cross-pair hazards
            // before enqueue. It is not a memmove implementation.
            for (a, b, c, d, n) in [
                (ip, ip, ip, op, 16usize),
                (ip, ip, op, op + 4, 16),
                (ip, op, op, op + 128, 16),
                (ip, ip, ip + 4, op, 16),
                (ip, ip, op + 1, op + 128, 16),
                (ip, ip, op, op + 128, usize::MAX),
            ] {
                let rc = unsafe {
                    k::memra_dsv4_compressor_copy_pair(
                        a as *const f32,
                        b as *const f32,
                        c as *mut f32,
                        d as *mut f32,
                        n,
                        sp(&stream),
                    )
                };
                assert_eq!(rc, 40071);
            }
            assert_eq!(
                unsafe {
                    k::memra_dsv4_compressor_copy_pair(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        sp(&stream),
                    )
                },
                0
            );
            drop(_or);
            drop(_ir);
            assert!(dtoh_f32(&stream, &out)?.iter().all(|x| x.to_bits() == 0));
            assert!(
                dtoh_f32(&stream, &inp)?
                    .iter()
                    .all(|x| x.to_bits() == 1.0f32.to_bits())
            );
            // Different allocation stream: keep cudarc's original dependency path.
            let other_stream = ctx.new_stream().map_err(e("copy gate peer stream"))?;
            let a = upload_f32(&other_stream, &[1.0, 2.0, 3.0])?;
            let b = upload_f32(&stream, &[4.0, 5.0, 6.0])?;
            let mut c = stream
                .alloc_zeros::<f32>(3)
                .map_err(e("copy gate fallback"))?;
            let mut d = stream
                .alloc_zeros::<f32>(3)
                .map_err(e("copy gate fallback"))?;
            assert!(!copy_pair(&stream, &a, &b, &mut c, &mut d, true)?);
            assert_eq!(dtoh_f32(&stream, &c)?, vec![1.0, 2.0, 3.0]);
            assert_eq!(dtoh_f32(&stream, &d)?, vec![4.0, 5.0, 6.0]);
            println!(
                "COMPRESSOR_COPY_COMPONENT device={device} sizes_f32=1,3,255,256,257,512,1024,2048,8192,65535,65536,65537 guard=17 bits_exact=true sources_unchanged=true overlap_refusals=6 cross_stream_fallback=true"
            );
        }
        println!(
            "PASS compressor paired-copy component cases={cases} numeric_class=bitwise_u32_copy"
        );
        Ok(())
    }
}
