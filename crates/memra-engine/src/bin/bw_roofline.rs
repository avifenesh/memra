//! bw-roofline: what this part actually streams, and what the MoE access pattern costs.
//!
//! WHY. Every "percent of the wall" in the glm5 B200 lane is now quoted against 4062 GB/s,
//! taken with `dsv4_hc_collapse_kernel` standing in as a streamer (`MEMRA_HC_BW_PROBE`).
//! That kernel collapses hc planes: one thread per output element, no grid-stride reuse,
//! four SCALAR 32-bit loads per thread. The bytes coalesce (32 lanes x 4 B = one 128 B
//! transaction) but it issues four times the load instructions a 128-bit access needs for
//! the same bytes, so the number it produces is a floor on the part, not a wall. 4062 GB/s
//! is 51% of B200's nominal 8 TB/s on a confirmed full-power 1000 W part at 3996 MHz
//! memory, which is low enough for a plain streamer that the instrument is the first
//! suspect. If the real wall is higher, every efficiency verdict in that lane moves and the
//! 230 tok/s target gets closer, not further.
//!
//! Arms:
//!   * `scalar` reproduces the old probe's ACCESS SHAPE at the same sizes, so the new number
//!     is anchored to the one it replaces rather than floating free.
//!   * `v4` is the same bytes through 128-bit loads.
//!   Both sweep ILP and grid so neither arm loses on a tuning choice the other got right.
//!   * `slabs` reads N disjoint slabs instead of one contiguous run. That is the MoE expert
//!     read: a token touches 9 of 288 experts per layer, each a ~14 MB slab scattered
//!     through a pool far larger than any cache. The default slab set is a WHOLE TOKEN's
//!     worth, 45 layers x 9 experts x 14 MiB = 5.5 GiB, because 9 slabs is 126 MiB and fits
//!     inside B200's 126 MB L2: sized per layer, the arm measures the cache, not the pattern.
//!     The gap between `v4` and `slabs` at the same total bytes is what the layout and
//!     residency levers are actually fighting.
//!
//! Timing is best-of-N wall around a synchronize, which is the same instrument the lane's
//! other bench bins use. Read-only by construction: the accumulator escapes through a
//! comparison that cannot hold, so no write lands in the timed loop.
//!
//! Usage: `bw-roofline [device] [pool_gib] [slab_mib] [nslabs]`.
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use memra_engine::dsv4_ffi as k;
use std::os::raw::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const REPS: usize = 7;

fn main() -> Res<()> {
    let arg = |i: usize, d: usize| -> usize {
        std::env::args()
            .nth(i)
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dev = arg(1, 0);
    let pool_gib = arg(2, 32);
    let slab_mib = arg(3, 14);
    // 45 layers x (8 routed + 1 shared) experts: a WHOLE TOKEN's expert read, 5.5 GiB, which is
    // the size that matters. Nine slabs would be 126 MiB and sit inside B200's 126 MB L2, so a
    // per-layer-sized slab arm measures the cache and not the pattern.
    let nslabs = arg(4, 405);

    let ctx = CudaContext::new(dev)?;
    let stream = ctx.default_stream();
    let pool_floats = pool_gib * 1024 * 1024 * 1024 / 4;
    let x = stream.alloc_zeros::<f32>(pool_floats)?;
    let mut sink = stream.alloc_zeros::<f32>(4)?;
    println!("[bw-roofline] device {dev}: pool {pool_gib} GiB ({pool_floats} f32), best of {REPS}");

    // ---- contiguous arms ----
    let mut best_overall: (f64, String) = (0.0, String::new());
    for (mode, name) in [(0, "scalar32"), (1, "vec128")] {
        for &ilp in &[1i32, 2, 4, 8] {
            for &blocks in &[148i32, 296, 592, 1184] {
                for &threads in &[256i32, 512] {
                    let mut best_us = f64::MAX;
                    for _ in 0..REPS {
                        stream.synchronize()?;
                        let t0 = std::time::Instant::now();
                        // SAFETY: pool and sink are live device allocations on this stream;
                        // the kernel only reads `pool_floats` elements of the first.
                        let rc = unsafe {
                            k::memra_bw_read(
                                x.device_ptr(&stream).0 as *const f32,
                                pool_floats as i64,
                                sink.device_ptr_mut(&stream).0 as *mut f32,
                                mode,
                                ilp,
                                blocks,
                                threads,
                                stream.cu_stream() as *mut c_void,
                            )
                        };
                        assert_eq!(rc, 0, "bw_read rc {rc}");
                        stream.synchronize()?;
                        let us = t0.elapsed().as_secs_f64() * 1e6;
                        if us < best_us {
                            best_us = us;
                        }
                    }
                    let gbs = (pool_floats * 4) as f64 / best_us / 1e3;
                    let tag = format!("{name} ilp{ilp} blocks{blocks} th{threads}");
                    println!("[bw-roofline]  {tag:34} {best_us:9.1} us  {gbs:7.0} GB/s");
                    if gbs > best_overall.0 {
                        best_overall = (gbs, tag);
                    }
                }
            }
        }
    }
    println!(
        "[bw-roofline] BEST CONTIGUOUS {:.0} GB/s ({})",
        best_overall.0, best_overall.1
    );

    // ---- scattered slabs: the MoE expert shape ----
    // Offsets are float4 indices, slab-aligned and spread across the whole pool so the read
    // cannot ride a cache or a single TLB entry.
    let slab_floats = slab_mib * 1024 * 1024 / 4;
    let slab_n4 = slab_floats / 4;
    let pool_n4 = pool_floats / 4;
    if nslabs * slab_floats > pool_floats {
        println!("[bw-roofline] slabs skipped: {nslabs} x {slab_mib} MiB exceeds the pool");
        return Ok(());
    }
    let span = pool_n4 / nslabs;
    let offs: Vec<i64> = (0..nslabs)
        .map(|i| {
            // deterministic spread: slab i sits at a fixed odd stride into its own span
            let base = i * span;
            let jitter = (i.wrapping_mul(0x9E37_79B9) % (span.saturating_sub(slab_n4)).max(1)) & !3;
            (base + jitter.min(span.saturating_sub(slab_n4))) as i64
        })
        .collect();
    let offs_d = stream.clone_htod(&offs)?;
    let bytes = (nslabs * slab_floats * 4) as f64;
    let mut best_slab: (f64, String) = (0.0, String::new());
    for &blocks in &[16i32, 32, 64, 128] {
        for &threads in &[256i32, 512] {
            let mut best_us = f64::MAX;
            for _ in 0..REPS {
                stream.synchronize()?;
                let t0 = std::time::Instant::now();
                // SAFETY: every offset is <= pool_n4 - slab_n4 by construction above, so the
                // kernel reads inside the pool allocation.
                let rc = unsafe {
                    k::memra_bw_read_slabs(
                        x.device_ptr(&stream).0 as *const f32,
                        offs_d.device_ptr(&stream).0 as *const i64,
                        nslabs as i32,
                        slab_floats as i64,
                        sink.device_ptr_mut(&stream).0 as *mut f32,
                        blocks,
                        threads,
                        stream.cu_stream() as *mut c_void,
                    )
                };
                assert_eq!(rc, 0, "bw_read_slabs rc {rc}");
                stream.synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6;
                if us < best_us {
                    best_us = us;
                }
            }
            let gbs = bytes / best_us / 1e3;
            let tag = format!("slabs blocks{blocks} th{threads}");
            println!("[bw-roofline]  {tag:34} {best_us:9.1} us  {gbs:7.0} GB/s");
            if gbs > best_slab.0 {
                best_slab = (gbs, tag);
            }
        }
    }
    println!(
        "[bw-roofline] BEST SLABS ({nslabs} x {slab_mib} MiB = {:.2} GiB) {:.0} GB/s ({}) = {:.0}% of contiguous",
        bytes / (1024.0 * 1024.0 * 1024.0),
        best_slab.0,
        best_slab.1,
        100.0 * best_slab.0 / best_overall.0.max(1.0)
    );
    Ok(())
}
