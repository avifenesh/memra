//! WP-A day 39 (DAY39.md section 1): the promote staging fill's price per thread count on this host.
//! `FILL shape=<27B|9B> threads=T N=5 ms median=.. min=.. max=.. gbps=.. bitwise=<bool>`, and with `--gpu`
//! `SPANS shape=.. N=5 ms median=..` (every staging buffer to a device buffer on one stream, between two events).
use cudarc::driver::{CudaContext, result, sys};
use std::time::Instant;

struct Buf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

fn shapes(name: &str) -> Vec<usize> {
    match name {
        // DAY31 1b's fill line: 96 buffers, 156,893,184 B (48 x 3 MiB ssm, 48 x 120 KiB conv).
        "27B" => [vec![3 << 20; 48], vec![120 << 10; 48]].concat(),
        // 48 buffers, 52,690,944 B (24 x 2 MiB, 24 x 96 KiB).
        _ => [vec![2 << 20; 24], vec![96 << 10; 24]].concat(),
    }
}

/// The fill's work list split into `t` contiguous byte shares: (plane index, byte offset, byte length).
fn shares(lens: &[usize], t: usize) -> Vec<Vec<(usize, usize, usize)>> {
    let total: usize = lens.iter().sum();
    let per = total.div_ceil(t);
    let mut out = vec![Vec::new(); t];
    let (mut k, mut used) = (0usize, 0usize);
    for (i, &n) in lens.iter().enumerate() {
        let mut off = 0;
        while off < n {
            let take = (n - off).min(per - used);
            out[k].push((i, off, take));
            off += take;
            used += take;
            if used == per && k + 1 < t {
                k += 1;
                used = 0;
            }
        }
    }
    out
}

fn fill(srcs: &[Vec<f32>], dsts: &[Buf], work: &[Vec<(usize, usize, usize)>]) {
    std::thread::scope(|s| {
        for part in work.iter().skip(1) {
            s.spawn(move || copy_part(srcs, dsts, part));
        }
        copy_part(srcs, dsts, &work[0]);
    });
}

fn copy_part(srcs: &[Vec<f32>], dsts: &[Buf], part: &[(usize, usize, usize)]) {
    for &(i, off, n) in part {
        // SAFETY: disjoint byte ranges of live buffers; a heap Vec and a pinned host allocation never overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(srcs[i].as_ptr().cast::<u8>().add(off), dsts[i].ptr.add(off), n)
        };
    }
}

fn main() {
    let gpu = std::env::args().any(|a| a == "--gpu");
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("FILL header available_parallelism={cores} gpu={gpu}");
    for shape in ["27B", "9B"] {
        let lens = shapes(shape);
        let total: usize = lens.iter().sum();
        let srcs: Vec<Vec<f32>> = lens
            .iter()
            .enumerate()
            .map(|(k, &n)| (0..n / 4).map(|i| (i as f32) * 0.5 + k as f32).collect())
            .collect();
        let dsts: Vec<Buf> = lens
            .iter()
            .map(|&n| Buf { ptr: unsafe { result::malloc_host(n, 0) }.unwrap() as *mut u8, len: n })
            .collect();
        for t in [1usize, 2, 4, 8, 12] {
            let work = shares(&lens, t);
            fill(&srcs, &dsts, &work); // warm
            let mut ms = Vec::new();
            let mut bitwise = true;
            for _ in 0..5 {
                for d in &dsts {
                    unsafe { std::ptr::write_bytes(d.ptr, 0, d.len) };
                }
                let t0 = Instant::now();
                fill(&srcs, &dsts, &work);
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                bitwise &= srcs.iter().zip(&dsts).all(|(s, d)| unsafe {
                    std::slice::from_raw_parts(d.ptr, d.len) == std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), d.len)
                });
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "FILL shape={shape} bytes={total} threads={t} N=5 ms median={:.3} min={:.3} max={:.3} gbps={:.2} bitwise={bitwise}",
                ms[2], ms[0], ms[4], total as f64 / ms[2] / 1e6
            );
        }
        if gpu {
            let dev: Vec<_> = lens.iter().map(|&n| stream.alloc_zeros::<u8>(n).unwrap()).collect();
            let mut ms = Vec::new();
            for run in 0..6 {
                let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                for (d, b) in dev.iter().zip(&dsts) {
                    let (dptr, _g) = cudarc::driver::DevicePtr::device_ptr(d, &stream);
                    unsafe { sys::cuMemcpyHtoDAsync_v2(dptr, b.ptr as *const _, b.len, stream.cu_stream()) }
                        .result()
                        .unwrap();
                }
                let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e1.synchronize().unwrap();
                if run > 0 {
                    ms.push(e0.elapsed_ms(&e1).unwrap() as f64);
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "SPANS shape={shape} bytes={total} N=5 ms median={:.3} min={:.3} max={:.3} gbps={:.2}",
                ms[2], ms[0], ms[4], total as f64 / ms[2] / 1e6
            );
        }
        for d in dsts {
            unsafe { result::free_host(d.ptr as *mut _) }.unwrap();
        }
    }
}
