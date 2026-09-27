//! WP-A day 40 (DAY40.md section 1): the span receipt's device-side form. Cell 1: `d2d_receipt_digest` over pinned host
//! memory through its host pointer (unified addressing), bitwise against the CPU oracle, cached and write-combined, every
//! size and offset. Cell 2: its price at the 27B's and the 9B's span shapes over device sources, over pinned staging one
//! launch per span, and over staging with every launch queued back to back. Cell 3: the CPU oracle's price here.
//! usage: day40-span-receipt-survey <tier_receipt.fatbin>
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, LaunchConfig, PushKernelArg, result, sys};

use memra_tier::conformance::{receipt_digest, receipt_digest_from_lanes};
use std::sync::Arc;
use std::time::Instant;

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + seed * 13 + (i >> 11)) % 253) as u8).collect()
}

fn shapes(name: &str) -> Vec<usize> {
    match name {
        "27B" => [vec![3 << 20; 48], vec![120 << 10; 48]].concat(),
        _ => [vec![2 << 20; 24], vec![96 << 10; 24]].concat(),
    }
}

struct Pinned {
    ptr: *mut u8,
    len: usize,
}
impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe { result::free_host(self.ptr.cast()) }.unwrap();
    }
}
fn pinned(len: usize, flags: u32) -> Pinned {
    Pinned { ptr: unsafe { result::malloc_host(len.max(1), flags) }.unwrap().cast(), len }
}

/// One `d2d_receipt_digest` launch over `n` bytes at `ptr` into the four lanes at `lanes` (zeroed by the caller), the
/// engine's own launch shape (`CudaTransfers::digest_on`).
fn launch(s: &Arc<CudaStream>, f: &cudarc::driver::CudaFunction, ptr: u64, n: u64, lanes: u64) {
    let blocks = n.div_ceil(8).div_ceil(2048).clamp(1, 2048) as u32;
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let mut b = s.launch_builder(f);
    b.arg(&ptr).arg(&n).arg(&lanes);
    unsafe { b.launch(cfg) }.unwrap();
}

fn lanes_of(s: &Arc<CudaStream>, lanes: &cudarc::driver::CudaSlice<u64>, k: usize) -> [u64; 4] {
    let v = s.clone_dtoh(lanes).unwrap();
    [v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]]
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: day40-span-receipt-survey <tier_receipt.fatbin>");
    let ctx = CudaContext::new(0).unwrap();
    unsafe { ctx.disable_event_tracking() };
    let s = ctx.new_stream().unwrap();
    let m = ctx.load_module(cudarc::nvrtc::Ptx::from_binary(std::fs::read(&path).unwrap())).unwrap();
    let f = m.load_function("d2d_receipt_digest").unwrap();
    println!("SURVEY header gpu={} fatbin={path}", ctx.name().unwrap());
    // Cell 1: the UVA read, bitwise.
    let (mut checked, mut bad) = (0, 0);
    for (kind, flags) in [("cached", 0u32), ("write-combined", sys::CU_MEMHOSTALLOC_WRITECOMBINED)] {
        for &len in &[1usize, 7, 8, 9, 4095, 4096, 122_880, (3 << 20) + 3] {
            for off in 0..8usize {
                let buf = pinned(len + off, flags);
                let data = pattern(len + off, len + off);
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr, len + off) };
                let lanes = s.alloc_zeros::<u64>(4).unwrap();
                let (lp, _g) = lanes.device_ptr(&s);
                launch(&s, &f, buf.ptr as u64 + off as u64, len as u64, lp);
                drop(_g);
                s.synchronize().unwrap();
                let got = receipt_digest_from_lanes(lanes_of(&s, &lanes, 0), len as u64);
                checked += 1;
                if got != receipt_digest(&data[off..off + len]) {
                    bad += 1;
                    println!("SURVEY UVA MISMATCH kind={kind} len={len} off={off}");
                }
            }
        }
    }
    println!("SURVEY UVA checked={checked} mismatches={bad} -> {}", if bad == 0 { "BITWISE" } else { "DIFFERS" });
    // Cells 2 and 3: the prices.
    for shape in ["27B", "9B"] {
        let lens = shapes(shape);
        let total: usize = lens.iter().sum();
        let hosts: Vec<Pinned> = lens.iter().map(|&n| pinned(n, 0)).collect();
        let datas: Vec<Vec<u8>> = lens.iter().enumerate().map(|(k, &n)| pattern(n, k)).collect();
        for (h, d) in hosts.iter().zip(&datas) {
            unsafe { std::ptr::copy_nonoverlapping(d.as_ptr(), h.ptr, h.len) };
        }
        let devs: Vec<_> = datas.iter().map(|d| s.clone_htod(d).unwrap()).collect();
        let lanes = s.alloc_zeros::<u64>(4 * lens.len()).unwrap();
        let (lp, _g) = lanes.device_ptr(&s);
        let dptrs: Vec<u64> = devs.iter().map(|d| d.device_ptr(&s).0).collect();
        for arm in ["device", "staging", "staging-queued"] {
            let mut ms = Vec::new();
            let mut ok = true;
            for run in 0..6 {
                // zero the lanes (the kernel adds into them)
                unsafe { sys::cuMemsetD8Async(lp, 0, 32 * lens.len(), s.cu_stream()) }.result().unwrap();
                let e0 = s.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                let t0 = Instant::now();
                for (k, &n) in lens.iter().enumerate() {
                    let src = if arm == "device" { dptrs[k] } else { hosts[k].ptr as u64 };
                    launch(&s, &f, src, n as u64, lp + 32 * k as u64);
                    if arm == "staging" {
                        // one launch per span, each observed before the next is queued
                        s.synchronize().unwrap();
                    }
                }
                let e1 = s.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                e1.synchronize().unwrap();
                let host_ms = t0.elapsed().as_secs_f64() * 1e3;
                let dev_ms = e0.elapsed_ms(&e1).unwrap() as f64;
                if run > 0 {
                    ms.push(if arm == "staging" { host_ms } else { dev_ms });
                }
                for (k, d) in datas.iter().enumerate() {
                    ok &= receipt_digest_from_lanes(lanes_of(&s, &lanes, k), d.len() as u64) == receipt_digest(d);
                }
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "SURVEY PRICE shape={shape} arm={arm} spans={} bytes={total} N=5 ms median={:.3} min={:.3} max={:.3} gbps={:.2} bitwise={ok}",
                lens.len(), ms[2], ms[0], ms[4], total as f64 / ms[2] / 1e6
            );
        }
        let mut ms = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            for d in &hosts {
                let bytes = unsafe { std::slice::from_raw_parts(d.ptr, d.len) };
                std::hint::black_box(receipt_digest(bytes));
            }
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "SURVEY CPU-ORACLE shape={shape} bytes={total} N=5 ms median={:.3} min={:.3} max={:.3} gbps={:.2}",
            ms[2], ms[0], ms[4], total as f64 / ms[2] / 1e6
        );
    }
}
