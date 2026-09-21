//! Hash-speed micro-cell for the HostPrefix contracts door review
//! (`research/spill-c-20260919/HOSTPREFIX-DOOR.md`, `WC-DESTINATIONS.md` item 2): the engine's own
//! `memra_tier::contracts::checksum` (sha2 0.10) over the SAME bytes in three host memory kinds,
//! cached pinned (`cuMemHostAlloc` flags 0, `PinnedHostBuf::new` and `PinnedKind::Cached`),
//! write-combined pinned (`CU_MEMHOSTALLOC_WRITECOMBINED`, `PinnedKind::WriteCombined`) and heap.
//! N passes per kind per order, two orders (cached, wc, heap then heap, wc, cached), one process,
//! one line per pass and one rule line. Diagnostic only: no engine path, no default, no flag.
//! Run ONLY through tools/tier-battery.py (it pins host memory under a CUDA context).
//! usage: hash-micro [--bytes N] [--n N]
use cudarc::driver::{CudaContext, result, sys};
use memra_tier::contracts::checksum;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Pinned {
    ptr: *mut u8,
    len: usize,
}
impl Pinned {
    fn new(len: usize, flags: u32) -> Result<Self> {
        // SAFETY: a live primary context is bound to this thread by main; the pointer is owned
        // here and freed exactly once in Drop.
        let ptr = unsafe { result::malloc_host(len, flags)? }.cast::<u8>();
        Ok(Self { ptr, len })
    }
    fn driver_flags(&self) -> u32 {
        let mut flags: std::ffi::c_uint = u32::MAX;
        // SAFETY: the pointer is a live cuMemHostAlloc allocation owned by self; only `flags`
        // is written.
        unsafe {
            sys::cuMemHostGetFlags(&mut flags, self.ptr.cast())
                .result()
                .unwrap()
        };
        flags
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    fn as_slice(&self) -> &[u8] {
        // SAFETY: as above; the bytes were fully written by `fill` before any read.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from malloc_host and is freed once, here.
        let _ = unsafe { result::free_host(self.ptr.cast()) };
    }
}

fn fill(dst: &mut [u8]) {
    // Fixed xorshift64 stream: every kind holds byte-identical content.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for chunk in dst.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let b = x.to_le_bytes();
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn main() -> Result<()> {
    let mut bytes: usize = 160 * 1024 * 1024;
    let mut n: usize = 5;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bytes" => bytes = args.next().ok_or("--bytes needs a value")?.parse()?,
            "--n" => n = args.next().ok_or("--n needs a value")?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if n == 0 || bytes < 4096 {
        return Err("n must be >= 1 and bytes >= 4096".into());
    }
    let ctx = CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let device = ctx.name()?;
    let t = Instant::now();
    let mut cached = Pinned::new(bytes, 0)?;
    let cached_alloc_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut wc = Pinned::new(bytes, sys::CU_MEMHOSTALLOC_WRITECOMBINED)?;
    let wc_alloc_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let mut heap = vec![0u8; bytes];
    let heap_alloc_ms = t.elapsed().as_secs_f64() * 1e3;
    fill(cached.as_mut_slice());
    fill(wc.as_mut_slice());
    fill(&mut heap);
    let (cached_flags, wc_flags) = (cached.driver_flags(), wc.driver_flags());
    println!(
        "hash-micro device=\"{device}\" bytes={bytes} n_per_order={n} alloc_ms cached={cached_alloc_ms:.3} wc={wc_alloc_ms:.3} heap={heap_alloc_ms:.3} driver_flags cached={cached_flags} wc={wc_flags} wc_bit_cached={} wc_bit_wc={}",
        cached_flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0,
        wc_flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0
    );
    let kinds: [(&str, &[u8]); 3] = [
        ("cached", cached.as_slice()),
        ("wc", wc.as_slice()),
        ("heap", &heap),
    ];
    let orders: [[usize; 3]; 2] = [[0, 1, 2], [2, 1, 0]];
    let mut ms: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    let mut per_order: [[Vec<f64>; 3]; 2] = [[vec![], vec![], vec![]], [vec![], vec![], vec![]]];
    let mut digests = std::collections::BTreeSet::new();
    for (oi, order) in orders.iter().enumerate() {
        for pass in 1..=n {
            for &k in order {
                let (name, slice) = kinds[k];
                let t = Instant::now();
                let d = checksum(slice);
                let dt = t.elapsed().as_secs_f64() * 1e3;
                digests.insert(d);
                ms[k].push(dt);
                per_order[oi][k].push(dt);
                println!(
                    "pass order={} pass={pass} kind={name} ms={dt:.3} gbps={:.3}",
                    oi + 1,
                    bytes as f64 / dt / 1e6
                );
            }
        }
    }
    let gbps = |m: f64| bytes as f64 / m / 1e6;
    let (c, w, h) = (median(&ms[0]), median(&ms[1]), median(&ms[2]));
    let rng = |v: &[f64]| {
        let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        format!("{mn:.3}..{mx:.3}")
    };
    println!(
        "HASH-MICRO rule device=\"{device}\" bytes={bytes} n_per_order={n} pooled={} orders=2 digest_equal={} wc_bit_cached={} wc_bit_wc={} cached_ms={c:.3} wc_ms={w:.3} heap_ms={h:.3} cached_range={} wc_range={} heap_range={} cached_o1={:.3} cached_o2={:.3} wc_o1={:.3} wc_o2={:.3} heap_o1={:.3} heap_o2={:.3} cached_gbps={:.3} wc_gbps={:.3} heap_gbps={:.3} wc_over_cached={:.3} cached_over_heap={:.3} two_hashes_cached_ms={:.3} one_hash_cached_ms={c:.3}",
        2 * n,
        digests.len() == 1,
        cached_flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0,
        wc_flags & sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0,
        rng(&ms[0]),
        rng(&ms[1]),
        rng(&ms[2]),
        median(&per_order[0][0]),
        median(&per_order[1][0]),
        median(&per_order[0][1]),
        median(&per_order[1][1]),
        median(&per_order[0][2]),
        median(&per_order[1][2]),
        gbps(c),
        gbps(w),
        gbps(h),
        w / c,
        c / h,
        2.0 * c
    );
    if digests.len() != 1 {
        return Err("digests differ across memory kinds; the fill or a read is wrong".into());
    }
    Ok(())
}
