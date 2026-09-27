//! Hash-speed micro-cell for the HostPrefix contracts door review
//! (`research/spill-c-20260919/HOSTPREFIX-DOOR.md`, `WC-DESTINATIONS.md` item 2): the engine's own
//! `memra_tier::contracts::checksum` (sha2 0.10) over the SAME bytes in three host memory kinds,
//! cached pinned (`cuMemHostAlloc` flags 0, `PinnedHostBuf::new` and `PinnedKind::Cached`),
//! write-combined pinned (`CU_MEMHOSTALLOC_WRITECOMBINED`, `PinnedKind::WriteCombined`) and heap.
//! N passes per kind per order, two orders (cached, wc, heap then heap, wc, cached), one process,
//! one line per pass and one rule line. Diagnostic only: no engine path, no default, no flag.
//! Run ONLY through tools/tier-battery.py (it pins host memory under a CUDA context).
//! `--two-step` (day 33, `DAY33.md`): the write-combined buffer hashed in place (the one
//! sequential single-pass read `checksum` makes: one `Sha256::update` over the slice) against a
//! `memcpy` of the write-combined buffer into a cached pinned buffer followed by the hash of that
//! copy; two orders (wc, wc_copy then wc_copy, wc), N passes per kind per order, one line per pass
//! and one `HASH-MICRO two-step rule` line. Without the flag the output is day 18's, unchanged.
//! usage: hash-micro [--bytes N] [--n N] [--two-step]
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

fn rng(v: &[f64]) -> String {
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    format!("{mn:.3}..{mx:.3}")
}

/// The `--two-step` arm: kind `wc` is `checksum` over the write-combined buffer in place; kind
/// `wc_copy` is `copy_from_slice` of the write-combined buffer into a cached pinned buffer
/// (`cuMemHostAlloc` flags 0) and then `checksum` over that copy, both steps timed separately and
/// summed. The digests of both kinds must agree.
fn two_step(bytes: usize, n: usize, device: &str) -> Result<()> {
    let mut wc = Pinned::new(bytes, sys::CU_MEMHOSTALLOC_WRITECOMBINED)?;
    let mut copy = Pinned::new(bytes, 0)?;
    fill(wc.as_mut_slice());
    copy.as_mut_slice().fill(0);
    let (wc_flags, copy_flags) = (wc.driver_flags(), copy.driver_flags());
    let wc_bit = |f: u32| f & sys::CU_MEMHOSTALLOC_WRITECOMBINED != 0;
    println!(
        "hash-micro two-step device=\"{device}\" bytes={bytes} n_per_order={n} driver_flags wc={wc_flags} copy={copy_flags} wc_bit_wc={} wc_bit_copy={}",
        wc_bit(wc_flags),
        wc_bit(copy_flags)
    );
    let orders: [[usize; 2]; 2] = [[0, 1], [1, 0]];
    let mut ms: [Vec<f64>; 2] = [vec![], vec![]];
    let mut per_order: [[Vec<f64>; 2]; 2] = [[vec![], vec![]], [vec![], vec![]]];
    let (mut memcpy_ms, mut hash_ms) = (vec![], vec![]);
    let mut digests = std::collections::BTreeSet::new();
    let gbps = |m: f64| bytes as f64 / m / 1e6;
    for (oi, order) in orders.iter().enumerate() {
        for pass in 1..=n {
            for &k in order {
                if k == 0 {
                    let t = Instant::now();
                    let d = checksum(wc.as_slice());
                    let dt = t.elapsed().as_secs_f64() * 1e3;
                    digests.insert(d);
                    ms[0].push(dt);
                    per_order[oi][0].push(dt);
                    println!(
                        "pass order={} pass={pass} kind=wc ms={dt:.3} gbps={:.3}",
                        oi + 1,
                        gbps(dt)
                    );
                } else {
                    let t = Instant::now();
                    copy.as_mut_slice().copy_from_slice(wc.as_slice());
                    let m = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    let d = checksum(copy.as_slice());
                    let h = t.elapsed().as_secs_f64() * 1e3;
                    let dt = m + h;
                    digests.insert(d);
                    ms[1].push(dt);
                    per_order[oi][1].push(dt);
                    memcpy_ms.push(m);
                    hash_ms.push(h);
                    println!(
                        "pass order={} pass={pass} kind=wc_copy ms={dt:.3} memcpy_ms={m:.3} hash_ms={h:.3} gbps={:.3}",
                        oi + 1,
                        gbps(dt)
                    );
                }
            }
        }
    }
    let (w, c) = (median(&ms[0]), median(&ms[1]));
    println!(
        "HASH-MICRO two-step rule device=\"{device}\" bytes={bytes} n_per_order={n} pooled={} orders=2 digest_equal={} wc_bit_wc={} wc_bit_copy={} wc_ms={w:.3} wc_copy_ms={c:.3} wc_copy_memcpy_ms={:.3} wc_copy_hash_ms={:.3} wc_range={} wc_copy_range={} wc_copy_memcpy_range={} wc_copy_hash_range={} wc_o1={:.3} wc_o2={:.3} wc_copy_o1={:.3} wc_copy_o2={:.3} wc_gbps={:.3} wc_copy_gbps={:.3} wc_copy_over_wc={:.3}",
        2 * n,
        digests.len() == 1,
        wc_bit(wc_flags),
        wc_bit(copy_flags),
        median(&memcpy_ms),
        median(&hash_ms),
        rng(&ms[0]),
        rng(&ms[1]),
        rng(&memcpy_ms),
        rng(&hash_ms),
        median(&per_order[0][0]),
        median(&per_order[1][0]),
        median(&per_order[0][1]),
        median(&per_order[1][1]),
        gbps(w),
        gbps(c),
        c / w
    );
    if digests.len() != 1 {
        return Err("digests differ between the in-place hash and the copied hash".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut bytes: usize = 160 * 1024 * 1024;
    let mut n: usize = 5;
    let mut two = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bytes" => bytes = args.next().ok_or("--bytes needs a value")?.parse()?,
            "--n" => n = args.next().ok_or("--n needs a value")?.parse()?,
            "--two-step" => two = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if n == 0 || bytes < 4096 {
        return Err("n must be >= 1 and bytes >= 4096".into());
    }
    let ctx = CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let device = ctx.name()?;
    if two {
        return two_step(bytes, n, &device);
    }
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
