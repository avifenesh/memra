//! WP-A day 38 (DAY38.md section 1): the hash-1 survey on one card. Cell 1: G's framed SHA-256 kernel, bitwise against
//! the receipt program and priced (one thread per item, one launch per batch, CUDA events on its own stream). Cell 2:
//! the host's write-combined read rate for the framed SHA-256, direct and after a streaming copy into cached memory.
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg, result, sys};
use sha2::{Digest as _, Sha256};
use std::time::Instant;

fn receipt(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"memra-tier\0v1\0");
    h.update((11u64).to_le_bytes());
    h.update(b"valid-bytes");
    h.update((bytes.len() as u64).to_le_bytes());
    h.update(bytes);
    h.finalize().into()
}

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 31 + seed * 7 + (i >> 9)) % 251) as u8).collect()
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let src = include_str!("kernel.cu");
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        src,
        cudarc::nvrtc::CompileOptions { arch: Some("compute_90"), ..Default::default() },
    )
    .unwrap();
    let m = ctx.load_module(ptx).unwrap();
    let f = m.load_function("framed_sha256").unwrap();
    println!("SURVEY header gpu={}", ctx.name().unwrap());
    // Identity sweep: odd and even sizes, offsets 0..3 into a buffer (the fast path's funnel).
    let mut bad = 0;
    let mut checked = 0;
    for &len in &[0usize, 1, 22, 23, 24, 86, 87, 88, 150, 4096, 61440, 61445, 1 << 20, (1 << 20) + 3] {
        for off in 0..4usize {
            let host = pattern(len + off, len + off);
            let d = stream.clone_htod(&host).unwrap();
            let (dptr, _g) = d.device_ptr(&stream);
            let ptrs = stream.clone_htod(&[dptr + off as u64]).unwrap();
            let lens = stream.clone_htod(&[len as u64]).unwrap();
            let mut out = stream.alloc_zeros::<u8>(32).unwrap();
            let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let one = 1i32;
            let mut b = stream.launch_builder(&f);
            b.arg(&ptrs).arg(&lens).arg(&mut out).arg(&one);
            unsafe { b.launch(cfg) }.unwrap();
            let got = stream.clone_dtoh(&out).unwrap();
            let want = receipt(&host[off..off + len]);
            checked += 1;
            if got[..] != want[..] {
                bad += 1;
                println!("SURVEY IDENTITY MISMATCH len={len} off={off}");
            }
        }
    }
    println!("SURVEY IDENTITY checked={checked} mismatches={bad} -> {}", if bad == 0 { "BITWISE" } else { "DIFFERS" });
    // Price: n items of `len` bytes each, one launch, N=5, events on the stream.
    for &(n, len) in &[(16usize, 61440usize), (32, 61440), (32, 1 << 20), (32, 4 << 20)] {
        let hosts: Vec<Vec<u8>> = (0..n).map(|k| pattern(len, k)).collect();
        let devs: Vec<_> = hosts.iter().map(|h| stream.clone_htod(h).unwrap()).collect();
        let ptr_vals: Vec<u64> = devs.iter().map(|d| d.device_ptr(&stream).0).collect();
        let ptrs = stream.clone_htod(&ptr_vals).unwrap();
        let lens = stream.clone_htod(&vec![len as u64; n]).unwrap();
        let mut out = stream.alloc_zeros::<u8>(32 * n).unwrap();
        let mut ms = Vec::new();
        for run in 0..6 {
            let e0 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            let cfg = LaunchConfig { grid_dim: (((n + 31) / 32) as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let n_i32 = n as i32;
            let mut b = stream.launch_builder(&f);
            b.arg(&ptrs).arg(&lens).arg(&mut out).arg(&n_i32);
            unsafe { b.launch(cfg) }.unwrap();
            let e1 = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
            e1.synchronize().unwrap();
            let t = e0.elapsed_ms(&e1).unwrap();
            if run > 0 {
                ms.push(t as f64); // run 0 is the JIT/warm-up launch
            }
        }
        let got = stream.clone_dtoh(&out).unwrap();
        let ok = (0..n).all(|k| got[32 * k..32 * k + 32] == receipt(&hosts[k])[..]);
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "SURVEY G items={n} bytes_each={len} N=5 ms median={:.3} min={:.3} max={:.3} bitwise={ok}",
            ms[2], ms[0], ms[4]
        );
    }
    // Cell 2: write-combined pinned memory, 1.1 MiB, the framed SHA-256 read directly, and after a streaming copy.
    let len = 1_153_434usize; // about the 9B's 64-token KV entry (1.1 MB)
    for (label, flags) in [("write-combined", sys::CU_MEMHOSTALLOC_WRITECOMBINED), ("cached", 0)] {
        let p = unsafe { result::malloc_host(len, flags) }.unwrap() as *mut u8;
        let data = pattern(len, 99);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, len) };
        let want = receipt(&data);
        let slice = unsafe { std::slice::from_raw_parts(p, len) };
        let mut direct = Vec::new();
        let mut streamed = Vec::new();
        let mut scratch = vec![0u8; len + 64];
        for _ in 0..5 {
            let t = Instant::now();
            let d = receipt(slice);
            direct.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(d, want);
            let t = Instant::now();
            stream_copy(p, scratch.as_mut_ptr(), len);
            let d = receipt(&scratch[..len]);
            streamed.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(d, want);
        }
        direct.sort_by(|a, b| a.partial_cmp(b).unwrap());
        streamed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "SURVEY WC kind={label} bytes={len} N=5 direct_ms median={:.3} min={:.3} max={:.3} streamed_ms median={:.3} min={:.3} max={:.3}",
            direct[2], direct[0], direct[4], streamed[2], streamed[0], streamed[4]
        );
        unsafe { result::free_host(p as *mut _) }.unwrap();
    }
}

/// A streaming copy (SSE4.1 `movntdqa`) from possibly write-combined memory into cached memory: 64-byte groups of
/// four 16-byte non-temporal loads, then ordinary stores; the tail byte by byte.
fn stream_copy(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{__m128i, _mm_storeu_si128, _mm_stream_load_si128};
        let mut i = 0usize;
        let aligned = (src as usize) % 16 == 0;
        if aligned {
            while i + 64 <= len {
                let s = src.add(i) as *const __m128i;
                let a = _mm_stream_load_si128(s);
                let b = _mm_stream_load_si128(s.add(1));
                let c = _mm_stream_load_si128(s.add(2));
                let d = _mm_stream_load_si128(s.add(3));
                let o = dst.add(i) as *mut __m128i;
                _mm_storeu_si128(o, a);
                _mm_storeu_si128(o.add(1), b);
                _mm_storeu_si128(o.add(2), c);
                _mm_storeu_si128(o.add(3), d);
                i += 64;
            }
        }
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
    }
}
