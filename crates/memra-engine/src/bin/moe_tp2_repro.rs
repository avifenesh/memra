//! MOE TP2 REPRO — does the grouped MoE serialize two ranks OUTSIDE the server? (2026-08-28)
//!
//! In the server the grouped MoE's two TP ranks run strictly one after the other at t=4096 (join
//! tracks span_sum 33.90 ms, not span_max 17.90 ms) while overlapping at t=416 and t=43. The
//! effect is reproduced across five builds and is worth ~780 ms of the 4k TTFT — MoE 1661 ms
//! would be ~830 ms if the ranks overlapped, with no kernel change.
//!
//! Eleven mechanisms have been eliminated: kernel tile form, occupancy, tile padding, B
//! double-buffering, register spills, tile-prefix truncation, host router, host CSR build, host
//! issue/allocation, cudarc's peer-copy event ordering, and shared device/context (ranks are
//! ordinal 0 and 1 with distinct contexts). What has been missing is a common time base for when
//! each rank BEGINS executing, which needs a profiler — and nsys cannot see through the server's
//! GPU worker thread but works fine on a standalone binary.
//!
//! So: the smallest program that issues the same two-rank chain. If it serializes here, nsys can
//! be pointed at it and the pieces can be deleted one at a time. If it does NOT serialize here,
//! the cause is something the server adds and the search moves there instead — which is equally
//! useful and is why this reproduces the shape rather than calling the engine's TP entry.
//!
//! usage: moe-tp2-repro [iters]
use cudarc::driver::CudaSlice;
use memra_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let n_expert = 288usize;
    let n_used = 8usize;
    let hidden = 4096usize;
    let local_ff = 640usize;
    // MEMRA_TP2_T: the geometry is an INPUT, not a constant. The server diverges at t=576 /
    // n_active=254 while this harness ran t=4096 / n_active~288, and per-expert GROUP TAILS are
    // exactly what changes with it — a sanitizer pass at the wrong geometry proves nothing about
    // the one that misbehaves.
    let t: usize = std::env::var("MEMRA_TP2_T")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096usize);
    let n_pairs = t * n_used;

    let engines = vec![Engine::new(0)?, Engine::new(1)?];
    println!(
        "ranks: {:?}",
        engines
            .iter()
            .map(|e| e.ctx().ordinal())
            .collect::<Vec<_>>()
    );

    // production-shaped routing, same deterministic tilt as moe-sk-repro
    let route = |p: usize| -> i32 {
        let mut h = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        ((u.powf(1.35) * n_expert as f64) as usize).min(n_expert - 1) as i32
    };
    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
    for p in 0..n_pairs {
        buckets[route(p) as usize].push(p as i32);
    }
    let (mut ex_ids, mut ex_off, mut ex_pairs) = (Vec::new(), vec![0i32], Vec::<i32>::new());
    for (id, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            ex_ids.push(id as i32);
            ex_pairs.extend_from_slice(b);
            ex_off.push(ex_pairs.len() as i32);
        }
    }
    let n_active = ex_ids.len();
    let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
    println!("shape: t={t} n_pairs={n_pairs} n_active={n_active} hidden={hidden} ff={local_ff}");

    // per-rank resident state, built once (the server caches its pointer tables too)
    struct RankState {
        tab_gu: CudaSlice<u64>,
        tab_d: CudaSlice<u64>,
        rb_gu: usize,
        rb_d: usize,
        exi: CudaSlice<i32>,
        exoff: CudaSlice<i32>,
        csr: CudaSlice<i32>,
        z: CudaSlice<f32>,
        act: CudaSlice<f32>,
        _banks: Vec<CudaSlice<u8>>,
    }
    let mut st: Vec<RankState> = Vec::new();
    for e in &engines {
        let mk = |in_f: usize, out_f: usize| -> Result<_, Box<dyn std::error::Error>> {
            let row_bytes = in_f / 64 * 36;
            // REAL BYTES, NOT alloc_u8. An uninitialised/zero bank dequants to zeros, the GEMM
            // computes 0 everywhere, and a sum of zeros is bit-identical REGARDLESS of
            // accumulation order — so every "bit-deterministic, maxdiff 0.0 over 20.9M elements"
            // result this harness produced may have been measuring degenerate data rather than
            // the kernel. MEMRA_TP2_RANDBANK=1 fills it with pseudo-random bytes so the
            // arithmetic is non-trivial and reordering can actually show.
            let bank = if std::env::var("MEMRA_TP2_RANDBANK").as_deref() == Ok("1") {
                let n = n_expert * out_f * row_bytes;
                let mut h: u64 = 0x243F_6A88_85A3_08D3;
                let bytes: Vec<u8> = (0..n)
                    .map(|_| {
                        h ^= h << 13;
                        h ^= h >> 7;
                        h ^= h << 17;
                        (h >> 24) as u8
                    })
                    .collect();
                e.htod_bytes(&bytes)?
            } else {
                e.alloc_u8(n_expert * out_f * row_bytes)?
            };
            let mut tab = vec![0u64; 3 * n_expert];
            {
                use cudarc::driver::DevicePtr;
                let s = e.stream();
                let (p, _g) = bank.device_ptr(&s);
                for ex in 0..n_expert {
                    let a = p + (ex * out_f * row_bytes) as u64;
                    tab[ex] = a;
                    tab[n_expert + ex] = a;
                    tab[2 * n_expert + ex] = a;
                }
            }
            Ok((bank, e.htod_u64(&tab)?, row_bytes))
        };
        let (b1, tab_gu, rb_gu) = mk(hidden, local_ff)?;
        let (b2, tab_d, rb_d) = mk(local_ff, hidden)?;
        st.push(RankState {
            tab_gu,
            tab_d,
            rb_gu,
            rb_d,
            exi: e.htod_i32(&ex_ids)?,
            exoff: e.htod_i32(&ex_off)?,
            csr: e.htod_i32(&csr_tok)?,
            z: e.htod(&vec![0.01f32; t * hidden])?,
            act: e.htod(&vec![0.01f32; n_pairs * local_ff])?,
            _banks: vec![b1, b2],
        });
    }
    for e in &engines {
        e.stream().synchronize()?;
    }

    // One "layer": issue rank0's three grouped GEMMs, then rank1's, then wait both. Exactly the
    // server's shape. Per-rank spans via CU_EVENT_DEFAULT events (new_event(None) would disable
    // timing and return INVALID_HANDLE — the trap that cost two build cycles in the server).
    // MEMRA_TP2_PEERCOPY=1 adds the ONE structural difference between this harness and the
    // server: the server copies z from the ROOT device to each rank every layer, and cudarc's
    // cross-context memcpy_dtod records an event on the SOURCE stream then makes the destination
    // wait on it. Without the copy this harness overlaps; if adding it serializes, the peer
    // dependency is the cause and the earlier pre-staging fix failed only because it hoisted the
    // copy WITHIN a layer while the root stream still carried the previous layer's work.
    let peer_copy = std::env::var("MEMRA_TP2_PEERCOPY").as_deref() == Ok("1");
    // simulate the root's per-layer trunk work sitting in front of the copy
    let root_busy: usize = std::env::var("MEMRA_TP2_ROOT_BUSY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // Copy SIZE is separated from copy ORDERING on purpose. A full-size peer copy costs ~16 ms
    // here (this harness never enables P2P, so it likely routes through host), and that cost
    // makes "rank 1 waited for rank 0, then worked" arithmetically indistinguishable from
    // "rank 1 copied and worked concurrently" — both land near 30 ms. A TINY copy keeps the
    // cross-device event ORDERING while removing the transfer cost, so join separates cleanly:
    // ~span_max means they overlap, ~span_sum means rank 1 was ordered behind rank 0.
    // MEMRA_TP2_LAYERS=N: run N "layers" back to back with a cross-device JOIN after each and
    // NO synchronize between them — the server's actual shape. Until now this harness synced
    // every iteration, which drains both devices and lets the next iteration start clean; that
    // per-iteration drain is a plausible reason the harness overlapped no matter where the copy
    // was placed while the server serialized regardless. If serialization appears only with
    // layers>1, the cause is queue-depth/join chaining across layers, not the copy at all.
    let layers: usize = std::env::var("MEMRA_TP2_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    // MEMRA_TP2_FRESH_ALLOC=1: allocate the per-rank working buffers FRESH every layer, the way
    // the server does (z_r ~67 MB, act ~84 MB, plus gate/up/down outputs — about six large
    // allocations per rank per layer). This harness reuses resident buffers, which is the next
    // untested difference: cudarc takes cuMemAllocAsync, and a stream-ordered allocation that has
    // to grow the pool can block until prior frees retire — indistinguishable from serialization.
    // MEMRA_TP2_DETERMINISM=1: run the SAME grouped MoE twice on IDENTICAL inputs and diff the
    // outputs. The server's prime is nondeterministic — one forward, temperature=0, max_tokens=1,
    // same process, and the argmax lands differently across reps at 400 and 1500 words. That
    // breaks every byte-identity gate on this path and blocks MEMRA_PP_BF16's correctness
    // receipt. This asks whether the grouped MoE is the source, and gives the jitter MAGNITUDE a
    // tolerance band would need. Same rank, same buffers, back to back — nothing varies but the
    // kernel's own execution.
    // MEMRA_TP2_QT: which kernel instantiation to exercise. THIS HARNESS PREVIOUSLY RAN qt=7
    // (QT_NVFP4) AND REPORTED THE GROUPED GEMM BIT-EXACT — while the server runs qt=107
    // (QT_NVFP4_V2, the slot-major bank with its own kq_fetch/kq_store specialisation). Stage
    // checksums in the live server then showed the FIRST grouped GEMM diverging on identical
    // inputs, so the clean harness result was for a kernel production does not use. Reproducing
    // the SHAPE of a call is not reproducing the CALL; the qtype selects the code path and is
    // part of it. v2 is a permutation of v1 with identical row_bytes, so the same synthetic bank
    // exercises either path — only the constant changes.
    let qt: i32 = std::env::var("MEMRA_TP2_QT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(memra_engine::QT_NVFP4);
    let determinism = std::env::var("MEMRA_TP2_DETERMINISM").as_deref() == Ok("1");
    let fresh_alloc = std::env::var("MEMRA_TP2_FRESH_ALLOC").as_deref() == Ok("1");
    let copy_first = std::env::var("MEMRA_TP2_COPY_FIRST").as_deref() == Ok("1");
    let copy_rows: usize = std::env::var("MEMRA_TP2_COPY_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    println!(
        "peer_copy={peer_copy} copy_first={copy_first} root_busy={root_busy} copy_rows={copy_rows}"
    );
    if determinism {
        // MEMRA_TP2_DETERM_CONCURRENT=1: keep rank 1 BUSY while rank 0's determinism reps run.
        // In the server, rank 0's grouped GEMM produces different output from byte-identical
        // inputs (z16/zs/bank/routing all checksum-identical across two calls), while this same
        // kernel is bit-exact here in isolation. Concurrency is the one structural difference
        // between the two contexts — the server has both TP ranks live, this harness runs one
        // rank serially. Stated as a hypothesis: nothing proves a kernel on the OTHER device can
        // change this one's output, and that would be surprising. This is the cheapest way to
        // find out, and a negative result pushes the aliasing/lifetime explanation to the front.
        let concurrent = std::env::var("MEMRA_TP2_DETERM_CONCURRENT").as_deref() == Ok("1");
        println!("determinism arm: concurrent_rank1={concurrent}");
        let e = &engines[0];
        e.bind_runtime_device(0)?;
        let s0 = &st[0];
        let (z16, zs) = e.moe_f16g_act(&s0.z, Some(&s0.csr), hidden, n_pairs)?;
        let mut prev: Option<Vec<f32>> = None;
        let mut worst = 0.0f32;
        let mut differing = 0usize;
        for rep in 0..8 {
            // queue work on rank 1 WITHOUT syncing, so it is executing while rank 0's rep runs
            if concurrent {
                let e1 = &engines[1];
                e1.bind_runtime_device(e1.ctx().ordinal() as i32)?;
                let s1 = &st[1];
                let (z1, zs1) = e1.moe_f16g_act(&s1.z, Some(&s1.csr), hidden, n_pairs)?;
                for _ in 0..3 {
                    let _ = e1.moe_f16_grouped(
                        &s1.tab_gu, 0, n_expert, &s1.exi, &ex_off, &s1.exoff, &z1, &zs1, hidden,
                        local_ff, n_active, n_pairs, qt, s1.rb_gu,
                    )?;
                }
                e.bind_runtime_device(0)?;
            }
            let y = e.moe_f16_grouped(
                &s0.tab_gu, 0, n_expert, &s0.exi, &ex_off, &s0.exoff, &z16, &zs, hidden, local_ff,
                n_active, n_pairs, qt, s0.rb_gu,
            )?;
            e.stream().synchronize()?;
            let cur = e.dtoh(&y)?;
            if let Some(p) = &prev {
                let mut md = 0.0f32;
                let mut n_diff = 0usize;
                for (a, b) in p.iter().zip(cur.iter()) {
                    let d = (a - b).abs();
                    if d > 0.0 {
                        n_diff += 1;
                    }
                    if d > md {
                        md = d;
                    }
                }
                if n_diff > 0 {
                    differing += 1;
                }
                if md > worst {
                    worst = md;
                }
                // NON-DEGENERACY GUARD. A bank of zeros makes every partial product 0, and a
                // sum of zeros is bit-identical in ANY accumulation order — so "BIT-DETERMINISTIC"
                // over an all-zero output proves nothing about the kernel. Print what the
                // comparison actually ran on: if nz==0 / absmax==0 the determinism verdict on this
                // line is VACUOUS and must not be cited as evidence.
                let nz = cur.iter().filter(|v| **v != 0.0).count();
                let absmax = cur.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let n_nan = cur.iter().filter(|v| v.is_nan()).count();
                let verdict = if nz == 0 {
                    "VACUOUS(all-zero output)"
                } else {
                    "live"
                };
                println!(
                    "  rep{rep}: maxdiff={md:.3e} differing_elems={n_diff}/{} | nonzero={nz} absmax={absmax:.4e} nan={n_nan} [{verdict}]",
                    cur.len()
                );
            }
            prev = Some(cur);
        }
        println!(
            "\nDETERMINISM: {} of 7 comparisons differed, worst maxdiff={worst:.3e} -> {}",
            differing,
            if differing == 0 {
                "the grouped MoE is BIT-DETERMINISTIC; the jitter is elsewhere"
            } else {
                "the grouped MoE is NOT deterministic — this is the jitter source"
            }
        );
        return Ok(());
    }
    let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut best_join = f64::MAX;
    let mut best_sum = f64::MAX;
    let mut best_max = f64::MAX;
    for it in 0..iters {
        let mut heads = Vec::new();
        let mut tails = Vec::new();
        let t0 = std::time::Instant::now();
        // COPY_FIRST: issue EVERY rank's peer copy before ANY rank's compute, so the event
        // cudarc records on the source stream sits ahead of rank 0's GEMM chain instead of
        // behind it. This is the fix the server attempt got wrong — there it was hoisted within
        // a layer while the root stream still carried the previous layer's work; here the
        // ordering is isolated and the effect is visible in seconds.
        if peer_copy && copy_first {
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for r in 1..engines.len() {
                let e = &engines[r];
                e.bind_runtime_device(e.ctx().ordinal() as i32)?;
                let n = (copy_rows * hidden).min(t * hidden);
                let (lo, hi) = st.split_at_mut(r);
                let mut dst = hi[0].z.slice_mut(0..n);
                e.stream().memcpy_dtod(&lo[0].z.slice(0..n), &mut dst)?;
            }
        }
        // put work on the ROOT stream first, the way a layer's trunk does in the server
        if root_busy > 0 {
            let e0 = &engines[0];
            e0.bind_runtime_device(0)?;
            for _ in 0..root_busy {
                let s0 = &st[0];
                let (z16, zs) = e0.moe_f16g_act(&s0.z, Some(&s0.csr), hidden, n_pairs)?;
                let _ = e0.moe_f16_grouped(
                    &s0.tab_gu, 0, n_expert, &s0.exi, &ex_off, &s0.exoff, &z16, &zs, hidden,
                    local_ff, n_active, n_pairs, qt, s0.rb_gu,
                )?;
            }
        }
        for (r, e) in engines.iter().enumerate() {
            let h = e.ctx().new_event(flags)?;
            h.record(&e.stream())?;
            e.bind_runtime_device(e.ctx().ordinal() as i32)?;
            if peer_copy && !copy_first && r > 0 {
                // cross-device: source is rank 0's buffer/stream, destination is this rank's.
                // split_at_mut so the source and destination borrows do not overlap.
                let (lo, hi) = st.split_at_mut(r);
                let n = (copy_rows * hidden).min(t * hidden);
                let mut dst = hi[0].z.slice_mut(0..n);
                e.stream().memcpy_dtod(&lo[0].z.slice(0..n), &mut dst)?;
            }
            let s = &st[r];
            // fresh per-layer working buffers, matching the server's allocation pattern
            // uninit, NOT htod: htod builds a 67 MB HOST vector each iteration, and the head
            // event then measures the stream sitting idle through ~50 ms of host allocation —
            // spans exploded to ~100 ms while join stayed at 14.4 ms, which is the signature of
            // measuring the harness rather than the device. uninit is a pure stream-ordered
            // device allocation, which is what the server actually does per layer.
            let (fz, fa);
            let (zsrc, asrc) = if fresh_alloc {
                fz = e.uninit(t * hidden)?;
                fa = e.uninit(n_pairs * local_ff)?;
                (&fz, &fa)
            } else {
                (&s.z, &s.act)
            };
            let (z16, zs) = e.moe_f16g_act(zsrc, Some(&s.csr), hidden, n_pairs)?;
            let (a16, a_s) = e.moe_f16g_act(asrc, None, local_ff, n_pairs)?;
            for (tab, rb, in_f, out_f, act, sc) in [
                (&s.tab_gu, s.rb_gu, hidden, local_ff, &z16, &zs),
                (&s.tab_gu, s.rb_gu, hidden, local_ff, &z16, &zs),
                (&s.tab_d, s.rb_d, local_ff, hidden, &a16, &a_s),
            ] {
                let _ = e.moe_f16_grouped(
                    tab, 0, n_expert, &s.exi, &ex_off, &s.exoff, act, sc, in_f, out_f, n_active,
                    n_pairs, qt, rb,
                )?;
            }
            let tl = e.ctx().new_event(flags)?;
            tl.record(&e.stream())?;
            heads.push(h);
            tails.push(tl);
        }
        // JOIN like the server: pull rank 1's output back to the root cross-device, on the
        // ROOT stream, before the next layer. This is the dependency the harness was missing.
        if layers > 1 {
            let e0 = &engines[0];
            e0.bind_runtime_device(0)?;
            let n = (copy_rows * hidden).min(t * hidden);
            let (lo, hi) = st.split_at_mut(1);
            let mut dst = lo[0].z.slice_mut(0..n);
            e0.stream().memcpy_dtod(&hi[0].z.slice(0..n), &mut dst)?;
        }
        let issue_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        for e in &engines {
            e.stream().synchronize()?;
        }
        let join_ms = t1.elapsed().as_secs_f64() * 1e3;
        let spans: Vec<f32> = heads
            .iter()
            .zip(tails.iter())
            .map(|(h, tl)| h.elapsed_ms(tl).unwrap_or(-1.0))
            .collect();
        if spans.iter().any(|v| *v < 0.0) {
            println!("iter {it}: span query FAILED — result invalid");
            continue;
        }
        let sum: f32 = spans.iter().sum();
        let mx = spans.iter().cloned().fold(0.0f32, f32::max);
        if it >= 2 && join_ms < best_join {
            best_join = join_ms;
            best_sum = sum as f64;
            best_max = mx as f64;
        }
        if it < 3 || it == iters - 1 {
            println!(
                "iter {it}: issue={issue_ms:.2}ms join={join_ms:.2}ms spans={spans:?} \
                 sum={sum:.2} max={mx:.2}"
            );
        }
    }
    // NOTE ON THE VERDICT: span_max is measured from a head event recorded BEFORE any wait, so a
    // rank that waits has the wait inside its span and join ~ span_max even under full
    // serialization. Judge against the NO-COPY baseline join instead, which is the only reading
    // that separates the two: ~baseline means overlapped, ~2x baseline means ordered.
    let verdict = if (best_join - best_sum).abs() < (best_join - best_max).abs() {
        "SERIALIZED"
    } else {
        "OVERLAPPED(by span_max — confirm against the no-copy baseline join)"
    };
    println!(
        "\nBEST(warm): join={best_join:.2}ms span_sum={best_sum:.2}ms span_max={best_max:.2}ms \
         -> {verdict}"
    );
    Ok(())
}
