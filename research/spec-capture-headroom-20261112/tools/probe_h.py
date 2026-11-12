#!/usr/bin/env python3
"""probe_h.py ARM_I_WORKER_RS OUT_WORKER_RS: arm H of memra#1221, arm I (main with trace lines)
plus the pool reservation off the scheduler thread: at the entry of a burst whose session holds
boundary captures (the cold prime has just finished) the planes the publisher will allocate are
sized; at the burst's second commit hook (after round 1) a helper thread allocates them from the
default pool on its own stream and frees them. The publisher is main's."""
import sys

src = open(sys.argv[1]).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


sub(
    "/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.",
    '''/// memra#1221 (arm H): the byte sizes of the planes the spec boundary publication will
/// allocate for this session's pending captures, in the publisher's order.
fn spec_boundary_plane_sizes(sp: &memra_engine::spec::SpecSession) -> Vec<usize> {
    let mut sizes: Vec<usize> = Vec::new();
    let cache = sp.cache_ref();
    for cap in sp.boundary_captures.iter() {
        let pos = cap.pos;
        for l in cache.kv.iter().flatten() {
            if l.len == 0 || l.len < pos {
                continue;
            }
            sizes.push(pos * l.k_tok_bytes);
            sizes.push(pos * l.v_tok_bytes);
        }
        if let Some((_, _, kt, vt)) = sp.draft_plane_ref() {
            sizes.push(pos * kt);
            sizes.push(pos * vt);
        }
    }
    sizes.retain(|&n| n > 0);
    sizes
}

/// memra#1221 (arm H): reserve `sizes` in the default pool from a helper thread, then free them.
fn spec_boundary_pool_reserve(engine: &Engine, sizes: Vec<usize>) {
    if sizes.is_empty() {
        return;
    }
    let trace = tick_trace_enabled();
    if trace {
        let (r, u) = engine.pool_reserved_used();
        eprintln!(
            "[specpub-probe] reserve-start planes={} bytes={} reserved={r} used={u} cached={} \\
             mem_available={:?}",
            sizes.len(),
            sizes.iter().sum::<usize>(),
            r.saturating_sub(u),
            memra_engine::unified_memory::host_mem_available_bytes(),
        );
    }
    let ctx = engine.ctx().clone();
    let _ = std::thread::Builder::new()
        .name("memra-pool-reserve".into())
        .spawn(move || {
            let t = Instant::now();
            let Ok(stream) = ctx.new_stream() else {
                return;
            };
            let mut held: Vec<CudaSlice<u8>> = Vec::with_capacity(sizes.len());
            let mut longest = 0u128;
            for n in sizes {
                let t1 = Instant::now();
                // SAFETY: the planes are never read; they are freed below.
                match unsafe { stream.alloc::<u8>(n) } {
                    Ok(p) => held.push(p),
                    Err(_) => break,
                }
                longest = longest.max(t1.elapsed().as_micros());
            }
            let alloc_us = t.elapsed().as_micros();
            let planes = held.len();
            drop(held);
            let _ = stream.synchronize();
            if trace {
                eprintln!(
                    "[specpub-probe] reserve-done planes={planes} alloc_us={alloc_us} \\
                     longest_us={longest} total_us={}",
                    t.elapsed().as_micros()
                );
            }
        });
}

/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.''',
)
sub(
    """        let mut emit = SpecBurstEmit::new(lm, s, plan.request_room);
        let spec = s.spec.as_mut().expect("checked above");
        let mut flush_cb = |slice: &[u32]| -> bool {
            emit.flush(""",
    """        let mut emit = SpecBurstEmit::new(lm, s, plan.request_room);
        let spec = s.spec.as_mut().expect("checked above");
        let mut reserve = Some(spec_boundary_plane_sizes(spec));
        let mut commits = 0usize;
        let mut flush_cb = |slice: &[u32]| -> bool {
            commits += 1;
            if commits == 2 {
                if let Some(sizes) = reserve.take() {
                    spec_boundary_pool_reserve(engine, sizes);
                }
            }
            emit.flush(""",
)
open(sys.argv[2], "w").write(src)
