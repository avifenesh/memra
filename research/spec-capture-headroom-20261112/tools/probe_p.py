#!/usr/bin/env python3
"""probe_p.py ARM_I_WORKER_RS OUT_WORKER_RS: arm P of memra#1221, arm I (main with trace lines)
plus a pool pre-reservation at the end of the cooperative prime, before the first token: for
every boundary capture of the session, allocate the planes the publisher will allocate at the
first burst boundary (per attention layer a K and a V plane of pos * tok_bytes, and the draft
plane), then free them all, so the default pool holds blocks of those sizes when the publisher
asks. A probe of the mechanism, not the shipped form: it runs on the scheduler thread."""
import sys

src = open(sys.argv[1]).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


sub(
    "/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.",
    '''/// memra#1221 probe (arm P): reserve the spec boundary publication's planes in the default
/// pool at the end of the prime, then free them.
fn probe_prewarm(engine: &Engine, sp: &memra_engine::spec::SpecSession) {
    let (r0, u0) = engine.pool_reserved_used();
    let t = Instant::now();
    let mut held: Vec<CudaSlice<u8>> = Vec::new();
    let mut bytes = 0usize;
    let cache = sp.cache_ref();
    for cap in sp.boundary_captures.iter() {
        let pos = cap.pos;
        let mut sizes: Vec<usize> = Vec::new();
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
        for n in sizes {
            if n == 0 {
                continue;
            }
            match engine.alloc_u8_uninit(n) {
                Ok(p) => {
                    bytes += n;
                    held.push(p);
                }
                Err(_) => break,
            }
        }
    }
    let alloc_us = t.elapsed().as_micros();
    let (r1, u1) = engine.pool_reserved_used();
    let planes = held.len();
    drop(held);
    let (r2, u2) = engine.pool_reserved_used();
    if tick_trace_enabled() {
        eprintln!(
            "[specpub-probe] prewarm captures={} planes={planes} bytes={bytes} alloc_us={alloc_us} \\
             total_us={} reserved={r0}->{r1}->{r2} used={u0}->{u1}->{u2}",
            sp.boundary_captures.len(),
            t.elapsed().as_micros(),
        );
    }
}

/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.''',
)
sub(
    """        s.prime_service.finish(walker)?;
        s.prefill_queue.clear();
    }""",
    """        s.prime_service.finish(walker)?;
        s.prefill_queue.clear();
        probe_prewarm(engine, spec);
    }""",
)
open(sys.argv[2], "w").write(src)
