#!/usr/bin/env python3
"""probe_i.py IN_WORKER_RS OUT_WORKER_RS: arm I of memra#1221, main's publisher with trace lines
only (MEMRA_TICK_TRACE=1). Per spec boundary publication it prints the default pool's reserved
and used bytes at entry and exit, the time and the pool growth of every plane allocation, and
the time of every copy launch. No allocation, copy or order of calls changes."""
import sys

src = open(sys.argv[1]).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


# the timed allocator and copy, used only by the publisher
sub(
    "/// One plane of a spec boundary publication:",
    '''/// memra#1221 probe (arm I): [`prefix_plane_alloc`] with a trace line.
fn probe_plane_alloc(
    engine: &Engine,
    bytes: usize,
    what: &str,
) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
    if !tick_trace_enabled() {
        return prefix_plane_alloc(engine, bytes);
    }
    let (r0, _) = engine.pool_reserved_used();
    let t = Instant::now();
    let out = prefix_plane_alloc(engine, bytes);
    let us = t.elapsed().as_micros();
    let (r1, u1) = engine.pool_reserved_used();
    eprintln!(
        "[specpub-probe] alloc {what} bytes={bytes} us={us} reserved_before={r0} \\
         reserved_after={r1} used_after={u1}",
    );
    out
}

/// One plane of a spec boundary publication:''',
)
sub(
    """    #[cfg(test)]
    if spec_plane_copy_fault() {
        return false;
    }
    bytes == 0 || engine.copy_u8_into(dst, 0, src, bytes).is_ok()""",
    """    #[cfg(test)]
    if spec_plane_copy_fault() {
        return false;
    }
    let t = Instant::now();
    let ok = bytes == 0 || engine.copy_u8_into(dst, 0, src, bytes).is_ok();
    if tick_trace_enabled() {
        eprintln!(
            "[specpub-probe] copy bytes={bytes} us={}",
            t.elapsed().as_micros()
        );
    }
    ok""",
)
sub(
    """                let (Ok(mut k), Ok(mut v)) = (
                    prefix_plane_alloc(engine, kb),
                    prefix_plane_alloc(engine, vb),
                ) else {
                    return; // alloc pressure""",
    """                let (Ok(mut k), Ok(mut v)) = (
                    probe_plane_alloc(engine, kb, "trunk-k"),
                    probe_plane_alloc(engine, vb, "trunk-v"),
                ) else {
                    return; // alloc pressure""",
)
sub(
    """        let (Ok(mut k), Ok(mut v)) = (
            prefix_plane_alloc(engine, kb),
            prefix_plane_alloc(engine, vb),
        ) else {
            return None; // alloc pressure: trunk-only entry""",
    """        let (Ok(mut k), Ok(mut v)) = (
            probe_plane_alloc(engine, kb, "draft-k"),
            probe_plane_alloc(engine, vb, "draft-v"),
        ) else {
            return None; // alloc pressure: trunk-only entry""",
)
# entry and exit readings
sub(
    """    let pos = cap.pos;
    if pos == 0 || pos > committed.len() || cap.logits.is_empty() {
        return;
    }
    if cache.has_swa_ring() {
        return;
    }""",
    """    let pos = cap.pos;
    if pos == 0 || pos > committed.len() || cap.logits.is_empty() {
        return;
    }
    if cache.has_swa_ring() {
        return;
    }
    let probe_t = Instant::now();
    if tick_trace_enabled() {
        let (r, u) = engine.pool_reserved_used();
        eprintln!(
            "[specpub-probe] enter why={why} pos={pos} reserved={r} used={u} cached={}",
            r.saturating_sub(u)
        );
    }""",
)
sub(
    """    trace_prefix_entry_state(engine, &e, e.pos, "spec-snapshot", why);
    px.insert_demoting(pool_key, e, why, engine, hpx);
}""",
    """    trace_prefix_entry_state(engine, &e, e.pos, "spec-snapshot", why);
    if tick_trace_enabled() {
        let (r, u) = engine.pool_reserved_used();
        eprintln!(
            "[specpub-probe] planes-done why={why} us={} reserved={r} used={u}",
            probe_t.elapsed().as_micros()
        );
    }
    px.insert_demoting(pool_key, e, why, engine, hpx);
    if tick_trace_enabled() {
        let (r, u) = engine.pool_reserved_used();
        eprintln!(
            "[specpub-probe] exit why={why} us={} reserved={r} used={u}",
            probe_t.elapsed().as_micros()
        );
    }
}""",
)
open(sys.argv[2], "w").write(src)
