#!/usr/bin/env python3
"""DIAG memra#1258 review round 1 (diagnostic builds only, never committed): the allocation ring.
Run after diag-worker.py. Adds to the engine's lib.rs a ring of the last 256 funnel allocations
(`memra_engine::diag_ring_dump`), and to worker.rs: a dump-and-clear after each DSpark prime
chunk, a dump before the next chunk (so that dump is the gap between two calls of one prime),
and a snapshot plus dump when a step runs out of memory.
usage: diag-ring.py LIB_RS WORKER_RS   (idempotent)"""
import sys

lib, worker = sys.argv[1], sys.argv[2]


def patch(path, subs, marker):
    s = open(path).read()
    if marker in s:
        print("already patched", path)
        return
    for old, new in subs:
        assert s.count(old) == 1, (path, s.count(old), old[:80])
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(
    lib,
    [
        (
            '''#[track_caller]
pub(crate) fn alloc_trace_hit(bytes: usize) {
    if alloc_trace_on() {
''',
            '''/// DIAG memra#1258 (diagnostic build only, never merged): the last 256 funnel allocations.
type DiagEntry = (u64, usize, &'static str, u32, u128);
static DIAG_RING: std::sync::Mutex<std::collections::VecDeque<DiagEntry>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static DIAG_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn diag_t0() -> &'static std::time::Instant {
    static T: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T.get_or_init(std::time::Instant::now)
}
/// DIAG memra#1258: dump the allocation ring (oldest first), then clear it.
pub fn diag_ring_dump(tag: &str) {
    let mut ring = DIAG_RING.lock().unwrap();
    eprintln!("[diag1255-ring] {tag}: {} entries (last is the newest)", ring.len());
    let mut sum = 0usize;
    for (seq, bytes, file, line, t_us) in ring.iter() {
        sum += *bytes;
        eprintln!(
            "[diag1255-ring] seq={seq} t_ms={:.3} bytes={bytes} at {file}:{line}",
            *t_us as f64 / 1e3
        );
    }
    eprintln!("[diag1255-ring] {tag}: sum {sum} bytes");
    ring.clear();
}
/// DIAG memra#1258: clear the allocation ring without printing it.
pub fn diag_ring_clear() {
    DIAG_RING.lock().unwrap().clear();
}

#[track_caller]
pub(crate) fn alloc_trace_hit(bytes: usize) {
    {
        let loc = std::panic::Location::caller();
        let seq = DIAG_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = diag_t0().elapsed().as_micros();
        let mut ring = DIAG_RING.lock().unwrap();
        if ring.len() >= 256 {
            ring.pop_front();
        }
        ring.push_back((seq, bytes, loc.file(), loc.line(), t));
    }
    if alloc_trace_on() {
''',
        )
    ],
    "DIAG_RING",
)
patch(
    worker,
    [
        (
            '''        diag1255_snap(engine, &format!("dspark-prime-chunk-before id={}", s.request_id));
''',
            '''        diag1255_snap(engine, &format!("dspark-prime-chunk-before id={}", s.request_id));
        if std::env::var("MEMRA_DIAG_RING").as_deref() == Ok("1") {
            // What was allocated since the last chunk of a prime ended (the gap between calls).
            memra_engine::diag_ring_dump(&format!("gap-before-chunk id={}", s.request_id));
        }
''',
        ),
        (
            '''            diag1255_snap(engine, &format!("dspark-prime-chunk-after id={}", s.request_id));
''',
            '''            diag1255_snap(engine, &format!("dspark-prime-chunk-after id={}", s.request_id));
            memra_engine::diag_ring_clear();
''',
        ),
        (
            '''                            eprintln!(
                                "[admit-oom] step OOM parked session back to queue \\
''',
            '''                            diag1255_snap(&engine, &format!("step-oom id={}", s.request_id));
                            memra_engine::diag_ring_dump("step-oom");
                            eprintln!(
                                "[admit-oom] step OOM parked session back to queue \\
''',
        ),
    ],
    "diag_ring_dump",
)
