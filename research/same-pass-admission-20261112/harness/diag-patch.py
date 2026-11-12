import os, re, sys
p=os.path.expanduser('~/.cache/memra-1255/diag/crates/memra-server/src/worker.rs')
s=open(p).read()
def rep(old,new,count=1):
    global s
    n=s.count(old)
    assert n==count, (old[:80], n)
    s=s.replace(old,new)
# helper
rep("fn effective_free_bytes(engine: &Engine) -> Option<(usize, usize)> {",
"""/// DIAG memra#1255 (diagnostic build only): one device reading line.
fn diag1255_snap(engine: &Engine, tag: &str) {
    let free = engine.ctx().mem_get_info().map(|(f, _)| f).unwrap_or(0);
    let (res, used) = engine.pool_reserved_used();
    let eff = free + res.saturating_sub(used);
    static LAST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    LAST.store(eff, std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "[diag1255] {tag}: driver_free={}MB pool_res={}MB pool_used={}MB pool_unused={}MB eff_free={}MB",
        free >> 20, res >> 20, used >> 20, res.saturating_sub(used) >> 20, eff >> 20
    );
}
/// DIAG memra#1255: a tick-phase reading, printed only when effective free moved by 32 MB or more.
fn diag1255_tick(engine: &Engine, tag: &str) {
    let free = engine.ctx().mem_get_info().map(|(f, _)| f).unwrap_or(0);
    let (res, used) = engine.pool_reserved_used();
    let eff = free + res.saturating_sub(used);
    static LAST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let last = LAST.swap(eff, std::sync::atomic::Ordering::Relaxed);
    if eff.abs_diff(last) >= 32 << 20 {
        eprintln!(
            "[diag1255] tick {tag}: driver_free={}MB pool_res={}MB pool_used={}MB eff_free={}MB (moved {}MB)",
            free >> 20, res >> 20, used >> 20, eff >> 20, (eff as i64 - last as i64) >> 20
        );
    }
}

fn effective_free_bytes(engine: &Engine) -> Option<(usize, usize)> {""")
# gate reading at admit
rep("""                    // ADMIT RECEIPT (memra#680): armed, every arrival this gate lets through is a""",
"""                    eprintln!(
                        "[diag1255] gate-pass id={} prompt={prompt_len} path={} cost={}MB draft={}MB reserve={}MB required={}MB required_eager={}MB eager_arm={eager_arm} booked={}MB limiting_free={}MB active={}",
                        req.request_id,
                        if estimate_spec { "spec" } else { "plain" },
                        cost >> 20, draft_state_bytes >> 20, reserve >> 20, required >> 20,
                        required_eager >> 20, pending_booked >> 20,
                        headroom.limiting_free_bytes() >> 20, active.len(),
                    );
                    // ADMIT RECEIPT (memra#680): armed, every arrival this gate lets through is a""")
rep("""            ensure_driver_headroom(&engine, &loaded, "prime");
""","""            ensure_driver_headroom(&engine, &loaded, "prime");
            diag1255_snap(&engine, "after-admit-trim");
""")
rep("""                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    active.push(s);""","""                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    diag1255_snap(&engine, &format!("after-admit id={} spec={} dspark_on={} cache={} dspark_sess={}", s.request_id, s.spec.is_some(), s.dspark_on, s.cache.is_some(), s.dspark.is_some()));
                    active.push(s);""")
rep("""        // 3. The tick. Three phases:""","""        diag1255_tick(&engine, "start");
        // 3. The tick. Three phases:""")
# step oom
rep("""                            eprintln!(
                                "[admit-oom] step OOM parked session back to queue \\""","""                            diag1255_snap(&engine, &format!("step-oom id={}", s.request_id));
                            memra_engine::diag_ring_dump("step-oom");
                            eprintln!(
                                "[admit-oom] step OOM parked session back to queue \\""")
# dspark prime start
rep("""            s.dspark_prime = Some(match s.dspark.take() {""","""            diag1255_snap(engine, &format!("dspark-prime-start-before id={} rows={}", s.request_id, tokens.len()));
            s.dspark_prime = Some(match s.dspark.take() {""")
rep("""        let mut walker = lm.model.dspark_prime_walker(engine, d, &mut s.dspark_prime);
        if !s.prime_service.advance(&mut walker)? {
            return Ok(true);
        }
        s.dspark = Some(s.prime_service.finish(walker)?);""","""        diag1255_snap(engine, &format!("dspark-prime-chunk-before id={}", s.request_id));
        let mut walker = lm.model.dspark_prime_walker(engine, d, &mut s.dspark_prime);
        if !s.prime_service.advance(&mut walker)? {
            diag1255_snap(engine, &format!("dspark-prime-chunk-after id={}", s.request_id));
            return Ok(true);
        }
        s.dspark = Some(s.prime_service.finish(walker)?);
        diag1255_snap(engine, &format!("dspark-prime-done id={}", s.request_id));""")
open(p,'w').write(s)
print("ok")
