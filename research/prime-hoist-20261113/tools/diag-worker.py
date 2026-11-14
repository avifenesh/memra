#!/usr/bin/env python3
"""DIAG memra#1258 (diagnostic builds only, never committed): add the [diag1255] lines the
harnesses read to a worker.rs on the 202e1411fe base: the admission gate-pass line
(b-sweep.py), the device snapshots around each DSpark prime chunk (b-spread.py), and the optional
pool drain before a cold prime (MEMRA_DIAG_TRIM_BEFORE_PRIME=1, the sweeps' drained mode).
usage: diag-worker.py WORKER_RS   (idempotent)"""
import sys

p = sys.argv[1]
s = open(p).read()
if "[diag1255] gate-pass" in s:
    print("already patched")
    sys.exit(0)


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


sub('''                        n_drafter_refusals += 1;
                        req.drafter_refused = true;
                        queue.push_front(req);
                        continue;
                    }
                    // ADMIT RECEIPT (memra#680)''',
    '''                        n_drafter_refusals += 1;
                        req.drafter_refused = true;
                        queue.push_front(req);
                        continue;
                    }
                    eprintln!(
                        "[diag1255] gate-pass id={} prompt={prompt_len} path={} cost={}MB draft={}MB reserve={}MB required={}MB required_eager={}MB eager_arm={eager_arm} booked={}MB limiting_free={}MB active={}",
                        req.request_id,
                        if estimate_spec { "spec" } else { "plain" },
                        cost >> 20, draft_state_bytes >> 20, reserve >> 20, required >> 20,
                        required_eager >> 20, pending_booked >> 20,
                        headroom.limiting_free_bytes() >> 20, active.len(),
                    );
                    // ADMIT RECEIPT (memra#680)''')
sub('''fn effective_free_bytes(engine: &Engine) -> Option<(usize, usize)> {''',
    '''/// DIAG memra#1255 (diagnostic build only): one device reading line.
fn diag1255_snap(engine: &Engine, tag: &str) {
    let free = engine.ctx().mem_get_info().map(|(f, _)| f).unwrap_or(0);
    let (res, used) = engine.pool_reserved_used();
    let eff = free + res.saturating_sub(used);
    eprintln!(
        "[diag1255] {tag}: driver_free={}MB pool_res={}MB pool_used={}MB pool_unused={}MB eff_free={}MB",
        free >> 20, res >> 20, used >> 20, res.saturating_sub(used) >> 20, eff >> 20
    );
}

fn effective_free_bytes(engine: &Engine) -> Option<(usize, usize)> {''')
sub('''            s.dspark_prime = Some(match s.dspark.take() {''',
    '''            if std::env::var("MEMRA_DIAG_TRIM_BEFORE_PRIME").as_deref() == Ok("1") {
                // DIAG memra#1258 (diagnostic build only): drain the pool's cached blocks to the
                // driver before the cold prime.
                let _ = engine.stream().synchronize();
                let (_, used) = engine.pool_reserved_used();
                let released = engine.pool_trim_to(used);
                eprintln!("[diag1255] trim-before-prime released={}MB", released >> 20);
            }
            diag1255_snap(engine, &format!("dspark-prime-start-before id={} rows={}", s.request_id, tokens.len()));
            s.dspark_prime = Some(match s.dspark.take() {''')
sub('''        let mut walker = lm.model.dspark_prime_walker(engine, d, &mut s.dspark_prime);
        if !s.prime_service.advance(&mut walker)? {
            return Ok(true);
        }
        s.dspark = Some(s.prime_service.finish(walker)?);''',
    '''        diag1255_snap(engine, &format!("dspark-prime-chunk-before id={}", s.request_id));
        let mut walker = lm.model.dspark_prime_walker(engine, d, &mut s.dspark_prime);
        if !s.prime_service.advance(&mut walker)? {
            diag1255_snap(engine, &format!("dspark-prime-chunk-after id={}", s.request_id));
            return Ok(true);
        }
        s.dspark = Some(s.prime_service.finish(walker)?);
        diag1255_snap(engine, &format!("dspark-prime-done id={}", s.request_id));''')
open(p, "w").write(s)
print("patched", p)
