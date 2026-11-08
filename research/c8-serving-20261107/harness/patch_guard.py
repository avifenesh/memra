"""Apply the prefix-cache headroom guard to worker.rs (idempotent check: refuses a second apply).

After a tick in which the prefix cache published an entry, a discrete card's driver must still
have the admission's driver-headroom floor; otherwise the cache yields LRU entries (and the pool
trims them back to the driver) until it does. Applied after cand4 is built.

  python3 patch_guard.py WORKER_RS
"""
import sys

p = sys.argv[1]
s = open(p).read()
assert 'fn prefix_yield_driver_headroom' not in s, 'already applied'


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


# 1. the function, after ensure_driver_headroom
rep("""/// One device's async-pool reading taken BEFORE a prefix-cache reclaim (memra#346, #445), so""",
"""/// THE PREFIX CACHE YIELDS THE DRIVER HEADROOM (2026-11-07). Called after a tick in which the
/// prefix cache published an entry, on a discrete card (unified memory has its own insert guard,
/// `prefix_insert_budget`). Admission keeps `admit_driver_headroom_bytes` of driver memory in
/// front of admitted work and reclaims the cache when an arrival needs its bytes, but an insert
/// made after the admission spends driver memory nobody re-checks before the next decode step.
/// So: trim the pool's cached bytes back to the driver (`ensure_driver_headroom`), and if the
/// driver is still under the floor, evict the oldest unleased entries until the shortfall is
/// freed, then trim those bytes back too. Returns (entries evicted, bytes freed).
///
/// Measured before it (24 GB RTX 5090 Laptop, Qwen3.8-27B, agentic c8, measured pass): seed
/// inserts of 549 MB and 593 MB took driver free from 3.8 GB to 83 MB within 42 s of decode, the
/// fa partial pool's next grow (about 71 MB) then failed with `CUDA_ERROR_OUT_OF_MEMORY` inside the
/// batched decode step, after the step's first KV append, so the step-OOM retry could not run, and
/// all eight streams ended `overloaded` after their first tokens: 2 of 2 runs
/// (`research/c8-serving-20261107/`).
fn prefix_yield_driver_headroom(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    px: &mut PrefixCache,
) -> (usize, usize) {
    let floor = admit_driver_headroom_bytes();
    if floor == 0 || px.total_bytes == 0 || device_is_unified(engine) {
        return (0, 0);
    }
    let min_driver_free = |engine: &Engine, loaded: &HashMap<String, LoadedModel>| {
        model_device_engines(engine, loaded)
            .iter()
            .filter_map(|owner| owner.ctx().mem_get_info().ok().map(|(free, _)| free))
            .min()
    };
    if min_driver_free(engine, loaded).is_none_or(|free| free >= floor) {
        return (0, 0);
    }
    ensure_driver_headroom(engine, loaded, "after a prefix-cache insert");
    let Some(free) = min_driver_free(engine, loaded) else {
        return (0, 0);
    };
    if free >= floor {
        return (0, 0);
    }
    let before = pool_readings(engine, loaded);
    let resident = px.total_bytes;
    let (n, freed) = px.evict_to_bytes(resident.saturating_sub(floor - free));
    if n == 0 {
        return (0, 0);
    }
    settle_reclaimed_prefix_bytes(engine, loaded, &before, freed, "prefix headroom");
    eprintln!(
        "[prefix-cache] headroom: driver free {:.0}MB under the {:.0}MB floor after an insert; \\
         evicted {n} entries ({:.0}MB, resident {:.0}MB -> {:.0}MB); driver free now {:.0}MB",
        free as f64 / 1e6,
        floor as f64 / 1e6,
        freed as f64 / 1e6,
        resident as f64 / 1e6,
        px.total_bytes as f64 / 1e6,
        min_driver_free(engine, loaded).unwrap_or(0) as f64 / 1e6,
    );
    (n, freed)
}

/// One device's async-pool reading taken BEFORE a prefix-cache reclaim (memra#346, #445), so""")

# 2. tick start: remember the insert count
rep("""        let batching = serve_batching();
        let mut finished: Vec<usize> = Vec::new();
""", """        let batching = serve_batching();
        let mut finished: Vec<usize> = Vec::new();
        let prefix_inserts_at_tick = px.inserts;
""")

# 3. before phase (c): the guard
rep("""            prefill_rate_mark =
                (primed_this_tick > 0).then_some((t_prefill_phase, primed_this_tick));
            // (c) batched decode, interactive rows first (stable sort by lane index: chunks""",
"""            prefill_rate_mark =
                (primed_this_tick > 0).then_some((t_prefill_phase, primed_this_tick));
            // The cache published this tick: before the decode step allocates, give back the
            // driver headroom an insert may have spent (`prefix_yield_driver_headroom`).
            if px.inserts != prefix_inserts_at_tick {
                let (n, _) = prefix_yield_driver_headroom(&engine, &loaded, &mut px);
                n_prefix_headroom_evictions += n as u64;
            }
            // (c) batched decode, interactive rows first (stable sort by lane index: chunks""")

# 4. counter + metrics
rep("""    let mut n_prefill_deadline_ticks = 0u64;
""", """    let mut n_prefill_deadline_ticks = 0u64;
    let mut n_prefix_headroom_evictions = 0u64;
""")
rep("""            m.prefill_deadline_ticks = n_prefill_deadline_ticks;
""", """            m.prefill_deadline_ticks = n_prefill_deadline_ticks;
            m.prefix_headroom_evictions = n_prefix_headroom_evictions;
""")
rep("""    pub prefill_deadline_ticks: u64,
""", """    pub prefill_deadline_ticks: u64,
    /// Prefix-cache entries evicted to give back the driver headroom after an insert
    /// (`prefix_yield_driver_headroom`).
    pub prefix_headroom_evictions: u64,
""")
open(p, 'w').write(s)

lib = p.replace('worker.rs', 'lib.rs')
t = open(lib).read()
old = '            "prefill_deadline_ticks": m.prefill_deadline_ticks,\n'
assert t.count(old) == 1
t = t.replace(old, old + '            "prefix_headroom_evictions": m.prefix_headroom_evictions,\n')
open(lib, 'w').write(t)
print('applied')
