#!/usr/bin/env python3
"""memra#1255 review round 1 (worker.rs): S1 the armed door books an unbuilt session's whole-prompt
stack (v2 counts it as an unstarted walker); nit 2 a settle wait only for an arrival the booking
alone keeps out (so no hold for one that cannot fit); S3 settle-wait time ends when the request
gets past the settle check, and the settled line prints when `admit` succeeds. Idempotent: refuses
when a needle is missing. usage: fix3-patch.py WORKER_RS"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        sys.exit(f"needle found {n} times, wanted {count}:\n{old[:200]}")
    s = s.replace(old, new)


# --- S1: an unbuilt session owes its whole-prompt stack (pending_prime_bytes_v2) ---
rep("""                unstarted_walker: s.spec.is_some() && s.mtp_prime.is_none(),
                cooperative: cooperative(&s.model),""",
    """                unstarted_walker: prime_walker_unstarted(
                    s.spec.is_some(),
                    s.mtp_prime.is_some(),
                    session_owns_route_state(s),
                ),
                cooperative: cooperative(&s.model),""")
rep("""/// One still-priming session as the corrected pending-prime term reads it (WP-B day 39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimeOwed {""",
    """/// Whether a still-priming session owes its whole-prompt stack, its walker not yet started: an
/// MTP spec session before its walker starts, and (memra#1255) any session that owns no route
/// state yet. A DFlash, gemma-spec or GLM cold prime builds its walker in the tick after `admit`,
/// so until then the armed door's pending-prime term must book that stack: the unbuilt booking
/// leaves the prefill workspace to this term on the armed door.
fn prime_walker_unstarted(mtp_spec: bool, mtp_prime_started: bool, owns_route_state: bool) -> bool {
    (mtp_spec && !mtp_prime_started) || !owns_route_state
}

/// One still-priming session as the corrected pending-prime term reads it (WP-B day 39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimeOwed {""")
rep("""    /// On the MTP spec route with its walker not yet started (its whole-prompt stack is owed).
    unstarted_walker: bool,""",
    """    /// Its walker not yet started, so its whole-prompt stack is owed (`prime_walker_unstarted`).
    unstarted_walker: bool,""")

# --- nit 2 + S2: settle only when the booking alone keeps the arrival out ---
rep("""/// SETTLE WAIT (memra#1255): an arrival the booked reading does not hold while an admitted
/// session still owes its unbuilt charge waits for the next pass. The tick builds that session
/// and the next reading shows what it took, so no reclaim, eager admission, drafter refusal or
/// capacity reject is decided on this one.
fn settle_wait(unbuilt_booked: usize, fits_full: bool) -> bool {
    unbuilt_booked > 0 && !fits_full
}""",
    """/// SETTLE WAIT (memra#1255): an arrival the booked reading does not hold, while an admitted
/// session still owes its unbuilt charge and the reading without that booking would hold it,
/// waits for the next pass. The tick builds that session and the next reading shows what it took,
/// so no reclaim, eager admission, drafter refusal or capacity reject is decided on this one. An
/// arrival the reading does not hold even without the booking gains nothing from a wait (the tick
/// can only lower the reading): it goes on to those verdicts at once, on the booked reading.
fn settle_wait(unbuilt_booked: usize, fits_booked: bool, fits_unbooked: bool) -> bool {
    unbuilt_booked > 0 && !fits_booked && fits_unbooked
}""")
rep("""#[derive(Debug, PartialEq, Eq)]
enum AdmissionHeadroom {""", """#[derive(Debug, Clone, PartialEq, Eq)]
enum AdmissionHeadroom {""")
rep("""                let measured_headroom =
                    admission_headroom(&engine, &loaded, device_requirements.as_deref()).map(book);
                if device_requirements.is_some() && measured_headroom.is_none() {""",
    """                let raw_headroom =
                    admission_headroom(&engine, &loaded, device_requirements.as_deref());
                // The reading without the unbuilt booking (memra#1255): only an arrival it holds
                // takes a settle wait.
                let fits_unbooked = raw_headroom.clone().is_some_and(|h| {
                    h.less_pending(
                        pending_booked.saturating_sub(unbuilt_booked),
                        primary_device,
                    )
                    .sufficient(required)
                });
                let measured_headroom = raw_headroom.map(book);
                if device_requirements.is_some() && measured_headroom.is_none() {""")
rep("""                    && settle_wait(unbuilt_booked, headroom.sufficient(required))
                {""", """                    && settle_wait(unbuilt_booked, headroom.sufficient(required), fits_unbooked)
                {""")

# --- S3: the settle time ends at the settle check; the line prints when admit succeeds ---
rep("""                    requeue.push_back(req);
                    continue;
                }
                if let Some(mut headroom) = measured_headroom {""",
    """                    requeue.push_back(req);
                    continue;
                }
                // Past its settle waits, if it took any: their time ends here, before any VRAM
                // defer that follows. The `settled:` line prints when `admit` succeeds.
                if let Some((since, passes)) = settling.remove(&req.request_id) {
                    let waited_ms = since.elapsed().as_secs_f64() * 1e3;
                    settle_wait_ms_total += waited_ms;
                    let entry = settled
                        .entry(req.request_id.clone())
                        .or_insert((Instant::now(), 0, 0.0));
                    entry.1 += passes;
                    entry.2 += waited_ms;
                }
                if let Some(mut headroom) = measured_headroom {""")
rep("""            if let Some((since, passes)) = settling.remove(&req.request_id) {
                let waited_ms = since.elapsed().as_secs_f64() * 1e3;
                settle_wait_ms_total += waited_ms;
                eprintln!(
                    "[admission] settled: model={model_key:?} ctx={admission_cap} admitted after \\
                     {passes} settle wait(s), {waited_ms:.1} ms after the first"
                );
            }
            ensure_driver_headroom(&engine, &loaded, "prime");""",
    """            ensure_driver_headroom(&engine, &loaded, "prime");""")
rep("""                        .prefill_workspace_bytes as usize;
                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    active.push(s);""",
    """                        .prefill_workspace_bytes as usize;
                    if let Some((_, passes, waited_ms)) = settled.remove(&s.request_id) {
                        eprintln!(
                            "[admission] settled: model={:?} admitted after {passes} settle \\
                             wait(s), {waited_ms:.1} ms in them",
                            s.model
                        );
                    }
                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    active.push(s);""")
rep("""    let mut settling: HashMap<String, (Instant, u32)> = HashMap::new();
    let mut settle_wait_ms_total = 0f64;""",
    """    let mut settling: HashMap<String, (Instant, u32)> = HashMap::new();
    // Requests past their settle waits and not yet admitted: when they got past, the passes they
    // waited and the milliseconds those took, for the `settled:` line at admission.
    let mut settled: HashMap<String, (Instant, u32, f64)> = HashMap::new();
    let mut settle_wait_ms_total = 0f64;""")
rep("""        settling.retain(|_, (since, _)| since.elapsed() < Duration::from_secs(600));""",
    """        settling.retain(|_, (since, _)| since.elapsed() < Duration::from_secs(600));
        settled.retain(|_, (since, _, _)| since.elapsed() < Duration::from_secs(600));""")
rep("""    /// Total time requests spent in settle waits before their admission, in ms (memra#1255).
    pub admission_settle_wait_ms: f64,""",
    """    /// Total time requests spent in settle waits, in ms (memra#1255): from a request's first
    /// settle wait to the pass that let it past, before any VRAM defer that follows.
    pub admission_settle_wait_ms: f64,""")

open(p, "w").write(s)
print("patched")
