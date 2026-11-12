#!/usr/bin/env python3
"""memra#1255: apply the unbuilt-session booking and the settle wait to worker.rs and lib.rs.
Idempotent: refuses when an anchor is missing or already patched."""
import os
import sys

W = os.path.expanduser("~/.cache/memra-1255/wt/crates/memra-server/src/worker.rs")
L = os.path.expanduser("~/.cache/memra-1255/wt/crates/memra-server/src/lib.rs")


def rep(s, old, new, count=1):
    n = s.count(old)
    if n != count:
        sys.exit(f"anchor count {n} != {count}: {old[:90]!r}")
    return s.replace(old, new)


w = open(W).read()
if "admitted_charge_bytes" in w:
    sys.exit("already patched")

# 1. metric field
w = rep(w, """    pub admission_vram_defers: u64,
""", """    pub admission_vram_defers: u64,
    /// Admission passes that requeued an arrival because the reading, less the charges of
    /// admitted sessions that own no route state yet, did not hold it (memra#1255): the
    /// reading changes when the tick builds those sessions, so nothing is decided on it.
    pub admission_settle_waits: u64,
""")
w = rep(w, """    let mut n_vram_defers = 0u64;
""", """    let mut n_vram_defers = 0u64;
    let mut n_settle_waits = 0u64;
""")
w = rep(w, """            m.admission_vram_defers = n_vram_defers;
""", """            m.admission_vram_defers = n_vram_defers;
            m.admission_settle_waits = n_settle_waits;
""")

# 2. session fields
w = rep(w, """    booked_w_bytes: u64,
    /// The predicted TOTAL completion tokens behind `shadow_kv_hat`; with""", """    booked_w_bytes: u64,
    /// The charge the VRAM gate admitted this session on (memra#1255), and the held
    /// prefill-workspace part of it. Booked against every later reading while the session owns
    /// no route state (`session_owns_route_state`); 0 for a test-constructed Session.
    admitted_charge_bytes: usize,
    admitted_workspace_bytes: usize,
    /// The predicted TOTAL completion tokens behind `shadow_kv_hat`; with""")
w = rep(w, """        booked_w_bytes: 0,
        vmm_floor_rows: n_prompt,""", """        booked_w_bytes: 0,
        admitted_charge_bytes: 0,
        admitted_workspace_bytes: 0,
        vmm_floor_rows: n_prompt,""")

# 3. helpers, next to the pending-prime terms
w = rep(w, """/// Prompt rows a session still has to prime (memra#680): its queued rows while the prime runs,
""", """/// Whether a session owns its route's device state (memra#1255). `admit` builds the plain cache,
/// the MTP spec session and every restored or resumed session; the DFlash, gemma-spec and GLM
/// cold primes build theirs in the tick (`dspark_prime_start`, `dspark_spec_session_new`,
/// `gemma_spec_session_new`, `glm5_prime_start`). Until then the session has allocated none of
/// its charge, so the live reading cannot see it.
fn session_owns_route_state(s: &Session) -> bool {
    s.cache.is_some()
        || s.spec.is_some()
        || s.mtp_prime.is_some()
        || s.gspec.is_some()
        || s.dspark.is_some()
        || s.dspark_prime.is_some()
        || s.glm5.is_some()
        || s.glm5_prime.is_some()
        || s.glm5_plain_prime.is_some()
}

/// What the admitted sessions that own no route state yet will allocate (memra#1255): the
/// charge the gate admitted each one on. Pure over `(owns_route_state, charge, workspace)` per
/// session. `workspace_booked` (the armed memory door) leaves out each charge's prefill
/// workspace, which the door's pending-prime term books already.
fn unbuilt_charge_bytes(
    sessions: impl Iterator<Item = (bool, usize, usize)>,
    workspace_booked: bool,
) -> usize {
    sessions
        .filter(|(built, _, _)| !built)
        .map(|(_, charge, workspace)| {
            if workspace_booked {
                charge.saturating_sub(workspace)
            } else {
                charge
            }
        })
        .fold(0usize, usize::saturating_add)
}

/// The unbuilt booking over the active set (`unbuilt_charge_bytes`).
fn unbuilt_session_booking(active: &[Session], workspace_booked: bool) -> usize {
    unbuilt_charge_bytes(
        active.iter().map(|s| {
            (
                session_owns_route_state(s),
                s.admitted_charge_bytes,
                s.admitted_workspace_bytes,
            )
        }),
        workspace_booked,
    )
}

/// SETTLE WAIT (memra#1255): an arrival the booked reading does not hold while an admitted
/// session still owes its unbuilt charge waits for the next pass. The tick builds that session
/// and the next reading shows what it took, so no reclaim, eager admission, drafter refusal or
/// capacity reject is decided on this one.
fn settle_wait(unbuilt_booked: usize, fits_full: bool) -> bool {
    unbuilt_booked > 0 && !fits_full
}

/// Prompt rows a session still has to prime (memra#680): its queued rows while the prime runs,
""")

# 4. the booking in the gate
w = rep(w, """                let pending_booked = pending_booked.saturating_add(vmm_owed);
                let book =
                    move |h: AdmissionHeadroom| h.less_pending(pending_booked, primary_device);""", """                let pending_booked = pending_booked.saturating_add(vmm_owed);
                // UNBUILT BOOKING (memra#1255): an admitted session whose route builds its state
                // in the tick (a DFlash, gemma-spec or GLM cold prime) has allocated none of its
                // charge, so a second arrival read in the same pass, or in a step-OOM replay's
                // pass, saw the first one's bytes as free and both were admitted on them. Every
                // reading is reduced by those charges until the tick builds the sessions. On a
                // single-device deployment only: a PP or TP route that builds its state in the
                // tick charges per stage, which this scalar booking does not split.
                let unbuilt_booked = if is_multi_device_deployment(&loaded) {
                    0
                } else {
                    unbuilt_session_booking(&active, admit_memory_cfg.armed)
                };
                let pending_booked = pending_booked.saturating_add(unbuilt_booked);
                let book =
                    move |h: AdmissionHeadroom| h.less_pending(pending_booked, primary_device);""")

# 5. the settle wait, before any reclaim or verdict
anchor = """                if let Some(mut headroom) = measured_headroom {
                    // EFFECTIVE free, not driver `free`: `mem_get_info` cannot see blocks the"""
w = rep(w, anchor, """                if let Some(headroom) = measured_headroom.as_ref()
                    && settle_wait(unbuilt_booked, headroom.sufficient(required))
                {
                    if settle_waits == 0 {
                        eprintln!(
                            "[admission] settle wait: model={model_key:?} ctx={} effective free \\
                             {:.0}MB after {:.0}MB booked for {} admitted session(s) with no \\
                             route state yet < cost {:.0}MB + reserve {:.0}MB; requeued for the \\
                             next pass [queue {}]",
                            admission_cap,
                            headroom.limiting_free_bytes() as f64 / 1e6,
                            unbuilt_booked as f64 / 1e6,
                            active.iter().filter(|s| !session_owns_route_state(s)).count(),
                            cost as f64 / 1e6,
                            reserve as f64 / 1e6,
                            queue.len() + requeue.len(),
                        );
                    }
                    settle_waits += 1;
                    n_settle_waits += 1;
                    requeue.push_back(req);
                    continue;
                }
""" + anchor)
w = rep(w, """        let mut vram_defers = 0usize;
""", """        let mut vram_defers = 0usize;
        let mut settle_waits = 0usize;
""")

# 6. record the admitted charge
w = rep(w, """                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    active.push(s);""", """                    // memra#1255: the charge this session was admitted on, booked while it owns
                    // no route state (`unbuilt_session_booking`).
                    s.admitted_charge_bytes = cost;
                    s.admitted_workspace_bytes =
                        crate::admit_predict::RequestCharge::from_physical_cost(
                            cost as u64,
                            context_cap_bytes as u64,
                            activation_bytes as u64,
                            draft_state_bytes as u64,
                            0,
                        )
                        .prefill_workspace_bytes as usize;
                    admission_book.admit(&s.model, s.booked_kv_bytes, s.shadow_kv_hat);
                    active.push(s);""")
open(W, "w").write(w)

l = open(L).read()
l = rep(l, """            "admission_vram_defers": m.admission_vram_defers,
""", """            "admission_vram_defers": m.admission_vram_defers,
            "admission_settle_waits": m.admission_settle_waits,
""")
open(L, "w").write(l)
print("patched")
