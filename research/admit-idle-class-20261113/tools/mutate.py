#!/usr/bin/env python3
"""mutate.py WORKER_RS ARM...: apply named red arms to worker.rs (exactly one match per edit).
The tests are kept; each arm restores or breaks one behavior and names the test that must fail
and a string its failure output must contain. Several arms may be applied to one file when no
two of them name the same test and none breaks another's test (the batches of dd-red.sh).
`mutate.py --list` prints `ARM TEST NEEDLE`; `mutate.py --batches` prints the batches."""
import sys

T_BOOK = "an_idle_unified_box_books_no_prefix_growth"
T_RETRY = "an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity"
T_WIRE = "the_idle_refusal_is_classified_against_the_post_load_capacity"
T_SITE = "unified_memory_admission_books_the_prefix_caches_allowed_growth"
T_PRIMARY = "a_primary_reading_takes_the_primary_devices_idle_capacity"
T_NOW = "the_idle_refusal_names_the_largest_request_that_fits_now"
T_HINT = "the_size_hint_is_a_request_the_caller_can_send"
T_COPY = "a_copy_publication_reserves_its_room_before_it_allocates"
T_PLANES = "a_copy_publication_counts_the_planes_it_copies"
T_TAIL = "a_copy_publication_counts_its_drafter_tail"
T_LATENT = "a_publication_counts_its_latent_rows"
T_HELD = "the_copy_preflight_is_held_to_the_live_reading"
T_STOP = "a_refused_copy_preflight_stops_the_publication"
T_HOST = "a_host_promotion_reserves_the_bytes_it_copies"
T_PROMO = "a_promotion_that_fails_leaves_the_caller_a_live_hit"
T_OVER = "a_publication_that_overruns_its_reservation_says_so"
T_DISCRETE = "a_discrete_card_keeps_its_copy_publication_order"
T_LIVE = "a_live_memory_refusal_names_its_cause_and_keeps_the_shape_warning"

ARMS = {
    # ---- The author's arms.
    # R1: the booking ignores idle.
    "R1": (T_BOOK, "left: 52411000000", [("    if unified && !idle {", "    let _ = idle;\n    if unified {")]),
    # R2: the idle refusal is always a 400.
    "R2": (T_RETRY, "left: ContextLength", [(
        "    if fits_capacity {\n        return EngineError::overloaded(",
        "    if fits_capacity && false {\n        return EngineError::overloaded(",
    )]),
    # R3: the reject branch answers EngineError::context_length again.
    "R3": (T_WIRE, "largest_now), );", [(
        "                            fail_request(\n"
        "                                req,\n"
        "                                idle_refusal(admission_cap, fits_capacity, largest, largest_now),\n"
        "                            );",
        "                            let _ = (fits_capacity, largest, largest_now);\n"
        "                            fail_request(req, EngineError::context_length(format!(\"request context {admission_cap} does not fit\")));",
    )]),
    # ---- The first reviewer's arms.
    # R4: the admission call site ignores idle (passes false).
    "R4": (T_SITE, "active.is_empty(), prefix_cache_budget_bytes()", [(
        "                    device_is_unified(&engine),\n"
        "                    active.is_empty(),\n"
        "                    prefix_cache_budget_bytes(),",
        "                    device_is_unified(&engine),\n"
        "                    false,\n"
        "                    prefix_cache_budget_bytes(),",
    )]),
    # R5: the post-load capacity no longer subtracts memra's pool growth since boot.
    "R5": (T_RETRY, "left: 6000000000", [(
        "            boot.free_bytes\n"
        "                .saturating_sub(now_pool_used.saturating_sub(boot.pool_used_bytes)),",
        "            boot.free_bytes + 0 * now_pool_used,",
    )]),
    # ---- The second reviewer's arm.
    # RA: a Primary reading is judged against the first boot device, whatever its own device is.
    "RA": (T_PRIMARY, "left: 2000000000", [(
        "                free_bytes: idle_capacity_free(free_bytes, primary_pool_used, boot(primary_device)),",
        "                free_bytes: idle_capacity_free(free_bytes, primary_pool_used, {\n"
        "                    let _ = (&boot, primary_device);\n"
        "                    capacity.first()\n"
        "                }),",
    )]),
    # ---- The first fold's arms.
    # R6: the size that fits now is computed on the capacity reading, not on the reading now.
    "R6": (T_WIRE, "let largest_now = largest_request_within( admission_cap, slack,", [(
        "                                &headroom.charges(primary_device, required),",
        "                                &at_capacity.charges(primary_device, required),",
    )]),
    # R7: the 503 does not name the size that fits now.
    "R7": (T_NOW, "free memory right now; retry after the Retry-After delay", [(
        "        return EngineError::overloaded(match largest_now {",
        "        return EngineError::overloaded(match largest_now.filter(|_| false) {",
    )]),
    # R8: the 400 names only the capacity's size, which then gets the 503.
    "R8": (T_NOW, "the largest admissible prompt plus max_tokens is 174315 tokens. Retrying will not help", [(
        "        (Some(n), Some(now)) if now < n => {",
        "        (Some(n), Some(now)) if false && now < n => {",
    )]),
    # R9: the spec publisher's reservation never applies (it reads the device as discrete).
    "R9": (T_COPY, "missing `let Some((to_allocate, held)) = spec_publication_reserve( px, pool_key, &copy_memory(engine),", [(
        "        px,\n        pool_key,\n        &copy_memory(engine),\n        &geometry,",
        "        px,\n        pool_key,\n        &CopyMemory {\n            unified: false,\n            ..copy_memory(engine)\n        },\n        &geometry,",
    )]),
    # R10: the preflight counts the bytes it is about to allocate as allocated (the clamp the
    # insert wrappers apply once the copy exists).
    "R10": (T_COPY, "left: (4166000000, 4166000000)", [(
        "    (to_allocate.saturating_add(held), held)\n",
        "    (to_allocate.saturating_add(held), to_allocate.saturating_add(held))\n",
    )]),
    # R11: the DSpark publisher hands over a tail whose bytes the reservation does not count.
    "R11": (T_COPY, "missing `bytes: dspark.draft_kv().tail_bytes(end),`", [(
        "\n            bytes: dspark.draft_kv().tail_bytes(end),\n",
        "\n            bytes: 0 * dspark.draft_kv().tail_bytes(end),\n",
    )]),
    # R12: a host promotion materializes its device entry without reserving.
    "R12": (T_COPY, "missing `let hi = host_promote_reserve(`", [(
        "    let hi = host_promote_reserve(\n"
        "        px,\n"
        "        host,\n"
        "        pool_key,\n"
        "        hi,\n"
        "        prepared.host_id,\n"
        "        memory,\n"
        "        &mut |host, dead| host_demote_prefix_entry(engine, host, dead),\n"
        "    )?;\n",
        "    let _ = (px.total_bytes, prepared.host_id, memory);\n",
    )]),
    # R13: a refusal by the live reading spends the one-time shape warning.
    "R13": (T_LIVE, "left: (2, true)", [(
        "\n        self.record_budget_skip(false, !live);\n",
        "\n        let _ = live;\n        self.record_budget_skip(false, true);\n",
    )]),
    # R14: the copy preflight reserves on a discrete card too.
    "R14": (T_DISCRETE, "left: Some(CopyReservation", [(
        "    if !memory.unified {\n        return None;\n    }\n"
        "    let (entry, allocated) = copy_preflight_split(to_allocate, held);",
        "    let (entry, allocated) = copy_preflight_split(to_allocate, held);",
    )]),
    # R15: the size hint counts one state of a choice group's charge.
    "R15": (T_WIRE, ".saturating_mul(choice_state_count),", [(
        "                                        .context_bytes(ctx, estimate_spec)\n"
        "                                        .saturating_mul(choice_state_count),",
        "                                        .context_bytes(ctx, estimate_spec),",
    )]),
    # ---- The second reviewer's round-2 mutations that passed the suite at ccdd965593.
    # MA: the publication counts no copied trunk or draft plane.
    "MA": (T_PLANES, "left: (0, 36000000)", [(
        "    if planes == SpecPlanes::Copy {\n"
        "        to_allocate += spec_copy_plane_bytes(pos, geometry.trunk.iter().copied());",
        "    if planes == SpecPlanes::Lend {\n"
        "        to_allocate += spec_copy_plane_bytes(pos, geometry.trunk.iter().copied());",
    )]),
    # MB: a host promotion reserves a 1024th of the bytes it will allocate.
    "MB": (T_HOST, "left: Some(", [(
        "    let bytes = host.entries[pool_key][hi].device_bytes;\n",
        "    let bytes = host.entries[pool_key][hi].device_bytes / 1024;\n",
    )]),
    # MC: the publication counts a 1024th of the drafter tail the publisher declared.
    "MC": (T_TAIL, "left: Some((", [(
        "    let tail_bytes = draft_tail.map_or(0, |tail| tail.bytes);",
        "    let tail_bytes = draft_tail.map_or(0, |tail| tail.bytes / 1024);",
    )]),
    # ME: the copy preflight never refuses (its reservation's verdict is discarded).
    "ME": (T_STOP, "assertion failed: !prefix_copy_preflight(", [(
        "        px.prepare_snapshot(pool_key, self.entry, self.budget, demote)\n    }",
        "        px.prepare_snapshot(pool_key, self.entry, self.budget, demote) || true\n    }",
    )]),
    # MF: the copy preflight's budget ignores the live reading (the configured budget, as on a
    # discrete card).
    "MF": (T_HELD, "left: InsertBudget { bytes: 50529000000", [(
        "        insert_budget_at(memory.configured, px.total_bytes, allocated, memory.live)\n    };\n",
        "        insert_budget_at(memory.configured, px.total_bytes, allocated, memory.live)\n    };\n"
        "    let budget = InsertBudget::from(budget.configured);\n",
    )]),
    # MH: the publication counts no latent rows.
    "MH": (T_LATENT, "left: 0", [(
        "    tail.len * tail.width + keys\n}",
        "    keys\n}",
    )]),
    # ---- The second fold's arms.
    # I1: on unified memory the promotion runs without the caller's hit leased.
    "I1": (T_PROMO, "left: Some(1)", [(
        "    if unified {\n        return px.with_hit_leased(pool_key, hit, promote);\n    }",
        "    if unified {\n        return (promote(px), hit);\n    }",
    )]),
    # I2: `admit` keeps its own index across the promotion (the promotion gets a copy).
    "I2": (T_PROMO, "host_promote_pin = host_promote_prefix_hit(engine, px, hpx, &pool_key, &prompt, &mut", [(
        "            host_promote_prefix_hit(engine, px, hpx, &pool_key, &prompt, &mut consumable_hit);",
        "            host_promote_prefix_hit(engine, px, hpx, &pool_key, &prompt, &mut consumable_hit.clone());",
    )]),
    # I3: the lease does not re-resolve the hit's index after the call.
    "I3": (T_PROMO, "left: Some(1)", [(
        "        let hit = lease.as_ref().and_then(|pin| self.id_index(pin));\n",
        "",
    )]),
    # H1: the hint is the charged context, not prompt plus max_tokens (the slack stays in).
    "H1": (T_HINT, "slack 8: the hint 110927 is charged", [(
        "        .checked_sub(slack)\n        .filter(|&n| n > 0)\n",
        "        .checked_sub(slack * 0)\n        .filter(|&n| n > 0)\n",
    )]),
    # H2: the hint is what fits on the first device.
    "H2": (T_HINT, "left: 100000", [(
        "        devices.iter().all(|d| {\n"
        "            let shed = context(d.device, admission_cap)",
        "        devices.iter().take(1).all(|d| {\n"
        "            let shed = context(d.device, admission_cap)",
    )]),
    # H3: the context term is a flat slope with the ring term beside it (the first fold's).
    "H3": (T_WIRE, "_ if device == primary_device => hint_costs .context_bytes(ctx, estimate_spec)", [(
        "                                    _ if device == primary_device => hint_costs\n"
        "                                        .context_bytes(ctx, estimate_spec)\n",
        "                                    _ if device == primary_device => ctx\n"
        "                                        .saturating_mul(bytes_per_token + ring_bytes_per_token)\n",
    )]),
    # H4: a figure that is all slack is named (zero tokens).
    "H4": (T_HINT, "left: Some(0)", [(
        "        .checked_sub(slack)\n        .filter(|&n| n > 0)\n",
        "        .checked_sub(slack)\n",
    )]),
    # O1: an entry larger than its reservation is not reported.
    "O1": (T_OVER, "left: None", [(
        "    entry_bytes\n        .checked_sub(to_allocate.saturating_add(held))\n",
        "    entry_bytes\n        .checked_sub(to_allocate.saturating_add(held).saturating_add(usize::MAX))\n",
    )]),
    # P1: the publication allocates on what its reservation evicted without reaping it.
    "P1": (T_COPY, "missing `if px.evictions > evictions { vmm_reap_for(", [(
        "    if px.evictions > evictions {\n        vmm_reap_for(\"copy preflight\");\n    }\n",
        "",
    )]),
    # T1: the host candidate's recency moves for a promotion the reservation then refuses.
    "T1": (T_HOST, "Instant {", [(
        "    if px\n"
        "        .snapshot_refusal(reservation.entry, reservation.budget.bytes)\n"
        "        .is_none()\n"
        "    {\n"
        "        host.touch(pool_key, hi);\n"
        "    }\n",
        "    host.touch(pool_key, hi);\n",
    )]),
}

# No two arms of a batch name the same test, and none breaks a test another arm of the batch
# names (R2 and R7 break T_RETRY or T_NOW; R13 breaks T_COPY and T_PLANES, which read the shape
# warning; MA breaks T_DISCRETE; R10 breaks T_PLANES, T_TAIL, T_LATENT, T_HELD, T_STOP and
# T_HOST; ME and MF break every refusal test; H1 and H4 edit the same lines).
BATCHES = [
    ["R1", "R2", "R3", "R4", "RA", "R9", "R14", "MB", "MC", "MH", "I1", "H1", "O1"],
    ["R5", "R6", "R11", "MA", "T1", "I2", "H2"],
    ["R8", "R15", "R12", "ME", "I3", "H4"],
    ["H3", "R10"],
    ["MF", "P1"],
    ["R13", "R7"],
]

if sys.argv[1] == "--list":
    for arm, (test, needle, _) in ARMS.items():
        print(f"{arm}\t{test}\t{needle}")
    sys.exit(0)
if sys.argv[1] == "--batches":
    seen = set()
    for batch in BATCHES:
        tests = [ARMS[a][0] for a in batch]
        assert len(set(tests)) == len(tests), batch
        seen.update(batch)
        print(" ".join(batch))
    assert seen == set(ARMS), set(ARMS) ^ seen
    sys.exit(0)
path = sys.argv[1]
src = open(path).read()
for arm in sys.argv[2:]:
    for old, new in ARMS[arm][2]:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{arm}: expected one match, found {n}: {old[:80]!r}")
        src = src.replace(old, new)
    print(f"{arm} applied")
open(path, "w").write(src)
