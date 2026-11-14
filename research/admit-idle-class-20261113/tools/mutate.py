#!/usr/bin/env python3
"""mutate.py WORKER_RS ARM: apply one named red arm to worker.rs (exactly one match required per
edit). The tests are kept; each arm restores or breaks one behavior and names the test that must
fail and a string its failure output must contain. `mutate.py --list` prints `ARM TEST NEEDLE`."""
import sys

T_BOOK = "an_idle_unified_box_books_no_prefix_growth"
T_RETRY = "an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity"
T_WIRE = "the_idle_refusal_is_classified_against_the_post_load_capacity"
T_SITE = "unified_memory_admission_books_the_prefix_caches_allowed_growth"
T_PRIMARY = "a_primary_reading_takes_the_primary_devices_idle_capacity"
T_NOW = "the_idle_refusal_names_the_largest_context_that_fits_now"
T_COPY = "a_copy_publication_reserves_its_room_before_it_allocates"
T_LIVE = "a_live_memory_refusal_names_its_cause_and_keeps_the_shape_warning"
T_DISCRETE = "a_discrete_card_keeps_its_copy_publication_order"

ARMS = {
    # The author's arms.
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
    # The first reviewer's arms.
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
    # The second reviewer's arm.
    # RA: a Primary reading is judged against the first boot device, whatever its own device is.
    "RA": (T_PRIMARY, "left: 2000000000", [(
        "                free_bytes: idle_capacity_free(free_bytes, primary_pool_used, boot(primary_device)),",
        "                free_bytes: idle_capacity_free(free_bytes, primary_pool_used, {\n"
        "                    let _ = (&boot, primary_device);\n"
        "                    capacity.first()\n"
        "                }),",
    )]),
    # The fold's arms.
    # R6: the size that fits now is computed on the capacity, not on the reading now.
    "R6": (T_WIRE, "let largest_now = largest_context_within( admission_cap, required, available,", [(
        "                            let largest_now = largest_context_within(\n"
        "                                admission_cap,\n"
        "                                required,\n"
        "                                available,",
        "                            let largest_now = largest_context_within(\n"
        "                                admission_cap,\n"
        "                                required,\n"
        "                                capacity,",
    )]),
    # R7: the 503 does not name the size that fits now.
    "R7": (T_NOW, "free memory right now; retry after the Retry-After delay", [(
        "        return EngineError::overloaded(match largest_now {",
        "        return EngineError::overloaded(match largest_now.filter(|_| false) {",
    )]),
    # R8: the 400 names only the capacity's size, which then gets the 503.
    "R8": (T_NOW, "the largest admissible context is 174315 tokens. Retrying will not help", [(
        "        (Some(n), Some(now)) if now < n => {",
        "        (Some(n), Some(now)) if false && now < n => {",
    )]),
    # R9: the spec publisher allocates without reserving its room.
    "R9": (T_COPY, "missing `if to_allocate > 0 && !prefix_copy_preflight(", [(
        "    if to_allocate > 0 && !prefix_copy_preflight(engine, px, hpx, pool_key, to_allocate, held) {",
        "    if false && to_allocate > 0 && !prefix_copy_preflight(engine, px, hpx, pool_key, to_allocate, held) {",
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
    "R12": (T_COPY, "missing `let hi = host_promote_reserve(", [(
        "    let hi = host_promote_reserve(engine, px, host, pool_key, hi, prepared.host_id)?;\n",
        "    let _ = (px.total_bytes, prepared.host_id);\n",
    )]),
    # R15: the fits-now arithmetic uses one row's slope for a choice group's charge.
    "R15": (T_WIRE, ".saturating_mul(choice_state_count as u64);", [(
        "                                .saturating_add(ring_bytes_per_token as u64)\n"
        "                                .saturating_mul(choice_state_count as u64);",
        "                                .saturating_add(ring_bytes_per_token as u64);",
    )]),
    # R14: the copy preflight runs on a discrete card too (reserving and evicting ahead of a copy
    # whose path there is unchanged).
    "R14": (T_DISCRETE, "missing `if !device_is_unified(engine) { return true; }`", [(
        "    if !device_is_unified(engine) {\n        return true;\n    }\n"
        "    let (entry, allocated) = copy_preflight_split(to_allocate, held);",
        "    let _ = device_is_unified(engine);\n"
        "    let (entry, allocated) = copy_preflight_split(to_allocate, held);",
    )]),
    # R13: a refusal by the live reading spends the one-time shape warning.
    "R13": (T_LIVE, "left: (2, true)", [(
        "\n        self.record_budget_skip(false, !live);\n",
        "\n        let _ = live;\n        self.record_budget_skip(false, true);\n",
    )]),
}

if sys.argv[1] == "--list":
    for arm, (test, needle, _) in ARMS.items():
        print(f"{arm}\t{test}\t{needle}")
    sys.exit(0)
path, arm = sys.argv[1], sys.argv[2]
src = open(path).read()
for old, new in ARMS[arm][2]:
    n = src.count(old)
    if n != 1:
        sys.exit(f"{arm}: expected one match, found {n}: {old[:80]!r}")
    src = src.replace(old, new)
open(path, "w").write(src)
print(f"{arm} applied")
