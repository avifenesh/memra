# WP-D v1.1 directed-admission fix

Repository **avifenesh/memra**, branch **lane/spill-d-20260919**.
Before source / v1.1 merge: **609ba4d2f7631a1bd4e1debf45206f7963803489** (merge of `ce49cd86`).
Checked implementation: **2bb7524cae72c8c2a203ae2287e8b4639502cd00**.
Local commits only: no push, PR, tag, main integration, deployment or GPU qualification.

## Result and scope

**D's finding is repaired: peer target 17 passed / 0 failed**, from 14 passed / 1 failed.
**The full crate is NOT green:** the unchanged A-owned
`day2::revision_v11_pinned_quarantine` still fails at frozen schedule line 215:
`left: Ok(())`, `right: Err(Busy)`. The requested full-crate-green criterion is blocked
on A's independent fix; D did not edit A or weaken/ignore the test. Lead must integrate
A and D and repeat the full battery. No other decision or contract proposal is needed.

Changes are confined to `crates/memra-tier/tests/peer/fake.rs` plus this receipt namespace:

- Replace the single global grants boolean with a map keyed by directed
  `(owner/source, consumer/destination)` device IDs. Each entry has independent context
  and pool grants plus `LinkHealth`; missing routes and non-AtMaximum health refuse.
- Admission checks that route before reserving through the same injected governor.
  Denial creates no allocation or charge. Existing borrowed release/Busy retry code is
  unchanged and its generic capacity/lifetime schedules still pass.
- Bind each frozen RouteFault to its own control. Run the unchanged directed schedule
  in both orientations; context/pool denial leaves the reverse admission intact.
- Add independent-control, missing/unknown/idle route, and reverse-downgrade coverage.
  Forward admission does not require reverse route health. These are injected directed
  route observations, not a claim that physical endpoint degradation is asymmetric.
- Replace submit's old global-grant check with the same directed predicate. Additional
  coverage revokes each control after reservation, requires all input ownership returned
  with no accepted ticket/charge change, then restores and drains successfully.
- No host-bounce path, second governor, external dependency, runtime flag or native
  implementation was added. Frozen contracts and `tests/contracts/*` are byte-unchanged
  from the merge; `git diff --exit-code` checked them explicitly.

## Contract assessment

`PeerPlan.owner_device` and `consumer_device` already identify a direction. `PeerCapacity`
can retain current grant/health authority in backend-owned state; no new persisted field
is necessary for this fake. **No contract proposal.** Native context/topology/binary
freshness and actual driver grant acquisition are still adapter/hardware gates; this
CPU fixture is not a native implementation or live route proof.

## Executed checks

| Command | Result |
|---|---|
| `cargo test -p memra-tier --offline --test peer` before | exit 101; 14 pass / 1 fail, reproduced exact directed failure |
| Same peer target after | exit 0; 17 pass / 0 fail |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo check -p memra-tier --offline --all-targets` | exit 0, macOS |
| Same check + `--target x86_64-unknown-linux-gnu` | exit 0, cross-target check only |
| `cargo test -p memra-tier --offline` | exit 101, only A storage failure; later doctests not reached |
| Same full test + `--no-fail-fast` | exit 101; 131 passed / 1 failed / 0 ignored, including 4 passing doctests |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0 |
| `git diff --check` | exit 0 |
| `bash tools/check-flags.sh` | exit 0; 864 reads, no uncovered names |

Full no-fail-fast target counts: bank 31, contracts 44, peer 17, placement 6,
storage 29 pass / 1 fail, doctests 4. No CUDA/nvcc/GPU, Linux execution, P2P/PCIe,
O_DIRECT hardware, model, server or G0–G7 qualification ran.

`v11-fix/commands.json` retains after-check argv/exit/time/hash;
`v11-fix/source.json` binds source and raw logs. Logs capture merged stdout/stderr
before any parsing. The initial and full before/after test outputs are reproduced
verbatim below. Other complete outputs, including no-fail-fast, are adjacent raw logs.

Effort: approximately **0.2 agent-hours**, no GPU time. No unrelated dirty changes
were found or staged. This is an open program worktree retained for lead integration,
not a closed/abandoned lane; no scratch directory or stash was created.

## Before peer — verbatim

`cargo test -p memra-tier --offline --test peer`

```text
   Compiling memra-tier v0.138.0 (/Users/avifen/tiyuvta/wt-spill-d/crates/memra-tier)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.88s
     Running tests/peer/mod.rs (target/debug/deps/peer-2abcdff8374b3a8b)

running 15 tests
test fake::capacity_grants_padding_and_homogeneous_ticket_refusal ... ok
test fake::caller_source_drop_keeps_owned_bytes_until_acknowledged ... ok
test fake::revision_v11_peer_submit_epochs_and_zero_accept ... ok
test fake::partial_acceptance_and_zero_accept_return_owned_inputs ... ok
test fake::exact_bytes_epochs_cancel_and_fence_issuer ... ok
test fake::revision_v11_peer_capacity ... ok
test fake::revision_v11_directed_capacity_lane_fix_required ... FAILED
test fake::quarantine_graph_pins_foreign_tickets_and_shared_budget ... ok
test fake::revision_v11_peer_complete_cancel_and_quarantine ... ok
test idle_downshift_is_deferred_active_downgrade_refuses ... ok
test fake::shared_schedule_runs_on_d_byte_fake ... ok
test fake::short_corrupt_and_wrong_context_completion_refuse ... ok
test observation_expires_with_context_topology_or_binary ... ok
test fake::revision_v11_peer_indexed_acceptance ... ok
test topology_is_directed_grant_bound_and_never_host_bounce ... ok

failures:

---- fake::revision_v11_directed_capacity_lane_fix_required stdout ----

thread 'fake::revision_v11_directed_capacity_lane_fix_required' (15802304) panicked at crates/memra-tier/tests/peer/../contracts/revision_v11.rs:567:18:
denying one directed route must not deny its reverse: Unsupported
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    fake::revision_v11_directed_capacity_lane_fix_required

test result: FAILED. 14 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `-p memra-tier --test peer`
```

## After peer — verbatim

`cargo test -p memra-tier --offline --test peer`

```text
   Compiling memra-tier v0.138.0 (/Users/avifen/tiyuvta/wt-spill-d/crates/memra-tier)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.24s
     Running tests/peer/mod.rs (target/debug/deps/peer-2abcdff8374b3a8b)

running 17 tests
test fake::capacity_grants_padding_and_homogeneous_ticket_refusal ... ok
test fake::directed_fault_controls_are_independent_and_missing_routes_refuse ... ok
test fake::exact_bytes_epochs_cancel_and_fence_issuer ... ok
test fake::quarantine_graph_pins_foreign_tickets_and_shared_budget ... ok
test fake::revision_v11_peer_capacity ... ok
test fake::revision_v11_directed_capacity ... ok
test fake::revision_v11_peer_complete_cancel_and_quarantine ... ok
test fake::caller_source_drop_keeps_owned_bytes_until_acknowledged ... ok
test fake::partial_acceptance_and_zero_accept_return_owned_inputs ... ok
test fake::directed_fault_after_reservation_returns_owned_copies_without_acceptance ... ok
test fake::revision_v11_peer_submit_epochs_and_zero_accept ... ok
test fake::revision_v11_peer_indexed_acceptance ... ok
test fake::short_corrupt_and_wrong_context_completion_refuse ... ok
test fake::shared_schedule_runs_on_d_byte_fake ... ok
test idle_downshift_is_deferred_active_downgrade_refuses ... ok
test observation_expires_with_context_topology_or_binary ... ok
test topology_is_directed_grant_bound_and_never_host_bounce ... ok

test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

```

## Before full crate — verbatim

`cargo test -p memra-tier --offline`

```text
   Compiling memra-tier v0.138.0 (/Users/avifen/tiyuvta/wt-spill-d/crates/memra-tier)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.42s
     Running unittests src/lib.rs (target/debug/deps/memra_tier-953de986bc31c7e6)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/bank/main.rs (target/debug/deps/bank-bd6b64183c1f3fb4)

running 31 tests
test bounds_wrong_ids_overflow_and_device_request_fail_closed ... ok
test alternate_layout_cannot_resurrect_masked_original_identity ... ok
test frozen_bank_cancel_schedule_runs_on_implementation ... ok
test discontiguous_row_scale_planes_are_required_and_exact ... ok
test failed_transfer_completion_enumerates_all_records_and_segments ... ok
test borrowed_view_and_alias_survive_busy_retirement ... ok
test actual_q2_q3_nvfp4_codes_are_preserved_without_kernel_promotion ... ok
test cancel_retains_charge_until_retirement_and_tombstone_ack ... ok
test frozen_row_order_schedule_runs_on_implementation ... ok
test read_planning_refuses_undeclared_tail_padding_and_invalid_granularity ... ok
test host_bridge::bridge_uses_per_record_metadata_and_preserves_existing_scale_planes ... ok
test missing_or_extra_checksums_and_false_uniform_declarations_refuse ... ok
test all_three_epochs_cross_service_and_publish_replay_refuse ... ok
test overlapping_cache_hit_tickets_hold_same_allocation_until_all_retire ... ok
test original_masks_foreign_catalog_and_wrong_domain_refuse_before_reads ... ok
test host_exps_uniform_mixed_original_mask_and_split_offsets ... ok
test refused_publication_retains_owned_charge_until_explicit_retirement ... ok
test pinned_ple_test_oracle_matches_native_function_bodies ... ok
test row_release_busy_preserves_retry_after_partial_alias_retirement ... ok
test revision_v11_rows_complete_cancel ... ok
test predictions_are_bounded_mask_checked_and_not_demand_heat ... ok
test shared_budget_atomic_refusal_and_tombstone_bound ... ok
test revision_v11_bank_cancel_identity_uniform ... ok
test uniform_proof_checks_actual_source_even_homogeneous_subset ... ok
test sparse_amplification_and_default_policy_are_explicit_arithmetic_not_io_benchmarks ... ok
test short_and_corrupt_reads_never_publish_partial_batch ... ok
test zero_small_full_cache_preserves_exact_bytes_and_scales ... ok
test revision_v11_corrupt_sibling_and_row_namespace ... ok
test revision_v11_rows_bytes_dedup_bounded_straddles ... ok
test synthetic_ple_trace_preserves_chunk_rewind_eos_and_native_expansion_bits ... ok
test rows_straddles_duplicates_pool_smaller_than_batch_and_retirement ... ok

test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s

     Running tests/contracts/mod.rs (target/debug/deps/contracts-8c4b776665dc29a3)

running 44 tests
test peer::revision_v11_directed_reference_routes ... ok
test peer::contiguous_bounds_overflow_and_foreign_owner_fail_closed ... ok
test peer::reusable_peer_schedule ... ok
test peer::peer_capacity_grants_budget_source_lifetime_and_explicit_release ... ok
test peer::revision_v11_peer_complete_cancel ... ok
test peer::peer_partial_vector_cannot_materialize_missing_sibling ... ok
test peer::peer_three_epochs_local_only_and_cancel_linearization ... ok
test services::bank_scale_plane_missing_is_not_a_complete_uniform_record ... ok
test services::bank_lease_keeps_backing_pinned_and_explicit_retirement_invalidates_aliases ... ok
test services::budget_device_headroom_and_priority_deadline_order ... ok
test services::common_governor_headroom_foreign_double_release_and_pin_lifetime ... ok
test services::object_partial_short_and_interrupted_root_never_publish ... ok
test services::object_cancel_before_and_after_commit_and_chunked_progress ... ok
test services::invalid_bank_publication_returns_charge_and_backing_for_explicit_release ... ok
test services::revision_v11_bank_and_rows_complete_before_cancel ... ok
test services::revision_v11_bank_row_unknown_retirement_hooks ... ok
test services::object_full_key_collision_corruption_and_version_refusal ... ok
test services::revision_v11_generic_governor_and_object ... ok
test services::bank_masked_ids_partial_completion_and_uniform_mixed_refusal ... ok
test services::reusable_object_tier_bank_row_schedules ... ok
test services::rows_rejected_read_and_cancel_never_publish_logical_subset ... ok
test rows::row_straddles_deduplicate_physical_reads_and_preserve_exact_duplicate_bytes ... ok
test transfer::completion_refuses_each_epoch_bad_checksum_missing_fence_and_short_vector ... ok
test transfer::copy_bounds_and_consumer_context_generation_are_checked ... ok
test services::tier_cancel_after_load_before_ready_and_unknown_retirement ... ok
test transfer::reusable_transfer_schedule ... ok
test transfer::revision_v11_zero_accept_preserves_owned_hosts ... ok
test transfer::transfer_cancel_after_completion_before_publication_revokes ... ok
test transfer::revision_v11_complete_cancel_and_lifetime ... ok
test services::rows_duplicate_order_dedup_charge_and_delayed_consumer_retirement ... ok
test services::tier_advisory_lookup_identity_completeness_and_epochs ... ok
test transfer::transfer_ready_take_is_owner_bound_and_post_publish_cannot_revoke ... ok
test transfer::revision_v11_indexed_acceptance ... ok
test transfer::transfer_zero_accept_error_returns_every_input_and_empty_refuses ... ok
test transfer::unknown_backend_shutdown_does_not_free_accepted_host_source ... ok
test transfer::unknown_cancel_and_delayed_fences_quarantine_until_retired ... ok
test transfer::transfer_partial_batch_and_broken_aggregate_are_detected ... ok
test wire::layout_duplicate_overflow_and_group_specific_counts_refuse ... ok
test wire::placement_negative_headroom_route_labels_and_estimates_survive_wire ... ok
test wire::heterogeneous_all_trailing_groups_aliases_and_padding ... ok
test wire::immutable_seal_and_committed_high_water_cannot_publish_rollback_tail ... ok
test wire::identity_domains_full_program_and_canonical_roundtrip ... ok
test rows::row_batch_larger_than_pool_progresses_and_one_bad_physical_read_refuses_all_output ... ok
test wire::golden_fixture_hashes_are_pinned ... ok

test result: ok. 44 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s

     Running tests/peer/mod.rs (target/debug/deps/peer-2abcdff8374b3a8b)

running 15 tests
test fake::caller_source_drop_keeps_owned_bytes_until_acknowledged ... ok
test fake::quarantine_graph_pins_foreign_tickets_and_shared_budget ... ok
test fake::partial_acceptance_and_zero_accept_return_owned_inputs ... ok
test fake::revision_v11_peer_capacity ... ok
test fake::capacity_grants_padding_and_homogeneous_ticket_refusal ... ok
test fake::exact_bytes_epochs_cancel_and_fence_issuer ... ok
test fake::revision_v11_peer_complete_cancel_and_quarantine ... ok
test fake::revision_v11_directed_capacity_lane_fix_required ... FAILED
test fake::revision_v11_peer_submit_epochs_and_zero_accept ... ok
test idle_downshift_is_deferred_active_downgrade_refuses ... ok
test fake::shared_schedule_runs_on_d_byte_fake ... ok
test observation_expires_with_context_topology_or_binary ... ok
test fake::short_corrupt_and_wrong_context_completion_refuse ... ok
test topology_is_directed_grant_bound_and_never_host_bounce ... ok
test fake::revision_v11_peer_indexed_acceptance ... ok

failures:

---- fake::revision_v11_directed_capacity_lane_fix_required stdout ----

thread 'fake::revision_v11_directed_capacity_lane_fix_required' (15803299) panicked at crates/memra-tier/tests/peer/../contracts/revision_v11.rs:567:18:
denying one directed route must not deny its reverse: Unsupported
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    fake::revision_v11_directed_capacity_lane_fix_required

test result: FAILED. 14 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

error: test failed, to rerun pass `-p memra-tier --test peer`
```

## After full crate — verbatim

`cargo test -p memra-tier --offline`

```text
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.15s
     Running unittests src/lib.rs (target/debug/deps/memra_tier-953de986bc31c7e6)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/bank/main.rs (target/debug/deps/bank-bd6b64183c1f3fb4)

running 31 tests
test alternate_layout_cannot_resurrect_masked_original_identity ... ok
test bounds_wrong_ids_overflow_and_device_request_fail_closed ... ok
test failed_transfer_completion_enumerates_all_records_and_segments ... ok
test borrowed_view_and_alias_survive_busy_retirement ... ok
test discontiguous_row_scale_planes_are_required_and_exact ... ok
test actual_q2_q3_nvfp4_codes_are_preserved_without_kernel_promotion ... ok
test frozen_bank_cancel_schedule_runs_on_implementation ... ok
test all_three_epochs_cross_service_and_publish_replay_refuse ... ok
test host_bridge::bridge_uses_per_record_metadata_and_preserves_existing_scale_planes ... ok
test cancel_retains_charge_until_retirement_and_tombstone_ack ... ok
test frozen_row_order_schedule_runs_on_implementation ... ok
test original_masks_foreign_catalog_and_wrong_domain_refuse_before_reads ... ok
test read_planning_refuses_undeclared_tail_padding_and_invalid_granularity ... ok
test missing_or_extra_checksums_and_false_uniform_declarations_refuse ... ok
test host_exps_uniform_mixed_original_mask_and_split_offsets ... ok
test pinned_ple_test_oracle_matches_native_function_bodies ... ok
test overlapping_cache_hit_tickets_hold_same_allocation_until_all_retire ... ok
test refused_publication_retains_owned_charge_until_explicit_retirement ... ok
test revision_v11_rows_complete_cancel ... ok
test predictions_are_bounded_mask_checked_and_not_demand_heat ... ok
test revision_v11_bank_cancel_identity_uniform ... ok
test row_release_busy_preserves_retry_after_partial_alias_retirement ... ok
test shared_budget_atomic_refusal_and_tombstone_bound ... ok
test sparse_amplification_and_default_policy_are_explicit_arithmetic_not_io_benchmarks ... ok
test uniform_proof_checks_actual_source_even_homogeneous_subset ... ok
test short_and_corrupt_reads_never_publish_partial_batch ... ok
test revision_v11_corrupt_sibling_and_row_namespace ... ok
test zero_small_full_cache_preserves_exact_bytes_and_scales ... ok
test revision_v11_rows_bytes_dedup_bounded_straddles ... ok
test synthetic_ple_trace_preserves_chunk_rewind_eos_and_native_expansion_bits ... ok
test rows_straddles_duplicates_pool_smaller_than_batch_and_retirement ... ok

test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s

     Running tests/contracts/mod.rs (target/debug/deps/contracts-8c4b776665dc29a3)

running 44 tests
test peer::contiguous_bounds_overflow_and_foreign_owner_fail_closed ... ok
test peer::peer_three_epochs_local_only_and_cancel_linearization ... ok
test peer::peer_capacity_grants_budget_source_lifetime_and_explicit_release ... ok
test peer::reusable_peer_schedule ... ok
test peer::peer_partial_vector_cannot_materialize_missing_sibling ... ok
test peer::revision_v11_peer_complete_cancel ... ok
test peer::revision_v11_directed_reference_routes ... ok
test services::bank_scale_plane_missing_is_not_a_complete_uniform_record ... ok
test services::budget_device_headroom_and_priority_deadline_order ... ok
test services::bank_lease_keeps_backing_pinned_and_explicit_retirement_invalidates_aliases ... ok
test services::common_governor_headroom_foreign_double_release_and_pin_lifetime ... ok
test services::invalid_bank_publication_returns_charge_and_backing_for_explicit_release ... ok
test services::object_cancel_before_and_after_commit_and_chunked_progress ... ok
test services::object_full_key_collision_corruption_and_version_refusal ... ok
test services::bank_masked_ids_partial_completion_and_uniform_mixed_refusal ... ok
test services::object_partial_short_and_interrupted_root_never_publish ... ok
test rows::row_straddles_deduplicate_physical_reads_and_preserve_exact_duplicate_bytes ... ok
test services::revision_v11_bank_and_rows_complete_before_cancel ... ok
test services::revision_v11_bank_row_unknown_retirement_hooks ... ok
test services::rows_rejected_read_and_cancel_never_publish_logical_subset ... ok
test services::reusable_object_tier_bank_row_schedules ... ok
test services::revision_v11_generic_governor_and_object ... ok
test services::tier_cancel_after_load_before_ready_and_unknown_retirement ... ok
test services::rows_duplicate_order_dedup_charge_and_delayed_consumer_retirement ... ok
test transfer::completion_refuses_each_epoch_bad_checksum_missing_fence_and_short_vector ... ok
test transfer::copy_bounds_and_consumer_context_generation_are_checked ... ok
test services::tier_advisory_lookup_identity_completeness_and_epochs ... ok
test transfer::reusable_transfer_schedule ... ok
test transfer::revision_v11_complete_cancel_and_lifetime ... ok
test transfer::revision_v11_zero_accept_preserves_owned_hosts ... ok
test transfer::revision_v11_indexed_acceptance ... ok
test transfer::transfer_cancel_after_completion_before_publication_revokes ... ok
test transfer::transfer_zero_accept_error_returns_every_input_and_empty_refuses ... ok
test transfer::transfer_ready_take_is_owner_bound_and_post_publish_cannot_revoke ... ok
test transfer::unknown_backend_shutdown_does_not_free_accepted_host_source ... ok
test transfer::unknown_cancel_and_delayed_fences_quarantine_until_retired ... ok
test transfer::transfer_partial_batch_and_broken_aggregate_are_detected ... ok
test wire::layout_duplicate_overflow_and_group_specific_counts_refuse ... ok
test wire::heterogeneous_all_trailing_groups_aliases_and_padding ... ok
test wire::placement_negative_headroom_route_labels_and_estimates_survive_wire ... ok
test wire::immutable_seal_and_committed_high_water_cannot_publish_rollback_tail ... ok
test wire::identity_domains_full_program_and_canonical_roundtrip ... ok
test rows::row_batch_larger_than_pool_progresses_and_one_bad_physical_read_refuses_all_output ... ok
test wire::golden_fixture_hashes_are_pinned ... ok

test result: ok. 44 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s

     Running tests/peer/mod.rs (target/debug/deps/peer-2abcdff8374b3a8b)

running 17 tests
test fake::capacity_grants_padding_and_homogeneous_ticket_refusal ... ok
test fake::caller_source_drop_keeps_owned_bytes_until_acknowledged ... ok
test fake::directed_fault_after_reservation_returns_owned_copies_without_acceptance ... ok
test fake::directed_fault_controls_are_independent_and_missing_routes_refuse ... ok
test fake::exact_bytes_epochs_cancel_and_fence_issuer ... ok
test fake::partial_acceptance_and_zero_accept_return_owned_inputs ... ok
test fake::quarantine_graph_pins_foreign_tickets_and_shared_budget ... ok
test fake::revision_v11_peer_capacity ... ok
test fake::revision_v11_peer_complete_cancel_and_quarantine ... ok
test fake::revision_v11_directed_capacity ... ok
test fake::revision_v11_peer_indexed_acceptance ... ok
test fake::revision_v11_peer_submit_epochs_and_zero_accept ... ok
test fake::shared_schedule_runs_on_d_byte_fake ... ok
test idle_downshift_is_deferred_active_downgrade_refuses ... ok
test observation_expires_with_context_topology_or_binary ... ok
test fake::short_corrupt_and_wrong_context_completion_refuse ... ok
test topology_is_directed_grant_bound_and_never_host_bounce ... ok

test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/placement/mod.rs (target/debug/deps/placement-699695d0bb79fdab)

running 6 tests
test staging_loader_scratch_and_routes_are_explicit ... ok
test odd_tails_and_replicas_are_counted_not_quartered ... ok
test report_is_frozen_wire_and_replica_breakdown_not_extra_charge ... ok
test invalid_geometry_and_integer_overflow_refuse ... ok
test equal_split_reproduces_census_slope_and_frontier ... ok
test alternate_bytes_and_explicit_reserve_are_not_format_fallbacks ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

     Running tests/storage/mod.rs (target/debug/deps/storage-9cc88276c49a2012)

running 30 tests
test day2::charged_pool_pins_backing_once_until_last_slice_returns ... ok
test bounded_reader_partial_acceptance_and_short_read_visibility ... ok
test cancel_late_completion_keeps_source_and_slot_until_io_retires ... ok
test corruption_header_payload_padding_truncation_and_future_version_refused ... ok
test day2::frozen_transfer_cancel_schedule_real_cpu_adapter ... ok
test day2::corrupt_backing_does_not_trap_explicit_charge_release ... ok
test day2::revision_v11_pinned_lease ... ok
test corrupt_or_missing_chunk_blocks_commit_and_read ... ok
test day2::revision_v11_pinned_quarantine ... FAILED
test day2::host_take_once_stale_triples_and_consumer_retirement ... ok
test exact_pread_eintr_partial_eof_overflow ... ok
test fake_pool_reserves_demand_headroom_and_charges_padding ... ok
test day2::mixed_epoch_batch_returns_all_owned_inputs_and_missing_chunk_refuses_publish ... ok
test day2::partial_acceptance_retains_rejected_device_and_refuses_batch_publication ... ok
test jsonl_escapes_strings_and_unknown_times_are_null ... ok
test day2::root_metadata_requires_canonical_v1_and_distinct_digest_domains ... ok
test day2::revision_v11_transfer_complete_cancel ... ok
test retirement::cancelled_transfer_needs_disk_dma_consumer_and_graph_retirement ... ok
test retirement::lost_transfer_completion_is_quarantined_on_drop ... ok
test incomplete_enospc_short_write_and_interrupted_commit_never_publish ... ok
test unknown_completion_quarantines_not_reuses ... ok
test root_collision_identity_and_immutable_conflicts_fail_closed ... ok
test day2::direct_alignment_and_short_eof_fail_closed ... ok
test publication_is_last_and_streams_object_larger_than_pool ... ok
test day2::revision_v11_object_publish_release ... ok
test real_filesystem_reopen_and_conflict_do_not_clobber ... ok
test day2::frozen_object_schedule_real_files_and_transaction_owner ... ok
test header_valid_vs_padded_and_full_integrity ... ok
test day2::persistent_restart_ignores_partial_pending_root_and_chunks ... ok
test day2::bench_real_roundtrip_reopen_restore_uncached_and_red_modes ... ok

failures:

---- day2::revision_v11_pinned_quarantine stdout ----

thread 'day2::revision_v11_pinned_quarantine' (15810598) panicked at crates/memra-tier/tests/storage/../contracts/revision_v11.rs:215:5:
assertion `left == right` failed
  left: Ok(())
 right: Err(Busy)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    day2::revision_v11_pinned_quarantine

test result: FAILED. 29 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.41s

error: test failed, to rerun pass `-p memra-tier --test storage`
```

