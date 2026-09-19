# WP-C v1.1 binding fix

Repository: **avifenesh/memra**. Branch: `lane/spill-c-20260919`.
Local lane only: no main integration, push, PR, tag, deployment, or GPU qualification.

## Source and decision

- Day-3 base: `e7188cd6ad85fc1fd7e272302a5d6fdedb7b5aa7`.
- Merged frozen lead tip `ce49cd863d36d317993c4fc71933efe5ac7b9489` with
  `git merge --no-ff lane/spill-lead2-20260919`.
- Merge commit: `30066e0b`. The sole append conflict in
  `crates/memra-tier/tests/bank/main.rs` retained both the day-3 helpers/integration
  module and the four v1.1 tests, unchanged, before reproducing the failures.
- Binding fix and exact source commit checked again after commit:
  **`4b50d980234ea75d7302542ca5f52fb3458a7ad0`**.

The diagnosis is confirmed: stage/gather enqueue work; `completion()` observes but
never executes that work. The new schedule callbacks inspected pending completions
and zero reader calls, because the lead bindings targeted day-2 synchronous code.

**Binding-only fix:** six callbacks now invoke the existing test-only `drain`
helper before observation: bank/row complete-before-cancel, bank uniform proof,
row bytes/dedup/straddles, and bank/row failed siblings. The helper invokes actual
`BankService::progress`, at most 10,000 times, and fails if the producer does not
terminate. Each progress step remains one bounded host read; completion includes
actual read/checksum failures rather than fabricated completion values.

Production stage/gather remain enqueue-only. Completion does not publish, retire,
or release resources. The schedules still explicitly test publication refusal,
cancellation, stale epochs, byte identity, duplicate capabilities, Uniform source
classification, partial-read failure, and delayed explicit retirement/release.

**Schedule proposal: none.** Existing mutable completion/observation hooks already
express the required completion point for an enqueue-then-progress design. No
changes to production implementation, frozen contracts, schedules, fixture pins,
root manifests, dependencies, flags, or any other lane were made by this fix.
`git diff --exit-code 30066e0b -- crates/memra-tier/src/contracts.rs
crates/memra-tier/tests/contracts crates/memra-tier/src/bank` passed.

## Checks (all actually executed)

Raw stdout+stderr (lossless `.log.gz`): [`v11-binding/`](v11-binding/); exact argv, exits, log SHA-256,
UTC, and checked source commit: [`checks.json`](v11-binding/checks.json).
Before/after bank outputs below are verbatim; the full suite was re-executed on
the committed fix. No stderr was discarded or parsed before capture.

| Command | Result |
|---|---|
| `cargo test -p memra-tier --offline --test bank` before | exit 101; 31 pass / 4 fail |
| Same command after | exit 0; 35 pass / 0 fail |
| `cargo fmt --all -- --check` | exit 0 |
| `cargo check -p memra-tier --offline --all-targets` | exit 0; macOS |
| Same check + `--target x86_64-unknown-linux-gnu` | exit 0; compile check only, no Linux runtime/link qualification |
| `cargo test -p memra-tier --offline` | exit 101; stops at inherited D failure |
| Same test + `--no-fail-fast` | exit 101; 132 pass / 2 fail / 0 ignored, including 4 doctests |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0 |
| `git diff --check` | exit 0 |
| `bash tools/check-flags.sh` | exit 0; 864 reads, none uncovered |

The full-suite failures are **A and D**, not only D. The normal required test run
stops at D and never reaches A's storage target. The no-fail-fast run reaches all
targets: bank 35, contracts 44, peer 14/1, placement 6, storage 29/1, doctests 4.
Both failures remain strict and are outside C's scope; the lead was notified.
Lead must integrate A/D fixes and rerun the integrated battery. This lane is not
claiming a green full crate or native serving readiness.

No GPU/nvcc, Linux execution, O_DIRECT runtime, model/engine/server, CUDA events,
graphs, or serving measurements ran. CPU producer completion is not GPU readiness.
Approximate effort: **0.15 agent-hours**, no GPU time.

## Before: verbatim bank output

```text
   Compiling memra-tier v0.138.0 (/Users/avifen/tiyuvta/wt-spill-c/crates/memra-tier)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 1.67s
     Running tests/bank/main.rs (target/debug/deps/bank-bd6b64183c1f3fb4)

running 35 tests
test alternate_layout_cannot_resurrect_masked_original_identity ... ok
test discontiguous_row_scale_planes_are_required_and_exact ... ok
test bounds_wrong_ids_overflow_and_device_request_fail_closed ... ok
test all_three_epochs_cross_service_and_publish_replay_refuse ... ok
test frozen_bank_cancel_schedule_runs_on_implementation ... ok
test borrowed_view_and_alias_survive_busy_retirement ... ok
test actual_q2_q3_nvfp4_codes_are_preserved_without_kernel_promotion ... ok
test cancel_retains_charge_until_retirement_and_tombstone_ack ... ok
test host_bridge::bridge_uses_per_record_metadata_and_preserves_existing_scale_planes ... ok
test integration::ngram_hint_is_byte_bounded_deduplicated_and_domain_separate ... ok
test failed_transfer_completion_enumerates_all_records_and_segments ... ok
test host_exps_uniform_mixed_original_mask_and_split_offsets ... ok
test missing_or_extra_checksums_and_false_uniform_declarations_refuse ... ok
test frozen_row_order_schedule_runs_on_implementation ... ok
test overlapping_cache_hit_tickets_hold_same_allocation_until_all_retire ... ok
test read_planning_refuses_undeclared_tail_padding_and_invalid_granularity ... ok
test original_masks_foreign_catalog_and_wrong_domain_refuse_before_reads ... ok
test integration::shared_governor_hints_cannot_starve_mandatory_loads ... ok
test pinned_ple_test_oracle_matches_native_function_bodies ... ok
test refused_publication_retains_owned_charge_until_explicit_retirement ... ok
test predictions_are_bounded_mask_checked_and_not_demand_heat ... ok
test row_release_busy_preserves_retry_after_partial_alias_retirement ... ok
test revision_v11_corrupt_sibling_and_row_namespace ... FAILED
test revision_v11_bank_cancel_identity_uniform ... FAILED
test revision_v11_rows_complete_cancel ... FAILED
test shared_budget_atomic_refusal_and_tombstone_bound ... ok
test uniform_proof_checks_actual_source_even_homogeneous_subset ... ok
test short_and_corrupt_reads_never_publish_partial_batch ... ok
test sparse_amplification_and_default_policy_are_explicit_arithmetic_not_io_benchmarks ... ok
test zero_small_full_cache_preserves_exact_bytes_and_scales ... ok
test revision_v11_rows_bytes_dedup_bounded_straddles ... FAILED
test synthetic_ple_trace_preserves_chunk_rewind_eos_and_native_expansion_bits ... ok
test integration::object_transfer_experts_partial_failure_never_publishes_successful_sibling ... ok
test rows_straddles_duplicates_pool_smaller_than_batch_and_retirement ... ok
test integration::object_store_transfer_rows_forced_misses_order_and_retirement ... ok

failures:

---- revision_v11_corrupt_sibling_and_row_namespace stdout ----

thread 'revision_v11_corrupt_sibling_and_row_namespace' (15822714) panicked at crates/memra-tier/tests/bank/../contracts/revision_v11.rs:823:5:
assertion failed: c.items.iter().flat_map(|i|
            &i.segments).any(|s|
        s.error.is_some() || s.status == ItemStatus::Failed)

---- revision_v11_bank_cancel_identity_uniform stdout ----

thread 'revision_v11_bank_cancel_identity_uniform' (15822713) panicked at crates/memra-tier/tests/bank/../contracts/revision_v11.rs:263:5:
assertion failed: c.producer_done
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- revision_v11_rows_complete_cancel stdout ----

thread 'revision_v11_rows_complete_cancel' (15822716) panicked at crates/memra-tier/tests/bank/../contracts/revision_v11.rs:287:5:
assertion failed: c.producer_done

---- revision_v11_rows_bytes_dedup_bounded_straddles stdout ----

thread 'revision_v11_rows_bytes_dedup_bounded_straddles' (15822715) panicked at crates/memra-tier/tests/bank/main.rs:1337:13:
assertion `left == right` failed
  left: 0
 right: 4


failures:
    revision_v11_bank_cancel_identity_uniform
    revision_v11_corrupt_sibling_and_row_namespace
    revision_v11_rows_bytes_dedup_bounded_straddles
    revision_v11_rows_complete_cancel

test result: FAILED. 31 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.38s

error: test failed, to rerun pass `-p memra-tier --test bank`
```

## After: verbatim bank output

```text
   Compiling memra-tier v0.138.0 (/Users/avifen/tiyuvta/wt-spill-c/crates/memra-tier)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.76s
     Running tests/bank/main.rs (target/debug/deps/bank-bd6b64183c1f3fb4)

running 35 tests
test alternate_layout_cannot_resurrect_masked_original_identity ... ok
test cancel_retains_charge_until_retirement_and_tombstone_ack ... ok
test frozen_bank_cancel_schedule_runs_on_implementation ... ok
test bounds_wrong_ids_overflow_and_device_request_fail_closed ... ok
test borrowed_view_and_alias_survive_busy_retirement ... ok
test discontiguous_row_scale_planes_are_required_and_exact ... ok
test failed_transfer_completion_enumerates_all_records_and_segments ... ok
test actual_q2_q3_nvfp4_codes_are_preserved_without_kernel_promotion ... ok
test host_bridge::bridge_uses_per_record_metadata_and_preserves_existing_scale_planes ... ok
test host_exps_uniform_mixed_original_mask_and_split_offsets ... ok
test integration::ngram_hint_is_byte_bounded_deduplicated_and_domain_separate ... ok
test frozen_row_order_schedule_runs_on_implementation ... ok
test missing_or_extra_checksums_and_false_uniform_declarations_refuse ... ok
test all_three_epochs_cross_service_and_publish_replay_refuse ... ok
test original_masks_foreign_catalog_and_wrong_domain_refuse_before_reads ... ok
test read_planning_refuses_undeclared_tail_padding_and_invalid_granularity ... ok
test overlapping_cache_hit_tickets_hold_same_allocation_until_all_retire ... ok
test refused_publication_retains_owned_charge_until_explicit_retirement ... ok
test integration::shared_governor_hints_cannot_starve_mandatory_loads ... ok
test predictions_are_bounded_mask_checked_and_not_demand_heat ... ok
test pinned_ple_test_oracle_matches_native_function_bodies ... ok
test revision_v11_rows_complete_cancel ... ok
test revision_v11_bank_cancel_identity_uniform ... ok
test row_release_busy_preserves_retry_after_partial_alias_retirement ... ok
test short_and_corrupt_reads_never_publish_partial_batch ... ok
test shared_budget_atomic_refusal_and_tombstone_bound ... ok
test uniform_proof_checks_actual_source_even_homogeneous_subset ... ok
test sparse_amplification_and_default_policy_are_explicit_arithmetic_not_io_benchmarks ... ok
test revision_v11_corrupt_sibling_and_row_namespace ... ok
test revision_v11_rows_bytes_dedup_bounded_straddles ... ok
test zero_small_full_cache_preserves_exact_bytes_and_scales ... ok
test integration::object_transfer_experts_partial_failure_never_publishes_successful_sibling ... ok
test synthetic_ple_trace_preserves_chunk_rewind_eos_and_native_expansion_bits ... ok
test rows_straddles_duplicates_pool_smaller_than_batch_and_retirement ... ok
test integration::object_store_transfer_rows_forced_misses_order_and_retirement ... ok

test result: ok. 35 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.39s

```

## Remaining failures: verbatim excerpts from committed-source full run

```text
---- fake::revision_v11_directed_capacity_lane_fix_required stdout ----

thread 'fake::revision_v11_directed_capacity_lane_fix_required' (15827651) panicked at crates/memra-tier/tests/peer/../contracts/revision_v11.rs:567:18:
denying one directed route must not deny its reverse: Unsupported
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    fake::revision_v11_directed_capacity_lane_fix_required

test result: FAILED. 14 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

```text
---- day2::revision_v11_pinned_quarantine stdout ----

thread 'day2::revision_v11_pinned_quarantine' (15827687) panicked at crates/memra-tier/tests/storage/../contracts/revision_v11.rs:215:5:
assertion `left == right` failed
  left: Ok(())
 right: Err(Busy)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


failures:
    day2::revision_v11_pinned_quarantine

test result: FAILED. 29 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.27s
```
