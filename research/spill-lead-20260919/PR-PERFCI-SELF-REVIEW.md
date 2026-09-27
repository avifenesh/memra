# perfci lane self-review (lead, 2026-09-21)

Read in full: `crates/memra-engine/src/lib.rs` (`test_support`), the three pair-only tests' heads
(`dsv4_graph.rs`, `model_memory.rs`), `tools/local-ci.sh` (lib stage, `run_cell`).

## Findings
1. **No assertion relaxed.** `skip_unless_native_pair` returns early only when the driver reports fewer than two
   devices; on the development pair the three tests run exactly as before and their `expect("... never skip")` calls
   still fire on any error. A one-device rig could never have executed them; it now says so on one line per test.
2. **The skip is counted.** `--show-output` surfaces the `SKIP-PAIR` lines from passing tests; local-ci greps them
   from a tee'd copy and prints the count and names, so the battery log accounts for every skip (#484). The stage's
   exit code comes from `PIPESTATUS[0]`, not from `tee`.
3. **`cudarc::driver::result::init()` before `get_count()`**: idempotent, same call `Engine::new` makes; a driver that
   fails to initialise reports zero devices and the pair tests skip with `found 0`, which is the truthful reading on
   a rig with no CUDA.
4. **Spec cells without a drafter.** `run_cell` now checks `$MODELS/$draft` (and the ranks file when named) before
   the model runs and prints `SKIP (no draft at ...)`, the vocabulary the model check already uses; the earlier
   `FAIL (no reading)` for the same situation discarded gemma-gate's stderr (`out=$(... 2>&1 || true)` never printed),
   against the evidence rule "never let a pipe swallow error output". A cell that really produces no reading now
   prints the last six lines of its output next to the verdict.
5. **Rows.** The failed second run already appended honest rows for `26b-plain-short` (208.85 tok/s) and
   `qwen9b-plain-short` (139.09 tok/s), `window_clean:true`, at the fixed tree `2463ff524`; the third run appends
   its own rows. Both sets stay in the append-only log; the board is unaffected (perf-ci.jsonl is the gate's record,
   not a published surface).

## Verification this review relied on
`lib-suite-after-fix.log`: `test result: ok. 24 passed; 0 failed`. Attempt-2 log: correctness GREEN, serve-smoke 0
failed, serve-stress green, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, `local-ci: SKIP 3 pair-only GPU test(s)`,
perf stage with the drafter-less cell as the only fail. Attempt 3 (final log in `perfci-main/`) with the SKIP fix.
