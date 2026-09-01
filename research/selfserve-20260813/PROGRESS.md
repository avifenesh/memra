# Tenant-budget and admin-seam progress

Date: 2026-08-13
Branch: `lane/cx-selfserve`
Pinned base: `v0.81.2` (`18885ec47`)

## Scope and gates

- [x] Add optional durable per-tenant micro-USD budgets without changing unbudgeted tenants.
- [x] Debit completed and abandoned ledger usage exactly once; reject exhausted tenants at HTTP admission with OpenAI-style 402.
- [x] Add a default-off, token-file-authenticated admin listener for key and tenant operations.
- [x] Cover exact cached-token charging, idempotent credits, journal replay, admin auth, and key hot reload.
- [x] Document the three environment variables, file semantics, and integer rounding rule.
- [x] Pass the focused memra-server tests, including HTTP budget and admin surfaces.
- [x] Pass final `cargo test --workspace -- --test-threads=1` and the flags drift gate.
- [x] Pass the full flock-coordinated box1 serving battery and one-cent exhaustion smoke.
- [x] Commit the final code, docs, progress, and raw evidence locally; do not tag, push, or merge.

## Constraints held

- Money is represented as integer micro-USD at the budget boundary; the existing ledger `Decimal`
  calculation remains the cost source of truth.
- `worker.rs` is outside this lane. Admission enforcement belongs in `main.rs`.
- No performance-board edits and no verification bypasses.
- Public names remain generic tenant-budget/admin terminology; product users, payments, and UI
  remain outside this repository.

## Timeline

- 2026-08-13: Started from a clean `lane/cx-selfserve` at the exact `v0.81.2` tag. Read
  `/home/avifenesh/.lanectl/inbox/cx-selfserve.md`; it matches the owner brief and contains no
  additional steering. Created this progress ledger before engine or documentation edits.
- 2026-08-13: Implemented additive TOML/JSONL budget grants, two-second hot reload, one-request
  exposure permits, exact-ledger terminal debits, append-and-sync debit/credit journaling,
  request-id crash repair, and compact balance snapshots. Positive fractional micro-USD costs
  round upward once at the request boundary.
- 2026-08-13: Added the loopback-only default-off admin router with token-file bearer auth,
  pre-handler audit records, key create/revoke, UTC-day usage, idempotent credit, and balance
  endpoints. Added focused router/auth, hot-reload, cached-price, abandonment, idempotency,
  crash-replay, usage aggregation, and HTTP 402 tests; focused test selections pass.
- 2026-08-13: Documented the three new flags and the generic self-serve contracts in
  `docs/FLAGS.md` and `docs/SERVING.md`.
- 2026-08-13: Inbox steering identified two admission/reload gaps. Replaced the positive-balance
  check with an atomic worst-case tariff reservation over the exact HTTP-rendered prompt and
  maximum completion bound; terminal worker-truth cost now settles the hold and refunds the
  remainder. Budgeted tenants are limited to one active request, bounding a pre-terminal crash to
  one request. Reload failures now retry the same source, expose operator counters, and fail
  admissions closed after three consecutive polls. Focused `cargo test -p memra-server` passes
  238/238 after these changes.
- 2026-08-13: Final local `cargo test --workspace -- --test-threads=1` and
  `tools/check-flags.sh` pass. `cargo clippy -p memra-server --no-deps` also exits zero; its 37
  warnings are confined to the pre-existing constrained/health/worker/main baseline, with none in
  the new admin or ledger code. Raw local logs are under `raw/local/`. The remote battery remains
  pending.
- 2026-08-13: A final crash-safety audit found that the first journal/audit file creation synced
  file data but not the containing directory. Added parent-directory sync for newly created
  append-only files and refreshed the full workspace/flags evidence before staging the immutable
  box1 candidate.
- 2026-08-13: Tightened the box1 wrapper's tenant-clean assertions to require both an empty CUDA
  compute-app list and no surviving `memra-server` process before or after the battery. The smoke
  persists only the generated key prefix and explicitly scans audit output for all bearer values.
- 2026-08-13: The first tenant-clean handoff correctly found no CUDA process but the initial
  process predicate matched an unrelated cleanup shell whose argument text mentioned
  `memra-server`. Narrowed it to exact process-name matching; the false-positive attempt stopped
  before build/GPU work and remains remote-only diagnostic evidence.
- 2026-08-13: A restarted lock owner expanded its active matrix after this lane queued. Made the
  harness lock wait configurable (`SELFSERVE_GPU_LOCK_WAIT_S`) with a 12-hour default so a healthy
  exclusive queue cannot expire before the requested battery begins.
- 2026-08-13: The first complete candidate run passed build, kernel-check, both run-gen arms,
  both K=1..8 run-spec arms, and the underlying serve-smoke (exit 0), but the wrapper copied the
  smoke server's `/tmp/serve-smoke.log` over the gate's stdout summary before checking it. Fixed
  evidence collection to select only current-run temp logs and store them under collision-free
  `tmp-*` names; a full rerun remains required.
- 2026-08-13: A concurrent lane was observed launching CUDA processes without inheriting the
  shared flock. Added a one-second compute-app monitor: any process outside this candidate's own
  `target/release` tree is recorded with timestamp/PID/path/memory and invalidates the battery,
  even if it exits before the final tenant-clean snapshot.
- 2026-08-13: The first monitored rerun reached green verdicts through the one-cent smoke, but the
  monitor classified this checkout's own `target/release/kernel-check` display string as foreign
  because `nvidia-smi` omitted its absolute prefix. The monitor now resolves each PID through
  `/proc/<pid>/exe`, preserving cross-lane rejection even when two checkouts use the same relative
  binary name. The attempt is diagnostic only because its monitor stopped at the false hit.
- 2026-08-13: Resume inspection found the next run had again passed every underlying gate before
  the monitor rejected a just-exited candidate PID: `readlink -f` weakly canonicalized the missing
  final `/proc/<pid>/exe` component into a literal path. The monitor now reads the live symlink
  directly, records already-vanished contexts separately, and still rejects every live PID whose
  executable is unreadable or outside this candidate tree.
- 2026-08-13: The direct-symlink rerun passed every functional gate but found the second terminal
  process race: a candidate `run-spec` PID remained as a zombie while `nvidia-smi` reported its
  torn-down context as `[No data]`, so `/proc/<pid>` existed without an `exe` link. Terminal Z/X
  contexts now join vanished contexts in the diagnostic log; every unresolved live context still
  fails closed. This attempt is diagnostic because the ownership assertion did not complete.
- 2026-08-13: Final box1 candidate `fdb70fc46c8399fd1f024ec34eaf6dfe6c38e853` held the single
  `/tmp/memra-gpu.lock` from clean admission through clean shutdown and passed all gates:
  kernel-check `ALL GREEN` (100 cells, 5 skipped), Q27/Q35 run-gen MATCH x2, Q27/Q35 run-spec
  K=1..8 PASS, serve-smoke `0 failed`, and serve-stress `ALL GREEN (c=64)`. The ownership monitor
  recorded no foreign process and no query errors; its three diagnostic rows are terminal Z-state
  candidate contexts. Both CUDA compute-app and `memra-server` pre/post snapshots are empty.
- 2026-08-13: The live one-cent smoke refused an oversized reservation with 402, admitted the
  bounded request, reconciled Decimal cost `0.009999004` USD to an exact ceil debit of 10,000
  micro-USD and zero balance, then returned OpenAI-style 402 for the next request. Admin 401/403,
  five secret-free audit rows, ledger/usage agreement, and tenant-clean shutdown all passed.
  Accepted raw is under `raw/box1/`; `MANIFEST.sha256` verifies and hashes to
  `77f5c6488d784524a06fc355e146a28664326c80ca67f9077e1c39e07d5c7be8`.

## Named follow-up

- `selfserve-journal-retention-v2`: v1 keeps debit request ids and credit idempotency keys for the
  process lifetime and retains the append-only journal indefinitely. Define a snapshot format with
  an authoritative request-ledger checkpoint before pruning; then retain at least 90 days of debit
  and idempotency replay history. Do not prune v1 in place, because startup request-ledger repair
  could otherwise reapply an older debit.
