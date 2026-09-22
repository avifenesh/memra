# WP-B day 30: memra#423 (the budget journal's `balance_after` chain under concurrency) and memra#464 (the boot backfill's carried rows), read from the code

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `da1f59bf6` (#631) at
`a88791f92` (`research/INDEX.md` resolved as the union of both sides; `tools/check-conflict-markers.sh` OK). Every
push today whose range touches engine source is refused `UNQUALIFIED` by the #589 hook on a plain push and goes out as
`MEMRA_RELEASE_QUALIFICATION_MODE=development git push` (`UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`,
logged in `.git/memra-gate-skips.log`). **No qualification is claimed anywhere in this record**; every GPU cell is
`executed-not-qualified`; no timing is compared across cards; no default changes today; no memra engine code changes
today (section 1 says why).

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU runs under the canonical lock
`/tmp/memra-5090.lock` (`flock -n`, a busy lock is a typed `REFUSED` retried boundedly), CPU-heavy work under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, `nvidia-smi --query-compute-apps` read before and after
every boot, no process this lane did not start is touched. The cells use a synthetic tenant and a synthetic budget on a
scratch ledger; no real tenant, key, box or balance appears anywhere in this record.

## 1. Where the journal lives, and what the brief assumed that the tree does not hold

The brief names "the existing tenant-budget seams and the admin API" as engine surfaces. They are not in this tree.
`docs/FLAGS.md` rows 304 to 306 record it: `MEMRA_TENANT_BUDGETS`, `MEMRA_ADMIN_ADDR` and `MEMRA_ADMIN_TOKEN_FILE`
were REMOVED from the stock binary by lane `engine-billing-extraction-20260829`, and the stock server FAILS STARTUP if
any of them is set. What memra keeps is the seam, `crates/memra-server/src/metering.rs`: the `Metering` trait
(`enforces_limits`, `is_limited`, `reserve`, `open`, `captures`, `limits_health`, `drain_kill`) and the `Receipt` trait
(`record_prompt_usage`, `record_completion_token`, `complete`, `complete_deadline_partial`, `reject`,
`settle_unbilled`). The seam speaks tokens and verdicts; it has no balance, no journal, no row. (Its module comment
still names an in-crate `ledger::ReferenceMetering`; no such module exists in this tree. The comment is stale, not a
lead.) The implementation behind the seam, including the budget journal, `balance_after_micro`, the boot backfill
`recover_request_ledger` and the `/admin` listener, is the deployment binary in darklanes:
`serving/darklanes-metering/src/ledger.rs` (6892 lines) and `admin.rs`, linked against the memra submodule pinned in
`engine/PIN` (today `1c0900e0e`, `v0.138.0`) by `serving/darklanes-serve`. So the defect in #423 is a darklanes
defect reached through a memra seam, and the memra half of this day is the seam's contract: what the engine promises
about the ORDER in which it calls `reserve`, `complete` and drop, and what it does not.

**What the engine does at the seam, and on which task.** `admit_tenant_budget` (`lib.rs` 9006) runs on the HTTP
handler's tokio task before the worker is asked anything: `is_limited`, then `reserve(tenant, principal, model,
prompt_tokens, completion_bound)`; the returned `Permit` rides `BudgetAdmission` into `open(meta, permit)` (three
call sites: 9490 chat/completions, 10100 the planned route, 22048 the retrieval routes). Every terminal row is
written from the same handler task: `receipt.complete(usage, elapsed)` inside the SSE generator once the worker
stream ends (`lib.rs` 10597 and 11158), `complete_deadline_partial` (11029), `settle_unbilled` (7563, 9119, 10493,
11007, 11126), `reject` (9098 and the `request_ledger_unavailable` arms). A receipt dropped without a terminal call
(client walk-away) reaches the implementation's `Drop` on whatever task drops it. The engine makes NO ordering
promise across requests: N handler tasks call `reserve` and `complete` concurrently, and the seam's `&self` plus
`Send + Sync` puts every serialization decision on the implementation's side. That is the correct contract for a
counting seam, and the fix does not need it to change.

**Where the rows are produced, on what thread, and how `balance_after_micro` is computed** (darklanes
`ledger.rs`, line numbers at darklanes `main` `9ebb64252`):

| row | producer | caller task | stamp |
|---|---|---|---|
| `kind: "debit"` (request settle) | `debit_reserved` (1177) via `BudgetPermit::settle` (1401) via `PendingReceipt::finalize` (3437) | the handler task's `complete`/`complete_deadline_partial`/Drop | `current + refund` where `current = accounts[tenant].balance_micro` and `refund = reserved_micro - amount_micro` (1205) |
| `kind: "debit"` (admin) | `admin_debit` (about 1100) | the admin listener's task | `current - amount` (1133) |
| `kind: "debit"` (boot backfill) | `recover_debit` (1301) from `recover_request_ledger` (1239) | the boot thread inside `TenantBudgets::open` | `current - amount` (1325) |
| `kind: "credit"` | `credit` (985) | the admin listener's task | `current + amount` (1020) |
| (no row) | `admit` (786) via `reserve_budget` (2959) via `Metering::reserve` | the handler task at admission | `balance_micro -= reserved_micro` (the worst-case hold) |
| (no row) | `impl Drop for BudgetPermit` (1457), unsettled | whichever task drops the receipt | `balance_micro += reserved_micro` (the hold refunded) |
| (no row) | `apply_source_reload` (about 1520) | any caller of `maybe_reload` | `balance_micro += (new source total - old source total)` |

Every one of these runs under ONE lock, `TenantBudgetsInner.state: Mutex<BudgetState>`, and the journal file handle is
a field of that state (`journal_file: File`), so `append_budget_row` (1841) writes and `sync_data`s INSIDE the same
critical section that computed the stamp and then mutates `balance_micro`. **The rows are in causal order.** There is
no window in which two settles compute stamps against the same `current` and land in the other order; the issue's
sentence "each one stamps a balance it read before the others landed" describes a race the code does not have.

**What breaks the chain is the three unjournaled movements of `balance_micro`.** `balance_micro` is the AVAILABLE
balance: the settled balance minus every outstanding worst-case hold. The hold is taken at `admit` with no row and
returned at settle folded into the stamp (`current + refund`) or at drop with no row. A settle row therefore stamps
`settled - (sum of the OTHER in-flight holds at that instant)`, and a walk that expects `stamp[n] = stamp[n-1] -
amount[n]` reads a break of exactly the other holds. When those holds settle, their own rows carry the opposite
break. The breaks cancel to zero once the tenant is quiescent, and the live gap between the journal's last stamp and
the admin balance is the sum of holds outstanding at the read, which moves with traffic. Both properties #423 reports
follow from this arithmetic; the sequence in section 2 makes it exact. The third movement, a source-file top-up
applied by `apply_source_reload`, moves `balance_micro` by the source delta with no row at all; it is a second,
independent chain break (a jump equal to the top-up) and is recorded in section 5 as a gap, not fixed today.

**What the readers assume.** The engine-side replay (`replay_budget_row`, about 1763) does NOT read a debit row's
stamp: it rebuilds balances as `source total + sum(credit amounts) - sum(debit amounts)` and reads a credit row's
`balance_after_micro` only into the idempotent `CreditResult` it replays to a repeated `POST .../credit`. The snapshot
(`write_budget_snapshot`, 1939) writes `balance_micro`, the available figure, and is a derived view no boot reads.
The readers that do walk the chain are outside the engine: the box-move seed ("balance_after_micro at a cut, then
replay amount_micro after it") and the conservation audits named in #423 and #464. They assume the stamp is the
settled balance after the row. Today it is the available balance after the row, which is the same number only when
no other hold is outstanding.

## 2. The defect as a sequence, and the pre-registration (written before the first run)

Tenant `T`, source balance `B0`. Requests `A` and `B` arrive on two handler tasks; their worst-case holds are `R_A`
and `R_B`; their exact settled debits are `a <= R_A` and `b <= R_B`. Everything in the right two columns runs under
the one `BudgetState` mutex.

```
handler task A        handler task B        BudgetState (one mutex)                 journal (appended inside the mutex)
reserve(A) ------------------------------> admit:  balance = B0 - R_A               (no row)
                      reserve(B) ---------> admit:  balance = B0 - R_A - R_B         (no row)
   ... worker streams A and B concurrently ...
complete(A) -----------------------------> debit_reserved(A): refund_A = R_A - a
                                             stamp_A = balance + refund_A
                                                     = B0 - a - R_B                  row 1: debit amount=a balance_after=B0 - a - R_B
                                             balance = B0 - a - R_B
                      complete(B) --------> debit_reserved(B): refund_B = R_B - b
                                             stamp_B = (B0 - a - R_B) + (R_B - b)
                                                     = B0 - a - b                    row 2: debit amount=b balance_after=B0 - a - b
                                             balance = B0 - a - b

chain walk expects   row 1: B0 - a            reads B0 - a - R_B     break  -R_B
                     row 2: (B0 - a - R_B) - b reads B0 - a - b       break  +R_B     (cancels)
live read between row 1 and row 2: admin balance B0 - a - R_B, journal last stamp B0 - a - R_B, source + sum(rows) B0 - a:
                     gap = -R_B, and it moves with every hold taken or released
```

The variant with a walk-away: `reserve(A)`, `reserve(B)`, `drop(B)` (hold `R_B` returned, no row), `complete(A)`:
row 1 stamps `B0 - a` and the chain holds, so the defect is invisible exactly when the peer never settles; a chain
that holds on one tape is not evidence the stamp is causal. The variant with a credit while a hold is out:
`reserve(A)`, `credit(c)` stamps `B0 - R_A + c`, `complete(A)` stamps `B0 + c - a`; the walk reads `-R_A` at the
credit row and `+R_A` at the debit row.

**Why the fix is bounded and where it lands.** The seam that already owns the balance is `BudgetState` under its one
mutex; every producer already computes and appends there. What is missing is the per-tenant sum of outstanding
holds, which the state tracks per KEY (`key_reserved`) but not per TENANT. With a `tenant_reserved: HashMap<String,
u64>` maintained at `admit` (+hold), `debit_reserved` (-hold) and the unsettled `Drop` (-hold), every stamp becomes
`balance_micro_after + tenant_reserved_after`, the SETTLED balance after the row, and the chain
`stamp[n] = stamp[n-1] + signed_amount[n]` holds by construction under any interleaving of reserve, settle, drop
and credit, because a hold's take and return never touch it. No new field on the row, no new `kind`, no new numeric
program, no flag, no thread: a reader of old rows parses new rows unchanged (BUDGET-JOURNAL.md property 1). It is a
change of what the stamp MEANS (settled, not available); the admin `balance_micro` reply and the snapshot keep the
available meaning, and section 5 states that divergence for the owner. The fix lives in darklanes `ledger.rs`, not
here; this lane carries it on a darklanes lane branch with its CPU tests and does not merge it (the owner decides,
same posture as this memra lane).

**Pre-registered cell (local RTX 5090, one boot per arm, executed-not-qualified).** Binary: `darklanes-serve` from
darklanes `main` `9ebb64252` against engine pin `1c0900e0e` (`v0.138.0`), built in this lane's darklanes worktree
(`serving/target/release/memra-server`, sha256 `7aa417abfa264c7ab60d73f6dbd6f0bacbe53460ee829e477c2b4ae7488888bb`
for the BEFORE arm; the AFTER arm's hash is recorded with its receipt). Model: the 9B NVFP4 GGUF the serve smoke
uses, alias `journal9`, `MEMRA_COMPAT=openai`. Accounting: `MEMRA_MODEL_METADATA` a synthetic TOML with one priced
model; `MEMRA_REQUEST_LEDGER` and `MEMRA_TENANT_BUDGETS` on a scratch dir; ONE synthetic tenant `cell-tenant` seeded
at `1_000_000_000` micro; `MEMRA_API_KEYS` inline with one random key hashed; the admin listener on loopback with a
mode-0600 token file. Load: `N=16` concurrent `POST /v1/chat/completions` (`max_tokens 48`, `temperature 0`,
`stream false`, 16 distinct prompts) fired at once from one driver; while they are in flight, one `POST
/admin/tenants/cell-tenant/credit` of `1000` micro (idempotency key `cell-credit-1`). After every request returns
and the tenant is quiescent, the driver reads the journal (`<ledger>.tenant-budget-journal.jsonl`, sealed segments
first, none expected) and `GET /admin/tenants/cell-tenant/balance`.

Assertions, verbatim before the run, evaluated by `research/spill-b-20260919/journal-order-cell.py`:

- `C1 order`: `unix_ms` is nondecreasing across rows (the writer holds the lock through the append; a violation is
  a clock step, reported not failed).
- `C2 chain`: with `signed(row) = +amount` for `credit` and `-amount` for `debit`, `stamp[0] == 1_000_000_000 +
  signed(row 0)` and for every `n >= 1` `stamp[n] == stamp[n-1] + signed(row n)`. Every violation is printed as
  `CHAIN n=<n> kind=<kind> amount=<a> prev=<stamp[n-1]> stamp=<stamp[n]> expected=<e> delta=<stamp - e>`.
- `C3 conservation at quiescence`: `admin balance_micro == 1_000_000_000 + sum(signed)` and `== stamp[last]`.
- `C4 cancellation`: `sum(delta over C2 violations) == 0`, reported.
- Verdict line: `JOURNAL-ORDER CELL arm=<before|after> N=16 rows=<r> chain_violations=<v> order_violations=<o>
  conservation=<ok|FAIL> cancel_sum=<s> -> <PASS|FAIL>` where PASS is `v == 0` and `conservation=ok`.

Expected before the fix, from section 1's arithmetic: `C2` violations at every settle row that had at least one peer
hold outstanding (up to 16 of 17 rows), `C4` sum `0`, `C3` ok (the available balance and the settled balance agree
once every hold is gone). Expected after: `chain_violations=0`. Both arms boot the same model, the same 16 prompts,
the same credit; nothing about the engine differs between them. A `REFUSED` (lock busy, model missing, port owned by
a foreign process, no admin readiness) is a typed line and is retried boundedly; it is never a verdict.
