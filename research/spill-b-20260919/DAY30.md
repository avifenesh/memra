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

## 3. The BEFORE arm (local RTX 5090, executed-not-qualified)

Receipts: `rtx5090-day30/before/` (`summary.json`, `journal.jsonl`, `requests.jsonl`, `server.log`, `budgets.toml`,
`metadata.toml`, `card-before.txt`, `card-after-boot.txt`, `card-after-stop.txt`). Binary sha256
`7aa417abfa264c7ab60d73f6dbd6f0bacbe53460ee829e477c2b4ae7488888bb` (darklanes `main` `9ebb64252`, engine pin
`1c0900e0e`). Card at boot: `15 MiB` used, no compute app; the same after stop. The first attempt did not boot: the
deployment binary refuses an inline keyring on the admin listener (`[admin] FATAL: MEMRA_ADMIN_ADDR requires
file-backed MEMRA_API_KEYS (inline keys cannot be provisioned)`, kept as `attempt1-inline-keys-server.log`); the driver
now writes a mode-0600 `[[keys]]` TOML. Boot to readiness `4.0 s` (the 9B was page-resident), 16 requests in `1.9 s`,
`ok=16/16`, every request `prompt_tokens 23, completion_tokens 48`, credit `200`. The journal read back (17 rows: the
credit first, then 16 debits of 76 to 78 micro), verbatim:

```
CHAIN n=0 kind=credit amount=1000 prev=1000000000 stamp=999999767 expected=1000001000 delta=-1233
CHAIN n=1 kind=debit amount=78 prev=999999767 stamp=999999767 expected=999999689 delta=78
CHAIN n=2 kind=debit amount=78 prev=999999767 stamp=999999767 expected=999999689 delta=78
CHAIN n=3 kind=debit amount=77 prev=999999767 stamp=999999767 expected=999999690 delta=77
  ... (rows 4 to 15 the same shape, delta 76 to 78) ...
CHAIN n=16 kind=debit amount=76 prev=999999767 stamp=999999767 expected=999999691 delta=76
requests ok=16/16 credit_status=200 boot_s=4.0 load_s=1.9
JOURNAL-ORDER CELL arm=before N=16 rows=17 chain_violations=17 order_violations=0 conservation=ok cancel_sum=0 -> FAIL
```

Reading, against section 2: the credit landed 0.5 s after the 16 admits and before any settle, so its stamp is
`seed + 1000 - 1233` where `1233` is the sum of the 16 outstanding worst-case holds (`delta=-1233`, one row, the
`-R_B` of the diagram sixteen times over). Every settle row then stamps `999999767` unchanged: with `max_tokens`
reached, each request's exact debit equals its hold, the refund is 0 or 1 micro, and `balance_micro` does not move
at settle because the hold already moved it at admit; the walk expects a decrement of the amount and reads `+amount`
instead (`delta` 76 to 78). `cancel_sum=0`: the sixteen positive deltas sum to the one negative one, `C4` as
pre-registered. `conservation=ok`: `seed + sum(signed) = 999999767 = admin balance = last stamp`, `C3` as
pre-registered. `order_violations=0`: the rows are in causal order. This is #423's tape in miniature: an issue that
reads as "rows not in causal order" and is in fact "the stamp is the available balance, which the peers' holds move
without rows."

## 4. The fix (darklanes `lane/budget-journal-order-20260922`, commit `20f8140e8`, not merged) and the AFTER arm

`serving/darklanes-metering/src/ledger.rs`, one file, `+177 -4` before formatting. `BudgetState` gains
`tenant_reserved: HashMap<String, u64>` (per tenant, the twin of `key_reserved`), maintained at `admit` (plus the
hold), `debit_reserved` (minus this hold, written after the append succeeds so a failed append leaves state as
before) and the unsettled `Drop` (minus the hold). Two helpers, `tenant_held(state, tenant)` and
`settled_stamp(held_after, balance_after)`, and every row producer (`credit`, `admin_debit`, `debit_reserved`,
`recover_debit`) stamps `balance_after_micro = balance_after + tenant_reserved_after`: the settled balance after the
row. The row struct's doc names the meaning and what older rows meant. No new field, kind, file, flag, thread or
numeric program; the admin `balance_micro` reply and the snapshot keep the available balance (what the next admission
sees). CPU tests, from the diagram, in `mod tests`: `journal_stamp_is_the_settled_balance_under_interleaved_holds`
(reserve A 100, reserve B 200, settle A 30, settle B 50: row A must stamp `999_970`, the walk from `1_000_000` ends at
`999_920`), `journal_stamp_ignores_a_walk_away_hold` (reserve A, reserve B, drop B, settle A),
`credit_and_admin_debit_rows_stamp_the_settled_balance_while_a_hold_is_outstanding` (the admin reply keeps
`1_000_900` available while the row stamps `1_001_000`), `journal_chain_walks_under_threaded_reserve_settle_and_drop`
(8 threads x 40 iterations of admit, settle or drop, then the walk). On the unfixed ledger
(`darklanes-metering-tests-red.log`), verbatim:

```
test ledger::tests::journal_stamp_ignores_a_walk_away_hold ... ok
test ledger::tests::journal_stamp_is_the_settled_balance_under_interleaved_holds ... FAILED
test ledger::tests::credit_and_admin_debit_rows_stamp_the_settled_balance_while_a_hold_is_outstanding ... FAILED
test ledger::tests::journal_chain_walks_under_threaded_reserve_settle_and_drop ... FAILED
assertion `left == right` failed: row A stamps the SETTLED balance (seed - a), not the available one (seed - a - R_B)
  left: Some(999770)
 right: Some(999970)
117 chain breaks, first: Some("n=2 kind=\"debit\" amount=78 prev=99999884 stamp=Some(99999743) expected=99999806")
test result: FAILED. 1 passed; 3 failed; 0 ignored; 0 measured; 88 filtered out; finished in 0.03s
```

The walk-away test passing on the unfixed code is the diagram's second variant: a hold that is dropped never shows
in the chain, so a tape with walk-aways only cannot expose the stamp. After the fix
(`darklanes-metering-tests-green.log`): `test result: ok. 92 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`
(88 existing tests unchanged, none re-pinned: no existing test held two permits at once). `cargo clippy -p
darklanes-metering --lib --tests -- -D warnings` clean (`clippy_rc=0`). `cargo fmt -p darklanes-metering -- --check`
listed my test lines (formatted before the commit) and pre-existing drift in `admin.rs` and `capture.rs` that this
lane does not touch (recorded in section 5).

AFTER arm run 1 (the pre-format tree, whitespace-identical code; binary sha256
`d0c811843d85b64a26978e0241ae4a52f549952bc234aec3e1b1609a270b86c4`; receipts `rtx5090-day30/after-run1-prefmt-tree/`),
same model, prompts, credit and driver; card `15 MiB`, no compute app at boot and after stop; `ok=16/16`, credit `200`,
`boot_s=4.0 load_s=1.9`, verbatim:

```
JOURNAL-ORDER CELL arm=after N=16 rows=17 chain_violations=0 order_violations=0 conservation=ok cancel_sum=0 -> PASS
```

The credit row now stamps `1_000_001_000` and the 16 debit rows step down by their amounts to `999_999_767`; the
admin balance after quiescence is the same `999_999_767` in both arms (the fix moves no money; `conservation=ok` in
both). Run 2 on the committed (formatted) tree is appended below when it lands.

AFTER arm run 2, the committed darklanes tree `20f8140e8` (binary sha256
`900f5de230ea478cfc1b3ca56d242a9ae7aaa7c8e894310638ad72997632c498`; receipts `rtx5090-day30/after/`, with
`build-chain.log`): card `15 MiB`, no compute app at boot and after stop; `ok=16/16`, credit `200`, `boot_s=4.0
load_s=1.9`; the journal's first rows `credit 1000 -> 1000001000`, `debit 78 -> 1000000922`, last `debit 76 ->
999999767`; verbatim:

```
JOURNAL-ORDER CELL arm=after N=16 rows=17 chain_violations=0 order_violations=0 conservation=ok cancel_sum=0 -> PASS
```

Three boots, one binary per arm, the same 16 prompts and the same credit: BEFORE `chain_violations=17 -> FAIL`,
AFTER `chain_violations=0 -> PASS` twice. Executed, not qualified; N=1 boot per arm is a mechanism check (the chain
holds by construction under the fix, section 2), not a performance claim, and no timing is compared.

## 5. Gaps and owner decisions, stated rather than taken

1. **The stamp's meaning changed, the schema did not.** `balance_after_micro` now means the settled balance after the
   row (source totals plus credits minus settled debits); before, the available balance (net of the other in-flight
   holds). Same field, same row shape, same `kind`s, same file. The admin `GET .../balance` reply and the snapshot
   keep the available meaning (what the next admission sees), so a live audit comparing "journal last stamp" to
   "admin balance" reads the outstanding holds as their difference, by design, and matches at quiescence. The
   idempotent replay of a repeated `POST .../credit` after a restart answers the row's stamp (settled at that time)
   where it answered the available figure before; both are stale by construction. If the owner wants the journal to
   keep the available meaning instead, the chain cannot be walkable without journaling holds (a new `kind`, which
   BUDGET-JOURNAL.md property 1 forbids without a format decision); that is the fork this lane did not take.
2. **Source-file top-ups have no row.** `apply_source_reload` moves `balance_micro` by the source delta and writes
   nothing; the chain jumps by the top-up at the next row, before and after today's fix. Options for the owner: a
   `credit` row per reload delta with an idempotency key derived from the tenant and the new source total (no new
   kind, rollback-safe, replays as money exactly once), or the audits fold the source totals into the walk (they read
   the source file already). About 0.2 agent-day with tests. Not done today: it decides how top-ups are recorded,
   which is a format decision.
3. **Existing journals keep their pre-fix rows.** A walk that crosses the fix boundary reads the old breaks up to the
   first boot on the fixed binary; from that boot on the chain is walkable (no hold is outstanding at boot, so the
   first new stamp equals the rebuilt balance). Audits should cut at that boot.
4. **The darklanes lane is pushed, not merged**: `lane/budget-journal-order-20260922` at `20f8140e8`, one file,
   `+177 -4` before formatting, tests 92/92, clippy clean. `cargo fmt -- --check` on that crate reports pre-existing
   drift in `admin.rs` and `capture.rs` on darklanes `main` that this lane did not touch. The owner decides the merge
   (a money path), and the engine pin does not move for it (the change is on the deployment side of the seam).
5. **memra `metering.rs` named a module that does not exist here.** Its header said the stock binary wires
   `ledger::ReferenceMetering`; the extraction lane moved that implementation out of this crate. Fixed today in the
   comment only (the seam's own doc; no code change), so the next reader is not sent to a module this tree lacks.

## 6. memra#464, reading only: what the backfill needs, and whether today's fix gives it

`recover_request_ledger` (darklanes `ledger.rs` 1239) walks `requests.jsonl`, and for every row whose `budget` is
non-null re-derives a debit (`recover_debit`, a `kind: "debit"` journal row for `budget.debit_micro`) unless the row's
`request_id` is already in `state.debits`, the replay guard. The guard is seeded ONLY from this box's journal:
`replay_budget_row` inserts every `debit` row's id (request settles and `admin-debit:` ids) during `open`. Nothing else
feeds it.

**What it needs to tell a carried row from an unbilled one:** a fact about the id's billing status that is not "this
box's journal is silent about it". Two rows are byte-identical to the backfill today: (a) a request row written by a
crash between `PendingReceipt::finalize`'s two appends (the request ledger row first, under `Ledger.file`, then the
journal row under `BudgetState`; a crash between them leaves `budget` set and no journal id), which is the case the
backfill was written for; (b) a request row carried onto a box that was seeded NET through one `credit` row, whose
source-box journal debit rows were deliberately not carried (carrying them would replay as money against a seed that
already accounts for them). Both have `budget.debit_micro` set and no journal membership. The request row itself
cannot carry the fact: it is written BEFORE the debit exists, so `finalize` cannot mark it billed, and the
`max_tokens`/`reserved_ctx` columns describe admission, not settlement.

**Does the ordering fix give it?** No. Today's change alters what the stamp means; it adds no id to the guard and no
marker to any row. What it does give a box move is the seed arithmetic: `balance_after_micro` at a cut is now the
settled balance, so "seed at the cut, replay `amount_micro` after it" conserves whatever holds were in flight at the
cut (the class of error #423 corrected with an explicit credit), and the cutover gate's `engine_balance - unguarded`
comparison is exact at quiescence rather than off by the outstanding holds. The double derivation of carried rows is
untouched by it.

**What would close #464 (engine side, in darklanes; the owner decides the format):**

- The sidecar #464 names: `requests.jsonl.accounted`, one request id per line, read at `open` into the guard
  (amount 0, no money) BEFORE `recover_request_ledger`, so a carried id is skipped and a crash-unbilled id is still
  re-derived. Keeps the journal pure; adds one reader and one file, whose name must avoid the `<journal>.<five digits>`
  segment shape.
- Or a zero-amount `kind: "debit"` row per carried id, appended to the destination journal by the move tool with
  `exact_cost_usd` naming the carry. `replay_budget_row` already seeds the guard from it (`checked_sub(0)` moves
  nothing; an older binary replays it unchanged, so no new kind and no rollback risk), `recover_request_ledger` then
  skips the id, and a later conflicting real row for that id fails the boot loudly instead of billing twice. No new
  file, no new reader; N extra rows per move.

Either is about 0.5 agent-day with tests (boot with carried rows plus the seed: `recovered 0`; boot with one
crash-unbilled row: `recovered 1`; a seed for an id that later settles for real: refused). Not started today, per the
brief; the gap is commented on #464.

## 7. Hygiene, receipts, hours

memra: no engine code change today (one doc comment in `crates/memra-server/src/metering.rs`, section 5 item 5), so
no touched-crate clippy is owed; `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh`,
`tools/check-conflict-markers.sh` and `python3 tools/check-public-boundary.py check` run before the final push (their
lines are in this section's commit message). Receipts: `rtx5090-day30/{before,after-run1-prefmt-tree,after}/` and the
two darklanes test logs beside them. Scratch `/tmp/spill-b-day30/` is deleted with this day; the darklanes worktree is
removed after its branch is pushed (the branch stays for the owner's decision). Hours: about 3.2 agent-hours against
the 4-hour budget (reading 1.3, cells and fix 1.2, record 0.7).
