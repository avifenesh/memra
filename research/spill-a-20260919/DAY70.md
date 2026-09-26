# WP-A day 70: L' charges a lease its class, so a short-prefix demote refuses where main admits (revuto round 2 on integ69); design Q registered

Integ69's second review round (revuto, comment 4112859255, `tier_transfer.rs:1592`) found a second L' defect. It is
placed below with a CPU cell on the merged code, and design Q is registered before any code. Every cell is
`executed-not-qualified`.

## 1. The defect, placed

- L1.2 (DAY63) charges a lease `lease_class(bytes)`, not its length:
  - at or below 1 MiB the next power of two, so a 600 KiB plane is charged 1 MiB (1.7x);
  - above 1 MiB, up to 1 MiB more per lease.
- The host LRU keeps residents at or under one budget of ACTUAL bytes (`total_bytes`, `e.bytes`). Under the door, each
  resident's pinned charge is its leases' charges: the residency charge takes pinned zero, and the leases carry it.
  So for short prefixes the residents' charges approach two budgets.
- The pinned ledger is `thrice(host_budget)`, planned as three parts: one budget of residents, one for the in-flight
  demote, and one for the pool's idle backings (cap one budget). The following exceeds it:
  - residents' class charges;
  - a pool of idle backings in other classes, which `take` cannot reuse because it matches the exact class;
  - the staging set's charges;
  - the next demote's classes.
- The demote's lease reserve then returns `Capacity` where main's two-budget ledger (length charges, no pool) admits.
- **The consequence is worse than a missed demote.** The contract route maps a lease refusal to `Alloc` (`pinned host
  alloc of N B failed: tier transfer engine Capacity`), and the caller latches the tier off, as the pre-door alloc path
  does. Main keeps serving through the same sequence.
- **Nothing recovers it.** Nothing gives the idle pool's charges back on the refusal. And while there is room, `put`
  keeps reserving idle backings, which then block the fresh allocation that would replace them.
- **The placement cell** (`day70/placement-test.patch`, run on integ69's `4f297e7bd`, `day70/placement.log`), verbatim:

      DAY70 PLACE L': residents 109 x 614400 B = 66969600 B of length charged 114294784 B; pool idle 67108864 B; staging 8388608 B; ledger 201326592 of 201326592 B; the demote's lease refused at Some((11, Capacity))
      DAY70 PLACE main: every demote lease admitted: true; ledger 85188608 of 134217728 B

  - The setup: a 64 MiB budget, residents of 600 KiB planes at one budget of length (109 planes), the pool full of
    4 MiB-class idle backings, and an 8 MiB staging charge.
  - The result: L' refuses a 16-plane short demote at its 12th lease. Main admits all 16 at 81 MiB of its 128 MiB.
  - The cell models L''s charge by its own statement (`request.bytes.pinned = class as u64`, which DAY63's census
    pins), and the pool through its real `put`.
- **The receipts (the lead's question 4): no sitting's demote count differed between arms, so no reading is
  affected.**
  - The `[prefix-host] demote:` counts per arm and order are equal in every A/B cell: L (DAY63 section 3) and L'
    (section 6), base against l; T-H (DAY65) and P2L2 (DAY67), both arms on L'. Cell by cell, per arm and order:
    - chain: 105 = 105 in L, L', T-H and P2L2;
    - demote: 45 = 45 in the same four;
    - promote: 55 = 55 in T-H and P2L2;
    - free: 45 = 45 in P2L2.
  - No A/B server log in those sittings, in F's (DAY64) or in DAY64B's carries `TIER DISABLED` or a `pinned host
    alloc` failure. The `TIER DISABLED` lines in the gate logs are the fault gate's injected arms.
  - Those cells' entries are the 27B's long chain entries (planes of about 2 to 3 MB, inflation at most 1 MiB each)
    and its standard prompts, far from the short-prefix shape.

## 2. Design Q (one unit for the LRU and the requests; the pool pays for everything it adds), registered

- **(Q.1)** A lease's request charge is its length, main's charge and the LRU's unit. Residents and the in-flight
  demote hold exactly what main's ledger holds.
- **(Q.2)** Everything the pool adds beyond the leases' lengths is charged to the pool tenant and capped together at
  the pool's cap (one host budget): each idle backing's capacity, and each live lease's slack (its backing's capacity
  minus its length). A new lease of `bytes` gets its backing one of three ways:
  - an idle backing of its class and kind, whose idle charge becomes the smaller slack charge, which always fits;
  - or a fresh backing at its class, when the pool can carry that slack within its cap;
  - or else a fresh backing at its length: main's allocation, no slack.
  - A dropped lease releases its length and its slack. Its backing parks only if it is class-sized and the pool has
    room (and is open, in the current epoch, DAY69's P.2); otherwise it frees.
- **(Q.3)** So the 3-budget pinned ledger is main's two budgets (residents, the in-flight demote, the staging set) plus
  one budget for the pool.
  - A request's reserve refuses only where main's would: the requests' charges are main's, and the pool holds at most
    the third budget.
  - The ledger still counts every pinned byte, which is what L1.2 wanted, now split between the request (its length)
    and the pool (the tail).
- **(Q.4)** A pool reserve that the ledger refuses (a park or a slack) is never an error. The backing frees, or the
  lease takes a fresh backing at its length.
- **Why not revuto's two options, or an LRU in charged bytes.** Correctness decides.
  - (1) Draining and retrying on a refusal leaves the residents' class inflation in the ledger. Residents near two
    budgets charged plus a short demote near its double exceed three budgets with an empty pool, so it still refuses
    what main admits.
  - (2) Bounding the pool's cap by the remaining headroom has the same gap.
  - An LRU budgeting charged bytes would change which entries stay resident, so hits, promotes and evictions would
    differ from main.
  - Q keeps the LRU and every request charge in main's unit. Only the pool's own accounting sees classes.
- **One program per request:** a lease's bytes, copies, receipts and length are unchanged. Only its backing's size,
  and who is charged for the tail, differ.
- **What it costs:**
  - When the pool's cap is full (idle plus slack), new leases are fresh at their length and never pool.
  - Short-prefix residents at about 1.7x inflation fill the cap with slack at about 1.4 budgets of length (0.7 budgets
    of slack for 1 of length). Past that, their demotes pay L''s base lease time.
  - The long entries the L' cells measured (slack at most 1 MiB per 2 to 3 MB plane) keep L''s program: the chain
    cell's 32 pooled leases per long demote, and the twin's free.
- **Superseded, recorded:**
  - L1.2's "the lease is charged its size class" is now "the lease its length, the pool its tail".
  - DAY63 section 7's worker test arithmetic `411177fea` (the class charge) returns to the length, because the worker
    fixtures' pool cap is 0, so no slack is charged.
  - `28c7aa6c1` (the refusal cell's three-budget co-tenant) stays.

## 3. Cells

- **(a) CPU test** `day70_short_prefix_residents_and_a_full_pool_admit_every_lease_main_admits`: section 1's sequence
  through the real `LeasePool::lease` and `put` with stand-in backings.
  - Every lease of the short demote is admitted.
  - The request charges equal main's (the same sequence on main's 2-budget length ledger, in the same test).
  - The pool's charge (idle plus slack) stays at or under its cap.
  - Red arm: the request charged its class again must refuse (`Capacity`).
- **(b) CPU test** `day70_the_pool_carries_the_tail_within_its_cap`:
  - a pooled take turns the idle charge C into the slack C - len;
  - a fresh lease takes its class only while the slack fits the cap, and is at its length past it;
  - a length-sized backing never parks;
  - a drop releases the slack before the park;
  - a drain leaves live leases' slack charged.
- **(c) CPU census:**
  - `alloc_host_kind` charges `bytes`;
  - the slack reserve names the pool tenant;
  - `put` parks only class-sized backings;
  - the allocation's drop releases the slack before `put`.
- **(d) GPU (native), for the target card's battery (BOX43):**
  - the day-63 and day-69 engine cells, whose pinned totals are unchanged by Q (the length plus the slack is the
    class);
  - a new engine cell `day70_a_lease_past_the_pool_cap_is_its_length_and_never_parks`;
  - the worker GPU span cells with the length arithmetic restored.
- **CPU battery:** fmt, clippy `--workspace --all-targets -D warnings`, the memra-engine and memra-server libs, and
  `tools/check-flags.sh`.
- **Budget:** 0.3 agent-day.

## 4. Design Q as built (`a57f85897` on `lane/spill-a-integ69-20260926`, over `4f297e7bd`; on this lane the same commit cherry-picked)

- **Built as registered.**
  - (Q.1) `lease_charge(bytes) = bytes`, which `alloc_host_kind` charges before it asks the pool.
  - (Q.2) `LeasePool::lease(class, len, kind)` returns `Pooled(backing, tail)`, `FreshClass(tail)` or `FreshLength`:
    - `reserve_slack` charges a tail under the pool tenant only while `held() = idle + tails` plus the tail fits the
      cap;
    - `release_slack` gives it back in the allocation's drop, before `put`;
    - `put` parks only a class-sized backing (`lease_class(c) == c`) while `held() + c` fits the cap.
  - (Q.4) A refused pool reserve returns `Err(())`, and the lease is allocated at its length.
  - DAY63's census now reads `request.bytes.pinned = lease_charge(bytes)`. The worker cells' `gpu_lease_charge` is back
    to the six leases' lengths (1392 B), because their pool cap is 0.
- **Cells:**
  - (a) Green: `day70_short_prefix_residents_and_a_full_pool_admit_every_lease_main_admits`. All 16 demote leases are
    admitted, the requests hold exactly main's charges, and the pool stays at or under its cap. It is section 1's
    placement sequence, now through the real `LeasePool::lease`.
  - (b) `day70_the_pool_carries_the_tail_within_its_cap` and (c) `day70_the_lease_is_its_length_and_the_pool_its_tail`
    are green.
  - **Red arm** `day70/red-arm-class-charge.patch` (`lease_charge` returns the class again, with a marker) fails all
    three (`day70/red-arm-class-charge.log`):
    - (a) on `every demote lease admitted`: the placement's refusal again;
    - (b) on `3000 + a 1096 tail` (`left: (1096, 5192) right: (1096, 4096)`);
    - (c) on `lease_charge .. bytes as u64`.
  - (d) The GPU cell `day70_a_lease_past_the_pool_cap_is_its_length_and_never_parks` is built (`NATIVE_CELLS` 19).
- **CPU battery on `a57f85897`:**
  - fmt;
  - workspace clippy `--all-targets -D warnings` clean (`day70/clippy.log`);
  - engine lib `581 passed; 0 failed; 52 ignored` (`day70/engine-lib.log`);
  - server lib `943 passed; 0 failed; 27 ignored` (`day70/server-lib.log`);
  - `tools/check-flags.sh`; `git diff --check`.
  - On this lane's tip: engine `581 passed`, server `949 passed`, workspace clippy clean.
- **The GPU cells for the target card's battery (BOX43)**, each `--ignored --exact --test-threads=1`:
  - engine `tier_transfer::tests::day70_a_lease_past_the_pool_cap_is_its_length_and_never_parks` (new);
  - engine `day63_a_dropped_lease_backs_the_next_same_class_lease`, `day63_a_drop_past_the_cap_frees_and_the_engine_closes_the_pool`
    and `day69_a_drained_backing_is_never_the_next_lease`. Their pinned totals are unchanged by Q: the length plus the
    tail is the class.
  - the server's `option_b_` and `option_c_` cells, 19 with DAY69's purge cell, with the length arithmetic restored.
  - Also worth the battery: the engine gate binaries `tier_transfer_gate` and `kv_tier_gate` (fault). They assert a
    lease's pinned charge equals its length (`used().pinned == bytes.len()`, `pinned-released`). They set no pool cap,
    so under Q they charge the length again, as they were written.
- **P2's integ branch** `lane/spill-a-p2-20260926` is rebased onto `a57f85897` as `409be61f8`: server lib `949 passed`,
  clippy `-p memra-server --all-targets` clean.

## 5. The two gate binaries for the lead's BOX43 battery (commands handed over; no build here)

- Built by the lead at integ69's merge of `a57f85897`: `cargo build --release -p memra-engine --bin tier-transfer-gate
  --bin kv-tier-gate`.
- **Neither binary takes the rig lock, so they run directly inside `run-held.sh`'s hold (FD 9).** They must not be
  wrapped in `tier-battery.py`: the collector opens its own handle on `/tmp/memra-gpu.lock` and `flock`s it
  `LOCK_NB`, which refuses while FD 9 holds the lock. Day 12 ran them with the collector as the holder.
- **`tier-transfer-gate`:**
  - `conformance`: exit 0, the 13 `PASS` lines of day 12, last line `PASS native governor zero after controlled
    drain`.
  - `roundtrip`: exit 0, six `PASS native D2H-H2D roundtrip bytes=.. byte_exact=true .. governor_zero=true` lines
    (4 KiB to 256 MiB).
  - It takes no `--out`. Timeout 700 s each.
- **`kv-tier-gate`, the seven fault arms on the 27B:** `--artifact <27B NVFP4/Q5K GGUF> --case active --context 8192
  --tiers host --same-program --kv-allocator pooled --fault <arm> --out <new dir>`.
  - It refuses any `MEMRA_*` variable but `MEMRA_NVCC`, `MEMRA_CUDA_ARCH` and `MEMRA_GPU_LOCK`.
  - It needs a **fresh `--out` per arm and per attempt**: `fs::create_dir` refuses an existing one.
  - Last lines, as day 12's: `FAULT-ARM PASS <arm> committed=8192 generated=128` for cancel-demote, cancel-restore,
    host-budget-short, device-short and require-resident; `committed=8064 generated=0` for corrupt-host and
    missing-host. Timeout 1800 s each.
- **Why these gates matter for Q:** `missing-host`'s `pinned-released` expects the plane's length, and the binaries
  set no pool cap. So they read a lease's length again under Q. Under L''s class charge a plane that is not a class
  size would have read its class.
