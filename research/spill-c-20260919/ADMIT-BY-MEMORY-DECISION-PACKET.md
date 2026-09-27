# Decision packet: `MEMRA_ADMIT_BY_MEMORY`, decide-by 2026-09-23 (draft for the owner)

Status: a DRAFT written by lane C on day 32 (2026-09-22) as the owner's input for the decide-by, in the shape of
`DOOR-DECISION-PACKET.md`. It recommends nothing. It becomes a `docs/decisions/` record only after the owner
decides; until then it lives here. Every number below was read from the receipt file named beside it (not from a
summary), every verdict and log line is quoted verbatim, every cell is `executed-not-qualified` development
evidence, and no number is compared across cards. The door is lane B's subject (`spill-b-20260919/DAY26.md`,
`KV-RESIDENCY-DESIGN.md` option (a)); this lane assembles, it did not measure. Paths are relative to `research/`;
`B` = `spill-b-20260919`, `C` = `spill-c-20260919`, `lead` = `spill-lead-20260919`; `darklanes` paths are in the
private repo and are pointers only (no host, id, price or location from them is restated here). Tree for the code
reading: this lane's tip after the integ41 union (`crates/` as on `main` `f10973ab7`, #644).

## The one page

### 1. The question

Whether `MEMRA_ADMIT_BY_MEMORY` becomes the naked default of the batched worker's admission seam, stays a door with
a new date and a named missing gate, or is deleted with its verdict moved to the removed-doors ledger. The door
landed 2026-09-10 (`ad02b8c9a`, #431, released in v0.138.0) with decide-by 2026-09-23 in its `docs/FLAGS.md` row (13
days after landing; the hygiene rule's default is 14). Lead record, integ30 (`lead/INTEGRATION-DAY12.md`, "B day 26"):
"Owner decisions flagged: the `MEMRA_ADMIT_BY_MEMORY` door's decide-by is 2026-09-23 with B's receipt as its
evidence; the park policy." B's design note (`B/KV-RESIDENCY-DESIGN.md` (a)): "Cost. 0.5 agent-day (the cell and the
FLAGS decision; the code exists). The decide-by is tomorrow." Nothing in either lane's records answers the question.

### 2. What the door is today (code reading, `crates/memra-server/src/{admit_memory.rs,worker.rs}`)

`MemoryAdmitConfig::from_env` (`admit_memory.rs:84`) reads `MEMRA_ADMIT_BY_MEMORY` once at worker start (strict
`"1"` arms; anything else is OFF), with `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` (default 8192) and
`MEMRA_ADMIT_DEFER_BUDGET_MS` (default 8000) read only under it, and prints one boot line
(`[admit-mem] door=ON|OFF open_output_tokens=N defer_budget_ms=N (...)`, `worker.rs:21241-21247`). Three parts:

- **(a) Open-output charge** (`request_ctx_cap`, `worker.rs:25992-26034`). OFF: a request bounding neither
  `max_tokens` nor `max_ctx` takes the `(None, MAX_NEW_CTX_BOUNDED)` arm and is charged and ALLOCATED the server
  context (`MEMRA_CTX`, else the checkpoint's own context; `prompt + server_ctx` when `prompt + 16 > server_ctx`;
  clamped to the model's trained context). ON: the same request is charged
  `charged_ctx_tokens(prompt, None, open_output_tokens, model_ctx)` = `prompt + 8192 + 8`, clamped the same way,
  never under `prompt + 8`. Because `ctx_cap` is both the charge and the `Cache::new_inner` allocation
  (`B/DAY26.md` 1.2: "Every session cache is born at `ctx_cap`"), (a) bounds the allocation, not only the book.
  Every other arm (a given `max_tokens`, a given `max_ctx`) is byte-identical under the door. A registry that pins
  `default_output_length` or `max_output_length` bounds `max_new` at the HTTP layer first and never reaches this arm
  (`B/DAY26.md` 1.1, `lib.rs:8646-8656` on B's tree); the naked `MEMRA_MODELS=` boot does. A client on a naked boot
  that omits `max_tokens` would stop at about 8,200 generated tokens with `finish_reason: "length"` where today it
  runs to the context (`B/KV-RESIDENCY-DESIGN.md` (a), "What would change for a client"); `docs/SERVING.md:912-914`
  states the current contract (omitted means context-bounded).
- **(b) Host-tier demotion at the admission reclaim** (`evict_all_demoting`, `worker.rs:11978-12027`; the call site
  `worker.rs:22417-22436`). OFF: the admission reclaim ladder's device prefix flush is `PrefixCache::evict_all`, every
  entry dropped. ON: the flush first settles any pending `Demoting`, `Promoting` or `Restoring` entry of the host
  tier with `ContractWait::Block` ("the admission reclaim"), then walks `oldest_evictable()` and, while
  `host.armed()` and the bytes demoted so far are under `demote_budget_bytes` (this arrival's SHORTFALL,
  `admit_memory.rs` `demote_budget_bytes`), calls
  `host_demote_prefix_ref(engine, host, entry, ContractD2h::OnTick)` before `remove_at`. Two consequences the
  decision should know: (1) with the host tier unarmed (`MEMRA_KV_HOST_MB` unset) `host.armed()` is false and (b)
  is the OFF program exactly (`HostDemoteOutcome::Off`, "byte-identical to today"); (2) the route is
  `ContractD2h::OnTick`, so under `MEMRA_KV_HOST_CONTRACTS=1` this demote is the day-16 SYNCHRONOUS tick program,
  not Move 1's copy-stream route (`C/DOOR-DECISION-PACKET.md` section 2, "The by-reference demote routes (the
  admission reclaim flush, the pause sweep, the handoff) keep the day-16 synchronous program (Move 1 owed item 3)").
  Entries the tier refuses (off, latched off, tenant share cap, copy failure) and everything past the budget drop
  as today; the flush always frees what it was called to free. Receipt when it fires:
  `[admit-mem] reclaim demoted N of M device prefix entries to the host tier (X MB kept warm; host Y MB of Z MB
  resident)` (`worker.rs:22440`).
- **(c) Bounded defer** (`worker.rs:22864-22930`). OFF: a memory-deferred request requeues FIFO with no bound. ON:
  the first memory defer latches `req.memory_defer_since`; each defer computes `estimate(admission_cap, bpt, ...)`,
  reads `device_free`, `demotable_device_bytes = px.evictable_bytes()` and `host_free` (0 when the tier is unarmed),
  and `decide` returns admit, demote-then-admit, defer, or refuse; refuse only when both tiers are exhausted AND
  `waited_ms` exceeds the budget, as a 429 with `Retry-After` clamped to 1..=60 s (5 s when unknown).
  `MEMRA_ADMIT_DEFER_BUDGET_MS=0` disables (c) and keeps (a) and (b). Every decision prints one grep-stable
  `[admit-mem] id=... verdict=... est_bytes=... device_free=... host_free=... demotable=... short_by=... waited_ms=...
  retry_after_s=...` line (`admit_memory.rs:322-345`).

OFF (unset or `0`) restores all three today's-behaviour arms (`docs/FLAGS.md` row, "RED ARM"). The row's rollback
clause couples the door to a launcher's session ceiling ("which must also restore `MEMRA_MAX_SESSIONS=4`; the two
move together"): the ceiling is a deployment knob outside this repo, and no cell in this research tree moved it.

### 3. Evidence B banked (every cell door OFF; per card, never across cards)

**The 129x receipt.** `arm=i L0 ... median=129.26` is the target card's open-arm allocated-over-used ratio at the
shortest prompt length, read from four receipt files that agree to the digit (greedy, identical prompts):
`B/pro-single-day26/cells/ab/REPORT.txt:103`, `cells/ba/REPORT.txt:103`, `cells/warm/REPORT.txt:106`,
`B/pro-single-day27/cells/after-ab/REPORT.txt:103`. Verbatim (the `ab` line):

```text
arm=i L0 N=5 P=[1481, 1481, 1484, 1484, 1483] G=[591, 369, 589, 457, 545] cached=[0, 0, 0, 0, 0] ratios=[126.52, 141.7, 126.46, 135.06, 129.26] min=126.46 median=129.26 max=141.70 alloc_B=8271167488 used_B_median=63987456 booked_MB=[9291, 9777, 9779, 9779, 9778] inherited=1
```

`ratio = (bpt x ctx_cap) / (bpt x (P + G))` with `bpt = 31,552` (spec path, 27B) and `ctx_cap = 262,144` (the
checkpoint's context, `MEMRA_CTX` unset): the allocation is the cap rule's, whatever the prompt. Every boot in
every cell below printed `[admit-mem] door=OFF open_output_tokens=8192 defer_budget_ms=8000 (...)` (one line per
`server.log`, counted); the runner (`B/run-day26-cell.sh`) never sets the door. So every number in this section is
a measurement of the OFF program; the ON arm has never booted on either card in this research tree (section 5).

Target card, one RTX PRO 6000 Blackwell at 600 W, Qwen3.8-27B NVFP4-Q5K MTP, `MEMRA_CTX` unset (262,144), spec path,
tree `1c66ff10e` (`cells/*/source.txt`), one binary (`cells/*/binary.sha256` equal across the three cells), collector
`tools/tier-battery.py --rig pro-single`, `/tmp/memra-gpu.lock`, `command.capture.json` `status executed-not-qualified`,
`qualification false`, `exit_code 0` in all three; 46 requests per cell (1 warmup + 45), `requests_ok=45 non200=0`,
`compute-apps-{before,after}.csv` header-only:

| Cell | Order | Regime (recomputed today from `command.gpu.csv`, 250 ms) | Rig line before / after | Receipt |
|---|---|---|---|---|
| `cell-ab` | AB (open then bounded inside each length; continuations deferred) | 778 samples, 32 to 64 C, 32.40 to 512.22 W | `32, 32.65 W` / `48, 90.71 W` | `B/pro-single-day26/cells/ab/{REPORT.txt,server.log,client.jsonl,samples.csv}` |
| `cell-ba` | BA | 777 samples, 44 to 64 C, 52.07 to 511.50 W | `44, 53.15 W` / `49, 89.28 W` | `B/pro-single-day26/cells/ba/` |
| `cell-warm` | AB, each continuation right after its first turn (the hit shape) | 731 samples, 37 to 65 C, 33.43 to 512.74 W | `37, 34.06 W` / `47, 90.92 W` | `B/pro-single-day26/cells/warm/` |

| Arm (N=5 per length per cell) | L0 (P about 1.48k) | L1 (P about 3.14k) | L2 (P about 5.80k) | Same in `ab`, `ba`, `warm` |
|---|---|---|---|---|
| (i) `max_tokens` omitted: ratio median (min..max); `alloc_B`; `used_B_median` | `129.26` (`126.46..141.70`); `8271167488`; `63987456` | `71.90` (`64.06..72.50`); `8271167488`; `115038592` | `41.61` (`40.79..42.58`); `8271167488`; `198777600` | yes (P, G identical to the token) |
| (ii) `max_tokens=96`: ratio; `alloc_B` | `1.00`; `57393088` | `1.00`; `97937408` | `1.00`; `183317120` | yes |
| (iii) continuation, `max_tokens` omitted: ratio median (min..max); `used_B_median` | `134.02` (`122.44..145.31`); `61715712` | `72.12` (`68.25..76.34`); `114691520` | `41.15` (`40.58..42.13`); `201017792` | yes; `warm` reads `cached=[1440 x5]`, `[3104 x5]`, `[5760 x5]` where `ab`/`ba` read `cached=[0 x5]` (the deferred shape's LRU eviction inside the day-26 budget; fixed on day 27, below), the ratio unchanged because the restore's carrier is a fresh `ctx_cap`-row cache |

Concurrency arithmetic (the cell's own numbers, not an admission run; `REPORT.txt:116-118` of each order): `L0: P~1483
open booked_MB=9778 -> 8 sessions; bounded booked_MB=1674 -> 49 sessions (free_ready=82759516160)` (AB; BA's bounded
arm ran before the learned `fixed` term rose and reads `booked_MB=1191 -> 69 sessions`), `L1 ... open 10571 -> 7;
bounded 2331 -> 35`, `L2 ... open 11066 -> 7; bounded 2975 -> 27`. `MEMRA_MAX_SESSIONS` was 64 in these boots.

Day-27 after cell on the same shape (`B/pro-single-day27/cells/after-ab/`, tree `de2c781e6`, order AB, deferred;
`status executed-not-qualified`, `exit_code 0`, 46 requests, `non-200=0`): after B's prefix-budget context fix the
(i) rows equal day 26 to the token and the ratio (`arm=i L0 ... median=129.26`, `REPORT.txt:103`), the 15
continuations hit (`cached=[1440 x5]`, `[3104 x5]`, `[5760 x5]`; `prefix-cache hit lines: 15`;
`prefix_cache_entries=45 prefix_cache_bytes=11986255872 ... prefix_cache_evictions=0`), retention at idle
`retained_by_process=39090913280`. Relevance to the door: part (b) demotes what the reclaim ladder would drop; on
the day-26 tree the ladder found 0.68 GB of entries and on the day-27 tree it finds 12.0 GB resident, so the
premise of (b) (entries worth keeping warm at the flush) exists on the target card only since `de2c781e6`.

Local RTX 5090 Laptop GPU (24,463 MiB, `power.limit [N/A]`), Qwen3.5-9B NVFP4 MTP, `MEMRA_CTX=65536` (the naked
262,144 does not fit the mix on this card, `B/DAY26.md` 2.1), spec path, tree `1c66ff10e` (`ab/source.txt`; `ba` and
`warm` stamp later record-only commits `26ce1ece2`, `6734be94d`), ONE binary (`binary.sha256` equal across the three
cells), `flock /tmp/memra-5090.lock`; 46 requests per cell, `requests_ok=45 non200=0`, `compute-apps` header-only:

| Cell | Rig line before / after (`gpu-{before,after}.csv`) | Sampler rows (`samples.csv`, memory gauges only) | Receipt |
|---|---|---|---|
| `ab` | `54, 9.32 W` / `76, 30.62 W` | 2,789 | `B/rtx5090-day26/ab/` |
| `ba` | `62, 17.85 W` / `75, 29.25 W` | 2,813 | `B/rtx5090-day26/ba/` |
| `warm` | `68, 34.04 W` / `76, 29.72 W` | 2,811 | `B/rtx5090-day26/warm/` |

| Arm (N=5 per length per cell) | L0 (P about 1.44k) | L1 (P about 3.10k) | L2 (P about 5.76k) |
|---|---|---|---|
| (i) ratio median (min..max); `alloc_B`; `used_B_median` | `11.30` (`10.10..12.23`); `1094713344`; `96916608` | `10.73` (`9.50..11.89`); `1094713344`; `101977920` | `6.62` (`6.20..7.27`); `1094713344`; `165319488` |
| (ii) ratio; `alloc_B` | `1.00`; `29683008` | `1.00`; `51147648` | `1.00`; `96348672` |
| (iii) ratio median (min..max) | `11.79` (`10.54..17.91`) | `10.62` (`8.78..11.62`) | `6.35` (`2.77..7.23` in `ab`; `2.72..` in `ba`; `2.95..` in `warm`: the `iii-L2-r3` run cut by the 90 s request deadline at 17,815 / 18,314 / 16,378 tokens) |

Concurrency arithmetic (`REPORT.txt:115-117`): `L0: P~1441 open booked_MB=1840 -> 8 sessions; bounded booked_MB=858 ->
18 sessions (free_ready=16029908992)` (BA: `739 -> 21`), `L1 ... 2436 -> 6; 1343 -> 11`, `L2 ... 2824 -> 5; 1824 -> 8`.
This model reasons before it answers (G 2,413 to 5,048 per one-sentence summary), which is why the open arm's ratio
is one order lower than the target card's: the numerator is the same cap rule, the denominator is this model's own
generation. On this card the bounded arm's booked cost is the prefill workspace and the fixed term, not KV
(`B/DAY26.md` 2.3).

**What the door's part (a) would allocate, as arithmetic on the receipts' own P and bpt (not a measurement, no ON
boot exists):** target card `(P + 8200) x 31,552` = `305,518,016 B` at P 1483, `357,736,576 B` at 3138,
`441,822,656 B` at 5803, against the receipts' `alloc_B=8271167488`; local `(P + 8200) x 16,704` = `161,043,264 B`,
`188,688,384 B`, `233,204,544 B` against `1094713344`. The bounded arm (ii) and the hit's private copy are untouched
by (a) (`B/KV-RESIDENCY-DESIGN.md` (a), "What it moves").

### 4. Correctness gates that cover the ON arm, verbatim, per card

The `docs/FLAGS.md` row names the gate: "GATE: CPU `cargo test -p memra-server admit_memory` plus the worker wiring
tests, and the card cell `cargo test -p memra-server memory_admission_admits -- --ignored` (1 x 900k + 8 x 4k
concurrently admissible on one B200 against live `mem_get_info`, with the tenth 900k arrival refusing). Receipt
pointer: darklanes `research/glm5-1m-b200-ship-20260906/` (requalification cell for the production pair) and this PR."

| Gate | Card | What it covers | Receipt (verbatim where one exists) |
|---|---|---|---|
| CPU, `admit_memory.rs` `mod tests` (11): `a_short_probe_is_not_a_long_session`, `open_output_charges_the_output_not_the_envelope`, `ring_geometry_caps_the_ring_class_only`, `decide_walks_device_then_host_then_time`, `one_long_and_eight_short_all_admit_on_one_card`, `retry_after_is_clamped_to_the_shed_window`, `waited_ms_is_zero_until_the_first_defer_stamps_it`, `memory_line_locks_fields`, `memory_line_renders_every_arm_and_its_placeholders`, `boot_line_states_the_door_and_its_knobs`, `demote_budget_is_the_shortfall_only` | none (pure arithmetic over worker-owned values) | the charge, the decision walk, the budget, the receipt line's fields | darklanes `research/glm5-memory-admission-20260909/LANE.md` "Bring-up receipts (2026-09-09, lane box)": "`cargo test -p memra-server` - **665 passed, 0 failed, 5 ignored**" (the door's landing tree); on today's tree: `C/day32/admit-cpu-tests.log` (this lane, run under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`; result recorded in `C/DAY32.md` 4) |
| CPU, worker wiring (2): `the_open_output_arm_charges_the_envelope_off_the_door_and_the_output_on_it` (the real `request_ctx_cap` at `MEMRA_CTX=1048576`: "On the 1M route these two answers differ by 118x"), `the_memory_admission_door_is_wired_at_the_admission_seam` (`include_str!("worker.rs")`: read once at worker start, threaded into `prepare_request`, the flush takes the demoting path, the defer site can refuse) | none | the seam is wired and stated | as above |
| Card cell, `memory_admission_admits_one_900k_and_eight_4k_on_one_card`, `#[ignore = "requires an exclusively locked CUDA device with >= 32 GiB free (the mixed-load admission cell)"]`; geometry `BPT: u64 = 13_312`, `FIXED: u64 = 155_000_000`, `MODEL_CTX: usize = 1_048_576` (the B200 mint's); red arm: a tenth 900k-class request must not read `admit` | one B200 SXM (rented for the door's lane, destroyed at its end; darklanes `research/glm5-memory-admission-20260909/LANE.md`) | (a) and (c)'s arithmetic against the driver's own `mem_get_info`, N=1 | `[cell] 1x900k + 8x4k resident: free 177.70 -> 164.39 GiB (13.31 GiB charged)` (quoted in the `ad02b8c9a` commit message and in darklanes `LANE.md`: "Every one of the nine admits on device alone against the driver's own `mem_get_info`, no demotion needed; the red arm (a tenth 900k-class arrival one byte past the card) defers inside its budget and refuses past it"). No receipt FILE for this cell exists in either repo's tracked tree; the quoted line is the only record. |
| Card cell | RTX PRO 6000 Blackwell | (a), (b), (c) under the door ON | **none.** No boot with `[admit-mem] door=ON` exists in `research/spill-*` (`rg -u` over every receipt: 0 hits); every B cell reads `door=OFF`. |
| Card cell | RTX 5090 Laptop GPU | as above | **none** (same search). |
| Serving boot with the door ON (not a gate of the door: the cell's subject was the stress-succession criterion) | a 2x B200 TP-2 boot | the boot line only | darklanes `research/glm5b200-stress-criterion-20260911/receipts/slot-A.log:47`: `[admit-mem] door=ON open_output_tokens=32768 defer_budget_ms=8000 (admission charges prompt+output, the reclaim flush demotes to the host tier before dropping, and a memory defer past the budget refuses 429 instead of queueing past the client's deadline)`; `env-posture.txt:5` `MEMRA_ADMIT_BY_MEMORY=1`. In the whole receipts directory: 1 `[admit-mem]` line (the boot line), 0 `[admit-mem] id=` decision lines, 0 `reclaim demoted` lines, 0 status-429 rows in `gate-A.log`, `gate-B.log`, `ledger-requests.jsonl`. The door was armed and never reached a decision in that window. |
| The requalification cell the FLAGS row points at ("1 x 900k + 8 x 4k + 8 x 32k concurrent" both arms, six pass conditions, boot nonce arm identity) | the target pair named in darklanes | (a), (b), (c) under load, control against door | darklanes `research/glm5-memory-admission-20260909/LANE.md` "REQUALIFICATION CELL for the production pair (**not yet run**)". The FLAGS row's pointer `research/glm5-1m-b200-ship-20260906/` holds no file naming the door or `[admit-mem]` (grep 0); the cell's definition lives in the door's own lane file. |

Byte identity under (a) is by construction (only the cap moves; the same prompt and tokens up to the stop), and B's
design note still owes it as a recorded digest comparison "on every request whose generation ends before the bound"
(`B/KV-RESIDENCY-DESIGN.md` (a), "Gate owed"); no such digest pair exists.

### 5. Unmeasured, unbuilt, or open (each with where it is stated)

1. **The ON arm has no receipt on either card of this program.** B's decision cell as pre-registered in
   `KV-RESIDENCY-DESIGN.md` (a) ("the day-26 mix with the door OFF and ON on the same binary, N>=5 per arm per order,
   both orders, both cards; per request the allocated and used bytes, `finish_reason`, and the `[admit-mem]` receipt
   line; the concurrency the card admits at the served context under OFF and ON") was not run (B `STATE.md`, day 30:
   "Open from earlier days: the admission-by-memory door decide-by 2026-09-23 (KV-RESIDENCY-DESIGN.md (a), 0.5
   agent-day)").
2. **No `[admit-mem] id=... verdict=...` decision line exists in any tracked receipt** in memra or darklanes (the
   string appears only in `admit_memory.rs` and `docs/FLAGS.md`). Part (c)'s 429 and part (b)'s `reclaim demoted`
   line have never been observed outside CPU tests.
3. **Part (b) has no serving receipt and its premise moved under it.** On the day-26 target-card tree the flush had
   0.68 GB of entries to demote; since `de2c781e6` (day 27) it has 12.0 GB resident. (b) also requires an armed host
   tier (`MEMRA_KV_HOST_MB > 0`); B's day-26 and day-27 mixes ran with the tier unarmed (no `[prefix-host]` line in
   those boots was sought or claimed). The stall (b) would put on the tick is bounded by the shortfall and priced by
   nothing: the nearest number is this lane's synchronous demote figure for the same route class
   (`C/DOOR-DECISION-PACKET.md` section 4, the pair rows), which is the by-entry route's cost, not this flush's.
4. **The contracts-door interaction is by reading, not by cell.** `evict_all_demoting` passes `ContractD2h::OnTick`,
   so under both doors ON the admission flush is the synchronous tick program (section 2). No cell has booted the
   two doors together.
5. **The receipt pointer in the FLAGS row does not resolve to a door receipt** (section 4, last row). The requal cell
   is "not yet run" in the door's own lane file, and that file's precondition for running it (a fleet pin moving to
   a release that carries the door) is a deployment fact recorded in darklanes, not here.
6. **The client-visible contract change is undocumented.** Under (a) a naked boot's open request stops at about
   8,200 tokens with `finish_reason: "length"`; `docs/SERVING.md:912-914` still states the omitted-means-context
   contract. A promotion would need that sentence changed in the same PR (a `docs/` edit, the lead's).
7. **`MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` is a floor, not a fleet fact.** The FLAGS row: "8192 is a floor, not a claim
   about the fleet ... a deployment whose advertised `max_output` differs sets this to that number: the
   qualification-env law means it is DERIVED from the registry the launcher ships, never hand-typed". The one ON
   boot with a receipt derived 32768 (section 4). A naked default would make 8192 the engine's own answer on every
   registry-less boot.
8. **Which memory the door would buy, per card, is arithmetic only** (section 3, last paragraph). On the target card
   the day-26 concurrency lines say the open arm admits 7 to 8 sessions at the served context and the bounded arm 27
   to 69; where (a) lands between them was not measured. On the local card the bounded arm buys 8 -> 18 to 21 at L0
   and 5 -> 8 at L2 because the prefill workspace and the fixed term, not KV, dominate the booked cost there.
9. **Sibling doors with their own dates**: `MEMRA_KV_HOST_CONTRACTS` (decide-by 2026-10-05, `C/DOOR-DECISION-PACKET.md`),
   `MEMRA_KV_PARK_COMPACT` (decide-by 2026-10-06 in its row's prose, value column still `0 = OFF by design`; the
   reconciliation proposal is `C/DAY32.md` 2), `kv-tier-gate --kv-allocator vmm` (2026-10-04,
   `docs/decisions/KV-PHYSICAL-RECLAIM.md`), the MoE slot cache door (2026-10-04, `C/MOE-SLOT-CACHE-DOOR.md`).

### 6. The three outcomes the door hygiene rule allows, and what each would require (not recommended)

- **A naked default (per card class; the per-hardware rule).** Delete the `MEMRA_ADMIT_BY_MEMORY` read and the OFF
  arms of (a), (b), (c): `request_ctx_cap`'s `None` branch of the open arm, the `px.evict_all()` else-branch at the
  reclaim, the unbounded-defer path; keep `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` and `MEMRA_ADMIT_DEFER_BUDGET_MS` as
  runtime parameters (their rows would lose "read only when `MEMRA_ADMIT_BY_MEMORY=1`") or fold them; the eleven CPU
  tests and the two wiring tests keep a single arm; the `#[ignore]` card cell stays as the card's red arm; change
  `docs/SERVING.md:912-914`; the FLAGS row becomes the naked-default statement. It requires what the rule requires
  of every default: correctness of the ON arm proven on the card class it becomes the default on, and the balanced
  same-window A/B (both orders, N>=5) that today exists on neither card (item 1); the hygiene rule's winner clause
  then deletes the rollback seam after two served weeks with the seam unused. The deployment-side coupling (the
  session ceiling and the derived output length) stays a launcher matter outside this repo.
- **A longer door with a new date and the missing gate named.** Requires the `docs/FLAGS.md` row to carry the new
  `decide-by:` and the reason in the row ("pending its X row" is a date, not a state). The candidates the records
  name: B's pre-registered OFF-against-ON cell on both cards (item 1; B priced it at 0.5 agent-day), a (b) cell with
  the host tier armed on the day-27 tree (item 3), the requal cell in darklanes (item 5), and the two-doors-ON
  boot (item 4). The row's receipt pointer would be corrected to the door's lane file at the same time.
- **Deletion, with the verdict and the receipt pointer moved to the removed-doors ledger.** Requires removing in one
  PR the three env reads (`admit_memory.rs:84-93`), `MemoryAdmitConfig` and the boot line, the `open_output_tokens`
  parameter of `request_ctx_cap` and its caller in `prepare_request`, `evict_all_demoting` and its call site (the
  flush returns to `px.evict_all()`), the bounded-defer block and `req.memory_defer_since`, the `[admit-mem]` line
  types, the thirteen CPU tests and the `#[ignore]` card cell, the three FLAGS rows and the FLAGS header section
  "Memory-shaped admission for a high session ceiling, 2026-09-09", with the verdict (the words the records hold:
  "absent receipt by 2026-09-23", darklanes `LANE.md` "Verdict handling") and pointers to `B/DAY26.md`,
  `B/KV-RESIDENCY-DESIGN.md` (a) and this packet moved to `docs/FLAGS.md`'s "Removed doors" ledger and the
  darklanes verdicts ledger. What is lost with it is named in B's design note: the only policy-level answer to the
  open arm's ratio (126 to 142 at L0 on the target card) "with no numeric change"; the bounded arm's ratio is 1.00
  with or without it, and the layout options (b), (c), (d) of that note buy the prefix copy and preemption, not
  this ratio. The launcher's marker gate on the binary (darklanes) would then arm nothing and keep its own default.

## Appendix

### A. How every number above was traced

The ratio, `alloc_B`, `used_B_median`, `booked_MB` and concurrency rows were read from the named `REPORT.txt` lines
(`grep -n "^arm=\|L0:"`), the boot lines from each cell's `server.log` (`grep -c "admit-mem\] door="`), the
`status`/`qualification`/`exit_code`/`elapsed_seconds` from `command.capture.json`, the regimes recomputed from the
target cells' `command.gpu.csv` (columns 6 and 9, 250 ms rows) and the rig lines from `gpu-{before,after}.csv`, the
tree from `source.txt`, binary identity from `binary.sha256`, the request counts from `REPORT.txt`
(`requests_ok=45 non200=0`), and the code lines from today's `crates/memra-server/src/{admit_memory.rs,worker.rs}`
with `grep -n`. The darklanes lines were read from the named files in the private repo; only the door's own log line
and the lane file's quoted sentences are restated, no host, id, price or location. The arithmetic paragraph of
section 3 is labelled arithmetic and uses only the receipts' `P` and the census's bpt.

### B. Numbers deliberately NOT carried into this packet

- B's `DAY26.md` prose ranges and sample counts: replaced by the counts and ranges recomputed today from the files.
- Retention and pool figures (`retained_by_process`, `pool_used max`, the 30 GB at idle): they belong to the park
  door and the VMM door's decisions, not this one; they are in `B/DAY26.md` 2.2 to 2.4 and `B/DAY27.md` 2.
- The "85.9x" and "118x" of the FLAGS row and the wiring test: the 1M route's arithmetic, restated where the test
  is quoted and not treated as a measurement.
- Cross-card ratios of any kind, and any comparison of the 27B's ratio with the 9B's.

### C. Untraced

None at the time of writing: every number in sections 3 and 4 was read from the file named beside it, or is labelled
arithmetic. If a later reader finds one that is not, it is to be struck rather than believed.

### D. Missing (numbers the decision would need that have no receipt anywhere; none was run today)

1. The ON arm's allocated-over-used, `finish_reason` and `[admit-mem]` lines on the target card and on the RTX 5090
   (B's pre-registered cell, both orders, N>=5 per arm; the local half is one collector hold on the 9B under
   `/tmp/memra-5090.lock`, not a one-boot cell, so it was not run under today's rule).
2. Part (b)'s `reclaim demoted` count, bytes and tick cost with the host tier armed, on either card.
3. Part (c)'s 429 with `Retry-After` observed on a card under a real memory defer (the CPU tests cover the arithmetic).
4. The concurrency the card admits at the served context under ON, against the OFF lines of section 3.
5. The completion-digest identity OFF against ON on requests ending before the bound (B's owed gate).
6. The requal cell of the FLAGS row, both arms, on the pair it names (darklanes; "not yet run").
