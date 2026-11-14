# DeepSeek-V4.1 route: a call flag names its rows (memra#1292, 2026-11-13)

Verdict: a router row that names no expert fails its own request and nothing else. On one GB10 the
rowfault gate passes 7 of 7 on both residencies, the serve smoke passes with the other request's answer
byte-identical to its solo run, and the kernel gate passes 185 of 185. A draft-head stage's fault and a
call flag no row map explains still fail the whole step.

Scope of that evidence: a GB10, trunk layers 0 and 1. The rowfault gate ran both residencies (9 of 9
each, at one rank and at two ranks on the card); every serve smoke ran the resident trunk. The serve
smoke on the one-card residency, the rowfault gate over all 40 layers and on a pair of cards, and the
support record's re-pin are RTX PRO 6000 runs that have not happened yet (Open, at the end).

Since #1281 the route stays alive after a request-caused numeric fault, but the call flag only said
that some router row of the step named no expert, so the route failed every request of that step.
Now each call keeps a row map, the plan marks the faulted rows in it, and the route fails only the
requests those rows belong to; the step's other requests, the other rows of the same batched call
included, are served.

## What changed

- **The plan marks rows.** `memra_v41_moe_plan` takes a row map (`rows_bad`), the slice's first row in
  its call (`row0`) and the map's capacity. Each pair whose expert id is outside `[0, ne)` still counts
  in `err`, is skipped and sets bit 0 of the rank's call word, and now also sets bit `row0 + t` of the
  map for its row t (the one-block plan and the three-kernel plan alike; the count kernel's thread is
  the row's).
- **Each call owns a map.** `Runner::forward` gives each call `ceil(s / 32)` zeroed words on every rank,
  `Runner::forward_batch` one map for the batched call; `Group::own_faults` points every rank's plans at
  the call's map around the call's blocks and back after them (`Ctx::fault_rows`). The map comes back
  in `Out::faults`. A plan outside every call (a draft-head stage, a gate's block) marks the rank's own
  word of plans no call owns (word 2 of `Ctx::route_bad`), cleared with the call word when a step opens.
- **The step names its rows.** On a group whose call flag is `attributed`, the reads of a call's logits
  no longer fail on the call word (a transport flag still loses the group). After a step's last read
  the route asks `Runner::step_faults` for each call's faulted rows: no call word, nothing faulted (no
  map is read); the call word with the unowned word, or with no map naming a row, fails the step as
  before; otherwise the rows. The words read with the step's last logits read are reused
  (`Group::seen`, cleared by every forward, batched step and draft), so a clean step adds no read.
- **The route fails those requests alone** (`faulted_requests`): a call of its own is its request's,
  row r of a batched call is the request at that row. Each such request gets the route's error, is
  booked as a request fault and leaves the step; the rest choose, commit and stream as usual.
- **The off-card MoE no longer fails the forward on a skipped pair.** With the experts behind the
  device cache (one or N ranks, the plan read back), a placed-pair count below the rows' top-k returned
  "route: X of Y pairs placed", so a request's non-finite router row failed every request of the step
  before any flag was read. The placed pairs are complete and the ranks' ranges, gathers and combine
  cover exactly them (the combine skips order -1), as the on-card and split paths already assumed; only
  a count above the pairs (a plan bug) still refuses.
- **The decode graph door.** A captured span records the plan's arguments, and a call's map moves from
  call to call, so the map is one more input of the FFN half's span (`ffn_spans`): a replay points the
  plan at the map of the call it runs for. The door stays off on the served layout (`Group::can_capture`).
- **The fault door** `MEMRA_DSV41_CALL_FAULT` takes `step:request` beside `step`: `step:i` starts the
  calls of the running request at index i in that step from NaN streams (`Runner::poison`), so its
  router rows name no expert in every layer, as a request's data can make them, through the real
  kernels; `step` alone keeps the whole-step injection (a call word no map names).
- **The gate's ranks and plan forms** (the memra#1315 review). `dsv41-gate rowfault` takes
  `--devices` (default one rank on device 0; `0,0` is two ranks on one card, the N-rank path on one
  device), and poisons a 50-row call beside a clean one: 300 pairs, past the 256 the one-block plan
  takes, so the three-kernel plan skips the pairs and marks the rows. The first three poisoned calls
  (23, 29 and 34 rows) all took the one-block plan.

## Which faults fail what

| fault | named by | fails |
|---|---|---|
| a trunk router row naming no expert in a prefill, a verification, a one-token call or a batched row | the call's row map | that request alone |
| a router row naming no expert in a draft-head stage | no call (the unowned word) | the step's requests |
| the door's whole-step injection | no map | the step's requests |
| a draft's non-finite logits, a refused allocation, any other error of a call | the call's error, which carries no request | the step's requests (unchanged) |
| a non-finite logit behind a clean router | the row's own choice (`choose`) | that request alone (unchanged) |
| a lost transport, a failed probe after a step fault, a caught panic | | the route stops (unchanged) |

Draft-head faults stay whole-step: a draft's router fault marks the unowned word, or, when its draft
logits come out non-finite, fails the draft call itself, and neither says which request's data caused
it (the draft's error is not typed). A draft head reads its own request's trunk taps, and a request
whose trunk rows went non-finite has already failed alone in the step that computed them, before its
next draft.

## Why the other rows keep their bytes

A call of one sequence runs alone through every block. A batched call's rows meet only in kernels that
compute each row on its own: the norms and mHC mixes, the GEMMs (activations quantized in 32-wide
blocks of their own row), attention over each row's own sequence keys, the router, and the plan, which
places only in-range pairs (a non-finite router row names no expert at all, so the row enters no expert
tile) and leaves the faulted row's order slots -1 for the combine. That is the batched-equals-solo
property `dsv41-spec batch-rows` gates on finite rows; the smoke below checks it with a NaN row beside a
live one.

## Gates

One GB10 (121a, CUDA 13.1), the gates with MEMRA_FULL_PREC=1. Two sets: the change's tree on main
`df1d23a764` (`f2efe6a3c9`, `receipts/gb10-f2efe6a3c9/`), then the code commit rebased onto main
`bbc6f83aea` (`52f869186c`, `receipts/gb10-52f869186c/`). The rebase brought in an admission change
for unified memory (memra#1257: `unified_memory.rs`, the server's `worker.rs`); `dsv41-gate` built from
both trees is the same binary (sha256 `73bef935...`), and `memra-server` differs.

| gate | on `df1d23a764` | rebased, `52f869186c` |
|---|---|---|
| kernels (`fixtures-gpu/rand`) | 185 of 185 (main: 184 of 184, `kernels/base-main/`) | 185 of 185 |
| rowfault, resident trunk (2 layers) | 7 of 7 | 7 of 7 |
| rowfault, `--experts-host` (the served one-card residency) | 7 of 7 | 7 of 7 |
| serve smoke on the resident trunk (2 boots) | PASS, 14 checks | PASS, 16 checks (`smoke/`), after a harness miss (`smoke-try1/`) |
| red arm: the off-card MoE's old check restored | resident 7 of 7, `--experts-host` fails | |
| serving-step A/B against main, the resident trunk | no difference at the log's 1 ms | |

**Kernels.** The new check is `schedule/moe_plan/out_of_range_id_rows700` (the three-kernel plan: 4
out-of-range pairs in 700 rows, rows 5, 650 and 651 marked and no other); `out_of_range_id` (the
one-block plan) now also checks the marked row, at a non-zero `row0`.

**`dsv41-gate rowfault`** runs the serving route's runner on the checkpoint's first 2 trunk layers,
with a request poisoned the way the fault door does it (`Runner::poison`):

- two own calls in one forward, the second poisoned: every row of the second call is named and none
  of the first, and the first call's last-row logits (129,280) are its clean run's bytes;
- a batched call of two rows, row 1 poisoned: row 1 alone is named, row 0's logits are its clean
  run's bytes, and so are the logits of row 0's next step, run alone;
- a call flag with no row named fails the step, and clean steps name nothing.

`rowfault-try1/` keeps the gate's first run (the tree one gate fix earlier): the resident arm passed
7 of 7 and the `--experts-host` arm stopped before its first check, because the gate reserved 16 GiB
beside the weights and the expert cache got no slot ("the expert cache needs 384 slots (7219445760
bytes) for one slice and 0 fit beside the resident weights and the 17179869184-byte reserve"). The
gate now reserves what its forwards need plus 1 GiB.

**The red arm** (`redarm/`, `red.patch`: one line of `Group::moe` back to main's check, placed pairs
`!=` the rows' top-k): the resident arm still passes 7 of 7, as that path never had the check; the
`--experts-host` arm passes the clean-steps check and then fails at the first poisoned forward with
`route: 0 of 174 pairs placed` (the poisoned call's 29 rows, top-6, none placed), before any flag is
read. That is main's behaviour on the served residency: the forward's error carries no request, so the
route fails every request of the step. With the change the same arm passes 7 of 7.

**Serve smoke** (`memra-server`, the route on a 2-layer trunk, `MEMRA_DSV41_TRUNK_LAYERS=2`: a
rehearsal of the serving loop, not the checkpoint's answers. Residency: the resident trunk. A GB10
holds a 2-layer trunk's routed experts, so the route keeps them on the card: every step line of every
boot here reads `cache_slots=[]`, and so does every step of the A/B below. These smokes therefore do
not run the off-card MoE, the one forward path this change touches; MEMRA_CTX=8192, greedy, `ignore_eos`, 32
tokens for A and B, 8 for the others, the checkpoint's page cache dropped before each boot):

- boot 1, `MEMRA_DSV41_CALL_FAULT=24:1,64`: requests A and B share batched calls; step 24 poisons B
  (row 1 of a two-row call). B fails alone (HTTP 500, `engine_error`), A completes its 32 tokens, and
  A's answer is byte for byte the answer of A sent alone afterwards. Step 64, the whole-step
  injection, fails request C; D is served after it. The log: 1 request failed alone, 1 whole-step
  failure, `the ranks probe clean after the fault: serving on`; health: 5 requests, 2 request faults,
  no dead reason.
- boot 2, `MEMRA_DSV41_CALL_FAULT=0:0,25:0`: W's prefill (its own call, rows 0-33) is poisoned and W
  fails alone; then A and B share batched calls, step 25 poisons A (row 0), A fails alone and B
  completes with its solo answer's bytes. 2 requests failed alone, no whole-step failure; health: 4
  requests, 2 request faults, no dead reason.
- A's and B's answers differ, so the byte comparison can fail.

The first set's smoke used steps 8 and 9 (`scripts/smoke.sh`) and passed its 14 checks. On the rebased
head the same script missed (`smoke-try1/`, SMOKE FAIL): B reached the server at step 11, after the
door's step, so nobody was poisoned (`MEMRA_DSV41_CALL_FAULT: step 8 has no running request 1`), B
ran on and the step-64 injection landed on A's solo request. The route did what the door asked; the
harness lost a race. `smoke3.sh` moves the poisoned steps to 24 and 25 and checks that each was one
batched call of two rows; that run is the 16-check pass.

**Serving-step A/B** (`ab/`; `memra-server` of main `df1d23a764` against `f2efe6a3c9`'s, no fault
door, the 2-layer trunk, one boot a run in the order A B B A B A, the page cache dropped before every
boot; a run is two 64-token requests sharing batched calls, then one alone). Median `step_s` of the
decode steps per run, logged in whole milliseconds:

| arm | 1-row steps (n 66 to 86 a run) | 2-row steps (n 51 to 61 a run) |
|---|---|---|
| main | 16, 16, 16 ms | 24, 32, 32 ms |
| this change | 16, 16, 16 ms | 25, 32, 24 ms |

The card idled at 47 to 49 C before a run and read 51 to 52 C at 2,184 to 2,190 MHz after its last
request; no run was dropped. A clean step adds no read (above), and the step time shows nothing at
this resolution. Each prompt's answer is the same text in all six runs, so the change moves no token of
a clean request. `ab-try1/` is the first attempt, with the server one gate fix earlier and no cache drop: its
fifth boot refused ("device 0: 3.0 GiB free after the load, the runs need 5.5 GiB (+0.0 GiB for a
streamed layer): does not fit"; GB10 memory is the host's, so the checkpoint's cached pages count as
used), and its five runs read 14, 14, 22 against 16, 16 ms at one row and 20, 44.5, 33 against 22, 29 ms
at two.

CPU, `52f869186c`'s crates on the build box (120a, toolchain 1.97.1, `receipts/cpu/`): `cargo fmt --all
-- --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass. memra-engine's dsv41 lib
tests pass 19 of 19; `step_faults_name_the_faulted_rows_of_each_call_or_fail_the_step` is new (the
call word with a row map, with the unowned word, with no row, and clean). memra-server's lib tests:
1,264 passed, 18 ignored; `a_step_with_a_fault_on_one_row_fails_only_that_rows_request` is new (a
two-request batched step with row 1 faulted fails request 1 and not request 0), and the door's parse
test now covers `step:request`.

## The review's runs (memra#1315, 2026-11-14)

The independent review asked for the three-kernel plan, for N ranks and for the residency of every
run. The head with the gate's 50-row case and `--devices` (`50413a3a4d`), one GB10 (121a,
MEMRA_FULL_PREC=1 for the gates), `receipts/gb10-50413a3a4d/`:

| gate | residency and ranks | result |
|---|---|---|
| kernels (`fixtures-gpu/rand`) | | 185 of 185 |
| rowfault, 2 layers | resident trunk, one rank | 9 of 9 |
| rowfault, 2 layers | one-card (`--experts-host`), one rank | 9 of 9 |
| rowfault, 2 layers | resident trunk, two ranks on the card (`--devices 0,0`) | 9 of 9 |
| rowfault, 2 layers | one-card, two ranks on the card | 9 of 9 |
| serve smoke, 2 boots (`scripts/smoke5.sh`) | resident trunk | PASS, 18 checks |

- **The three-kernel plan.** The two new checks: a 50-row call poisoned beside a clean one has its
  rows 0 to 49 named and none of the clean call's, and the clean call's logits are its clean run's
  bytes. 50 rows at top-6 are 300 pairs, and the plan takes its one-block form only up to 256
  (`V41_PLAN_SMALL`), so the skipped pairs go through the three-kernel plan, and under
  `--experts-host` through `Group::moe`. A clean run of the same two calls names nothing.
- **N ranks.** Two ranks on one card run the N-rank path (each rank's plan over its expert range, the
  gathers) on one device. The fully poisoned calls, whose blocks are empty on both ranks, pass on
  both residencies; nothing there showed a limit. Two cards are a PRO 6000 run (Open).
- **The smoke's residency** is checked by the script now: both boots ran the resident trunk (72 and
  71 steps, none with an expert cache, no host-experts line). Its 18 checks are the 16 above and one
  residency check a boot.

CPU at `50413a3a4d` (`receipts/cpu-50413a3a4d/`, the build box): `cargo fmt --all -- --check` and
`cargo clippy --workspace --all-targets -- -D warnings` pass, memra-engine's dsv41 lib tests 19 of 19,
memra-server's lib tests 1,264 passed (18 ignored). `first-try-stale-build-script/` keeps the first
try, whose test builds stopped with "cc1plus: fatal error: cu/dsv41_cutlass.cu: No such file or
directory": the box's shared target dir handed this tree the build script compiled from the trial tree
below, whose kernel list has that file. The passing run touched this tree's sources first.

### The trial rebase onto the memra#1286 landing head

memra#1286 replaces the GEMM bodies under the plan and lands before this change, with memra#1312. A
trial: this change's three commits replayed on that landing branch's head `e225735efd`, memra#1312's
three commits replayed first (`29d734d95c`, a local commit; the branch itself stays on main until
memra#1286 is there). Three places conflict, the gate's usage text, `docs/FLAGS.md` and
`docs/KERNELS.md`, and each keeps both sides (`receipts/trial-29d734d95c/scripts/`, the two resolution
scripts); the route sources merge by themselves.

| check on `29d734d95c` | result |
|---|---|
| fmt, clippy `-D warnings` on all targets (`cpu/`) | pass |
| memra-engine's dsv41 lib tests, memra-server's lib tests | 20 of 20; 1,264 passed (18 ignored) |
| kernels, GB10 (`gset/`) | 1,027 of 1,027 |
| rowfault, 2 layers, GB10: resident and one-card, each at one rank and at `0,0` | 9 of 9 in each of the four arms |
| serve smoke on the resident trunk, GB10 (`smoke/`) | PASS, 18 checks |

So over memra#1286's GEMM bodies the other rows keep their bytes on both residencies (the rowfault
gate's byte checks) and through the serving loop on the resident trunk (the smoke's solo comparison).

**The rebase itself** (2026-11-14). memra#1286, memra#1294 and memra#1312 landed as main `63c351a2f4`,
and this change now sits on it. The trial's resolutions applied as recorded, and the rebased head's
`crates/` are byte for byte the trial tree's (the only difference under `crates/`, `docs/` and
`tools/` is main's own re-pin of the support record), so the trial's GB10 runs above are runs of this
code. Its kernel gate, 1,027 of 1,027, is also the one `dsv41-gate kernels` run on the combined
head; the landing ran none. The CPU checks ran again on the rebased tree (`receipts/cpu-rebased/`):
`cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings` pass,
memra-engine's dsv41 lib tests 20 of 20, memra-server's lib tests 1,264 passed (18 ignored).

## Open: the RTX PRO 6000 window

Not run yet. These need PRO 6000 cards and the commit this change lands as:

- **The serve smoke on the one-card residency**: the whole 40-layer trunk on one card, where the
  route picks that residency by itself. A GB10 cannot run it: it holds a 2-layer trunk's experts, and
  the route has no switch that puts them off a card with room.
- **`dsv41-gate rowfault --layers 40 --experts-host`** on one card. Everything above is trunk layers 0
  and 1 (ratio 0): no compressor layer (2 to 19) and no index-selection layer (20 to 39) has seen a
  non-finite row on any device.
- **`dsv41-gate rowfault --devices 0,1`** on a pair of cards.
- **The support record.** This is the first route-code change after the record's pin:
  `memra model verify checkpoint` and `verify serve` on one card at the landing commit, and the
  re-pin.
