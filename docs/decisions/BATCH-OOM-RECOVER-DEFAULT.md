# A batched decode chunk's untouched OOM retries once on the RTX PRO 6000 Workstation Edition, full power and Max-Q (2026-11-04, Max-Q 2026-11-05)

**Chosen:** on the full-power RTX PRO 6000 Blackwell Workstation Edition and, since 2026-11-05, the
Max-Q Workstation Edition (each from its own cell), when a batched decode
chunk fails with a quoted CUDA OOM before any session-state write (the engine's `step_guard`:
before the engine call, or inside the generic unsplit body before its first KV append or
conv-ring update), the worker takes one reclaim rung (parked sessions dropped, the device prefix
cache halved, the teardown fence, the model pools trimmed, the VMM graveyard reaped) and runs the
same batched step once more. The retried step is the same program on unchanged state, so each
request keeps one numeric program. A second failure, a torn step or a non-OOM error ends the
chunk as before. `MEMRA_BATCH_OOM_RECOVER` is deleted; the boot line `[admit-oom] batch OOM
recovery: ON|OFF (...)` states the card's arm. Code: `batch_oom_recover_default` in
`crates/memra-server/src/worker.rs`.

**Rejected:**

- The previous default on that card, where one quoted OOM on a batched chunk ends every session
  of the chunk with the typed overloaded error, peers included (DAY47 2.1). On the deciding
  serving shape it lost 8 of 9 requests on the spec-default route and 5 of 9 on the plain route.
- The retry on the RTX 5090 class for now. Its gate cells pass (arm i twice, arm j and j-vmm,
  each red twin red; DAY49 2.5, 2.6), but its registered serving shape read three of eight boots
  (`O1-off`, `O1-on`, `O2-on`) before the idle wait ran out, and the plain route's boots never
  ran. The 5090 keeps the error arm until those boots read.
- The Server Edition and every other card: no cell.
- Keeping the env door as a rollback seam. Owner, 2026-11-04: winners with positive receipts
  become defaults keyed to the card each receipt came from, and their switches go.

**Measurement that settled it:** `research/spill-b-20260919/DAY49.md`, one RTX PRO 6000 Blackwell
Workstation Edition at 600 W. Gate arm j (the fault aimed at a batched chunk of at least two
sessions, addendum D) PASS four times on a 3-session chunk, its red twin red each time; j-vmm
green with one reap per retry, red with none (2.3). The serving shape (2.4), both routes and both
orders: every `on` boot 9 of 9 `200` with one retry and one `retried (ok)`; `off` ends the faulted
chunk's sessions (8 of 9 `503` on the spec default, 5 of 9 on the plain route). Arm i's unaimed
fault landed on a solo spec step on the 27B (2.1), which is why the gate's default list runs arm j
(addendum E).

The Max-Q row is DAY49 2.7 (`pro-maxq-day49f/`), pre-registered in addendum F before any Max-Q
boot: one Max-Q card at a 300 W limit beside training on the box's other cards, the same 27B,
gates and serving shape. Arm j PASS on both gate units, its red twin red each time; every `on`
boot 9 of 9 `200` with one retry and one `retried (ok)` on both routes and both orders; `off` lost
8, 8, 3 and 5 of 9. j-vmm reads `n/a` there: the on-demand KV planes are off on the Max-Q, so the
reap it checks cannot run.

**What would change it:** the five unrun 5090 boots of `rtx5090-day49d/run.sh` (on its
registered tree) reading as the target card did adds the RTX 5090 class. A cell on the Server
Edition adds that edition.
