# GPU bank budget refusal: options for the lead

Status: **lead decision needed.** No default, flag parse, or `docs/FLAGS.md` row
changes in this lane. `MEMRA_MOE_SLOTS` (`docs/FLAGS.md` row "force an exact SLRU
slot count") keeps its documented behavior until the lead picks an option.

## The seam as it is

`MoeSlotCache::new` (`crates/memra-engine/src/moe_cache.rs`):

- `forced_slots = MEMRA_MOE_SLOTS.parse::<usize>().ok()`; an invalid string
  silently falls back to auto sizing (`MEMRA_MOE_VRAM_FRAC`, default 0.85).
- `requested_bytes = n.saturating_mul(max_block_bytes + 8)`; an absurd `n`
  saturates to `usize::MAX` and then `min(hard_bytes)` turns it into the hard
  ceiling instead of a refusal.
- `budget_bytes = requested_bytes.min(hard_bytes)` where `hard_bytes` is
  `MEMRA_MOE_HARD_VRAM_FRAC` (default 0.80) of free VRAM minus two blocks.
- `n = (budget_bytes / (max_block_bytes + 8)).max(8)`: forced 0 to 7 allocates
  eight slots anyway, above the requested bytes and possibly above the hard
  ceiling on a tight card.

So the GPU side of the `--experts-via-tier` door cannot express "this budget
cannot hold the bank; refuse before touching the device". The host side can:
`host_bank_slots` (`banked_residency.rs`) refuses below one record and above the
256 MiB qualification ceiling, and `DAY8.md` banked a native refusal cell for it.
The day-nine 8 GiB cells used `MEMRA_MOE_SLOTS=9986`, well above the clamp, so
they never met this seam.

Fixed for every option below: refusal happens before the first CUDA slot
allocation, before any source read, pending H2D, or published bank record; the
message names the requested and the minimum bytes; a collector wrapper maps only
that exact typed refusal to `REFUSED: <reason>` exit 2 (lead ruling 6); any new
`MEMRA_*` read gets its FLAGS row in the same commit; native cells run on the
target card through the collector and are N=1 development evidence.

## Option A: keep the clamp, add a separate byte budget for the door

Legacy `MEMRA_MOE_SLOTS` is untouched. The installer takes a gate-only budget,
mirroring the existing `--expert-bank-host-bytes=N`: `--expert-bank-gpu-bytes=N`
(CLI, so no FLAGS row; decide-by in `MOE-SLOT-CACHE-DOOR.md`). When present the
installer computes the slot count itself with checked arithmetic and passes an
exact count into the cache; when absent, behavior is exactly today's.

- Fail-closed: `N < 8 * (max_block_bytes + 8)` refuses
  `experts-via-tier GPU bank budget cannot hold the eight-slot minimum
  (requested N, minimum M)`; `N > hard_bytes` refuses `... exceeds the hard VRAM
  ceiling (requested N, ceiling H)`; checked multiply overflow refuses
  `... overflow`. Never raise `N` to fit eight slots; never silently take the
  ceiling. The cache constructor gains a `forced_exact: Option<usize>` that
  refuses if the count cannot be allocated instead of clamping.
- Test shape: CPU pure function `gpu_bank_slots(bytes, max_block_bytes,
  hard_bytes) -> Result<usize, &'static str>` next to `host_bank_slots`, with
  cells zero, minimum minus one, exact minimum, minimum plus one, `u64::MAX`,
  hard ceiling below minimum; a `run-gen` argv test that the flag is parsed only
  with `--experts-via-tier`. Native: minimum minus one refuses before any
  `[expert-gpu-slru]` or `[expert-host-slru]` line; exact minimum runs gen and
  spec with tapes identical to the day-nine controls and explicit evictions;
  legacy default and forced 9986 controls unchanged.
- Cost: two knobs express one quantity; the legacy clamp stays a silent floor for
  everyone outside the door. Smallest change; touches `moe_cache.rs` and
  `native.rs` (engine files, so the perf-ci pre-push gate applies).

## Option B: lift the clamp on `MEMRA_MOE_SLOTS`

Change the legacy flag: forced `0..=7` refuses, invalid strings refuse, overflow
refuses. `.max(8)` remains only for the auto path, where it is a floor on a
computed value, not on a user request.

- Fail-closed: `MoeSlotCache::new` returns
  `MEMRA_MOE_SLOTS=<n> below the eight-slot minimum (requested N bytes, minimum M)`;
  `Engine::with_moe_cache` propagates it on the first MoE dispatch. That is after
  model load and inside the first forward, so the refusal is late relative to
  options A and C, and every caller of `with_moe_cache` (run-gen, run-spec,
  kernel-check, the server worker) sees the new error class.
- Test shape: extract `plan_uniform_slots(forced: Option<usize>, budget_bytes,
  max_block_bytes) -> Result<usize, SlotPlanError>` as a pure function and unit
  test the same boundary cells plus invalid string and auto-path floor. Native:
  forced 7 refuses; forced 8 runs; the documented spill controls 64 and 512 run
  with unchanged tapes; auto default unchanged.
- Cost: changes a documented flag's behavior (FLAGS row edit plus an owner
  decision recorded in `docs/decisions/`), and any launcher that sets a value
  below eight starts failing. A repo grep finds only 64 and 512 in receipts and
  9986 in day nine, none below eight, but external users of the public engine may
  differ. Single knob, most honest, latest refusal point.

## Option C: refuse at plan time with a typed budget

The typed proposal from day nine. `CacheBudget::LegacyAuto` and
`CacheBudget::Strict { bytes: u64 }` at the constructor; `try_plan_slots(layout,
budget) -> Result<SlotPlan, CacheBudgetRefusal>` runs before the first CUDA
allocation; the installer requests `Strict` through a typed argument, never by
reinterpreting an environment variable. Legacy env translation stays
`LegacyAuto` until separately decided.

- Fail-closed: checked addition and multiplication for every exact capacity plus
  the eight-byte tail; overflow is `InvalidLayout`, never saturation. Requested
  bytes and measured hard headroom are enforced independently; never raise the
  caller ceiling to fit eight slots. Uniform layouts refuse `BelowMinimum {
  requested, minimum }` under `8 * (max_block_bytes + 8)`. Mixed layouts need a
  separately proved minimum per capacity class; unsupported class minima refuse
  rather than substituting max-sized uniform slots. The exact plan bytes are
  reserved through the governor before allocation; partial allocations and the
  reservation unwind on failure and the original error is kept (a budget refusal
  is not CUDA OOM). On refusal no slots, reads, pending H2D, or published records
  exist.
- Test shape: CPU boundary tests zero, minimum minus one, exact minimum, minimum
  plus one, integer overflow, hard headroom below minimum, mixed-class refusal;
  allocation count and charge restoration on each injected allocation failure.
  Native: minimum minus one refuses before dispatch; exact minimum runs gen and
  spec with unchanged tapes and explicit evictions; legacy default and forced
  controls as regressions.
- Cost: largest change (new types, a refactor of `MoeSlotCache::new`, a governor
  reservation path in engine code), so the most native cells and the longest
  perf-ci exposure. Covers mixed layouts, which A and B do not.

## Comparison

| | A: clamp + door budget | B: lift clamp | C: typed plan-time |
|---|---|---|---|
| Legacy flag behavior | unchanged | changed (FLAGS row, decision record) | unchanged until decided |
| Refusal point | installer, before load finishes | first MoE dispatch | constructor, before first allocation |
| Mixed layouts | no | no | yes |
| FLAGS row needed | no (CLI) | edit existing row | no, unless env translation changes |
| Engine files touched | 2 | 1 | 1 plus governor wiring |
| CPU test surface | pure function | pure function | pure function + injected alloc failures |

Recommendation held back on purpose: the lead owns flag policy and the PP budget
design this interacts with. This lane will implement whichever option is chosen
with the listed tests and the native target-card cells, and will not change a
default before that.
