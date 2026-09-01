# The A/B, rebased onto the residency config — plan, and the one thing that must be settled first

## The finding that changes the A/B, found from source before any box time was spent

Full two-card expert residency puts every routed expert in a device-resident slab. That makes
`dev_exps` present on every stage engine, which makes `slab_local` `Some` — and the fused
epilogue's predicate carried `slab_local.is_none()`. **On the config that is now the serving
config, the arm was denied outright.** An A/B run against it would have reported 0 dispatches and
read as "the fusion does nothing".

The engagement counter would have caught it after the fact; source caught it before. Fixed and
gated: `moe_fused_epi_launch` is now the single launch path for both provenances, and the gate
carries a `Placement` dimension. On the 5090, 13/13 pass, with the slab arm at **89/89
engagement** (asserted as `== OPPORTUNITIES`, not `> 0`: a slab holds every expert by
construction, so a shortfall means a predicate is denying the arm on the placement the product
serves on), fused-vs-unfused **0 bits differ**, and fused **SLRU-vs-SLAB 0 bits differ** — the
two pointer provenances feed the same kernel pair, so that is a provenance-only pair and the
strongest form of the claim.

## Integration branch, because the measurement needs both lanes in one binary

`lane/glm53-epilogue-x-pp` = `lane/glm53-epilogue` + `origin/lane/glm53-pp` (merge `459aa6445e`,
clean, no conflicts). The residency walk (`decode_step_hyper_ppn`) is PP's and is not in the
epilogue lane's base; the fused epilogue is not in PP's. Neither lane's history is disturbed:
this branch exists to be measured, not merged.

## THE BLOCKING UNKNOWN, and the self-check that settles it without guessing

`RESIDENCY-CELL.md` states the R2 result but not the exact env, and PP's scratch has been removed
from the box, so the invocation cannot be recovered. The reconstruction below is derived from the
receipt's own stated facts plus the code's env names — but it is **not** to be trusted on
derivation. It is trusted only if it reproduces R2's banked numbers:

    MEMRA_PP_SHARD / MEMRA_PP_DEVICES=0,1   (2-card sharded loader; "stage0=dev0 stage1=dev1")
    MEMRA_PP_SPLITS=24                       (stated in RESIDENCY-CELL.md)
    MEMRA_MOE_SLOTS=18144                    (21 MoE layers x 288 x 3 = 18,144 blocks per card)
    MEMRA_MOE_HARD_VRAM_FRAC=0.95            (the pin PP had to raise from 0.80)
    NO MEMRA_BF16_MMV                        (f32 trunk; owner decision outstanding)
    MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0

**Acceptance for the config itself, before a single A/B number is quoted.** The `=0` arm must
reproduce all three, or the config is wrong and the A/B is not run:

| check | R2's banked value |
|---|---|
| p5 greedy sha | `4ec98d8aeb7a30e6` |
| p5 sampled sha (seeded) | `bec0d19fbf181d1b` |
| p7 greedy sha | `fdf5109149b4ece8` |
| p5 greedy | 29.947 tok/s / 33.39 ms/token |
| VRAM card0 / card1 | 93,685 / 94,039 MiB |

Those shas are also the real-width bit-identity oracle this lane never got to close: the `=1` arm
must reproduce them too. If it does, real-width identity is settled on the serving config, at
`n_used=8`, across two devices — a far stronger statement than the fixture's 128/64.

## Traps carried into the run, all of them stated by the coordinator

1. **`moe_cache_stats()` is not wired through PP stage engines.** Under cross-device PP,
   `steady.py` reports `MB_per_tok 0.0` and `miss_per_tok 0.0` on every two-card row, which reads
   exactly like "staging went to zero" and actually means "not measured". The finding-3
   follow-up depends on those very numbers, so a zero from that path is not evidence. Staging
   will be derived from VRAM occupancy against the block arithmetic (18,144 blocks x 4.5 MiB =
   81,648 MiB per card) and the method will be named in the receipt.
   The engagement counter is NOT affected: `MOE_FUSED_EPI_DISPATCHES` is a process-global atomic
   and `PpNRt` binds one thread to each stage context within one process, so it aggregates
   across stages. `moe_cache_stats()` is per-Engine and does not.
2. **Headroom is under 4 GiB** at `MEMRA_MOE_HARD_VRAM_FRAC=0.95` (peak 96.1% of VRAM). The fused
   arm's only new allocation is the batched activation buffer, `n_used * n_ff_exp` f32 = 8 x 2048
   x 4 = 64 KiB per token-layer, against the sequential loop's 8 KiB per expert — the same order,
   transient, and ~5 orders below the headroom. VRAM peak will still be reported per arm rather
   than argued.
3. **The sampled p5 row loops in both arms** ("User references User references...") with the same
   sha, identical across placements, so it is a model/prompt property. Per
   LAW:greedy-is-the-instrument it is flagged and EXCLUDED from aggregates, and the exclusion is
   stated. p7 greedy carries the second prompt.

## Protocol

Interleaved x5 at boot granularity (these are process-level env behind `OnceLock`, the deviation
METHOD.txt already documents), 4 reps inside each boot, `idle-check.sh` before EVERY boot, p5
greedy + p5 vendor-default sampled + p7 greedy. Flag stays default OFF regardless of outcome
until the owner rules on the flip; the FLAGS row already names this A/B as the condition.

Still outstanding after it, and named so the flip is not surprised at review: the
LAW:multiturn-cache-twin 8-turn larger-prompt cache-on/off twin, which this lane has not written.

## Run status: config-validation boot started, box then became unreachable

`abres.sh V0 0` (the `=0` validation arm, reconstructed env above) was started on an idle box at
15:56:09Z after `idle-check.sh` passed and the integration binary built clean (4m45s).

Last observed state, ~90 s into the load: **card 0 = 3 MiB, card 1 = 20,563 MiB.** The load was
still in progress. Note the asymmetry — at that point weights were landing on card 1 only, which
is either the sharded loader filling stage 1 first or a sign the reconstructed `MEMRA_PP_SHARD` /
`MEMRA_PP_DEVICES` pins are not producing the `stage0=dev0 stage1=dev1` split R2 logged. It was
too early to tell, and it is exactly what the acceptance table above exists to decide.

The box then stopped responding to SSH (six attempts, connection timed out). **No conclusion is
drawn about the cause.** There is no evidence either way: this lane's arm was mid-load with
`MEMRA_MOE_HARD_VRAM_FRAC=0.95` against under 4 GiB of headroom, which is a plausible pressure
source, and the box independently rebooted once already today from an apt/kernel path that had
nothing to do with any lane. Determining which needs the box back and its logs read
(`journalctl -b -1`), the same way `BOX-INCIDENT-20260828.md` settled the first reboot from
evidence rather than inference.

**No A/B number exists and none is quoted.** The validation arm did not reach a single steady
row, so the reconstructed env is neither confirmed nor refuted against R2's banked shas.

What is ready the moment the box returns: the integration branch and its built binary
(`~/memra-epix`), `abres.sh`, the acceptance table above, and `idle-check.sh`. The first action is
not the A/B — it is reading the boot log to establish why the box went away, then re-running
`V0 0` and checking it against `4ec98d8aeb7a30e6` / `bec0d19fbf181d1b` / `fdf5109149b4ece8`
before any arm is treated as a measurement.
