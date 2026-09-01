# The A/B on the residency config — acceptance, characterization, results (2026-08-28)

Box: a rented 4x RTX PRO 6000 Blackwell Server 96 GB box (identity in the private ops repo;
all four cards verified at the full 600 W). Artifact `/root/models/glm53-nvfp4`, byte-verified
against the published HF repo. Binary: `ab-epi-x-pp` = `lane/glm53-epilogue-x-pp` merged with
lane head `ba9877b68` (the prefix-restore guard). `NVIDIA_TF32_OVERRIDE=0` everywhere;
`reasoning_effort` pinned `low`; `MEMRA_PREFIX_CACHE_MB=0` pinned in every A/B cell and stated.

## 0. Two config corrections found the hard way, both banked

- **The PP door is `MEMRA_PP_STAGES=2`.** `MEMRA_PP_SHARD` is only the rollback seam (`=0`
  disables sharding); setting `=1` opens nothing. The first V0 boot carried `MEMRA_PP_SHARD=1`
  and no `MEMRA_PP_STAGES`, so `pp_cuts()` returned `None`, the whole model loaded on one card
  (94 GB on card 1, 3 MiB on card 0), and the SLRU thrashed at 8 tok/s with 1,326 MB/token of
  staging. The acceptance table caught it exactly as designed: config fingerprint wrong, run
  discarded, no number quoted. This is ALSO what the earlier V0 boot was showing before the previous box
  died — card 1 filling alone was the one-card path, not "stage 1 loads first".
- **Residency on this recipe is the SLRU at full slot count, not load-time slabs.** The planner
  logs `experts 97.84GB vs budget 97.24GB -> SLRU cache` per stage, so `dev_exps` is None and
  the fused epilogue's SLRU provenance is the live one. The slab provenance added for this A/B
  is exercised by the fixture gate, not by this config — stated so nobody reads the slab arm's
  fixture receipts as covered-by-serving.

## 1. Acceptance-table verdict: env CORRECT, and the sha bank itself was contaminated

Corrected env (`abres.sh`): `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=24
MEMRA_MOE_SLOTS=18144 MEMRA_MOE_HARD_VRAM_FRAC=0.95`, f32 trunk, `MEMRA_CTX=8192`,
`MEMRA_MAX_SESSIONS=4`. Fingerprint reproduces R2: `stage0=dev0 stage1=dev1` transport line,
93.4 / 95.1 GB VRAM split.

Three boots isolate every variable (all on this box):

| boot | binary | prefix restore | p5 greedy sha | p5 sampled sha | p7 greedy sha |
|---|---|---|---|---|---|
| R2 banked (other host) | PP commit `876959009` | ON (pre-guard) | `4ec98d8aeb7a30e6` | `bec0d19fbf181d1b` | `fdf5109149b4ece8` |
| R2X | PP commit `876959009` | ON (pre-guard) | `4ec98d8aeb7a30e6` | `bec0d19fbf181d1b` | `fdf5109149b4ece8` |
| R2C | PP commit `876959009` | OFF (pinned) | `13614b57cd055adc` | `bd86f3dc82ef6b66` | `f0e2b850702f131c` |
| V0 | merged head | OFF (pinned + guard) | `13614b57cd055adc` | `bd86f3dc82ef6b66` | `f0e2b850702f131c` |

- **R2X = R2 exactly, on a different host.** All three shas byte-identical. The reconstructed
  env is right, and residency's byte-identity claim now holds CROSS-HOST on the real artifact.
- **R2C = V0 exactly.** The merged head introduces NO numeric change on the cold path; R2's
  commit and the merged head produce the same bytes when both run cold.
- Therefore the entire sha delta between V0 and the R2 bank is **restore-vs-cold**: R2's banked
  shas were produced through pre-guard prefix restores on a latent-plane model, i.e. they carry
  exactly the contamination class the prefix-restore-guard lane shipped to stop. The cold shas
  are the honest bank going forward. **Finding for the PP lane and the residency receipt, not a
  defect in this A/B.**
- **The banked 29.947 tok/s does not transfer to this host**: R2X (same binary, same env, same
  shas, restore on so elapsed ≈ decode-only) measures 26.27 tok/s p5. The ~12% gap is the host
  (same card class, different PCIe/system), which is why cross-box rate comparisons are made
  within one box only; every A/B number below is same-box, same-binary, interleaved.

## 2. Protocol

Interleaved at boot granularity (process-level env behind OnceLock; METHOD.txt deviation),
x5 boots per arm: V0/Z2/Z3/Z4/Z5 (=0) against E1/E2/E3/E4/E5 (=1), alternating, idle-checked
before every boot, 4 reps per battery after a warm pass, p5 greedy + p5 vendor-default sampled
(temp 1.0 / top_p 0.95, seeded) + p7 greedy, 192 tokens.

With `MEMRA_PREFIX_CACHE_MB=0` (and the guard refusing restores on this family regardless),
every rep pays a fresh prefill, so `server_toks` includes prefill. Decode-only rates are derived
as `completion_tokens / (server_elapsed_s - ttft_s)` per rep and reported alongside raw
`server_toks`; the method is stated here once and used identically for both arms.

`epi_per_tok` counts BOTH prefill and decode dispatches over completion tokens: p5 = (192
prompt + 192 completion) x 42 layers / 192 completion = 84.0 exactly, and p7 = (281 + 192) x 42
/ 192 = 103.47 exactly. Both match the measured medians to the last digit, which is the
engagement receipt: **42/42 MoE layers take the fused arm for every token, prefill and decode,
with zero fall-through** across every =1 boot.

`moe_cache_stats()` is per-Engine and unwired through PP stage engines — every PP row reads
`acc_per_tok 0.0`, confirming the trap. No staging number in this file comes from it. Staging is
derived from: (a) `disk_mib` per rep = 0.0 on every row after warm-up, (b) VRAM occupancy
(93.4 + 95.1 GB ≈ the full 2 x 18,144-block mass + trunk + KV, stable across reps and boots),
(c) the per-stage planner line. With every block resident, misses are structurally zero for BOTH
arms; the fused arm's admit-all-first order admits nothing on an all-hit token, so **finding 3's
1.19 MB extra staging is a thin-margin artifact and is ZERO in the serving regime.**

## 3. THE A/B RESULT (interleaved x5 boots per arm, 20 reps per cell, medians)

Raw rows: `ab-residency/ab-{V0,E1,Z2,E2,Z3,E3,Z4,E4,Z5,E5}.txt`. decode_only =
`completion_tokens / (server_elapsed_s - ttft_s)` per rep (prefill excluded; both arms
identically). `server_toks` includes the per-rep fresh prefill (prefix pinned off).

| battery | `=0` decode | `=1` decode | delta | `=0` ms/tok | `=1` ms/tok | ms saved |
|---|---|---|---|---|---|---|
| p5 greedy (instrument) | 26.109 | **30.097** | **+15.3%** | 38.30 | 33.23 | 5.07 |
| p5 sampled (THE PRODUCT) | 24.132 | **27.514** | **+14.0%** | 41.44 | 36.35 | 5.09 |
| p7 greedy | 25.861 | **29.798** | **+15.2%** | 38.67 | 33.56 | 5.11 |

| battery | `=0` TTFT | `=1` TTFT | prefill speedup |
|---|---|---|---|
| p5 (192 prompt tok) | 2.118 s | **1.058 s** | **2.00x** |
| p7 (281 prompt tok) | 3.086 s | **1.521 s** | **2.03x** |

- **The saving is a flat ~5.1 ms/token on every battery**, greedy and sampled, both prompts —
  the signature of a launch-overhead removal, not a bandwidth effect, and it is sized right:
  the attribution put the MoE loop's share of launch overhead at ~1750 of ~3200 launches inside
  17.2 ms, and this arm deletes ~1,890 launches/token (49 -> 4 per token-layer x 42).
- **Prefill halves** because glm5_next's prefill rides the same per-token sequential loop (the
  pairs/dev prefill arms are denied on this arch), so the fused epilogue removes the same
  launches there. Not predicted in the plan; measured, and mechanically explained.
- Separation is total: every =1 boot beats every =0 boot on every battery (5/5 sign-ordered);
  within-arm boot spread 0.8% (=1) and 3.5% (=0) against a 14-15% effect.
- **Byte identity at real width, closed.** All 30 STEADY rows across both arms carry the same
  sha per battery (`13614b57cd055adc` / `bd86f3dc82ef6b66` / `f0e2b850702f131c`), greedy AND
  seeded vendor-default sampled — the fused epilogue is byte-identical to the sequential loop
  at n_used=8 on the real 190.7 GB artifact under cross-device PP.
- **Engagement receipt:** `[moe-fused-epi] snapshot dispatches=303912` on E5 vs `0` on Z5;
  per-token medians 84.0 (p5) and 103.47 (p7) match (prompt+completion)x42/completion to the
  last digit = 42/42 layers, prefill and decode, zero fall-through. The rollback seam is proven
  at real width (=0 arm: zero dispatches).
- **VRAM:** card 0 93,429 (=0) vs 93,461 MiB (=1): +32 MiB, noise against the ~4 GB headroom.
- **The sampled-p5 loop did not occur** in any of the 20 cold sampled reps (coherent 825-char
  output, tail inspected). The banked loop observation came from restore-era rows; on the cold
  path at 192 tokens it is absent. Stated rather than silently un-excluded.

## 4. Flip recommendation, against the pre-registered condition

The FLAGS row pre-registered: "an interleaved x5 A/B on the RTX PRO 6000 bench class with the
real 190.7 GB artifact, plus the vendor-default sampled twin". That condition is now MET, on the
serving card class, on the residency serving config, with the product-shape row (+14.0%
sampled), byte identity, total engagement, a proven rollback seam, and no staging or VRAM cost.

**Recommendation: default ON for the glm5_next arch** (the predicate already scopes the arm to
sigmoid-router + PRE-clamped families; no other arch can enter it). Per the flags law the
default change itself is an owner decision and ships as its own FLAGS.md row edit; this receipt
is the measurement it points to. Named residual before the SERVING RECIPE adopts it (distinct
from the engine default): the LAW:multiturn-cache-twin 8-turn larger-prompt cache-on/off twin,
plus post-deploy sampled verification per the serving law.

## 5. Token ledger after this arm (all on this host's clock)

decode-only p5 greedy: 38.30 ms unfused -> 33.23 ms fused. The attribution's residual
(roofline ~15.9 ms f32 + non-MoE launch structure) now dominates: the KDA (~600 launches),
mHC pre/post (~540) and MLA chains are the remaining launch mass, and `MEMRA_BF16_MMV` (the
numeric-class door, owner acceptance outstanding) is the remaining roofline halving.
