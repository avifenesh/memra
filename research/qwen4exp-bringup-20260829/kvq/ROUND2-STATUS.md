# qwen4_exp round 2 — status at the THIRD spot reclaim (2026-08-31)

Round 2 (KV-quant + TP2-prefill: box baseline, TP2-prefill class gate, the 1M ladder,
spec at depth) is **BLOCKED ON HARDWARE**, not on code. This file is the honest state so
the next box starts from receipts instead of from scratch.

## The reclaim

| field | value |
|---|---|
| box | round-2 box 3, a **preemptible (spot) two-card box**. Provider, region, instance class and instance/request ids are fleet state and live in darklanes, not in the public engine repo. |
| cards | 2x RTX PRO 6000 Blackwell **Server Edition**, 97,887 MiB each, 600 W limit (verified on the box: same class as round 1, so round-1 numbers ARE the comparison) |
| launched | 2026-08-31 09:46:21 GMT |
| terminated | 2026-08-31 10:28:35 GMT — **42 minutes** |
| reason | provider-initiated preemption: **no preemptible capacity available** for the request |
| request type | **one-time** — nothing auto-replaces it |

Third preemption in this lane (2026-08-29 x2, 2026-08-31 x1). No replacement hardware was
provisioned from the lane — capacity hunting is the owner's path, and which provider is
allowed is recorded in darklanes.

## What DID reproduce on box 3 before it died (work item 1, partial)

Phase 0 ran with **no seam env at all** (the PROFILE-9 §7 doctrine: a default is a claim
about what runs when nobody passes a flag, so it is verified by arming nothing and
asserting an outcome). Note the binary's reference-parity pin: `--goldens`/`--prompts`
force the f32 exactness-instrument cache arms even under the flipped serving defaults, so
the no-env hidden/greedy rows are the **kvq0** rows and round 1's kvq0 receipts are the
right comparison.

| arm | round-1 receipt | box 3 | verdict |
|---|---|---|---|
| tiny gate (`qwen4exp_gpu_gate`), no seam env | `tiny-gate-defaults-on-box.tsv`, 263 rows / 0 failures | every arm PASS, rc=0 | **AGREES** |
| real-checkpoint hidden goldens | `hidden-gate-kvq0-defaults.tsv` | logits rows **IDENTICAL TO EVERY PRINTED DIGIT** on all 10 rows; argmax **10/10** | **AGREES (exact)** |
| greedy first-divergence, raw, f32 arm | kvq0 pattern `-1 / 8 / -1 / 48` | `-1 / 8 / -1 / 48` | **AGREES** |
| verify-bit 24 (ship admission) | 24/24 bit-identical | `rows=24 mismatched=0 policy=bit-identity pass=true` | **AGREES** |
| spec byte-identity >= 256 tokens, raw | PASS | `policy=byte-identity pass=true` | **AGREES** |

**Box-baseline verdict: no delta found.** The strongest arm is the hidden-goldens
comparison, and it is the strongest for a specific reason — the two runs printed the same
`max_abs`/`max_rel`/`mean_abs`/`ref_absmax` on all 10 rows and the same 10 top-1 ids, on
two different physical machines, which is exact agreement rather than agreement inside a
tolerance. Same card class, and the numbers behave like it.

Transcribed side by side (round-1 file left, box-3 run right — the full rows are in
`box/hidden-gate-kvq0-defaults.tsv`; box 3's own TSV died with the box, see below):

```
logits        10  248320  6.183e0  4.901e0  2.689e-1  1.950e1     <- both runs
row 0  3.553e-1  3.553e-1  1.231e1  20438  20438  true            <- both runs
row 4  1.883e0   1.825e0   1.888e1    888    888  true            <- both runs
row 9  6.183e0   4.901e0   1.638e1    271    271  true            <- both runs
# logits_argmax_agreement  10/10                                  <- both runs
```

### NOT reached (the box died mid-battery)

- greedy raw with `MEMRA_Q4E_SEAMS=kvq` armed (expected the kvq1 pattern `-1/8/-1/26`):
  the run completed rc=0 but its rows were never read or copied.
- spec byte-identity thinkon; `--tp2-gate 24`; the live router audit row count.
- Work items 2-4 entirely: the TP2-prefill class-gate calibration, the 1M ladder
  (100k/262k/600k/1M), spec at depth. **No number from any of those is claimed anywhere.**

### Receipts LOST, and that is a process failure, not bad luck

Phase 0's TSVs were written to `~/realgate/kvq2` on the box and **never scp'd to the rig**
— they were read over ssh instead. The lane's own instruction was to bank receipts
promptly because the box is spot, and 42 minutes was not enough runway for "bank at the
end of the phase". The five agreements above survive only because their key lines happened
to be read out loud before the box died; the raw files are gone.

**Corpus-worthy, spot-box class:** on a spot box, a receipt is banked when it is on the
rig, not when it is written. Copy each receipt in the same breath as the run that produces
it (`scp` per arm, not per phase) — or better, have the battery script push each TSV as it
lands. Reading a verdict over ssh is not banking it.

## What is READY on the rig for the next box (no GPU needed to redo)

The two reproducibility gaps that made the previous reclaims expensive are closed, and
both were verified working on box 3:

- `gpu-eager/expand-goldens.py` — rebuilds the `--goldens` f32 bins from the in-repo
  `hidden-goldens.pt` with torch alone. `prep-real-gate.py` needed the **336 GB BF16
  checkpoint** only to re-tokenize `prompts.tsv`, and `prompts.tsv`/`manifest.tsv`/
  `input-ids.txt` are all already mirrored in this lane — so that download was never
  actually required. Re-derives the manifest and input-ids and hard-compares them against
  the mirrored copies. Verified on box 3: *"expanded 50 records (28.3 MiB of f32 bins),
  probe T=10; manifest.tsv + input-ids.txt VERIFIED identical to the mirrored copies"*.
- `yarn/make-ladder-ids.py` — mints the ladder corpus (`--ladder-ids`, the yarn cell's
  `ids=1150000`) from the memra tree's own Rust/CUDA/doc sources in a pinned sorted order
  using the artifact's `tokenizer.json`. That file only ever existed on a box and every
  reclaim took it. Verified on box 3: *"ladder-ids: 1150000 tokens from 295 files
  (14422038 chars), corpus_commit=0e0ef7c69"*. It refuses to repeat the corpus to reach
  width, because a repeated prefix would make the depth rungs measure self-similar text
  instead of real long context.
- `yarn/mk-override.py` — **fixed; as committed it could not have worked.** It minted the
  yarn-1M dir with `os.symlink` while YARN-CELL §1 records the opposite in the same lane
  ("Symlinks are refused by the loader's snapshot containment check — hardlinks are the
  working form"), so the yarn cell must have hardlinked by hand. Now `os.link`, and it
  skips directories (a downloaded mint carries an hf `.cache/` dir and `os.link` refuses
  directories with EPERM, which aborted the whole mint on box 3). Verified: 20 files
  hardlinked, 0 extra disk, `rope_type=yarn factor=3.814697265625 original=262144
  mpe=1000000` read back.
- `kvq/run-round2-p0.sh` — the phase-0 baseline battery, as run.

## Engine work banked this round (rig-built, gates NOT yet run on a box)

Commit `8a1b7348b`. Three seams; the code compiles clean and no perf or exactness number
is claimed for any of it yet.

1. **Pluggable per-layer expert placement**, `MEMRA_Q4E_EP_MAP`, default OFF = even split.
   Reads the FROZEN shared `memra-ep-map-v1` minted by
   `tools/build_expert_placement_map.py` (merged on main), not a lane-local format, so a
   map minted from qwen4_exp traces is comparable with the glm5 arm's. The split was
   hardcoded in three places (bank upload, decode route, prefill route) and is now ONE
   `LayerPlacement` carried on the shard, so upload and routing cannot disagree. Card 1's
   bank became a gather in ascending-global-id order; for the even arm that gather
   concatenates exactly the bytes the old suffix slice handed over, so **even split =
   control arm is a statement about bytes, not a hope**. Fail-closed on format, `ranks!=2`,
   expert-count mismatch, uncovered layer, out-of-range rank, and an **unbalanced** layer
   (the card-1 bank halves are equal-size allocations, so unbalanced is out-of-bounds, not
   merely slower).
2. **Route traces in the shared `MEMRA_MOE_TRACE` format** (`<layer> <t> <id,id,...>`),
   byte-compatible with `hybrid_forward.rs::trace_moe_routes`. Default OFF. Honest limit,
   stated in code: TP2 keeps the host router twin by construction so TP2 batteries trace
   for free, but the shipped single-card default routes on DEVICE with no readback at all
   (`routerdev` deleted exactly that sync, PROFILE-9), so single-card tracing needs
   `MEMRA_Q4E_ROUTER_AUDIT=1` and rides that host recompute at zero new syncs.
3. **Per-rank engagement counters + a two-regime TP2 class gate.** See the next section —
   the gate replaced a bar that could not have caught what it was for.

## The TP2-prefill gate had two real defects (found by reading, not by measuring)

The pre-existing `--tp2-prefill-gate` policy was `max_rel <= 0.01 && argmax match` on
every row. Both halves of that were wrong:

- **The bar was calibrated against nothing.** 1e-2 is ~50x looser than the prime band the
  glm5 TP-2 lane calibrated for the identical numeric class (2e-4, itself 10x its measured
  green worst of 4.85e-5) and ~300x looser than what our own `--tp2-gate` receipt already
  measured for the t=1 join class (3.0e-5). A bar that loose passes a genuinely wrong
  program whose error happens to be small.
- **It compared only the chunked prefill's LAST ROW** — a t==1-shaped read of a t>=2
  program. An interior-row prefill defect was structurally invisible to it.

Replaced with a two-regime class gate: PRIME (t>=2) compares **every row** of one
full-head forward per route against a calibrated near-tie band plus greedy tape identity;
DECODE (t==1) gets its own tighter band. It deliberately does **not** claim decode byte
identity — glm5 got that because their program was column-parallel-over-gather and ours is
not — but reports `decode_byte_identical` as a measured field, so byte identity would show
up as a finding rather than being assumed as the bar. RED arms
(`MEMRA_Q4E_TP2_GATE_RED=skip-peer-moe|peer-local-ids|reverse-peer-weights`) must land
past `RED_FLOOR` or break the tape, because a band is only a bar if a wrong program lands
orders outside it. `--tp2-class-calibrate` measures without barring.

**The band constants in the code are PLACEHOLDERS.** They are set at 10x a *borrowed*
green worst, and the calibrate-downward law says a band comes from this gate's own
measurements on this artifact and card class. The first thing the next box does is the
calibration run; until that receipt exists, no green from this gate should be quoted.

## Checked by construction, so the next box does not have to discover it

**TRAP:grouped-prefill-monolithic-workspace does NOT apply to this TP2 prefill.** The glm5
lane's 9-GiB-free shape walled at ~7-8k prompt tokens because a per-REQUEST-sized
workspace was handed the whole prompt (their `prime_cache_hyper_ppn` short-circuited the
chunk walk, ~0.8-1.0 MiB/token/card). Read for the same shape here:

- `Qwen4ExpState::reserve` is the workspace-slot reserve unit and `alloc_state_reserve`
  caps it at the **CHUNK** bound, with the reason in-comment at both the single-card and
  TP2 sites ("a long-context TP2 state (1M rows) must not reserve capacity-sized plane
  slots (~10 GB each)"). `prefill_extend_tp2` walks `ids.chunks(chunk)`; there is no
  monolithic short-circuit equivalent to the ppN twin.
- `tp2_moe_rows` sub-batches at `SLOT_CAP = 8192` **slots**, so the grouped-MoE staging
  transients are bounded by the slot cap regardless of `t`.
- The other O(capacity) risk at 1M — the dense `[t, capacity]` attention mask slot — is
  never allocated at the shipped defaults: with `kvq` ON the quantized cache has no
  masked-kernel form at all and the block-list path is the only read path at every depth.

Still to be **measured** on a box (reading is not measuring): per-chunk VRAM held across a
deep TP2 fill should be flat in prompt depth, not growing. That is a row in the ladder's
VRAM table, and it is the empirical version of the check above.

## Correction to carry forward: the glm5 EP fractions are DERIVED, not measured

The tp2 brief cites the glm5 lane's naive even split as "measured ~1.57x effective, peer
rank touched ~99.3% of layer-tokens, slowest rank ~64% of expert bytes". Those numbers are
real but they are **closed-form derivations**, not measurements: `1 - 2*C(144,8)/C(288,8)`
for the touch fraction and `E[max(k0, 8-k0)]/8` with `k0 ~ Hypergeometric(288,144,8)` for
the byte fraction. Their branch has no per-rank token-touch or byte-fraction metric in code
or receipts, and §9 of their SHARD-MAP says so ("Expected value, restated honestly (no
measurement claim)"). What they DO log is a narrower engagement counter
(`GLM5_EP_PEER_SLOT_DISPATCHES`), and only because a first seed search found a token stream
that never routed a peer expert — which would have made the arm's identity claim vacuous.

This lane's counters are the measured version for the qwen4_exp geometry, and any A/B
against an expert split here reports `peer_slot_fraction` and `both_card_fraction` from
counted rows beside the verdict. Do not restate the glm5 fractions as measurements.

## Resume order on the next box (ascending cost)

1. `expand-goldens.py` on the mirrored `hidden-goldens.pt`; scp the shape prompts; mint the
   yarn-1M hardlink dir; `make-ladder-ids.py`. All CPU, minutes, no download.
2. `run-round2-p0.sh` — finish the baseline (the three arms box 3 did not reach), **scp'ing
   each TSV as it lands**.
3. `--tp2-class-calibrate` → read the measured prime/decode worst → set the band constants
   from that receipt (downward) → re-run as a gate → run the three RED arms and confirm
   each is loud and engaged.
4. The 1M TP2 ladder ASCENDING (100k / 262k / 600k / 1M) so a reclaim keeps the rungs
   already banked. Per-card VRAM, decode tok/s at depth, prefill wall, continuation
   coherence, and the flat-workspace check above.
5. Spec at depth (single-card by construction — `--ladder-tp2` refuses a spec arm),
   32k/100k per shape at ship admission, card-1 draft.
6. Route traces from every battery (`MEMRA_MOE_TRACE`), banked by shape and depth, then
   `tools/build_expert_placement_map.py` for the placement lane's input. Minting the map
   is in scope; the clustering/placement A/B is the NEXT lane.
