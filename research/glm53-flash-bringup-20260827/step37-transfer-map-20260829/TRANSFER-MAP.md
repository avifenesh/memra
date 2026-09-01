# step37 -> glm5_next transfer map (2026-08-29)

Owner question this answers: did we go over ALL of step37's work for ideas transferable to
GLM-5.3-Flash, or only opportunistically? This document is the systematic pass. Sources
walked: the step37 squash `068cbc4253` and its lane history, every `MEMRA_STEP_*` /
`MEMRA_BF16_MMV` / `MEMRA_PP_BF16` / `MEMRA_SWA_RING` / `MEMRA_PRIME_TROWS` /
`MEMRA_STEP_GEMM_PRIME*` row in `docs/FLAGS.md`, `research/step37-*` on origin/main
(nvfp4-tp2, decode-headroom, bf16-mmv, sampled-quality, draft-graph, 30k-cell,
reasoning-effort incident, affinity), `research/gemm-suffix-20260828`,
`research/dualpp1-20260811`, and the curated TP2 join-diet corpus. FLAGS line anchors below
are as of origin/main `4d9cf5747`; lane anchors as of origin/lane/glm53-flash-bringup
`dd7f1d11d`.

Standing law (no-generic-model-support): step37 (step35 arch class: GQA attention, SWA/global
4-cycle, 288-expert sigmoid MoE, NVFP4 experts, 3 MTP heads) and glm5_next (34 KDA linear
layers + 11 MLA/DSA layers, mHC 4-stream hyper-connections residual, 288-expert sigmoid
`noaux_tc` MoE + shared expert, NVFP4 experts + BF16 trunk, 1 NextN/MTP layer) are distinct
semantic programs. Every row below is a HYPOTHESIS carrying step37's evidence, never a
support claim; each names whether the mechanism is architecture-neutral or step-specific,
and the acceptance gates glm5 owes before any of it is believed on glm5.

## The two targets, and where glm5 stands

- Decode: ~22 tok/s serving-shape today (box A/B, PP3 full residency, 4.6k to 6.5k
  contexts: greedy 21.4 / 22.1 / 22.0 OFF-arm, 21.3 / 22.0 / 21.9 ON-arm;
  `moe-grouped-prefill-receipts/box-ab-20260829/BOX-AB.md`). Short-prompt residency cell:
  29.95 tok/s = 33.39 ms/token, split "~0 staging / 15.9 roofline / 17.2 launch", "LAUNCH
  STRUCTURE IS NOW 51% OF THE TOKEN" (`decode-attribution-receipts/ROADMAP.txt` step 3).
  Owner target ~90 tok/s = 11.1 ms/token.
- Prefill: 616 to 639 tok/s after the `MEMRA_MOE_GROUPED_PREFILL` default-ON flip
  (TTFD 54.16 -> 7.51 s / 65.53 -> 8.90 s / 75.86 -> 10.25 s, sampled twin 7.24 s;
  BOX-AB.md; lane `docs/FLAGS.md:475`). Owner target: thousands.

Already transferred from step37, recorded as DONE, not re-proposed here:

- `MEMRA_BF16_MMV` (the door itself; ratification still owed, lever 3).
- `MEMRA_PP_BF16` tensor-core trunk prefill: lane/glm5-tc-trunk-prefill has the L2 arm and
  a pre-registered box A/B (`3cb38ff0e`, `88c4841a4`).
- The grouped NVFP4 GEMM prefill class: `MEMRA_MOE_GROUPED_PREFILL` generalized the step37
  `MEMRA_STEP_GEMM_PRIME` kernel class (measured there "170-270 TFLOP/s",
  `docs/FLAGS.md:553`) off the TP runtime; flipped ON with receipts (`dd7f1d11d`).
- Most of the measurement craft: interleaved x5 fresh-boot arms, engagement announce lines
  in BOTH arms ("the step37 trap, closed", BOX-AB.md), vendor-default sampled twin,
  reasoning_effort pinning, binary sha per boot. Remaining gaps are lever 7.

---

## Ranked transfer map

Ranking is by expected value against the two targets, discounted by evidence quality and by
what is already in flight on sibling glm5 lanes. "ARITHMETIC" marks derived numbers;
everything else is quoted from its receipt.

### 1. KDA + mHC decode launch diet: fused multi-projection GEMV and epilogue-class kernel consolidation (decode, and the post-L1 prefill residue)

- Mechanism: single-input multi-output GEMV fusion plus persistent-workspace / no-host-sync
  restructuring. Architecture-NEUTRAL mechanism; glm5-specific census and algebra.
- step37 evidence FOR:
  - `MEMRA_STEP_TP_QKV_FUSED` (`docs/FLAGS.md:988`): ONE launch per rank replaces "~12
    chunked cuBLASLt calls per rank", 22.88 -> 24.99 tok/s, numeric class "2.4e-3 ->
    4.2e-3", acceptance = run-gen argmax gate + 12-boot battery.
  - `MEMRA_STEP_TP_DECODE_V2` (`docs/FLAGS.md:989`): v1 spent "81% of its 1550us/layer
    wall" on per-token allocs, host round-trips and host stream syncs; v2 is the same
    kernels on a persistent workspace with "exactly one cuMemAlloc" per layer per token.
  - `MEMRA_BF16_MMV` head receipt (`docs/FLAGS.md:986`): "head 1.40 -> 0.74 ms/token" from
    a one-block-per-row bf16 matvec, the kernel class glm5's BF16 trunk already rides.
- glm5 evidence FOR: the launch term is measured, invariant, and dominant: "17.1
  ms/token above the bandwidth roofline, UNCHANGED by both bandwidth levers"
  (`decode-attribution-receipts/ATTRIBUTION.txt`); launch census ~3200/token at "~5.3 us,
  which is exactly kernel-launch latency" (same file). After the fused epilogue removes the
  MoE share, "The KDA (~600), mHC pre/post (~540) and MLA chains are NOT touched by the
  epilogue, so the residue still needs its own launch-structure arc" (ROADMAP.txt step 4).
  The KDA projections (q/k/v, f_a/f_b, g_a/g_b, out) all read the same post-norm input and
  are all BF16 (`CENSUS.md`: "All KDA projections are BF16"), exactly the shared-input
  shape the fused kernels exploit.
- Evidence AGAINST / risks: the step37 fused kernels are F32-mirror and q8 kernels over
  step-TP canonical chunks; none of the kernel code transfers as-is, only the shape of the
  program. KDA adds the short conv and the decay/gate algebra between projections; fusion
  boundaries must be chosen from a fresh nsys census, not copied. mHC sinkhorn is already
  batched per site (~8-10 launches per layer per chunk at prefill, `PREFILL-GAP.md` 1.5);
  its decode t=1 launch count (~540/token) is the number to attack, likely via merging the
  pre/expand/collapse/post chain per layer.
- Expected value (ARITHMETIC): KDA ~600 + mHC ~540 launches at ~5.3 us is ~6.0 ms of the
  17.2 ms launch term. A 3-4x consolidation in the QKV_FUSED class saves ~4-4.5 ms:
  post-epilogue ~33.2 ms -> ~28-29 ms, and it compounds with every later lever. It also
  trims the post-L1 prefill residue (trunk kernels are chunk-wide but the same chains run).
- Acceptance owed (the step37 form, transferred verbatim): run-gen prefill/decode argmax
  gate with the logit-maxdiff CLASS stated, fresh-tape identity where the change claims
  bit-exactness, 12-boot battery, interleaved x5 on the serving card class, sampled twin.
- Status: NOT started; no sibling lane owns it. This is the map's top new lever.

### 2. Speculative decode over the NextN/MTP layer, riding the new batched mHC walk (decode)

- Mechanism: MTP-head drafting + t-column batched verify. The verify walk and the policy /
  gate corpus are architecture-neutral craft; the draft chain and head loading are
  model-specific programs.
- step37 evidence FOR: spec-on is step37's single biggest serving lever: plain 78.41 ->
  spec 92.13 tok/s median (+17.5%), "byte identity spec-vs-plain under sustained load,
  25/25 zero-fault gates" (`research/step37-decode-headroom-20260828/RESULTS.md`;
  `068cbc4253` body). The whole measurement corpus transfers: spec engagement proven from
  `usage.spec` in the response body (never from a 200), the policy sweep form (K3/H3/P0.5
  won all three stages at 146.62 median over 9 interleaved cells), and the trap ledger
  (prime-program-differs-by-spec: spec-on primes t=1440 vs t=1024 on the same prompt, so
  byte-identity gates need a prompt where both arms prime identically).
- glm5 evidence FOR: the blocking primitive just landed: lane/glm53-batched-decode built
  the mHC `[B, streams, n_embd]` batched walk with its gate (`eb526c161`), cap derived at
  the shexp decode-exact knee B=15 (`cef4e11c3`), B=16 knee measured as the low-order-bit
  cuBLASLt class ("first diff ref=-0.901187 vs -0.90118694", `92b7687a2`). Verify is a
  batched t>1 walk of the same trunk; the refusal wall is now one arm, not the topology.
  glm5 carries 1 NextN layer with its own q/kv projections (`CENSUS.md`: "45 decoder
  layers + 1 MTP (NextN) layer = 46").
- Evidence AGAINST / risks: every spec entry point still `refuse_hyper`s
  (`BRINGUP.md`: "No spec-engagement receipt is possible on this model"). glm5 has ONE
  MTP head where step37 ships three, and step37's own sweep says heads matter: h3-off
  142.66 vs h1-off 126.78 medians (draft-graph corpus row), so the 1-head ceiling is
  materially lower than +17.5%; treat the uplift as unknown until glm5's own
  acceptance-parity cell runs (BRINGUP.md phase 3 names LAW:acceptance-parity-gate).
  Vendor defaults are temp 1.0 / top_p 0.95: the sampled chain must be the
  serving-equivalent device sampler inside the chain (step37 built exactly this,
  `eed70e226`), and the sampled draft-graph exclusion (pure-temp only) applies.
  Acceptance under KDA recurrent state needs its own rewind contract (KDA state rollback
  on rejected tokens has no step37 analog; SWA-ring rewind headroom is the nearest class).
- Expected value: at step37's accept rates a 1-head chain is plausibly +10-15%
  (ARITHMETIC from the h1-vs-h3 spread; NOT a measurement). At a post-lever-1 ~28 ms token
  that is ~3-4 ms.
- Status: blocked on the verify walk; sequence after lever 1 lands or in parallel on a
  second pod.

### 3. `MEMRA_BF16_MMV` ratification, and the precision-door ORDERING step37 paid to learn (decode roofline)

- Mechanism: bf16-resident large 2D tensors + bf16 matvec at decode. Engine-generic door,
  already engaging on glm5.
- step37 evidence: the door is worth a third of the token and its acceptance form is fully
  written (`docs/FLAGS.md:986`): OFF 58.58 vs ON 78.06 median (+33.3%), argmax MATCH at
  logit maxdiff "8.779e-2 to 9.807e-2" vs the OFF arm's "1.965e-3 to 2.699e-3", first-token
  identity at max_tokens=1 on four real prompts, empty completions rejected because the
  model is thinking-class, load-time `[bf16-mmv] RESIDENT` announce making engagement a
  receipt. The named remaining step there: "the 12-boot battery at the full serving env, in
  the 12/12 form MEMRA_STEP_TP_GRAPH and MEMRA_NVFP4_BANK_V2 document".
- glm5 evidence: already measured as A3/A4 arms: roofline 15.9 -> 10.1 ms, greedy 20.348 ->
  23.260 tok/s at 12000 slots, 25.912 at 14000, sha stable across both arms at
  `ca1e3cc2e4ea7104` (ATTRIBUTION.txt). The ROADMAP correction stands: BF16 is NOT on the
  residency critical path (the f32 trunk fits under PP), so this is a pure roofline door.
- What transfers beyond the door: the ORDERING. step37 measured three precision classes and
  their argmax outcomes: reduction-order doors at ~4e-3 (QKV_FUSED) pass trivially;
  bf16-residency at ~9e-2 passes the max_tokens=1 discriminator; weight-precision q8
  mirrors at "9.798e-1" to "2.055e0" need a quality battery, not an argmax line
  (`docs/FLAGS.md:1019`). glm5 numeric doors should be admitted in that order and judged
  by that ladder, and glm5's grouped-prefill owner acceptance already cited "the
  MEMRA_BF16_MMV acceptance class" for its near-tie flip, so the ladder is live precedent.
- Acceptance owed on glm5: run-gen argmax gate on real prompts (exists in class), the
  12-boot battery, owner ratification (numeric-class default flips are the owner's call).
- Expected value: measured, not arithmetic: ~5.8 ms/token of roofline (15.9 -> 10.1).

### 4. Prefill stage overlap under PP, and the join-serialization instrument (prefill to thousands)

- Mechanism: pipeline the prime chunks across PP stages so more than one card computes at a
  time, and PROFILE for serialized joins before believing any overlap claim.
  Architecture-neutral scheduling; the mHC state transit is the glm5-specific part, and it
  is already proven exact cross-device ("18/18 PASS", `ppn-hyper-gate/XDEV-FINDINGS.md`).
- step37 evidence: the 2026-08-29 re-baseline found the grouped prime's rank joins STILL
  serialized at t=4096: "med join 18.70 ms" vs "med span_max 8.50 ms", serial ratio 1.42,
  and at 39.5k tokens "joins total 7.64 s of the 15.31 s TTFT (50%)"; collapsing join to
  span_max is "worth ~4.1 s" (`research/step37-decode-headroom-20260828/RESULTS.md`,
  Task A). The transferable half is the instrument: per-layer `[grp-prof]` join/span
  decomposition plus `MEMRA_PRIME_PROF` phase splits, which is how a "parallel" program is
  caught running serial.
- glm5 evidence: PP3 prefill runs one stage at a time by construction (the BOX-AB serving
  shape), so at 616-639 tok/s the two idle stages are the largest untapped prefill
  resource; ARITHMETIC ceiling ~3x from chunk pipelining alone, and the PREFILL-GAP
  sequencing already estimates "L1+L2+L3 to an estimated 1,500-3,500 tok/s". Overlap
  stacks with L2 (tensor-core trunk, in flight) and L3 (`MEMRA_KDA_CHUNKED`, landed
  default OFF, `e69ed0600`).
- Risks: KDA recurrent state and the DSA kpool are sequential across chunk boundaries
  within a session, so overlap is across stages for one stream, not across chunks of one
  layer; the mHC boundary payload is `[t, 4, hidden]` f32, 4x a serial trunk's transit
  (PREFILL-GAP.md 1.5), so boundary bytes grow with chunk size and want the bulk-transfer
  discipline (see the TP_PREFILL dead end below). Exactness gate = the existing ppn-hyper
  gate class extended to pipelined chunk schedules.
- Status: NOT started. First action is measurement, not code: run
  `prefill-gap-20260829/profile-prime-phases.sh` on the grouped-ON arm and add a per-stage
  busy/idle decomposition; only then size the pipelining arc.

### 5. The warm-turn arc: session affinity at depth, continuation primes, prefix cache (TTFT for the agentic ICP)

- Mechanism: keep long sessions resident (affinity), give continuation/suffix primes the
  fast prime path, re-enable prefix reuse. Engine-generic machinery; glm5-specific
  blockers named.
- step37 evidence:
  - Affinity at 36-40k context: warm follow-up "2.275 / 2.318 / 2.258 s" vs "35.12 / 35.35
    / 35.06 s" without it, ~15x, zero faults, and it was UNMEASURABLE until the ring-aware
    checkpoint restore landed (`c9a617ca99`; `research/step37-30k-cell-20260829/RESULTS.md`).
  - `MEMRA_STEP_GEMM_PRIME_SUFFIX` (`docs/FLAGS.md:554`): continuation primes were the hole
    after fresh primes got the GEMM path; measured "walk continuation 0.2529 s + 5.5978
    ms/suffix-token ... vs this path at 0.99 ms/token", "warm-turn TTFT 0.58 s (door) vs
    7.15 s (walk)", flipped ON only after the 8-turn blind quality A/B and the cache-on
    twin.
- glm5 evidence: every glm5 receipt pins `MEMRA_PREFIX_CACHE_MB=0` (restore defect, its own
  lane); the grouped-prefill flip covered FRESH primes, and whether a `cache.pos > 0`
  continuation chunk rides the grouped arm is an open, unmeasured question (the step37
  lesson is that it silently will not until someone builds and gates it). glm5's ring
  restore precondition is already paid on the SWA side by `c9a617ca99`; the latent-plane
  restore defect is the remaining blocker.
- Expected value: for the buyer profile (agentic multi-turn), this multiplies every cold
  win; step37's own numbers say the warm shape is where 10-15x lives. No cold-decode
  contribution.
- Acceptance: the owner's 8-turn larger-prompt cache-on twin law, per-turn TTFT, plus the
  30k-cell soak-threshold form (counters rewound = illegal = #87 = lap = panics =
  fullprime = 0).

### 6. TP2 over the KDA trunk + MoE: the join-diet playbook, applied late (decode roofline)

The owner's question 1 answered directly: TP2 would NOT beat PP for glm5 decode TODAY, and
TP4 even less, because the token is launch-bound, not bandwidth-bound. TP attacks the 15.9
ms roofline term (parallel weight reads) and does nothing for the 17.2 ms launch term; it
roughly doubles per-layer launches (two rank walks) and adds 2 joins per layer x 45 layers
of boundary latency, the exact costs step37's join-diet campaign existed to pay down. At
B=1 PP with full residency, glm5 pays only N-1 = 2 stage handoffs per token of
`[streams, n_embd]` state; there is no per-layer seam to diet. AFTER levers 1-3 land, the
picture inverts: a ~10.1 ms BF16 roofline becomes the dominant term, and TP2 halving it
(ARITHMETIC ~5 ms, minus join costs) is what closes the last gap to 11.1 ms.

- step37 evidence FOR (when its turn comes): TP2 native P2P is step37's qualified serving
  topology (78.6 eager stable; 92.13 spec). The join-diet doors carry exact receipts:
  `MEMRA_OPROJ_DIRECT` +0.6 (57.1 -> 57.7), `MEMRA_MOE_DIRECT` +1.85 (57.9 -> 59.8),
  `MEMRA_ROUTES_PRESTAGE` +2.5 (59.6 -> 62.0), `MEMRA_SHEXP_OVERLAP` +5.1 (62.2 -> 67.3)
  with its load-bearing placement law "after rank issue (early placement starved dev1,
  -8)" (`docs/FLAGS.md:994-997`). The one-shot PCIe allreduce design is banked with
  community numbers ("6.1-11.8 us for 1-64 KB vs NCCL 13.2-71.2 us", end-to-end "+11.3%",
  `research/step37-nvfp4-tp2-20260820/ENGINE-BASICS-SWEEP.md`). The canonical-chunk
  contract (`MEMRA_STEP_TP`, `docs/FLAGS.md:260`) is the design law for bit-stable
  sharding across rank counts.
- glm5-specific constraints: expert sharding must respect NVFP4 block geometry and the
  per-expert `weight_scale_2` macro (the fused-epilogue fold already solved the macro once;
  reuse its gate's five red arms). The mHC residual means every joined quantity is a
  4-stream state, 4x step37's seam payload. The KDA recurrent state is rank-local only if
  heads shard cleanly (64 heads x 128: TP2/TP4 divide; check the conv and dt/decay planes
  in the census before design, per the asym-split lesson below).
- Which join-diet dead ends bind glm5's topology (all architecture-neutral, hardware-level,
  receipted; do NOT re-run): stream memops (`cuStreamWriteValue32`) rejected on peer memory
  over PCIe and on async-pool allocs, legal form = device-kernel P2P store + LOCAL
  waitValue; rank0-stream merge (-2.7); work migration to the idle device (-5: the idle
  sits BEFORE its input exists); token pipelining under exact greedy (hard argmax->embed
  edge, the +4 probe was an illegal schedule). The patterns that pay (direct join,
  prestage, prejoin-overlap filler, replicated deterministic compute, lazy len mirrors)
  transfer to ANY TP seam glm5 ever opens.

### 7. Serving-admission and measurement craft still owed (both targets, and correctness)

Transfers that are checklists, not kernels. glm5's lanes adopted most of the corpus; these
are the measured gaps:

- The 12-boot battery form ("12-boot 200-token identity battery (12/12)",
  `docs/FLAGS.md:991`; named as the admission step in the BF16_MMV and BANK_V2 rows) has
  not been run for any glm5 numeric door. It is the named remaining step for BF16_MMV
  ratification and should be the standing form for lever-1 doors.
- Short-prompt margin gates. The `MEMRA_NVFP4_BANK_V2` incident (`docs/FLAGS.md:993`): the
  ON arm corrupted generated text at a 25-token prompt while "every gate prompt was >= 613
  tokens" and the soak "metered request success rather than answer quality"; the damage was
  "MARGIN-dependent, not length-dependent". glm5 gate prompt sets must include the
  short/low-margin class, and soaks must sample answer quality.
- The dead-rollback boot refusal pattern. `MEMRA_STEP_GEMM_PRIME` "=0 REFUSED AT SERVER
  BOOT since 2026-08-28" because "A rollback that cannot serve is an outage with a flag
  name" (`docs/FLAGS.md:553`). glm5 analog, decision owed: with grouped prefill default ON,
  the `MEMRA_MOE_GROUPED_PREFILL=0` arm serves 85 tok/s prefill and blows the 90 s
  first-token deadline for cold prompts past ~7.6k tokens (BOX-AB arithmetic). Either the
  serving recipe documents that bound or the server refuses the dead arm at serving ctx.
- The blind pre-registered quality rubric (`research/step37-sampled-quality-20260828`:
  rubric committed before any generation, 72/72 valid rows, per-row engagement receipts) is
  the form any glm5 weight-precision door (a future W8 class) must pass, argmax alone being
  insufficient at that maxdiff class.
- The 30k-cell soak thresholds and instrument fixes (unique port per arm x round,
  pgrep-clear wait, PID-verified boot after health-200) transfer as-is to glm5's 1M-context
  serving admission.

### 8. Cheap measured knobs worth one probe each (decode)

- `MEMRA_RMS_BLOCK=1024` (`docs/FLAGS.md:992`): step37 t=1 norms went "19us -> single-block
  4096-elem class", full stack 42.0 -> 44.0 eager. glm5 runs 45 layers of norms plus mHC
  rowsq/post chains at t=1, same hidden 4096; same latency class. Numeric-class change:
  pinned-value acceptance (fresh-tape identity + battery), one interleaved cell.
- `MEMRA_SWA_RING`'s LESSON, not its code (`docs/FLAGS.md:273`): allocation shaped to the
  attention class freed "16.4 GB" throughput-neutral on step37 and un-blocked every
  VRAM-hungry door (W8, 3-head MTP). glm5's analog audit: only 11 of 45 layers are MLA and
  need KV planes at ctx scale; KDA layers carry fixed-size recurrent state. The ring-sizing
  lane already owns part of this; the transfer is the audit habit: per-layer-class
  capacity math before any "does not fit" verdict (it is also what makes lever-6 mirrors
  or MTP scratch fit later).
- Dual-wave PP scheduling (`research/dualpp1-20260811/PROGRESS.md`: balanced two-wave ticks
  behind `MEMRA_DUAL_PP=1` + `MEMRA_PP_OVERLAP=1`, ">=+15% c>=8 floor" acceptance): NOT a
  B=1 lever, but with the glm5 batched walk landed (cap 15) it is the named mechanism for
  multi-stream throughput on the PP3 shape once the batched hyper walk takes the ppN door.
  Aggregate-throughput class only; keep it off the 90 tok/s single-stream critical path.

---

## Do-not-transfer list (receipted against, with the receipt)

1. Whole-token CUDA graph capture (`MEMRA_STEP_TP_GRAPH`, `docs/FLAGS.md:991`). Re-measured
   on the current step37 stack 2026-08-25: eager 75.99 / 75.98 / 75.84 vs graph 59.39 /
   61.27 / 61.09 tok/s, -20%; "the remainder is dependency LATENCY (405 small serialized
   kernels that cannot fill the device), not launch overhead, so graph replay cannot
   recover it". On glm5 additionally: every graph path `refuse_hyper`s the topology, and
   the per-layer router D2H exists because "the host needs `sel` because it DRIVES
   ADMISSION" (ROADMAP.txt step-4 correction), which no graph can capture. CAVEAT that
   keeps this honest: glm5's launch regime differs from step37's (~3200 launches at ~5.3
   us of true launch latency vs step37's ~405 dependency-bound children), so AFTER lever 1
   plus a device router (the `MEMRA_STEP_TP_DEV_ROUTER` pattern, `docs/FLAGS.md:987`,
   "the prerequisite for whole-token CUDA-graph capture", now structurally possible on the
   full-residency placement) a bounded inner-region capture may be re-evaluated. If it
   ever is, step37's capture discipline transfers whole: alloc-free children, param-update
   retargeting instead of rebuilds, eager fallback on ineligible shapes, the 12-boot
   identity battery. Until that day: do not build graphs on glm5.
2. `MEMRA_NVFP4_BANK_V2` slot-major expert banks (`docs/FLAGS.md:993`). Refused at server
   boot for step37 since 2026-08-29: the ON arm changes generated text and the v1-vs-v2
   reader defect is not yet localized to a kernel. glm5 serves the same
   `qmatvec_nvfp4_dp4a` kernel family; do not adopt the v2 layout anywhere until the
   device-side bank oracle lane lands.
3. `MEMRA_STEP_TP_W8` as a straight lift (`docs/FLAGS.md:1019`). Three reasons in its own
   receipts: weight-precision class "9.798e-1" / "1.570e0" / "2.055e0" logit maxdiff
   (~200x the fused-kernel class) needs a quality battery step37 never ran; the first
   wiring was SLOWER (79.52 vs 80.72: three launches + activation quantize beat the halved
   bytes) until the single-launch fused kernel; and the mirrors cost "~0.9 GB per card"
   plus "roughly 1.7 GB" more, which "does NOT fit beside a 262144-token cache
   reservation". On glm5 the same VRAM competes with expert residency, the single most
   valuable resident bytes on the box (staging costs 53 GB/s pinned). If a glm5 W8 class
   is ever opened: fused single-launch first, quality battery mandatory, VRAM accounted
   against residency, and only after BF16_MMV is ratified (precision-door ordering,
   lever 3).
4. Whole-expert EP at B=1 (`MEMRA_STEP_NVFP4_EP2`, `docs/FLAGS.md:1040`). "RECEIPTED
   NEGATIVE at B=1 (-4.5%: 72.57/72.59/72.60 vs 75.97/76.03/76.06)": top-8 over 2 owners
   is binomially imbalanced ("expected critical path ~5.1 full experts vs TP's
   always-balanced 4"). glm5 is also top-8 over 288: the same math forbids whole-expert
   ownership at B=1 on any glm5 multi-card expert layout; shard projections instead.
5. Asymmetric TP splits. Killed on step37 by census + math (GQA n_head_kv=8 makes the
   smallest step 37.5/62.5 against a needed ~56/44, and MoE asymmetry breaks the NVFP4
   canonical geometry; join-diet corpus). The transferable law: run the head/expert/block
   granularity census BEFORE designing any split, and expect glm5's KDA 64-head / NVFP4
   16-block geometry to dictate, not permit.
6. Token-row P2P transport at prefill (`MEMRA_STEP_TP_PREFILL` high-context NO-GO,
   `docs/FLAGS.md:265`): "about 61,440 peer copies ... per 4K attention layer", TTFT
   "113.252s" at 16K vs vLLM "1.099s". Any cross-device glm5 prefill (lever 4's pipelining
   included) uses dense bulk transfers (`MEMRA_STEP_TP_BULK_P2P` class receipt: TTFT
   "113.252s to 23.487s", peer copies "61,452 -> about 21"), never per-token-row copies.
7. The join-diet micro dead ends (lever 6 list): stream memops on PCIe peer memory,
   rank0-stream merge, work migration to the idle device, token pipelining under exact
   greedy. All receipted negative on the exact card class glm5 serves on.
8. `MEMRA_MOE_CSR_NVFP4`'s cached form: "remains rejected by batch-composition identity
   gates; never enable in serving" (FLAGS row). glm5's grouped prefill already took the
   correct (sort + grouped GEMM) shape instead.

## The six-fix arc, read as glm5's failure-class forecast

From the `068cbc4253` squash body, each fix names a class glm5 will meet:

1. "NVFP4_V2 shared-memory overrun (KQ_CB_WORDS)": shared-array-sized-by-macro vs stager
   mismatch in templated CUDA kernels. glm5 mints new NVFP4 kernel twins (epilogue,
   grouped prefill, future lever-1 kernels): audit smem sizing per new qtype/shape.
2. "SWA-ring MTP lap (rewind headroom)": ring rewind must reserve headroom for
   speculative rewinds. glm5's DSA index ring plus any future NextN spec inherits this
   class; glm5 already paid the sibling class once (`c9a617ca99`, restore "copy the
   window, not the absolute length").
3. "solo prefill widening (ckpt_at is a boundary, not carried state)": prime-chunk
   boundary state treated as carried state. glm5's 1m-context receipts already banked a
   chunk-order red (`1m-context-20260828/03-red-mutation-chunk-order.txt`); every new
   prime schedule (lever 4) re-gates this.
4. "spec row-table stale-pointer memo": a process-lifetime pointer-keyed cache with no
   invalidation on KV drop; presented as ILLEGAL/#87 before affinity worked and as SILENT
   WRONG OUTPUT after. glm5 has pointer-keyed caches of the same shape (the
   `matvec_bf16_via_q8_mirror` mirror cache is built on first decode use, keyed by
   pointer): audit invalidation before affinity + reuse land on glm5.
5. "dead session affinity reuse (TP KV mirror predates the checkpoint)": generation checks
   between mirrors and checkpoints. glm5's affinity bring-up (lever 5) gates this
   explicitly.
6. "request-absolute seq_end in the batched prime (SWA arm selection)": absolute vs
   window-relative position confusion. glm5's ring/kpool offset math is the same class;
   the batched walk and any pipelined prime re-gate it.

Plus the two flanking lessons: `MEMRA_STEP_GEMM_PRIME=0` boot refusal (dead rollbacks fail
loudly at load, lever 7) and the spec K=0 floor fix (`a49c363ea`: a request-shape edge
panicked the GPU worker; engine panics are fleet-fatal, so glm5 spec bring-up guards its
input domains first).

## The decode arithmetic to 90, stated honestly

All ARITHMETIC except where receipted; short-prompt shape (the 4.6k+ serving shape carries
an additional depth term that must be measured per lever):

| step | ms/token | tok/s | basis |
|---|---|---|---|
| full residency PP (measured) | 33.39 | 29.95 | ROADMAP step 3 |
| + fused epilogue (measured, flag OFF pending flip receipts) | 33.23 | 30.10 | lane/glm53-epilogue A/B (+15.3% on its own host baseline) |
| + BF16_MMV ratified (roofline 15.9 -> 10.1, measured arms) | ~27.5 | ~36 | ATTRIBUTION A3 intercept 26.70 |
| + lever 1 launch diet (~-4.5 of the launch term) | ~23 | ~43 | ARITHMETIC at ~5.3 us/launch |
| + lever 2 spec, 1 head | ~20-21 | ~48-50 | ARITHMETIC, unmeasured on glm5 |
| + lever 6 TP2 roofline split | ~15-16 | ~62-66 | ARITHMETIC, minus join costs |

The stack does NOT reach 90 on paper. That is the map's most important honest finding: 90
tok/s single-stream needs either a deeper launch collapse than the QKV_FUSED class
(epilogue-degree fusion on KDA/mHC, or the re-evaluated bounded graph after the host syncs
are gone), a multi-head draft (the NextN chain is 1 layer; step37's h3-vs-h1 gap was ~16
tok/s), or acceptance of an aggregate-throughput framing (the batched walk + dual-wave
already point there). Sizing that choice is exactly what lever 1's census cell buys first.

## What to start tomorrow

Lever 1, and its first cell is measurement: an nsys decode launch census on the residency
config with the fused epilogue ON (the decode twin of `profile-prime-phases.sh`; step37's
`launch_econ.rs` and `decode-kernel-census` instruments transfer), bucketing the ~1140
KDA+mHC launches by chain and sizing the fusion boundaries. It attacks the measured 51%
term, needs no owner numeric ratification to begin, collides with no sibling lane
(epilogue owns MoE, tc-trunk owns prefill GEMMs, batched-decode owns width), pays into
both targets, and its census simultaneously answers the graph question (launch latency vs
dependency latency) with data instead of analogy. Hardware time: one bench-box window for
the census, one for the first fused-door A/B + battery.
