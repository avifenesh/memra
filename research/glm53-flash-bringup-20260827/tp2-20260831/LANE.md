# glm5 TP-2 lane (lane/glm5-tp2, 2026-08-31)

Owner bar: **GLM-5.3-Flash does not serve under 100 tok/s single-stream in any scenario**
(LAW:glm53-100toks-serving-bar). Plain is 35.41 tok/s on the 3-card PP shape; the deployed
BF16_MMV roofline is 99 tok/s on one card; TP-2 streams the token from two cards' HBM
(~197 tok/s roofline pre-haircut, `decode-gap` ATTRIBUTION.md §2). This lane builds TP-2
for glm5_next, correctness-first, on the receipts that already exist: the pro6000-multigpu
transport qualification, the MEMRA_STEP_TP house pattern, and the step37 tp2-join-diet
playbook. TP-3 is architecturally disfavored (64 attn heads, 32 indexer heads: neither
divides by 3); the box topology law is TP-2 inside a NUMA pair, PP-2 across pairs, never
TP-3/4 on this host class (docs/decisions/PRO6000-MULTICARD.md).

Base: `origin/lane/glm53-flash-bringup` @ 34e0c0bf2. Worktree `~/projects/wt-glm5-tp2`.

## Stage 1 — transport-fix port (DONE, gates below)

The hy3 PP-4 qualification (darklanes `lane/hy3-pp4-qual-20260830`,
`research/hy3-pp4-qualification-20260830/README.md`) proved the ORIGINAL peer-probe
diagnostic produced intermittent false-red failures (10/50 processes across the ladder)
while every production-slot probe passed, and closed it with the live-transport reprobe:
50/50 fresh PP-4 processes, 200/200 runtime probes, zero byte mismatch on the exact
RTX PRO 6000 card class. That chain is NOT on the glm5 line; this stage ports it.

Cherry-picked onto 34e0c0bf2, clean (no conflicts; `peer_probe_copy`,
`run_production_peer_probe`, and the runtime-probe scheduler predate the multigpu branch's
wave scheduler and are shared ancestry):

| lane/glm5-tp2 SHA | upstream SHA | subject |
|---|---|---|
| 18cfaf774 | 7682dc0a0 | test(pp): drain due ticks before runtime peer probes |
| 610425804 | 991178caf | fix(pp): publish runtime probes to receiver stream |
| 57a6b16d8 | 20d4ec705 | fix(pp): reprobe the live boundary transport |

(991178caf also exists as cherry-pick 175d7e8e9 on the hy3 combined line
`lane/hy3-pro6000-serving-20260830`; content identical.)

What the fixes are: (1) `peer_probe_copy` publishes the peer write to the receiving
context through a TX-stream event the RX stream waits on — the live `BoundarySlot`
ordering contract — instead of a host-side TX-only synchronize that reads the destination
context unordered; (2) the RUNTIME peer probe re-routes through the production-slot path
(`run_production_peer_probe_widths`) instead of the legacy allocation/readback pass that
produced the false reds.

NOT ported (assessed): `a3e60fcdc` "experimental 2-4 GPU serving" (the `MEMRA_PP_WAVE`
PP-3/PP-4 wave scheduler, multi-card placement advisor, decode-batch pp mode). TP-2 and
TP-2+PP-2 do not need the wave scheduler (PP-2 keeps its independently qualified
dual-active default per PRO6000-MULTICARD.md), and the commit sits on v0.121.0 +
the business-tier extraction — a ~40-commit stack the glm5 bringup line has not merged.
Bringing it across is a consolidation-lane decision, not a TP-2 prerequisite. If the box
arm later composes TP-2 with PP-3+, the wave scheduler port re-opens.

### Stage 1 gates (rig, exactness only, flock /tmp/memra-5090.lock)

| gate | invocation | result |
|---|---|---|
| clippy | `cargo clippy --workspace --all-targets` @ 57a6b16d8 | zero warnings, exit 0 |
| spec-ppn matrix (stages 2+3) | `bash ppn-verify-20260830/run-spec-ppn-gate.sh` (OUT=`tp2-20260831/stage1-gates`, release build @ 57a6b16d8, flock, TF32 off) | **ALL ARMS PASS** — 8 arms (n2 even/split1/split3/streams0/overlap0, n3 even/asym/streams0), logs `stage1-gates/1[0-8]-n[23]-*.log`, every arm exit=0 incl. R1/R2/R3 reds biting |
| hyper-ppn matrix | `bash ppn-hyper-gate/run-ppn-hyper-gate.sh` (same OUT/build) | **ALL ARMS PASS** — 10 arms (n2 x6 incl. shard0 + longer, n3 asym, n4 even/streams0), every arm exit=0 |
| local-ci | `tools/local-ci.sh --perf` @ 3be882e66 | correctness stage green (kernel-check, run-gen argmax, run-spec K=1..8, gemma stream, VERIFY-GATE, serve-stress, spec-on-cache-hit qwen); perf stage 0 fail 0 warn (qwen9b 138.91 tok/s; 31b/26b/e4b cells SKIP, models absent on rig); row appended to perf-ci.jsonl |

## Stage 2 — shard map (the review artifact)

See `SHARD-MAP.md` in this directory: per tensor class the TP-2 axis / block legality /
replicate-or-owner decision, the EP-2-vs-top-8 arithmetic (peer touched on ~99.3% of
layer-tokens; slowest-rank expected 64% of expert bytes, ~1.57x not 2x), the per-rank
byte budget, and the reuse-vs-build port map against tp.rs/parallel.rs.

**Flag choice (stated per the brief): a glm5 twin door, `MEMRA_GLM5_TP`, on the shared
tp.rs machinery — not an extension of `MEMRA_STEP_TP`'s surface.** The grammar
(`LAYER[-LAYER]@DEVICE,DEVICE[;...]`, `parse_step_layer_specs` with its flag name
already parameterized), the preflight shape, the snapshot-once registry, the rank
runtime + P2P ladder, the NVFP4 shard validators, and the EP route/combine contracts
all REUSE. What cannot be shared is the model contract itself:
`ModelParallelContract::from_plan` is a hard is_step37 pin that refuses non-
SlidingGatedMoe plans (parallel.rs:177-292) — glm5 compiles to a HyperConnections
program — and the step surface's divisibility law is FP8-128x128 while glm5's is
NVFP4-64/head-128/EP-whole-expert. Overloading the step flag would run glm5 through
step-worded refusal strings and step-named engagement markers, breaking the receipts
discipline (every marker names its seam), and the step FLAGS row's semantics (SWA ring,
GQA shards, official TP4 trunk) are all false statements about glm5. Twin flag, shared
skeleton, separate FLAGS row with its own defaults and receipts.

## Stage 3 — TP-2 execution (BUILT, gated in stage 4)

The `MEMRA_GLM5_TP` seam, implemented per the shard map. What was built vs reused:

| piece | disposition |
|---|---|
| `crates/memra-engine/src/glm5_tp.rs` | NEW: flag/parse (grammar REUSED via `parse_layer_specs_for_trunk` — the step parser generalized with the trunk as a parameter, step wrappers unchanged), fail-closed preflight (`prepare_glm5_tp_load`: non-glm5 plans, heads/experts not divisible, KDA head_dim != 128, co-armed PP>1 / step doors, one device pair, owner-first), `Glm5TpRt` (peer Engine; gate-only same-device constructor), shard builders (`shard_kda_layer`, `shard_mla_layer` — load full layer through the unchanged loader laws, slice rows on the outermost axis, host-bounce to the peer, drop the full copy; per-layer transient VRAM = one layer), `arm_moe_ep` (contiguous expert halves as per-rank slabs), the KDA TP walk (`kda_tp_cached`: per-rank `kda_core_gated` + gather + column-`wo` + concat), gate-red knobs |
| `kda.rs` | `kda_core` split at the wo boundary (`kda_core_gated` returns the gated mixer output; the plain wrapper is byte-for-byte the pre-split body); `KdaAttnLayer.tp` sidecar field; fail-closed refusals at the `kda_core` choke point (covers every plain entry) + `kda_scan_replay` |
| `hybrid.rs` | `MlaAttnLayer.tp` + `MoeWeights.glm5_ep` fields; loader arming inside the layer loop (preflight BEFORE the loop, per-layer shard+replace after each push) |
| `hybrid_forward.rs` | `mla_attn_core` split pre-wo + `mla_attn_cached_pre_wo` (one plumbing copy for both arms); `mla_tp_attn_cached` (replicated wq_a/wkv_a/indexer/kpool on both ranks, per-rank latent replicas — peer plane geometry-cloned lazily, `ensure_mla_peer_latent`); `moe_ffn_glm5_ep` (root router via the unchanged `moe_route_sigmoid_cfg`, per-slot sequential expert program on the owning rank, slot-ordered `axpy` combine on root — the plain fmaf chain reproduced operation for operation); `moe_shexp_add` extracted verbatim so both walks share ONE shared-expert body; TP branches in `hyper_range_decode`, `hyper_range_prime`, AND `prime_chunk_hyper` (the single-engine chunk walk — missed on the first pass, caught by the gate's prime refusal); stateless `mla_attn` refusal |
| `memra-kv` | `Cache.glm5_tp_recur` (per-layer `[root, peer]` shard-geometry conv/ssm planes, engine-hydrated lazily) + `Cache.glm5_tp_latent_peer` (peer latent replica); `snapshot`/`snapshot_into` REFUSE while TP state is live (per-rank planes are outside CacheSnapshot); the PP stage-split cache moves the slots like every other class |
| `memra-server/worker.rs` | spawn-time refusal of `MEMRA_GLM5_TP` (serving wiring — per-session TP admission/rollback — is the named box increment) |
| `glm_spec.rs` | `glm5_spec_session_new` co-refuses while the door is armed |
| `docs/FLAGS.md` | `MEMRA_GLM5_TP` row + the two gate-harness knob rows, same PR |

v1 transport is host-canonical staging (the step seam's correctness transport). NOT built,
deliberately: native P2P engagement (machinery inherited: `configure_native_p2p` ladder),
join-diet doors, lm_head vocab split (pure-perf, zero-arithmetic-risk, box arc), batched
decode TP (refused), spec composition (co-refused), serving wiring (worker refuses).

## Stage 4 — gates (GREEN: `glm5-tp-gate` ALL ARMS PASS)

`glm5-tp-gate` (fixture: the ppN gates' mini glm5_next with 2 KDA heads x 128 + 2 MLA
heads + 4 experts top-2; KDA projections Q8_0-encoded — as F32 they ride cuBLAS whose
K-reduction split is SHAPE-dependent, the measured 1-ulp class; Q8_0 rides the per-row
MMVQ class that matches the real serving kernels' shape). Invocation:
`run-tp-gate.sh` (P=16 N=12, flock, TF32 off); log `stage34-gates/01-tp-gate-p16-n12.log`.

**The two-regime bar, and why.** DECODE (t=1, this lane's product surface): BYTE IDENTITY
vs the door-OFF plain walk — achieved on every arm (28 t=1 steps, full logits + tape).
This is real because the program is column-parallel-over-gather: every arithmetic op runs
the same per-row kernel over the same values; the one cross-rank arithmetic site (EP
slot-ordered axpy) reproduces the plain fmaf chain exactly. PRIME (t>=2): batched GEMM
widths select shape-dependent K-reduction splits (cuBLASLt f32/f16, MMQ tiles), so a
128-row shard legally differs by ulps from its 256-row full tensor — the documented
`Engine::linear` m-dependence class (FLAGS MEMRA_PRIME_CHUNK precedent). Bar: calibrated
band 2e-4 (10x margin over the measured green worst 4.85e-5) + tape identity + repetition
byte-identity; reds must land orders above.

| arm | verdict |
|---|---|
| B tp-all (whole trunk: KDA+MLA shards, EP x3) | decode BYTE-IDENTICAL; prime max_rel 3.6e-5, tape identical |
| C tp-kda-only / C0 layer-0 / C2 layer-2 | decode BYTE-IDENTICAL; prime 2.8e-5..4.9e-5 in band |
| D tp-mla-only (the NEW verification surface: cross-rank absorbed decode + replicated kpool selection) | decode BYTE-IDENTICAL; prime BYTE-IDENTICAL (max_rel 0.0) |
| self-consistency (all arms x2 repetitions) | bit-identical |
| E stateless-poison / F spec-co-refusal / G PP-composition-refusal | all refuse BY NAME |
| R1 swapped shard (wo halves crossed) | bites: max_rel 1.4e2, tape diverges |
| R2 wrong expert weights (root gate/up swapped) | bites: 2.7e2 |
| R3 peer-combine dropped (non-vacuity: peer contributes) | bites: 6.7e1; EP peer-slot dispatches 75-225 counted per arm (a first seed-search run found a token stream that NEVER routed peer experts — the gate now proves engagement before claiming identity) |

Regression: spec-ppn (stages 2+3) + hyper-ppn matrices re-run green on the stage-3 tree
(`stage34-gates/regress/`), unit tests green, clippy zero, local-ci --perf (see log row).

## Stage 5 — box arm (NAMED, NOT RUN)

The rig gate proves the shard/join program; what it cannot prove is named for the box
(4x RTX PRO 6000 Blackwell, two same-NUMA pairs — the hy3 qualification host class):

1. **Real-artifact class gate**: the fixture classes are F32/Q8_0; the artifact serves
   BF16-resident KDA (custom matvec), NVFP4 MLA/experts (MMVQ + fused epilogue + macros),
   f32 b_proj/f_a/g_a. Same gate shape (decode byte identity + prime band) on the real
   checkpoint, TP-2 within the NUMA pair (`all@0,1`, REAL devices — the same-device knob
   never leaves the rig). NOTE for the arm design: sharded-vs-plain f32 cuBLAS sites
   (b_proj [64,4096] -> [32,4096]) may put even DECODE into the near-tie class on the real
   geometry — measure first, then set the bar from the measurement (the calibration law).
2. **Native P2P engagement**: inherit `configure_native_p2p` (16KiB..64MiB ladder), then
   the transport A/B vs host-canonical (byte-identity gate between transports).
3. **Join-diet ladder**: direct join -> prestage -> prejoin overlap (shexp filler), each
   step identity-gated vs the banked tape + interleaved x3 (the step37 43.1->69.7 recipe;
   placement law: the hook goes AFTER the host's rank-issue walk).
4. **TP-2 + PP-2-across-pairs composition**: lift the `MEMRA_PP_STAGES>1` refusal behind
   its own gate (TP pair 0-1 as stage 0, pair 2-3 as stage 1); requires the wave-scheduler
   port decision (a3e60fcdc) to be revisited if PP-3+ is ever composed.
5. **The 100-bar re-price battery**: interleaved x5 fresh boots, plain-vs-TP2 on the
   serving card class, vendor-default sampled twin + 8-turn cache-on twin per the
   multi-turn law, spreads reported. Priced against the decode-gap table (TP-2 expected
   ~42-43 tok/s pre-diet, ~60 post-diet, 100+ only with the diet+spec levers stacked) —
   the box arm PRICES the lever, it does not promise the bar.
6. **Serving wiring**: per-session TP state admission accounting (per-rank planes are
   unpriced today), rewind/retry seams, then the worker refusal lifts.

Box time is coordinated through the lane owner; the rig never takes timing numbers
(LAW:rig-exactness-only).

## Merge-forward onto the bringup head (lane/glm5-tp2-fwd, 2026-08-31)

The bringup line took two lanes after this lane's base (34e0c0bf2): verify-batch
(3f4accf13, batched verify walk) and decode-diet (536eb510c, four doors). The TP-2 seam
merged forward onto 1c7285e3e as a REAL semantic merge; branch `lane/glm5-tp2-fwd`,
receipts in `fwd-merge-gates/`.

### Resolution per file

- `kda.rs`: `kda_core_gated` carries the CURRENT core (verify-batch `KdaStash`/rows arm,
  `scan_clock` trace instrument, `MEMRA_KDA_FUSED_PROJ` door) and returns the gated
  mixer output; the `kda_core` wrapper keeps the TP fail-closed choke point (now
  covering the batched rows entry too) AND the rows-exact `wo` routing. The wo dispatch
  moved with the split; its routing did not change.
- `hybrid_forward.rs`: `mla_attn_core_pre_wo` threads `rows_exact` (the `mm` closure,
  kpool select, the TC-prefill `!rows_exact` guard); `mla_attn_cached_inner` is the one
  choke point covering the plain and rows entries and applies the rows-exact `wo` after
  the pre-wo core; `mla_tp_attn_cached` calls `pre_wo` with `rows_exact=false` on both
  ranks (the TP walk serves prime/decode only).
- `glm5_tp.rs`: `kda_tp_cached` passes `KdaStash::None` and no scan clock;
  `prepare_glm5_tp_load` refuses the four decode-diet doors by name before any TP CUDA
  state (`GLM5_TP_REFUSED_DOOR_FLAGS` + `refuse_glm5_tp_door_composition`, unit-tested
  env-mutation-free).
- `docs/FLAGS.md`: both sides' rows kept; the `MEMRA_GLM5_TP` row carries the
  composition matrix. No door default changed in this merge.
- `perf-ci.jsonl`: union, both sides kept in ts order, every line validated as JSON.

### Composition matrix (v1: refuse every unproven pair by name)

| pair | verdict | seam | receipt |
|---|---|---|---|
| TP x MEMRA_GLM5_VERIFY_BATCH | REFUSED via spec co-refusal (the batched walk exists only inside glm5 spec sessions) + mixer choke points | `glm5_spec_session_new`, `kda_core`, `mla_attn_cached_inner` | tp-gate arm F; choke refusals arm E |
| TP x MEMRA_HC_FUSED_PRE | REFUSED at load preflight | `prepare_glm5_tp_load` | `red-door-MEMRA_HC_FUSED_PRE.log` exit=1, named |
| TP x MEMRA_HC_DECODE_WS | REFUSED at load preflight | same | `red-door-MEMRA_HC_DECODE_WS.log` exit=1, named |
| TP x MEMRA_KDA_FUSED_PROJ (either arm) | REFUSED at load preflight | same | `red-door-MEMRA_KDA_FUSED_PROJ.log` exit=1, named |
| TP x MEMRA_MLA_DECODE_SPLIT | REFUSED at load preflight | same | `red-door-MEMRA_MLA_DECODE_SPLIT.log` exit=1, named |
| TP x PP>1 / spec doors / step doors | REFUSED (pre-existing, unchanged) | preflight + session | tp-gate arms F, G |

No pair composes in v1; each unlocks only with its own composition gate. The refusal
unit test (`every_refused_door_composition_bites_by_name`) holds the matrix; the four
red-door logs prove the live path end to end (refusal fires at preflight, before CUDA).

### Gate table (merged tree d8107a4d9-line, rig 5090, flock, TF32 off, exactness only)

| gate | result |
|---|---|
| `glm5-tp-gate` P=16 N=12 | ALL ARMS PASS: decode BYTE-IDENTICAL on B/C/C0/C2/D (28 t=1 steps, logits + tape); prime band max_rel 2.800e-5..4.850e-5 (band 2e-4), D byte-identical even at prime; E/F/G refuse by name; R1 1.429e2 / R2 2.705e2 / R3 6.742e1 all bite; EP peer-slot dispatches 69..231 per arm. Verdicts IDENTICAL to the seam-commit run |
| diet gates | `hc_fused_pre_gpu` 4/4, `hc_decode_ws_gpu` 2/2, `kda_fused_proj_bf16_gpu` 5/5, `kda_fused_proj_gpu` (q8 re-bite) 5/5, `mla_decode_split_gpu` 3/3 |
| verify-batch gates | `glm5_verify_batch_gpu` 3/3, `glm5_tparallel_verify_gpu` 9/9 |
| suites | `glm5_spec_session_gpu` 9/9, `glm5_dflash_session_gpu` 10/10, `glm5_mtp_head_gpu` 5/5, `hyper_connections_gpu` 6/6, `glm5_kpool_indexer_gpu` 14/14 |
| ppn matrices | spec-ppn 8/8 arms PASS, hyper-ppn 10/10 arms PASS (`fwd-merge-gates/ppn/`) |
| server suite | 481/481 |
| clippy / fmt / check-flags | zero warnings / clean / 723 reads covered, none uncovered |
| `tools/local-ci.sh --perf` | see `fwd-merge-gates/local-ci.log` and the perf-ci row |
