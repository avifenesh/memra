# glm5 CO-ACTIVATION EXPERT PLACEMENT (lane/glm5-ep-place, 2026-08-31)

Owner directive, now law (darklanes `agent-knowledge/gpu/kernel-craft.md`
`LAW:coactivation-expert-placement`): expert placement for TP/EP is MEASURED, never
even-split — (1) measure per-layer expert co-activation on real traffic, (2) bundle
co-active experts per card under VRAM balance, (3) pin the always-active set (the SHARED
expert plus top-frequency routed experts) to a KNOWN card the token visits FIRST and
leaves immediately. Target: the TP-2 lane's naive EP-2 arithmetic (SHARD-MAP.md §3) —
peer-touch on ~99.3% of layer-tokens, slowest rank ~64% of expert bytes, effective
~1.57x not 2x. Co-activation bundles are the lever that turns layer-tokens single-rank
and closes toward 2x.

Base: `origin/lane/glm53-flash-bringup` @ 94e6e4872 (the matvec merge — carries the
full TP-2 machinery: `glm5_tp.rs` shard builders, SHARD-MAP.md, the EP walk with the
slot-ordered fmaf combine). Worktree `~/projects/wt-glm5-ep-place`, branch
`lane/glm5-ep-place`.

## 0. Fleet reconciliation (coordinator, mid-lane 2026-08-31) — division of labor

Two forks were averted while this lane was mid-build:

- **The trace tap already exists**: `MEMRA_MOE_TRACE` / `MEMRA_MOE_WEIGHT_TRACE`
  (`trace_moe_routes`, hybrid_forward.rs) log `<layer> <t> <ids>` /
  `<layer> <t> <id:weight,...>` from the host route readbacks. This lane's own JSONL
  tap (`MEMRA_MOE_SEL_TRACE`) was built, gated green, then DROPPED in favor of the
  fleet tap. What survived of the measurement leg: (a) the glm5 **EP walk is now wired
  into `trace_moe_routes`** (it was the one host-routed MoE walk missing the call — a
  blind spot for any co-activation mint on a TP-armed boot); (b) the **device-routed
  step TP walk now REFUSES an armed tap by name** (its selection never returns to
  host; a trace silently missing whole layers poisons a placement mint — this
  protects the hy3 EP-4 re-cell the coordinator named); (c) the gate's **arm T**
  (trace-ON byte identity + exact row counting) now bars `MEMRA_MOE_WEIGHT_TRACE` on
  the glm5 walks.
- **The partition tool + format land shared**: `tools/build_expert_placement_map.py`
  (stdlib-only; strategies coactivation/frequency/even; selftest 10/10 with a proven
  red arm), frozen JSON format `memra-ep-map-v1` (spec + example receipts:
  `research/ep-placement-map-20260831/REPORT.md`), merged to origin/main @ 4e46be545
  mid-lane. This lane's own Rust partitioner (written before the coordination
  message, gated green, fixture-demo'd) is DROPPED; anything it does better goes to
  the shared tool as a follow-up PR, not a parallel tool. **Base choice, noted per
  the coordinator**: origin/main and the bringup line were mid-merge by another
  agent, so this branch stays based on the bringup head and CHERRY-PICKS the tool
  lane's three commits (6f132b3c2 tool, 365f76ff0 selftest receipt, 6ded915b0
  spec+examples -> 915df846e/dc127ad9c/6787e261e here) — the tool, spec and example
  maps are in-tree, and the reader's unit test is byte-anchored on the committed
  example emission.

This lane's retained scope: the **engine consumption leg** (`MEMRA_GLM5_EP_MAP`
reading `memra-ep-map-v1` in the glm5 shard builders, fail-closed), the **skewed-map
identity gates**, the glm5 wiring + fail-closed hardening of the fleet taps, and the
**box measurement plan** (fleet-first co-activation cell: measured NOWHERE yet;
hy3's EP-4 re-cells with the shared tool once this pattern holds).

## 1. Measurement leg — the fleet tap, glm5-complete and fail-closed

`MEMRA_MOE_WEIGHT_TRACE=<path>`: one row per (MoE layer, forward),
`<layer> <t> <expert:weight,...>` with t*k token-major pairs. glm5's serving walks
already read every router selection host-side (the 42 per-layer `(sel[8], w[8])`
readbacks/token: sequential decode, batched-verify, hyper-batch walks) — the tap rides
exactly that seam, ZERO new device syncs. Rows carry `t`, so one trace separates
decode (t=1) / spec-verify (2..15) / prime (>=16) events, and token events per layer
are chunking-invariant.

This lane's three changes to the tap's world (no format change):

| change | why |
|---|---|
| glm5 EP walk calls `trace_moe_routes` | the one host-routed MoE walk that skipped the taps; a TP-armed trace boot would have silently missed every EP layer |
| device-routed step TP walk refuses an armed tap by name | selection stays device-side there; tracing would need a NEW sync, and a trace missing whole layers poisons placement mints (hy3's EP-4 re-cell rides this) |
| `glm5-tp-gate` arm T | trace ON is BYTE-IDENTICAL to the tap-cold plain walk in both regimes, and rows are COUNTED exactly (56 token events per MoE layer = 2 walks x (P+N); 123 rows shape-checked) |

FLAGS.md: the `MEMRA_MOE_WEIGHT_TRACE` row now states both consumers (m-invariance
razor + co-activation measurement), the EP-walk coverage, the refusal, and the arm-T
receipts.

## 2. Consumption leg — `MEMRA_GLM5_EP_MAP=<file>` (the engine reader)

Reader: `crates/memra-engine/src/ep_map.rs` — the frozen `memra-ep-map-v1` JSON
(load-bearing fields `format`/`ranks`/`entry_rank`/`expert_count`/
`layers[].{layer,assignment}`; the mint's self-receipting `stats`/`traces`/`params`
ride along uninspected). Parsed with the house minimal JSON reader
(`memra_gguf::config::JsonObj`, `raw` widened to pub) — no serde dependency; a
string-aware splitter walks the layers array so paths/shas in the self-receipt can
never desynchronize it. Unit-tested: render/parse round-trip, every refusal
spelling, the layer-cover law, and a BYTE-ANCHOR on the tool's committed example
emission (`parses_the_committed_example_map_bytes` include_str!s
`example-map-coactivation.json` — the reader is proven against the real artifact
bytes, not a hand-typed imitation).

**The entry-rank law**: the format carries `entry_rank` (the law's first-hop card).
The glm5 TP-2 loader REQUIRES `entry_rank == 0` — root is the first hop by
construction (router, slot-ordered combine, shared expert all live there); a map
minted with any other entry rank refuses by name ("re-mint with --entry-rank 0")
rather than silently remapping rank ids.

`arm_moe_ep` generalized: ownership = the validated map row when armed, else
`EpMap::even_owners` — the even split expressed as the degenerate map. Per-rank slabs
pack owned experts ascending (the even split packs as the contiguous half: same bytes,
same single-slice upload — byte-unchanged by construction, re-proven by the full gate
re-run being verdict-identical). The EP walk indexes through `owner_of`/`local_of`
(`Glm5EpExps`); contiguous runs upload directly, scattered rows stage once at load.

**Correctness is placement-independent by construction**: ownership selects WHICH rank
runs the identical per-expert dot program over identical host-canonically fanned-out
bytes; the combine stays slot-ordered on root either way. The map changes bytes MOVED,
never bytes COMPUTED — a PERF artifact, gated for identity here and priced on the box.

Fail-closed, at the load preflight BEFORE any TP CUDA state: unreadable/malformed
file, ranks != 2, expert count != the model's bank, a layer row missing from or
foreign to the EP-armed set, an empty rank, a set-but-empty flag — all refuse by name.
The map with `MEMRA_GLM5_TP` COLD refuses at load on glm5-class (KDA-carrying) plans:
a placement that silently reverts to the even split is the trap (check scoped so
co-loaded non-glm5 models never trip it — the MEMRA_FRSPEC_TRIM lesson). The armed
load prints the map sha256 in `[glm5-tp-preflight] ep-map armed`.

FLAGS.md rows in this PR: `MEMRA_GLM5_EP_MAP` (ABSENT by design — ships only with box
A/B receipts), the `corrupt-ep-map` red-arm spelling, the WEIGHT_TRACE row update.

## 3. Gate table (rig 5090, flock /tmp/memra-5090.lock, NVIDIA_TF32_OVERRIDE=0,
exactness only, 2026-08-31; log `gates/01-tp-gate-p16-n12.log`)

`glm5-tp-gate` P=16 N=12 @ release build of this head — **ALL ARMS PASS**:

| arm | verdict |
|---|---|
| B/C/C0/C2/D (even split, pre-lane arms) | decode BYTE-IDENTICAL (28 t=1 steps, logits+tape, x2 repetitions); prime band 2.8e-5..4.85e-5 (band 2e-4), D byte-identical at prime — verdicts IDENTICAL to the pre-lane run (the map leg left the even split byte-unchanged) |
| **M map-skew** (`MEMRA_GLM5_EP_MAP`, a real memra-ep-map-v1 JSON with all-but-one expert on rank 0; rank-1 singleton = expert 3, provably routed via the probe walks' moesd union) | decode BYTE-IDENTICAL to plain; prime max_rel 3.637e-5 in band — IDENTICAL numbers to arm B: placement independence proven, not assumed. EP peer-slot dispatches 132 (non-vacuous); `[glm5-tp-preflight] ep-map armed` prints the map sha256 + entry_rank |
| **H1 missing file / H2 wrong assignment length / H3 missing layer row / H4 map-without-TP / H5 wrong entry_rank** | all REFUSE BY NAME at load (fail-closed receipts in the log) |
| E/F/G (poison, spec co-refusal, PP refusal) | refuse by name, unchanged |
| R1 swap-wo / R2 swap-ep-gateup / R3 skip-peer-combine | bite at 1.429e2 / 2.705e2 / 6.742e1 (unchanged) |
| **R4 corrupt-ep-map** (rank-0 local-slot table reversed under the armed skew map — owner table and slab bytes disagree) | BITES: max_rel 5.129e1, tape diverges — the map indirection is load-bearing |
| **T trace-identity + rows-counted** (`MEMRA_MOE_TRACE` + `MEMRA_MOE_WEIGHT_TRACE` armed together — the exact pair the box trace cells arm) | trace ON byte-identical BOTH regimes; 56 token events/MoE layer exactly (=2*(P+N)); 123 rows all shape-checked |

Unit tests (`cargo test -p memra-engine --lib`): ep_map (5) + moesd + glm5_tp laws —
10/10 green. Standing batteries (default arm — tap OFF, map absent; the shipped program
byte-for-byte untouched): all 17 standing GPU suites green (`gates/*.log`), server
suite, split matrices, clippy zero (workspace, all-targets), fmt clean,
`tools/check-flags.sh`, `tools/local-ci.sh --perf` — see the status log.

## 3b. MERGE-FORWARD onto the doors-ON head (2026-08-31, second agent)

The lane was based at 94e6e4872; `origin/lane/glm53-flash-bringup` had moved 114 commits
to **9158ea5d5**, where the four glm5-matvec doors T/X/K/W are **DEFAULT ON**
(a59c9d6b4, mv-battery box receipts). Merged forward at **38734863a**.

Merge shape: every code file auto-merged (upstream heavily rewrote `hybrid_forward.rs`,
`lib.rs`, `hybrid.rs`); the lane's four load-bearing pieces were re-verified present
after the merge — `pub mod ep_map`, the EP-walk `trace_moe_routes` call, the
device-routed step-TP-walk tap refusal, and the `ep.local_of[ex]` indirection at the EP
combine. Two conflicts, both append-shaped: `docs/FLAGS.md` resolved row-by-row
(upstream wins the T/X/K/W rows, this lane wins `MEMRA_GLM5_TP` / `MEMRA_GLM5_EP_MAP` /
`MEMRA_GLM5_TP_GATE_RED`), and `research/tune-data/perf-ci.jsonl` taken as the union of
both sides ordered by `ts` (append-only shared journal — nothing dropped).

### Gate re-run on the merged head — `gates/02-tp-gate-p16-n12-mergefwd.log`, exit=0

`glm5-tp-gate` P=16 N=12: **ALL ARMS PASS, 40/40**, and the verdict lines are
**BYTE-IDENTICAL to the pre-merge run** modulo the per-run tempdir pid in the H1/H2/H3/H5
refusal strings (`diff` of the PASS lines with `epmap-<pid>` normalized is empty). Every
number above still holds on the doors-ON head: arm M decode byte-identical with prime
max_rel 3.637e-5 = arm B exactly, reds R1/R2/R3/R4 at 1.429e2 / 2.705e2 / 6.742e1 /
5.129e1, arm T 56 events/MoE layer and 123 rows. The arm-T fixture traces re-emitted
**byte-identically** (the files were not even dirty afterwards) — tap determinism holds
across the doors flip.

### The doors flip broke its own gates — found here, fixed here (92626d44e)

Re-running the standing battery on the merged head turned up a real defect the flip left
behind. Doors T/X/K/W read `!= Ok("0")`, so **an OFF arm expressed by leaving the
variable UNSET is now an ON arm**, and all three reference arms in
`crates/memra-engine/tests/glm5_matvec_doors_gpu.rs` were unset-shaped:

| door | what the flip did to its gate |
|---|---|
| **T** (route A/B) | **FAILED LOUDLY** — `flag-off arm moved the wide-tcols dispatch counter, left: 1 right: 0`. Caught only because this gate also asserts its dispatch counter stays flat on the off arm. That counter assertion is the whole reason the class was discovered. |
| **X** (`x1 vs x4` bitwise, t=2..=8) | passed **VACUOUSLY** — the `y_x4` "reference" ran with the var unset, i.e. through the x1 grid: x1 compared against x1. |
| **K** (`sharded vs standing` top-k) | passed **VACUOUSLY** — the `(v_off, i_off)` "standing" reference ran at n_cols=20000, above door K's 16384 engagement threshold, so the shard split was compared against itself. Its planted-tie spot assertions are absolute and did keep their teeth, which is exactly why nothing looked wrong. |
| **W** | unaffected — its gate in `glm5_spec_session_gpu` already pins `"0"`. |
| **M** | unaffected — reads `== Ok("1")`, genuinely default OFF, docs correct. |

Fix: a `without_flag` helper that PINS `"0"`, applied to those three reference arms
(owner law, 2026-08-25: the OFF arm of any flag is pinned `=0`, never merely unset).
Re-run green with every arm proven live — receipt banked with `--nocapture` so the
engagement announces and reds are visible: door T 8 t-widths bit-identical + RED 1596
outputs differ; door X 7 t-widths + RED 8 outputs differ; door K 15x16 value+index
identical + RED row-4 indices move.

Also corrected the flip's **stale defaults**, which said OFF everywhere while the code
said ON: `docs/FLAGS.md` default column `off`->`on` for T/X/K/W (the row prose already
said DEFAULT ON with receipts — only the column was missed), `docs/KERNELS.md` three
kernel rows, and 4 rustdoc headers + 3 inline comments in `lib.rs`. The historical
"Was default OFF at ship BY DESIGN" prose is left intact: correct history, not a stale
default. This is the written-flags law's own failure mode — a default that contradicts
the code — so it is fixed rather than noted.

### Standing batteries on the merged head

17/17 GPU suites PASS (`run-battery.sh`; matvec doors 4/4 after the fix, verify_batch
4/4, tparallel 9/9, spec_session 10/10, dflash 10/10, moe_epilogue 9/9, mtp_head 5/5,
kpool 14/14, hyper 6/6, hc_fused_pre 4/4, hc_decode_ws 2/2, mla_decode_split 3/3, kda
suites 5/5+5/5+4/4, mla_gpu_forward 5/5). Split matrices ALL ARMS PASS: spec-ppn,
hyper-ppn, hyper-batch (`gates/{ppn,hppn,hbatch}/` + the three matrix logs).
`memra-server` 492/492 (upstream added 11 since the 481 run). `cargo test -p
memra-engine --lib` 271/271. clippy zero (workspace, all-targets), fmt clean,
`tools/check-flags.sh` 743 runtime reads, none uncovered.

### End-to-end fixture demo with the SHARED tool (FIXTURE traffic, labeled)

Banked: `fixture-weight-trace.txt` (+`.ids`, the MEMRA_MOE_TRACE twin) — 123 rows,
the mini 4-expert top-2 fixture through both regimes, preserved by the gate's
TRACE_OUT arg. Tool selftest 10/10 in this tree; mints deterministic and sha-pinned
(`fixture-map-{coactivation,even}.json`, reports `fixture-mint-*.txt`):

| layer | even peer_touch | coactivation peer_touch |
|---|---|---|
| 1 | 0.488 | 0.585 (WORSE — greedy bundle loses to contiguous here) |
| 2 | 0.463 | 0.463 (tie) |
| 3 | 0.756 | 0.390 (the win the strategy exists for) |

NOT a co-activation claim (fixture routing is near-uniform and 4-expert; nothing
transfers) — but layer 1 is a real finding about the TOOL: the greedy bundle
strategy can LOSE to even on a layer, and the format already supports per-layer
mixed maps. **Follow-up flagged for the shared tool's author (a PR-addition, not a
fork): per-layer strategy selection — take the better of {coactivation, even} per
layer using the self-receipting stats the tool already computes.** The box mint
(step 2 below) should sanity-scan per-layer stats vs the even baseline either way.

## 4. The box plan (NAMED, NOT RUN — box time coordinated through the lane owner)

Host class: the TP-2 box arc host (2-4x RTX PRO 6000 Blackwell, NUMA pair — the
tp2-battery box). Priors this plan builds on, from the banked tp2-battery window
(`tp2-battery-20260831/RESULTS.md` @ a5477a318 on lane/glm5-tp2-battery): the
real-artifact class gate is **BAND for the EP-MoE decode class** (~3-5e-2 per layer,
saturating 5.2e-2 full trunk — the SHARD-MAP §3 pre-registered fused-vs-sequential
fallback; KDA decode byte-exact; swap-wo red 20-100x louder), and **bare TP-2 v1
measured 22.65 tok/s** with a 13-18 ms/tok join tax — the placement lever prices
INSIDE the TP-2 follow-up stack (EP dispatch diet, grouped prime, native P2P), not
against the 42-43 row. Serving laws apply throughout: real prompts only, interleaved
A/B, greedy is the instrument, vendor-default sampled twin on any serving-shaped row.

1. **Real-traffic trace cells** (measurement, no A/B — the FLEET-FIRST co-activation
   measurement): serve the real `glm53-nvfp4` artifact through the SERVED path with
   `MEMRA_MOE_WEIGHT_TRACE` armed, on the owner pools per traffic class — the
   ranks-mint precedent (darklanes `research/glm53-ranks-mint-20260830/`): **agentic
   (SXC pools) as the serving-default candidate**, prose, mixed. Also split by row
   shape (t=1 decode vs the ship shape's spec-verify rows through DFlash2 +
   VERIFY_BATCH — the l3/SXC pools). Bank per-class traces + shas. The tap forces
   observation mode and appends per layer-forward: NOT a perf cell — no timing number
   leaves a traced boot.
2. **Per-class mints** with the SHARED tool (in-tree:
   `tools/build_expert_placement_map.py --trace <id> --weight-trace <w> --ranks 2
   --entry-rank 0 --expert-count 288 --decode-only`, strategy coactivation vs
   frequency vs the even control): per-class `memra-ep-map-v1` maps + the tool's
   self-receipting per-layer stats (intra-rank co-occurrence fraction, expected
   max-rank expert-touch vs the even baseline, peer-touch fraction) — the first
   honest read of how much co-activation structure glm5's sigmoid noaux_tc-balanced
   router actually has (the balancing may flatten it; the trace decides, not the
   hope). Strategy sweeps are CPU, free. Scan per-layer stats vs even before arming
   any map (the fixture demo showed greedy can lose to even on individual layers).
3. **Real-artifact map identity gate** (correctness before pricing): the tp2-battery
   class gate re-run with the minted agentic map armed vs the even split — KDA decode
   byte-exact, EP-MoE decode inside the banked BAND bars (5.2e-2 class), tape holds at
   tiny prime, plus the corrupt-map red once on the real geometry. Same instrument
   (`glm5-tp2-box-probe` tape mode), TP-2 within the NUMA pair, real devices.
4. **The placement A/B** (the pricing cell): naive even-split vs measured map, same
   binary, same config, interleaved x5 fresh boots per arm (LAW:interleaved-ab; boot
   nonce per arm — arm identity, not liveness), single-stream decode on the real
   pools, greedy instrument + vendor-default sampled twin, loop-law screen. Report
   tok/s spread per arm AND the counters the mint predicted (peer-touch fraction,
   per-rank expert-byte split) from trace replay of a separate NON-TIMED repeat —
   never arm the tap inside the timed boots. Expected effect size is bounded by the
   join-tax finding: if EP dispatch cost dominates the peer-touch delta, the map's
   win shows up only after the EP-dispatch-diet follow-up — state the composition in
   the cell design, price both orders if the window allows.
5. **Follow-ups gated on 4** (named, not designed here): per-class map selection at
   boot (agentic default per the ranks-mint precedent); map x join-diet composition;
   hy3 EP-4 re-cell with the shared tool (coordinator-named); N-way maps when a wider
   TP seam exists (the reader is N-way ready; v1 refuses ranks != 2).

What the rig CANNOT prove, named: any throughput number; the real 288-expert top-8
co-activation structure; NVFP4/fused-epilogue class behavior under the map (box gate
#3 owns it, with the banked BAND bars).

### Box-plan amendment for the doors-ON baseline (merge-forward, 2026-08-31)

The base is now 9158ea5d5+, where doors T/X/K/W are DEFAULT ON. Three consequences the
box cells must carry, or their rows will not mean what they say:

1. **The A/B baseline is the doors-ON ship config**, not the pre-flip program. Step 4's
   "naive even-split vs measured map" arms both run doors-ON; the mv-battery winner
   (T+X+K+W = 70.458 tok/s) is the reference the placement delta is measured against,
   and the tp2-battery bare-TP-2 22.65 tok/s row predates the flip — do not subtract
   across the flip.
2. **Every OFF arm is pinned `=0`, never unset** — this is now load-bearing on the
   command line, not just in gates: `MEMRA_BF16_TCOLS_WIDE=0 MEMRA_BF16_TCOLS_X1=0
   MEMRA_TOPK_SHARDS=0 MEMRA_GLM5_VERIFY_WS=0` is the only spelling of "doors off". An
   unset variable is an ON arm, which is exactly how three of the four doors' own gates
   went vacuous (see 3b). Any cell that wants a doors-off control writes the four pins
   explicitly and prints them into its receipt.
3. **Door composition with the map is UNMEASURED and must be stated per cell.** The TP
   preflight's composition matrix names refusals for HC_FUSED_PRE / HC_DECODE_WS /
   KDA_FUSED_PROJ / MLA_DECODE_SPLIT / VERIFY_BATCH, but T/X/K/W carry **no** TP
   refusal and are now silently ON under `MEMRA_GLM5_TP`. The rig gate proves that
   composition is exactness-clean at fixture scale (40/40 on the doors-ON head, arm M
   numbers unchanged), which is a correctness statement only — whether the doors help
   or fight the EP dispatch pattern on the real geometry is a box question. Price the
   map delta with doors ON (the ship shape) and, if the window allows, once with the
   four pinned `=0`, so the two levers are separable rather than confounded.

## 5. Files

| piece | path |
|---|---|
| map reader (frozen JSON format, fail-closed parser, cover + entry-rank laws; unit-tested incl. the committed-example byte anchor) | `crates/memra-engine/src/ep_map.rs` (+ `JsonObj::raw` widened pub in `memra-gguf/src/config.rs`) |
| shared mint tool + spec + examples (cherry-picked) | `tools/build_expert_placement_map.py`, `research/ep-placement-map-20260831/` |
| tap wiring + hardening | `hybrid_forward.rs` (EP walk -> `trace_moe_routes`; device-routed step walk refusal) |
| engine consumption | `glm5_tp.rs` (`load_glm5_ep_map`, `arm_moe_ep` placement, `owner_of`/`local_of`, `corrupt-ep-map` red), `hybrid.rs` (arming + map-without-TP refusal), EP walk lookup in `hybrid_forward.rs` |
| moesd routed-union exposure (gate arm M's singleton picker) | `crates/memra-engine/src/moesd.rs` (`MoesdLayerUnion.experts`) |
| gate arms M/H1-H4/T/R4 | `crates/memra-engine/src/bin/glm5_tp_gate.rs` |
| FLAGS rows | `docs/FLAGS.md` (`MEMRA_GLM5_EP_MAP` new, `MEMRA_MOE_WEIGHT_TRACE` + `MEMRA_GLM5_TP_GATE_RED` + `MEMRA_GLM5_TP` rows updated) |
| receipts | this dir: `gates/`, `fixture-weight-trace.txt`(+`.ids`), `fixture-map-{coactivation,even}.json`, `fixture-mint-*.txt`, `run-battery.sh` |

## 6. Status log

- Lane open 2026-08-31 @ 94e6e4872. Law read verbatim (darklanes feed35586);
  SHARD-MAP §3 + tp2 LANE.md merge-forward read first; the tp2-battery class-gate
  BAND receipts (a5477a318) folded into the box plan when they landed mid-lane.
- First build: own JSONL tap + own Rust partitioner + consumption leg; full gate
  green (ALL ARMS incl. the six new), fixture mint demo banked deterministic.
- **Fleet reconciliation applied mid-lane** (coordinator): own tap and partitioner
  DROPPED; glm5 EP walk wired into the fleet tap; device-routed refusal retargeted at
  the fleet tap; arm T re-pointed at `MEMRA_MOE_WEIGHT_TRACE`; map magic fixed to the
  frozen `memra-ep-map-v1`; FLAGS rows redone. Full gate re-run: ALL ARMS PASS,
  pre-existing arms verdict-identical.
- Standing batteries re-run green AFTER the reconciliation (default arm untouched):
  17 GPU suites ALL PASS (`run-battery.sh`, incl. matvec doors 4/4, verify_batch 4/4,
  tparallel 9/9, spec_session 10/10, dflash 10/10, moe_epilogue 9/9, mtp_head 5/5,
  kpool 14/14, hyper 6/6, diet doors, kda suites, mla_gpu_forward 5/5) +
  `kda_fixture_gpu` 3/3 (non-ignored twin run). Split matrices ALL ARMS PASS:
  spec-ppn 8/8, hyper-ppn, hyper-batch (`gates/{ppn,hppn,hbatch}/`). `memra-server`
  suite 481/481. `tools/check-flags.sh`: 729 runtime reads, none uncovered. clippy
  zero (workspace all-targets), fmt clean. `tools/local-ci.sh --perf`: see
  `gates/local-ci-perf.log` + the perf-ci row.
- **Shared tool landed mid-lane** (origin/main merge 4e46be545): three lane commits
  cherry-picked in (tool + selftest receipt + spec/examples); the reader REWRITTEN
  for the frozen JSON (JsonObj, no serde), `entry_rank==0` law added (+ gate arm H5),
  arm T re-armed with BOTH taps (id + weight — the tool's two inputs), TRACE_OUT
  preserves both. Full gate re-run on this head: ALL ARMS PASS (M/H1-H5/T/R4 +
  pre-existing arms verdict-identical). Tool selftest 10/10 in-tree; fixture demo
  minted with the SHARED tool, deterministic, per-layer stats banked (incl. the
  layer-1 greedy-loses-to-even finding, flagged as a PR-addition follow-up).
- `tools/local-ci.sh --perf` green TWICE (pre-JSON head and the final head; exit 0
  both): correctness stage GREEN; perf 0 fail 1 warn both runs (qwen9b 134.91 /
  135.29 vs rolling median 138.59, -2.7%/-2.4% — the known rig-thermals warn class,
  not a lane effect: the cell never touches this lane's code). perf-ci rows appended;
  final log `gates/local-ci-perf.log`.

### Handover (first agent died to provider errors mid-run; second agent from here)

- A THIRD `local-ci.sh --perf` was in flight when the first agent died. It completed:
  correctness GREEN, perf **1 fail** — qwen9b 133.99, -3.32% vs the 138.59 median.
  Banked rather than discarded as `gates/local-ci-perf-run3-7af67eb3e-FAIL.log` with
  its perf-ci row (46afd6758). House rule: a WARN passes, a FAIL is retried in a
  quieter window, spaced — honored by the post-merge re-run below, which is the head
  that actually ships.
  The drift is machine state, not a diff: after the merge brought upstream's rows in,
  **six** runs sit in the 08:54-09:38 window across **five different commits**, all
  133.99-135.34 against the 138.59 median, and none of those commits touches the
  qwen9b path.
- Merge-forward onto 9158ea5d5 (doors default ON) at 38734863a; gate re-run 40/40 ALL
  ARMS PASS with verdict lines byte-identical to the pre-merge run (tempdir pid aside).
  See section 3b.
- **Defect found and fixed in-lane** (92626d44e): the doors flip left doors T/X/K's own
  reference arms unset-shaped, so door T's gate FAILED and doors X and K passed
  VACUOUSLY (door-vs-itself). `without_flag` pins `=0`; all four re-run green with
  engagement announces and reds biting. The flip's stale "default OFF" docs (FLAGS
  column x4, KERNELS x3, lib.rs x7) corrected to ON. Section 3b has the table.
- Standing batteries, split matrices, server suite (492/492), lib tests (271/271),
  clippy zero, fmt, check-flags (743 reads) all green on the merged head.
- **`tools/local-ci.sh --perf` on the merged head @ 92626d44e: exit 0, correctness
  GREEN, perf stage 0 fail 0 WARN — qwen9b-plain-short 138.17 tok/s [OK]**, i.e. back
  on the 138.59 rolling median (`gates/local-ci-perf.log`, perf-ci row
  2026-08-31T10:28:23Z, load 6.41). This CLOSES the inherited FAIL: the same cell, same
  harness, quieter window, returns to the median while carrying this lane's whole diff
  plus 114 upstream commits. The 133.99-135.34 band across five commits was the rig,
  exactly as the log's own drift-tripwire note predicted — no interleaved A/B needed
  because the tripwire cleared itself on the shipping head.
- The three split matrices' `hppn/` and `hbatch/` arm logs came back **byte-identical**
  to the pre-merge run (git saw no change at all); `ppn/` differs only in per-run
  incidentals. Independent determinism evidence for the split walks under doors-ON.
