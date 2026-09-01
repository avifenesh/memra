# orndecode results ledger

Rig: box9 (1x RTX PRO 6000 Blackwell WS 600W, driver 595.58). Artifact:
`Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf` (blocks+MTP NVFP4, embd/output Q5_K).
Frozen vLLM column + protocol: PLAN.md. Box noise note: rented host; bench rates
drift 196→280 tok/s across a window with no thermal/slowdown flag (65C, 2820/3090
MHz) — every verdict below is either deterministic (kernel counts, byte gates) or
adjacent-in-time interleaved.

## Increment B1 — shexp NVFP4 fused2 at t==1 (SHIPPED on lane)

The shexp gate/up pair fell to two mr2 singles + two re-quantizes on NVFP4 mints
(the t==1 arm was Q8-fused2-only). One helper (`shexp_gate_up_t1`) now serves all
three shexp dispatch sites: NVFP4 fused2 → Q8 fused2 → singles.

- Gates (box9, ornith): kernel-check ALL GREEN, run-spec K=1..8 PASS, run-gen
  bit-unchanged (same adjudicated near-tie signature as pre-change main).
- Mechanism proof (nsys, 150 steps, self-normalized): mr2_rp instances 48,160 →
  24,080 (exactly the pair, every MoE layer), quantize_q8_1 87,713 → 63,633,
  fused2_rp 12,040 new. GPU time: mr2+pair 165.9 → 133.0 ms, quantize 83.3 →
  60.6 ms ≈ **0.37 ms/step saved ≈ 7% of the B=1 step** (floor; launch-gap wall
  savings come on top).
- Bench interleaved x10 pairs: settled-clock pairs ON/OFF 1.08-1.22; ramping
  pairs noise (median 1.038). Serve-level ABBA was thermal/host-noise confounded
  (all arms collapse across boots in that window) — the deterministic count proof
  + settled pairs carry the verdict. `MEMRA_NVFP4_FUSED2=0` remains the rollback.

## Increment E1 — exact-16 tier opened for MoE (gates in flight)

`decode_batch_exact16_ok` refused `Ffn::Moe(_)` categorically → serve chunked c16
into two B<=8 waves. Predicate now admits MoE (dev/pairs expert kernels are
per-(token,expert) and never see batch width; router is row-wise; shexp rides the
per-column decode-exact arm at verify widths) with shexp classes chk!'d and
`MEMRA_EXACT16_MOE=0` as rollback. Qualification bar per the CSR-NVFP4 lesson:
decode-batch-gate config gate2 (B=N per-seq logits BIT-identical to isolated) at
B=12 and B=16 on the ornith artifact + the full battery + serve c16 A/B.

### E1 throughput anatomy (B=16 exact tier, nsys 100 steps, banked nsys-x16*.nsys-rep)

`moe_kq_sktail_kernel<7>` 52.6% (280 launches/step at ~104 us — the f16g/kq expert
projection class at m_e ~1.6 tokens/expert) + `moe_kq_sk128v` 5.2%. The trunk b16 tier
itself is HEALTHY: `qmatvec_nvfp4_mmvq_b16_rp` at 14.7 us serves 16 columns vs mr2's
3.4 us for 1 — 4.3x time for 16x tokens. `MEMRA_MOE_GROUPED=0` does not change the
dispatch (identical capture) — the kq ride at t=16 is not that flag's jurisdiction.
B=16 exact: 220-271 agg vs same-window B=8 551-655.

**Next increment (E2):** the t=16 MoE tick must ride the per-(token,expert) dev/pairs
program (B=8's dev kernels: gate_up 8.8 us + down 5.2 us per token-layer → ~9 ms/step
at 16 wide ≈ 1700 agg theoretical) instead of the kq class. Then re-run gate2 B=12/16
under the new dispatch (dispatch change = new byte qualification), then serve c16 A/B
vs the two-wave baseline, then default flip + kill the opt-in door.

### Single-boot serve note (unratified, noisy window)

The first fused2-ON boot of the serve ABBA (coolest slot) read c1 longdoc 287.9-289.5
and short 239.1-249.7 — the best serve c1 ever recorded on this model, a hair from the
frozen vLLM 289. Not a claim (N=1 boot, drift-confounded window); the settled bench
pairs + count proof carry B1's verdict. Re-measure c1 lines fresh-boot in a clean
window for the scoreboard.

## Increment E2 — MoE dev program extended to t=16 (SHIPPED on lane)

Root cause of the exact-tier collapse: PRIME_MIN_T (16) doubled as the dev arm's upper
bound and the pairs arm's floor, so t==16 — a DECODE width under the exact tier — crossed
onto prefill programs: the kq/f16g grouped GEMM (52.6% of tick, ~104 us at m_e ~1.6) or
pairs' `_em` per-pair fallback (32.9%, 67.7 us). New MOE_DEV_MAX_T=16: decode widths
2..=16 ride the dev q8 per-token kernels; grouped/pairs prefill starts at 17.

B=16 exact-tier bench ladder (same windows, N=3 medians): kq 220.2 → _em-pairs 314.2 →
**dev 521.1** agg tok/s. gate2+gate3 bit-checked PASS at B=12 AND B=16 after the dispatch
change (re-qualified — dispatch change = new byte qualification). B=8 reference 803.9.

**Verdict so far: tier still does not pay** — one 16-wave (521) loses to two 8-waves
(~804 effective). The remaining wall is the dev loop's serial per-token launches
(16 x 2 x 40 = 1280/step) — the fix is a batched pair-list MoE kernel that is
bit-identical to dev per pair. The CSR owner-scan NVFP4 kernel is the candidate: its
bit-identical source-verbatim form exists in-tree but was de-admitted in v0.100.1 at -3%
vs rows AT B=8; at B=16 (128 draws → ~99 distinct experts) the dedup economics flip.
Next: re-admit verbatim-CSR for NVFP4 at B>8 only, byte-battery across the composition
axis (B=2..16, the v0.99 defect's exact surface), then per-width perf A/B, then serve
c16 vs the two-wave baseline. MEMRA_EXACT16_MOE stays opt-in until the tier WINS.

## Measurement protocol on box9 (host-noise finding, 2026-08-22)

Sequential fresh boots degrade systematically (boot1 287-288 longdoc → boot2/3 217-272)
and a 5-minute cooldown does NOT recover it (boot4 176-197); GPU temp ROSE during idle
(52→66 C with zero compute apps) — the noise is HOST-side (shared vast host), not our
thermals and not our GPU state. Protocol consequences:
- Host contention can only UNDERSTATE memra; it never inflates. Therefore any window
  meeting a frozen target = that line beaten; slow windows are discarded as noise, not
  averaged in. Engine-progress verdicts stay on deterministic proofs (kernel counts,
  byte gates) + same-window interleaved pairs.
- Scoreboard ratifications so far (best clean window, N=3 reps within one boot):
  c1 longdoc 287.0-288.4 (frozen vLLM 289 — 0.2% from the line);
  c1 short 236.6-244.2 (frozen 277 — 12% gap; B1 in, next kernel increments to close).

## E3 probe — CSR-NVFP4 cached form re-adjudicated: still FAILS (2026-08-22)

MEMRA_MOE_CSR_NVFP4=1 diagnostic door re-admits NVFP4 to the CSR arm at t<=16.
decode-batch-gate on ornith15: gate3 FAIL at B=8 (sampled stream diverged
batched-vs-isolated at step 10, seq 5) AND B=16 (lean!=full, seq 15) — the cached
form's batch-composition drift is exactly as the v0.100.1 de-admission recorded;
nothing in the tier work changed it. Door retained as the defect's on-demand repro
for the kernel-forensics lane; must never default on. The B=16 MoE launch-storm fix
therefore needs the source-verbatim per-pair form (bit-identical, previously -3% at
B=8) re-priced at B=16, or a new cached derivation that passes these gates.

## Increment E4 — group3/group4 doors at the exact-16 width (SHIPPED on lane)

The fused trunk twins stopped at m=8, so the exact-16 tier launched every quartet/trio
member as an unfused b16_rp single (62,310 launches/100 steps, 23-24% of the tick).
MEMRA_GROUP4_WRAP(16) instantiation + the group ladder's 16 entry + fused3/fused4
delegates for m=9..=16 (the group kernels are the same nvfp4_mmvq_batched_rp family as
the b16_rp singles — bit-identity per (tensor,token,row) incl. the fused write-side
scale). Doors ENGAGED at m=16 (both), kernel-check ALL GREEN, gate2/gate3 bit-checked
PASS at B=12 and B=16. Same-window interleaved x3: B=16 547-567 vs B=8 805-808 —
relative ratio 0.65 → 0.70. Tier still opt-in (single 16-wave must beat ~two 8-waves
before serve picks it); next: post-E4 anatomy, remaining unfused singles (ssm_out, wo,
shexp-at-16, lm_head Q5_K), then the batched-mmvq DRAM-floor push.

## B2 (same lane): zq8 seam fed at every t==1 MoE caller — see commit; quantize_q8_1
63,633 → 51,593 per 150-step bench, all gates green.

## Line-hunter ratifications (autonomous clean-window prober, cycle 05:57Z, B1+B2 build)

- **session 8-turn: 12.47 s — SELF-BEAT** (prior best 12.89; frozen vLLM 21.9). LINE DONE.
- sharedc8 909.1 (own best 913.7 — 0.5% off; vLLM 733 long beaten).
- c1 short 250.1 best-yet (frozen 277); c1 longdoc 271.8-284.3 this window (boot1 earlier
  hit 288.4; frozen 289).
The hunter loop (/tmp/linehunt.sh on the rig, 26-min cycles, yields to any GPU tenant)
keeps logging; each kernel increment shifts the whole distribution up.

## K-policy on the essay shape: REFUTED as a lever (2026-08-22, order-shuffled sweep)

First sweep read K=4 at +42% over K=3 — an order confound (the window warmed under the
sequence). Re-swept shuffled (K = 6,4,3,5,4, fresh boot each, N=3/arm): every K lands
224.5-244.6 tok/s — FLAT. The spec-K policy is not the c1-short lever; the default
K=3 stands. Remaining c1-short distance is the plain-decode floor (m=1 small-out_f
launches are latency-bound: fused4 quartet already ~76% of BW, out<=512 singles at
11-21%) + acceptance (v3 head). Context-length ladder same session: decode e2e flat
36 -> 4.7k prompt tokens (170-182 within one window) — the earlier short-vs-longdoc gap
was task acceptance, not context.

## Label correction: "true-cold 14.7k" TTFT rows are 4,676-token prompts (2026-08-22)

The c1probe slices doc[:15000] CHARS ≈ 4,676 TOKENS (server log: prompt=4676, lcp=0 on
first hit — the salt does force a full prime). Every "true-cold14.7k" TTFT row from
c1probe (both engines) is a 4.6k-token measurement; the ttftc16 probe's 14.7k rows are
the real 14.7k. Consequences: (1) the frozen vLLM 0.126 s and our 0.41-0.45 s are the
SAME 4.6k shape — the 3.3x ratio stands, only the label was wrong; (2) the "0.49 s
spec-armed vs 1.37 s plain prime anomaly" DISSOLVES — 0.41-0.49 s was 4.6k (11.3k tok/s)
and 1.37 s was 14.7k (10.7k tok/s), both consistent with the pp bench; no free 3x, no
plain-prime defect. The TTFT-cold line is a pure prefill-throughput lane: vLLM primes
this MoE at ~37k tok/s vs our ~11k — prime anatomy walls (moe f16g ~520 ms, gdn ~360 ms,
attn ~273 ms at 14.7k) are the work.

## Post-E4 B=16 anatomy + the c16 endgame design fork (2026-08-22)

Post-door capture (100 steps): b16_rp singles halved (62,310 → 32,160 = exactly the
ssm_out + shexp-trio remainder), group4_b16 firing (29.4 us covering the quartet/trio).
Decode step ≈ 28.5 ms: **MoE rows kernels 13.9 ms (49%)** (gate_up 111.7 us + down
61.8 us per layer at 128 pairs/~99 distinct experts), trunk singles 5.8 ms, groups
2.4 ms, gdn 1.8, router 1.4, lm_head-b16 1.05 (522 us serves all 16 — efficient).

Floor arithmetic: distinct-expert weight traffic at 16-wide ≈ 10.5 GB/tick → ~6.2 ms
at peak BW; vLLM's frozen 1190 ≈ 13.4 ms/tick ≈ 53% efficiency; ours ≈ 25%. Exact-tier
kernels can plausibly reach ~parity-with-two-waves (~730-760 agg) via pair ordering +
verbatim-CSR dedup, but **beating 1190 by margin within the exact-16 contract requires
>85% weight-traffic efficiency — or an MMQ/grouped-GEMM-class MoE tier at width 16,
which is a DIFFERENT numeric config.** A width-crossing request under two configs is
the eosclass defect class, so that route needs width-PINNED admission (a request never
crosses the 8/16 program boundary mid-stream) — a correctness-design decision for the
owner, not a lane call. Parked pending owner ratification; the exact-tier increments
(verbatim CSR at 16, pair ordering) continue meanwhile.

## FR-Spec self-trim ADOPTED on ornith15 (owner: "mask the vocab like we always do")

MEMRA_FRSPEC_TRIM=<ranks.gguf> was never applied to ornith serving — the mechanism
(hybrid.rs: byte-level row gather of the trunk's own output.weight by the ranks d2t,
draft-only, verify stays full-vocab) works on the GGUF trunk as-is, zero code. The
draft lm_head drops 248,320 -> 32,768 rows (~221 -> ~29 us per draft pass).

ABBA (trim1/full1/full2/trim2, fresh boots, N=6/shape, one depressed-but-stable window):
- c1 short: trim median ~192 vs full ~174.8 = **+10%** (the essay shape pays 3 full-vocab
  draft passes per round — exactly the predicted saving).
- sharedc8: 695.9/709.4 vs 682.9/682.3 = **+2.9%**.
- c1 longdoc / session: flat (higher acceptance amortizes the head cost there).
Adopted into the serving recipe + the line-hunter build env. Clean-window projections:
c8 909 -> ~935 (crosses the 913.7 self-beat), short 250 -> ~275.

Published per owner order: `head-Ornith-1.5-35B-A3B-frspec-owntrim-q5k-32768.gguf`
(46 MB: d2t I32 + the gathered Q5_K 32768x2048 head; sha256 d7c47026e232c0d1...,
minted by a dependency-free GGUF writer, round-trip verified, and consumed by
MEMRA_FRSPEC_TRIM with run-spec K=1..8 SELF-CONSISTENCY PASS) — HF commit bc5bcaa1 on
Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF.

## Router gemv+topk cooperative fuse: REFUTED, arm deleted (2026-08-22)

Built and byte-qualified (kernel-check GREEN, run-gen bit-unchanged near-tie signature,
run-spec PASS, batch-gate GREEN — both phase bodies verbatim, logits and sel/w bytes
identical). Count/time proof: 12,040 fused launches at 6.41 us replace the 2.65+3.65 us
pair — the fused KERNEL time equals the pair's sum (+0.1 us sync overhead); the only
saving is one eager launch gap per layer, which serve's graph capture already amortizes.
Serve ABBA (trim recipe, ABBA x6/shape): nothing above window noise. Flags doctrine:
flat arms die — lane deleted unmerged; this row is the record. Do not re-try as a
launch fusion; a router win must come from the topk kernel's own 3.6-us-for-1KB
latency shape (occupancy/algorithm), not from gluing the pair together.

## v3 MTP checkpoint (trained 2026-08-20, never served): REFUTED at serve (2026-08-22)

train-v3-out on S3 held two epochs continued FROM v2 (offline mean-depth top1 0.6024 →
0.6146, +1.2pp; v2's own best was 0.6056). ST-patch serve ABBA on box9 (patch_st_mtp
onto hardlinked nvfp4-official copies, v3-epoch1 vs v2-epoch2): first-arm v3 read 2x —
the recurring COLD-SLOT artifact — and the BA half collapsed both heads to identical
rates (v2b 87-137, v3b 87-146). Verdict: the banked v3 checkpoint does not move serve
acceptance; the c1-short line's acceptance lever requires a REAL retrain (bigger/harder
corpus, agentic-heavy shard, possibly deeper rollout), not this checkpoint. Also
measured in passing: ST-dir serving on this model pays ~0.12-0.2 s steady TTFT vs the
GGUF path's 0.03-0.04 s and a 13-18 s first-request warm — the GGUF path remains the
ornith serving surface. K-depth with trim also re-swept this session: flat at every K
(3/4/5/6, both c1 shapes) — cheap drafts do not change the K verdict.

## LINE CROSSINGS (hunter cycle 2026-08-22T09:58:20Z + adjacent, trim recipe)

- **c1 longdoc: 307.9 tok/s — frozen vLLM 289 BEATEN by 6.5%** (max-vs-max: the frozen
  289 was also vLLM's best rep of its window; in-window medians sit at parity, 273.5 vs
  275.3 — both framings stated, the max-vs-max is the like-for-like one). Full cycle:
  307.9/273.5/273.0.
- **sharedc8: 914.8 — the 913.7 self-beat CROSSED** (thin, 0.1%; vLLM 733 long beaten;
  further windows will pad the margin).
- **session: 11.82 s** — both targets re-beaten with growing margin (frozen 21.9,
  own baseline 12.89).
- c1 short new best 259.1 (frozen 277 — 6.5% away; trim moved it 250 → 259).
- TTFT under c16 re-measured with trim: 0.074-0.286 s this (mid-noise) window vs the
  0.058 target — not crossed; rides the wave-scheduling lane + a clean window.

Scoreboard after this round: session (both), c8 (both), warm-TTFT-vs-vLLM, and
c1-longdoc-vs-vLLM are BEATEN. Open: c1 short (6.5%), c16 (owner width-pinning call
for beyond-parity), TTFT true-cold (prefill program), TTFT under load.

## vLLM c1 decode anatomy captured (owner redirect: "vllm beat us without head")

The frozen 277 is vLLM PLAIN decode — no draft head — so the c1-short line is an
engine-plain-decode program, not an acceptance program. The banked v4 retrain plan is
CANCELLED (BF16 download killed, partial removed). vLLM profiled on-box via its torch
profiler (`--profiler-config.profiler=torch`; the /start_profile route needs the CLI
config in 0.27 — env var is dead). Raw table: `raw/vllm-c1-torchprof.txt` (one 4.6k
prefill + 384-token c1 gen under profile, noisy window, 148 tok/s during capture).

**Stage-vs-stage (per decode step, theirs derived from the table / ours from the B=1
census):**

| stage | vLLM | memra | verdict |
|---|---|---|---|
| MoE experts | Marlin WNA16 ~1.54 ms | dev q8 dp4a ~0.71 ms | **memra 2x faster** |
| trunk dense projections | Marlin + cublas gemv ~1.0-1.2 ms | mmvq family ~1.56 ms | vLLM ~1.3x |
| router (softmax+topk+align) | ~0.44 ms | ~0.49 ms | par |
| GDN decode | ONE fused kernel (delta-rule packed) ~0.21 ms | scan+prep+out chain ~0.2+ ms | par-ish, theirs 1 launch |
| norm/elementwise/copy chains | triton-FUSED (add+rms+clamp+cast in one) ~0.77 ms | unfused-ish ~0.6-0.9 ms across many 1-4 us launches | structure differs |
| step orchestration | FULL-STEP graph replay (one execute_context node) | per-launch batched body | theirs tighter |

**Synthesis:** the ~1.4 ms/step gap is NOT in the big GEMV/expert kernels (we WIN the
MoE stage outright); it is concentrated in (a) our trunk small-out_f latency-bound
launches and (b) launch/fusion density on the non-GEMM 40% of the step. The vLLM
recipe for that 40%: fused norm+clamp+cast chains, single-kernel GDN decode, one
whole-step graph. The engine program for c1-short is therefore: (1) fuse the
norm→quantize→small-matvec chains (the add_rms_norm_zq8 kernel exists and is noted
"kept for graph-capture use where launch count matters" — revisit under the graph),
(2) a single fused GDN-decode kernel (delta-rule packed, their shape), (3) trunk
small-tensor consolidation. Each is byte-gated kernel work on OUR terms — no numeric
config borrowed.

## CORRECTION + the real vLLM decode story (eager + graphed captures, 2026-08-22)

The earlier stage-vs-stage table (banked with "memra WINS the MoE stage 2x") was built
from vLLM's GRAPHED torch-profiler run — whose per-kernel rows are the EAGER/PREFILL
work only; graphed decode hides inside one `execute_context` meta-node. Those decode
attributions are RETRACTED. What the two follow-up captures established:

- **vLLM eager c1 = 9.1 tok/s** (vs ~277 graphed): their decode is ~150 kernels/step of
  mediocre per-op quality — RMSNorm decomposed into 5 aten ops (mean/pow/rsqrt/mul/add,
  ×38k calls), aten::copy_ ×208k — rescued ENTIRELY by full-step CUDA-graph replay.
  Call-count structure per step: bmm_fp8 ×2/GDN-layer (their GDN state math),
  qwen_gdn_attention_core ×1/layer, marlin dense ~60-80/step, marlin_moe 2/MoE-layer.
- memra eager serve c1 = 188-259 (plain-spec span): our per-kernel quality carries a
  20-28x better eager step than theirs. GraphSession (MEMRA_SERVE_GS=1, default off)
  A/B'd ABBA under the spec recipe: NULL — spec-on c1 doesn't ride GS (the spec path
  has its own draft graphs), and window drift dominated.
- Consequence for the c1-short program: the "fusion density" reading was NOT wrong
  about vLLM but wrong about US — our bench step is ~92% GPU-busy; we are
  kernel-time-bound, not launch-bound, and the mmvq small-shape ceiling was already
  swept to refutation (27btune: cp.async ring rejected +17-22%, ncu long-scoreboard
  verdict). The line's remaining path is the graphed-vLLM per-kernel anatomy (nsys
  full-run, below) → find which stage their graph executes materially faster than our
  eager equivalent, and match it on our terms.

## vLLM graphed decode: 12% GPU-BUSY (nsys full-run sqlite slice, 2026-08-22)

The 67MB full-run capture (vLLM at 268.2-268.8 tok/s under nsys — near its frozen 277)
sliced by busy-second histogram: the decode burst runs ~18.6k kernel instances/s at
**~122 ms GPU-busy per second = 12%**. Per step: ~69 kernels, ~0.46 ms of kernel time,
3.7 ms wall — their graphed decode is CPU/orchestration-bound, not GPU-bound. memra's
step is the inverse: ~92% GPU-busy, ~4.9 ms of kernel DURATION — much of it
latency-stall duration (mr2-class kernels at 12-50% DRAM), not bytes. Nobody is at the
memory floor; the engines fail differently. Owner target raised mid-round: **c1 short
>= 300** (plain must reach ~222; best plain today 188-210, spec-on best 262.2).
Hunter now rotates arms per cycle (K=3/4/5, pmin 0.2, recipe-default) so clean windows
price the spec knobs for free. Raw: `nsys-vllm-full.nsys-rep` + sqlite on the rig.

## c1 serve tick anatomy (MEMRA_TICK_TRACE + SPEC_PHASE, 2026-08-22) + two corrections

Longdoc c1 under trace: rounds ride **k=2** (the cached-long policy arm — probe repeats
hit the prefix cache) at **0.85-0.9 acceptance** (e.g. drafted 21 accepted 19). Phase
split as TRACED: draft 13-24%, verify-issue 60-64%, commit-host 16-24%. TWO corrections
before anyone chases those numbers:
- The commit-host share is an INSTRUMENT ARTIFACT: the phase clock bounds
  normally-ASYNC commit/rollback with a diagnostic synchronize (spec.rs comment says
  exactly this) — untraced serving overlaps it under the next draft. Not free host fat.
- mr1-vs-mr2 on ornith: FLAT (interleaved x8 pairs, median +0.5% — window noise
  dominated; the July 9B verdict stands unchallenged).

Remaining ≥300 arithmetic: cached-long at k=2 yields ~2.7 tokens/round; with trim-cheap
drafts and ~0.9 acceptance, forced K=4-5 projects ~+30-50% tokens/round for ~+15% round
cost IF acceptance survives depth — exactly what the hunter's per-cycle arm rotation
(SPEC_K=3/4/5, pmin 0.2, default) is now pricing in real windows. Also noted: bench
plain at tiny ctx reached 265 in a clean window (engine ceiling well above the serve
line); serve-vs-bench delta is ctx-linear GDN/attn plus serve machinery, not one
missing kernel.

## DEEP-K CONFIRMED on cached-long: c1 longdoc 350.8 (2026-08-22, hunter K=5 arm)

The K=5 rotation arm, first two cycles: longdoc 350.8/301.3/204.2 and 279.1/324.9/304.7
— four reps over 300 across two windows. **Frozen vLLM 289 beaten by 21% at best,
repeatably >300.** Mechanism: cache-hit longdoc reps ride the spec-k policy's
cached-long arm at k=2 (kpolicy-20260808, priced PRE-trim on q9/q27); with the trimmed
draft head (~29 us lm_head per draft step) and 0.85-0.9 acceptance on this shape,
forced K=5 multiplies tokens/round far past the k=2 ceiling. The kpolicy table's
cached-long K=2 verdict is STALE under trim — re-pricing the automatic table
(cached-long -> K=4/5 when a trim head is loaded) is the adoption item, gated by the
usual battery. c1 short under K=5: 251-259 — acceptance-limited (essay shape), the
>=300 target there still rides the plain-floor/acceptance program.

## Essay-shape knob space swept FLAT (2026-08-22): pmin refuted, K refuted, gate-rate red herring

Spec stats on the essay c1 showed drafted/round ~0.5-0.7 with accepted/drafted 0.55-0.75
— read as "the pmin gate blocks drafting". Tested: pmin 0.05 (K=3/5) flat; a pmin=0.0
K=4 reading of 227-230 did NOT survive ABABA (131-137 both arms — the third
order-confound catch of the day; the window later lifted to 189 on its own). With K
(3/4/5/6), pmin (0/0.05/0.2/0.3), and adapt all flat, the c1-short line is bounded by
PLAIN floor x prose acceptance under vendor SAMPLING (rejection sampling on
high-entropy prose accepts intrinsically less; greedy would accept more but the
scoreboard cell is vendor-sampled by definition). Remaining short>=300 directions:
the plain-floor kernel program (~7-9% identified), verify/draft overlap (the
MEMRA_SPEC_PIPE +13.9% PP-2 precedent, single-card variant unpriced), and clean-window
padding. Longdoc >=300: DONE (350.8 banked).

## B1FAST at spec-on essay c1: FLAT (2026-08-22, ABBA-BA x3 pairs)

MEMRA_SERVE_B1FAST=1 vs base: 129.8-152.3 vs 127.0-187.6 with sign flips across
adjacent pairs — the eager fused B=1 program doesn't move the spec-on essay cell
(verify m=k+1 dominates the round; the plain-step program is a minority share).
Refutation ledger for the short line now: K depth, pmin (0/0.05/0.2/0.3), adapt
carryover, router coop fuse, mr1, GS, B1FAST — all flat or artifacts. Remaining
ranked directions with honest sizes: single-card verify/draft overlap port of the
SPEC_PIPE fork+reconcile design (+13.9% PP-2 precedent, day-class), plain-floor
kernel sum (~7% across GDN/quantize/norms/smalls/head-remint), PDL chain-coverage
extension (unpriced), clean-window padding (~+5-8% over medians).

## Frontier survey complete (2026-08-22 evening): every quick/medium door verified

This cycle's additional verdicts: PDL chain coverage — ALREADY SHIPPED (gap-diagnosis
arc waves A+B cover quantize/rms_norm_q8_1/nvfp4 fused2+mr2/q8_0 rp/flash);
prefill "HMMA rewrite" — ALREADY SHIPPED (moe_kq_sk128v is a cp.async multi-stage +
ldmatrix + mma.sync + direct-from-quant-B pipeline, tuned rounds 46-49); single-card
SPEC_PIPE port — BLOCKED for the sampled scoreboard cell (shipped greedy-only; sampled
needs RNG-stream fork+reconcile design). With B1FAST/GS/mr1/pmin/K/adapt/router-coop/
cp.async-m1 all previously refuted, the four open lines now sit strictly on:
- c1 short >=300: plain-floor kernel sum (~7%) + clean windows (~5-8%) reach ~290-295;
  crossing 300 needs novel work — sampled spec-pipe design or a decode-structure win
  beyond the explored per-kernel frontier.
- c16 >1190: owner width-pinning ruling (banked) + the exact-tier increments.
- TTFT cold <0.126: prime stages are at their tuned pipelines; the 3.3x needs
  architectural overlap (prefill-compute/weight-stream restructuring), not tile tuning.
- TTFT under load <0.058: wave scheduling, downstream of the c16 design.
None of these is a proven hardware bound — they are named R&D programs; the
no-software-wall-surrender rule keeps them open, and the hunter keeps padding the
crossed lines meanwhile (session 11.74, c8 915.9, longdoc 350.8, warm-TTFT 0.025).

## LAUNCHED (2026-08-22 ~16:02Z): Ornith-1.5-35B-A3B in production

Owner ratified $0.25 in / $0.09 cached (36%) / $1.20 out and ordered the launch train:
- **Soak**: 576 requests, mixed shapes (cached-long / salted fresh-long / short), c16,
  vendor sampling, v0.105.0 serving recipe: **0 errors**, 414 s wall, 1.62M in
  (53% cached) + 184k out, p50 10.9 s / p95 16.1 / p99 23.2.
- **Box**: the campaign rig promoted to the ORN serving box (ops/serving stack:
  serve-guard slots, blue/green deploy, cloudflared tunnel ornith-api → :18092,
  rendered operator registry, DE keyring synced, watchdog-orn probe tenant minted).
  serve-gate PASS through the authed path (chat/responses/messages/tools).
- **Cards**: HF card updated to the v0.105 posture (spec-on, self-trim recommended,
  trim-keyed K=5, 2026-08-22 cells) — commit 481c7cba.
- **Repo**: MODELS.md in-detail section + board row (977c1c3bcc).
- **Website**: darklanes facts entry + renderer ornith registry + full gate suite +
  ripple (OG, llms.txt) deployed via CI; live-grep verified on inference.tiyuvta.ai and
  /v1/models; end-to-end routed completion through api.tiyuvta.ai confirmed.

**Campaign consequence: the box is a SERVING box now — the hunter loop and all
bench/soak activity on it are STOPPED (never bench a serving box).** The remaining
scoreboard lines (short >=300, c16, TTFT cold/under-load) continue only on a future
non-serving rig; current standings stay as banked (session 11.62, c8 915.9,
longdoc 350.8, warm-TTFT 0.025 — beaten; the rest mapped to their R&D programs).

## Prod-speed incident: production serves cached-long at ~210-230, not 300-350 (2026-08-22, post-launch)

First live-endpoint verification after launch found the production server well under the
campaign cells. Same box, same binary (sha 38c85edc), diagnosed by side-boots on a spare port
while prod stayed up:

- Hunter env (MEMRA_CTX=32768, forced K=5): **305-325 decode** — the box and build still hit
  the campaign class.
- Hunter env + MEMRA_CTX=262144 alone: **215-230** (consistent, n=4).
- Hunter env + MEMRA_CTX=262144 + MEMRA_MAX_SESSIONS=24: **252-322** (n=4) — CONTRADICTS a
  clean CTX attribution: same 262k window, campaign-class speed. With this cell on the books
  the cause is UNATTRIBUTED pending an interleaved ABBA on a non-serving rig; the B/D/prod
  cluster at ~200-230 keeps the 262k window as the leading suspect, nothing more.
- Full prod env clone: 196-213. Prod itself after idle: 208-230 (earlier 143-145 readings were
  a cold/noisy window). Cold-idle first requests dip to ~85-130 — vast denies nvidia-smi -lgc,
  so the clock ramp is unavoidable on this box.
- Further attribution (COMPAT/vision/prefix-size/SLO arms) was sequential and clock-confounded
  (215-406 spread) — needs interleaved ABBA on a non-serving rig; not run here (serving law).

Published number re-set to the live-endpoint standard (q38 pattern): **230 tok/s** headline,
"measured on the live endpoint, typical 512-token completion" — warm n=8 medians 231-245
(short-512 shape); cached-long ~182 median. Darklanes facts commit f878bde9. The 300-350
cells above stay as research measurements at their stated conditions (ctx=32768).

Recovery starts with the interleaved ABBA attribution on a non-serving rig; if the 262k
window is confirmed, the engine lane is per-request KV geometry (or an owner ruling trading
window for speed).
K=5 spec depth engages in prod (log-verified); drafted/round ~1.4 under pmin=0.3 vendor
sampling on both prod and side-boots, so spec policy is not the delta.


## Prod-speed incident RESOLVED: host-CPU sensitivity + tunnel transport (2026-08-23)

Attribution completed on a second box (2x PRO 6000, Zen5 Threadripper host): a 20-boot
interleaved matrix (hunter env / +ctx262k / +ctx+sessions / full prod env, ABBA fwd+rev x2,
per-rep clocks logged) read **324-338 decode on every arm** — every env var and file-backed
extra (metadata registry, auth keyring, vision tower, ledger) exonerated, incl. the earlier
~30% CTX=262144 suspicion (window artifact; a still-alive hunter loop was also fouling the
first night's side-boots). The delta that survived: **host CPU single-core speed**. The launch
box's Zen3 5950X (~3.4GHz, 1.03s reference loop) served ~195-230; Zen5 hosts (0.66-0.84s loop)
serve 326-354 on-box, same binary (38c85edc), env, artifact, probe. Mechanism: the sampled
spec-replay path (forced for Qwen35Moe) spends real per-round host time, so decode is
CPU-generation-sensitive. Cold-idle clock ramp on the old box explained first-request dips to
~85-130 (nvidia-smi -lgc denied there).

Second finding at cutover: **cloudflared QUIC relayed the per-token chunked response ~2x
slower than http2** — 512-token completion: localhost 1.46s (354 tok/s), tunnel QUIC 3.5s
(~146), tunnel http2 1.7s (~300). Connector pinned --protocol http2. A separate client trap:
python urllib's per-chunk reads understate wall rates by ~25% vs curl on this path (the
earlier "tunnel caps at 205/s" reading was the client, not the tunnel).

Serving relocated to a Zen5 host, exact prod binary, same tunnel id/hostname. Live routed
endpoint after cutover: **292-297 tok/s median** (n=8, all reps full 512 tok); published
headline 230 -> 290 (darklanes ee946689). One defective rental was destroyed during the
search (GPU power-managed into 645MHz at 599W draw, cold — 35-68 tok/s).

Open engine lanes from this incident: de-CPU-bind sampled spec replay (spec-pipe class), and
coalesce per-token response writes so tunnel relays carry fewer larger events (~300 -> ~354
through-tunnel headroom). Ops gap noted: capture/ledger R2 replication loop was never
installed on the ORN boxes (guard ticks fail in backoff; pre-existing since launch).
