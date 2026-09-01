# What the composition lane transfers to a TP-2-NVLink B200 shape

Written at lane close on the owner pivot (2026-09-01): the qualified 3-card PP3 shape is
the serving shape and the 100@262k bar is retired as a launch gate; this lane's verdict is
re-framed from launch gate to UPGRADE path; the follow-on arc is 2xB200 bring-up + 1M
support. This doc is the load-bearing handoff for that arc — what carries over as
machinery, what carries over as LAW, and what must be re-measured because it is a number,
not a structure. (Reviewer-hardened: a first draft of this doc carried five claims that
contradicted the banked tree; every correction below names its receipt.)

## Re-framed verdict (the upgrade-path reading)

Composed spec x TP on the current 2-card PCIe class prices at **29.684 tok/s** (engine
twin) against the shipped shape's battery rows — spec 45.654 greedy / 46.66
vendor-sampled (flip-reprice), best single-stream 71.489 greedy / 61.82 vendor-sampled
(struct-battery dhon arm; the two are the SAME arm's greedy and sampled medians, never a
greedy range). **There is no blue/green candidate in this lane's numbers.** LIKE-FOR-LIKE
LAW when pricing any upgrade: those are INSTRUMENT rows; the shipped shape's SERVED class
is banked separately (tp2-battery: served PP-3 35.36-35.42, and the explicit trap that
engine twins cannot price PP arms — the eager driver reads 0.254x of served). Compare
served against served, instrument against instrument.

The upgrade levers with receipts behind them: the **EP-aware vrows arm** — which helps
SHARDED shapes only (the sequential vrest wall exists exactly where the EP walk preempts
the batched vrows pair; the shipped PP3 shape has no EP walk and already rides the batched
fit 33.5+11.2K ms, so this lever does NOT move the shipped rows) — CUDA-graph capture of
the decode round, and the batched-EP prime. `MEMRA_GLM5_SPEC_TP` stays default OFF.

## Transfers WHOLESALE (machinery, fabric-agnostic by construction)

1. **The transport seam.** `tp_transport` (general since extraction PR #77) is the direct
   B200 asset: named hop shapes (fanout / gather / concat / block-returns), one pub + one
   release event per rank, and the all-ordered-pairs byte-integrity arm ladder. On NVLink
   the same copy-engine `memcpy_dtod` calls ride the new fabric with zero code change,
   and the ladder is the FIRST thing to run there — stated honestly: the ladder has only
   ever PASSED (PCIe real-fabric receipts, mismatches=0); its refusal path has never
   fired outside unit scope (extract2 notes it as the one refusal site never
   parameterized), so treat it as a necessary check, not a proven fabric gate, and pair
   it with the `simpleP2P`-class kernel peer-read probe as on PCIe.
2. **KEEP PULL, for BOTH banked reasons.** The ordering rationale (a consumer-issued copy
   is ordered against the consuming kernel for free) is fabric-independent and remains
   the headline for the copy-engine path. The PCIe SM-issued read/write asymmetry was
   never the copy-engine reason — per `tp_transport.rs` it is the standing reason a
   future FUSED collective must be pull-shaped and the push twin is never built. NVLink
   symmetry for SM-issued traffic is UNMEASURED in this repo: the never-build-push law
   stands until someone measures it, and even then the ordering argument alone still
   favors pull.
3. **The rank envelope.** TP-2 IS the original qualified envelope — a 2xB200 shape needs
   zero rank work, and TP-4 exists gated if a 4xB200 host ever appears. `glm5-tp-gate`
   runs on any card the engine compiles for (the composition arms S2/Q-S4/SF1-4/SW/SD
   included) — but see item 9: B200 does not compile today.
4. **The spec x TP wiring — engine-level only.** Model-truth admission, per-rank
   verify/rollback (`kda_tp_core` rows arm, per-replica MLA truncation), the
   ARMED-at-return receipt: all keyed on the model's shards, never the fabric. What does
   NOT exist anywhere: **TP serving wiring** — the memra-server worker refuses
   `MEMRA_GLM5_TP` outright at spawn (FLAGS.md names it the unbuilt box increment). Any
   1M-on-TP-2-NVLink SERVING plan must budget building that increment; it is not a
   transfer, it is new work.
5. **The EP dispatch diet.** It cuts hop COUNT; the placement-map multiplier (a rank with
   no routed experts moves zero activation bytes) is fabric-independent. How much the
   diet is worth on NVLink is unknowable from here — see item 8's cost-model honesty.

## Transfers as LAW (structure, not numbers)

6. **The TP walk is dispatch-structure-bound, not byte-bound.** The lane's decomposition:
   TP-2 host-canonical 50.6 ms/tok -> peer-pull 43.7 -> +diet 38.8. The banked tax law
   (`tp_transport.rs`): the movement tax is a ROUND-TRIP and LAUNCH-COUNT tax, not a byte
   tax. The per-token launch count and the per-slot sequential EP walk inside the 38.8 ms
   residual are fabric-independent terms NVLink cannot touch. **Land the EP-aware vrows
   arm before pricing any COMPOSED-SPEC cell on B200** — it is the biggest term on the
   composed round (and only there; see the re-framed verdict for why it does not move
   the shipped shape). Plain decode shape-selection rows do not need it.
7. **The vrows preemption transfers unchanged.** `moe_ffn_inner` returns through the EP
   arm before the batched vrows pair is consulted, on any fabric. The
   `[glm5-tp-ep] verify rows ride the SEQUENTIAL EP walk` announce is the receipt; the
   unsharded 33.5+11.2K ms round fit does not transfer to any sharded execution until
   the EP-aware arm lands.
8. **There is NO banked NVLink cost model — refuse to inherit one.** Zero NVLink
   measurements exist in this repo. Do not size a B200 transport budget from a bandwidth
   ratio: the tax is round-trip/launch-count-shaped, and how much of the PCIe hop cost
   (the ~11.8 ms/tok the transport levers recovered, plus whatever host-boundary residue
   remains) survives on NVLink is exactly what the first interleaved A/B on the pair
   exists to measure. A predicted number here would be the same class of miss this
   lane's own transport prediction was (predicted 29.0-35.7 engine twin, measured
   22.902 — banked as a miss).
9. **Acceptance parity is an open gate, and it is shape-keyed, not fabric-keyed.** The
   same-box control showed one-card acc 0.582 vs sharded 0.640 (part of the 1.151x
   composition multiplier is acceptance, mechanism not established). Any B200 sharded
   serving arm owes the acceptance-parity gate before its spec rows are read as
   time-only.
10. **Prime near-tie bands are per (rank count, shard shape, kernel arch).** The 2-rank
    band (2e-4, worst 4.85e-5) and the quad band (4e-3, worst 4.013e-4) are calibration
    rows for THIS kernel arch; recalibrate on the first B200 run with the banked
    10x-over-worst procedure. Decode BYTE identity stays the bar at t=1 regardless of
    arch.

## Does NOT transfer (re-measure, re-gate, or BUILD — never assume)

11. **Every multiplier.** 1.159x (peer-pull), 1.126x (diet), 1.305x (together), 1.151x
    (spec on TP), 1.012x (bare TP-2 over one card) are PCIe-2-card, PRO-6000-class,
    host-ST-caveated numbers.
12. **The build itself — CORRECTED 2026-09-01 by the b200-prep census (this item's
    original text is superseded; the census is authoritative:
    `research/glm5-b200-prep-20260901/LANE.md`, "The SM100 native-FP4 question,
    settled").** Three of this item's claims aged out or were never quite right:
    (a) sm_100a NOW COMPILES — the polarity bugs named here were already fixed
    2026-08-23; the actual last blocker was `cu/mmq_q8_0_f32acc.cu`'s
    `__CUDA_ARCH__ >= 1000` guard, fixed by lane/glm5-b200-prep-20260901 (PR #91,
    merged to main 2026-09-01); the 29-cell census is green and ci.yml carries a
    100a compile cell.
    (b) The fatbin census now passes: the two absent lookups are DECLARED exceptions
    with property-guarded call sites (`MEMRA_FP4` door + kernel_check Stage-C).
    (c) NVFP4 experts are NOT "routed through the accuracy-safe W4A8 int8 path" on
    sm_100a — `cu/mmq_nvfp4_w4a8.cu` is a fail-closed STUB on every non-120a arch (its
    block-scale int8 form is the same sm_120a-only MMA kind). What an sm_100a build
    actually runs for NVFP4 is the dp4a decode matvec family (`qmatvec_nvfp4_dp4a_sel*`,
    plain int8 dp4a) plus cuBLAS/cuBLASLt-class prefill; B200's own FP4 tensor cores
    (tcgen05) are unreached by any memra kernel and remain a port lane. What STANDS from
    this item: the SM100 re-gate law (every decode-exact/rows-exact class contract
    re-proven on the new arch, on the path that actually executes) and the launch-diet
    census as the first instrument once it boots. `detect_arch()` still refuses to
    auto-select 100a — now because nothing has ever RUN on the silicon, not because it
    does not build.
13. **The residency premise — with the arithmetic done honestly, SKU-sensitively.** The
    artifact models 190.6-190.75 GB total (171.2 GB routed bank — which EXCLUDES the
    4.08 GB MTP bank — plus ~14.7 GB non-expert trunk/head). Against a 192 GB B200 that
    is at best ~1-6 GB of headroom BEFORE any 262k latent/indexer planes or workspaces;
    against the 180 GB SXM SKU it does not fit at all. So full single-card residency is
    not a plannable arm. The real 262k comparison on the pair is **residency-split
    PP-2** (each card carries whole layers — locality, zero per-token joins) vs
    **TP-2-NVLink** (every layer joined, both banks resident): this lane's law (joins
    cost, locality wins) predicts PP-2, but the NVLink join tax is unmeasured (item 8),
    so MEASURE both. A single-card arm exists only with SLRU'd experts, which
    reintroduces the staging wall the attribution lane priced at 84.1 ms/tok on PCIe
    hosts. **TP-2-NVLink's clearest role is the 1M arc** — at 1M the latent KV + DSA
    kpool planes outgrow any split's slack and the second card's memory is the point.
    The prior 1M receipts (1m-battery WINDOW.md: 3-card resident prime OOMs at the DSA
    k-pool, layer 31; INDEX.md: PP4 splits 13,26,39 = the only demonstrated 1M posture;
    banked in the private corpus as the 1m-needs-pp4-slru verdict) are 96 GB-card
    residency artifacts — re-derive the 1M placement from the B200 memory map, not from
    the PP4 demo.
14. **The prefix-cache caveat transfers as a model fact.** glm5_next latent planes are
    not carried by prefix entries (snapshot fail-closed), so multi-turn TTFT on B200
    still models full re-prefill per turn until the prefix-latent lane lands — fabric
    changes nothing there.

## First actions on a B200 pair, in order

0. THE ENGINE LANE FIRST (item 12) — DONE 2026-09-01: lane/glm5-b200-prep-20260901
   (PR #91, merged to main 2026-09-01) fixed the last compile guard, declared and
   guarded the two fatbin census exceptions, and added the ci.yml 100a compile
   cell; a 100a binary exists and
   stays buildable at every merge. Still owed ON THE BOX: the class-contract re-gates on
   the paths that actually execute on sm_100a (dp4a NVFP4 decode + cuBLAS prefill — NOT
   the W4A8 int8 MMQ, which is a fail-closed stub there; see the corrected item 12).
1. Arm ladder + kernel peer-read probe + `glm5-tp-gate` (fixture, all arms) — fabric and
   code sanity.
2. Launch-diet census (us/launch constant) — the number every later receipt divides by.
3. Real-artifact class gate (the W2-G1 shape: teacher-forced tapes, EP band metrics,
   transport twin byte-identity) with FRESH prime-band calibration.
4. Residency-split PP-2 vs TP-2-NVLink PLAIN decode rows at 262k-class depth (full
   single-card residency does not fit — item 13). These are shape-selection cells; they
   do not need the vrows arm.
5. IF the composed-spec question re-opens on the winning shape: EP-aware vrows lands
   first (item 6), then the acceptance-parity gate (item 9), then the composed rows.
6. The 1M residency map (where the KV/kpool planes land across the pair), then the 1M
   prime walls (grouped EP prime, chunk schedule) — the arc's actual target. If 1M
   serving rides any TP shape, the TP serving wiring (item 4) is a named build item on
   the critical path.
