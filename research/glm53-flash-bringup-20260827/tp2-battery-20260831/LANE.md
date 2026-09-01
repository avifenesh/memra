# glm5 TP-2 box battery (lane/glm5-tp2-battery, 2026-08-31)

The stage-5 box arm of `tp2-20260831/LANE.md`, pre-registered here BEFORE the window ran.
Box: the glm53 second 4-card (4x RTX PRO 6000 Blackwell Workstation, 96 GB each — identity
stays in the private ops repo). Build pin: **2c9e7fff6** (lane/glm53-flash-bringup
merge-forward head: TP-2 + decode-diet + verify-batch + hyper-batch, TP compositions
fail-closed) + this branch's probe-bin commit (bin + scripts only; the engine tree is
byte-identical to 2c9e7fff6 — verified by `git diff 2c9e7fff6 --stat` in the receipts).

## Topology receipt (gates the whole window)

`nvidia-smi topo -m`: ALL pairs `NODE`, single NUMA node 0, CPU affinity 0-191 — this box
class has NO strong/weak pair asymmetry (unlike the hy3 0-1/2-3 class). The topology law
(TP-2 inside a pair, never across a weak edge) is satisfied by ANY pair; pair 0,1 chosen.
Full receipt + P2P capability probe banked in `receipts/topology/`.

## The instrument, stated up front

The serving worker REFUSES `MEMRA_GLM5_TP` by design (worker.rs:7604; serving wiring —
per-session TP admission/rollback — is stage-5 item 6, deliberately NOT this window). So:

- Every arm of every comparison runs through ONE engine-level binary,
  `glm5-tp2-box-probe` (the card3-probe program class: `prime_cache` + `decode_step`,
  host argmax, per-step walls). Arms differ ONLY in env (probe_arm.sh owns the tables).
- One SERVED PP-3 boot (serve.sh, the exact flip-battery recipe env) + run_pool timed
  ties the engine instrument to the banked 35.41 tok/s served baseline — the
  instrument-offset receipt. No TP number is ever compared to a served number without
  this offset row next to it.
- decode tok/s = steps/(sum of step walls after the first emitted token) — the streamed
  `(ct-1)/(t_last-t_first)` estimator shape. 128-token floor applied BY NAME (agg.py).
- Vendor-default sampled twin: temperature 1.0 / top_p 0.95 (the artifact's
  generation_config.json), seeded host sampler. A traffic-SHAPE twin, never a
  serving-sampler parity claim.

## Cells (in order; any silent-wrong signature = STOP)

1. **Real-artifact class gate.** TP-2 (`all@0,1`, host-canonical v1 transport) vs
   single-card plain (CVD=0, MOE SLRU posture — the card3/attribution program) on
   4 decode prompts + WARM: decode t=1 BYTE compare (200-token tape + first 8 step
   logits, full vocab f32) + prime last-token logits. Bar per the two-regime law:
   decode t=1 byte identity expected, BUT the lane predicts sharded-vs-plain f32 cuBLAS
   sites (b_proj [64,4096]->[32,4096]) may put even decode into the near-tie class on
   the real geometry — MEASURE FIRST, calibrate the band only if bits genuinely differ,
   and run the swap-wo RED arm on the real artifact so reds land orders louder.
   Output-sample gate (fluent, loop-law screened) per boot.
2. **Transport receipt.** v1 ships host-canonical ONLY (`configure_native_p2p` inherited
   but not wired to the glm5 seam — LANE stage 3 "NOT built, deliberately"). This cell
   banks the host-canonical cost inside the cell-3/4 numbers and NAMES the native-P2P
   engagement A/B as the follow-up arm. P2P capability of the box is probed and banked
   (topology receipt) so the follow-up lane knows the transport is available.
3. **Join-cost row (measurement only).** Measured TP-2 ms/token minus the decode-gap
   attribution terms (bandwidth/2 with the EP-2 1.57x haircut, measured latency class
   from the PP-3 twin) = the measured join+overhead per token, priced against the
   table's assumed ~1-2 ms. The diet DOORS (direct join / prestage / prejoin-overlap)
   are NOT built in v1; the ladder is the named follow-up, sized by this row.
4. **BARE TP-2 PRICING** (TIMED, marker up, interleaved x3 fresh boots per arm, x5 on
   anomaly): TP-2 vs PP-3 recipe, both engine-level; decode pool + l3 deep rows
   (TTFT 0.4k/3.7k); one vendor-default sampled row per boot; engagement receipts
   ([glm5-tp-preflight/kda/mla/ep] on TP boots, zero TP lines on plain). Single-card is
   NOT an arm: one 96 GB card cannot hold the 192.5 GB artifact resident (SHARD-MAP §7);
   stated, not measured. Priced against the decode-gap table: TP-2 pre-diet ~42-43 tok/s.
5. **TP-2 + PP-2-across-pairs composition** — ONLY if TP-2 beats PP-3: the composition
   is REFUSED at preflight in v1 (merge-forward matrix); lifting it is a gated engine
   increment, scoped honestly if the gate fires.

## Protocols

TIMING-IN-FLIGHT raised for every timed cell; engine probes are foreground processes
(no orphan risk); the served boot uses the pidfile+exe-scoped stop. Rebuild-attribution:
`git log -1`, build wall, binary sha256, strings probe for the TP announce literals
(`[glm5-tp-preflight] armed`, `[glm5-tp-kda]`, `[glm5-tp-mla]`, `[glm5-tp-ep]`) in every
receipt. Receipts scrubbed of box identity before push (public repo boundary).
