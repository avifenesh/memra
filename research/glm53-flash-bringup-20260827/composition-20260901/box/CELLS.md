# Composition-lane box cells (pre-registered; runs when hardware lands)

DOOR PINS ON TP ARMS (moe-loc §4.5 vacuous-green law: pin, never leave unset): the
instrument-continuity arms pin `MEMRA_MLA_TC_PREFILL=0` (its default flipped ON after the
tp2-battery banked its rows) and the diet doors `MEMRA_GLM5_EP_DIET=0
MEMRA_GLM5_EP_GROUPED_PRIME=0` unless the arm names them ON.

READ THE TC RECEIPT RIGHT (#82 review): the `[mla-tc-prefill] DECLINED on glm5-TP head
shards` line only prints when the flag is ON and the shape reaches the chain — so on the
PINNED `=0` arms it CANNOT appear, and its absence there proves nothing. To observe the
decline, run one deliberate arm with `MEMRA_MLA_TC_PREFILL=1` on a TP boot at a prefill
width (t >= 16, real artifact kv_rank 512): the line is the receipt that shards fall
through to the f32 kernels. The TC-prefill x TP composition gets its own band-gated box
arm before any TTFT claim rides it.

Protocol laws bound at lane open (LANE.md "Pre-registered protocol"): interleaved x3 fresh
boots (x5 on >0.5% spread), boot-nonce arm identity, real prompts (sxc pools), capped
max_tokens, loop-law screening, `git log -1` in every receipt, doors-OFF arms pinned `=0`,
spec engagement receipts from the log (`route=spec K=` / `[glm5-acc]`), never boot-trust.

## 2-card window (any 2 free cards; the interim box)

The composed-route MULTIPLIER evidence — every cell here transfers to the 4-card arithmetic.

- W2-G0 fixture gate: `glm5-tp-gate 16 12` on the box (real fabric; the rig's same-device
  emulation cannot prove fabric engagement). MUST be ALL ARMS PASS before any timed cell.
- W2-G1 real-artifact TP-2 class gate (tp2-battery cell-1 shape): tape mode, teacher-forced
  vs single-card reference tapes; byte bar non-MoE, measured band EP-MoE (3-5e-2/layer
  saturating), argmax+margin on deep primes. BOTH transports. STOP rule: any decode
  divergence not orders-below the red class = SILENT-WRONG-SUSPECT, window stops.
- W2-T transport re-price (the tp-transport lane's named box window): timed decode rows,
  arms {host-canonical, peer-pull} x {diet=0, diet=1}, interleaved x3. Anchors: the banked
  22.65 tok/s v1 row must reproduce on the =0 arm (instrument continuity). Prediction under
  test: TP-2 peer-pull 29.0-35.7 engine twin.
- W2-S spec x TP-2 composed rows (THE NEW CELL, needs MEMRA_GLM5_SPEC_TP=1 +
  MEMRA_GLM5_DFLASH=<drafter> + MEMRA_GLM5_SPEC=1): engine-twin spec tok/s + acc/cycle +
  round-wall fit a+bK at K in {1,2,3,5}, greedy + vendor-default sampled twin, vs the same
  box's plain TP-2 rows (W2-T) and vs the unsharded 1-card spec rows for the multiplier.
  Acceptance must be byte-identical to the unsharded acceptance (1.839/2.458/2.907 tok/cyc
  at K=1/2/3 on the deployed head — the walk moves time only, never acceptance).
  KNOWN COMPOSED-SHAPE WALL (pre-registered so the round-wall fit is read right): on a
  sharded trunk the EP walk PREEMPTS the batched vrows MoE pair at verify widths, so the
  composed round re-inherits the sequential per-(token,expert) vrest class the vrows lane
  removed on the unsharded shape (the unsharded 11.2 ms/K fit does NOT transfer). The boot
  receipt is the `[glm5-tp-ep] verify rows ride the SEQUENTIAL EP walk` line; the named
  lever is an EP-AWARE vrows arm (per-rank grouped pairs over the EP slabs).
- W2-D doors-on-composed: T/X/K/W (+D/H explicitly =1) engagement counters on the composed
  shape (doors fire on the verify walk's t=2..16 shapes; bare TP decode is vacuous by
  construction — matvec lane §4). Pinned-=0 OFF arms.

## 4-card window (the owner's OD box, when granted)

- W4-G0/G1: fixture gate + real-artifact class gate at `all@0,1,2,3` (both transports, red
  arms, ep-map arms with a 4-rank mint via tools/build_expert_placement_map.py --ranks 4).
- W4-B TP-4 base decode rows: {host-canonical, peer-pull} x {diet 0/1}, interleaved x3.
  Prediction under test: TP-4 peer-pull 30.3-36.9 engine / 36.1-48.7 served-class = only
  1.03-1.05x over TP-2 (driver-primitive bound). This number DECIDES the composed route.
- W4-S spec x TP-4 composed rows: as W2-S on the 4-rank shape.
- W4-R 262k residency: EP-4 sharded experts (~43 GB routed bank per card) leave room for
  the 262k KV + workspaces — boot at MEMRA_CTX=262144, prime a ~250k real prompt (prefill
  wall expected ~6 min at the flat ~700 tok/s class; grouped EP prime
  MEMRA_GLM5_EP_GROUPED_PRIME=1 arm A/B), then decode rows at depth. VRAM-at-ready CSV per
  card banked.
- W4-BAR the bar battery: composed shape at 262144, interleaved x3 (x5 on spread), greedy
  instrument + vendor-default sampled + 8-turn larger-prompt twin (glm5 prefix cache is
  structurally dead — the twin measures the full-re-prefill reality and says so), per-turn
  TTFT + accept. VERDICT vs the 100 tok/s single-stream bar stated plainly.
  NOTE the serving-shape caveat: the memra-server worker still refuses MEMRA_GLM5_TP
  (serving wiring = a named increment this lane takes only if the engine-twin numbers
  justify it); until then the bar battery's serving-shape rows ride the engine twin +
  the tp2-battery's instrument-offset calibration, and the verdict says so.

## The arithmetic the cells settle (banked predictions, quoted)

- best single-stream today: 71.49 tok/s (3-card PP3 + spec + doors + D/H, struct-battery)
- TP-2 v1 bare: 22.65; TP-2 peer-pull predicted 29.0-35.7 engine / 34.5-47.1 served-class
- TP-4 peer-pull predicted 30.3-36.9 engine / 36.1-48.7 served-class (1.03-1.05x over TP-2)
- spec multiplier measured on ppN: 1.29x (45.65/35.41); public TP4 prior: 1.64-2.33x
- doors measured: 1.1288x (T+X+K+W) x 1.0154x (D+H) ~= 1.146, NOT the charter's 1.25
- composed candidate: TP-4-pp x spec x doors — if TP-4 base lands <= ~50 engine, the
  composed route cannot reach 100 with the measured multipliers and the verdict is
  REFUTED-as-composed with the named unlocks (CUDA-graph capture of the peer copies,
  fused pull collective, batched-EP dispatch diet at prime).
