# step37 draft-graph lane: the dcw door, and where capture actually is

Lane: make the speculative-decoding DRAFT chain graph-captured on Step-3.7-Flash.
Named blocker in the brief: `fa_decode_dc` cannot express the MTP block's SWA view offset
(the boot WARN "step35 has no captured draft chain ...").

Box: the rented dev box (2x RTX PRO 6000 Blackwell), model `/root/models/step37-flash-nvfp4`,
prompt curve-0400 (real 613-token chat payload). Code: memra branch
`lane/step37-mtp-masked-vocab-20260825` commits c460c747d9 + 2eb05f35cc + b22b411697
(box twin: branch `lane/step37-draft-dcw-20260829`, tip dfe718016).
Binaries fingerprinted by strings, never cargo's Finished line:
server md5 e42d389d01bac504833368925fbb64d9 (dcw_flag=2, arm_dcw=1),
run-spec md5 637d3b51826bad537a63a65047b50643.

## 1. The fix (landed, default OFF)

`MEMRA_STEP35_DRAFT_DCW=1` routes the step35 MTP draft attention through the WINDOWED
device-counter family the step TP graph arc already built (`append_kv_quantized_dcw` +
`fa_decode_dcw`): write slot, key bound and SWA view offset derive entirely from device
state (`len_d`, a new `KvLayer::base_d` ring-base mirror written only at host-side
rebases inside `prepare_kv_append`, and the block's window 512). bucket = min(cap,
window), so the capture-time grid covers every replayed len (one-partition law). BOTH
draft modes run the one launcher, so graph-vs-eager draft parity holds by construction.
Ring headroom for captured appends is pre-armed host-side at capture and round start
(`MtpScratch::ensure_dcw_headroom`); the eager arm keeps a per-step prepare. The cap-side
FFN now reads the step35 SHEXP clamp exactly like the dev arm; round-stream stays refused
(its verify has no step35 twin). FLAGS.md row in the same commit.

## 2. Gate receipts (raw/ in this dir)

GATE A2, run-spec greedy identity, MEMRA_MTP_HEADS=1, K=1..8, NGEN=160, curve-0400
(`raw/dcw-battery2.txt`, logs `raw/dcw2-runspec-*`):

- off arm: `capture_warn_count=8` (the brief's WARN, verbatim), receipt
  `[mtp-geom] arm=eager`, SELF-CONSISTENCY PASS at every K.
- dcw-graph arm (door on, capture live): `capture_warn_count=0`, receipt
  `[mtp-geom] arm=dcw`, SELF-CONSISTENCY PASS at every K. THE CHAIN CAPTURED.
- acceptance per K is IDENTICAL across off / dcw-eager / dcw-graph
  (78/81, 89/144, 90/207, 90/276, 90/345, 90/414, 90/483, 90/552): drafts are
  bit-identical across arms, capture is a pure latency door.
- serving-policy twin (K=3 PMIN=0.5 PMIN0=1): acceptance 77/85 = 90.6% in all three
  arms, PASS; WARN 1/0/0.
- heads=3 twin (battery 1 partial, `raw/dcw-battery.txt`): off and dcw-eager arms
  PASS K=1..8 with identical acceptance (110/153 at K=3 etc.); no WARN in either arm,
  see finding 3a.
- illegal=0, sentinel87=0 in every cell of every battery.

CELL C2, vendor-default SAMPLED serving (the product shape: NO sampling params;
temp 0.5 / top_p 0.9 applied by the server, `chat_defaults=Some(0.5)/Some(0.9)`),
serving spec policy (SERVE_SPEC=1 K=3 PMIN=0.5 PMIN0=1), interleaved x5, one boot per
cell, engagement from `usage.spec` in the response body per boot, thinking-model hygiene
in the probe (empty completion INVALID, loop rows excluded):

| arm | tok/s median | min | max | spread |
|---|---|---|---|---|
| h3-off (QUALIFIED shipping shape) | 142.66 | 123.50 | 149.19 | 25.69 |
| h3-on | 136.67 | 121.58 | 147.56 | 25.98 |
| h1-off | 126.78 | 122.51 | 133.77 | 11.26 |
| h1-on (captured-capable) | 128.37 | 125.93 | 132.51 | 6.58 |

Pairwise per round: h3 on-minus-off median -1.63 (values -21.08, +2.75, -6.88, +5.58,
-1.63), inside a 25 tok/s per-arm spread: the door is measured NEUTRAL at the shipping
head count. h3-off beats h1-on in 4 of 5 rounds. `usage.spec` engaged (rounds>0,
accepted>0) in all 20 boots. NOTE these absolute numbers are the MERGE branch
(origin/main merged 2026-08-28) and are not comparable to the 92.13 qualification number
taken on the pre-merge lane tip; only the arm-relative deltas are claims.

## 3. The findings that reshape the brief

3a. THE SHIPPING CONFIG NEVER ATTEMPTS CAPTURE. Step-3.7-Flash carries THREE MTP heads
(`[mtp-draft] embedded chain: heads=3 blocks=45..=47 scratch=per-head`), and the
qualified serving config is MEMRA_MTP_HEADS=3. `graph_draft` carries an
`mtp_extra.is_empty()` conjunct, so at heads=3 no capture is attempted and no WARN is
printed (verified: zero WARN in every heads=3 log, server and run-spec). The brief's
boot WARN belongs to 1-head boots only.

3b. THE VENDOR-DEFAULT SHAPE EXCLUDES THE SAMPLED GRAPH BY EXACTNESS LAW. The sampled
draft graph is pure-temp-only (the in-graph draw is gumbel-max over the RAW softmax;
composing it with a filtered accept test is the unconditional-accept exactness bug the
graph-s-key lane closed). step37 vendor defaults are temp 0.5 / top_p 0.9, so even at
heads=1 the captured chain never launches for a vendor-default request: all 20 C2 boots
show `capture_warn=0` because capture is never even attempted on the sampled path.
The regime where this lane's capture IS live today: greedy requests at heads=1
(the exactness-instrument shape), proven in Gate A2.

3c. THE 3-HEAD CHAIN IS WORTH MORE THAN CAPTURE. h3-off 142.66 vs h1-on 128.37 median:
trading the 3-head chain for a captured 1-head chain LOSES ~10%. Acceptance at K=3:
heads=3 71.9% vs heads=1 43.5% (full-chain sweep), 90.6% under the serving PMIN policy.

## 4. What "captured draft chain in the shipping config" still needs (costed)

1. IN-GRAPH FILTERED SAMPLING (prerequisite, removes 3b for penalty-free requests):
   the eager sampled draft already computes `filter_stats` + `gumbel_perturb_filtered`
   entirely on device per row; capturing them as nodes (th/z/mx device intermediates,
   seed/temp baked, sctr in g_ctr) widens the sampled graph beyond pure-temp. Work is
   capture wiring + `SampledGraphKey` widening + seeded-reproducibility and accept-q
   gates. Engine-lane sized: about a day of agent time, GPU-hours dominated by the
   run-spec sampled battery.
2. MULTI-HEAD CHAIN CAPTURE (removes 3a): the chain is step-modulo with PREFIX-REPLAY
   and per-head scratch, so a captured form needs either per-(head, T) graphs
   (K x heads small graphs) or a T-padded fixed-shape twin, plus the per-head kv_fill
   at round boundaries staying outside the graphs. This is the genuinely large piece.
   Value bound before paying it: at ~140 tok/s serving, a round is ~12-20 ms and the
   eager 3-head chain costs a few ms of it; capture recovers launch/glue latency only,
   so the ceiling is single-digit percent. Measure the eager chain's wall share first
   (MEMRA_SPEC_ANATOMY on the serving shape) before opening the lane.
3. The dcw door itself is the kernel prerequisite for BOTH and is landed and gated.

## 5. Door decision

MEMRA_STEP35_DRAFT_DCW default OFF, serving env does not set it. Reasons: at the
shipping head count it is measured neutral (median pairwise -1.63 inside a 25 spread)
and enables nothing there (3a); at heads=1 it enables capture for greedy shapes only
(3b) and heads=1 loses to heads=3 (3c). The door ships as the kernel seam the two
follow-up lanes build on, with its exactness receipts already banked.

## 6. Shared-box incident (paid for, documented)

Battery v1 rebuilt the SHARED checkout binary in place and broke the sampled-quality
lane's md5 pin (their gate caught it, 22:20Z). Restored: commits parked on
`lane/step37-draft-dcw-20260829`, checkout reset --keep to 8695bdef4, binary rebuilt
(md5 07b49d66ceb880b7e9a640271d6c45e8, NOT bit-identical to the original f45c36...;
strings fingerprint verified equivalent), incident note appended to BOX-README.txt and
/root/sq/DCW-BINARY-NOTE.txt. Battery v2 runs private /root/dcw-* binaries and takes
the GPU lock per cell. Also killed one ORPHANED sq memra-server (PPID 1, zero
connections, GPU 0%) whose inherited fd held the gemmprime lock and had starved both
lanes for 1.5h; documented in the same note.
