# Box ctxprobe: the ring fix verified on serving hardware (vast 4-card, 2026-08-29)

The probe the FLAGS row named as owed: lane/glm53-box's three-arm ctxprobe re-run on a binary
carrying the fix. Rented 4x RTX PRO 6000 Blackwell Server 96 GB box; no box identifiers in
these receipts.

## Setup, laws honored

- Binary built on the box in a detached worktree at the consolidated head f929dda914
  (origin/lane/glm53-flash-bringup: my fix 52732cef75 verified an ancestor, the 1M lane's
  chunked mHC prime and the prefix-restore latent guard both in history). `git log -1` and a
  binary-newer-than-every-source assertion in `00-verify-binaries.txt` (LAW:
  rebuild-after-checkout-attribution).
- Every arm booted with MEMRA_PREFIX_CACHE_MB=0 pinned (defence in depth against the
  latent-plane restore defect; the pin is printed in each arm's boot receipt and was verified
  on the live pid's /proc environ).
- PID-verified stop between arms, never pkill (serve.sh).
- The PRE-RING arm is UNAVAILABLE on this box and stated rather than substituted: the only
  candidate binary (~/memra-r2, lane/glm53-pp tip 876959009) carries the ring UNFIXED
  (strings: ring-flag=3, pre-fix guard msg=1, post-fix msg=0; f7ec IS its ancestor). It serves
  instead as the RED-reproduction arm. The 08-28 prior-box receipt (7312 tokens) remains the
  pre-ring reference.
- ~/memra on the box is another lane's checkout (branch ab-epi-x-pp, 7 unpushed commits, fix
  NOT an ancestor) and its prebuilt binary carries ZERO post-fix strings despite the handoff
  describing it as the consolidated head. Strings census caught a wrong-binary handoff for the
  second box in two days; nothing in ~/memra was touched beyond `git fetch`.

## The three-arm verdict (ctxprobe.py byte-identical to lane/glm53-box's, MEMRA_CTX=8192)

| prompt tokens | MERGED ring ON (default) | MERGED ring OFF (=0) | UNFIXED ring ON (r2) |
|---|---|---|---|
| 940 .. 4630   | 200 | 200 | 200 |
| 5550          | 200 | 200 | **500 tail ring lapped: 5120 rows** |
| 6470          | 200 | 408 deadline (see below) | 500 same |
| 7300          | 200 | 408 deadline | 500 same |
| **largest SERVED** | **7300** | 5550 | **4630** |

- **THE FIX VERDICT: ring ON at the merged head serves the full configured window, 7300
  tokens, zero failures, zero ring-guard errors.** Identical row-for-row to what ring OFF and
  the pre-ring binary served on 08-28 (7300/7312). The regression is closed on serving
  hardware through the product surface.
- **The RED reproduced in-session**: the unfixed binary caps at 4630 with the same
  `indexer tail ring lapped: 5120 rows` error, cache pinned off, so the 08-28 finding was the
  ring and only the ring.

## The ring-OFF 408s are a DEADLINE, not admission, and they inverted the rollback assumption

The OFF arm's failures are `deadline of 90000 ms ... elapsed`, never a refusal and never a
ring error; a streamed 5550-token request on the OFF arm completed at 186.8s, so admission is
fine on every row. What the timing shows (ttftprobe.py, streamed TTFD, A,B,B,A boot
interleave):

| prompt tokens | ring ON (boot 4) | ring OFF (boots 2 and 3) |
|---|---|---|
| 4630 | 57.4s | 90s deadline / (served in-ladder earlier) |
| 5550 | 67.1s | 186.8s |
| 6470 | 78.8s | 90s deadline |

Same binary, one env flag, both ON boots fast and both OFF boots slow: on this box the
FLAT plane's prefill is ~2.4x slower than the ring's at these sizes and blows the 90s
platform deadline from ~6.5k tokens (nonstream). N=2 per arm on a rented box, so this is an
observed direction, not a tuned perf row; but the operational consequence stands either way:
**MEMRA_DSA_INDEX_RING=0 is no longer the safe arm.** A rollback "for safety" now costs
usable context through the product surface on this hardware class. Attribution of the OFF
slowdown (suspect: a scorer/dispatch path keyed on the ring'd plane) is engine-lane work,
not this probe's.

## Files

| file | what |
|---|---|
| 00-verify-binaries.txt | git log -1, freshness assertion, strings census of all three binaries |
| 01-build.log | on-box build at f929dda914, rc=0 |
| 02-ctx-arms.txt | the three arms, boot receipts inline (prefix-cache pin visible) |
| 04-ring-off-warm-retry.txt | OFF arm re-boot + warm-up: the 408s reproduce warm |
| 05/06-*-timing.txt | streamed TTFD rows, OFF vs ON |
| ctx-*.json | per-row structured results, all four runs |
| serve.sh / run-arms.sh / ctxprobe.py / ttftprobe.py | the harness that ran, byte-for-byte |

## The wall ladder: no OOM anywhere, and the wall is the deadline, not memory

One boot, merged head, ring ON (default), MEMRA_CTX=262144, streamed rungs 8k/16k/32k/64k/
128k/250k (03-wall-bracket.txt, wall-262k.json, 07-wall-no-oom.txt):

- **ZERO CUDA-OOM at any rung.** The serve log has zero out-of-memory/panic matches after
  the full ladder; the server was alive at close; VRAM peaked at 85 GB on the serving card
  with the 262k planes resident. The 08-28 failure class at ~50k
  (`CUDA_ERROR_OUT_OF_MEMORY`) did not reproduce: on the pre-chunked binary the monolithic
  prime's whole-prompt transient failed fast at allocation; on the merged head nothing
  fails fast, which is the chunked prime doing its job.
- **Every rung, including 8000, ended as `408 deadline of 90000 ms ... before the first
  token`, at exactly 90s.** The 90s first-token bound is a platform contract
  (`TIMEOUT_MS_MAX`, "we answer inside 90s or you don't pay") and it binds streamed
  requests too, so on this box config (experts host-resident, MEMRA_MOE_RESIDENT=0,
  single serving card) the product surface cannot carry a prompt whose prefill exceeds
  ~90s. Measured prefill walls (ring ON): 57.4s at 4630 tokens, 67.1s at 5550, 78.8s at
  6470, deadline at ~7.4k. **The honest ceiling through the product surface on this box is
  ~7.5k prompt tokens, and it is a prefill-throughput number, not a capacity number.**
- Capacity above ~7.4k processed tokens is therefore UNTESTED through the product surface:
  the deadline cancels prefill before memory can be exercised. What IS banked: the 262k
  boot allocates and holds its planes (85 GB, stable, alive), and no rung produced an
  allocation failure while it ran.

For the model card this means the context statement is currently bounded by serving
prefill throughput per box config, not by the engine's memory plan. Raising the ceiling is
a serving-config and prefill-throughput question (expert residency across the idle cards
is the obvious first lever on this class), which is other lanes' work.

Also observed, pre-existing: `[admission] request cost ... = 0 B/token x ctx + 0MB fixed`
for this model, i.e. per-token admission cost is zero. Consistent with the 08-28 finding
that nothing enforces the admission limit before prefill; serve lane's item, unchanged.

Note on scrubbing: one phrase of this lane's own script echo named the prior box's cloud
provider; it is scrubbed to "prior-box" in run-arms.sh, the banked 02-ctx-arms.txt, and this
document (public-boundary rule: fleet placement stays in darklanes). No measured value was
touched; the receipts on the box retain the original bytes.
