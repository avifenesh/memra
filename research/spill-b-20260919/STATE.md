# WP-B checkpoint 2026-09-27 11:3xZ: integ71 merges with the doors-off program green (revuto round 2 approved 451ccd0c3).
The RW cell's R1 FAIL placed on its wording (the G=32 later turns resume through the exact path at the settle point;
all 20 resumed on both exact boots, R2 matched cold); DAY44 2.2 and addendum D; the revised reader at f7901d987; the
lead reruns c13 on the lane tip. lane/spill-b-integ71-fixes not needed after the merge; remove wt-b-fix71 then.
# WP-B checkpoint 2026-09-27 07:0xZ: the four #844 items fixed on 24bddc8cb's line (lane/spill-b-integ71-fixes at
f0b824ff5: the kv fault OOM formats without libcuda, the remove-var allowlist, DAY44 addendum C's settle checkpoint and
memory gate with the RW cell), merged into the lane (2d84b98a5); DAY48 2.1 read (V1 to V4 PASS, no measurable change).
GPU owed on the fix tree: the DAY44 mini cell, the RW cell (R1, R2), cell 12 with MEMRA_RESUME_EXACT=1.
The fix worktree wt-b-fix71 (branch fix/spill-b-integ71) goes once integ71 merges.
# WP-B checkpoint 2026-09-27 05:3xZ: integ71's GPU battery on BOX43 (e8af7cf53) all green for this lane's twelve cells
(the lead's receipts: research/spill-lead-20260919/integration-day12/integ71-pro-run1/lane-b/; serve-smoke fails only
#777's Q35 line). The lead's CRLF finding: 209 committed *.hdr captures had been stored LF under core.autocrlf=input
(90 checked against their manifests, 119 local against the untouched working copies); restored to their CRLF bytes
with research/spill-b-20260919/.gitattributes (*.hdr, *.body -text -whitespace) and listed in hdr-crlf-record.tsv.
# WP-B checkpoint 2026-09-27 05:0xZ: DAY39 2.4 (the 5090 GREEN; O5 read on both classes) and DAY50 2.2 (the 5090 stage 0:
arm O on this class too) read; DAY50 addendum C registered (arm O's census, the shadow, stage 1 next); the eighteenth
sitting (DAY48) queued on BOX43 behind integ71's battery; local: DAY49D, DAY46C, DAY48, queue-m (DAY40), chain-r5 (gates).
# WP-B checkpoint 2026-09-27 04:3xZ (NEED TARGET CARD, the eighteenth sitting, DAY48): DAY46 2.2 read (the W release
admits 11 of 32 of the second wave against 1 of 32); DAY48 coded (`5a6f1898f`, after integ71's base 24bddc8cb, so it
goes into the next integ); the DAY46C and DAY48 local runners run from wt-b-integ71; integ71's battery runs on BOX43.
# WP-B checkpoint 2026-09-27 03:4xZ: integ71's GPU cells for this lane sent to the lead (12 cells: grid_capture_gpu,
the continuation gate, the rewind probe, prime-gate, run-spec, run-gen, the health gate g,h,j and j under VMM, the VMM grow
series, the serving gates under VMM, a DAY44 mini cell with EXTERNAL_LOCK=1, admit-mem burst with the W release); tip
bdaaf311f (the gate's default arms drop i, DAY49 addendum E; day44-run.sh EXTERNAL_LOCK). The seventeenth sitting (DAY46C)
runs on BOX43.
# WP-B checkpoint 2026-09-27 03:0xZ: the integ71 merge `c2c32539e` (main caf5b7d28 into the lane, one conflict, both kept;
all suites green) pushing; DAY46 2.1 read (P1 to P5 PASS; the value reading owed, addendum C, the seventeenth sitting);
DAY50 stage 0 read (GPU-bound: arm O); the chains' manifests exclude themselves; the lead-overwritten manifests noted.
Working tree for commits: `wt-b-integ71` (branch merge/spill-b-integ71, pushed to lane/spill-b-20260919). The live
checkout `wt-spill-b` stays at 1df7e7852 while its runners (queue-m, chain-r5, DAY49D, DAY50 stage 0) read it; ff it
and remove wt-b-integ71 when they finish. The DAY46C local rerun runs from wt-b-integ71.
# WP-B checkpoint 2026-09-27 01:3xZ: the fifteenth sitting (DAY46) queued on BOX43 behind integ70's battery; DAY50 stage 0
built (`5d94e26ae`, the sixteenth sitting ready, the 5090 half running); DAY47 2.4 and DAY49 2.5 read (the 5090 halves);
the lane goes into integ71: merge main in a separate worktree once integ70's sha lands (keep both behaviors), then fmt,
clippy, the engine, kv, server and tier suites under the cap, and report the sha
# WP-B checkpoint 2026-09-27 (NEED TARGET CARD on BOX43, the fifteenth sitting, DAY46): DAY44 2.1 read (exact: E1 0
flips everywhere; E2 FAIL 8 of 24, TTFT); DAY50 pre-registered (the overlap revision, stage 0 first); DAY46 coded
(client, reader, runner, chain; `15a6ec634`); integrable at 15a6ec634 (told the lead); the DAY46 local runner started
# WP-B checkpoint 2026-09-26 21:3xZ: O14 read on the target card in full (DAY49 2.3 gates, 2.4 serving shape: on 9/9 on
both routes, off loses the chunk); BOX35 released; the ninth sitting still on BOX33; 5090 halves waiting for the card
# WP-B checkpoint 2026-09-26 21:1xZ (NEED TARGET CARD, the DAY49D boots-only rerun on BOX35): DAY49 2.3 read (arm j
PASS x4 on a 3-session chunk; j-vmm as registered); the chain's WT export fixed with a BOOTS_ONLY=1 entry (`09b15280f`)
# WP-B checkpoint 2026-09-26 (NEED TARGET CARD, fourteenth sitting on BOX35): DAY49 2.1 and 2.2 read (the fault never
reached a multi-session batched chunk; i-vmm green PASS); addendum D coded (`8926ccfb3`, `batch:<n>`, arm j); the rc
audit done (lane scripts fixed, DAY31-D4, DAY32 and DAY37 2.8 corrected); the ninth still on BOX33
- Running scripts left as they are (lead): queue-m and queue-l still carry the old `"$(date) ... rc=$?"` line on
  their aggregate 'boots rc=' lines, noted in their logs; the per-boot lines are correct. Every stopped script is fixed.
- The first commit of the twelfth and thirteenth mirrors caught an ELF (`pro-single-day49/box/bins/tip/memra-server`)
  the lead removed after my add; the push's boundary check refused it, the blob was dropped by amend and never
  reached the remote. Scan staged mirrors for ELF before every commit.
# WP-B checkpoint 2026-09-26 (eleventh sitting read; NEED TARGET CARD for the thirteenth): DAY45 2.2 every clause PASS
on the target card; DAY47 2.3 g, g-red, h, h-red PASS twice; DAY46 addendum A registered; the twelfth running on BOX35
since 17:48Z (tree 77fe12114), the thirteenth follows; the ninth still on BOX33
# WP-B checkpoint 2026-09-26 (NEED TARGET CARD, thirteenth sitting): DAY49 addendum C (the batch reclaim's VMM reap,
- 17:48Z: queue-k stopped by the lane in an idle wait (DAY39B's boots after v2-G2 went unrun: v2-L64's 7200 s idle
  wait ran out at 17:38Z and day39-run.sh stops the call there). queue-m (pid 2617798) runs queue-k's items 2 to 9 with
  the same specs and readers; a spec that did not start is asked again (up to eight times), one that started never is.
  The card has been held back to back by another lane's M1 queue since about 15:48Z; chain-r5's A2 collector has retried
  the lock every 120 s since 16:35Z and records `collector.exit=1` at its 61st try (about 18:37Z) if it stays held; A2
  would then be asked again under a new label (a not-run is not a result).
`95d35c383`) and its i-vmm cell ready for the target card (`pro-single-b-sitting13.sh`, about 45 min, can follow the
twelfth on BOX35); O1's 5090 rerun (r5) and queue-l running locally; the ninth sitting still on BOX33
# WP-B checkpoint 2026-09-26 (O14 queued): the eleventh and twelfth sittings queued on BOX35 (the lead's /root/b11-chain.sh, then /root/b12-chain.sh behind it; receipts /root/spill-receipts/b-day45b, b-day47, b-day49); the ninth sitting still on BOX33
- Local runners now: queue-k (pid 2946742, DAY39B onward), chain-r5 (O1's r5), and queue-l (pid 2114207, from
  17:07Z: DAY47 a2 a3, DAY49 g1 g2 on the 02dbdfa40 worktree, DAY49C green and red, DAY49's boots). queue-l replaced
  three idle-waiting runners (stopped by the lane, nothing run) whose `flock -w 600` wrapper would record an empty run
  when two of the lane's runners took the same free window. Lane F's M1 queue holds the card in back-to-back cells.
- DAY49 addendum C: the seven review patterns read against O14's arm found the batch reclaim skipping the VMM reap
  (the other two reclaim paths reap before their trim); fix `95d35c383`, census pins all three. Cell `i-vmm` local
  (`rtx5090-day49/run-c.sh`, worktrees `target/wt-day49c-*`, removed at its end); the target half rides the next sitting.
- Rule kept: after any engine or server commit, rebuild the lane's `target/release` under the quota at once, because
  `tools/health-fault-gate.sh` builds in its tree inside the lock hold with no quota (DAY47's and DAY49's local gates run it).
- O1's 5090 rerun registered (DAY37 addendum H, 1.17): the whole cell on the addendum-B tree r5 (`02dbdfa40`, main
  `2c5edcb4c`), `rtx5090-day37/build-r5.sh` then `chain-r5.sh` (runner from 16:28Z; receipts `rtx5090-day37/r5/`).
- OWED statuses refreshed: every half that named queue-e, f, i or j now names its queue-k item (queue-k: item 1 done,
  item 2 DAY39B running since 15:20Z). DAY38's 5090 reading is in (2.4: addendum D PROMOTE-ELIGIBLE), so O2 is read on
  both cards.
- Test discipline (lead, 2026-09-26): the full memra-server lib suite (and the full memra-engine lib suite when engine code moves) runs before each push; a filtered run is never the pre-push gate. The filtered run is how 09badfe57 shipped the admit_predict_shadow_wiring red.

# WP-B checkpoint 2026-09-26 (O14 built): the eleventh sitting queued on BOX35 (the lead's); O14 coded and CPU-gated, the twelfth sitting ready (NEED TARGET CARD); the ninth sitting still on BOX33
- O14 (DAY49): the engine step guard (`memra_engine::step_guard`, census-pinned state-write marks in the generic batched body) and the worker's one reclaim-then-retry of the same batched step under `MEMRA_BATCH_OOM_RECOVER` (`6102fb63a`, `02dbdfa40`); the gate's arm i (control, recovery, red twin); the serving shape (burst 8 x 6,144, no warm, the fault on the burst). Local runner `rtx5090-day49/run.sh`; target `pro-single-b-sitting12.sh`.
- `09badfe57` (the eleventh sitting's DAY45B source) left `admit_predict_shadow_wiring` red on the retire seam's distance (my filtered test run missed it); `6102fb63a` moves the receipt after the book retire. The runtime program of 09badfe57 is the same (the receipt line prints before rather than after the retire).

# WP-B checkpoint 2026-09-26 (later): the tenth sitting read (DAY45); addendum B coded; the eleventh sitting ready (DAY45B then DAY47; NEED TARGET CARD); DAY37's A1 red repro placed; O6, O8, O14 pre-registered; the ninth sitting still on BOX33
- DAY37 2.8: addendum G's repro on the 5090: r4 RED twice (8 OOMs), v3 GREEN: the A1 red is the missing DAY39 addendum B.
- DAY45 2.1 (target card): W1 PASS; W4 FAIL on both arms identically (the burst past the card without an admission door: 53 x 503 from the batched prime OOM); W2 and W3 failures placed on by-design exits and the drop set. Addendum B: the memory door on both arms, a second wave, `w-retire-unreleased` (`09badfe57`).
- DAY47 (O7): h PASS on every 5090 run; g FAIL on the batched chunk (registered shape) then on the gate's literal (addendum A's shape met every term); g-red PASS; g-batch DOCUMENTED. Addendum B fixes the literal; a2 and a3 run locally.
- O14 opened (the batched chunk's OOM ends every session) and pre-registered (DAY49, text only). O6 (DAY46) and O8 (DAY48) pre-registered, text only.
- Target card: `pro-single-b-sitting11.sh` (DAY45 addendum B, 4 boots, then DAY47, the gate twice).
- Local: queue-k (yielding), the DAY47 runner (a2, a3).

# WP-B checkpoint 2026-09-26 (DAY45): the ninth sitting running on the target card (the lead's); DAY37 r4's A1 red placed (DAY39 2.3); O4 pre-registered and coded (DAY45); the tenth sitting ready (NEED TARGET CARD); the 5090 queue-k yields the card between cells
- Resync at resume: `git fetch`; tip `5edac849b` then this session's commits; queue-i finished DAY39r's eight boots and its chain ran the gate; queue-j waited.
- Card sharing (the lead's request): the runners sleep YIELD_S after each boot (DAY40 between orders, the probe between cells); queue-i and queue-j stopped by the lane (their logs say so) and replaced by queue-k (pid 2946742, YIELD_S=240) for the unrun items: DAY37 addendum G's repro, DAY39B, DAY40, DAY41, DAY42e, DAY41B, DAY43, DAY44, DAY45.
- DAY39 2.3 (the registered 5090 half): NOT-GREEN (v2 parks 3 prefill OOMs per G2 burst; the gate RED with 8). It places DAY37 2.7's A1 red (the same binary `580fe677...`): the missing addendum-B terms. Addendum G's repro (r4 twice, v3 once) is pre-registered and queued.
- DAY45 (O4): pre-registered `3182da256`, addendum A; code `21b081ee1` (MEMRA_ADMIT_W_RELEASE); the tenth sitting `pro-single-b-sitting10.sh` (4 boots); the reader dry-read on synthetic lines.

# WP-B checkpoint 2026-09-26 (DAY44): the eighth sitting read (DAY43: the clamp resumes 60 of 60 spec turns); O11 revised on the owner's direction and built (the exact and fast resume, DAY44); the ninth sitting ready (NEED TARGET CARD); the 5090 queue-i then queue-j
- Resync at resume: `git fetch`; the rig rebooted at 07:28Z (queue-e to queue-h died); receipts banked (DAY37 r4 with two server logs gzipped, DAY38, DAY38D, DAY39's two interrupted boots); queue-i (pid 1164292) relaunched for the unrun cells (DAY39 registered rerun as rtx5090-day39r, DAY39B, DAY40, DAY41, DAY42e, DAY41B, DAY43); lanes A and F share the card.
- DAY43 2.1 (target card): C2 to C5 PASS; C1 FAIL as registered (turn-3 prompts follow turn 2's completion; all same-prompt rows equal). The clamp: 60 of 60 spec turns resumed (34), later-turn TTFT p50 211 against 1,622 ms, p95 417 against 9,149 ms.
- The 5090 halves read: DAY37 r4 FAIL (no reading) on A1 (the pooled arm's admit-mem burst: 7 parked prefill OOMs; vmm GREEN); DAY38 registered P1 FAIL (as the target card); DAY38D PROMOTE-ELIGIBLE on the 5090 class.
- DAY44 (O11 revised): pre-registered `2720da1c5` (race cases, E1 to E6); addenda A and B; code `35ece04e7` (engine capture), `e9772790b`, `9fa281cff` (server), `138790651`, `7a4abb4c9` (the prime-only spec settle). memra-engine 578, memra-server 955 passed; clippy clean; GPU tests on the 9B pass (capture equals split; resume and settle-then-resume cold-exact).
- Local smoke: 0 flips on both routes; gapped turns faster than keep; the zero-gap G=256 cost as 1.4 anticipated.
- Target card: `pro-single-b-sitting9.sh` (DAY44, 16 boots plus offprev and the fault boot, about 12 h).
- Local: queue-i (running), queue-j (DAY44, behind queue-i).

# WP-B checkpoint 2026-09-26 (later): the seventh sitting read (DAY41 2.2: R1, R3, R4 PASS on both routes, R2 PASS on rewind 60 of 60; spec keep 34 of 60 is today's overshoot miss); O13 pre-registered and coded (MEMRA_SPEC_BUDGET_CLAMP); the eighth sitting ready (NEED TARGET CARD)
- Resync at resume: `git fetch`; the lead's mirror `pro-single-day41/box-b` checked (no ELF); main merged (`9e75001b0`, 91 commits, no conflict); queue-e at DAY38 done (04:56Z), queue-f and queue-g waiting behind it.
- DAY41 2.2 (target card): flips keep 24/60 and 6/34, rewind 0; re-primed G+32 rows at 6,144 and 30,720; TTFT plain x2.0 to x2.6, spec x1.01 to x1.32; 122,880 (nominatable, the control token's point) plain x7.3 and x14.9, spec x3.1 and x6.1.
- O13 (DAY43): the clamp truncates the request's last round at the budget like the grammar truncation; C1 to C5 pre-registered; the reader dry-read on DAY41's spec keep boots.
- Target card: `pro-single-b-sitting8.sh` (DAY43, 5 boots, about 3.5 h).
- Local: queue-e, queue-f, queue-g (DAY41B), queue-h (DAY43, behind queue-g).

# WP-B checkpoint 2026-09-26: the third and sixth sittings read (DAY40 V-DOOR PASS; DAY41 rewind EXACT, R2 FAIL placed, addenda B and C coded; DAY42e F1 to F4 PASS, warmth 10 of 24 on both arms); the seventh sitting ready (NEED TARGET CARD); the 5090 healthy and running queue-e, then queue-f, then queue-g
- Resync at resume (after the rig's reboot): `git fetch`; STATE and OWED re-read; the lead's mirrors at `pro-single-day40/box`, `pro-single-day41/box`, `pro-single-day42/box-e` checked (no ELF committed); queue-e (pid 109603, the lead's relaunch) running DAY37 r4 on the healthy card, queue-f (pid 111251) behind it; main merged (`0543cf5a1`, #727, no conflict).
- DAY40 2.1 (O3, target card): every DAY34 1.6 term PASS, V-DOOR PASS; SELECT R1 and R2 none, R3 32768; `on32768` 46 x 200 and 18 x 429 (day 36: 44 and 20); the seed cap bound on every ON boot. The box's FAULTS.txt was a chain bug (the boots root passed to the lister); re-listed locally, clean.
- DAY41 2.1 (O11, target card): probe `rewind: EXACT`, `hist: NEAR-TIE-CLASS` at all four; R1 plain PASS, R3 PASS, R4 PASS; R2 FAIL (plain rewind 40/60: turn 3 armed no checkpoint when the boundary equals the resumed depth; spec 34/60 and 36/60: the parked committed carries the final burst's overshoot); R1 spec FAIL on the line half (10 affinity rewinds, no differ). The workload's literal `<|im_start|>` text in docs/SERVING.md put the 30,720 and 122,880 checkpoints at an interior control token. Flips: keep 24/60 plain, 6/34 spec; rewind 0. Price at 6,144: plain TTFT x2.0 and x2.6, spec x1.04 and x1.29. FX: fanout lost, 7.0 -> 9.07 s. The reader's FX crash fixed and re-read locally.
- DAY41 addenda B and C (`e2ce82ba4`, `4e2ad014e`, code `23c296b58`): B1 unnominatable prompts arm the guard window, B2 a resumed session arms past the resumed depth with the fed-start floor, B3 the armed spec exact probe on the public stream. memra-server 943 passed, clippy clean.
- DAY42 2.3 (O12, target card, sixth sitting): P0 EXERCISED, F1 to F4 PASS on all eight; warmth 10/24 both arms; max gap 4.06 and 3.96 s against 8.93 and 9.02 s; TTFT p50 7.9 and 6.9 s against 11.8 and 12.0 s; offtick admits 15 and 16 against 17; time to 429 9.4 and 11.2 s against 15.9 and 16.0 s (local reading).
- O13 opened (the spec pool's overshoot miss on the default route): next to pre-register (DAY43) and code.
- Target card: `pro-single-b-sitting7.sh` (DAY41 addenda B and C, 9 boots, about 6 h).
- Local: queue-e (DAY37 r4 remainder, DAY38/39, DAY38D, DAY39B, DAY40, DAY41), queue-f (DAY42e), queue-g (pid 889141; builds target/day41b, then DAY41B).

## Earlier checkpoint
# WP-B checkpoint 2026-09-25 18:30Z: DAY42 exercised on BOX26 (F1 to F3 PASS; off-tick flush cut the tenants' stall and the burst's TTFT but kept less warmth); addenda E and F (one worker-level demote queue) coded; the sixth sitting ready (NEED TARGET CARD); BOX14 (DAY40, DAY41) still running
- Resync at resume (18:05Z): `git fetch`; main at 5228ff0cd already merged (no new commits); STATE and OWED re-read; queue-e and queue-f still waiting for the 5090's health.
- Resync at resume (15:52Z): main merged (`f2a7a5d10`, integ62; the FLAGS.md conflict resolved row by row; two lane-A census windows at the loop head widened from 4,500 to 6,000 characters because the head grew past them with integ62 plus this lane's idle refresh); queue-e and queue-f still waiting for the 5090's health.
- DAY42 2.1 (BOX20): F2 FAIL in both orders and F4 source-flip FAIL are no reading of the arm: no reclaim pass ran (every burst arrival fit), and the warm phase published nothing (spec-route `max_tokens=1` parks in the spec pool). F1 and F3 PASS as read. Addendum C: warm at `max_tokens=16` with more entries, a P0 exercised line. Addendum D: the burst books and allocates a large `max_ctx` and generates 64 tokens.
- Resync at resume (10:26Z): `git fetch`; main merged into the lane again (`e365aa112`, integ61; no conflict); STATE and OWED re-read; queue-e still waiting for the card's health (no cell run); no script the running BOX14 sitting reads was edited (BOX14's DAY41 chain fetches the tip, so the day-41 scripts stay as they were).
- O12 (DAY42, the lead's ruling at integ62): `MEMRA_ADMIT_RECLAIM_OFFTICK` (default unset, decide-by 2026-10-09; only with `MEMRA_ADMIT_BY_MEMORY=1` and `MEMRA_KV_HOST_CONTRACTS=1`), a deferring reclaim flush: today's drop set at once, the demote set off the tick one landing at a time, the arrival parked on the landing within the door's defer budget, then the typed 429; `[admit-mem]` lines gain `reason=`. Addendum A (the landing-failure fault is `d2h-source-flip`), addendum B (replan an empty plan). CPU: memra-server 935 passed, clippy clean. O10 folded into its cell.
- Target card: the fourth sitting, `pro-single-b-sitting4.sh` (DAY42, about 3 h, 96 GB host RAM for the 32 GB pinned tier, iproute2).
- Local: queue-e (every earlier unrun cell) then queue-f (DAY42), both waiting for the card's health.
- Resync at resume (05:48Z): `git fetch`; main merged into the lane (`7bcb6364d`, integ59 and integ60; one FLAGS.md row conflict resolved keeping both sides); STATE and OWED re-read; queue-b and queue-c were only timing out against the card (`[GPU requires reset]` since 01:25Z) and were stopped (their own pids); queue-e now waits for the card's health (it only reads nvidia-smi) and then runs every unrun local cell in decide-by order.
- O1 (DAY37): target class PROMOTE-ELIGIBLE (2.5 plus 2.6's rerun: every gate cell PASS on both arms, A5-ENSURE and A6 PASS, A5-MAPPER N/A under inline). Stage 0 disagrees across two boxes of the class (55.3 against 271.2 us).
- O2 (DAY38): target class PROMOTE-ELIGIBLE under addendum D (2.3); the red arm is red.
- O5 (DAY39): GREEN on the target card (2.2).
- O3 (DAY40): day 36's cell on the final booking, pre-registered; `pending_seed_uncapped=` printed (`b46ae200e`).
- O11 (DAY41): `MEMRA_RESUME_GRID_REWIND` (default unset, decide-by 2026-10-09, `424b6756e`) and the probe's `--rewind` arm; the price cells pre-registered (addendum A: RX with the prefix cache off, FX in its own boots, off-prev = the tip minus the door).
- Target card: `pro-single-b-sitting3.sh` (DAY40 then DAY41), about 12 h, needs iproute2.
- Next: the sitting's reading, the 5090 halves when the card is back, then O4, O6, O7, O8, O10.
- Day 36 checkpoint (kept): WP-B day 36 checkpoint: memra#680 closed under the door (days 33 and 35, plus the review cap); the owner's decision cell read V-DOOR PASS on both cards
- Branch lane/spill-b-20260919 on origin/main 25bbb91f5. Pushes in the announced development mode (`UNQUALIFIED DEVELOPMENT`, logged; no qualification claimed). Engine changes under the default-OFF door only; no default moved.
- Day 33 (`30a5ab697`, on main since #692): the armed gate books each still-priming session's prefill workspace (`pending_prime`), prints a `verdict=admit` line per admission, and parks a pre-emission prefill OOM. 5090 green at G2; the burst gate `tools/admit-mem-burst-gate.sh` wired into local-ci. BOX3 was lost before its PRO half.
- Day 34 (`9f335ac48`): the day-32 cell with V-ALLOC on the engine form, V-OOM, G-BOOK and PARK. The 5090 half reads V-DOOR PASS, banked as a record of that tree: no OOM, the 32768 burst 19 x 200 and 13 typed 429s, R1 8192 R2 8192. On BOX4 (a replacement Workstation card, not BOX3) memra#680's owed boots: red 46 x 503, green 0 x 503 but 3 parked prefill OOMs from unbooked prefix seeds. Part B not run, by the lead's order.
- Day 35 (`809c16444`, pre-registered at `164777767` before code, option (a)): the armed gate also books the prefix entry each armed session will publish (`pending_seed`); the cache's contents are unchanged. BOX4 red reproduces the 3 parks; green twice has 0 OOM lines, 44 x 200 and 20 x 429; V-ID-FIX, V-ID and V-OFF 16/16; card GREEN. The 5090 has no local red for the seed term; green G2 twice. CPU 891/0, clippy and fmt clean.
- Day 36 (pre-registered `88bb3b1e2`), the owner's decision cell: day 34's cell on `809c16444`. V-DOOR PASS on both cards, no OOM, no park. 5090: R1 8192, R2 8192, the 32768 burst 19 x 200 and 13 typed 429s. BOX4: R1 none, R2 none (max natural stop 193,178), the 32768 burst 44 x 200 and 20 typed 429s. R3 32768 and R4 context on both. It measured the uncapped seed booking: the review fix `34a4b7f23` (integ55) caps it at the prefix cache's remaining budget, and warm-cache ON rows may be over-booked (DAY36.md 2.2).
- Day 35 addendum: `34a4b7f23` on BOX4 under day 35's reader, G-NOOM 0 OOM lines twice, card GREEN. BOX4 released after a 559-file sha256-checked mirror.
- Owed: the door decision (owner, decide-by 2026-10-07); optionally the ON concurrency rows rerun on the capped booking (integ55 tree), 0.1 agent-day plus about 4.5 h local and 6 h on a target card.
- Open from earlier days: `--kv-allocator vmm` decide-by 2026-10-04; `MEMRA_KV_PARK_COMPACT` decide-by 2026-10-06 with its deciding cell; memra#476 boot pre-grow booking point (owner); outstanding-only `W` release at prime completion; DAY24's step-OOM and client-disconnect fault arms; the `[spec-vg]` predictive gap on MoE + linear families; darklanes `lane/budget-journal-order-20260922` at `20f8140e8` (owner, money path) and memra#464 guard seed (0.5 agent-day).
- RULE (lead, 2026-09-21): never signal a process you did not start on this rig; the driver stops only a server it spawned whose cwd is this cell's scratch dir.
