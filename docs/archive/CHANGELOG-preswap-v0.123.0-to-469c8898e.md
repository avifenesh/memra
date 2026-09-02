# Pre-swap changelog: v0.123.0 to 469c8898e (old history, 108 non-merge commits)

The memra repository history was rebuilt from a zero-commit content snapshot on 2026-09-01.
The snapshot commit (`49d1d6f65`, "catch-up sync: memra@469c8898e") is the root of the
current history and corresponds to old-history commit `469c8898e` (old main tip), which was
151 commits past the last release tag, `v0.123.0` (old-history commit `bc0952fe5`,
crates.io memra-engine 0.123.0 published 2026-09-01T04:59Z). Those 151 commits never had a
tag, so the first release cut on this history (v0.124.0) covers them AND everything since the
snapshot. `tools/changelog.sh` cannot see them (the objects are not in this history); this
file is the record, generated with the same script against the archive bundle:

    bash tools/changelog.sh v0.123.0 469c8898e   # run inside memra-FINAL-preswap-all-refs.bundle

Archive bundle: rig `~/repo-backups-20260901/memra-FINAL-preswap-all-refs.bundle`, and R2
bucket `tiyuvta-capture` prefix `repo-bundles/20260901/`. The bundle carries
`refs/tags/v0.123.0` (tag object 4826ab9bb, peels to bc0952fe5) and `refs/heads/main` at
469c8898e among 373 refs. The 38 files that differ between 469c8898e and the snapshot root are
the provider-mention purge re-applied to research raw files (42 insertions, 42 deletions, no
code). Subjects below are verbatim commit subjects and read as such, with one exception marked inline: the provider-name scrub the tree carries applies to this file too, so that subject names the scrub without the provider. 34 subjects are lane bookkeeping
(perf-ci rows, battery receipts, run-N gate tables) and carry rig timing numbers, which are
exactness-only by law and never published; they are moved below the `<!-- bookkeeping -->`
marker so the record is complete while `tools/changelog.sh` stops emitting at the marker.

Changes since v0.123.0:

## Fixes
- system_fingerprint is a real, rewrite-proof build identity
- restore parent-scoped paired Q8 gate-up
- structured_output advertisement mirrors the real refusal, not a template heuristic
- de-race the gate-template fixture's rc-75 lock-timeout arm
- the public-boundary scan must survive a path it cannot stat, and no tracked symlink may leave the repo
- a ppN body must publish EVERY stage stream to its caller, not just the last

## Documentation
- MEMRA_VISION_OVERLAY_PUBLISH — overlay row residency, both arms, the rollback seam
- seal HY3 paired-Q8 qualification
- cold-prime rate gap under concurrency, scoped + named GPU gate (competitive-bench debt 2)
- the three MEMRA_PRIME_CHUNK=0 inline promises now state the PRIME_CHUNK_LAUNCH_CAP bound (revuto review finding: the seam narrowed to 65,520 and these sites still said any-length; no working behavior lost, the old walk always died at launch above the grid.y wall)
- say why pp_exit_publish's loose polarity is deliberate (review note)

## Other
- provenance: a lane that looks unlanded because it landed as a squash
- rebase onto origin/main 88c7caed0: re-gated, audited, and arm D takes main's new build identity
- B200-TRANSFER item 12: corrected against the landed b200-prep census (sm_100a compiles; NVFP4 on 100a is dp4a decode + cuBLAS prefill, NOT the W4A8 int8 MMQ)
- box battery: fixtures pinned by sha with every refusal path executed, and arm D interleaves boots in one window
- vision: execute the residency refusal on one card, and prove it bites
- vision: write down the two ordering facts new_published depends on
- vision-ppn: the residency law shared by every consumer, boot-time placement admissibility, and the lane bank
- vision: the overlay carries its own row residency, and ppN publishes it into stage 0's context
- moeu spec: neutralize the box class name in the header (merged from main after the scrub commit; measurements untouched)
- provider-zero: remove every provider mention from the working tree (owner order, cross-repo with the darklanes twin lane, 2026-09-01) [subject edited: the same scrub applies to this file]
- clippy 1.97: while-let the handoff drain-demote loop (repairs the branch's clippy-zero bar)
- serve: first-token deadline admission gate, MEMRA_FIRST_TOKEN_DEADLINE_GATE default OFF (competitive-bench debt 3: deadline behavior under thrash)
- clippy: the host-demote drain loop is a while-let (unbreaks main's clippy-zero gate)
- prefix cache: stable-boundary plain capture, MEMRA_PREFIX_STABLE_BOUNDARY default OFF (competitive-bench debt 1: promote starvation)
- review round 2 hardenings (PR #96): identity-keyed absent-plane convention + doc truth + inert-flag warning
- ci: retrigger (synchronize runs never fired for the stage-B pushes)
- battery2: C1b cold reference runs in its own cache namespace
- lane: slot-B qualification battery (spec-ON arms + cache-bust control + C1b continuation byte oracle) + runbook
- glm5 spec x prefix (MEMRA_GLM5_SPEC_PREFIX, default OFF) + PR #93 review fixes
- ci: re-trigger (Actions created no run for the previous two pushes; GitGuardian saw both)
- settle the SM100 native-FP4 question authoritatively (cross-lane conflict)
- moeu: sync the corpus rows to their banked form (darklanes PR #49)
- moeu: box reclaimed mid-lane; verdict scoped to rig+code, tightening cell written up as owed
- moeu: interleave the probe's arms, and record what the fix was worth
- serve: carried suffixes ride the prime program on hyper trunks (MEMRA_HYPER_SUFFIX_PRIME, default OFF)
- moeu: NO-GO write-up, the sensitivity sweep that refutes the tempting kill, and the parked box cell
- review fixes: kernel_check's Stage-C FP4 arm keyed on the 120a property + count/attribution corrections
- B200-TRANSFER: reviewer pass — three inverted laws corrected, leak scrubbed, close hygiene
- moeu: price the MoE routed-union gather on the shipped kernels before writing one
- teeth arm 13: the uncompiled-advisory fixture derives its own ci.yml
- lane doc: fold in the composition lane's B200-TRANSFER handoff (PR #89)
- kv host tier: cross-deploy warm handoff (drain-demote export + drip import), default OFF
- rebase onto c48f63d0e: one site from the ornith-cold-long merge, allowed as intended
- B200-TRANSFER: do the residency arithmetic honestly (review finding)
- clippy gate: pin the toolchain to the workspace rust-version; stable moved mid-PR
- rebase onto 680cf01f5: the composition merge brought two sites, both machine-identities
- PR #87 review follow-up: local-ci runs the clippy gate, headers say what they mean
- clippy-zero restored and gated: main was ~200 warning sites deep with no lint gate
- b200 prep: sm_100a compiles again (one guard was the whole wall) and CI keeps it that way
- VFUSE.md: bank the measurement timeline — the cell never got the lock, then the box went unreachable
- composition lane: B200-TRANSFER handoff + the owner-pivot re-framing
- VFUSE.md: the verdict, the kill criteria, and where the verify's time actually is
- review: restore the lane-scoped test's teeth (MIN arm through the retry), clear_poison, DrainingRestore
- VERDICT: correct the acceptance claim, the transport prediction, and the ceiling arithmetic
- server tests: one flake must fail alone — poison-proof the shared locks, admit-retry the deadline family
- rig gate: ALL ARMS PASS after absorbing the extraction rename (PR #77)
- rig gate: ALL ARMS PASS on the rebased tree (main now carries PR #73's ppN publish fix)
- composition lane: the VERDICT (composed TP route REFUTED vs the 100 bar) + #82 review hardening
- glm5-tp: MLA shards decline the TC prefill door — an ungated composition closed
- tp2-box-probe: BOXP_MODE=spec — composed spec rows for the composition lane's box window
- vfuse: FLAGS row + the cost model, and the code fact that settles the lever
- verify-cost probe: the prof pass collected nothing, because take() disables
- qwen4exp mtp12: the vfuse seam + a verify-cost probe, so the fused-verify lever gets priced before any kernel is written
- extract2: verification-pass fixes — the ladder advice, the door-H clear in the RIGHT gates, and an unsourced cert line in my own table
- prime schedule: cap monolithic/oversized prime ranges at the CUDA grid.y wall (ornith cold-long 66k defect)
- accrace LANE.md: retire the fatbin named-item — another lane closed it, and in the kernel
- extract2: peer-review fixes (PR #77) — two real defects and a set of honesty findings
- accrace: fifth re-gate; boundary-gate conflict resolved as a synthesis, not a side
- extract2 item 6+8b: MEMRA_WORKER_AFFINITY -> MEMRA_WORKER_CPUSET, and the sampler audit closes
- accrace: fourth re-gate on origin/main (bringup stack + host-audit landed) — FIX 0/12, CTL 6/12
- accrace: third re-gate on the final tip (main advanced 4 more commits)
- accrace: re-gated on origin/main (owner merge-to-main workflow) — FIX 0/12, CTL 5/12
- accrace: re-verified on the merged lane tip (base moved 117 commits)
- extract2 item 4+8a: the TP transport seam goes general, and its two fleet tools become first-class
- accrace battery: 48/48 loaded runs green, and the arm that first missed it got moved
- extract2: door-H suite module doc names the general flag
- extract2: the lane doc, the phase-2 battery runner, the door-H alias arms, and the premise correction
- extract2: four phase-2 extractions — the flag-alias law for boolean doors, three general doors, and the draft-source load seam

<!-- bookkeeping -->
## Dropped as lane bookkeeping (kept for the record, not for release notes)
- perf-ci: board rows from the lane's full --perf battery (0 fail, 0 warn)
- perf-ci: clippy-zero lane battery receipt banked
- perf-ci: CLEAN-WINDOW battery row settles the #82 review's perf finding
- perf-ci: battery row for the review-hardening push
- perf-ci: battery row for the probe-branch push (tp_shard guard + BOXP spec mode)
- perf-ci: DIRTY-window FAIL kept for the v0.123 tip, not deleted and not read as a regression
- perf-ci row: extract2 tree at e05dab650 — qwen9b-plain-short 137.83 tok/s, window_clean, [OK] vs the rolling median
- receipts(rig): the gate matrix re-run on the final tree
- clippy 1.97 while_let_loop (main-red, host-tier drain-demote loop) + post-merge perf battery row (0 fail, 0 warn, MSRV toolchain)
- perf-ci: board rows from the lane's full --perf battery (0 fail, 0 warn)
- bank the third push-gate battery row
- round2 box receipts: bank replacement-box state ahead of possible account sweep (spec262kv1-thinkon mid-prefill, no rung rows yet; corpus_commit=06918148c remint receipts, qA4v2/qA5 queue logs)
- b200 prep: bank the two perf-ci rows from the push-gate batteries (qwen9b-plain-short 138.37 / 137.84 tok/s, 0 fail 0 warn)
- perf-ci: clippy-zero lane battery receipt banked
- admit-predict shadow: boot-derived budget default + dual booked/booked_real receipt (stress-campaign FINDING 3 calibration)
- ci: retrigger on the rewritten head (public-boundary scrub of the box-receipt header)
- perf-ci: CLEAN-WINDOW battery row settles the #82 review's perf finding
- perf-ci: battery row for the review-hardening push
- glm5 spec x TP: harden per the #80 review — model-truth admission, honest ARMED receipt, per-rank walk unification
- perf-ci: battery row for the probe-branch push (tp_shard guard + BOXP spec mode)
- extract2: run-10 receipts — the seventh-absorb tree is fully green (battery ALL GATES PASS, local-ci exit 0, correctness GREEN, perf 0 fail 0 warn, 138.22 tok/s [OK])
- extract2: run-9 receipts on the rank-widened tree — battery ALL GATES PASS (tp-gate ALL ARMS PASS over main's TP-2 AND TP-4 fixtures, all four alias arms non-vacuous), local-ci exit 0 correctness GREEN perf 0 fail 0 warn 138.22 tok/s [OK]; the first perf attempt's [FAIL] is recorded, not dropped — a contended window the runner caught, re-measured clean
- perf-ci: DIRTY-window FAIL kept for the v0.123 tip, not deleted and not read as a regression
- extract2: run-8 receipts — the twice-reviewed tree is green (battery ALL GATES PASS, local-ci exit 0 correctness GREEN perf 0 fail 0 warn 137.56 tok/s [OK]); cert lines now match the banked receipts exactly (325 engine / 517 server)
- accrace: sixth re-gate; correct a receipt note that declared committed evidence absent
- extract2: run-7 receipts — the reviewed tree is green (battery ALL GATES PASS, local-ci exit 0 correctness GREEN perf 0 fail 0 warn 137.41 tok/s [OK]); 325 engine + 519 server units, 17 added by this lane
- extract2: run-6 receipts on the final tree (main absorbed through a5130e228) — battery ALL GATES PASS, local-ci exit 0, correctness GREEN, perf 0 fail 0 warn, 138.18 tok/s [OK]
- extract2: run-5 receipts + final gate table — the full assigned scope closed and green (battery ALL GATES PASS, local-ci exit 0 correctness GREEN perf 0 fail 0 warn 138.45 tok/s [OK]); main absorbed a fifth time (nvfp4 quad-symbol) and re-verified
- extract2: run-4 receipts on the fast-forwardable tree — battery ALL GATES PASS, local-ci exit 0 correctness GREEN; the single perf WARN settled by diff (one comment line changed, zero executable bytes) rather than by re-measuring
- extract2: run-3 gate table — the main-merged tree is fully green (battery ALL GATES PASS, local-ci exit 0, 138.73 tok/s [OK], clippy zero after fixing main's 12)
- extract2: local-ci --perf receipt on the merged tree — exit 0, correctness GREEN, perf 0 fail 0 warn, qwen9b-plain-short 138.19 tok/s [OK]
- extract2: lane doc + run-2 receipts — items 4 and 8a EXECUTED, full battery re-run green on the merged tree
- perf-ci row: extract2 tree at e05dab650 — qwen9b-plain-short 137.83 tok/s, window_clean, [OK] vs the rolling median
- extract2: gate table + receipts — 30 GPU suites green non-zero, tp-gate ALL ARMS PASS with the two alias arms as each other's red
