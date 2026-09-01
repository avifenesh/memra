# Box window receipts: battery OFF/ON + zqx (2026-08-30). 262k cell NOT RUN.

Window taken on the queue box (identity in the private ops repo, never here), branch
`window-launchdiet-20260830`, after the launch-diet done-line.

Banked-copy transformation, named: every `system_fingerprint` value in the raw response
and SSE files is redacted to `memra-FINGERPRINT-REDACTED` per the public-boundary
policy's `live_fingerprint` rule. The observed value was the fingerprint form of the
merge commit recorded below (its first 12 hex), corroborating that the serving binary
was built from that commit. Originals stay on the box until the window's final close.

- Merge: `origin/lane/glm5-prefix-latent` merged clean.
- Commit in every receipt: `bec785165b3badd9c6e72ec8f65e07fc5288b9ab`
  ("Merge remote-tracking branch 'origin/lane/glm5-prefix-latent' into window-launchdiet-20260830").
- Rebuild check per the runbook (NOT mtime):
  `strings target/release/memra-server | grep -c "minted before or without latent capture"` = **1**.
- `NVIDIA_TF32_OVERRIDE=0` on every boot. `MEMRA_DSA_INDEX_RING` left at DEFAULT (the ring).
- Residency posture note, as the runbook requires: the launch-diet window's receipts
  (launch-diet-20260830/LANE.md) adopted NO serving residency posture (census + gates only,
  A/B deferred), so the runbook default `MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16` was
  kept verbatim.

Serve env (both arms, cards 0/1): `MEMRA_SPILL_STATS=1 MEMRA_MOE_RESIDENT_GB=98
MEMRA_MOE_SLOTS=16 MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1
CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=<artifact>
MEMRA_ADDR=127.0.0.1:18400 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 MEMRA_CTX=8192`
plus per-arm: OFF `MEMRA_PREFIX_CACHE_MB=2000` (no flag); ON `MEMRA_PREFIX_LATENT=1
MEMRA_PREFIX_CACHE_MB=2048`.

## The window-dominating finding: the pre-registered env OOMs prefill at depth

With `MEMRA_MOE_RESIDENT_GB=98` on the 2-card shape, cards settle at 97237/96983 MiB of
97887 MiB at ready (~0.65 GiB headroom). The runbook's own 2048 MiB cache sizing note
assumed the residency-cell plateau headroom of 3.8-4.2 GiB/card; this boot does not
exhibit it. Every prefill >= ~2.4k tokens failed with
`[engine-error] class=Overloaded prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`
(serve-off.log lines 315-352, serve-on.log 6 engine-error lines). Admission books glm5 at
0 B/token + fixed ~233-270 MB (the very budget gap this lane's derivation fix names), so
admission admits what the device cannot serve. NO fix was improvised on the box; the env
is the pre-registered one.

## OFF arm (boot READY 85s, pid-verified stop after)

- Pre-registered bar: **PASS**. C1 one greedy sha per prompt (p5 `98eac17a79c093c0` x4,
  p7 `ef197d4dd6c6cb6a` x4), `cached_tokens=0` everywhere, refusal lines PRESENT
  (`grep -c "snapshot failed (latent" serve-off.log` = **12**). battery.py exit 0.
- C2: turns 0-1 served (ttft 2.228s / 3.266s, cached 0); turns 2-7 dead (the OOM class).
- C3: 8000 chars OK (2211 tok, cold ttft 3.337s, repeat 3.328s, cached 0);
  16k (4565 tok) and 32k (8167 tok) dead (OOM).

## ON arm (boot READY 104s, pid-verified stop after; server left DOWN)

- C1 (THE acceptance bar): **PASS**. One sha per prompt across cold + 3 restored reps,
  byte-equal to the OFF arm's cold shas (p5 `98eac17a79c093c0`, p7 `ef197d4dd6c6cb6a`);
  `cached_tokens == prompt_tokens` on every restored rep (180/180, 269/269);
  `[prefix-cache] hit: 180 of 180 prompt tokens` lines in serve-on.log.
- Refusal transition: `grep -c "snapshot failed (latent" serve-on.log` = **0** (bar met).
- Insert probation shows real per-token bytes, not the 152.6 MB flat defect:
  97 tok = 155.0 MB, 180 = 156.9 MB, 269 = 159.1 MB, 326 = 160.5 MB, 1521 = 189.0 MB
  (= 152.6 MB floor + ~23.9 KiB/token, matching DESIGN.md par.2 exactly).
- C2: turn0 cold ttft 2.239s; turn1 cached 1521/2175 ttft **29.458s**; turn2 cached
  1521/2770 ttft **56.378s**; turns 3-7 dead (OOM). The restored-turn TTFT inflation
  (29.5s/56.4s vs 2.2s cold) is banked unexplained; candidates (not adjudicated in this
  window): expert-staging thrash at ~0.65 GiB free, restore-path cost. No timing claim
  is made from these rows beyond "inflated under this env".
- C3: all three depths dead, including 8000 chars which the OFF arm had served (by then
  cache resident 1456.4 MB of 2147 MB).
- battery.py exit **2**, 8 violations: C2 turn3..7 "cached_tokens=0, no warm-turn
  engagement" (dead rows), C3 8000/16000/32000 hit rows dead.
- **Pre-registered C2/C3 bars: FAIL** (measured; the failure mechanism is the env OOM
  above, and by these receipts it is not a latent-restore defect: every request that fit
  engaged and byte-matched).

## zqx latentprobe (ON arm, 4 reps x {tool,recall,bare} x {greedy,sampled})

- All 12 greedy rows PASS, cold AND restored (tool: exact `zqx_fetch_glimb_status`
  with `vault_sigil=VS-7`, `glimb_mode=thrum`; recall + bare: `QUARTZ-77-NIMBUS-4`
  returned verbatim). Restored reps engaged (cached == prompt) on every row.
- Sampled: 11/12. The miss: BARE/SAMPLED rep3 (restored, cached 96/96, engagement
  proven) answered a refusal sentence instead of the passphrase; reps 1-2 of the same
  cell passed. A sampled-variance row, named per protocol. Against DESIGN.md's
  "18/18 restored" phrasing this is **17/18 restored** with the single miss on a
  sampled bare row; latentprobe.py itself exits 0 unconditionally (no gate), the
  verdict is read from latentprobe.json rows.

## Not run

The 262k cell was NOT started (executor stop point; C2/C3 bar failure reported to the
coordinator). Server DOWN, all four cards 0 MiB, out-dirs retained on the box for the
window's final close.

## Deviation arm r92 (coordinator-ordered rerun, 2026-08-30): FALSIFIED by its OFF arm

Coordinator order: rerun the timed cells with exactly one env change,
`MEMRA_MOE_RESIDENT_GB=92`, rationale "92 restores the ~5-6 GiB/card headroom the
runbook's own cache sizing assumed". Receipts: `out-plx-off-r92/` (DEVIATION.txt in-dir,
`vram-at-ready.txt`, `vram-post-battery.txt`, serve log, battery stdout).

What 92 actually does on this shape, from serve-off-r92.log (the knob is a PER-DEVICE
all-or-nothing residency gate, not a headroom setter):

    [moe] resident-experts decision (PP dev0): experts 97.84GB + trunk 0.00GB vs free 99.24GB (expert budget 92.00GB) -> SLRU cache
    [moe] resident-experts decision (PP dev1): experts 89.69GB + trunk 0.00GB vs free 98.16GB (expert budget 92.00GB) -> RESIDENT

At the 24-split, stage0 (dev0) carries 97.84 GB of experts: budget 98 admits it resident
(the 97.2 GiB packing and prefill OOM of the first pass), budget 92 rejects it wholesale,
so dev0 streams experts through the SLRU cache. VRAM at ready: dev0 11.7 GiB, dev1
94.3 GiB (post-battery 18.9 / 97.2). Result: every C2/C3 row, including C2 turn0
(1521 tok) which had SERVED at 98, died with HTTP 408 Request Timeout (server alive,
requests admitted, one stray engine-error OOM line 261). No value of this knob yields
dev0 residency AND multi-GiB headroom: residency needs >= 97.84 GB on a card with
~99.2 GB free (~1.4 GiB max headroom); the balancing lever is the split itself, which is
a different deviation nobody pre-registered, so it was NOT improvised.

- OFF r92 formal bar: battery exit 0, C1 one sha per prompt (p5 `bd270fc4afd87f7e` x4,
  p7 `ef197d4dd6c6cb6a` x4), cached 0 everywhere, refusals present (9). The exit code is
  not the verdict: all 14 C2/C3 rows are 408-dead.
- Cross-posture note: p7's greedy sha matches the 98-boot and the 98-ON arm byte-for-byte;
  p5's differs across postures (`98eac17a79c093c0` at 98 vs `bd270fc4afd87f7e` at 92,
  within-boot identity intact in both). Placement changes the numeric program; within-boot
  identity remains the lane's bar.
- The ON r92 arm and the r92 zqx rows were NOT run: with C2 turn0 dying cold at this
  posture, the coordinator's question (does the 98-arm restored-turn TTFT inflation
  normalize at 92?) is UNANSWERABLE here, and banking timed rows under a rationale line
  the OFF arm had already falsified would manufacture receipts. The 98-arm C2 inflation
  (2.24s -> 29.46s -> 56.38s) therefore stays banked UNEXPLAINED and OPEN: neither
  normalized nor persisted; it could not be re-measured. The PRODUCT-SUSPECT flag
  decision needs a posture where C2 survives, which does not exist under this knob on
  this split.

## Deviation arm s23 (coordinator-ordered, 2026-08-30): ABORTED at the boot gate

Order: one change against the original runbook env, `MEMRA_PP_SPLITS=23` with
`MEMRA_MOE_RESIDENT_GB=98`, plus a pre-registered boot acceptance gate (both devices
`-> RESIDENT` AND >= 3.5 GiB free per card at ready) before any battery row. Receipts:
`out-plx-off-s23/` (DEVIATION.txt, ABORT.txt, residency-decisions.txt, boot log,
vram-post-fatal.txt). SPLITS=23 is a different numeric program than the L2-adopted 24;
no cross-posture sha claim is made (none could be: no request was served).

Result: the split arithmetic behaved exactly as predicted, and the posture still cannot
exist. Gate condition 1 was met on the decision lines, perfectly balanced:

    [moe] resident-experts decision (PP dev0): experts 93.77GB + trunk 0.00GB vs free 99.24GB (expert budget 98.00GB) -> RESIDENT
    [moe] resident-experts decision (PP dev1): experts 93.77GB + trunk 0.00GB vs free 98.56GB (expert budget 98.00GB) -> RESIDENT

but the load itself died before READY, immediately after the LAST expert slab
(blk44-down) populated:

    [server] FATAL: worker init failed: load zai/glm-5.3-flash: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")

Gate condition 2 was therefore unreachable; the arm was aborted per the gate, no battery
row sent, cards returned to 0 MiB.

Why no split can pass this gate, from banked numbers: the 24-split boot (the only one
that reaches READY) settles at 97237 + 96983 MiB of 2 x 97887 MiB, ~1.5 GiB of TOTAL
slack across both cards. Full dual-residency of this artifact's experts plus two stage
trunks/slabs consumes essentially the whole 2-card capacity; the 24-split boots only
because it leaves ~0.65 GiB/card, and moving one expert layer (~3.8 GiB) onto dev1
(already at 96983 MiB under 24) exceeds the card. The residency decision lines compare
experts vs free at decision time and do not carry the trunk/slab/cache bytes that land
later in the load, which is why the gate's condition 1 can pass on a boot that cannot
complete. There is no MEMRA_PP_SPLITS value at MEMRA_MOE_RESIDENT_GB=98 with both
stages resident AND multi-GiB headroom on this 2-card shape; the bytes do not exist.

Postures that can host the timed battery on this box, for the coordinator to choose
from (each a different numeric program, none improvised here): SLRU on both devices
(the parent lane's own measured posture, rebaseline-and-surface-20260828 serve.sh:
`MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000`, where the ~13.4 s
repeated-prefix TTFT that motivates this lane was banked), or a 3-card shape (box-ab
precedent, `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30`, claims card 2 from co-tenants).

## SLRU arm (coordinator go 3): closed post-boot by OWNER RULING, zero rows

Ruling (verbatim, in OWNER-RULING.txt): "we are not fast enogh ffor 2 card slru" —
2-card SLRU is not a serving posture. The arm had booted READY (194s) with the boot gate
holding (no RESIDENT lines under `MEMRA_MOE_RESIDENT=0`, VRAM at ready 11987/13013 MiB,
~84 GiB free/card). Receipts in `out-plx-off-slru/`: boot log, vram-at-ready,
empty residency-decisions (the SLRU receipt), DEVIATION.txt, OWNER-RULING.txt.

## The serving-shape arm: 3-card resident (owner-ruled posture) — the battery finally runs

DEVIATION (in every -3c dir): 3-card resident, `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30
MEMRA_PP_DEVICES=0,1,2` + runbook residency (98/16), box-ab precedent. Boot gate PASS
both arms: all three devs `-> RESIDENT` (61.15/61.15/65.23 GB experts), VRAM at ready
54547/65651/68085 MiB (~42/31/29 GiB free/card — the owner's session-capacity receipt),
READY 81s (OFF) / 104s (ON).

### OFF 3c: full battery serves, bar PASS (exit 0)

C1 one sha per prompt (p5 `98eac17a79c093c0`, p7 `ef197d4dd6c6cb6a`), cached 0
everywhere, refusals present (22), zero engine errors. C2 all 8 turns cold-served,
TTFT 2.25 / 2.80 / 4.35 / 5.36 / 6.68 / 7.73 / 8.65 / 9.93 s (1521 to 6273 prompt
tokens). C3 all depths: cold 3.39 / 6.87 / 12.86 s at 2211 / 4437 / 8039 tok
(~625 tok/s prefill at depth), repeats equal-cold, cached 0.

### ON 3c: C1 green, C2/C3 bars FAIL on a NEW defect class (exit 2, 7 violations)

- C1 PASS: one sha per prompt, byte-equal to the OFF arm colds; cached == prompt on
  every restored rep; 11 hit lines; refusals 0; insert probation per-token bytes
  (1521 tok = 189.0 MB again).
- C2: turn0 cold 2.25s; turns 1-3 ENGAGED (cached 1521) but inflated **15.28 / 54.00 /
  77.36 s** vs the OFF arm's 2.80 / 4.35 / 5.36 s. The suffix-length arithmetic fits a
  restored-suffix-through-the-decode-program mechanism: suffix 346 / 1196 / 1700 tokens
  x ~33-45 ms/token = 11-16 / 39-54 / 56-77 s. Banked as the candidate mechanism, not
  adjudicated.
- C2 turns 4-7 and ALL C3 rows dead. The server log names the defect (8x):
  `[engine-error] class=Engine prefill error: pp host-bounce payload <N> exceeds
  geometry-sized capacity 16777216 (n_embd=4096, max prime tokens=4096)` with
  payload = prompt_tokens x 16384 B (f32 hidden states). The SAME prompts served on the
  OFF arm, so the un-chunked host-bounce sits on the restored/cache-armed path of the
  3-stage PP shape. One request also shows the 90s deadline abort
  (`prompt 4347 (1521 cached), 0 generated, 89.99s`).
- After the first host-bounce errors, INSERTS CEASED (3 insert lines total), so the zqx
  cell ran with cached=0 on every "restored" rep: content 24/24 PASS (tool exact,
  passphrase verbatim, both modes) but the restored arm never actually restored —
  the zqx cell is content-green, engagement-void, and does not count as a restore
  receipt on this posture.

### PRODUCT-SUSPECT flag (the coordinator's step-2 question, now answerable)

The restored-turn TTFT inflation is flagged **PRODUCT-SUSPECT (restore-path cost)**:
it persists on the serving shape with ~30-40 GiB free per card, which eliminates the
expert-staging-thrash attribution from the 98-posture window. Two concrete engine
defects for the isolated Memra lane: (1) restored-suffix prime runs ~10-14x slower than
a cold full prime (decode-program arithmetic above); (2) the restored path's PP
host-bounce is not chunked against its geometry-sized buffer on 3-stage shapes
(errors above). With both present, MEMRA_PREFIX_LATENT must NOT flip on for serving:
a hit is slower than a miss and can 500 the request.

### Co-tenancy incident, 02:47Z, owned by this window

At the ON-arm stop, the window's stop loop (same basename-wide logic as the parent
lane's serve.sh) killed BOTH its own server and the card3 co-tenant lane's
memra-server (pid 79009, port 18500, started 01:51:54Z per BOX-QUEUE). Heads-up +
apology written to BOX-QUEUE. Fix installed for every subsequent boot/stop in this
window: a scoped serve script that PID-verifies AND requires
`MEMRA_ADDR=127.0.0.1:18400` in /proc/<pid>/environ before signaling
(mla-tc-ab-3c-serve-scoped.sh, banked with the A/B receipts).

## CORRECTION (coordinator, 2026-08-30): every arm above is the f32-trunk arm

The owner-accepted serving recipe (BRINGUP.md adopted recipe, L2) pins
`MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1`; the window runbook env predated L2 and lacked them.
Scope every conclusion above accordingly:

- The 98-arm prefill OOMs, the r92 SLRU falsification, the s23 load-OOM, and the
  "no resident 2-card posture has prefill headroom" claim are all **f32-trunk arm**
  results. L2 receipts show the BF16 trunk frees ~10 GiB (2-card resident at
  88.6/88.9 GiB used, ~9 GiB free), so the 2-card no-headroom conclusion is
  **scoped to the f32-trunk arm**; the recipe arm was never OOM-tested on 2-card in
  this window (owner has moved the serving shape to 3-card, so it will not be).
- The 3-card f32-trunk battery rows are kept and labeled; the decision battery re-ran
  on the pinned recipe (the -3cr dirs below). Observed recipe-arm residency at ready:
  51443/62771/64021 MiB, ~3 GiB/card lighter than the f32 arm's 54547/65651/68085.

## MLA TC prefill A/B, f32-trunk arm (mla-tc-ab-3c-f32trunk/): COMPLETE, NON-DECISIONAL

Ran before the correction landed; labeled and kept (F32-TRUNK-ARM.txt). 5 interleaved
rounds x fresh boots, arm identity from /proc/environ, scoped stops, ZERO violations
(engagement in every ON boot, zero mla-tc lines in every OFF boot, no cuBLASLt
declines, no engine errors), binary 9216ccdd (strings-mlatc=2).

Headline rows (medians across 5 rounds, greedy cold primes):

| prompt | OFF TTFD | ON TTFD | OFF tok/s | ON tok/s | door |
|---|---|---|---|---|---|
| A4630 (4614 tok) | ~8.3-8.6 s | ~3.4-3.9 s | 538-574 | 1186-1377 | ~-58% TTFD |
| C6470 (6455 tok) | ~10.4-10.7 s | ~3.9 s | 606-621 | ~1673 | ~-63% TTFD |

Decode ms/token unchanged across arms (~54-55 greedy rows, ~47.5 WARM rows, both arms:
decode never enters the door). Engagement announce names the chain
(`absorb/decompress = strided-batched bf16 TC GEMMs, attention = fa_mla_gathered_bf16
(t=4614, t_kv=4614, nh=64, width=2051)`).

First-token argmax gate: A4630 first chunk IDENTICAL across arms in all rounds.
C6470 diverges STABLY (OFF " The wiki has a Muon opt" every round, ON " The wiki
currently cove" every round; shared prefix " The wiki ", divergence at the 4th word,
so the flip is mid-stream not first-token, and it is deterministic per arm, not
near-tie noise). The 8-draw census belongs to the decisional pinned-recipe A/B below.

## Pinned-recipe battery (the decisional arm: -3cr dirs, recipe pins BF16_MMV+PP_BF16+GROUPED_PREFILL)

Boot gate PASS both arms: 3x RESIDENT, VRAM at ready 51443/62771/64021 MiB
(~3 GiB/card lighter than f32), READY 95s (OFF) / 60s (ON).

### OFF 3cr: bar PASS (exit 0, refusals 22, zero engine errors)

C1 one sha per prompt (p5 `ba59f88262cec835`, p7 `5f92ffbde53efa9e` — recipe-arm shas,
distinct from the f32 arm's as expected, within-boot identity intact). C2 all 8 turns
(2.04 to 9.32 s, 1521 to 6399 tok). C3 cold 3.11 / 6.35 / 11.83 s at 2211/4437/8039 tok
(~8% faster than the f32 arm, consistent with L2's TTFD claim), repeats equal-cold.

### ON 3cr: C1 + zqx-content green; C2/C3 bars FAIL — the SAME defect class as f32

- C1 PASS: one sha per prompt, byte-equal to the OFF 3cr colds (p5/p7 verified from
  battery.json both arms); cached == prompt on every restored rep; refusals 0;
  probation per-token bytes again (1521 tok = 189.0 MB).
- C2 turn0 cold 2.05 s; turn1 26.18 s (cached 1521/2215), turn2 61.31 s (1521/3127),
  turns 3-7 dead. Decode-program arithmetic fits again: suffix 694 / 1606 tokens x
  ~33-45 ms = 23-31 / 53-72 s vs measured 26.2 / 61.3 s.
- 9x the same engine error: `pp host-bounce payload <N> exceeds geometry-sized
  capacity 16777216 (n_embd=4096, max prime tokens=4096)`; inserts ceased after 3;
  zqx restored reps cached=0 (content 24/24 pass, engagement void).
- battery exit 2, 8 violations (C2 turn3-7, C3 all).

**PRODUCT-SUSPECT confirmed on the owner-accepted serving recipe.** Both engine
defects (restored-suffix ~10-14x slow prime; un-chunked restored-path pp host-bounce
on 3-stage shapes) are present with the recipe pins. MEMRA_PREFIX_LATENT stays OFF
for serving; the two defects go back to an isolated Memra lane
(worker restored-suffix prime program; pp host-bounce chunking on the restore path).

## MLA TC prefill A/B, PINNED RECIPE (mla-tc-ab-3cr/): the decisional cell — GREEN

5 interleaved rounds x fresh boots per arm, /proc/environ arm identity, scoped stops,
binary 9216ccdd (merge cc718b988), recipe pins in BOTH arms, MEMRA_PREFIX_CACHE_MB=0
(named). ZERO violations: `[mla-tc-prefill] engaged` + dispatch counter in every ON
boot, zero mla-tc lines in every OFF boot, zero cuBLASLt declines, zero engine errors.

| row (n=5 each) | OFF TTFD med [range] | ON TTFD med [range] | door | OFF/ON prefill tok/s |
|---|---|---|---|---|
| A4630 greedy | 7.449 s [7.358-7.646] | 2.832 s [2.788-2.999] | **-62%** | 619 / 1629 |
| C6470 greedy | 9.571 s [9.524-9.583] | 2.986 s [2.984-2.991] | **-69%** | 674 / 2162 |
| A4630 sampled (vendor-default) | 6.731 s | 2.197 s | -67% | 686 / 2100 |

Decode ms/token UNCHANGED: WARM med 33.09 (OFF) vs 32.98 (ON) — decode never enters
the door. First-token argmax gate: **ZERO flips across all 10 boots**; C6470's first
chunk is byte-identical across arms on the recipe (`' The wiki has a Muon opt'` both).
The f32-trunk arm's stable C6470 divergence does NOT reproduce on the pinned recipe,
so the 8-draw census is moot on the decisional cell (census script banked anyway).
The A/B workload pool (box-only file l3-ab/prompts.json) is banked here as
`l3-ab-prompts.json`, with the driver (`mla_ab.py`) and the scoped serve script.

Verdict handed to the coordinator: door green (-62/-69% TTFD, argmax clean, engagement
receipted both arms); the 1M cell inherits `MEMRA_MLA_TC_PREFILL=1` per the running
order; the default flip itself stays the owner's call per the lane.

## Box A close (owner kill order, 2026-08-30)

Box A is being destroyed (owner keeps the cheaper second 4-card). Everything above is
banked on `lane/glm5-prefix-latent-window`; nothing window-related remains only on
box A (the unused arm dirs held only DEVIATION.txt copies, reproduced here; the
card3 co-tenant lane's material is its own). The remaining sequence (pinned-recipe
battery, A/B, 1M cell) migrates to the replacement box per the coordinator's order;
the A/B and battery above already COMPLETED on box A, so box B re-runs only what the
coordinator names.

## Box B (replacement box; receipts under boxb/): the window re-run on serving silicon

Box A was destroyed on owner order; box B carries the same artifact (sha-verified corpus,
same mint) on RTX PRO 6000 Blackwell WORKSTATION Edition at 600 W (full power; the
gpu_max_power law checked in the first minutes). NO timing number from box A is ported;
every row below is fresh. Binary: `fe1fc438f98ca4c939e971c0dfe745e8bd7df748` = window
branch merged onto cc718b988 PLUS `origin/lane/glm53-1m-demo` (required by the ordered
`MEMRA_TIMEOUT_MS_MAX` pin, and carrying 93927b1fac "ppN prime walks the chunk
schedule"); strings latent=1 mlatc=2 timeout=1 after a real rebuild (the first build was
a 0.09s no-op off a silently failed merge — caught by the strings check, the
rebuild-attribution law working as designed).

### Pinned battery, box B: the defect set NARROWS to one

- OFF: bar PASS (exit 0, refusals 22, zero engine errors). C2 cold 1.77-8.57 s across
  8 turns; C3 cold 2.77 / 6.19 / 11.19 s.
- ON C1: PASS with engagement, and the shas are BYTE-IDENTICAL ACROSS BOXES on the same
  binary/recipe/placement (p5 `ba59f88262cec835`, p7 `5f92ffbde53efa9e` on both boxes,
  Workstation vs Server Edition silicon).
- **The box A host-bounce defect is GONE on box B: zero engine errors.** The box B
  binary's only relevant delta is the 1m-demo lane, whose 93927b1fac chunks the ppN
  prime; the box A un-chunked host-bounce path no longer exists. Defect #2 therefore
  already has its fix in-tree on `lane/glm53-1m-demo`.
- **C3 restored rows are green and near-instant**: cold 2.77/6.18/11.21 s vs repeat
  1.09 s / 0.011 s / 0.017 s with cached == prompt (2211/4437/8039) — a whole-entry hit
  with a ZERO-length suffix restores fast and correctly.
- **Defect #1 stands, isolated**: C2 restored turns with a non-zero suffix prime the
  suffix at decode speed. Box B fit is near-exact: suffixes 469/1407/1899 tokens at
  ~33 ms/token predict 15.5/46.4/62.7 s; measured 15.10/46.13/62.49 s. Turns 4-7 die on
  the (default, non-overridden) 90 s deadline with NO engine error. battery exit 2,
  4 violations (the deadline-dead turns).
- PRODUCT-SUSPECT verdict sharpened: ONE engine defect remains for the isolated memra
  lane — the restored-suffix prime path runs the decode program per token instead of
  the chunked prefill program. Flag stays OFF for serving.

### Pinned A/B, box B silicon (boxb/mla-tc-ab-b/): GREEN again, fresh receipts

5 interleaved rounds/arm, 0 violations, 0 flips (C6470 first chunk identical across
arms on this box too). Medians (n=5): A4630 greedy 6.579 -> 2.507 s (**-62%**,
701 -> 1840 tok/s); C6470 greedy 8.911 -> 2.863 s (**-68%**, 724 -> 2255 tok/s);
A4630 vendor-default sampled 6.416 -> 2.202 s (-66%). Decode untouched (28.80 vs
28.77 ms/token). Same verdict on both silicon editions, independently measured.

## 1M serving-config receipt, box B (boxb/out-1m-b/): the answer is NO on this shape

Cell: the window's 3-card resident recipe + `MEMRA_MLA_TC_PREFILL=1` (per the A/B
verdict) + `MEMRA_CTX=1048576 MEMRA_PREFIX_CACHE_MB=0 MEMRA_TIMEOUT_MS_MAX=64800000`
(the 1m-demo lane's measurement override), corpus = the demo's sha-banked Gutenberg
file rebuilt on-box and sha-verified (`a07d4fcd...`), deep prime in the vendor-default
shape (deviations named in DEVIATION.txt: port 18400, MAX_SESSIONS=4 recipe admission,
vendor per the serving law with the demo's 0.01% greedy/vendor prefill agreement).

- Boot green (3x RESIDENT, ready VRAM 51.4/62.8/64.0 GiB class), W1K warm rung healthy:
  1108 tok prime at **1123.7 tok/s** (MLA_TC engaged), decode 34.1 tok/s, coherent answer.
- **Deep prime FAILED at wall 427.9 s**:
  `[engine-error] class=Overloaded prefill error: layer 31: DSA k-pool selection failed:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY)`. Error census of the boot: exactly 1. The
  server survived (request-level error, no panic); usage None (the prime died before
  first token).
- Per-card peaks over the prime (vram-1m.csv, 20 s cadence): 65048 / 77688 / **97242**
  MiB (dev3 = co-tenant, flat 80830). Dev2 (stage 2, layers 30-44: fixed expert
  residency 65.23 GB + output head + whole-prime hidden stack + the kpool score
  transient) is the wall; layer 31 is its second layer.
- Verdict: the 3-card RESIDENT recipe is NOT a 1M-context config. The fixed-residency
  posture has no arena lever; the demo's PP4 + arena-capped SLRU
  (MOE_SLOTS=12000 + RESIDENT_HEADROOM_GB=36, splits 13,26,39) remains the only
  demonstrated 1M configuration of this artifact. The depth ceiling of the 3-card
  resident shape sits between the battery's proven 8k and the ~7-minute point of this
  prime; locating it precisely is a follow-up ladder, not this window's charter.

## Wall-time actuals (UTC, approximate to the minute)

23:58 window state check; 00:01 START line; 00:02 merge; 00:06 rebuild + strings-check;
00:07 OFF boot (READY 85s); 00:10-00:17 OFF battery; 00:19 ON boot (READY 104s);
00:22-00:37 ON battery + zqx; 00:39 stop, cards verified 0 MiB; 00:45 banked.
Estimate to this point was ~1h15m; actual ~47m (the OOM-dead rows return fast).

r92 rerun: ~01:00 resume line + DEVIATION.txt; 01:02 OFF r92 boot (READY 75s);
01:06-01:47 OFF r92 battery (408-dead rows burn ~2-4 min each at the server deadline);
01:50 stop (SIGTERM escalated to SIGKILL on the same re-verified pid/exe), cards 0 MiB;
~02:00 banked. ON r92 arm not booted (see the deviation section).

s23 arm: ~01:33 resume line + DEVIATION.txt; ~01:35 OFF s23 boot attempt; FATAL load
OOM after the last slab populate, before READY (~4 min in); ~01:42 abort confirmed
(process gone, cards 0 MiB on their own), no battery row sent, no timing marker needed;
~01:50 banked.

Box B: 03:50 window resume line; 03:52-04:00 merges (window branch + 1m-demo; first
merge attempt silently failed on git identity, caught by the 0.09s build + strings=0,
redone properly) + rebuild (45.9s, strings 1/2/1); corpus rebuilt + sha-verified in
parallel. 04:02 OFF boot (READY 32s), battery PASS; ~04:15 ON boot (READY 36s),
battery exit 2 (deadline turns only, zero engine errors); ~04:25-04:38 pinned A/B
(10 boots, DONE 0/0); 04:41 1M boot + W1K; 04:42:33Z deep prime START (warning line);
04:49:41Z prime OOM receipt; 04:55 stop, cards 0/1/2 idle, done-line written.
Box B segment ~65 min end to end.

SLRU arm: 01:43 resume; 01:45 boot (READY 194s); owner ruling landed; ~01:52 stopped,
zero rows. 3-card f32 arm: 01:56 OFF boot (READY 81s); 02:04-02:20 OFF battery PASS;
02:22 ON boot (READY 104s); 02:24-02:44 ON battery+zqx (exit 2, host-bounce class);
02:47 stop (co-tenant incident, owned above). 02:50-03:00 merge cc718b988 + rebuild
(strings-latent 1, strings-mlatc 2). f32 A/B: ~03:02-03:45, 5 rounds complete.
Env correction landed; pinned battery: 03:52 OFF boot (READY 95s), battery PASS;
~04:15 ON boot (READY 60s), battery+zqx (exit 2, same class); pinned A/B:
~04:35-05:20, 5 rounds, 0 violations, 0 flips. Box A evacuation + close ~05:30.
