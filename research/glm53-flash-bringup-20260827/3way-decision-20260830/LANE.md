# THE DECISION WINDOW: plain vs native-MTP-spec vs DFlash2-spec (glm5, box B, 2026-08-30)

End-to-end three-way served comparison on the deployed serving shape. The owner decides the
drafter from this packet. **Decode tok/s leads; acceptance is a diagnostic** (owner, verbatim:
"it is more then only acceptance"). The loop under test is FIRST-GENERATION and untuned — see
"FRAMING" before reading any spec-vs-plain ratio.

- Build: `lane/glm5-dflash-draft-src` @ **f8f35bd911f79d11bc0dde0dfcde46532818b27d** (spec route
  + ppN + MLA-TC default ON + DFlash2 draft source; based on bringup 9053d538d).
  Rebuild-attribution receipt (`receipts/build-f8f35bd91.log`): `cargo build --release -p
  memra-server` exit 0, **real 46.7s** (incremental over the prior window's cache), binary mtime
  1788090751 == `BUILD_END` exactly, `git log -1` == f8f35bd91 in the same receipt.
  Strings probe on the built binary: `draft source = dflash2` **present**, plus
  `draft source = native-mtp`, `native MTP head `, 13 `glm5-spec`, `glm5-acc`, `mla-tc-prefill`,
  `RED-ARM tap-shift`, `DFlash2 drafter blocks`.
- Drafter artifact: `incoai/GLM-5.3-Flash-DFlash2` pinned @ revision
  `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, fetched to `/root/models/glm53-dflash2`.
  **sha256 of `model.safetensors` VERIFIED** =
  `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` — byte-for-byte the
  wiring lane's pin (`b33c0347`); the boot receipt echoes the same sha8. Census read from the
  safetensors header: 81 tensors, **1,171,080,448 params, all BF16, 2233.7 MiB on disk**; no
  `embed_tokens` and no head of its own (it borrows the target's, per the wiring lane's
  contract). Largest tensor `fc.weight [4096, 20480]` = the 5-tap feature projection.
- Placement (identical on EVERY boot of EVERY arm — the comparability requirement; arms differ
  ONLY in the source flags):
  `MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 CUDA_VISIBLE_DEVICES=0,1,2
  MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_MOE_RESIDENT_GB=98
  MEMRA_MOE_SLOTS=16 MEMRA_CTX=131072 MEMRA_MAX_SESSIONS=4 MEMRA_PREFIX_CACHE_MB=0
  NVIDIA_TF32_OVERRIDE=0 MEMRA_COMPAT=openai` + the model pin, port 18400.
  - MLA-TC is **default ON at this head** (no pin) — engagement asserted per boot from the log
    rather than assumed: `[mla-tc-prefill] engaged: ... fa_mla_gathered_bf16 (t=239, ...)` on
    all three arms.
  - Named deviations: `MEMRA_TIMEOUT_MS_MAX=600000` on all boots (measurement cell — the deep
    l3 prompts must not 408; FLAGS.md measurement override); `MEMRA_PREFIX_CACHE_MB=2000` on
    the 8-turn cache-twin boots only.
  - Serve harness `box/serve.sh`: pidfile + `/proc/<pid>/exe` + `MEMRA_ADDR` check before any
    signal (**never pkill, never basename matching**), boot-nonce written into the serving
    process environ and re-read from `/proc` at gate time (arm identity, not liveness),
    VRAM-at-ready snapshot per card on every boot.
- Arms:

  | arm | flags | draft source |
  |---|---|---|
  | PLAIN | (none) | none — zero `[glm5-spec]` lines is the gate |
  | NATIVE | `MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1` | the embedded `layers.45` NextN MTP head |
  | DFLASH | `MEMRA_GLM5_DFLASH=/root/models/glm53-dflash2 MEMRA_GLM5_SPEC=1` (**no MTP flag**) | the pinned DFlash2 drafter; head NOT loaded |

- Pools (real prompts only, never synthetic): decode-attribution pool (10 prompts, 6 code +
  4 prose, `decode-attribution-receipts/prompts.json`) + l3-ab deep pool (WARM ~0.4k, A4630
  ~3.7k / 14.5k chars, B5550 20k, C6470 23.2k; sha256 `de57a7a4...` verified on the box).
- Prior receipts this window is measured against:
  - `spec-battery-20260830/` (native arm, same placement, same pools): spec OFF **35.4 tok/s**,
    native ON **27.5** (0.777x), TTFT **5.5-6.1x worse** from the sequential per-token
    MTP-plane warm in `glm5_spec_session_new`; acceptance notrim K3 greedy **1.443 acc/cyc
    (2.443 tok/cyc)**, K5 1.473; 56/56 byte identity; break-even stated there as tok/cyc >= 3.14.
  - `acceptance-probe-20260830/` + the DFlash2 probe band: **3.06 tokens/cycle** all,
    **4.66 tool-wire**, **acc@1 0.73** (teacher-forced on real agent traffic).

## Cell log

### CELL 1 — boot receipts x3 arms + VRAM-at-ready: GREEN

All three arms GATES GREEN (nonce re-read from the serving pid's environ, >= 3 RESIDENT lines,
fluent output-sample gate, source/engagement lines as designed, zero `[glm5-spec]` lines on
plain). Receipts `receipts/c1/`, `receipts/logs/boot-c1-*.{gates,identity,vram}`.

| arm | VRAM-at-ready dev0 / dev1 / dev2 (MiB) | boot | source receipt |
|---|---|---|---|
| plain | 51444 / 62772 / **66166** | 32 s | (no `[glm5-spec]` line at all) |
| native | 51444 / 62772 / **70294** | 32 s | `serve route ARMED: MTP head loaded; draft head FULL target vocab` + `draft source = native-mtp` |
| dflash | 51444 / 62772 / **66774** | 38 s | `serve route ARMED: draft source = dflash2 @ b33c0347; draft head FULL target vocab; native MTP head NOT loaded (the q38 pattern: a full MoE trunk layer of VRAM saved)` |

- **The no-head saving is REAL and measured: 3520 MiB on dev2** (native +4128 vs plain; dflash
  +608 vs plain). dev0/dev1 are byte-identical across all three arms — both the MTP head and
  the drafter land on the **head engine** (stage 2, the device that owns the trunk `lm_head`
  the drafter projects through), exactly as the wiring lane specified.
- native's +4128 MiB = the `blk.45.nextn` NextN block (a full MoE trunk layer: 288 routed
  experts + MLA + indexer); its load line is `[mtp-glm5] MEMRA_GLM5_MTP=1: loading the
  glm5_next NextN block` + `RESIDENT blk.45.nextn.eh_proj.weight`.
- **dflash's +608 MiB is attributed, not a partial load**: the boot log reads
  `[dspark] precision=q4 (MEMRA_DFLASH_PREC unset)` — the 1.171B / 2233.7 MiB-bf16 drafter is
  loaded as **q4_0**, which is the owner-ratified DEFAULT since lane/dflash2-head-trim
  (2026-08-25, measured at unchanged acceptance on this very card class: RTX PRO 6000
  dspark_q38_gate x3 interleaved 157.9 q4 vs 152.4 q8 spec tok/s, accept 0.662 vs 0.656, all
  exact). 2233.7 MiB bf16 -> q4_0 ~= 558 MiB + scales/overhead == the measured 608 MiB.
  `MEMRA_DFLASH_PREC=q8|mixed|bf16` are the named alternates (q5 was measured DEFECTIVE and is
  not a serving mode). The census also confirms `[dspark] harvest=dflash (checkpoint census
  dflash2=true strategy_dspark=false)` — the loader identified the artifact by census, not by
  filename.
- The `[spec-k] automatic table` prints on every capable boot: `prompt<1024 -> K=3; cold-long ->
  K=3; prompt>=1024 and cached>=1024 -> K=2 (K=5 when the loaded MTP head is rank-trimmed)`.
  Since glm5 prefix entries cannot snapshot (spec-battery finding, re-receipted in cell 4), the
  `cached>=1024` rows are unreachable and **the policy default is K=3 everywhere** on both spec
  arms.
- MLA-TC engaged on all three arms at the same shape (t=239) — the default-ON flip is live and
  identical across arms, so it is not an A/B variable here.

### CELL 2 — DFlash2 served byte-identity spot battery: GREEN (the window continues)

Served-path spec-vs-plain greedy byte identity, K in {1,3,5} x **8 prompts spanning both
pools** (d00/d02/d04/d05 code, d06/d09 prose, l3-WARM, l3-A4630 — including the d02/d04 rows
the spec-battery lane identified as rejection-heavy), max_tokens 256, non-streaming, tape =
reasoning + `\0` + content bytes. K is a **boot pin** (`MEMRA_SPEC_K`): there is no
request-level `spec_k` on this server, so each K arm is its own fresh boot (a request field
would have been silently ignored and the "K sweep" would have compared one K three times).

- **24/24 tapes byte-identical to the plain boot** (8 per K arm) — `receipts/c2/`,
  `receipts/cell2.log`. Any divergence would have stopped the window; a draft source may only
  move acceptance, never output, and on the real artifact it does not.
- Loop-law screen: **0 flagged of 32** tapes. Aggregates carry no exclusions.
- Engagement receipts on every spec boot: `[glm5-spec] route=spec K=<pin> ... cold=1` and
  `[glm5-acc]` per-burst lines (66 bursts on the K=1 arm alone); the plain boot has zero.
- Acceptance already visibly ABOVE the native arm at matched shape (counts, greedy, K=3):
  d00 164 accepted / 91 rounds = **1.802 acc/cycle**, d04 158/74 = **2.135 acc/cycle** vs the
  native arm's pool-wide 1.443. Deepest prompt cumulative: K1 0.816 acc@1, K3 164/279,
  K5 179/400. The band comparison is cell 3; the perf consequence is cell 4.

### CELL 3 — DFlash2 acceptance sanity vs the probe band + tap-shift RED: GREEN, with one finding

Shape deliberately IDENTICAL to spec-battery stage 2 (both pools = 14 real prompts, max_tokens
128, greedy AND vendor-default with NO sampling params on the wire) so every row is directly
comparable to the banked native-MTP numbers. K pinned on both spec arms (the card-3 K-policy
finding: an unpinned comparison silently compares different K). Receipts `receipts/c3/`,
`receipts/cell3.log`, table `receipts/c3/c3-acc.txt`.

- **Tape identity 42/42 byte-identical to the plain boot** (14 prompts x {k3, k5, k3-red}).
- Loop-law screen: **0 flagged of 112** tapes. No exclusions in any aggregate below.

| arm | mode | acc/cyc | tok/cyc | acc rate | agentic | prose | l3deep | vs native (banked) |
|---|---|---|---|---|---|---|---|---|
| DFlash2 K=3 | greedy | **1.821** | **2.821** | 0.607 | 1.681 | 2.078 | 1.809 | native 1.443 -> **+26.2%** |
| DFlash2 K=3 | vendor-default | **1.889** | **2.889** | 0.630 | 1.935 | 1.819 | 1.893 | native 1.365 -> **+38.4%** |
| DFlash2 K=5 | greedy | **2.290** | **3.290** | 0.458 | 2.043 | 2.759 | 2.280 | native 1.473 -> **+55.5%** |
| DFlash2 K=5 | vendor-default | **2.147** | **3.147** | 0.429 | 2.109 | 2.160 | 2.194 | native 1.386 -> **+54.9%** |
| DFlash2 K=3 RED tap-shift | greedy | 1.729 | 2.729 | 0.576 | 1.662 | 1.876 | 1.695 | -5.1% vs its own green arm |
| DFlash2 K=3 RED tap-shift | vendor-default | 1.694 | 2.694 | 0.565 | 1.560 | 1.909 | 1.709 | -10.3% vs its own green arm |

- **The probe band is reproduced and exceeded.** Probe: 3.06 tokens/cycle (all), acc@1 0.73,
  teacher-forced. Served here: **3.290 tok/cycle at K=5 greedy**, and cell 2's K=1 arm gives
  acc@1 **0.794-0.969** per prompt. The feature contract (stream-mean of completed trunk layers
  [5, 14, 24, 33, 42], host-staged through `Cache::hc_taps`) is therefore correct on the real
  artifact and on the 3-stage ppN split — the wiring lane's named risk (a wrong layer set is
  fluent and silent) did not materialize.
- **Sampled acceptance does NOT decay on this drafter** — vendor-default K=3 (1.889) is HIGHER
  than its own greedy (1.821), whereas the native arm's sampled row LOST ground against its
  greedy (1.365 vs 1.443). The DFlash2 sampled path rides the same `dspark_accept_sampled`
  rejection walk the q38 serve route ships, and it holds. This matters more than the greedy row:
  vendor-default IS the served traffic shape.
- The **prose class is the drafter's best** (K5 greedy 2.759 acc/cyc, tok/cyc 3.759); agentic is
  the weakest (2.043) yet still above every native row. The probe's 4.66 tool-wire figure is not
  reproduced here because these pools carry no tool-wire prompts (a named pool gap, not a miss).

#### FINDING: the tap-shift RED arm does not collapse on the real artifact (gate-craft, not a defect)

The window expected acceptance to COLLAPSE under `MEMRA_GLM5_DFLASH_GATE_RED=tap-shift`. It did
not: **1.729 vs 1.821 acc/cyc, only -5.1%** (greedy; -10.3% vendor). Attributed before reporting,
and the instrument is NOT dead:

- `[glm5-spec] RED-ARM tap-shift: drafter tap layers shifted +1 (gate instrument, never a
  serving flag)` printed **29 times** on the red boot and **0 times** on the control boot;
- **12 of 14** per-prompt accept/round pairs differ between the arms (e.g. l3-A4630 77/52 -> 69/58,
  d06-prose 92/35 -> 88/40) — a dead instrument would have reproduced all 14 exactly;
- tapes stayed **14/14 byte-identical**, which is the invariant the red arm is really guarding.

So the honest reading: a +1 shift over a 61-layer trunk lands on adjacent layers whose residual
streams are highly correlated, and the drafter tolerates them. **`tap-shift` proves the feature
seam is WIRED and load-bearing; it is NOT a decisive oracle for the layer SET on the real
artifact.** What is decisive is the positive evidence: the absolute band match above, plus the
fact that DFlash2 out-accepts the native MTP head — a head that reads the true hidden state — by
26-55%, which is impossible if the taps carried no signal.

Named follow-up (deliberately NOT run in this window): a large-shift or reversed-tap red arm
would make the layer-set claim decisive. It needs an engine knob (`dflash_precision`-style match
in `glm_spec.rs` accepts only `tap-shift`), and adding one mid-window would have broken the
one-binary-across-all-arms requirement of the timed comparison.

### CELL 4 — THE THREE-WAY (timed, marker held for the whole window): DONE

18 boots, **0 failures**, interleaved x5 (plain -> native -> dflash per round, fresh boot every
arm every round) + 3 cache twins. `/root/TIMING-IN-FLIGHT` raised before the first boot and held
to the last, so arm conditions are consistent across the whole comparison. Boot-nonce arm
identity verified from `/proc/<pid>/environ` on every one of the 18 boots. Streamed greedy pool
at max_tokens 256; tapes are byte-identical across arms by cells 2-3, so tok/s is a pure speed
comparison. Loop-law screen: **0 flagged of 210** tapes. Receipts `receipts/s4/`,
`receipts/s4/DECISION-TABLE.txt`, `receipts/cell4.log`, `receipts/logs/boot-s4*`.

#### THE DECISION TABLE (median over pool, then median of the 5 boot medians)

| arm | decode tok/s | deep tok/s | TTFT 0.4k | TTFT 3.7k | acc/cyc | tok/cyc | vendor t/s | boot s | VRAM d0/d1/d2 MiB |
|---|---|---|---|---|---|---|---|---|---|
| **plain** | **35.41** | **30.01** | **0.422** | **2.208** | n/a | n/a | 33.5 | 32 | 51444/62772/**66166** |
| **native-MTP** | 27.49 | 22.11 | 2.299 | 12.303 | 1.452 | 2.452 | 27.2 | 34 | 51444/62772/**70294** |
| **DFlash2** | **31.71** | **26.04** | **1.780** | **4.612** | **1.907** | **2.907** | 32.0 | 39 | 51444/62772/**66774** |

| ratio vs plain | decode | deep decode | TTFT 0.4k | TTFT 3.7k | vendor |
|---|---|---|---|---|---|
| native-MTP | 0.777x | 0.737x | 5.45x worse | 5.57x worse | 0.810x |
| DFlash2 | **0.896x** | **0.868x** | 4.22x worse | **2.09x worse** | **0.954x** |

- **Per-boot stability is exceptional**: plain 35.41-35.43, native 27.49-27.50, DFlash2
  31.71-31.72 across five fresh boots each (spread **< 0.04%**), and TTFT identical to the
  millisecond. Box clock drift is nil inside the held window; the interleave was honoured and
  the numbers need no error bars.
- **Cross-window reproducibility**: plain and native reproduce the banked spec-battery stage-4
  numbers exactly — 35.41 vs 35.4, 27.49 vs 27.5, pool TTFT 0.362 vs 0.362, deep 2.208/12.303
  vs 2.212/12.258, acc/cyc 1.452 vs 1.443. Two independent windows on the same placement agree,
  so the DFlash2 row can be read against the banked native arc with confidence.
- Engagement receipts: every spec boot carries `serve route ARMED` + `route=spec K=3` +
  `[glm5-acc]` bursts + `usage.spec` on all 14 rows; **every plain boot has zero `[glm5-spec]`
  lines and no `usage.spec`**. The vendor-default row (NO sampling params on the wire) carried
  `usage.spec` on every spec boot — the never-serve-greedy receipt.

#### DFlash2 vs native-MTP: DFlash2 wins on every single axis

| axis | native | DFlash2 | DFlash2 advantage |
|---|---|---|---|
| decode tok/s (c=1) | 27.49 | **31.71** | **+15.4%** |
| deep-pool decode tok/s | 22.11 | **26.04** | **+17.8%** |
| TTFT @ ~3.7k cold | 12.303 s | **4.612 s** | **2.67x faster** |
| acc/cycle (timed pool) | 1.452 | **1.907** | **+31.3%** |
| VRAM on the head-engine card | 70294 MiB | **66774 MiB** | **-3520 MiB** |
| multi-turn TTFT @ 7.9k | 19.52 s | **6.34 s** | **3.08x faster** |
| boot time | 34 s | 39 s | -5 s (the only axis native wins) |

#### But at the K=3 policy default DFlash2 still loses to plain — and by how much

Per-prompt, each spec row paired against the SAME prompt's plain row over all 5 rounds
(`receipts/s4/DECISION-TABLE.txt`):

- DFlash2: **1 WIN / 13 loss of 14**. The win is l3-WARM (tok/cyc 3.556 -> **1.119x plain**,
  38.90 vs 34.76 tok/s). The worst losses are the rejection-heavy code prompts d02 (0.753x) and
  d01 (0.784x).
- Native: **0 WIN / 14 loss of 14**.
- Solving the measured (tok/cycle -> ratio) relation for ratio == 1:
  - **DFlash2 TIE POINT: tok/cycle 3.251 (acc/cycle 2.251)**; measured 2.922 -> below by 0.329.
  - Native TIE POINT: tok/cycle 3.139 (acc/cycle 2.139); measured 2.493 -> below by **0.647**.
  - **DFlash2 halves the distance to break-even that native leaves.** And cell 3 measured
    DFlash2 at K=5 greedy at **3.290 tok/cycle — above the 3.251 tie point** — which is exactly
    why the K sweep (cell 6) is decisive rather than optional here.

#### TTFT attributed: DFlash2 does NOT have the native failure mode

The window's standing instruction was to attribute a DFlash2 TTFT regression in code before
reporting it, because native's was a wiring bug (the sequential per-token MTP-plane warm loop in
`glm5_spec_session_new`). Measured penalty shape, spec minus plain, median per prompt:

| prompt tokens | native penalty | DFlash2 penalty |
|---|---|---|
| ~0.3-0.5k (10 pool prompts) | +1.49 to +2.11 s | **+1.11 to +1.49 s** |
| 4626 (l3-A4630) | **+10.06 s** | +2.40 s |
| 5547 (l3-B5550) | **+11.60 s** | +2.36 s |
| 6467 (l3-C6470) | **+13.37 s** | **+1.90 s** (it SHRINKS) |

- Native's penalty is **O(prompt)** — ~2.5 s per 1k tokens, i.e. the sequential warm running at
  ~400 tok/s, exactly as spec-battery attributed it.
- DFlash2's penalty is **near-constant and does not scale with depth** (it is smaller at 6467
  tokens than at 4626). There is no MTP-plane warm on this route, as the wiring lane predicted.
- It is **per-session, not a one-time boot warm**: within a single boot the penalty does not
  decay by request order (request 1 = 1.626 s, request 10 = 1.653 s), and boots dfl1 and dfl5
  reproduce every per-prompt TTFT to the millisecond. So the residual ~1.1-1.5 s is per-session
  drafter setup plus the context-feature ingest, not a warm-up artifact and not a leak.
- **Verdict on this axis: a tuning/engineering cost, not a wiring finding.** The wiring lane
  already names the candidate fix (the tap sink is host-staged by deliberate first-light choice;
  "a device-resident tap diet is a named follow-up if the box A/B shows it in the round wall" —
  this window is that A/B, and it does show it).

#### 8-turn larger-prompt cache-on twin (owner law, 2026-08-21)

All three arms, `MEMRA_PREFIX_CACHE_MB=2000` (named deviation, twin boots only), vendor-default
sampling, 8 user turns growing 4626 -> 7852 prompt tokens, max_tokens 128.

- **The twin ran honestly and the refusal IS the receipt**: `cached_tokens=0` on all 8 turns of
  all 3 arms, with the budget line proving 2097 MB was configured. The plain arm carries the
  loud refusal **9 times**: `[prefix-cache] snapshot failed (latent (MLA/DSA) KV planes are not
  carried by prefix entries); prefix not cached`. The spec arms show **0 refusal lines and 0
  hits** — a spec session never attempts the snapshot at all. Either way nothing is cached, so
  cache-on vs cache-off is a no-op for glm5 and the automatic K table's `cached>=1024 -> K=2`
  row remains dead code for this family (spec-battery's finding, re-receipted per arm here).

| turn | prompt tok | plain TTFT | native TTFT | DFlash2 TTFT | native acc/cyc | DFlash2 acc/cyc |
|---|---|---|---|---|---|---|
| 1 | 4626 | 2.232 | 12.274 | **4.104** | 1.224 | **2.023** |
| 2 | 4985 | 2.329 | 12.835 | **4.377** | 1.529 | 1.761 |
| 3 | 5394 | 2.462 | 13.987 | **4.681** | 0.868 | 1.761 |
| 4 | 6077 | 2.716 | 15.412 | **4.875** | 1.048 | **2.024** |
| 5 | 6650 | 2.934 | 16.618 | **5.217** | 0.984 | **2.342** |
| 6 | 7289 | 3.183 | 18.078 | **5.418** | 1.309 | **2.486** |
| 7 | 7601 | 3.296 | 19.156 | **5.953** | 1.228 | 1.612 |
| 8 | 7852 | 3.398 | 19.515 | **6.338** | 1.268 | 1.600 |

- On the multi-turn agentic shape this model actually serves, DFlash2's TTFT penalty holds at
  **1.84-1.87x** across the whole depth ladder while native's holds at **5.5-5.7x** — and
  DFlash2's sampled acceptance (1.60-2.49 acc/cyc) roughly **doubles** native's (0.87-1.53).

#### MEASUREMENT TRAP caught in this window (and guarded, not reported)

5 of the 15 vendor-default rows were short sampled completions (`completion_tokens` 31-99,
`finish=stop`). The streamed tok/s estimator is `(ct-1) / (t_last_chunk - t_first_chunk)`; when a
short completion's tail arrives in one or two SSE chunks that span collapses and the estimate
explodes — `s4-dfl2` read **310.8 tok/s** against a real ~31.7, which would have dragged the
DFlash2 vendor median to 169.8 tok/s and a nonsense 5.06x "win". A **128-token floor** now
excludes such rows from the vendor median by name, with the exclusion and the token count
printed per boot (`summarize.py`, `VENDOR_TOK_FLOOR`). The greedy pool rows that carry the
headline numbers are unaffected: **all 140 are >= 232 completion tokens** (checked, not assumed).
This is the same class spec-battery reported as the unexplained range "19.6-59.3 tok/s (sampled-
length noise)" — same artifact, now named and screened rather than absorbed into a range.

### CELL 6 — K sweep on DFlash2 (the leading spec arm): the decisive cell

Cell 4 left DFlash2 0.329 tok/cycle short of its 3.251 tie point while cell 3 had measured K=5 at
**3.290** tok/cycle — above it. So the sweep was run to settle whether a deeper K crosses plain.
One fresh boot per K (K is a boot pin), same binary, same placement, marker held, `route=spec K=<pin>`
verified in every boot log. K=1 was added after the first four made the trend clear — measured, not
extrapolated. Receipts `receipts/c6/`, table `receipts/c6/K-SWEEP.txt`, loop-law 0 flagged.

| K | decode tok/s | ratio vs plain | deep tok/s | TTFT 0.4k | TTFT 3.7k | tok/cyc | round wall | tok/cyc needed | short by |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **34.98** | **0.988x** | 28.55 | 1.525 | **3.978** | 1.839 | 52.6 ms | 1.862 | **+0.023** |
| 2 | 34.32 | 0.969x | 28.11 | 1.495 | 4.180 | 2.458 | 71.6 ms | 2.536 | +0.078 |
| 3 (policy default) | 31.72 | 0.896x | 26.05 | 1.781 | 4.614 | 2.907 | 91.6 ms | 3.245 | +0.338 |
| 5 | 26.14 | 0.738x | 21.40 | 2.186 | 5.094 | 3.421 | 130.9 ms | 4.635 | +1.214 |
| 7 | 21.10 | 0.596x | 17.56 | 2.695 | 5.819 | 3.662 | 173.5 ms | 6.145 | +2.483 |

**The K=5-crosses hypothesis is REFUTED, and the sweep explains why no K can win here.**
Acceptance rises monotonically with K (1.839 -> 3.662 tok/cycle) but decode falls monotonically
(34.98 -> 21.10): the round cost grows faster than the accepted tokens pay back, so the gap
*widens* with K and the optimum is the SMALLEST K.

#### The structural blocker, isolated by the round-wall fit

Round wall is strikingly linear in K over all five points: **round_wall = 31.6 + 20.1 * K ms**.
That separates the two costs the sweep exists to separate:

- **per-draft marginal cost: 20.1 ms** per extra draft token;
- **FIXED per-round cost: 31.6 ms** — against plain's **28.24 ms** per decoded token.

So **the spec round's fixed cost alone is 1.119x a plain decode step**: every verify round starts
~3.4 ms in the hole before a single draft is judged. That is a ppN pipeline-overhead cost, not a
drafter-quality cost, and it is precisely flip condition 2 that spec-battery named ("a cheaper
verify cycle under ppN — the walk pays 3-stage pipeline overhead per round").

Ceiling arithmetic at K=1 (measured round wall 52.6 ms):

- a **PERFECT** drafter (acc@1 = 1.0, tok/cycle 2.0) would reach ~38.0 tok/s = **1.07-1.09x
  plain**. That is the entire headroom available to speculation on this placement.
- to merely TIE plain at K=1 the drafter needs tok/cycle >= 1.862 (acc@1 >= 0.862); DFlash2
  measures **1.839 (acc@1 0.839)** — short by 1.2%.
- Caveat stated rather than hidden: the linear fit's own K=1 estimate (51.7 ms) puts the tie
  threshold at acc@1 0.832, i.e. marginally BELOW the measured 0.839. The direct per-K
  measurement is authoritative and reads **0.988x**. The honest summary is that **K=1 is within
  ~1% of a tie — indistinguishable from parity for practical purposes, and certainly not a win.**

### CELL 5 — concurrency on DFlash2 (the leading spec arm): the shed policy is correct

c=4 mixed pool (2 code + 1 prose + l3-A4630), greedy, max_tokens 256, plus a c=1 reference on the
same boot. Three arms because the K-shed policy makes nopin and pinned different questions.
Receipts `receipts/c5/`, table `receipts/c5/CONC-TABLE.txt`.

| arm | c | wall | tokens | aggregate tok/s | spec rows | per-row tok/s |
|---|---|---|---|---|---|---|
| plain | 1 | 7.6 s | 256 | 33.8 | 0/1 | 35.5 |
| plain | 4 | 33.7 s | 1024 | **30.4** | 0/4 | 7.7 / 8.4 / 8.4 / 8.4 |
| DFlash2 nopin (**deployed**) | 1 | 10.1 s | 256 | 25.5 | 1/1 | 30.3 |
| DFlash2 nopin (**deployed**) | 4 | 33.6 s | 1024 | **30.5** | **0/4** | 7.7 / 8.5 / 8.5 / 8.5 |
| DFlash2 K=3 pinned (counterfactual) | 4 | 44.6 s | 1024 | **23.0** | 4/4 | 6.4 / 7.0 / 6.5 / 7.2 |

- **The K-shed fires exactly as designed and it is protective.** 3 PP stages is not the pp2
  cross-device shape, so the placement default is LOW=2/HIGH=4 (`[spec-gate] policy
  placement=single-or-non-pp2 LOW=2 HIGH=4 source=placement-default spec-admission=on`). At c=4
  `choose_spec_k` returns K=0 with reason=Concurrency: the nopin arm logged **4 `route=plain`
  lines and 0 `route=spec`** for the c=4 rows, and every row came back with **no `usage.spec`**.
  Its aggregate (30.5) matches plain's (30.4) to within 0.3% because at c=4 the nopin arm *is*
  plain. The receipt is the admission line, not an inference from the tok/s.
- **The counterfactual proves the shed is worth having**: pinning K=3 through c=4 (an operator pin
  disables automatic demotion, and the boot log says so) costs **23.0 vs 30.4 aggregate tok/s =
  0.757x** — a 24% throughput loss. So if `MEMRA_GLM5_DFLASH` were ever flipped on, it must be
  left on the automatic policy, never a blanket K pin.
- Adjacent observation (about plain serving, not the drafter): this placement does **not** gain
  aggregate throughput from concurrency — 35.5 tok/s at c=1 versus ~30.4 aggregate at c=4. The
  3-stage ppN pipeline is already saturated by a single stream, which is the same pipeline
  overhead the round-wall fit exposed in cell 6, seen from the batching side.

## THE DECISION TABLE (the deliverable)

Everything below is measured on ONE binary (f8f35bd91), ONE artifact, ONE placement, with only
the source flags differing. Decode leads; acceptance is a diagnostic.

> **CAVEAT — untuned glm5 loop.** These are FIRST-GENERATION spec-loop numbers: T-parallel verify
> is in, but none of the maturity mechanisms that carried qwen and step past plain (adaptive draft
> length / PMIN-class confidence gating, draft-verify latency hiding, pipelined rounds that do not
> stall on verify readback, drop on first rejection, spec-from-token-0 without a sequential plane
> warm). **Both qwen and step also lost to plain at this stage.** The mature-loop port is a named
> follow-up. Read the spec-vs-plain rows as a measurement of THIS loop, and the native-vs-DFlash2
> rows as the drafter comparison — the latter is robust to the loop port because both arms share
> the loop and the drafter accounts for only ~2.8% of round cost.

| | PLAIN | NATIVE-MTP spec | **DFLASH2 spec** |
|---|---|---|---|
| **decode tok/s, c=1** (5-round interleave, K=3 policy) | **35.41** | 27.49 (0.777x) | **31.71 (0.896x)** |
| decode tok/s, c=1, **best K** | 35.41 | — | **34.98 at K=1 (0.988x)** |
| **aggregate tok/s, c=4** (deployed nopin) | **30.4** | — | **30.5 (shed to plain, 1.00x)** |
| aggregate tok/s, c=4, K=3 pinned | 30.4 | — | 23.0 (0.757x) |
| deep-pool decode tok/s, c=1 | 30.01 | 22.11 (0.737x) | **26.04 (0.868x)** |
| **TTFT short (~0.4k, cold)** | **0.422 s** | 2.299 s (5.45x) | **1.780 s (4.22x)** |
| **TTFT deep (~3.7k, cold)** | **2.208 s** | 12.303 s (5.57x) | **4.612 s (2.09x)** |
| TTFT @6.5k cold | 2.868 s | 16.242 s (5.66x) | **4.770 s (1.66x)** |
| multi-turn TTFT @7.9k (8-turn twin) | 3.398 s | 19.515 s (5.74x) | **6.338 s (1.87x)** |
| TTFT penalty shape | — | **O(prompt)**, ~2.5 s per 1k | **near-constant**, per-session |
| **acc/cycle** (timed pool, greedy K=3) | n/a | 1.452 | **1.907 (+31.3%)** |
| acc/cycle, greedy K=3 (128-tok cell) | n/a | 1.443 | **1.821 (+26.2%)** |
| acc/cycle, **vendor-default sampled** K=3 | n/a | 1.365 | **1.889 (+38.4%)** |
| acc/cycle, greedy K=5 | n/a | 1.473 | **2.290 (+55.5%)** |
| multi-turn sampled acc/cycle (8 turns) | n/a | 0.87 - 1.53 | **1.60 - 2.49** |
| acc@1 (K=1 arm) | n/a | 0.771 - 0.939 | **0.794 - 0.969** |
| **VRAM-at-ready, head-engine card (dev2)** | **66166 MiB** | 70294 MiB (+4128) | **66774 MiB (+608)** |
| VRAM dev0 / dev1 | 51444 / 62772 | 51444 / 62772 | 51444 / 62772 (identical) |
| **boot time** | **32 s** | 34 s | 39 s |
| byte identity vs plain | — | 56/56 (banked) | **66/66 this window** |
| loop-law flags | 0 | 0 | 0 (**0 flagged of 436 tapes** across all cells: 32 + 112 + 210 + 12 + 56 + 14) |
| per-prompt record vs plain (K=3) | — | 0 WIN / 14 loss | **1 WIN / 13 loss** |
| tie point (tok/cycle needed) | — | 3.139, measured 2.493 (**-0.647**) | **3.251, measured 2.922 (-0.329)** |

### FRAMING: this is a FIRST-GENERATION glm5 spec loop (owner input, 2026-08-30)

Read every spec-vs-plain row below against this: the glm5 spec loop has T-parallel verify and
**none of the maturity mechanisms** that took the qwen and step spec arcs from slower-than-plain
to faster — adaptive draft length / PMIN-class confidence gating, draft-verify latency hiding,
pipelined rounds that do not stall on verify readback, immediate drop on first rejection,
spec-from-token-0 without a sequential plane warm, and the rest (a lessons lane is extracting the
full table). **Both qwen and step LOST to plain at exactly this stage.** So "spec loses to plain"
here is the EXPECTED result for an untuned loop, not a DFlash2 failure and not a verdict on
speculation for this model. The numbers below are not softened; what follows separates the loop's
cost from the drafter's so the two questions can be answered independently.

### The loop cost and the drafter cost, separated by measurement

The engine at this head carries **no per-phase timing** — `[glm5-acc]` emits counts only
(`ctx=`, `burst=a/d`, `cum=`), and there is no draft-ms / verify-ms / host-gap instrumentation to
grep. Adding it mid-window would have broken the one-binary requirement of the timed comparison,
so the separation below is done by **measured A/B instead of profiler attribution**, which is
stronger anyway. Two facts make it work: (1) the wiring lane's seam means **everything from the
verify walk on is shared and untouched between the two spec arms** (`glm5_verify_rows`, the accept
walks, `glm5_verify_rollback`, commit bookkeeping, K policy), so at matched K the round-wall
difference between arms is *purely* draft-source cost; (2) the K sweep's linear fit splits the
DFlash2 round into a fixed and a per-K term.

| quantity | native-MTP | DFlash2 | reading |
|---|---|---|---|
| round wall @ K=3 | 89.20 ms | 91.67 ms | **+2.48 ms = +2.8%** — the entire cost of swapping the draft source |
| tok/cycle @ K=3 | 2.452 | 2.907 | **+31.3% acc/cycle** — what that 2.8% buys |
| cost per emitted token | 36.38 ms | **31.54 ms** | plain is 28.24 ms |
| cost per token vs plain | 1.288x | **1.117x** | DFlash2 removes 60% of native's excess |
| shared FIXED loop term (K=0) | 31.6 ms (same loop) | 31.6 ms (same loop) | **34% of the round, and 1.119x a whole plain decode step** |
| per-K marginal term | — | 20.1 ms per draft | verify-dominated, see below |

- **The per-K marginal cost is verify-dominated, not draft-dominated.** The two arms use radically
  different drafters — native runs K sequential forwards through a *full MoE trunk layer*
  (`layers.45` NextN: 288 routed experts + MLA + indexer), DFlash2 runs a 5-layer 4096-hidden q4
  model plus host feature staging — and yet their round walls differ by only 2.8%. If drafting
  dominated the 20.1 ms/K term, that swap would have moved the round wall far more. So the 20.1 ms
  is mostly the extra batched verify row through the 61-layer trunk: a LOOP cost, shared.
- **Therefore the blocker is entirely on the loop side of the seam.** The fixed 31.6 ms plus a
  verify-dominated marginal are both shared-loop properties; the drafter contributes ~2.8% of
  round cost and ~31% of acceptance. This is why the drafter question and the flip question
  separate cleanly, and why **loop maturity work pays off MORE for DFlash2 than for native**: at
  equal round cost it already banks 31% more accepted tokens per round, and every mechanism in the
  mature-loop list (drop-on-first-rejection, latency hiding, pipelined rounds, adaptive K) acts on
  the shared terms that dominate.

### The recommendation, in two separable parts

#### (a) WHICH DRAFTER WINS — the decision available today, and it is robust to loop tuning

**DFlash2, unambiguously. Make it the drafter of record.**

Both arms run the *same* loop, and the loop contributes the fixed 31.6 ms plus a verify-dominated
20.1 ms/K while the drafter contributes ~2.8% of round cost. So the drafter comparison is
measured on the axis the drafter actually controls, and **nothing in the mature-loop port changes
its sign** — it can only widen the margin, because DFlash2 banks 31% more accepted tokens per
round at equal round cost.

| axis | native-MTP | DFlash2 | margin |
|---|---|---|---|
| acc/cycle @ K=3 greedy | 1.452 | **1.907** | **+31.3%** |
| acc/cycle @ K=3 vendor-default sampled | 1.365 | **1.889** | **+38.4%** (native's sampled row *lost* to its own greedy; DFlash2's *gains*) |
| acc/cycle @ K=5 greedy | 1.473 | **2.290** | **+55.5%** |
| multi-turn sampled acc/cycle, 8 turns | 0.87 - 1.53 | **1.60 - 2.49** | ~**2x** |
| draft cost (round wall @ K=3) | 89.20 ms | 91.67 ms | **+2.8%** — all it costs |
| VRAM, head-engine card | 70294 MiB | **66774 MiB** | **-3520 MiB** (the native head is not loaded at all) |
| TTFT penalty shape | **O(prompt)**, +13.4 s @6.5k | **near-constant**, +1.9 s @6.5k | removes flip condition 1 outright |
| decode tok/s @ K=3 (loop included) | 27.49 | **31.71** | +15.4% |

The two facts that make this decision safe to take now: DFlash2 **exceeds the vendor's own
teacher-forced probe band** (3.290 tok/cycle measured at K=5 vs the 3.06 band; acc@1 0.794-0.969
vs 0.73), and it does so while *removing* the native arm's O(prompt) prefill regression rather
than trading against it.

#### (b) DOES SPEC BEAT PLAIN TODAY — no, for either arm, and that is the expected first-generation result

**Keep `MEMRA_GLM5_SPEC` and `MEMRA_GLM5_DFLASH` default OFF.** Not because DFlash2 underperforms
— it beat its own vendor band — but because the untuned loop's overhead is larger than the whole
prize. Both qwen and step lost to plain at this same stage; this is that stage.

The break-even arithmetic, stated once:

- A spec round emits `tok/cycle` tokens for `round_wall = 31.6 + 20.1*K` ms (measured, 5 K points,
  linear). Plain emits one token per 28.24 ms. So spec wins iff
  **`tok/cycle > (31.6 + 20.1*K) / 28.24`**.
- K=1 needs 1.862, DFlash2 delivers 1.839 -> 0.988x. K=3 needs 3.245, delivers 2.907 -> 0.896x.
  Every larger K loses by more. **No K crosses.**
- The fixed term is the whole story: at K=0 the requirement is already 31.6/28.24 = **1.119
  tokens per cycle just to break even on loop overhead**, before a single draft is judged. That is
  why no acceptance improvement can rescue the arm on this loop — and why the fix is a loop fix.

What this changes, concretely:

1. **DFlash2 replaces native MTP as the drafter for all future glm5 spec work**, and the native
   head stops being loaded at all (3520 MiB of head-engine VRAM back). This decision does not
   wait on the loop port.
2. **Both spec flags stay default OFF** (unchanged from the FLAGS.md rows). The flip condition is
   now one number instead of a hope: **the round's fixed cost must drop below ~28.24 ms**, i.e.
   spec-battery flip condition 2 (a cheaper verify cycle under ppN) — plus whatever the
   mature-loop mechanisms take off the marginal term. Land those and re-run cell 4 + cell 6 of
   this window unchanged; the harness, pools and protocol are banked here for exactly that.
3. **The mature-loop port is the named next lane, and this window sizes its prize.** The
   mechanisms qwen/step used act on precisely the terms measured to dominate here: the fixed
   31.6 ms (pipelined rounds that do not stall on verify readback, spec-from-token-0, drop on
   first rejection) and the verify-dominated 20.1 ms/K (adaptive draft length / PMIN-class
   confidence gating, draft-verify latency hiding). To beat plain at K=1 the round must reach
   1.839 * 28.24 = **51.93 ms against the measured 52.6 ms — 0.67 ms, a 1.3% loop saving, flips K=1 today**,
   and each further ms off the fixed term raises the ceiling above the current ~1.09x.
4. **If spec is ever flipped on, leave it on the automatic K policy.** Cell 5 shows the c>2 shed
   is protective: pinning K=3 through c=4 costs 24% aggregate throughput while the automatic
   policy sheds to plain and matches plain exactly.
5. **The tap-shift red arm should not be trusted as a layer-set oracle** on real artifacts (-5.1%
   only). A large-shift/reversed-tap variant is the named follow-up.
6. Adjacent finding for the ppN lane, independent of the drafter: this placement gains **no**
   aggregate throughput from concurrency (35.5 tok/s at c=1 vs ~30.4 at c=4). The single-stream
   pipeline overhead the round-wall fit isolated shows up on the batching side too, and it is
   probably the same seam.

### Named follow-ups (out of scope for this window, deliberately)

- **The mature-loop port** (owner's framing lane): adaptive draft length / PMIN-class confidence
  gating, draft-verify latency hiding, pipelined rounds, drop-on-first-rejection,
  spec-from-token-0. This window's round-wall fit is the budget to spend against, and cells 4+6
  are the re-run gate.
- **Verify-cycle cost under ppN** — the largest single term. Target: fixed per-round cost
  < 28.24 ms (a 1.3% / 0.67 ms round saving already flips K=1).
- **Per-round phase instrumentation** — the engine has none at this head (`[glm5-acc]` carries
  counts only), so draft-ms / verify-ms / host-gap had to be inferred by cross-arm A/B. A phase
  timer behind a flag would make the loop port's progress directly measurable instead of inferred.
- **Per-session drafter setup / device-resident tap diet** — the ~1.1-1.5 s per-session TTFT
  penalty. The wiring lane already predicted this A/B would surface it; it did.
- **Large-shift red arm** for the feature-contract oracle (needs an engine knob; not added
  mid-window to preserve one binary across every timed arm).
- **Tool-wire pool** — the probe's 4.66 tokens/cycle tool-wire figure has no counterpart here
  because these pools carry no tool-call prompts. A tool-wire pool would likely show DFlash2's
  best acceptance, and it is the shape the two buyer profiles actually send.

## Wall actuals (from the BOX-QUEUE timestamps, box clock)

| cell | window | wall |
|---|---|---|
| build + drafter fetch + sha verify + harness | 11:50-11:56Z | 6 min |
| 1 boot receipts x3 arms + VRAM | 11:56-12:02Z | 6 min |
| 2 byte identity (4 boots, 32 tapes) | 12:02-12:13Z | 11 min |
| 3 acceptance + tap-shift red (4 boots, 112 tapes) | 12:13-12:29Z | 16 min |
| 4 THE THREE-WAY (18 boots, timed, marker held) | 12:29-13:35Z | **66 min** |
| 6 K sweep K 2/3/5/7 (4 boots) | 13:35-13:54Z | 19 min |
| 6 extended, K=1 measured (1 boot) | 13:54-14:02Z | 8 min |
| 5 concurrency (3 boots, c=1 + c=4 each) | 14:02-14:18Z | 16 min |
| **total window** | 11:50-14:18Z | **2 h 28 min** (estimate was ~3 h; 32-40 s warm boots beat it even with two cells added mid-window on the evidence) |

Window totals: **37 boots across 6 cells** (3 + 4 + 4 + 18 + 5 + 3), **0 boot failures**, 66/66 served tapes byte-identical to
plain, loop-law **0 flagged of 436** tapes, 1 measurement trap caught and guarded. All four cards
released at 1 MiB, `/root/TIMING-IN-FLIGHT` down, server stopped pidfile-verified.
