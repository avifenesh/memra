# memra #641: results (lane/decode-exact-641-20260923)

Pre-registration: `DAY1.md` (pushed before the first boot). Rig: the local RTX 5090 Laptop, every
GPU run under `flock /tmp/memra-5090.lock` with an idle-card check (`raw/*.log` first lines). Model:
Qwen3.5-9B NVFP4 MTP GGUF, sha256 `52c9cceb...a8f39de`. All cells `executed-not-qualified`.

Verdict: #641 reproduces on `9c07b398b` and is fixed. A request primed inside a fresh concat
batch took the varlen FA arm, which attended bf16 of the pre-quantization K/V while the solo prime
attends the quantized cache view; the arm is deleted, every prime shape is now bit-identical to the
solo prime, and the served peer-cold-b text equals the solo control on 5 of 5 on-shape runs (5 of 5
differed before).

## Cell 1: engine replay (tickshape), base `9c07b398b` + the probe arm (`1ca71af5e`)

Binary: `concat-prime-probe` sha256 `6b9a3ffb...` (`raw/concat-prime-probe-1ca71af5e.sha256`).
Prompts: the gate's exact ids (A 2048, B 2048, C 4096 tokens), tick 1024, 32 teacher-forced steps,
join at step 4. N=1 per env, as pre-registered.

Default env (`raw/engine-tickshape-default.log`):

```
arm ref2 verdict: EXACT
arm tick verdict: EXACT
arm bp prime: logits bitdiff=248319 maxabs=1.816733e-1 argmax ref=82 arm=82 | h_seed bitdiff=4096 | hidden bitdiff=8388594 maxabs=9.308960e0 first_row=Some(0) | cache digests differ 55/66 first=["L7.k", "L7.v", "L11.k", "L11.v", "L15.k", "L15.v"]
  first argmax flip at decode step 8: ref tok=596 (2nd 2513, margin 0.010529) arm tok=2513 (2nd 596, margin 0.084015); ref text to here "s \t</div>\n</div"
arm bp verdict: DIFFERS (greedy text diverges at token 8)
arm bps verdict: DIFFERS (greedy text diverges at token 8)
arm wave verdict: EXACT
tickshape verdict: AT LEAST ONE ARM DIFFERS
```

`MEMRA_FA_VL=0` (`raw/engine-tickshape-favl0.log`): every arm EXACT,
`tickshape verdict: ALL ARMS EXACT`.

Reading, per the DAY1 rule: ref2 is EXACT, so the cell is valid. bp and bps are one-program-law
violations. The divergence starts at hidden row 0 of the first (fresh) `[A, B, C]` batch. The
digest walks the KV layers first, in layer order, and the first differing entry is L7.k: L3, the
first full-attention layer, writes K/V rows computed from identical inputs, so the difference
enters at L3's attention output and reaches the cache at the next attention layer. Chunked solo priming (tick) and
the mixed `[B, C]` decode wave are bit-identical to the solo program. With the favl door off the
batched prime is bit-identical too, so the concat GEMMs at m=3072, the varlen GDN core and the
batched lm_head are not part of the difference on this model (no f16 mirrors are built for it, so
the lm_head takes the per-sequence m=1 matvec).

Cause, from the code: the fresh varlen arm (`use_favl` in `prime_cache_batch_inner`, task #18)
attends bf16 copies of the pre-quantization f32 K/V (`fa_mirror_vl` then
`fa_prefill_bf16kv_vl`). Since the 2026-08-05 chunk-invariance fix every solo chunk, chunk 0
included, attends the quantized cache view (q8_0 K / q5_1 V through `fa_prefill_view_ws`). The two
arms were bit-identical when task #18 landed (research/concat-prime-exact-20260802) and stopped
being so when the solo arm moved; twoprog W3 recorded the pair on 2026-08-13 as open.

## Fix (`58d3a225b`)

Per the law, the two programs are made one: the fresh varlen arm is deleted and every batch, fresh
or carried, runs the per-sequence attention core (`full_attn_prime_core_inner`), which is the solo
prime's program. Deleted with the door: the `use_favl` branch in `prime_cache_batch_inner`,
`fa_prefill_vl8`, `attn_pre_vl8` and their argument structs, the VL kernels in `flash_attn.cu`
(`fa_mirror_vl`, `q_gate_split_vl`, `attn_rms_vl`, `attn_rope_vl`, `append_kv_vl`,
`fa_prefill_bf16kv_vl` plus its hd128 twin) and the sm_90a `memra_fa3_vl` twin in
`fa3_prefill.cu`. `MEMRA_FA_VL` moves to the Removed doors ledger in `docs/FLAGS.md`. No new flag.
The varlen GDN core stays: with the FA door off every arm was already exact (cell 1).

## Gate cell: `ptick` / `ptickc` (`tools/prime-tick-exact-gate.sh`)

The engine replay of the #641 prime trace (cell 1's arms, the scheduler's shapes: two 1024-row
concat batches `[A, B, C]`, first fresh then carried, C's tail in solo ticks, B's decode joining a
`[B, C]` wave at step 4), every arm compared bitwise against `prime_cache(B)` in one call. Prompts
are regenerated from pinned seeds and checked against pinned sha256s. Liveness: a log carrying
`carried-prime.v1 unqualified` fails NOT-LIVE (the batched entry would fall back to solo primes).
Canary: `--canary` replaces B's first token inside the bp/bps batches only; the comparator must
report both DIFFERS. Wired into fast-gate (`models.tsv` rows `ptick`, `ptickc`; `map.tsv` routes
`flash_attn.cu`, `fa3_prefill.cu`, `hybrid_forward.rs`, the probe and the gate script to both).

Window 1, one lock hold, card idle (`raw/gate-ptick-window1.log`), N=1 per arm (the replay is
deterministic: ref2 is the determinism pin):

Base tree (`9c07b398b` + the probe arm, probe sha256 `6b9a3ffb8b6e37ae`), red:

```
    arm bp verdict: DIFFERS (greedy text diverges at token 8)
    arm bps verdict: DIFFERS (greedy text diverges at token 8)
    tickshape verdict: AT LEAST ONE ARM DIFFERS
prime-tick-exact-gate: FAIL rc=1 (a request primed inside a tick batch or a wave took a
base naked rc=1
```

Fix tree (probe sha256 `93f76e401eb0bbea`), green:

```
    arm bp prime: logits bitdiff=0 maxabs=0.000000e0 argmax ref=82 arm=82 | h_seed bitdiff=0 | hidden bitdiff=0 maxabs=0.000000e0 first_row=None | cache digests differ 0/66 first=[]
    arm bp verdict: EXACT
    arm bps verdict: EXACT
    tickshape verdict: ALL ARMS EXACT
prime-tick-exact-gate: PASS (every prime shape is bit-identical to the solo prime; log ...gate-ptick-fix-naked.probe.log)
fix naked rc=0
```

Fix tree canary, teeth:

```
    arm bp verdict: DIFFERS (greedy text diverges at token 2)
    arm bps verdict: DIFFERS (greedy text diverges at token 2)
    arm wave verdict: EXACT
prime-tick-exact-gate: CANARY OK (bp and bps DIFFER with B's batch prompt changed; log ...gate-ptick-fix-canary.probe.log)
fix canary rc=0
```

## Cost of the fix: batched prime wall, base vs fix (window 2, `executed-not-qualified`)

`prime-batch-gate <9B> --batch 3 --bench 1024` (three fresh 1024-token prompts, the #641 tick
shape), one process per run, 6 pairs in alternating order (AB, BA, ...), so N=6 per arm; each
run's figure is that process's own median of 5 alternating serial/batch reps. Binaries:
`prime-batch-gate` base sha256 `5c33c228dc657cb6`, fix `0e69b6c71680534d`. Thermal regime: hot
laptop card, 76-87 C at run starts, SM 1590-1732 MHz under load (`raw/perf/telemetry-250ms.csv`,
250 ms). Raw: `raw/perf-window2.log`, `raw/perf/pbg-<arm>-pair<p>.log`.

| arm | batch_wall_ms median (N=6) | min | max | batched vs serial, median |
|---|---|---|---|---|
| base (favl live) | 662.83 | 638.90 | 667.54 | +6.05% |
| fix (per-seq core) | 663.89 | 652.56 | 666.06 | +5.90% |

Paired fix minus base, per pair: +13.66, -7.53, +3.94, +2.97, -0.50, -1.47 ms; median +1.23 ms
(+0.19%). Flat at this shape on this card: the deleted arm bought no measurable wall time, and the
batched prime keeps its gain over serial primes. Pair 1's base run started from an idle clock
(187 MHz), which is its low 638.90 figure. One card, one shape (B=3, T=1024); no PRO 6000 or H100
figure here.

## Cell 2: serving repro (`repro641.py`), base and fix

Pre-registered order and rule as in `DAY1.md`, one fresh boot per run, card idle before every
window (`raw/serve-window{3,4}.log`, `raw/serve-fix-window6.log` first lines). Base server
`/tmp` copy of `9c07b398b` + the probe arm, sha256 `c0c57308...` (first line of each window log);
fix server built from `58d3a225b`, sha256 `53f9a109c9ca7cc0`. R = C0's peer-cold-b text, sha16
`06bfb5126effdd4c`. Per-run receipts: `raw/serve/<tag>-cell.json` plus `-server.log`, and
`raw/serve-fix/` for the fix.

| tree | run | yield | on-shape | peer-cold-b sha16 | vs R |
|---|---|---|---|---|---|
| base | C0 | 1 | control | 06bfb5126effdd4c | equal |
| base | Y0-1 .. Y0-5 | 0 | 5 of 5 | 85dbe38937836437 (all five) | DIFFER |
| base | Y1-1 .. Y1-3 | 1 | n/a | 06bfb5126effdd4c (all three) | equal |
| fix | C0f | 1 | control | 06bfb5126effdd4c | equal |
| fix | Y0f-1 .. Y0f-5 | 0 | 5 of 5 | 06bfb5126effdd4c (all five) | equal |
| fix | Y1f-1 .. Y1f-3 | 1 | n/a | 06bfb5126effdd4c (all three) | equal |

Verdict by the fixed rule: REPRODUCES on the base (5 of 5 on-shape Y0 differ, deterministically:
the same wrong text every boot, `s \t</div>\n</body>\n</html>\n```...` against R's
`s \t</div>\n</div>\n</div>...`). Controls clean. On the fix every run equals R. The served
divergence is the same program split cell 1 located: the on-shape Y0 run primes peer-b inside the
fresh `[prime-batch] B=3 tokens=3072 carried=0 partial=3` batch, and the yielding arm never forms
that batch. The other requests' texts (long, peer-hit) are identical on every run of both trees.
peer-cold-a returns an empty completion (`finish=stop`, sha16 `e3b0c44298fc1c14`, the empty
string) on every run of both trees: its first greedy token is EOS, so its bytes carry no evidence
either way.

## GPU battery on the fix (window 5, `raw/battery-window5.log`, card idle, `executed-not-qualified`)

Binaries: `memra-server` sha256 `b806e1b780b22a8f` (target/release, same engine source as
`53f9a109`; the re-link after the KERNELS.md commit moved the embedded revision), `prime-batch-gate`
`0e69b6c71680534d`, `decode-batch-gate` `e30aef5b1f410daa`.

```
== prime-batch-gate exact-b3-p24 (--batch 3 --exact)
   rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
== prime-batch-gate exact-b4-p1100 (--batch 4 --plen 1100 --exact)
   rc=0 ALL GREEN: prime-batch gate (batch=4, uneven lengths)
== prime-batch-gate exact-b6-p600 (--batch 6 --plen 600 --exact)
   rc=0 ALL GREEN: prime-batch gate (batch=6, uneven lengths)
== prime-batch-gate carried-b3-exact (--batch 3 --plen 600 --carried --exact)
   rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
== decode-batch-gate config B=8
   rc=0 ALL GREEN: decode_step_batch exactness battery
== decode-batch-gate strict B=4 equalized
   rc=0 ALL GREEN: decode_step_batch exactness battery
== serve-smoke
   rc=0 ... serve-smoke: 0 failed
== prime-fairness-gate fix
   rc=0 PRIME-FAIRNESS: long=131072 peers=3x1 bytes=yes peer_p95=52.10/1.43s peer_max=52.10/1.43s tick_max=53608/2386ms long_ttft=53.61/56.36s yields=142 engaged=yes -> PASS
```

serve-smoke skipped its spec, gemma4 and Q35 arms (no draft or model on this rig; lines in
`raw/battery/serve-smoke.log`). KV-HOST-SPILL identity was not run: the fix does not touch the
spill path.

Diagnostic (window 7, `raw/battery-base-pbg-window7.log`): the standing `prime-batch-gate --exact`
on the BASE binary (`5c33c228dc657cb6`) is red on this model, `exact-b3-p24 rc=1 Error:
"prime-batch-gate: 6 FAIL(s)"` and `exact-b4-p1100 rc=1 Error: "prime-batch-gate: 8 FAIL(s)"`
(`seq 0: exact logits diff 248320/248320 h_seed diff 4096/4096 hidden diff 98304/98304`). The
exact gate already existed and would have caught #641, but no battery runs it on the 9B: the only
fast-gate rows for it are Step35's `pbatch35`/`pbatch35c`, and local-ci does not call it. The new
`ptick` row is the serving-shape cell; the 9B `--exact` coverage gap is recorded here for the lead.

## Second commit (`53dac5a56`): the batched o_proj f16 epilogue takes the solo guard

The per-sequence attention loop in `prime_cache_batch_inner` fed `wo` the f16 epilogue whenever one
existed, while the solo `full_attn_prime_core` skips it for an artifact carrying an AWQ o_proj
scale (`fa.wo_pqs`, memra#253), because that epilogue never applies the scale. The batch arm now
carries the same `fa.wo_pqs.is_none()` condition. The 9B artifact has no
`attn_output.pre_quant_scale` tensor (its GGUF header carries no `pre_quant_scale` name), so the
added condition is always true on this model and its program cannot move; the change removes a
latent second program (and an unscaled o_proj) for a scaled artifact that also builds f16 mirrors. No such
artifact is on this rig, so that pairing has no receipt here.

Final-tip GPU battery (window 8, `raw/final-window8.log`, tree `53dac5a56` clean, card idle,
`executed-not-qualified`; probe `bd1ec128f602c3c7`, prime-batch-gate `6d405899264cedec`;
serve-smoke re-linked `memra-server` from the same clean tree before booting it, and the fairness
gate used that binary):

```
prime-tick-exact-gate: PASS (every prime shape is bit-identical to the solo prime; log .../raw/final/gate-ptick-naked.probe.log)
ptick naked rc=0
prime-tick-exact-gate: CANARY OK (bp and bps DIFFER with B's batch prompt changed; log .../raw/final/gate-ptick-canary.probe.log)
ptick canary rc=0
   pbg exact-b3-p24 rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
   pbg exact-b4-p1100 rc=0 ALL GREEN: prime-batch gate (batch=4, uneven lengths)
   pbg carried-b3-exact rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
   serve-smoke rc=0 serve-smoke: 0 failed
   fairness rc=0 PRIME-FAIRNESS: long=131072 peers=3x1 bytes=yes peer_p95=52.61/1.54s peer_max=52.61/1.54s tick_max=54112/2456ms long_ttft=54.12/57.92s yields=142 engaged=yes -> PASS
```

The cell 2 fix runs (window 6) used the `58d3a225b` server; the guard commit cannot change the 9B's
program (above), and the final-tip ptick and exact gates are bit-identical to the solo prime.

CPU battery on `53dac5a56` (`raw/final/cpu/`, run under a CPU quota):

```
clippy rc=0      (cargo clippy --release --all-targets -p memra-engine -p memra-server -- -D warnings)
server tests rc=0
  test result: ok. 871 passed; 0 failed; 16 ignored; 0 measured; 0 filtered out; finished in 5.56s
  test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
engine lib rc=0
  test result: ok. 539 passed; 0 failed; 31 ignored; 0 measured; 0 filtered out; finished in 4.50s
fmt rc=0         (cargo fmt --all -- --check)
check-flags rc=0
  check-flags: no uncovered runtime names
  check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)
diff-check rc=0  (git diff --check 9c07b398b..HEAD)
shellcheck rc=0  (tools/prime-tick-exact-gate.sh)
```

## Part 2: the merged tree, the 9B exact gate in the battery, the target card

### Merge with main `d544c6b82` (`2cec07c15`)

No code conflict. `lib.rs`, `FLAGS.md`, `TESTING.md` and `KERNELS.md` merged in separate regions;
`research/INDEX.md` conflicted and is the union of both sides (set-diff against both parents
empty, no conflict marker left). One behavior fix-up: main's `pub mod dsv4_source_tape;` line moves
`lib.rs` by one line, so the `fa3_prefill.cu` row in `KERNELS.md` now cites `lib.rs:28008-28020`
and `lib.rs:2156-2169`. Main's side (#665 `worker.rs`, #668 the spec round guards, #666, #556, the
dsv4 PRs) touches no prime program: `git diff 9c07b398b d544c6b82 -- crates` has no line naming
`prime_cache_batch`, `fa_prefill` or the varlen arm, and `hybrid_forward.rs` changed on this side
only.

### The 9B exact gate joins the battery (`47831292c`)

`tools/prime-batch-exact-gate.sh` runs the three `prime-batch-gate --exact` rows (b3-p24,
b4-p1100, carried b3-p600), unsets `MEMRA_REWRITE_BUNDLE` and refuses a log carrying
`carried-prime.v1 unqualified`. Its `--canary` runs the new `prime-batch-gate --canary` (seq 0's
first token changes inside the batched prime only): seq 0 must differ and seqs 1 and 2 must stay
bit-identical. Wired as fast-gate rows `pbg9`/`pbg9c` (routes: prefill FA `.cu`,
`hybrid_forward.rs`, `prime_batch_gate.rs`, the script) and as a `tools/local-ci.sh` stage after
decode-batch-gate that runs both pairs, `prime-batch-exact-gate` and `prime-tick-exact-gate`, on
the 9B (`MEMRA_CI_PRIME_EXACT=0` skips; FLAGS and TESTING rows). `prime-tick-exact-gate` now
removes its scratch dir on a green run without `--log`. CPU battery on `9e3b7250e`
(`raw/merged/cpu/`): clippy `-D warnings` rc=0, server tests 875 + 7 passed, engine lib 545
passed, fmt, check-flags, `git diff --check origin/main..HEAD`, shellcheck and `bash -n
tools/local-ci.sh` all rc=0.

### Target card: one RTX PRO 6000 Blackwell, the 27B (`pro6000/`, `executed-not-qualified`)

Order: the PRO card ran first (07:39:40 to 07:44:35 UTC), while lane B's order held the local 5090;
the 5090 cells ran after (next section). One collector hold of `/tmp/memra-gpu.lock`
(`pro6000/box/collector/`, lock proof `pro6000/box/out/LOCK.json`), card idle at start and end.
RTX PRO 6000 Blackwell Server Edition, 600 W limit, driver 580.178.04, nvcc 13.2 on the box. Model
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, sha256 `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
Fix tree `9e3b7250e`: `prime-batch-gate` `2d595076`, `concat-prime-probe` `9ec68ead`,
`memra-server` `92576099`. Base tree main `d544c6b82`: `prime-batch-gate` `487d87ff`. Raw:
`pro6000/run.log` (the box run log plus the done signal), `pro6000/box/out/`.

Fix, green:

```
    exact-b3-p24 rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
    exact-b4-p1100 rc=0 ALL GREEN: prime-batch gate (batch=4, uneven lengths)
    carried-b3-exact rc=0 ALL GREEN: prime-batch gate (batch=3, uneven lengths)
prime-batch-exact-gate: PASS (every batched prime is bit-identical to its individual prime)
prime-batch-exact-gate: CANARY OK (seq 0 DIFFERS: exact logits diff 248320/248320 h_seed diff 5120/5120 hidden diff 122880/122880; seqs 1 and 2 bit-identical)
    arm bp prime: logits bitdiff=0 maxabs=0.000000e0 argmax ref=82 arm=82 | h_seed bitdiff=0 | hidden bitdiff=0 maxabs=0.000000e0 first_row=None | cache digests differ 0/130 first=[]
    tickshape verdict: ALL ARMS EXACT
prime-tick-exact-gate: PASS (every prime shape is bit-identical to the solo prime; log /root/e641/out/ptick-naked.probe.log)
prime-tick-exact-gate: CANARY OK (bp and bps DIFFER with B's batch prompt changed; log /root/e641/out/ptick-canary.probe.log)
SPEC-CTX-EDGE GATE: ALL GREEN
```

The ptick gate applies: the 27B loads through `HybridModel` (the qwen hybrid family), 130 cache
digests. spec-ctx-edge ran with `SCE_CTX=384` as pre-registered (the 27B's prompt is 73 tokens,
under the 128-token switch): 13 PASS, 0 FAIL, door ON r1 to r4 `completion_tokens=64
expected=64`, `SCE (plain) message equals the on arm's r1: plain=0ba35b0ee89ade1d
spec=0ba35b0ee89ade1d -> PASS`, door OFF r1 to r3 `completion_tokens=309/311/309 room=311`, every
boot census `panicked=0 argmax_sentinel=0 worker_fatal=0 respawn=0 verify_refused=0`.

Base (main `d544c6b82`, the varlen arm live), red on this card as on the 5090:

```
   pbg-base-exact-b3-p24 rc=1
    seq 0: exact logits diff 248320/248320 h_seed diff 5120/5120 hidden diff 122879/122880
    Error: "prime-batch-gate: 6 FAIL(s)"
   pbg-base-exact-b4-p1100 rc=1
    Error: "prime-batch-gate: 8 FAIL(s)"
   pbg-base-carried-b3-exact rc=1
    Error: "prime-batch-gate: 8 FAIL(s)"
```

Cost A/B, `prime-batch-gate <27B> --batch 3 --bench 1024`, 6 pairs AB BA AB BA AB BA (A = base),
one process per run, N=6 per arm, each figure that process's median of 5 alternating serial/batch
reps. Regime: 250 ms telemetry (`pro6000/box/out/telemetry-250ms.csv`), SM 2355 to 2422 MHz,
42 to 61 C, up to 503 W across the A/B window; run starts 45 to 51 C.

| arm | batch_wall_ms median (N=6) | min | max | serial_wall_ms median | batched vs serial, median |
|---|---|---|---|---|---|
| base (main, varlen arm live) | 792.487 | 792.453 | 792.534 | 818.946 | +3.3% |
| fix (per-seq core) | 790.772 | 790.756 | 790.811 | 818.926 | +3.6% |

Paired fix minus base, per pair: -1.711, -1.735, -1.676, -1.678, -1.755, -1.718 ms; median
-1.715 ms (-0.22%). By the pre-registered rule this is flat (under 1%), with the direction plain:
on this card the fix is faster in all six pairs, both orders, by about 1.7 ms of 792. The removed
varlen arm was not a PRO 6000 win at B=3, T=1024 on the 27B.

## Still owed

- A varlen FA twin that attends the dequantized cache view (the solo program) could come back
  only with its own bit-identity receipt against the solo prime. Neither card shows a wall-time
  loss from removing the old one at B=3, T=1024 (5090 9B: +0.19%, flat; PRO 6000 27B: -0.22%,
  flat, the fix faster in 6 of 6 pairs), so the twin has no measured gap to recover at this shape.
- No H100 re-measure. The sm_90a `memra_fa3_vl` twin is deleted with the door; Hopper batched
  primes now run the per-seq core too.
- The AWQ o_proj scale plus f16 mirror pairing (`53dac5a56`) has no receipt: no scaled artifact is
  on either rig.
