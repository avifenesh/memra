# memra #641: results (lane/decode-exact-641-20260923)

Pre-registration: `DAY1.md` (pushed before the first boot). Rig: the local RTX 5090 Laptop, every
GPU run under `flock /tmp/memra-5090.lock` with an idle-card check (`raw/*.log` first lines). Model:
Qwen3.5-9B NVFP4 MTP GGUF, sha256 `52c9cceb...a8f39de`. All cells `executed-not-qualified`.

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
