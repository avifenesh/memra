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
