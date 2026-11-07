# fp16 GDN recurrent state on GB10: magnitude census, then the fp16 decode class (2026-11-07)

Model: Qwen3.8-27B (GGUF NVFP4+Q5_K trunk, masked sxc32768 MTP draft, the `gguf-mtpmask` serving
arm). Hardware: GB10 (DGX Spark, `121a`). Branch `research/gdn-state-fp16`, base main
`05a8a63503`. Follows the two rejected bf16 forms (`research/gdn-state-bf16-storage`, `f41085a693`;
`research/gdn-state-bf16-decode`, `7826562451`; `docs/decisions/GDN-STATE-BF16.md` there): both
scored long HumanEval 155/164 against a same-binary fp32 158 (band 156-159), and removing every
prime rounding recovered nothing, so the loss sits in the per-token rounding of decode and verify.
fp16 has the same bytes and a 10-bit mantissa against bf16's 7, so each rounding is about 8x finer,
but its range is small: 65504 at the top, subnormals (multiples of 2^-24) below 2^-14 = 6.1e-5,
and zero at or below 2^-25 = 3.0e-8. Step 1 asks whether the state fits that range; step 2 runs the
fp16 decode class only if it does.

## Step 1: magnitude census

**Verdict: fp16 holds the state.** No element of any sample comes within 600x of fp16's largest
value, and the subnormal range costs precision only in the smallest heads, where one fp16 rounding
is still at least 5.2x finer than one bf16 rounding.

### Instrument

`MEMRA_GDN_STATE_CENSUS=<dir>` (memra-server, `crates/memra-server/src/gdn_census.rs`, commit
`a8db6a5dc3`; a diagnostic, the numeric program is unchanged). Between worker ticks it copies each
live session's GDN states (`RecurLayer::ssm_state`: 48 layers x 48 value heads x 128 x 128 fp32) to
the host and records per layer and per head:

- the largest |x|, against 65504, and elements that would overflow;
- the fraction in fp16's subnormal range (0 < |x| < 2^-14), the fraction that would flush to zero,
  exact zeros, non-finite values;
- the share of the head's energy (sum of x^2) carried by the subnormal-range elements;
- the relative L2 error of one fp16 rounding and of one bf16 rounding of the head
  (`||round(S) - S|| / ||S||`), which folds the mantissa and the subnormal loss into one number;
- the head RMS, and a log2 |x| histogram per layer.

Per head because the GDN output goes through a per-head gated RMSNorm: a head with a small state is
scaled back up, so its relative error reaches the output whatever its absolute size.

Cadence: a session is sampled when first seen, every 10 s while it primes (the cooperative prime
walker yields between chunks, so mid-prime states are caught), on its first decode tick, and every
128 committed rows of decode. A sample costs about 24 ms (3 ms copy, 21 ms statistics on 16 host
threads; the first one 1 s). The host rounding functions agree with the device's
`__float2half_rn` and `__float2bfloat16_rn` on 49.8M random finite fp32 values and on every finite
fp16 value (`scripts/xcheck.rs` run on the rig, plus the module's own tests).

The decision rule was fixed before the run: fp16 holds if no sample overflows and the energy below
fp16's normal range is negligible, which shows as every head's fp16 rounding error staying near
the normal-range value. A per-layer or per-head scale is considered only if a few heads leave the
range.

### Run

Binary `arm-2d6ce42cf5` (`121a`: the census commit plus the fp16 decode class, served in the fp32
class with `MEMRA_GDN_STATE_FP16=0`), one fresh boot of the 264-task long-context panel
(`qual_panel.py`): the 131,550-token prime, its 64-token answer, then the long HumanEval tasks,
each a prefix hit on the 131K entry with its own tail prime and decode, 8 concurrent. The boot was
stopped after 78 of the 164 HumanEval tasks to free the unit for another lane's landing record; the
census counts only states, so the partial panel still covers the long prime, the decode after it
and 90 task sessions at 131.6K-132.3K rows. Receipts: `census/panel/` (`census.jsonl`, one row
per sample; `agg.json`, per layer and head extremes; `summary.md`; `server.log`; the partial
panel's records under `panel-partial/`, not a quality score).

### Numbers (263 samples, 91 sessions, 9.93e9 element samples)

| quantity | value |
|---|---|
| largest \|x\| (any layer, head, sample) | 101.7 (layer 0, head 9, at 110,592 prime rows); every other layer below 7.8 |
| fp16 headroom | 644x; overflows 0; non-finite in fp32 0 |
| elements in fp16's subnormal range | 10.6% of all (2.0% to 47% per layer: the early layers carry the small values) |
| elements that would flush to zero | 1.2e-4 of all; each is at most 3.0e-8, so the energy lost is below 1e-7 of any head's |
| worst head energy share below 2^-14 | 6.8% (L2 share 26%), layer 1, a head with RMS 8.6e-5 |
| smallest head RMS in any sample | 8.6e-5 (layer 1) |
| one fp16 rounding, relative L2 error, per layer | median 2.07e-4, max 2.75e-4 |
| one bf16 rounding, relative L2 error, per layer | median 1.65e-3, max 2.09e-3 |
| one fp16 rounding, worst head in any sample | 4.13e-4 (layer 2) |
| one bf16 rounding, worst head in any sample | 3.37e-3 (layer 2) |
| bf16/fp16 error ratio per head (worst over samples) | median 8.00, min 5.87; per layer and sample median 8.00, min 5.24 |

The typical element is small: the log2 |x| histogram peaks at 2^-11 to 2^-10 (31% of elements) and
17.5% sit below 2^-13. That is why a tenth of the state is subnormal in fp16, and also why it does
not matter much: in the smallest heads the subnormal spacing (6e-8) is about 1/1400 of the head
RMS, so those heads round at roughly twice the normal-range error, still 5.9x below bf16. Layer 0's
largest element grows with depth in the prime (48.7 at 4,096 rows, 81 at 61K, 101.7 at 110K) and
is 62 to 67 in the task sessions after it.
