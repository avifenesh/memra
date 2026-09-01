# CANARY arm: the `seq_end` defect isolated, with no m-dependence confound

`MEMRA_STEP_GEMM_PRIME_SUFFIX=1 MEMRA_STEP35_PRIME_BATCH_TSEND=1` — the batched suffix arm with
the pre-fix chunk-local `seq_end` restored. Perf is unchanged from the ON arm, as it must be
(`0.3062 s + 0.7309 ms/tok` vs ON's `0.3060 + 0.7286`; ratio median 3.391x vs 3.388x): the seam
moves an attention MASK, not any work.

## The clean isolation: LEG F

| arm | LEG F sha (fresh 4338-token prompt) |
|---|---|
| off | `7c8c50730a683b43` |
| on | `84ddc832a39c92ff` |
| canary | `41f9b3df5e88909b` |

**canary != on is the whole demonstration.** The canary and ON arms have IDENTICAL chunk
decomposition — same doors, same `m` at every chunk, same implementation — and differ in one
thing only: the `seq_end` value handed to `step35_attn_pre_wo`. So this pair isolates the
defect with none of the m-dependence confound that muddies off-vs-on. A fresh 4338-token prompt,
no session reuse anywhere, changes its answer purely because a sub-512 trailing chunk selected
the unwindowed FA arm. That is the live bug, reproduced on demand.

It also retires the worry pre-registered in PREDICTION.md that a canary MATCH would be
ambiguous between "seam not read" and "defect not byte-visible": the seam is provably read.

## The predicted window-straddle signature appears

| row | on | canary | |
|---|---|---|---|
| s-warm/s0700 | `fe32811b97ed12c7` | `fe32811b97ed12c7` | **MATCH** |
| s-warm/s0030 | `f2af5e80a515b53c` | `a16398da8222ea5f` | differ |
| s-warm/s0250 | `27b487bd3778b7c7` | `1b0f04b6fee948d9` | differ |
| s-warm/s0450 | `8a55858e859cc770` | `04b625b84ad8703c` | differ |
| s-warm/s1200 | `c689f639d10a35f4` | `15a7c5631e8cfe84` | differ |
| s-warm/s4400 | `d11c58590a32c93a` | `571ee46c005104ba` | differ |

s0700's warm suffix is `t=704 base=1440` plus `t=36 base=2144`. The 704-token chunk exceeds
win=512 under both values, so it takes the same arm either way; the 36-token tail differs in
arm, but at `base=2144` the trim gives `off = (2144-511) & !31 = 1632` and `t_kv = 548`, so only
a single key (1632) is forbidden-but-unmasked for the earliest query. One key was not enough to
move a token. That is the perturbation-scales-with-length behaviour pre-registered before the
run, showing up where predicted.

## Instrument caveat the cross-arm table exposed, and it must be stated

`s-cold/s0700` in the canary arm and `s-cold/s0250` in the ON arm both hash to
`808b4df7b0aa772d`. That is NOT a hash collision — a 64-bit collision across ~90 samples is not
plausible — it is genuinely the same completion text for two different prompts.

The reason is a property of this sweep's construction: LEG S prompts differ ONLY in how many
meaningless filler words ("alpha beta gamma …") are appended to the SAME question. The model
answers the same question the same way regardless of filler length, so cross-row sha equality is
expected and carries no information.

Consequences, both ways:

* **Only within-pair comparisons are valid** (warm vs cold at the same sweep point, identical
  bytes, verified by identical `seq_end`). That is exactly how the verdict uses them, and the
  driver's "6 distinct completions over 6 swept prompts" line is a weaker safeguard than it
  looks — it is luck, not a guarantee.
* **It sharpens the identity finding.** Output here is insensitive to appending hundreds of
  filler tokens, yet the SAME prompt primed two ways produces different text with different
  lengths. The numeric divergence between chunk shapes is therefore not a marginal near-tie
  flip; it moves the answer further than a large prompt edit does.
