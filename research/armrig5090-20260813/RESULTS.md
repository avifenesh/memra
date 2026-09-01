# RTX 5090 eager-arm cost — result

Date: 2026-08-13

## Verdict

**Keep both eager paths OFF by default on this RTX 5090 class.** EAGER failed the
required byte-identity gate at restored c=1, intermittently at restored c=16, and on
every EAGER restored-hit cell in the mixed-cache money shape. The only valid
performance comparison was restored c=4, where EAGER was inside the observed spread
and therefore **FLAT**.

This campaign did not observe early EOS or truncation: all full-hit requests completed
512 tokens and all mixed requests completed 60 tokens with `finish_reason=length`.
The failure was output divergence, so throughput is withheld for every affected cell.

## Scored table

Primary full-hit metric: aggregate tokens after the first visible token divided by the
shared decode window. Each arm is N=5 fresh-server runs. Spreads are min..max. Delta is
the EAGER median versus the REPAIRED median.

| Model / shape | Concurrency | REPAIRED median tok/s [spread] | EAGER median tok/s [spread] | Delta | Exactness-gated verdict |
|---|---:|---:|---:|---:|---|
| Q27 NVFP4, fully restored, 512 out | 1 | WITHHELD | WITHHELD | WITHHELD | BYTE MISMATCH in 5/5 pairs; all 5 EAGER requests differed from the stable REPAIRED hash |
| Q27 NVFP4, fully restored, 512 out | 4 | 84.98 [73.61..85.24] | 85.17 [84.98..85.25] | +0.224% | BYTE IDENTICAL in 5/5 pairs; ranges overlap: **FLAT** |
| Q27 NVFP4, fully restored, 512 out | 16 | WITHHELD | WITHHELD | WITHHELD | BYTE MISMATCH in 2/5 pairs; 2/80 EAGER requests diverged |
| Q27 NVFP4, mixed 90% hit / 10% miss, 4860+60 | 4 | WITHHELD | WITHHELD | WITHHELD | All 90/90 EAGER restored hits disagreed with their own cold-seed golden; REPAIRED was 0/90 |

For the valid restored-c=4 cell, the paired-delta median was +0.126%. The end-to-end
window independently remained flat at +0.235% by median. REPAIRED repetition 5 was a
73.61 tok/s low observation; it completed correctly and no failure was captured, so its
cause is unknown and it remains in the published spread.

## Exactness findings

The stable c=1 divergence repeated in all five interleaved pairs. The first scored pair
reported, verbatim:

```text
BYTE MISMATCH across A/B pair: {'full-q27-c1-r1-repaired': ['210933f9d1c2e8e111b6a633829934d518f149d5c92204ffa70d1933f2e1e70e'], 'full-q27-c1-r1-eager': ['c1491b4b22305fb2e74aa77fb111a7a4dfb6f4832c69a1239210be2831dbf1e3']}
```

At c=16, EAGER request 12 in repetition 1 and request 13 in repetition 4 produced
`3130013ced5bbd4c62067c13eef0909ea0c029aaa00d8ef58f56396dbaceba10`
instead of the otherwise stable
`210933f9d1c2e8e111b6a633829934d518f149d5c92204ffa70d1933f2e1e70e`.
Both divergent requests still completed the full 512-token budget.

The mixed-cache result is the strongest crossing receipt. Every REPAIRED cold seed and
all 90 REPAIRED restored hits agreed. Under EAGER, each of the five cells recorded
`"golden_mismatches_observed": 18`: all 90 restored hits disagreed with the cold seed
created under that same EAGER process. Four of five A/B request pairs also had different
output sets; repetition 5 happened to agree across arms, but still failed its internal
EAGER seed-to-restored identity gate.

Because this A/B deliberately enables B1FAST and GraphSession together, it establishes
that the combined EAGER policy is unsafe; it does not assign the divergence to one door
individually.

## Protocol and thermal regime

- One binary for both arms, built from `57ebcf8d319dc8ea9bb351b39fc1ab28d18c20db`
  (engine base `01df75ac2`): SHA-256
  `1b460e29f642b93e86ed287c144129e6c5b3a6c1cca56abf2aacff92393b809c`.
- REPAIRED left `MEMRA_SERVE_B1FAST` and `MEMRA_SERVE_GS` unset. EAGER set both to
  `1`. Activation logs contain the expected GraphSession graph census.
- One uninterrupted `/tmp/memra-5090.lock` hold from 07:41:05Z through completion of
  the last scored point at 08:26:01Z. Odd repetitions ran REPAIRED first; even
  repetitions ran EAGER first. No artificial cooldown was inserted.
- Owner-declared thermal clock cap: 210–1200 MHz. No clock-changing command was run.
  Continuous 250 ms telemetry captured 10,701 samples; busy samples reported
  180–1200 MHz, the ceiling never exceeded 1200 MHz, temperature was 50–67 C, and
  power was 21.42–102.68 W. These numbers are relative-only within this rig and are not
  comparable to PRO 6000 absolutes.
- Q27 was selected because it fits the 24 GB card and exercises both eager doors. The
  campaign peaked at 21,656 MiB used. c=16 passed a 16/16 x 512 qualification and is the
  highest native exact decode chunk class; wider request sets would be scheduler-chunked.
- The initial post-run reduction rejected NVIDIA telemetry's `[N/A]` power-limit field
  after all GPU measurements had completed. The parser was corrected and the existing
  raw window was reduced without rerunning or mixing thermal holds.

## 5090 recommendation

Do not add a 5090-class device-detected EAGER default from this evidence. At the sold
c=4 restored shape, the repair is free within spread. At c=1, c=16, and mixed-cache c=4,
the required output-identity gate failed, so no valid speed win exists to trade against
correctness.

Any future requalification should isolate B1FAST and GraphSession, then require an
immutable per-request numerical-program choice. Promotion may happen only at admission
when lifetime-solo execution is guaranteed; once any peer is queued, promotion must be
refused, and a request must never demote or switch programs mid-flight. Device detection
belongs only after that crossing proof and same-prompt golden gates pass; the env vars
remain rollback and measurement seams.

## Evidence

- Reduced machine-readable result: `raw/attempt2-local5090/summary.json`
- Full request and point ledger: `raw/attempt2-local5090/full-points.jsonl`
- Mixed request, seed, and point ledger: `raw/attempt2-local5090/mixed-points.jsonl`
- Continuous thermal telemetry: `raw/attempt2-local5090/gpu-250ms.csv`
- Pair-gate quotes: `raw/attempt2-local5090/exactness/`
- Full campaign transcript and initial reducer failure: `raw/attempt2-local5090/driver.log`
- Post-run successful reduction: `raw/attempt2-local5090/reduce-fixed.log`
