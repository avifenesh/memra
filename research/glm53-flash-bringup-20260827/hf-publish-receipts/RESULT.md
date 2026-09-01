# Gate re-run RESULT: PASS on the primary bar

Run on the upload box, over `~/models/glm53-nvfp4`, which is the directory the Hub
upload reads from. No copy sits between the gated bytes and the Hub.

- Binary: `glm5_checkpoint_runner` (`crates/memra-reference`), release, built on the
  upload box from the lane head. `--self-test` PASS (streamed trunk matches `execute()`
  bit-for-bit).
- Invocation: `MEMRA_ORACLE_OUT=<tsv> glm5_checkpoint_runner ~/models/glm53-nvfp4 1 2 3 4`
- Wall time: 2670.4 s, 45 trunk layers streamed, 154,880 last-position logits written.
- Output: `nvfp4-oracle-rerun.tsv` beside this file.

## Primary bar: PASS

```
cmp nvfp4-oracle-rerun.tsv ../mint-receipts/nvfp4-oracle.tsv  ->  IDENTICAL
```

The re-run is **byte-for-byte identical** to the banked gate receipt, all 3.3 MB of it.
This is the strongest available outcome and it settles the coverage question directly:
the banked `MINT GATE PASSED` receipt is not merely consistent with these bytes, it is
reproduced by them, on a different box, at a later engine commit. The gate now covers
the exact artifact being published rather than a directory that shared its name.

## Fallback bar, computed anyway as a cross-check: also PASS

Independent of bit identity, the gate's own bar re-derived on the re-run output:

| comparison | argmax | top-k rank-identical | max_abs | mean_abs |
|---|---|---|---|---|
| re-run NVFP4 vs BF16 twin (the mint's own source) | MATCH (id 5) | top-3 | 3.117 | 0.534 |
| vendor FP8 vs BF16 twin (the calibration row) | MATCH (id 5) | top-3 | 3.489 | 0.490 |
| re-run NVFP4 vs vendor FP8 | MATCH (id 5) | top-6 | 4.184 | 0.705 |

Our 4-bit mint's deviation from full precision (3.117 / 0.534) sits inside the vendor's
own 8-bit deviation from the same source (3.489 / 0.490) on max_abs, at half the bit
width. The calibration row is why the first row means anything: an absolute logit delta
has no interpretation without a same-instrument reference point.

Both pre-registered bars are met. The acceptance rule was written into GATE-COVERAGE.md
before this run produced output.

## Bytes bound to this result

`SHA256SUMS.txt` beside this file, 30 entries, re-verified with `sha256sum -c` on the
upload box AFTER the reboot and immediately before upload. 28 of the 30 are published;
the vendor's own `README.md` and `config.json.pre-keeplist-fix` are not.

## What this result does NOT claim

Unchanged from the original gate, and repeated here so the card cannot quietly widen it:
serving accuracy over long generations, long-context behaviour, sampled decoding
quality, and the engine's fused 4-bit kernels are all outside it. The runner dequantizes
to f32; a serving path riding 4-bit matmul is a separate gate that has not been run.
The vision tower is present in the architecture and is not exercised here.
