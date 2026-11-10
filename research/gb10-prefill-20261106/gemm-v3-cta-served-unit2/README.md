# F8F4 v3 served on GB10 unit 2: #1102 confirmed, the CTA-form follow-up measured (2026-11-08)

Ledger rows 43 and 45. One campaign ran cells B L N N L B on one DGX Spark (unit 2), one fresh
boot per cell and one cold prompt per length per boot. The model was Qwen3.8-27B NVFP4+Q5_K GGUF
with the masked MTP draft, at 262,144 context.

- **B:** #1102's binary with `MEMRA_MMQ_F8F4_V3=0` (the v2 mainloop).
- **L:** #1102's binary (`landed-SHA256SUMS`, built from `173cd71c57`). Its device code is that of
  `4924aeddf2`, the binary of #1102's unit 1 A/B.
- **N:** the follow-up's binary (`followup-SHA256SUMS`, `476818d451`). It has the CTA-local copies
  and gate/up on v3, with the same server code as L.

Cell 1 ran 10:54-11:05 in a first directory (`campaign-part1.log`). The campaign then paused at that
cell boundary for another lane's landing record and resumed at 11:51 for cells 2-6
(`campaign-part2.log`, `pause-campaign.sh`, `b23d-fup2.sh`). The six cell directories here are
numbered in campaign order. `summary.txt` is `summarize.py` over all six.

| arm | 32,784 tokens | 131,051 tokens | 259,951 tokens |
|---|---|---|---|
| B, v2 mainloop | 24.405, 24.371 s | 127.080, 127.985 s | 333.763, 333.204 s |
| L, #1102 | 24.054, 24.047 s | 126.137, 126.144 s | 330.659, 330.153 s |
| N, follow-up | 23.054, 23.019 s | 121.990, 121.979 s | 322.052, 322.270 s |

- **L against B:** -1.38% / -1.09% / -0.92%. Unit 1 had -1.51 / -1.01 / -0.89%.
- **N against L:** -4.22% / -3.29% / -2.50%.
- **N against B:** -5.54% / -4.35% / -3.40%.
- The answer text is identical in all six boots, all needles were found and no request failed.

`argmax16k-v3/` is the 16k argmax-margin gate on N, which PASSED: flips 3, budget 3, every flip
margin-explained. Its probe table (the gate log without timing lines, before the home path was
written as `~`) hashes to `a42e74b3f28cbd96`. That is the hash of #1102's tables on unit 1, under
both mainloops.

> **Note, 2026-11-10 (memra#1199).** "Every flip margin-explained" is true of any flip and is
> not evidence. What this cell shows is 3 flips against a budget of 3 and a probe table
> byte-identical to #1102's. See `docs/decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md`.
