# OWED 18 on the RTX PRO 6000 Blackwell Max-Q: the KV handoff O_DIRECT export (2026-11-05)

One RTX PRO 6000 Blackwell Max-Q Workstation Edition (GPU 0 of a four-card box, driver 615.71.09),
ext4 on a whole-disk PCIe NVMe. The cell is `M1-PREREG.md` section H and its two power amendments,
registered 2026-11-04 and 2026-11-05 before any Max-Q boot. Kit `kit/run.sh` from the deciding
tree `114ef768d` (`kit/SOURCE_SHA`, `commit.txt`), binaries rebuilt from that tree
(`binaries.sha256`, `kit/PROVENANCE.txt`), run 2026-11-05 04:04:31Z to 04:40:57Z. Receipts mirrored
file for file (`MIRROR-CHECK.txt`: 505 of 505 `OK` against the kit's `MANIFEST.sha256`) and exported by
`../m1-box-sanitize.py`, which replaced the box name, the local time zone and the remote home path
in nine files (`EXPORT-MANIFEST.json` lists each with its original and exported sha256). The raw
mirror and the run's private part (the proof's raw device fields) stay outside git in the private
store, as for BOX36.

## Proof, box and power

- M1 proof PASS, `nvme-local-direct`, reasons 0 (`m1-proof/`, `m1-proof.driver.log`).
- The slot started with no compute app on any card (`runner/wait-quiet.log`) and ended with GPU 0
  at 2 MiB (`runner/status.txt`). As registered, the runner checked only the card under test during
  the cell; the other three cards were not sampled in between.
- The runner set GPU 0 to 300 W (from 250 W, max 325 W) and read it back (`card.json`). The box
  re-applies its own per-card caps every 10 minutes (a system timer, 250 W on this card), and at
  04:10:12Z it returned the card to 250 W, where it stayed for the rest of the cell; the runner's
  restore after the cell found 250 W already set (`runner/power-set-handoff-direct.log`). So
  `handoff-1g` (04:04:42Z to 04:13:43Z) ran its first 5.5 minutes, about 12 of its 20 cycles, at
  300 W and the rest at 250 W, and `handoff-8g` (04:13:43Z to 04:40:54Z) ran entirely at 250 W. The
  10 s samples read 34 at 300 W and 184 at 250 W; the 250 ms telemetry 1,326 rows at 300 W and
  7,333 at 250 W (`card-telemetry.csv`, `runner/thermal-handoff-direct.csv`). The summary line in
  `runner/power-handoff-direct.txt` reports the limit it read at the start (300 W); the samples
  are the record. The registered rule reads within-pair ratios and does not depend on the limit.
- Throttle summary (218 samples at 10 s): maximum 83 C, fan 54 %; SW power cap active in 109
  samples (9 of the 34 at 300 W, 100 of the 184 at 250 W), SW thermal slowdown, HW thermal
  slowdown and HW power brake in 0; no thermal stop. The power cap is the fill prompts' prefill
  reaching the limit; the amendment records it and does not judge it.

## Verdicts

All 40 cycles passed (export ok, import `skipped = 0`, every probe restored with text equal to the
cold reference), 10 pairs per size, `failed_cycles=[]` and `malformed_pairs=[]` in both cells.
Registered verdicts, direct over buffered within each pair (`handoff-1g.pairs.log`,
`handoff-8g.pairs.log`, `VERDICT.txt`):

```
M1-HANDOFF-VERDICT handoff-1g export_ms direct/buffered: winner median_ratio=0.8602963708794817 pairs=10
M1-HANDOFF-VERDICT handoff-1g write_ms direct/buffered: flat median_ratio=1.0016031327496069 pairs=10
M1-HANDOFF-VERDICT handoff-1g fsync_ms direct/buffered: winner median_ratio=0.010498404600322997 pairs=10
M1-HANDOFF-VERDICT handoff-1g import_s direct/buffered: flat median_ratio=1.0606617647058822 pairs=10
M1-HANDOFF-VERDICT handoff-8g export_ms direct/buffered: winner median_ratio=0.9156872557932629 pairs=10
M1-HANDOFF-VERDICT handoff-8g write_ms direct/buffered: loser median_ratio=1.066101836718885 pairs=10
M1-HANDOFF-VERDICT handoff-8g fsync_ms direct/buffered: winner median_ratio=0.0033601915591593857 pairs=10
M1-HANDOFF-VERDICT handoff-8g import_s direct/buffered: flat median_ratio=1.013157894736842 pairs=10
M1-HANDOFF-VERDICT pro6000-maxq export_ms direct/buffered: winner handoff-1g=winner@0.8602963708794817/10 handoff-8g=winner@0.9156872557932629/10 default=direct card_index=0 power_limit_w=300.00 card="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
```

Per arm, medians of 10 cycles each (`*/visits/cycle-*/cycle.json`):

| Cell | Entries, file | Export ms, buffered -> direct | Write ms | Fsync ms | Wall |
|---|---|---:|---:|---:|---:|
| handoff-1g | 17, 2,161.8 MB | 1,710.0 -> 1,471.0 | 1,233.4 -> 1,227.7 | 221.0 -> 3.1 | 540.6 s |
| handoff-8g | 76, 9,664.7 MB | 6,221.0 -> 5,691.0 | 5,126.1 -> 5,458.6 | 871.0 -> 2.9 | 1,630.8 s |

## Reading, as section H fixed it

`export_ms` reads winner in both counted cells: **default=direct**, the default extends to the
Max-Q. On this card and storage the direct export is 14.0% (1 GiB) and 8.4% (8 GiB) shorter at the
median ratio. As on the Workstation card at 600 W (BOX36: 0.917 and 0.916), `O_DIRECT` removes the
writeback at fsync; here the 1 GiB write is flat instead of slower, and the 8 GiB write is slower
by less (1.066 against 1.212). `write_ms`, `fsync_ms` and `import_s` decide nothing, as registered.

Scope: this card class on this storage class (ext4 on a local PCIe NVMe), in this chassis, at the
box's 250 W cap for all of `handoff-8g` and most of `handoff-1g` (300 W for the first 12 cycles),
power cap included. A box whose handoff path is on other storage is outside the receipt.
