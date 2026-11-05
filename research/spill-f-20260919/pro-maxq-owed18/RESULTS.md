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
- The runner set GPU 0 to 300 W (from 250 W, max 325 W), read it back (`card.json`, every
  `card-telemetry.csv` row) and restored 250 W after the cell (`runner/power-set-handoff-direct.log`).
- Throttle summary (`runner/power-handoff-direct.txt`, 218 samples at 10 s): maximum 83 C, fan
  54 %; SW power cap active in 109 samples, SW thermal slowdown, HW thermal slowdown and HW power
  brake in 0; no thermal stop. The power cap is the fill prompts' prefill reaching the 300 W limit;
  the amendment records it and does not judge it.

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

Scope: this card class on this storage class (ext4 on a local PCIe NVMe), at a 300 W limit in this
chassis, power cap included. A box whose handoff path is on other storage is outside the receipt.
