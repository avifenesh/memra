# BOX36 PRO 6000 sitting: OWED 17, 18 and 26 on the target card

One RTX PRO 6000 Blackwell Workstation card, Ryzen 9 9950X host, 123 GB RAM, host-local xfs NVMe
at `/scratch`. Run by the lead from lane tip `114ef768d` (`commit.txt`), 2026-09-26 13:08Z to
14:43Z, `pro-sitting/run-sitting.sh`: `SITTING DONE`, every step rc=0. Receipts mirrored raw
into the private store (984 files checked against the box-side manifest) and exported here by
`../m1-box-sanitize.py` (16 files carried the rented volume id; `EXPORT-MANIFEST.json`).
Card state in every timed visit: 42 to 44 C, SM clock median 2,865 to 2,872 MHz, no throttle.

## Proof and OWED 26

- M1 proof PASS, `nvme-local-direct` (`m1-proof/`).
- OWED 26 GPU cells: `test result: ok. 7 passed; 0 failed ... finished in 0.32s` (`owed26-cells/`),
  including the four OWED 26 cells; the start and end fall in the same second because the seven
  cells take 0.32 s.
- Serving shape on the target card: zero fallbacks and zero demand-wait timeouts in all 63 run-gen
  visits (smoke, cold, bounded).

## OWED 17: cold read-once bypass (`MEMRA_MOE_COLD_BYPASS`)

Correctness: the smoke's three arms generated 128 tokens equal to the byte oracle file (the same
ids as BOX27 and the 5090); `mapped_serves=667196` on the mapped arm; `=== SELF-CONSISTENCY PASS ===`
for `bypass-staged` and `bypass-mapped` (run-spec K=1..8).

Registered verdicts (`f17-cold.pool.log`, `f17-bounded.pool.log`), fallback visits unclean:

```
cold    M1-5090-VERDICT arm=bypass-staged vs worker16: flat median_ratio=1.0017673731075383 pairs=10
cold    M1-5090-VERDICT arm=bypass-mapped vs worker16: loser median_ratio=0.8725199069985191 pairs=10
bounded M1-5090-VERDICT arm=bypass-staged vs worker16: flat median_ratio=1.0095147478591817 pairs=9
bounded M1-5090-VERDICT arm=bypass-mapped vs worker16: loser median_ratio=0.8957934990439769 pairs=9
```

(The `M1-5090` prefix is the pool tool's line label; these are the target-card cells.) Both
regimes scored; zero contaminated visits in cold, one worker16 visit in bounded.

| Regime | worker16 | bypass-staged | bypass-mapped |
|---|---|---|---|
| cold, median tok/s (N=10) | 22.67 (22.52 to 22.84) | 22.69 (22.48 to 23.06) | 19.79 (19.66 to 20.01) |
| bounded, median tok/s (N=10) | 10.54 (9.67 to 16.18) | 10.68 (10.54 to 13.93) | 9.45 (9.30 to 9.51) |

Reading: on the target card, reading a cold block in place over PCIe is 10 to 13% slower than
copying it into a slot first, in both regimes. Skipping admission for first misses (`staged`) is
neither better nor worse. The door stays default off; its decision waits for the 5090 row
(section F: both rigs before a default), and a loser on both rigs deletes the value.

## OWED 18: KV handoff O_DIRECT (`MEMRA_KV_HOST_HANDOFF_IO`)

All 40 cycles passed (probes restored with text identical to cold, entries imported equal
exported, `io=` lines). Registered verdicts, direct over buffered within each pair
(`handoff-1g.pairs.log`, `handoff-8g.pairs.log`):

```
M1-HANDOFF-VERDICT handoff-1g export_ms direct/buffered: winner median_ratio=0.917038670189261 pairs=10
M1-HANDOFF-VERDICT handoff-1g write_ms direct/buffered: loser median_ratio=1.2153829424597005 pairs=10
M1-HANDOFF-VERDICT handoff-1g fsync_ms direct/buffered: winner median_ratio=0.0021578316851375017 pairs=10
M1-HANDOFF-VERDICT handoff-1g import_s direct/buffered: flat median_ratio=1.0 pairs=10
M1-HANDOFF-VERDICT handoff-8g export_ms direct/buffered: winner median_ratio=0.9164044948115526 pairs=10
M1-HANDOFF-VERDICT handoff-8g write_ms direct/buffered: loser median_ratio=1.2116343093147717 pairs=10
M1-HANDOFF-VERDICT handoff-8g fsync_ms direct/buffered: winner median_ratio=0.0009088471883929809 pairs=10
M1-HANDOFF-VERDICT handoff-8g import_s direct/buffered: flat median_ratio=1.0434782608695652 pairs=10
```

| Cell | Arm | Export ms | write ms | fsync ms | Import s |
|---|---|---|---|---|---|
| 1 GiB (17 entries, 2,161.8 MB) | buffered | 1,459.5 | 946.9 | 325.4 | 1.6 |
| | direct | 1,338.0 | 1,151.0 | 0.7 | 1.6 |
| 8 GiB (76 entries, 9,664.6 MB) | buffered | 5,831.5 | 4,255.8 | 1,393.4 | 6.9 |
| | direct | 5,344.0 | 5,155.9 | 1.3 | 7.2 |

Medians of 10 per arm. The buffered arm reproduces BOX27's B2 on a different box (1,456 to 1,460 ms
and 5,772 to 5,898 ms). Reading: `O_DIRECT` removes the writeback at fsync but pays part of it back
in a slower write, for a net 8.3% (1 GiB) and 8.4% (8 GiB) shorter export; import is unchanged
(its clock is the DONE line's 0.1 s resolution). No single decision metric was registered;
the end-to-end export time is where the handoff's storage cost lands, and the default waits for
the 5090 row.
