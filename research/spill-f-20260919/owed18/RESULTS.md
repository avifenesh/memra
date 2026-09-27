# OWED 18: KV handoff O_DIRECT arm (`MEMRA_KV_HOST_HANDOFF_IO`), results so far

Registration: `../M1-PREREG.md` section E, its 5090 sizing and the sizing correction. Code:
`crates/memra-server/src/handoff_io.rs`, `worker.rs` export and import. Frozen binaries: `build/`.

## Correctness (green)

- Unit cells: `handoff_io::tests` (byte-identical files from both writers at lengths around every
  4096 and 4 MiB boundary, both readers read both files, `O_DIRECT` refusal path) and
  `host_handoff_file_arms_are_byte_identical_and_cross_readable` (frames through both arms,
  cross-read, truncation breaks the stream the same way). Server suite 940 passed.
- Forced OFF and ON on the 5090 (`5090/handoff-1g/round-01`, 1 GiB, spec off, tenant 100%):
  `M1-B2-CYCLE 1 passed=True` (buffered) and `M1-B2-CYCLE 2 passed=True` (direct); 17 entries
  exported and imported, zero skips, all four probes restored 6,496 cached tokens with text
  identical to the cold reference, and each log line carries its `io=` mode.
- The first pair (`5090/refused-handoff-1g-tenant-cap`) failed on sizing, not on the door: with a
  4,096 MB budget the default 50% tenant cap dropped the oldest entry in both arms alike.

## Timing so far (one pair, never a verdict)

| Arm | Export ms | write ms | fsync ms | Import s |
|---|---|---|---|---|
| buffered | 4,353 | 1,186.3 | 2,798.2 | 2.0 |
| direct | 1,385 | 1,025.2 | 1.0 | 2.0 |

The direct arm removes the fsync writeback (the page cache copy never happens), which is the
whole export difference in this pair; import is unchanged. Rounds 2 to 10 and the 8 GiB cell are
queued; the PRO 6000 half needs a target card.
