# The host-tier handoff export writes with O_DIRECT on the RTX PRO 6000 Workstation Edition, full power and Max-Q (2026-11-04, Max-Q 2026-11-05)

**Chosen:** on the RTX PRO 6000 Blackwell Workstation Edition, the full-power card and the Max-Q
(since 2026-11-05, from its own cell), the host-tier deploy handoff export (`MEMRA_KV_HOST_HANDOFF`) writes its `.tmp` file with `O_DIRECT` through one
4096-aligned 4 MiB buffer: the last block zero-padded, the file truncated to its logical length,
then `fdatasync` and the same rename. Every other card keeps the buffered writer (a 4 MiB
`BufWriter` through the page cache, then `fsync`). The import reads through a 4 MiB `BufReader`
on every card. Both writers produce the same bytes, so either file imports. A filesystem that
refuses `O_DIRECT` at open (`EINVAL`) gets the buffered writer, and the export line reads
`io=buffered (O_DIRECT open refused ...; the export buffered)`. `MEMRA_KV_HOST_HANDOFF_IO` is
deleted. Code: `HandoffIo::for_device` and `HandoffWriter::create_for_export` in
`crates/memra-server/src/handoff_io.rs`.

**Rejected:**

- The buffered export on that card, the previous default: the registered cell read it 8.3% (1 GiB)
  and 8.4% (8 GiB) longer end to end.
- The direct import reader, on every card. It read flat on both rigs (PRO 6000 median ratios 1.0
  and 1.043 at the DONE line's 0.1 s resolution; RTX 5090 1 GiB 0.907, flat), so it is deleted
  with the code only it reached.
- The direct export on the RTX 5090. Its 1 GiB cell read flat (export median ratio 0.843 with the
  pair signs disagreeing; write 1.524, flat) and its 8 GiB cell never ran.
- A loud failure where `O_DIRECT` is refused. That was the measurement door's contract, so a
  cell could not silently measure the buffered program. As a card default it would turn a
  handoff path on a filesystem without a direct path into a cold start after every flip; the
  buffered writer is the program every earlier handoff ran.
- Keeping the env door as a rollback seam. Owner, 2026-11-04: winners with positive receipts
  become defaults keyed to the card each receipt came from, and their switches go.

**Measurement that settled it:** `research/spill-f-20260919/box36/RESULTS.md`, OWED 18, one RTX
PRO 6000 Blackwell Workstation card at 600 W on host-local xfs NVMe, 10 interleaved pairs per
size, all 40 cycles passing (probes restored with text identical to cold, entries imported equal
exported). Direct over buffered, medians within each pair: export 0.917 (1 GiB, 17 entries,
2,161.8 MB: 1,338.0 against 1,459.5 ms) and 0.916 (8 GiB, 76 entries, 9,664.6 MB: 5,344.0 against
5,831.5 ms); write 1.215 and 1.212 (slower); fsync 0.002 and 0.001 (0.7 against 325.4 ms, 1.3
against 1,393.4 ms); import flat. `O_DIRECT` removes the writeback at fsync and pays part of it
back in a slower write, for a net shorter export. The RTX 5090 row:
`research/spill-f-20260919/owed18/5090/handoff-1g.pairs.log`.

| Card | Storage | Export, direct / buffered (1 GiB, 8 GiB) | Write | Fsync | Import |
|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell Workstation, 600 W (`box36/`) | xfs, local NVMe | 0.917, 0.916 winner | 1.215, 1.212 loser | winner | flat |
| RTX PRO 6000 Blackwell Max-Q, 300 W limit (`pro-maxq-owed18/`, 2026-11-05) | ext4, local PCIe NVMe | 0.860, 0.916 winner | 1.002 flat, 1.066 loser | winner | flat |
| RTX 5090, 1 GiB only (`owed18/5090/`) | | 0.843 flat | flat | winner | flat |

The Max-Q row is `M1-PREREG.md` section H, registered before any Max-Q boot with the BOX36 tree,
driver, reader, arms and schedule: all 40 cycles passed, 10 pairs per size, export medians 1,710.0
-> 1,471.0 ms (1 GiB, 17 entries, 2,161.8 MB) and 6,221.0 -> 5,691.0 ms (8 GiB, 76 entries,
9,664.7 MB), buffered -> direct. The card ran at a 300 W limit with the SW power cap active in 109
of 218 samples (the fill prefill) and no thermal or power-brake slowdown.

**What would change it:** the 5090 8 GiB cell reading a winner on the export adds the RTX 5090
class; a Server Edition cell reading a winner adds that card. The win is a storage property as
much as a card one: both PRO 6000 cells ran on local NVMe (xfs on the Workstation card, ext4 on
the Max-Q), and a box whose handoff path is on other storage is outside the receipts.
