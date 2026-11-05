# OWED 18 on the RTX PRO 6000 Blackwell Max-Q: pre-registration

Written 2026-11-04, before any Max-Q cycle. Nothing below may change after the first cycle.

## Question

`docs/decisions/HANDOFF-DIRECT-EXPORT.md` makes the host-tier handoff export write with `O_DIRECT`
on the full-power RTX PRO 6000 Blackwell Workstation Edition only, because its deciding cell ran
on a rented Workstation card at 600 W (`research/spill-f-20260919/box36/`). Owner, 2026-11-04:
the RTX PRO 6000 target is the Max-Q. This cell decides whether the default extends to the
"NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".

## The cell: the Workstation cell, replayed

Same tree (`114ef768d95ece38fcdfff9f74dfbc2cce814258`, BOX36 `commit.txt`), same driver
(sha256 `20aa4115...87fd`, equal to BOX36 `identity.json`), same reader, same arms, order, N,
prompts, sizes, host budget and tenant shares as `pro-sitting/run-sitting.sh` step 7:

- `handoff-1g`: `--size-bytes 1073741824`, default tenant share; `handoff-8g`:
  `--size-bytes 8589934592 --tenant-pct 100` (B2 amendment 3). Both `--host-mb 16384`.
- Schedule per cell: 20 cycles, 10 alternating pairs, `buffered,direct` then `direct,buffered`,
  five pairs in each order. The arm is `MEMRA_KV_HOST_HANDOFF_IO` per cycle.
- Each cycle: boot the gate (spec off, ctx 8192, 4 sessions, prefix cache 1024 MB), fill the host
  tier with the 128 registered 6,500-word prompts to the size, SIGUSR1 export, SIGTERM, sha256 the
  file and evict it from the page cache, boot again for the import, four probes against the cold
  reference from one stock `memra-server` boot with the host tier off.

Every difference from the Workstation sitting is in `DEVIATIONS.md`.

## The rule, quoted from the Workstation registration

`M1-PREREG.md` section E (registered 2026-09-26, before any code):

> Verdict per metric on the within-pair ratio direct / buffered: winner at median <= 0.95 with at
> least 4 of 5 pairs below 1 in each order, loser at median >= 1.05 with the same agreement,
> otherwise flat.

`m1-handoff-pairs.py` lines 8-12, the reader that wrote the Workstation verdict (run unchanged):

> Every cycle must have passed (probes restored, text identical to cold, entries imported equal
> exported, the `io=` lines); a cell with any failed cycle gets no verdict. Per metric (export ms,
> its write and fsync parts, import s), the ratio is direct / buffered within the pair: winner at
> median <= 0.95 with at least 4 of 5 pairs below 1 in each order, loser at median >= 1.05 with
> at least 4 of 5 above 1 in each order, otherwise flat.

The cycle pass condition, `M1-PREREG.md` B2 amendment:

> a cycle passes when the export answered ok, the import finished with `skipped = 0`, and every
> probe has `cached_tokens > 0` and text equal to the reference;

The metric that set the Workstation default, `box36/RESULTS.md`:

> No single decision metric was registered; the end-to-end export time is where the handoff's
> storage cost lands

The Workstation default rests on `export_ms` reading winner in both cells (median ratio 0.917 at
1 GiB, 0.916 at 8 GiB, 10 pairs each, all 40 cycles passing).

## The Max-Q decision, fixed now

1. A cell counts only with 10 pairs (20 cycles), `failed_cycles=[]` and `malformed_pairs=[]`.
2. **default=direct** (the default extends to the Max-Q): `export_ms` reads winner in both
   counted cells.
3. **default=buffered** (the code stays, the decision record gains a rejected Max-Q row): any
   counted cell reads `export_ms` flat or loser. One such cell decides it, whatever the other
   cell does.
4. **default=none** (no verdict, the Max-Q row stays owed): anything else. That covers a cell not
   run, a cell cut short by the window, a thermal stop, and a cell with a failed cycle.
5. `write_ms`, `fsync_ms` and `import_s` get their verdict lines and decide nothing, as on the
   Workstation (there, write read loser, fsync winner, import flat).

## Max-Q specifics recorded with the reading

- The card is chosen at run time (`CARD=<nvidia-smi index>`). `card.json` records its index,
  name, `power.limit`, `power.max_limit` and the enforced limit. On 2026-11-04 <box> reads
  250/250/280/280 W for GPUs 0-3 (max 325/300/300/325 W). The owner has not changed them, and
  the kit does not. The reading is the Max-Q row at the recorded limit; the verdict line carries it.
- The runner refuses unless `nvidia-smi -i $CARD` names exactly
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", shows under 2000 MiB used and no
  compute apps.
- Storage: <box>'s scratch is ext4 on a whole-disk PCIe NVMe (Samsung 9100 PRO 8TB) that also holds
  the box's data. BOX36 was xfs on a local NVMe. The decision record: the win "is a storage
  property as much as a card one", and other storage "is outside the receipt". A Max-Q winner on
  <box> covers this card class on this storage class. A flat or loser may come from the filesystem
  rather than the card, and the row must say so.
- Quiet box: nothing else runs on <box> during the cell (the window rule; run-sitting.sh "Nothing
  else may run on the box"). The runner checks only the card under test.
- The owner's Max-Q thermal rule: stop only when the card reads >= 92 C with its fan >= 95 %,
  three samples in a row. A stop means no verdict.

## Output

`maxq-verdict.py OUT` runs the registered reader on each cell (its lines are written to
`handoff-1g.pairs.log` and `handoff-8g.pairs.log` exactly as run-sitting.sh wrote them), then prints
one line, which `run.sh` puts last in `OUT/VERDICT.txt`:

```
M1-HANDOFF-VERDICT pro6000-maxq export_ms direct/buffered: <winner|loser|flat|no-verdict> handoff-1g=<verdict>@<median_ratio>/<pairs> handoff-8g=<...> default=<direct|buffered|none> [(why)] card_index=<i> power_limit_w=<W> card="<name>"
```

The same reader on the Workstation receipt (`box36/`, from the repo at `2ccb038d1c`) reproduces
both registered `.pairs.log` files and `handoff-pairs.json` byte for byte, then prints:

```
M1-HANDOFF-VERDICT pro6000-ws-box36-replay export_ms direct/buffered: winner handoff-1g=winner@0.917038670189261/10 handoff-8g=winner@0.9164044948115526/10 default=direct
```

## Window and time

The window is <box> 2026-11-05 06:00-07:00 <local-tz>, shared with the MoE prefetch cell, which
runs first. `run.sh` bounds every step by `DEADLINE` (default 06:50) minus 120 s, runs
`handoff-1g` first, and starts `handoff-8g` only when 0.9 x (3.0 x the measured 1 GiB wall) fits
the time left (`FORCE_8G=1` overrides). A cell the deadline cuts short is not counted (rule 1).

Expected wall time, from BOX36's own clocks: 444.3 s (1 GiB) and 1281.0 s (8 GiB) at 600 W. In
those cells the card drew more than 250 W for 170 s (mean 494 W) and 720 s (mean 522 W); that is
the prefill of the fill prompts, the part a 250 W cap slows. At 1.26x to 1.5x longer in those
phases (clock ~ power^(1/3) to a flatter curve) the cells take about 8.1-8.8 min and
24.5-27.4 min. With the artifact sha256 (18.2 GB), prompts, proof and reader, about 33-37 min end
to end; the Workstation floor is 29.3 min, and 2x in the hot phases gives 44 min. A start later
than about 06:12 is likely to leave `handoff-8g` unrun or cut, which means no verdict.

## What each outcome changes

- default=direct: `crates/memra-server/src/handoff_io.rs:44` keys the direct writer to every RTX
  PRO 6000 Blackwell whose name contains "Workstation" (full power and Max-Q; the Server Edition
  stays buffered). The test `the_export_writer_follows_the_card` asserts Direct for the Max-Q
  name. Docs: the decision record (title, Chosen, a Max-Q row, What would change it),
  `docs/decisions/README.md`, `docs/FLAGS.md`, `docs/SERVING.md`, `docs/rigs/rtx-pro-6000-blackwell.md`.
  Receipts in a new dir under `research/spill-f-20260919/` with raw logs, after
  `m1-box-sanitize.py`.
- default=buffered: the code stays; the decision record gains a rejected Max-Q row with the
  storage caveat above.
- default=none: nothing changes; the Max-Q row stays owed.
