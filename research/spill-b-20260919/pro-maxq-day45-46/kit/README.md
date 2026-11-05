# admit-w-release Max-Q kit (<box>)

Decides whether the W-release default (`docs/decisions/ADMIT-W-RELEASE-DEFAULT.md`) extends from the full-power RTX
PRO 6000 Blackwell Workstation Edition to the Max-Q. It replays the two registered halves (DAY45 2.2, DAY46 2.2) at
their deciding trees on one <box> card. Rule and clauses: `PREREG.md`. Every change from the registered cell:
`DEVIATIONS.md`. Binaries and sources: `PROVENANCE.txt`, `bins/SHA256SUMS`, `KIT-MANIFEST.sha256`.

## Run

By the session that holds the GPU window (it already holds `/mnt/Data/work/evals/gpu2.lock` on FD 9):

    CARD=2 bash /mnt/Data/work/memra-maxq/admit-w-release/run.sh

If one gap cannot hold the whole cell, run the halves in separate gaps, in either order:

    CARD=2 bash /mnt/Data/work/memra-maxq/admit-w-release/run-part1.sh   # DAY45 half, 4 boots
    CARD=2 bash /mnt/Data/work/memra-maxq/admit-w-release/run-part2.sh   # DAY46 half, 6 boots

The second part's verdict reads the other half from the newest `out-*` whose boots all finished, and prints which
dir it took. `TOTAL_TIMEOUT_S` caps the run (defaults: 13800 s whole, 6600 s part 1, 9000 s part 2); pass the time
left in the window if it is shorter. `DRY_RUN=1` runs every check, the lock and the chains' dry runs without a boot.

The runner refuses (exit 3, `out-*/REFUSED.txt`) unless card `CARD` is a Max-Q with under 2000 MiB used and no
compute app, the model and binaries match their sha256, port 18526 is free, and `/tmp/memra-gpu.lock` is free
within 300 s. It never touches FD 9, another card, or a process it did not start. Do not edit a script here while a
run reads it (bash reads scripts as it goes).

## Expected wall time

Workstation at 600 W: DAY45 boots 33.3 min (4 x about 8 min), DAY46 boots 48.5 min (shadow 14.9 min x 2, enforce 3.5
min x 2, enforce-wrel 5.0 and 6.6 min). The boots are bound by batched 30,720-token primes. Assumed Max-Q factor
1.7 to 2.0x (sustained SM clock at 250 to 280 W about 1.35 to 1.6 GHz against about 2.6 to 2.8 GHz at 600 W; no
same-model Max-Q prefill receipt exists): DAY45 about 57 to 67 min, DAY46 about 83 to 97 min, whole cell about 2.4
to 2.8 h plus about 3 min of checks and model hashes. At a 2.5x factor the whole cell is 3.4 h, inside the 3 h 50
min default timeout.

## Disk

The model (15.7 GB) is pre-staged at `/mnt/Data/work/memra-maxq/models/`. A run writes `out-<stamp>/` (about 3 MB
of receipts plus about 4 MB of 250 ms telemetry for 2.5 h) and `scratch-<stamp>/` (CUDA JIT cache and TMPDIR,
expected empty: the binaries carry sm_120a SASS), which is deleted on exit.

## Output

`out-<stamp>/VERDICT.txt` holds the one line:

    VERDICT admit-w-release-default card=pro6000maxq day45=<PASS|FAIL|INCOMPLETE>(n/12) day46=<...>(n/24) -> <PASS|FAIL|INCOMPLETE>

`verdict.out` adds the readings (the second wave's admitted count per enforcing arm, the card and its power limit,
telemetry maxima, `gpu_watch_fault_lines`). Raw receipts: `day45/` and `day46/` in the registered layout
(`chain.log`, `run.log`, `card.csv`, `read.log`, `SUMMARY.txt`, `boots/<boot>/server.log`, `client.jsonl`, ...),
`telemetry.csv` (card only), `compute-apps-card.csv`, `guard.log`, `run.out`, `MANIFEST.sha256`.

## Layout

- `run.sh`, `run-part1.sh`, `run-part2.sh`, `cell.sh`: the runner.
- `lib/guard.py`: telemetry, thermal stop, deadline. `verdict.py`: the rule.
- `d45/`, `d46/`: the cell's files from chain trees `0ac7f6413` and `43dbdb29f` (`SOURCE.txt`), three scripts each
  ported (`DIRTY.txt`, `patches/`), the clients, parsers, readers and `docs/SERVING.md` byte-identical.
- `bins/s45/`, `bins/s46/`: `memra-server` built on the laptop from `09badfe575fc` and `8926ccfb3e8a`, with
  `source.commit` and `build.log`.
