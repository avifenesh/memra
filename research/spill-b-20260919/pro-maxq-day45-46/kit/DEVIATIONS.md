# Deviations from the registered cell (DAY45 2.2 and DAY46 2.2 on the Workstation)

Unchanged: the arms and their env, the boot orders, N=5, the client flags, the prompts (`docs/SERVING.md` from each
chain tree; `inputs.sha256` of every boot reproduces the Workstation's: `022b1d50...` and `12ae9444...` for
SERVING.md, `53f5dece...`/`64568d59...` and `073e2dc0...`/`e25d5075...` for the clients and parsers), the clients,
parsers and readers (byte-identical copies), the model bytes (sha256 `1facf36c...`), `MEMRA_CTX` unset,
`MEMRA_TIMEOUT_MS_MAX=3600000`, port 18526, `NO_SCOPE=1`, `BOOT_CTX=''`, `MODEL_KEY=q38`. Per-file port diffs are in
`patches/`; each boot's `dirty.txt` lists the three ported scripts of its tree.

## Arms choice

D0. **Replay at the deciding trees, not HEAD.** The S45 (`09badfe575fc`) and S46 (`8926ccfb3e8a`) binaries still read
`MEMRA_ADMIT_W_RELEASE` and `MEMRA_ADMIT_BY_MEMORY` and have no Workstation keying, so the Max-Q runs exactly the
Workstation cell's program. At HEAD both env reads are deleted and both mechanisms are keyed to the Workstation, so
HEAD cannot run the off/on arms on a Max-Q without patched arm binaries for two mechanisms, and it would run beside
other card-keyed defaults (KV-VMM, batch OOM retry, park compaction) that differ between the two editions. This is
the same basis as the Workstation default. No extension patch is in the kit.

## Hardware and box

D1. **Card:** one RTX PRO 6000 Blackwell Max-Q Workstation Edition (the index is `CARD` at run time) at its <box> cap
(250 or 280 W; GPU 2 reads 280 W), against one Workstation Edition at 600 W. Driver 615.71.09 against 580.65.06.
Both have 97887 MiB.
D2. **Shared box:** GPUs 0, 1 and 3 run training during the cell (the Workstation box ran nothing else). Every clause
is untimed pass/fail; readings are labeled as taken on a shared box.
D3. **Build:** the binaries were built on the laptop (`build.sh`: `cargo build --release -j2 -p memra-server --bin
memra-server`, default features, `MEMRA_CUDA_ARCH=120a`, nvcc 13.4.92, rustc 1.97.1), not on the box by
`build-arms.sh` (nvcc 13.1 auto-detected, arch auto-detected as 120a). Same SHAs, same crate set, so the program is
the same source; the SASS may differ by compiler release. The clauses compare arms of one binary (W3, P4) or read
receipts, so a compiler difference cannot pass or fail a clause by itself. The binary hashes differ from the
Workstation's (`2efafc68...`, `35719237...`); `PROVENANCE.txt` records the kit's.

## Ported scripts (portability only)

D4. `run-day26-cell.sh` (both trees): `source.txt` and `dirty.txt` come from `$WT/SOURCE.txt` and `$WT/DIRTY.txt`
(no git on <box>); the four `nvidia-smi` snapshots read `-i $CARD`; a new `device.txt` records the card pinning.
D5. `day45-run.sh`, `day46-run.sh`: `idle()` reads compute apps on `CARD` only; its lock probe is skipped when the
runner holds the lock (`RIG_LOCK_HELD=1`), and the boots run with `LOCK=none` (`CELL_LOCK`), so `lock.txt` reads "lock
held by the collector (external)" and `shape.txt` reads `lock=none` (the Workstation: per-boot `flock -w 3600` on
`/tmp/memra-gpu.lock`). The idle wait is `IDLE_WAIT_S` (600 s from `run.sh`) instead of 7200 s, so a busy card
fails fast inside a 4 h gap.
D6. `pro-single-day45/chain-b.sh`, `pro-single-day46c/chain.sh`: `R`, `WT`, `MODEL`, `BIN`, `CARD` come from the
runner (were `/root/...`); no `git fetch`/`merge --ff-only` (the lane branch is gone), no `rev-parse`/`cat-file`
(source from `SOURCE.txt`; the binary's `source.commit` must equal S45 or S46); no on-box build (`bins/tip/` holds
`source.commit`, an empty `source.patches`, `prebuilt.txt`, and `SHA256SUMS` in the registered format); no
`PATH=/root/.cargo/bin`; the card check is `nvidia-smi -i $CARD` and greps "RTX PRO 6000 Blackwell Max-Q Workstation"
(the Workstation grep cannot match the Max-Q name); the reader label is `pro6000maxq` (was `pro6000`, a label only).
D7. The sitting scripts (`pro-single-b-sitting11.sh`, `-sitting17.sh`: `cd /root/wt-b`, fetch, and DAY47 in
sitting 11) are replaced by `run.sh` and `cell.sh`. DAY47 is not part of this cell and does not run.
D8. Both halves run in one invocation (DAY45 then DAY46), or as `run-part1.sh` and `run-part2.sh` in separate gaps;
the Workstation ran them about 10 h apart in two sittings.

## Environment added by the runner

D9. `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD` (the Workstation box had one card, logical 0).
D10. The whole cell runs at `nice -n 10` (server, clients, scripts); no core pinning. The Workstation ran at the
default nice with `NO_SCOPE=1`, as here.
D11. `CUDA_CACHE_PATH` and `TMPDIR` point at `scratch-<stamp>/` under the kit (the <box> root disk is full), deleted on
exit. The fatbins carry sm_120a SASS, so nothing is expected to JIT.
D12. Inherited `MEMRA_*` variables are cleared before the cell (each one is logged), and `PORT=18526` is exported, so
nothing from the caller's shell changes the program.

## Added by the kit (no registered counterpart)

D13. `lib/guard.py`: 250 ms telemetry of `CARD` only (`telemetry.csv`), its compute apps every 10 s
(`compute-apps-card.csv`), the owner's Max-Q thermal stop (>= 92 C with fan >= 95 % on 3 consecutive 10 s checks, or
>= 95 C) and the total deadline. A stop sends SIGTERM, then SIGKILL after 60 s, to the cell's own process group only,
and leaves the cut half INCOMPLETE. The Workstation cells had no thermal guard and no kit deadline.
D14. `verdict.py` prints the one VERDICT line. The registered readers print clause lines only; the Workstation
verdicts were sentences in DAY45.md and DAY46.md. `verdict.py` encodes those sentences (PREREG.md section 3); on
the Workstation receipts it prints PASS and the registered readers reproduce both `read.log` files byte for byte.
D15. `box-state-start.csv` and `box-state-end.csv`: one read-only all-card snapshot each (index, memory.used,
power.draw, temperature), a label for the shared box. No check reads them.
D16. The runner checks and refuses before the first boot: the Max-Q name, under 2000 MiB used and no compute app on
`CARD`, the model sha256, the binaries' sha256, tools, port, 24 GB host memory, 5 GB disk, and the memra lock within
`LOCK_WAIT_S` (300 s).

## Left as registered, with a shared-box caveat

D17. The server's GPU watcher (`MEMRA_GPU_WATCH`, default on) is unchanged. At these SHAs it reads box-wide sources:
the first row of an unpinned `nvidia-smi` query (GPU 0 on <box>) for uncorrected ECC and row-remap failure, and every
card's kernel Xid lines. A fatal Xid or ECC fault on a training card would latch a fault in this server and fail
W4/P2 for a reason outside the mechanism. `verdict.py` prints `gpu_watch_fault_lines` per half so such a FAIL can be
placed from the server log before anything is concluded.
D18. Each boot still hashes the 15.7 GB model (`run-day26-cell.sh`), and each chain hashes it once more; with the
file in page cache that is seconds per hash.

## Receipt namespace

D19. Receipts go to `/mnt/Data/work/memra-maxq/admit-w-release/out-<UTC stamp>/{day45,day46}` with the registered
layout under each. When committed, they need their own receipt dir and `research/INDEX.md` row (AGENTS.md), not the
Workstation's `pro-single-day45/box-b/` or `pro-single-day46/box-c/`.
