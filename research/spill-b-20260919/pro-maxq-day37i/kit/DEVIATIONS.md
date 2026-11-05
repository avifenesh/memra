# Deviations of the Max-Q kit from the registered DAY37 O1 cell

The registration is DAY37.md section 1 (1.6 clauses, 1.7 rule, addenda A to F; addendum H's pair unit), run on the
Workstation by `pro-single-day37/chain.sh` (sitting 1, harness 826ff8d78) and `pro-single-day37/rerun-f.sh` (sitting 2,
harness 3476ea4dc). Unchanged: the lane source (c6f9282c2), the main source (17dceb981), the model and its sha256, the
arms, the fault lists, the gate cells and their order, the boot names, kinds and orders, N, the clients, the prompts
(`docs/SERVING.md` at 3476ea4dc, identical at c6f9282c2), `MEMRA_CTX` unset, `MODEL_KEY=q38`, `HOSTGATE_MB=256`,
`STREAM_CONC=16`, `BOOT_CTX=''`, `NO_SCOPE=1`, the corrected reader (`day37-read.py` at 3476ea4dc, unmodified).
Every harness file the kit changed is listed with its source blob and both sha256s in `harness/PORTED.txt`;
`provenance/port.py` applies them to a `git archive` of 3476ea4dc with asserted counts.

## Tree and build

1. **One sitting, one root.** The Workstation reading combined two sittings (DAY37 2.6). Here every clause runs into
   one receipt root: chain.sh's boot list (chain.sh:100-109) with `off-lane` inserted before `fault-mapper`, as
   rerun-f.sh:64 ran it, every boot through `boots-f.sh` (the scope's "use boots-f.sh for every boot").
2. **Built off the box.** The bins are built on another machine, release, default features, `MEMRA_CUDA_ARCH=120a`
   (the box auto-detected 120a), nvcc 13.1 through `MEMRA_NVCC` (the box auto-detected 13.1), rustc 1.97.1 (the
   tree's pin), `cargo build -j2`. The lane cargo line is chain.sh:48-49's, main's is chain.sh:60's. The binary hashes
   differ from the Workstation's (the builds are not byte-reproducible across machines: sitting 1 and sitting 2 built
   the same source to 3dc05d17 and 7e586b8c). At run time the CUDA runtime libraries are the box's
   (`/usr/local/cuda`, 13.4, `libcudart.so.13`); driver 615.71.09 against the rented box's 610-series driver.
3. **serve-smoke runs the prebuilt lane binary.** Registered: gates.sh made a worktree at the lane's `source.commit`
   and serve-smoke ran its own `cargo build --release -p memra-server` there (gates.sh:50-68, serve-smoke.sh:54). The
   box has no cargo and no checkout, so serve-smoke runs from the harness tree with `target/release/memra-server`
   linked to the lane binary, and its cargo line is replaced by a check that the binary exists. `tools/` and
   `docs/SERVING.md` are identical between c6f9282c2 and 3476ea4dc (port.py asserts it), so the smoke script is the
   one the lane source carries. On the Workstation the smoke binary was a separate build of the same source with the
   same features (560fbb6d and ade3242a against the lane's 7e586b8c); here it is the lane binary itself.
4. **No git on the box.** `git rev-parse HEAD` reads `harness/SOURCE.commit` (3476ea4dc) instead: gates.sh `TREE`,
   run-grow.sh `source.commit`, run-day26-cell.sh `source.txt`, the `commit` line of admit-mem-burst-gate.sh and
   spec-ctx-edge-gate.sh. run-day26-cell.sh's `dirty.txt` holds `harness/PORTED.txt` instead of `git status --short`.

## Card, lock and box

5. **Card pinning.** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD` for every step (the rented box had one
   card). The gates' `${CUDA_VISIBLE_DEVICES:-0}` then resolves to the card.
6. **nvidia-smi reads the card only** (`-i $CARD`): the compute-apps idle checks (boots-f.sh, the chains), the 250 ms
   samplers (tier-battery.py and the three clients: day26-client.py, day33-client.py, day37-stream-client.py), the
   before/after snapshots (gates.sh, run-day26-cell.sh, admit-mem-burst-gate.sh, spec-ctx-edge-gate.sh, the twin
   gate). Registered: box-wide, which on one card was the card. Box-wide reads remain only where the box is the
   subject: one regime snapshot per part start and end (`preflight/box-*.csv`), and the timed half's quiet-box checks
   (before the part and before every stream boot, `KIT_IDLE_SCOPE=box`, the registered form).
7. **The lock is held by the runner for the whole part.** Registered: each step took `/tmp/memra-gpu.lock` itself
   after an idle wait, released between steps. Here run.sh and run-timed.sh hold it on FD 7 from preflight to exit;
   the collector's `campaign_lock` takes a dup of that open file description (proved by an independent open, never
   unlocked) so `lock.json` and the gates' `tier-lock-proof` read the same `inherited-flock-same-open-description`;
   the boots run with `LOCK=none` (run-day26-cell.sh's registered "collector holds the lock" mode) and boots-f.sh's
   idle check proves the hold instead of probing it with `flock -n`. Nothing else can take the card between steps.
8. **The window.** The runner refuses unless FD 9 is open and flock-held on `/mnt/Data/work/evals/gpu$CARD.lock` (the
   eval session's window), the card reports the Max-Q name, under 2000 MiB used and no compute process. New stop
   conditions not in the chain: a total deadline (`WINDOW_MIN`, default 225 min untimed and 120 min timed) and the
   box's Max-Q guard (>= 92 C with fan >= 95 % on 3 consecutive 10 s samples, or >= 95 C). A stop kills every process
   carrying the run's marker, then the reader runs on what exists.
9. **Idle wait 600 s, not 7200 s** (`KIT_IDLE_WAIT_S`), per step and per boot: a gap is bounded, and a card that is
   not idle in a gap will not become idle by waiting. The >= 24 GB host-memory check of boots-f.sh is kept.
10. **A2's lock retries** `ATTEMPTS=2 RETRY_SECONDS=60` (registered 60 x 120 s); run-grow.sh retries only lock
    contention, which cannot happen under the runner's hold.
11. **CPU and I/O.** The whole runner runs at `nice -n 10` and `ionice -c2 -n7`; no core pinning (the trainer owns its
    cores). Registered: `NO_SCOPE=1` on the box (no systemd scope) and the smoke build's scope, gone with the build.
12. **Environment.** Inherited `MEMRA_*` names are dropped at start (listed in `chain.log` as `dropped_env`), and
    `CUDA_LAUNCH_BLOCKING` and the MPS variables are unset; the chains then set exactly the registered names.
13. **Telemetry.** Beside the registered per-step samplers, a 250 ms sampler of the card runs for each whole part
    (`telemetry/card<idx>-<part>-<stamp>.csv`), with fan speed and clock event reasons.

## Order and placement

14. **Timed clauses split out.** Registered order (1.8 step 6): stage 0, A2, A1, A5, A6, A7, A3, A4 in one sitting.
    AGENTS.md's evidence discipline allows only untimed pass/fail cells beside a campaign on a multi-card box, so the
    untimed half (A2, the gate set, the 19 untimed boots) runs in a gap (run.sh) and the timed half (stage 0 and the
    20 stream boots, carrying A3 (i), A3 (ii) and A1-STREAM) runs later on a quiet box (run-timed.sh), into the same
    root. Inside each half the registered order holds; the stream pairs still come after every other boot.
15. **Placement pinned to inline for the whole cell** (`MEMRA_KV_VMM_GROW=inline`), as addendum F pinned it for the
    second Workstation sitting; sitting 1 took it from its own stage 0. Stage 0 runs as a reading at the start of the
    timed half. PREREG.md M.3 gives the reasons.
16. **Resume across windows.** A finished step is never run again: A2 with `collector.exit`, a gate arm with
    `gates done arm=`, a boot with its exit file. A step a deadline or a stop cut before it finished is moved to
    `cut/` (or `boots/cut/`) with a stamp and asked again under its registered name: A2 whole, a gate arm whole (one
    collector hold), a boot whole, a stream pair whole (both boots, so a pair never spans two windows; addendum H's
    unit). Registered: addendum H asks again only boots the idle wait did not run; 1.9 reruns a cell whole under a new
    name after a failure. A cut is not a failure finding (no outcome was produced); `chain.log` and `cut/` record each
    one. A FAIL that a step did produce is never retried.
17. **Names.** New receipt names for the new cell (1.9): `grow-32768-maxq`, `gates-maxq-pooled`, `gates-maxq-vmm`;
    the reader's free card label is `pro6000-maxq`. Boot names are the reader's registered ones.
18. **The rule is applied by a script.** The Workstation rule was read by hand from the reader's lines (DAY37 2.6).
    `day37-verdict.py` applies 1.7 to the same lines and prints one VERDICT line; on the Workstation receipts,
    assembled as 2.6 assembled them, it prints `-> PROMOTE-ELIGIBLE`.

## Inputs

19. **Paths.** Model at `/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (registered
    `/root/artifacts/...`), same sha256 required before the first boot of each part (a resumed part reuses the hash
    when the file is unchanged since). The runner refuses if serve-smoke's spec draft, Gemma-4 12B or Qwen3.6-35B
    exists at its default `/data/ai-ml` path, so the smoke skips the arms the Workstation smoke skipped. The 35B
    staged beside the 27B is not used by this cell.
20. **Python 3.14.4** on the box (the rented box's version is not recorded). The corrected reader and the rule script
    were run on 3.14.4 against the Workstation receipts and reproduce DAY37 2.6's lines.
21. **Power.** The card's limit is 250 or 280 W (300 W max) against the Workstation's 600 W; recorded, never changed.
