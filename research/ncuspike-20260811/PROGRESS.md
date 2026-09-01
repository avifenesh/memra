# cx-ncuspike — progress log

Lane: `lane/cx-ncuspike`, created from `main` at `1592253f`.
Deliverable: per-token post-sigrouter decode anatomy for resident Step-3.7 IQ4_XS on box1,
the rented cloud-2card with 2x RTX PRO 6000 Blackwell Server Edition.

## Constraints

- Measurement-only lane: no runtime code, board, formatting, merge, tag, or push changes.
- Profile an idle, stock-clock GPU; hold `/tmp/memra-gpu.lock` for the complete GPU block.
- Launch the profiling driver detached from minute one with `setsid nohup`.
- Rebuild release from the exact resident-sigrouter commit on box1 before profiling.
- Use the on-box CUDA 13.2 profilers (Nsight Systems 2026.1.3 and Nsight Compute 2026.1.0);
  run NCU under `sudo` and keep text/CSV/raw logs in this research directory, never `/tmp`.
- Treat Nsys as the unperturbed timing receipt and NCU replay as mechanism evidence only.
  Record N, the stock-clock/thermal regime, and `window_clean=false` because the host is shared.
- Use the Server Edition card's actual 1,597 GB/s GDDR7 bandwidth for hardware SOL while also
  retaining the frozen lane model's 1.79 TB/s denominator for an apples-to-apples comparison.
- Commit the progress checkpoint before decode-path inspection or GPU profiling.

## Work log

- [x] Read `~/.lanectl/inbox/cx-ncuspike.md`, `/home/avifenesh/projects/bw24/CLAUDE.md`,
  and `research/solgap-20260811/REPORT.md` in the requested order.
- [x] Confirmed a clean dedicated branch/worktree at the requested base commit.
- [x] Recovered the standing profiler-disk, clock, thermal, and evidence constraints.
- [x] Resolved the clean capture harness (`decode-window-profile`, the eager
  `decode-bench` protocol with prime and four warmups outside `cuProfilerStart`) and installed
  profiler versions (Nsight Systems 2025.5.2; CUDA 13.1 Nsight Compute 2025.4.1).
- [x] Preflight found the GPU idle at stock clocks, but found two blocking contract failures:
  the Step artifact is absent locally and host profiler pressure is red. Full receipt:
  `raw/preflight-blocker.txt`.
- [x] Read the corrected brief, retained the local block analysis, and retargeted only the
  measurement execution to box1.
- [x] Resolved the exact target as `1808220ead39d515a0854df49d1bb6452b558209`, the clean
  resident-sigrouter checkout used by the same-commit N=5 throughput receipt.
- [x] Created a clean detached source worktree on box1, rebuilt full release with `nice -n 15`,
  and recorded run-gen SHA-256
  `706a6c7ccec59088750e7fe1351d7b9225b1701de33d316b765c3cdac618527a`.
- [x] Launched every GPU driver detached. The Nsys driver polled the flock without touching the
  GPUs while `dualpp1` held its uninterrupted block, then acquired the lock only after box1 was
  idle. No ncuspike process contended with another lane.
- [x] Rejected the first `run-gen` trace because it included prompt replay/steady-state work and
  155 full-logit D2H endpoints; retained it as a diagnostic rather than slicing away the failure.
- [x] Rejected the legacy `decode-window-profile` trace after its `Cache::new` allocation put
  stage-1 KV on device 0 under PP2. Built a separate measurement-only harness that uses the
  production `pp::new_cache` and B=1 sampled/lean worker seams without changing runtime source.
- [x] Captured one detached, lock-held Nsys N=32 decode-only range at depth 512: 9.401 ms/token
  wall, 8.846 ms/token GPU-busy union, and 0.555 ms/token (5.90%) launch gap.
- [x] Ran the focused NCU replay on representative device 0 for four dominant symbols and 12
  launch configurations; retained CSV/text/raw logs and the binary report's SHA-256, then removed
  the binary report before collection.
- [x] Wrote `RESULTS.md`, froze the raw receipts, and reduced them into `summary.json`.
- [x] Re-ran the reducer deterministically; checked all report links, shell syntax, ShellCheck,
  raw return codes, profiler-artifact exclusions, secret patterns, and authored-file whitespace.
  Raw Nsys/NCU text exports retain their original padded tables byte-for-byte.

## Blocked preflight

- The local Step Hugging Face cache contains only a 40-byte ref; there is no Step GGUF under
  `/data`. The complete 104,993,562,624-byte IQ4_XS checkpoint is receipted only on box1.
- The required zero-DtoH path is guarded by `sigmoid_resident_dev_eligible`: Step-only, uniform
  q8, and a same-device `DevExps` slab. Existing PP-2 receipts put the smaller expert stage at
  45.72 GB plus a 3.92 GB trunk; the local RTX 5090 has 24,463 MiB. A spill run is therefore a
  different pre-sigrouter-dispatch program and cannot answer this brief.
- `/tmp` is 20/31 GB and swap is 30/31 GB used. The standing pressure gate forbids a profiler
  launch in that state; this lane owns none of those bytes and did not delete them.

The owner chose the exact box1 path. The local blocker remains valid evidence explaining why the
5090 cannot answer this question; it is not treated as a failed box1 measurement or removed.

## Box1 retarget

- Host: `<private-host-redacted>`; 2x RTX PRO 6000 Blackwell Server Edition, stock 600 W limits.
- Artifact: the three-part IQ4_XS checkpoint, 104,993,562,624 bytes total; the profiler contract
  records each pinned part hash from the existing artifact receipt.
- Isolated source: `/opt/scratch/nvme/memra-cx-ncuspike-src`; isolated target:
  `/opt/scratch/nvme/memra-cx-ncuspike-target`.
- Remote raw staging: `/opt/scratch/nvme/ncuspike-20260811`; committed copies live under
  `raw/box1/`.
- Queue state at launch: the idle gate was green but `dualpp1` acquired the shared flock first;
  the detached ncuspike driver correctly remained queued. No overlapping ncuspike GPU process
  was started.

Final remote state: both profilers returned zero; the GPU lock is free; `nvidia-smi compute-apps`
is empty; and the isolated runtime checkout remains clean at `1808220`. The lane is ready for its
research-only commit; no runtime, board, formatting, merge, tag, or push operation is part of it.
