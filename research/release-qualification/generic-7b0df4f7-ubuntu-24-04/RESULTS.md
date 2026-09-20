# Exact-source generic release qualification

Tested source: `7b0df4f70540aa71e40f85812f620a5725032d3c`.
Source inputs SHA-256: `4cc77a957b05776382280170e881881aac8a08668bd7d57ed5ada94649c73e8c`.

The controlled v3 build and generic release battery passed on one RTX PRO 6000
Blackwell Server Edition, Ubuntu 24.04 / glibc 2.39, driver 580.178.04,
Rust 1.97.1 and CUDA 13.1.115. The compiler ran in the verified private source
view with empty CUDA visibility. The native run held one exclusive physical-card
lease. Wrapper, child and battery exited 0; cleanup found no lingering compute.

- Kernel coverage: 110 total cells, 103 executed, all 10 required, 7 named skips
  within the unchanged budget of 11. Every skip is retained in the raw evidence.
- Ornith 1.5 35B A3B: calibrated argmax passed (`flips=1 bad=0`); greedy K=1..8
  each passed exactly once and matched the plain target.
- Qwen3.8 27B: calibrated argmax passed (`flips=0 bad=0`); greedy K=1..8 each
  passed exactly once and matched the plain target.
- Source, six executable identities, four named artifacts, numerical environment
  and hardware/topology agree before and after the run.

The immutable [record](record.json) has SHA-256
`39edf96bfb60f87a0cf523099e4812d8b8115494fd7281e83747b93555931e89`.
It binds the full source inventory, build record, completed lease, raw cells and
250 ms telemetry. The build record SHA-256 is
`cc75f2a8424eef891f26261051f768e9ae39687f4c1f7b715158da159797e966`.

## Router evidence from the same run

The [router descriptor](router-baseline-proof.json) binds the same record,
kernel-check ELF, actual 35B oracle and completed lease to these exact raw results:

```text
router batch-twin bit-identity (real q35 router, 32 m-points 1..2048): mism=0 OK
router batch-twin m-invariance (rows vs plain m=2048 prefix): mism=0 OK
```

They occur exactly once at lines 515 and 516 of [the kernel raw log](cells/1-kernel.log),
SHA-256 `abf1be97b2ae418e5c7aa229f44d0cbfdee12c00c6b155b55340dda38996da36`.
The initial inspection searched only for the internal cell name and missed these
human-readable labels. That absence claim was corrected; the raw run and sealed
record are unchanged. No duplicate router GPU job was launched. This proof uses
the positive lines and exact identities, not the aggregate kernel count.

## Profile and publication scope

This record qualifies the recorded Ubuntu 24.04 executable bytes and workload.
It does not qualify Ubuntu 22.04, prove a second Ubuntu 24.04 build reproducible,
or establish release-asset packaging. A tag still requires valid records for both
shipping profiles and exact rebuilt ELF matches before packaging; a missing
profile or changed executable is unqualified.

This publication preserves the tested commit and binaries. It grants no new
binary qualification, broader serving/support claim or performance attribution.
