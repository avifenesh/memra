# DFlash2 composed verify, PP1 p32k, 2026-09-09

**KDA+MLA with legacy PMIN measures 81.359039 tok/s versus same-boot plain
80.651195, a 0.877661% median win. Corrected PMIN does not beat plain in the
requested throughput cell: K6 is 77.503363 and auto K is 80.574876 tok/s.**
The speed-only winning arm retains the known selected-token PMIN bias. Recommend
plain for correct vendor-default sampling on this shape; keep all three doors
OFF in source. This is a small diagnostic win for the composed verifier, not a
corrected-sampling deployment qualification. rev: 2026-09-23

## Source and protocol

Base main: `9396a817e` (v0.137.0). Merged, in order:

| PR | Exact merged head |
|---|---|
| [#388](https://github.com/avifenesh/memra/pull/388) | `188ab8bf13151c99d87d700c71088e7ed1798829` |
| [#394](https://github.com/avifenesh/memra/pull/394) | `1733ef0d17567d8d7767c6793da3e5a4378fb775` |
| [#393](https://github.com/avifenesh/memra/pull/393) | `f197eb413f6c23ab1e05eb41316c8d5f8e61dc5c` |

Composed source `532dc1ce211d829fc92a803ec4bfca0d0f024741`, plus the archived
`instrumentation.patch`. Conflicts were only appended research-index and kernel
catalog entries; both sides were retained. Measured server SHA256:
`d014669626a2efe24026880805e1ca1ed82b1c242ca5690d716739817f1199c3`.
The final engine source contains the three original default-OFF mechanisms;
research controller, logging and oracle code are retained only as a patch/script.
The public branch banks the identical composed tree as one commit on main: the
push guard otherwise re-scans an unchanged, already-published release archive
against secondary merge parents. The merged source history remains in the private
source bundle; no boundary exception or inherited archive edit is made.

One B200, sm100a, CUDA 13.1.115, PP1. Full native mint and drafter hashes match
the root-cause inventory, including all 19 model weight shards. Root-cause
launcher/env/client, with a private port/target, model metadata path and tick
trace. The supplied p32k text renders to 29,781 prompt tokens. Vendor requests
omit temperature/top_p/top_k; metadata supplies temperature=1.0/top_p=.95.
PMIN=.7, no FR-Spec trim, prime chunk 256, full-cover spec prefix restore.

One boot; two excluded 64-token warmups; five 160-token greedy tapes; six
fixed-seed sampled requests; then 21 timing requests (7 arms, forward/reverse/
rotated order, N=3 each). Every scored request hits all 29,781 cached tokens.
All 34 requests complete HTTP 200 and terminal SSE; 4,697 spec rounds reconcile
with usage. Four timing requests stop naturally before their 512-token cap:
MLA r0=263, MLA r1=174, both r2=477, auto r2=502. They remain in the table, using
actual completion_tokens/wall. No loop flags. Telemetry every 250 ms spans
36..49 C over load, warmup and all cells. Artifact hashing completed before
scored throughput. No concurrent GPU tenant entered this cell.

Build: private target, nice 19, -j16. Every GPU phase held the shared lock.
No Cargo, server, benchmark or GPU gate ran on the rig. The shared host remains
for its other lanes. No production pair access or deployment occurred.

## Exactness first

The complete worker terminal token-ID snapshots are 160 IDs each; these are
not merely concatenated text comparisons. Text hashes agree within the same
exact groups. The first OFF versus MLA-ON token divergence is zero-based
index 114: 4226 versus 3060.

| Pair(s) | 160-token IDs and text | MLA same-input numerical contract |
|---|---|---|
| OFF vs KDA-only | Byte-identical | No MLA candidate |
| MLA-only vs KDA+MLA | Byte-identical | PASS |
| MLA-only vs KDA+MLA+PMIN | Byte-identical | PASS |
| KDA+MLA vs KDA+MLA+PMIN | Byte-identical | PASS |
| Either OFF/KDA-only vs any MLA-ON arm (all 6 pairs) | Not byte-identical | PASS at latent-row boundary; no final-token identity claim |

MLA checks ran inside the greedy cell on identical query/cache/index bytes:
1,782 calls, 462,528 latent rows, widths t2..7, zero argmax flips, finite outputs,
and every row within `1e-5 + 1e-4*maxabs(reference)`. Worst error/band ratio:
0.057847664. Changed f32 bits are retained in summary.json. This validates the
component's argmax/band class, not a byte-exact MLA program or plain-token oracle.
KDA engagement is recorded by its candidate-specific dispatch counter; request
logs record the exact controller bits. Greedy timings, including the expensive
MLA oracle, never enter throughput medians.

Token-ID SHA256 (compact JSON array encoding):

- OFF and KDA: `7086dbe2b19030d0bc97050da7cc3f4bc94a37d9597beba37a80a5d6e56bc0aa`
- MLA, both and all: `c3d1c394e342f07841b0dece697e40d4db74d2a3dfe92d4640507e4a7d949b33`

## Causal PMIN model-scale check

Both CPU tests pass: target-distribution counterexample after an accepted prefix,
and prefix/slot-zero cutoff contracts. On the composed binary, KDA+MLA are ON in
both sampled arms; only causal PMIN differs. Seed 39320260909 is fixed for all
six requests, with vendor sampling defaults and 512 outputs each. Each arm's
three repeats reproduce its exact token tape and counts. These are deterministic
repeats, not three independent statistical samples.

| PMIN correction | Requests | Outputs | Rounds | Accepted drafts | Accepted/round | Drafted/round |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 3 | 1536 | 603 | 933 | 1.547264 | 2.616915 |
| ON | 3 | 1536 | 507 | 1026 | 2.023669 | 3.088757 |

| Normalized empirical histogram | OFF count | ON count | Total variation | Jensen-Shannon bits |
|---|---:|---:|---:|---:|
| Accepted token IDs | 933 | 1026 | 0.427850 | 0.322428 |
| Accepted length per round | 603 | 507 | 0.149460 | 0.020826 |
| All output token IDs | 1536 | 1536 | 0.402344 | 0.300074 |

All 1,110 captured acceptance prefixes replay exactly from p/q/u using the
unchanged rejection rule. The CPU argument and observed controller behavior
support the causal cutoff; the model-scale ON/OFF divergence is explicitly
measured. A histogram across different autoregressive continuations cannot prove
conditional target-distribution equality. No equivalence threshold was specified,
and none is invented here. Penalties, eight-turn continuation and broader
sampling-distribution qualification are not established by this cell.

## Throughput

Each column is the median of three per-request values. Independently selected
medians need not reconstruct each other. HTTP tok/s is completion_tokens/wall;
server tick ms/token sums measured decode ticks (spec: tick-spec wall) divided
by actual completion tokens. Sync round instrumentation is present in every arm;
shadow sampling and latent-row oracles are absent from timing.

| Arm | K | N | Accepted/round | Acceptance | HTTP ms/round | Tick ms/token | HTTP tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 0 | 3 | - | - | - | 12.231445 | 80.651195 |
| All OFF | 6 | 3 | 1.870787 | 0.701031 | 39.151574 | 13.496830 | 73.468425 |
| KDA only | 6 | 3 | 1.652850 | 0.662551 | 36.829108 | 13.767967 | 72.031333 |
| MLA only | 6 | 3 | 1.777174 | 0.656780 | 36.089155 | 12.823574 | 77.103737 |
| KDA+MLA | 6 | 3 | 1.664804 | 0.665236 | 32.753638 | 12.166889 | 81.359039 |
| KDA+MLA+PMIN | 6 | 3 | 1.557214 | 0.605416 | 32.866492 | 12.776979 | 77.503363 |
| KDA+MLA+PMIN | auto (3) | 3 | 1.454545 | 0.672646 | 30.826906 | 12.295080 | 80.574876 |

KDA+MLA beats same-boot plain by 0.707844 tok/s (0.877661%), and the supplied
historical plain 80.636 by 0.723039 tok/s. It is 0.646961 tok/s below the
conditional 82.006 prediction, and 7.540961 below the reference 88.9 bound.
That root-cause bound held acceptance fixed; it is not a fresh ceiling for these
changed continuations. The combined arm's three rates are 80.516580, 86.747543,
81.359039; one is below every current plain row. N=3 on one prompt supports a
narrow median observation, not a robust workload-wide speed guarantee.

The combined median verify wall is 25.210840 ms/round versus OFF 31.620684.
Its lower round wall accompanies a different width/acceptance mix, so the
6.409844 ms difference is not a matched-input estimate of kernel savings.
Random timing requests also differ in acceptance; the KDA-only rate does not
refute the byte-exact component saving measured on fixed inputs in #388.

Corrected K6 trails plain by 3.147832 tok/s (3.903020%); corrected auto by
0.076319 tok/s (0.094629%). At each request's observed accepted work, exact HTTP
break-even round saving is `1000*(wall_s - completion_tokens/plain_rate)/rounds`.
The median additional savings needed are **1.282786 ms/round for corrected K6**
and **0.028770 ms/round for corrected auto**. The latter is below the observed
request variability and establishes no win. The older steady approximation
`round_ms - 1000*(accepted/round+1)/plain_rate`, applied to separate column
medians, gives 1.159412 and 0.392819 ms; it mixes independent medians and ignores
cap/EOS boundaries, so it is not the decisive shortfall calculation.

## Served posture and TP2 implications

Recommended PP1 p32k posture now: **plain K=0, all three source doors OFF**.
The candidate speed posture K6/KDA=1/MLA=1/causal-PMIN=0 retains known sampling
bias and changes the greedy tape at the MLA numerical boundary. It is not the
recommended sampled serving configuration. KDA/MLA remain measured useful
components, while corrected K6/auto need more saving or accepted work. No default
or fleet pin changes are made. rev: 2026-09-23

[#387](https://github.com/avifenesh/memra/pull/387), inspected at
`a6e702ad668822e51e5285398d85aa4b0f104438`, owns TP2 DFlash verification and its
pending O1/O2/O3 pair gates. This PP1 receipt does not qualify it. In particular,
#394 explicitly admits 64 heads; TP2's 32-head shards do not enter that arm.
Do not transfer the 3.091277 ms t4 saving or this composed rate to the pair.
The pair must qualify its actual sharded verify/rollback and numerical program,
then compare corrected-PMIN TP spec with same-window TP plain, including sampled
cache continuation. The narrow legacy-PMIN PP1 win is no reason to enable the
pair's default-OFF spec door. rev: 2026-09-23

## Receipts and validation

Private Darklanes PR #440 custody:
`research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/compose-20260909/compose-raw.tar.gz`.
Archive SHA256: `3a031c4cc01a5017e160d4e4035e5a51131f086a83932f98bb50c601844ec120`.
All 216 member hashes and lengths match after transfer. The public boundary
rejected one provider-name pattern in the archive; no allowlist exception was
added. Full boot, model/prompt hashes, source bundle and patch, build/test logs,
HTTP requests, SSE, complete token tapes, round/acceptance/oracle logs, and
250 ms telemetry are retained intact. Public summary.json and raw-manifest.json
bind the tables to that archive; summarize.py reproduces the calculations.

Both causal CPU tests and the model-scale MLA numerical gate passed. Final
uninstrumented source formatting passed remotely; git diff --check is clean.
Push uses MEMRA_SKIP_PERF_CI=1 under the owner's temporary no-rig-gates order;
GitHub CI remains the merge gate. This draft does not merge or deploy the three
component PRs. Shared hardware is retained for the other owners.

publicity: skipped - maintenance research record.

Claude-Session: https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj
