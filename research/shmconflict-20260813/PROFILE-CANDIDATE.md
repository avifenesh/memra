# Candidate v1 shared-memory profile

Date: 2026-08-13
Candidate server SHA-256:
`73a049376dc31bb5081d4369d2429a37a02dad34952f482f5019df767baa86ec`

This profile reuses the exact frozen request and NCU section set from
`main@055adf47d:research/fa3softmax-20260813/PROFILE.md`, as recorded in the imported request
harness and each committed NCU session export. Both launches held
`/tmp/memra-5090.lock`, began with an empty compute-app check, reported `cached_tokens=0`, and
retained the baseline output hashes for Q27 and Q35. The local RTX 5090 remained under the
owner-imposed 210--1200 MHz cap; these observations are relative-only.

## Direct mechanism result

| Model / shape | Baseline actual | Candidate actual | Ideal | Baseline ratio | Candidate ratio | Excess reduction |
|---|---:|---:|---:|---:|---:|---:|
| Q27 / 4,096 | 873,541,632 | 142,356,480 | 135,966,720 | 6.424672x | 1.046995x | 99.133680% |
| Q27 / 764 | 354,302,976 | 57,513,600 | 54,912,000 | 6.452196x | 1.047378x | 99.131036% |
| Q35 / 4,096 | 582,361,088 | 94,904,320 | 90,644,480 | 6.424672x | 1.046995x | 99.133680% |
| Q35 / 764 | 236,201,984 | 38,342,400 | 36,608,000 | 6.452196x | 1.047378x | 99.131036% |

The same per-array result occurs on all four captures:

| Array / instruction | Baseline | Candidate v1 | Result |
|---|---:|---:|---|
| Q `LDSM.16.M88.4` | 8.00x | 1.00x | conflict eliminated |
| K `LDSM.16.M88.4` | 8.00x | 1.00x | conflict eliminated |
| P `STS` | 4.00x | 2.00x | residual two-wave service |
| P `LDSM.16.M88.4` | 4.00x | 2.00x | residual two-wave service |
| V `LDSM.16.MT88.4` | 8.00x | 1.00x | conflict eliminated |

Recurring K/V loads contributed 97.03--97.22% of the baseline excess and are now ideal. The
remaining aggregate 1.047x ratio is almost entirely P's two-wave residual. The deterministic
before/after summary is
[`raw/shared-attribution-before-after.csv`](raw/shared-attribution-before-after.csv); the complete
per-PC surface is [`raw/shared-pcs-before-after.csv`](raw/shared-pcs-before-after.csv).

## Resource and duration check

The XOR permutation uses the same dynamic shared allocation, so it cannot create a second CTA.
NCU confirms 8.33% theoretical occupancy on every candidate capture.

| Model / shape | Baseline duration | Candidate v1 duration | Baseline local spill requests | Candidate local spill requests |
|---|---:|---:|---:|---:|
| Q27 / 4,096 | 13.47 ms | 9.10 ms | 0 | 811,008 |
| Q27 / 764 | 5.70 ms | 3.79 ms | 0 | 327,168 |
| Q35 / 4,096 | 9.26 ms | 6.22 ms | 0 | 540,672 |
| Q35 / 764 | 4.27 ms | 2.84 ms | 0 | 218,112 |

The profiled kernel is materially shorter even with the spill, but these are single NCU captures,
not scored timing evidence. Static extraction and SASS agree on the new cost: candidate v1 has an
8-byte stack frame and exactly two `STL` / two `LDL` instructions, whereas the baseline has none.
A source-liveness-only v2 preserved the exact swizzle and moved values not used by Q staging out
of that peak-pressure region. It did not remove the 8-byte stack frame and moved the spill slot
into the PV/output accumulator path, so it was rejected statically and the profiled v1 source was
restored. Candidate v1's direct mechanism result remains the retained mechanism evidence.

## Artifact boundary

The NCU reports remain outside the repository under `/tmp/memra-shmconflict-20260813` and are
identified by SHA-256 in each `ncu-postflight.log`. The repository retains only the launch/request
logs, session export, raw/details/source-SASS CSV exports, extracted SASS, and derived tables under
`raw/profile-candidate-{q27,q35}/`.
