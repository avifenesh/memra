# cx-budgetsize results

## Verdict

**NO-GO for the combined derived-default change under the frozen acceptance table.** Criteria
1–4 pass: arm A reproduces the silent-refusal defect, arm B reaches 4/5 hits on every boot with
exact geometry, the derived request stays below boot free VRAM, and explicit 4,096 MiB behavior is
unchanged. Criterion 5 is RED: the full-shape Q27 c=64 cell recorded 7,072 VRAM-admission deferral
decisions. Zero step-OOM parks and zero evictions do not waive the explicit zero-deferral bar.

The refusal-observability half is independently shippable: it is isolated in the arm-A commit and
made the defect measurable. The derived-default half should not be promoted from this lane unless
the full-shape concurrency mechanism changes or the owner explicitly revises the c=64 acceptance
contract. A matched explicit-4,096-MiB c=64 control was frozen but not run: the daily ColBERT
rebuild occupied the CUDA device outside this lane's lock and may run for two hours. That optional
classifier cannot turn the frozen zero-deferral criterion green, so it is deferred rather than run
in a contaminated window.

## Arm provenance

**Arm A is deliberately metrics-instrumented; it is not a pristine `v0.81.3` binary.** The
instrumentation adds refusal counters/publication and the one-time first-budget-refusal warning so
the before arm can expose the defect. The naked budget behavior remains the `v0.81.3` 256 MiB
default. That common observability patch is not part of the A/B mechanism comparison.

| Arm | Budget mechanism | Source commit | Frozen binary SHA-256 |
|---|---|---|---|
| A | `v0.81.3` naked 256 MiB default plus common refusal observability | `093a214a9e1bc7170dd655bb417b0fd7fc6d13c8` | `ec0c2fed4aa25fa904ab072fc2af53cee34dbee7c352d0eefb257c52f88a2a2f` |
| B | geometry-derived naked default, corrected to exclude MTP/NextN head layers | `13b4918ee5cc69b73bd045c036440d065303fd9a` | `29f1a64e8935bfc5b97ea1e9b6cf02e5fd4b562c05dd844b2b6566a53f9b77a8` |

Arm A's executable is 52,775,640 bytes; arm B's is 52,789,376 bytes. Both were rebuilt cleanly
from the exact commits after `/tmp` was reclaimed, with isolated build targets and disk-backed
`TMPDIR`. The runner verifies each binary hash before creating a cell.

## Frozen protocol

- Local RTX 5090 Laptop GPU, exclusive `/tmp/memra-5090.lock` per cell.
- Persistent 210–1200 MHz SM range; every raw 250 ms sample must remain inside it.
- Relative latency and behavior evidence only; no absolute-throughput claim.
- Qwen3.6 27B NVFP4 MTP GGUF SHA-256
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`.
- Frozen 4,860-token prompt plus 60-token greedy completion, five sequential requests per boot,
  arms alternated, three boots per arm.
- `MEMRA_CTX=8192`, no `MEMRA_PREFIX_CACHE_MB` in scored A/B cells. Explicit 4,096 MiB is a
  separate compatibility control.

## Corrected geometry

The first candidate binary at `772235e526b64f6ed2f02aa7cca853b9d858e299` produced a useful but
excluded diagnostic cell. It derived 415,367,168 B per Q27 entry instead of the measured
400,162,816 B. The exact 15,204,352 B excess was one MTP/NextN layer: 1,856 B/token x 8,192.
Prefix snapshots retain 64 trunk layers, not Q27's 65 total blocks, so final arm B derives both KV
and recurrent bytes over the trunk range. Its live boots must report 400,162,816 B per maximum
entry and 800,325,632 B for the documented two-entry default before they are eligible.

## Measurements

All A/B rows below are relative behavior/latency evidence at a persistent 210–1200 MHz SM range;
they are not absolute-throughput claims. Each arm has three eligible boots, alternated A/B.

| Arm | Boots / requests | Hits / misses / inserts | Budget skips | Evictions / session defers / VRAM defers / OOM parks | First-request TTFT median (range) | Hit TTFT median (range) |
|---|---:|---:|---:|---:|---:|---:|
| A | 3 / 15 | 0 / 15 / 0 | 15 | 0 / 0 / 0 / 0 | 5,689.766 ms (5,687.680–5,690.670) | n/a |
| B | 3 / 15 | 12 / 3 / 3 | 0 | 0 / 0 / 0 / 0 | 5,688.697 ms (5,684.452–5,695.864) | 2.124 ms (1.794–3.718), N=12 |

Arm A's `prefix_cache_skips_budget == prefix_cache_misses == 15`; the new metric exactly exposes
the 0% hit-rate defect. Arm B inserts once per boot, then hits the next four requests: 12/15 total
(80%, and 12/12 after the three seed misses). Its first-hit TTFT median is 2.268 ms across the three
boots. All 30 sequential requests emit the same greedy text SHA-256:
`200ec271e8c0eb57fb6b7d42d3ed53e4590c5e72f0303b5ef3c74d363eab88e7`.

The clock/sample receipts were:

| Arm / repetition | Samples | SM MHz min..max | Peak used / minimum free |
|---|---:|---:|---:|
| A1 | 181 | 210..1192 | 19,260 / 4,724 MiB |
| B1 | 92 | 210..1192 | 19,256 / 4,728 MiB |
| A2 | 181 | 210..1192 | 19,256 / 4,728 MiB |
| B2 | 92 | 210..1192 | 19,256 / 4,728 MiB |
| A3 | 181 | 210..1200 | 19,256 / 4,728 MiB |
| B3 | 96 | 210..1192 | 19,256 / 4,728 MiB |

A1 completed the frozen replay with a PASS but the original shell clock check compared a cleaned
field lexically. Its immutable 181 samples were revalidated numerically and the exact recovery
receipt is pinned by the reducer; no measurement row was rerun or altered.

### Boot geometry

All three final arm-B boots reported the same values:

- maximum Q27 entry at `MEMRA_CTX=8192`: 400,162,816 B;
- documented entry count: 2;
- requested/active derived budget: 800,325,632 B;
- boot CUDA-driver free: 9,957,277,696 B;
- post-1.5-GiB-reserve clamp: 8,346,664,960 B.

The request is therefore exactly two maximum entries and remains far below both measured boot free
VRAM and the post-reserve clamp.

### Explicit 4,096 MiB compatibility

Arm A and arm B each produced one miss/insert followed by four hits, the same cached-token sequence,
the same output SHA-256, and byte-identical counter deltas: zero skips, evictions, admission defers,
and OOM parks. The explicit override remains authoritative; the candidate's derived-default logic
does not move configured behavior.

### Full-shape c=64

The derived-default cell seeded one 301,215,744 B entry, then launched 64 simultaneous requests for
the same frozen 4,860-token prefix and 60-token completion. Results:

- 64/64 HTTP requests completed with 4,860 cached tokens each;
- 64 hits, zero misses/inserts during the burst, zero evictions and cache skips;
- zero session-count defers and zero step-OOM parks;
- **7,072 VRAM-admission deferral decisions**;
- 23,448 MiB peak used / 536 MiB minimum free from 239 GPU samples, all 210..1192 MHz;
- final driver free 562,036,736 B; CUDA pool cached 6,598,816,510 B;
- server failure scan empty.

The captured server line names the mechanism: at 14 active sessions, effective free fell to
752..660 MiB while one more session required 465 MiB plus a 465 MiB admission reserve, so the FIFO
gate queued requests. This is not called an OOM and no failure cause is inferred beyond that quoted
receipt.

The cell also exposed a separate exactness problem: none of the 64 concurrent streams matched the
cold seed. Fifty-nine shared one alternate hash and five shared another. This lane does not assign
that divergence to prefix-budget derivation; the neighboring exactness lanes are already isolating
decode width/row provenance. The committed `explicit4096-a-c64` harness mode can classify whether
it is pre-existing in a future clean window.

## Correctness gates

- Post-fix workspace `cargo test`: PASS, including memra-server 224/224 and all doc-tests.
- Final-code named GPU battery: PASS — kernel-check ALL GREEN (106 cells, 1 explicit skip), Q35
  run-spec K=1..8 PASS, correctness stage GREEN, serve-smoke 0 failed, standing short-request c=64
  ALL GREEN, served-spec acceptance 1 pass / 0 fail, and separate Q27/Q35 run-gen argmax MATCH.
- Clean-window postcondition: DIRTY. `sxc-refresh-colbert.service` joined mid-battery with a
  1,390 MiB CUDA process despite this lane's lock, so the enclosing wrapper exited 1 after all named
  gates. No timing conclusion uses this battery; see `raw/gates/POSTCHECK.md`.

## Raw evidence

`raw/MANIFEST.sha256` seals 158 files and has SHA-256
`91a648293d1bbc9895d2389251dd66ebb235c5ca645ad930f910f17bd338f73c`; every entry verifies. The
reducer intentionally exits 1 while criterion 5 remains RED and accepts only the exact observed
failure class; it still emits the validated complete summary at `raw/reduction/summary.json`. The
excluded initial-candidate diagnostic is retained under `raw/local-ab/02-b-r1/` and is not included
in arm B's N=3 reduction.
