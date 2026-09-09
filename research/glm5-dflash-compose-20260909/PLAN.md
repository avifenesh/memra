# DFlash2 composed verify cell, 2026-09-09

Owner target: compose PRs #388, #394, #393 on current main and measure PP1 p32k
DFlash2 against plain. Baseline and conditional prediction from the fully read
root-cause diagnosis: plain 80.635647 tok/s, K6 71.850085, 1.850000 accepted/round,
39.588602 HTTP ms/round, 4.244432 ms saving needed; t4 component prediction
82.006 tok/s assumes 4.834859527 ms saving and unchanged acceptance.

Merged source identity is in source-identity.json. Conflicts in research/INDEX.md
and docs/KERNELS.md retained both independent entries. Runtime doors remain OFF.
The research instrument changes per-request controls, synchronous round timing,
conditional token snapshots, and same-input MLA latent-row validation only in the
remote disposable checkout. instrument.py reproduces that patch. The worker
reads one serial controller before requests, then atomic bits select KDA=1,
MLA=2 and causal-PMIN=4; 8 enables the greedy MLA oracle, 16 token snapshots.
No switch occurs during a request. The K pin is independently controlled.

The root-cause launcher and SSE client are retained with a lane-specific port,
metadata path and target. MEMRA_TICK_TRACE=1 adds server phase receipts.
One boot, 64-token spec/plain warmups; five greedy 160-token arms OFF/KDA/MLA/
both/all; three paired PMIN OFF/ON 512-token requests using fixed seed
39320260909 with the other two doors ON; then seven throughput arms interleaved
forward, reverse and rotated, each N=3, 512-token cap and omitted sampling params.

Greedy records the complete terminal worker token snapshot and concatenated SSE
text. MLA ON simultaneously compares candidate and current outputs on identical
query/cache/index bytes, with finite outputs, per-latent-row argmax and the
component band abs <= 1e-5 + 1e-4*maxabs(reference). Timing does not run this oracle.
PMIN samples record proposals, p/q/u, reached acceptance lengths and emitted IDs.
Rejection prefixes must replay exactly. Accepted-token, accepted-length and output
histograms report total variation and Jensen-Shannon divergence. Three repeated
fixed-seed continuations cannot prove conditional distribution equality; the
analytic CPU counterexample/regression provides the causal sampling argument.

The server and all GPU gates hold the existing shared lock. Build uses a private
target, nice 19, -j16, CUDA sm100a on the assigned development B200. No cargo or
GPU work on the rig. Other lanes and the production serving pair are untouched.
Any apparatus failures are archived and excluded, never counted as timing rows.

The release lane was observed holding the outer GPU lock while a descendant
waited on that lock. The owner was notified; no other lane process was altered.

Bank complete raw source/build/test/request/SSE/log/telemetry receipts, member
hash manifest and archive SHA. If the public boundary rejects raw bytes, retain
the complete archive under private Darklanes custody and the public digest here.
