# GLM verify tally, 2026-09-09

Baseline: dcfeab7c738912a150ebbfea277112724bb99de4, plus the recorded NVTX diagnostic.
One B200, CUDA 13.1.115, sm_100a. Builds and GPU phases run remotely; GPU
phases hold the common lock. No local cargo or GPU qualification.

Read first: the 2026-09-08 three-request phase profile, KDA mechanism receipt,
PR294 real-input numeric failure, PR383 exact but 0.247601569 ms weighted
saving, PR386 superseded mechanism, and the 09-05/09-06 primitive and nsys
method records. None of those component verdicts is re-opened here.

Capture a p32k, 64-output-token HTTP request with vendor sampling parameters
omitted, DFlash2, PMIN 0.7 and K auto, PP1. Use the existing K pin for missing
widths, counting actual t after confidence truncation. Initial NVTX trace has
no SPEC_PROF/SPEC_TRACE drains; the separate phase diagnostic is not scored
as an untraced throughput result. The diagnostic labels whole rounds, verify
walks, 45 layers, and rows-exact weight shapes and resident byte counts.

Join CUDA kernel correlation IDs to host launch APIs inside NVTX ranges.
Exclude drafting, prefill, acceptance and rollback from verify. Require 45
layers in every included round. Report top15 by kernel duration per round,
actual launch counts, logical weight bytes where derivable, effective GB/s
and percent of 8000 GB/s, plus GPU timeline gaps. Effective weight bandwidth
is not a hardware-counter measurement of HBM traffic. Overlapping intervals
are unioned for GPU busy/gap analysis.

Choose the single largest evidenced saving to 5600 GB/s (70% of peak), or
fusion of its adjacent launches. Write the candidate arithmetic before code.
Candidate door default OFF, decide-by 2026-09-23. Oracle first on identical
captured real inputs at t2/4/7, then warmed ABBA x5. KEEP requires conditional
width-weighted saving >=0.5 ms/round; otherwise remove the candidate and its
door in this lane. If no single kernel can reach that bar, stop after the
candidate/fusion proposal with the measured bound. Never reinterpret an
unmeasured gate as PASS.
