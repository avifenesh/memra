# G2 pre-registered single-card development envelope

Frozen before native calibration/scoring. Target: one RTX PRO 6000 Blackwell,
96 GB, 600/600 W. This is copy-envelope development evidence, not a board move,
serving qualification, spill-speed claim, or comparison across rigs.

- A = pageable; B = cacheable pinned (engine PinnedHostBuf, flags=0).
- Sizes: 4 KiB, 64 KiB, 1 MiB, 16 MiB, 256 MiB; both H2D and D2H.
- Use F's native `h2d-probe --copies --repeats 1`, unchanged. Each invocation
  runs both directions and arms; two distinct-pattern correctness controls per
  arm/direction precede each measured visit. Final full-buffer SHA256 equality
  and the comparator corruption-red control are required.
- Calibrate all sizes first, within the same collector campaign. Start with
  1000 copies for <=1 MiB, 10 copies otherwise; increase counts using the fastest
  of the four visits to target >=500 ms (safety margin over the 250 ms floor).
  At most five calibration attempts/size and 100000 copies/visit; failure stops
  the campaign. Freeze per-size counts before scoring. Calibration is excluded
  from N and medians.
- Score sizes in ascending order. For each size run AB then BA, repeated five
  times, yielding **5 AB + 5 BA pairs**, N=10 visits per size/direction/arm.
  Each scored visit must be >=250 ms or the campaign is not scored; no selective
  rerun/drop of an individual short/slow observation.
- Metric: median of per-visit wall_ns/copies (us/copy); report median effective
  completed_bytes/wall_ns (GiB/s). Wall interval includes host API submission,
  per-copy stream fencing and event setup; this is not DMA-only bandwidth.
  Event_ms sums per-operation owner-stream event intervals, also including
  host submission/fence gaps. Allocation/setup/hash verification excluded.
- Thermal regime: sequential calibration-warmed visits, no imposed steady-state
  thermal soak; record observed temperature, clocks, power and link ranges at
  250 ms from collector raw CSV. Publish this regime next to N and medians.
- One uninterrupted collector `/tmp/memra-gpu.lock` across calibration and all
  scored visits. Bounded acquisition wait <=45 minutes, campaign timeout 30 min.
  Verify inherited canonical lock FD and pass to children; children stay in
  collector process group. No GPU command runs before lock acquisition.
- Refuse nonempty compute-app inventory between visits, non-600/600W samples,
  changed binary hash, missing matrices, wrong arm order, copy/byte/hash failures.
  Require complete telemetry and capture-integrity replay before summary.
- Retain calibration, every scored visit, raw hashes, failures and the complete
  collector journal/telemetry. Do not alter a probe's original N=1 classification:
  the outer campaign aggregates ten independent visits, not the copy count.
