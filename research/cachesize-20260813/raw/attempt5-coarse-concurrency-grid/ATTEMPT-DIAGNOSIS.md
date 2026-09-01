# Excluded attempt 5 — coarse sold-latency boundary

This attempt is excluded from scoring. All six Q27 budget boots and the Q35 1,024 MiB boot
passed; Q35 4,096 MiB was in progress when the owned sweep timeout process was terminated, which
made the fail-closed runner stop its server and samplers. Both GPUs returned to 0 MiB, memra's
ports cleared, and `/tmp/memra-gpu.lock` became free.

The Q27 32 and 49 GiB cells established 100% working-set hits, c=4 within the sold ~22 ms p95
class, and c=8 outside it. But a c=4/8 grid only brackets the requested maximum at c=4..7. The
final campaign adds c=5/6/7 for both models and restarts every budget/repetition from zero under
one uninterrupted lock. No row from this attempt is used in the scored reduction.
