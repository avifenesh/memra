# Excluded attempt 3 — batched seed numerical class

This attempt is excluded from scoring. It completed Q27 1,024 / 4,096 / 8,192 MiB and reached
16,384 MiB before the harness failed closed at c=8.

One full 4,860-token cache hit returned HTTP 200 with `finish_reason=stop` after 11 of the frozen
60 completion tokens (`cmpl-5fd64be53904ee8f9032b2e9adeb1372`). The cell counters reconciled
exactly: 20 admitted/completed, 34,020 cached tokens, seven hits, 13 misses, 1,151 output tokens,
zero admission/session/VRAM defers, and zero step-OOM parks. The server log contains no captured
CUDA error, out-of-memory text, panic, or fatal marker.

The harness had seeded the 96-entry working set in concurrent batches of eight. That diverged
from the frozen sell-gate's sequential hot-cache setup. Prefix snapshots inherit the numerical
class of the prime configuration that produced them, and earlier excluded sell-gate evidence
also contains Q27 11-token EOS outcomes outside the final qualified run. The harness now restores
sequential one-request seeding. A focused reproduction of this exact 16,384 MiB boot subsequently
passed all four widths: 80/80 requests completed with exactly 60 tokens and no server failure
line. No row from this attempt is used in the scored reduction.
