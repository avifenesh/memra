# OWED 10: host/storage 250 ms sampler (2026-09-25)

**Landed CPU-verified.** Tool `m1-host-sampler.py` (`sample` and `validate`), stdlib only,
procfs/sysfs reads only. Registration: `CPU-PREREG.md` OWED 10. Raw files in `owed10/`.

| Control | Result |
|---|---|
| green: `nvme0n1,nvme1n1,dm-0`, target pid a lane-owned `sleep`, 5 s | `validate=PASS ticks=20 problems=0`; worst tick lateness 0.24 ms; eight NVMe hwmon sensors and the target's `/proc/<pid>/io` captured |
| SIGTERM mid-run | sampler exit 0, `validate=PASS ticks=8 problems=0` |
| red: an absent device `nvme9n9` | `validate=FAIL`, every tick lists `missing devices ['nvme9n9']`, exit 3 |
| red: four ticks removed from the green file | `validate=FAIL problems=2` (`sequence gap: seq 8 follows 3`, `gap 1250.0 ms > 500 ms`), exit 3 |

The runner (OWED 12) starts it inside the collector-executed worker so the canonical lock covers
it, gives it the proven leaf set from the M1 proof plus the top device, and scores a visit only
when `validate` passes over the visit window. Host-wide device counters minus the visit
process's `read_bytes`/`write_bytes` give the co-tenancy indicator of `M1-PREREG.md` B.
