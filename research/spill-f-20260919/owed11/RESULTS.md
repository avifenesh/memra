# OWED 11: cache-regime helper (2026-09-25)

**Landed CPU-verified.** Tool `m1-cache-regime.py` (`cold`, `warm`, `residency`, `balloon`),
stdlib plus libc through ctypes, no privilege, no global cache drop. Registration:
`CPU-PREREG.md` OWED 11. Raw logs `owed11/01..10-*.log`, on a lane-owned 256 MiB file on this
rig's ext4 (removed after).

| Control | Result |
|---|---|
| 01 residency right after writing | 65,536 / 65,536 pages resident |
| 02 cold | PASS, 0 resident, first attempt |
| 03 warm | PASS, 65,536 / 65,536 resident |
| 04 cold again | PASS, 0 resident |
| 05 red: cold while another lane-owned process maps and touches the file | **FAIL**, 65,536 still resident after 5 attempts, reason quoted, exit 3 |
| 06 cold after that process exits | PASS, 0 resident |
| 07 balloon 4 MiB, floor 1 GiB, hold 1 s | LOCKED then RELEASED, exit 0 |
| 08 red: balloon 64 MiB under an 8 MiB `RLIMIT_MEMLOCK` | REFUSED with the limit quoted, exit 2 |
| 09 red: balloon whose floor exceeds MemAvailable | REFUSED with both values, exit 2 |
| 10 floor breach while held (MemAvailable readings patched) | RELEASED-FLOOR, exit 4 |

Control 05 is the constraint the runner must respect: a visit's cold regime is applied only after
the previous visit's process has exited and nothing else maps the artifact. On this rig the
balloon regime is capped by the 8 MiB memlock limit; the rented container's limit is checked
before B3 regime (iii), and a refusal there is recorded as the regime's refusal (OWED 20).
