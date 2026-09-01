# A1 clock-validator recovery

The replay itself completed with `verdict=PASS` and `replay.exit=0`, but the original runner exited
afterward at `FAIL: SM clock sample escaped 210-1200 MHz`. That message came from a validator bug:
after `gsub`, this host's `awk` compared field 7 lexically. It therefore rejected valid values.

The immutable `gpu-250ms.csv` was revalidated with explicit numeric coercion (`$7 + 0`): 181
samples, minimum 210 MHz, maximum 1192 MHz, zero escapes. The same post-run recovery also found no
server failure signature, reconfirmed the frozen arm-A binary SHA-256, and found no remaining GPU
compute process. See `clock-validation-recovery.log`, the empty `server-failure-scan.log`, and
`recovery-audit.log`.

The runner now uses the numerically coerced check. A1 remains eligible; no measurement row or raw
sample was changed, and the completed cell was not rerun.
