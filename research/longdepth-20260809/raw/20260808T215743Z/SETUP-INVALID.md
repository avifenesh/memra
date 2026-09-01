# Rejected setup: no GPU work started

The effort-matched prompt rendered successfully, but the first cell script exited before creating
the cell directory or acquiring `/tmp/memra-gpu.lock`: its rendered-prompt existence check used an
unset `RUN` variable. No generation from this run is part of the scored matrix.

The one-line harness correction defines `RUN` from the already-frozen `LANE` and `RUN_ID`; a new
commit and run id are used for the matrix rather than changing this run's recorded provenance.
