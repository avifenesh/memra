# Excluded attempt 4 — wrapper pre-created the fail-closed output directory

This launch stopped in source preflight before the GPU lock was acquired, before a server was
started, and before any measurement was made. The launcher had placed its own log and PID file
inside `raw/scored`; the campaign harness correctly refuses a pre-existing output directory.

The empty launcher log and PID receipt are retained here. The corrected launcher writes those
files beside `raw/scored`, leaving the scored path absent for the harness to create atomically.
No row from this attempt is used in the scored reduction.
