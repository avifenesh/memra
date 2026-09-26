# B0 runner: device envelope and the io_uring screen (2026-09-25)

**Runner landed CPU-verified against a stub fio; no device measurement has run.** Tool
`m1-fio-envelope.py` (`plan`, `run`), stub `m1-stub-fio.py` (fio json+ shaped output, never a
measurement), tests `test-m1-fio-envelope.py` (`b0/test-fio-envelope.log`: 4 of 4 pass).
Registration: `M1-PREREG.md` B0 and B5 (B5's two thresholds clarified with this runner, before
any run: "faster" also needs 4 of 5 rounds agreeing in each order, "matched" is a median
bandwidth ratio of at least 0.97).

- `plan`: 88 fio invocations in registered order: 1 prepare (16 GiB sequential direct write,
  `end_fsync`), 45 grid cells (5 block sizes x 3 depths x 3 engines, engine order reversed on
  alternate cells; `psync` uses `numjobs = depth`, `io_uring` and `libaio` use `iodepth = depth`),
  40 screen visits (557,056-byte reads, depths 2 and 16, 5 AB plus 5 BA), 2 write-envelope runs.
  About 19 minutes of box time.
- `run`: verifies the inherited lock and the proof identity on the fio directory; per
  invocation starts the sampler first (so the device window covers the run), then fio with its
  own json+ file; CPU seconds come from `wait4` rusage of the fio process (all threads); the
  envelope file is removed at the end; the summary carries the B5 screen verdicts and the
  sustained read maximum with its 70% headroom figure.
- Tests: plan shape and argument mapping; the B5 rule (faster, sign disagreement in one order,
  matched-and-cheaper, matched-not-cheaper, slower-but-cheaper, incomplete); an end-to-end stub
  run (88 rows, both depths `io_uring-justified` at a stub 1.12x, envelope file removed, sampler
  files present); refusals for a wrong-identity proof and for the stub bypass with a real name.

fio is not installed on this rig (installing a package here is not this lane's to do); on the
box it is a bootstrap step (`apt-get install -y fio` inside the container).
