# Gate postcheck

The required named gates all passed on final lane code:

- `tools/local-ci.sh --correctness` exited 0.
- `kernel-check: ALL GREEN (106 cells, 1 skipped)`.
- `run-spec K=1..8 self-consistency: PASS (Qwen 35B, 8/8)`.
- `correctness stage: GREEN`.
- `serve-smoke: 0 failed`.
- The standing short-request serve stress completed 64/64 with `ALL GREEN`.
- The served-spec acceptance smoke reported 1 pass / 0 fail.
- Separate Q27 and Q35 `run-gen` commands each exited 0 and reported `MATCH` without `MISMATCH`.

The enclosing lane wrapper nevertheless exited 1 at its additional final assertion that no CUDA
compute process remained. The process was PID 1106664, a 1,390 MiB CUDA process under
`sxc-refresh-colbert.service` (`sxc index-colbert`). `ps` recorded its start as 2026-08-13
06:03:57 +0300, during the battery and before the separate Q27/Q35 checks. It did not participate
in `/tmp/memra-5090.lock`.

Therefore this is a PASS for the named behavior/exactness gate contracts, but not a clean-window
receipt. No throughput or latency conclusion is drawn from this battery. The separate lane c=64
full-shape result remains RED independently; the short-request standing stress cannot waive it.
