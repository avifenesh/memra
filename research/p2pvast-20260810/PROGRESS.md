# P2P on the Vast serve home — progress

Date: 2026-08-10
Lane: `lane/cx-p2pvast`
Base: `3f8ca2ef`

## Contract

This lane answers, in order:

1. Is the production-shaped PP-2 runtime on the Vast 2x RTX PRO 6000 box actually using native
   CUDA peer transfer, and what bandwidth/latency does that path deliver at decode and prime
   activation sizes?
2. In live serving, how large are the decode-round and prime-chunk boundaries relative to their
   adjacent stage compute, and are the copies overlapped or serialized?
3. Only if a non-overlapped boundary serializes at least 3% of the measured round or prime time,
   add a default-off `MEMRA_PP_COPY_OVERLAP=1` experiment, prove byte identity, and run an
   interleaved N=5 A/B on this box.

The peer receipt, anatomy measurements, decision, and any gated A/B belong in `RESULTS.md`. Raw
logs stay under this directory. Serving is stopped only for bounded measurement blocks and is
restored after every block; the lane does not push, merge, tag, or alter published perf boards.

## Initial state

- Local worktree was clean on `lane/cx-p2pvast` at `3f8ca2ef`.
- The Vast checkout was detached at `019428e217e297cb5981d201a4a520aee69222a6` and
  `GET /v1/models` answered before measurement work began.
- The requested lane inbox path, `~/.lanectl/inbox/cx-p2pvast.md`, did not exist at lane start.
  It remains a mandatory recheck before each bounded block; absence means no queued override was
  available, not that the check was skipped.

## Gate ledger

- [x] Block 1: capability + engine probe + boot-log grants captured; serving and soak restored.
  The box reports bidirectional peer capability and memra enables the production peer path, but
  synchronized payload checks fail. The custom 16 KiB transfer leaves 16,320/16,384 bytes wrong;
  memra's exact boundary roundtrip fails all four slots; NVIDIA `simpleP2P` also fails under both
  the CUDA 13.1 compatibility library and the host driver library. Apparent bandwidth is therefore
  a no-op/corruption artifact, not a usable peer-active receipt.
- [x] Block 2: decode short/4k anatomy + prime-pipeline diagnostic; serving restored. The
  apparent median `(tx + rx) / round` was 0.0873% short and 0.0998% at 4k; the pp512-class
  pipeline advanced 3/3 overlap counters. All timings are non-scored because the boundary bytes
  are corrupt.
- [x] Decide build versus hold. Block 1 already excludes an overlap implementation: optimizing a
  transfer that does not preserve bytes cannot pass the lane's correctness gate.
- [x] If built: byte-identity gate and interleaved N=5 A/B. Not applicable — HOLD, no build.
- [x] Commit raw logs and `RESULTS.md`.
- [x] Final `GET /v1/models` and streamed-completion receipt with the production launcher restored.
  The endpoint and SSE framing are live, but the stream is sixteen repeated BOS tokens; service
  transport health must not be mistaken for model-output correctness.
