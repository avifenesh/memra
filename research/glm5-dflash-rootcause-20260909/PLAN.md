# DFlash2 acceptance and verify-cost diagnostic protocol

Owner request, 2026-09-09: explain the greedy versus sampled performance gap on
the single B200 development card, using the served path. No production deployment
or engine default changes. The separately discovered PMIN bias permits the
requested default-OFF corrective patch; it is absent from the scored executable.

## Final cell

- Base `6285210078609c7e11aa23ae070ad581c184c153` plus the research-only
  `instrumentation.patch`, reproducible with `instrument.py` on a disposable
  checkout of that base. `source-identity.json` and `binary.sha256` bind it.
- Native mint, DFlash2, PP1, context 65536, one session, prefix budget 2048 MB,
  PMIN=.7, and the performance pins in `posture.json`. Prime chunk is 256.
  Existing full-cover spec restore is explicitly enabled for identical repeats.
  Admission and the 1.5 GiB transient reserve remain enabled.
- The local streaming deadline is 900000 ms, header commit 300000 ms, and queue
  ceiling 900 s. Requests have one second between them, outside timed HTTP wall.
  These local diagnostic deadlines are not product defaults.
- `cell.py` warms spec and plain, then runs 20 sampling/K arms and vendor-default
  plain, in forward, reverse and rotated order. Each arm has three requests in
  one successful boot, each capped at 512 outputs. Natural EOS is retained.
  Spec requests must carry positive usage.spec; plain must not carry spec usage.
  All bodies, complete SSE, concatenated output text and log windows are retained.
- Sampling: temperature=0; temperature=.6/top_p=.95; vendor parameters omitted
  (metadata resolves 1/.95); temperature=1/top_p=1; temperature=1/top_p=1/top_k=40.
  The last disables top-p to isolate top-k. K pins 2/4/6/auto and 0 for plain are
  updated through atomic research-only control files between serial requests.
- Per-round verify clocks drain before and after the target walk; round clocks
  drain at completion. These are synchronized diagnostic rates, not a release
  qualification. HTTP rate includes TTFT; engine round wall separates restore
  and admission overhead. The plain post-first-token interval is not an isolated
  GPU duration. No spec event cadence is treated as per-token latency.
- After timing, one additional 512-token request per speculative arm captures
  proposal IDs, target argmax, independent target samples, p, q and uniforms.
  The shadow stream never advances serving RNG counters. It is excluded from
  timing medians. Report all-position and reached-position denominators: rows
  after the first rejection have hypothetical draft prefixes.
- `summarize.py` reconciles usage/log round counts and replays captured p/q/u.
  `tables.py` derives the tables. `seal.py` checks completion and hashes every
  member of `rootcause-raw.tar.gz`. The intact archive is banked under the private
  ship lane's `receipts/dflash2-20260908/rootcause-20260909/`; public Memra retains
  its hash, manifest, summaries and harness. The archive was transferred and every member
  hash rechecked. The loop screen flags four repeated 12-word sequences; none
  of the scored or agreement outputs triggered it.

## Preserved apparatus attempts, excluded from scoring

| Namespace | Finding | Change before the next attempt |
|---|---|---|
| failed-attempt1 | Plain warmup succeeded; following admission needed 6.07 GB with only 4.88 GB attainable | Reduced prime chunk from default 4096 to 1024 |
| failed-attempt2 | Plain-first prefix had no drafter tail; spec requests fell back to plain | Spec-first warmup and mandatory engagement check |
| failed-attempt3 | Spec engaged; following request exceeded a 10 s preheader queue deadline | Explicit local deadlines and one-second spacing |
| interrupted-preflight4 | Source review found identical full-cover repeats require the existing restore flag | Stopped before scoring and enabled full-cover restore |
| failed-attempt5 | Both warmups passed; 512-token admission had 3.780 GB versus about 4.368 GB including reserve | Reduced prime chunk to 256 |
| invalid-config6 | Equal 900000 ms commit/deadline values refused before model load | Commit 300000 ms, below deadline 900000 ms |

## Execution and references

All GPU phases hold `/tmp/memra-gpu.lock`; long jobs run under nohup/setsid with
logs. Build uses a separate target directory, CUDA 13.1.115, sm_100a, nice 19,
-j16. No cargo, GPU gate, benchmark or server ran on the local rig. The authorized
pre-push command ran its metadata censuses and recorded MEMRA_SKIP_PERF_CI=1.
The shared tune box is retained; only this lane's files and processes are removed.

The older tally is copied from commit
`188ab8bf13151c99d87d700c71088e7ed1798829` into `tally-reference/`; its source lane
was cleaned during this run. It measured `dcfeab7c` plus its own instrumentation,
with a different prime recipe. Its GPU spans are not this lane's wall clocks.

The PMIN counterexample uses exact rational enumeration on the development host.
The proposed cutoff uses max(q), independent of the current selected token given
the preceding prefix. Two remote Rust CPU tests passed. Model-scale ON-arm
sampling/continuation qualification remains pending; the new door stays OFF.
