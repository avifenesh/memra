# v0.136.0 release qualification

Base: `15f4bf96fc3a1bfcaa401944515af7f18877307d` (#370), immediately after v0.135.0.
Versioned candidate: `12790ac96728`. Authorized non-production RTX 5090, sm_120a, CUDA 13.0. All builds and gates ran on the box with `/tmp/memra-gpu.lock`; the local rig ran none. The standard roster symlinks were used unchanged. `cargo update --workspace --offline` updated all 11 workspace packages; Cargo.toml has the workspace version and all nine internal pins at 0.136.0.

| Cell | Model | Route or identity receipt | Result |
|---|---|---|---|
| kernel-check | Ornith artifact | 95 cells, 21 recorded skips | PASS |
| argmax-margin | Ornith-1.5-35B-A3B | flips=1 bad=0 | PASS |
| run-spec | Ornith-1.5-35B-A3B | K=1..8 identical to plain target | PASS |
| argmax-margin | Qwen3.8-27B | flips=0 bad=0 | PASS |
| run-spec | Qwen3.8-27B | K=1..8 identical to plain target | PASS |
| sampled HTTP | Qwen3.8-27B DFlash | No request sampling parameters; 512 output tokens, 170 rounds; dspark-acc cum=345/546=0.632 | PASS |
| release guard | workspace | version, nine pins, remote claim | PASS |

Greedy cells are identity instruments only. This short sampled cell establishes route engagement, not sustained reuse or a throughput claim.

Server binary SHA256: `dab92051e3cf3c7bebac56528226b8f29c41b39005e5c46c6a895d11177f09eb`.
Raw battery log SHA256: `b40c9d925acbd65d81ba873e94faa6a301c4420c0b00c67e1c37c1e43adb20c7`.
`receipts.tar.gz` SHA256: `8e6bbcc2125a39906003478a2482d1c2c226611fe9654ab8b13e152902febc38`.
The archive contains the complete traced battery output, build/guard logs, sampled request/response/profile/log, binary hashes, and harnesses. The annotated tag records the subsequent final-commit battery.

Two harness setup errors preceded the passing sampled request. The first public-server boot refused `MEMRA_REQUEST_LEDGER`: `FATAL: MEMRA_REQUEST_LEDGER is a deployment-binary surface; this build ships no accounting/admin/capture.` That setting was removed. The next boot reached dspark but the harness polled `/ready` instead of `/readyz`; its own server was stopped by PID, the URL was corrected, and the cell reran. These logs are retained; neither setup attempt is counted as a passing request.

# Change evidence

[DFlash qualification](../dflash-tap-storage-20260908/qualification.json) binds the cold/resumed tap memory and exactness results to #370. The +0.05 to +0.37% TTFT comparison at 2k/8k/32k (three boots per arm) is explicitly historical combined-candidate evidence in that file. It is not an isolated benchmark of this release source. No new serving flag. Warm admission remains #372. Publicity: skipped, maintenance release.
