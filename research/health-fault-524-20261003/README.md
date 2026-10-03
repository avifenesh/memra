# Health and readiness fault acceptance

Readiness waits for a private planned cache, native prime and two route-aware eager decode calls. Every placement owner context is synchronized before the cache drops, covering the separate PP stage and readback streams. The primary context is restored afterward. Admission calibration controls cannot skip this warmup.

Serial OOM recovery retries only before output and within the retry budget. Terminal serial and prefill OOM paths fence device state before retirement. Diagnostic targets cover both prime scheduler paths and serial failures after output. Runtime defaults and native math are unchanged.

[PROOF.json](PROOF.json) records 13 focused CPU controls, coherent guard-removal controls, 62 source-bound native vendor-default rows and six queued-work lifetime controls. Those predicates passed on the historical tuple recorded there. The later placement-aware warmup fix is CPU-validated; its current native replay is pending hardware availability. The queued-work red uses the stock-before binary; the fixed binary synchronizes before request-cache drops. The helper forges the OOM after observing actual pending work. It does not establish allocator exhaustion or an explicit CUDA context handle. Its peer request tests sequential recovery.

The CUDA profile covers actual vendor-default single and three concurrent requests. These observed routes show native kernel engagement and no graph capture or launch. No graph, model support state or fleet pin is promoted.

Input artifacts, vendor metadata, private wiring, billing and raw hardware receipts remain in the private evidence bank. The public proof contains their admission digests. Owner review and current PR checks gate the merge.

The current wrapper uses the existing planned cache allocator, prime selector and eager HC/PP dispatch. Tokenwise overrides and frozen mixed expert residency keep their existing policy. The numerical engine and rewrite-qualification guards are unchanged. CPU routing evidence does not grant HC or PP pair qualification. Merge waits for the applicable current-source acceptance.
