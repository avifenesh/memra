# Router traces for the co-activation placement lane (box, 2026-09-01/02)

Three MoE route traces at depth 32,768 (2,048-token chunks, 60 decode tokens, K=5 spec),
single-card mtp-dev1 layout, `MEMRA_Q4E_ROUTER_AUDIT=1` (the audit readback is where the
tap lives, `route_topk_device`), corpus_commit = the box remint of ladder-ids (375180a73-era
main; token-comparable rows are NOT the point of a trace — co-occurrence SHAPES are):

| shape | file | bytes (gz) | note |
|---|---|---|---|
| thinkoff | moe-thinkoff-32768.trace.gz | 62,655 | first pass, 2026-09-01 21:47Z |
| raw | moe-raw-32768.trace.gz | 54,205 | first pass, 2026-09-01 21:58Z |
| thinkon | moe-thinkon-32768.trace.gz | see ls | third attempt 2026-09-02 01:56Z — attempt 1 was killed by an operator pkill that matched the shared binary name, attempt 2 by the host OOM killer when three lanes loaded at once (both receipted in kvq2-queue/QUEUE.log) |

Consumer: memra `tools/build_expert_placement_map.py` -> `MEMRA_Q4E_EP_MAP` (the placement
plumbing has 18 unit tests as of PR #42; uniform split is the day-1 map). Whether a measured
map pays is an open question the EP2 verdict bounds: routed-expert work is 32.8% of the K=5
round, so placement can only move that slice.

Also here: `kvq2-queue/` — the q2 queue's log, the three trace run logs/ladder receipts, and
the spec262kv1-thinkon HANG receipt that opened memra #53.
