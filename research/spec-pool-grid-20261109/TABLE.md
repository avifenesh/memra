| case | sampling | base bytes (t2/t3) | base warm path | fix bytes (t2/t3) | fix warm path |
|---|---|---|---|---|---|
| chat-short-open | greedy | PASS (=/=) | parked end 49, affinity 69 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-short-open-t1 | T=1 | FAIL (69!=68/95!=100) | affinity 17 (off grid), affinity 49 (off grid) | PASS (=/=) | 1 park skip, prefix restore 64 |
| chat-short-open-t1-pen | T=1 pen | FAIL (161!=147/217!=101) | affinity 17 (off grid), affinity 49 (off grid) | PASS (=/=) | 1 park skip, prefix restore 64 |
| chat-story-open | greedy | PASS (=/=) | affinity 32, affinity 256 | PASS (=/=) | affinity 32, affinity 256 |
| chat-story-open-t1 | T=1 | FAIL (1090!=1048/682!=563) | affinity 32, affinity 256 | PASS (=/=) | affinity 32, affinity 256 |
| chat-short-256 | greedy | PASS (=/=) | affinity 17 (off grid), affinity 49 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-short-256-t1 | T=1 | FAIL (69!=68/95!=100) | affinity 17 (off grid), affinity 49 (off grid) | PASS (=/=) | 1 park skip, prefix restore 64 |
| chat-mid-open | greedy | FAIL (1204!=1050/=) | affinity 30 (off grid), affinity 286 (off grid) | PASS (=/=) | 1 park skip, prefix restore 288 |
| raw-short-open | greedy | PASS (=/=) | parked end 49, affinity 69 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| raw-long-nopc | greedy | PASS (=/=) | affinity 928, affinity 960 | PASS (=/=) | affinity 928, affinity 960 |
| raw-long-nopc-t1 | T=1 | FAIL (13!=12/=) | affinity 928, affinity 960 | PASS (=/=) | affinity 928, affinity 960 |
| raw-story-open-t1 | T=1 | FAIL (1090!=1048/682!=563) | affinity 32, affinity 256 | PASS (=/=) | affinity 32, affinity 256 |
| chat-think-1024 | greedy | PASS (=/=) | affinity 32, affinity 64 | PASS (=/=) | affinity 32, affinity 64 |
| chat-think-1024-t1 | T=1 | PASS (=/=) | affinity 32, affinity 64 | PASS (=/=) | affinity 32, affinity 64 |
| chat-think-1024-t1-pen | T=1 pen | FAIL (=/33!=0) | affinity 32, affinity 64 | PASS (=/=) | affinity 32, affinity 64 |
| chat-grid-open | greedy | PASS (=/=) | affinity 32, affinity 77 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-grid-open-t1 | T=1 | FAIL (9!=12/2!=122) | affinity 32, affinity 60 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-grid-open-t1-pen | T=1 pen | FAIL (9!=12/79!=139) | affinity 32, affinity 60 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| raw-grid-open | greedy | PASS (=/=) | affinity 32, affinity 77 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-exact-open | greedy | PASS (=/=) | parked end 64, affinity 84 (off grid) | PASS (=/=) | affinity 64, rewind 64->32 |
| chat-exact-open-t1 | T=1 | FAIL (9!=12/=) | affinity 32, affinity 60 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
| chat-exact-open-t1-pen | T=1 pen | FAIL (9!=22/75!=139) | affinity 32, affinity 60 (off grid) | PASS (=/=) | affinity 32, 1 park skip |
