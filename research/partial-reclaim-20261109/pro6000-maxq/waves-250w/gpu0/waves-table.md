| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 49.5 / 89.4; 5, 49.4 / 89.4; 5, 49.4 / 89.3 | 0 of 16, 0, 11.09 | 116/116 |
| waves-2-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 49.4 / 89.0; 5, 49.3 / 89.2; 5, 49.6 / 89.5 | 16 of 16, 425984, 0.12 | 114/116 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
- waves-2-B: pressure-1-6; pressure-1-7 <HTTPError 408: 'Request Timeout'>
