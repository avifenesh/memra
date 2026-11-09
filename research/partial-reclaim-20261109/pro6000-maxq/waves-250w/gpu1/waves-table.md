| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 49.8 / 88.6; 5, 49.9 / 88.7; 5, 49.9 / 88.6 | 16 of 16, 425984, 0.12 | 114/116 |
| waves-2-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 49.9 / 88.6; 5, 49.9 / 88.8; 5, 49.8 / 88.6 | 0 of 16, 0, 11.28 | 116/116 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
- waves-1-B: pressure-2-6 <HTTPError 408: 'Request Timeout'>; pressure-2-7
