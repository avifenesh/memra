| arm | reclaims (partial, still short) | entries / GB freed (largest) | booking on / off | OOM lines | pressure waves: errors, TTFT p50 / max s | repeats: hits, cached tokens, TTFT p50 s | greedy texts = A |
|---|---|---|---|---|---|---|---|
| waves-1-A | 1 (0, 0) | 57 / 54.2 (57 / 54.2) | 0 / 0 | 0 | 5, 46.9 / 88.3; 5, 46.7 / 88.1; 5, 46.8 / 88.1 | 0 of 16, 0, 10.56 | 116/116 |
| waves-2-B | 7 (7, 0) | 36 / 34.3 (15 / 14.2) | 2 / 2 | 0 | 5, 46.8 / 88.2; 5, 46.7 / 88.4; 5, 46.8 / 88.2 | 16 of 16, 425984, 0.12 | 114/116 |

Rows whose greedy text differs from the first A arm (tag, the row's error if any):
- waves-2-B: pressure-1-6; pressure-1-7 <HTTPError 408: 'Request Timeout'>
