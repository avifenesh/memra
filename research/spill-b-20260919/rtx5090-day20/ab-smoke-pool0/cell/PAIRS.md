| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 183.931/280.552 | 182.229/183.787 | 75.0->70.0 | 70.0->68.0 |
| BA-0 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.033/280.11 | 181.427/182.597 | 67.0->68.0 | 68.0->67.0 |
