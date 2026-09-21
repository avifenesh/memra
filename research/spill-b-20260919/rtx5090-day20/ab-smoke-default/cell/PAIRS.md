| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 186.301/281.798 | 184.494/185.217 | 74.0->69.0 | 69.0->67.0 |
| BA-0 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 195.675/285.217 | 183.919/185.197 | 68.0->72.0 | 67.0->68.0 |
