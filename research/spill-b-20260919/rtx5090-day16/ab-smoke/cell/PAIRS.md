| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31700 | 29550 | 2150 | lru | 0 | 1700 | 0 | 0 | 191.16/280.406 | 184.097/187.013 | 74.0->71.0 | 71.0->69.0 |
| BA-0 | BA | 31700 | 29550 | 2150 | lru | 0 | 1700 | 0 | 0 | 188.215/285.607 | 193.092/197.588 | 75.0->73.0 | 69.0->75.0 |
