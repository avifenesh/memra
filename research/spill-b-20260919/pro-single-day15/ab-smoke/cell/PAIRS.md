| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.896/266.271 | 157.801/159.469 | 49.0->47.0 | 47.0->46.0 |
| BA-0 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.674/266.074 | 158.199/159.475 | 46.0->47.0 | 46.0->46.0 |
