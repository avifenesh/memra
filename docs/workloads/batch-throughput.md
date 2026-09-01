# Batch throughput

| | Recommended use |
|---|---|
| **Start with** | A model-and-card pair with a current batch receipt |
| **Prioritize** | Request mix, output length, admission limits, and aggregate completion rate |
| **Validate** | The real HTTP surface at the intended concurrency, not only a kernel benchmark |
| **Read next** | [Serving](../SERVING.md), [Performance](../PERFORMANCE.md), [Testing](../TESTING.md) |

Do not assume the fastest single-stream path is the best batch path. Memra selects and qualifies
decode modes per request shape; use the board and the server gate for the workload you will run.
