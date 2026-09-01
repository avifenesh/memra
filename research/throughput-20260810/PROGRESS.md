# Box1 PP-2 throughput ceiling — progress ledger

Status at 2026-08-10T00:04:20Z: **measurement complete**. The final verdict, tables,
serving recommendation, method, and receipt index are in [`RESULTS.md`](RESULTS.md).

The current `2d9359df` runtime binary was measured without runtime source changes. Its SHA-256 is
`8d69e0027d34cf90ed32febc66e84a5e2f8671268f6c4846ab063435928cdd54`.

Two scored blocks completed under exclusive box1 GPU lock holds:

- grouped OFF/ON c-curve plus mixed traffic: 42 cells, 1,248 requests, 319,488 output tokens;
- grouped-on default/2048 prefill-tick A/B: 6 cells, 96 requests, 24,576 output tokens.

All 1,344 scored requests reached the requested 256 output tokens. Every admission, queue, OOM,
client-rc, and failure-signature gate is clean. The excluded pilots and their reasons remain in the
raw ledger alongside both scored blocks.

Nothing was pushed, tagged, merged, released, or added to the generated performance board.
