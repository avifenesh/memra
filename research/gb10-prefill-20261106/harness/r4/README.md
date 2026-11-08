# Round-4 harness (2026-11-08)

The scripts the round-4 receipts ran, as they ran on the units (`$HOME/.cache/memra-gb10-r4` there;
`box/` in the logs is this directory). They are the 2026-11-07 lanes' harness with that path, plus:

- `parts.py`: also classifies the F8F4 v3 tile (`mul_mat_f8f4v3`, which the v2 pattern misses and the
  glue pattern caught by its `mul`).
- `parts_req.py`: per-part time on the request's own window (from the prime's first kernel, minus the
  host time before it, for TTFT seconds). `parts_last.py` ends its window at the last prime attention
  launch, which on the MTP route is followed by the boundary tail chunk and the draft fill, so its
  window started before the request arrived.
- `gaps_api.py`: idle gaps with the CUDA API calls inside them.
- `tsrun.py`, `ts_probe.sh`: the server log with every line stamped (unix time), for the phase
  timeline (request, trunk chunks, boundary tail, draft fill, first token).
- `tok_time.py`, `tok_probe.sh`: `/v1/tokenize` time on the served documents.
- `unit_a.sh` (probe, main build, profile), `unit_b.sh` (fill lane build and A/B), `unit_c.sh` (the
  fast-exp probe arms), `gates_attn.sh` (the battery for an attention lever).
