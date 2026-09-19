# Native capture schema (integrity, not qualification)

`tools/tier-battery.py::validate_capture` and `validate_cell` enforce the capture
contract. This is **not** `telemetry.schema.json` (the structured route-counter
schema), `runs.schema.json` (forced byte-identity arms), or A's pending storage-cell
join. Existing archived capture version 1 remains readable.

## Additive day-6 fields

| Location | Field / type | Meaning |
| --- | --- | --- |
| `command.capture.json`, completed `CELL.jsonl` row | `gpu_power_limits: array<object>` | All distinct observed device/limit triples from the raw sampler CSV; empty if absent/empty. Older captures may omit it. |
| Each `gpu_power_limits` entry | `device: string` | Raw GPU `index`, not UUID/address. |
| Each entry | `power.limit: string|null` | Raw CSV `power.limit [W]` value; e.g. `400.00 W`. Preserve `N/A`; null only when the column is unavailable. |
| Each entry | `power.max_limit: string|null` | Raw CSV `power.max_limit [W]` value; e.g. `600.00 W`. Same unknown handling. |
| Capture, both CELL rows in external-lock mode | `lock_proof: {path, bytes, sha256}` | Hash-bound collector `lock.json`; canonical inode metadata plus inherited-FD protocol. |
| Capture/completed CELL | `status: executed-not-qualified|failed|refused` | Refused requires exit 2, no timeout and a last-line `REFUSED:` or `Error:` diagnostic; exact `failure_quote` must match. No status means qualified. |

The sampler queries `power.draw,power.limit,power.max_limit` every 250 ms. These
limit fields are strings intentionally: they retain NVML's units and unknowns,
not an invented zero or assumed maximum. The validator re-derives the list from
hash-checked `gpu_telemetry.raw_csv` and compares the completed CELL mirror.

Consumers needing a numeric cap must explicitly parse watts and reject unknown
or changing values for a fixed-envelope performance claim. A 400/600 W sample is
restricted-power development evidence, never a full-power baseline. Empty or
unavailable telemetry cannot establish a power envelope.
