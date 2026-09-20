# Native capture schema (integrity, not qualification)

`tools/tier-battery.py::validate_capture` and `validate_cell` enforce the capture
contract. This is **not** `telemetry.schema.json` (the structured route-counter
schema), or `runs.schema.json` (forced byte-identity arms). A's explicit
`--schema storage-cell` diagnostic join first passes current capture integrity,
then strictly binds the command, raw sample and run-id envelope. Existing
archived capture version 1 remains readable.

## Additive day-6 fields

| Location | Field / type | Meaning |
| --- | --- | --- |
| `command.capture.json`, completed `CELL.jsonl` row | `gpu_power_limits: array<object>` | All distinct observed device/limit triples from the raw sampler CSV; empty if absent/empty. Older captures may omit it. |
| Each `gpu_power_limits` entry | `device: string` | Raw GPU `index`, not UUID/address. |
| Each entry | `power.limit: string|null` | Raw CSV `power.limit [W]` value; e.g. `400.00 W`. Preserve `N/A`; null only when the column is unavailable. |
| Each entry | `power.max_limit: string|null` | Raw CSV `power.max_limit [W]` value; e.g. `600.00 W`. Same unknown handling. |
| Capture, both CELL rows in external-lock mode | `lock_proof: {path, bytes, sha256}` | Hash-bound collector `lock.json`; canonical inode metadata plus inherited-FD protocol. |
| Capture/completed CELL | `status: executed-not-qualified|failed|refused` | Refused requires exit 2, no timeout and a last-line `REFUSED: <reason>` or `kv-tier-gate: REFUSED: <reason>` diagnostic; exact `failure_quote` must match. Neither status means qualified. |

The sampler queries `power.draw,power.limit,power.max_limit` every 250 ms. These
limit fields are strings intentionally: they retain NVML's units and unknowns,
not an invented zero or assumed maximum. The validator re-derives the list from
hash-checked `gpu_telemetry.raw_csv` and compares the completed CELL mirror.

Consumers needing a numeric cap must explicitly parse watts and reject unknown
or changing values for a fixed-envelope performance claim. A 400/600 W sample is
restricted-power development evidence, never a full-power baseline. Empty or
unavailable telemetry cannot establish a power envelope.

## Day-8 explicit refusal token contract

A refusal is exit **2**, not timed out, with a final diagnostic line matching
`^(?:kv-tier-gate: )?REFUSED: .+`. The line is retained verbatim as `failure_quote`.
Both B's original prefixed spelling and the new line-start `REFUSED:` spelling are
accepted. A generic `Error:` with exit 2 is **failed**, not refused. A token embedded
later in a line, an empty reason, a trailing non-diagnostic line, a different exit
status, or a timeout is never a refusal. This changes classification, not acceptance:
neither failed nor refused is a positive gate. Archived generic-error captures are
not rewritten into refusals.

## Day-8 storage command and filesystem binding

The collector resolves literal `storage-bench roundtrip|restore OBJECT_DIR [BYTES]
[buffered|uncached|direct]` argv, optional literal environment prefixes, and the
canonical `bash -c` single-command wrapper
(`shlex.join(shlex.split(script)) == script`). Opaque/noncanonical shell storage
commands are rejected before execution. The resolved command requires
`--storage-root` even when wrapped. Shell expansion, substitutions, redirects and
compound commands are not a supported escape hatch.

New storage captures carry `storage.object_binding` in the capture and **both**
CELL rows: resolved root/object paths, stat device number, statfs filesystem id,
and Linux mount id from `/proc/self/fdinfo`. `storage.object_argument` binds the
original command's object argument. Before execution the backend object directory (or its existing
parent for creation) must be at or beneath the resolved root and on the **same mount
and filesystem**; symlink escapes and different nested mounts refuse. The same
binding is checked after execution. A changed mount/path aborts completion (a
start-only journal cannot validate). This is filesystem identity evidence, not
NVMe proof. Existing findmnt/lsblk ancestry, raw hashes and explicit unproven
opt-in still apply. Offline validation checks the recorded identity and command
join without querying the reader's filesystem. Legacy captures lacking this
additive binding remain legacy integrity evidence, never new filesystem proof.
