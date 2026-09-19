# Second rented RTX 5090 topology fixture

Read-only probe from source `41507fb9`, captured timestamp in
`research/spill-d-20260919/day6/probes/inventory.json`. Original topology JSON
SHA-256 before identity minimization:
`3c1ca27c736c58b2eb217095e8fd260a784fd33300c134f51ae58890a55fba4f`.
The fixture preserves command outcomes, topology/NUMA text and only the GPU Link
Info subsection of `nvidia-smi -q`; addresses/UUIDs/serials are not published.

One RTX 5090: current Gen1, host/device/effective ceiling Gen5, width 16x.
This differs from the first rental fixture's Gen4 host ceiling, but proves no
sustained bandwidth or direct P2P route. Fixture replay is CPU parser coverage.
Read-only power query records **400 W limit / 600 W maximum**. No setting changed.
NVMe and md0 sysfs entries exist without corresponding /dev nodes: local NVMe
access is still unproven; no storage throughput claim follows.

One successful SSH attempt. Bootstrap `--status --pidfile /root/spill-bootstrap.pid`
returned exit 1, `not-running`; that file was absent and no matching bootstrap
pidfile was found beneath the receipt directory. This only describes those
pidfile paths, not an independent proof that no bootstrap process exists.

Archived `receipts-box2/` validation succeeded for three completed CELLs. This is
capture integrity only, not new GPU execution or full hardware qualification.
