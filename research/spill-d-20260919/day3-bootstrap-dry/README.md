# Inherited interrupted-session CPU dry run — retained, superseded

These files were already present, uncommitted, when D day 3 resumed. They are
retained unchanged as the earlier stub run, not discarded or relabeled as current.
`BOOTSTRAP.json` predates the inherited script's final edits: it lacks the later
`pp-transport-smoke` target, bootstrap binary-hash table and rustup installer steps.
Its topology schema also predates the current probe output. The original source
revision was not recorded; the all-ones source SHA is explicitly a **stub**, not git.
The CUDA `accept` file is text marked `CPU STUB NOT EXECUTABLE`.

The successor run is `../day3-bootstrap-dry-final/`; current offline check receipts
are `../day3-final-checks/`. Neither run compiles CUDA, allocates GPU memory, performs
SSH, installs packages or qualifies a rig. No hardware or model result exists here.
