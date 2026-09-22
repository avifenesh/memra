# Selected bundle finalization correction

TC542-SELECTED-01: the initial selected runner quarantined its provisional index
only for case failures. After all cases succeeded, a finalization or publication
failure could leave the index installable despite a failed/incomplete result.

The outer selected-run failure handler now covers those later boundaries too.
It moves the index, reseals the changed evidence tree and records failed status,
including when atomic replacement completed before raising. It never alters a
pre-existing unowned output directory. Shared manifest serialization stays one
implementation; no runtime mathematics or permission checks changed.

Actual-index and real-SIGTERM controls reproduce the old failure on immutable
aa660. The corrected runner passes21 caller controls,7 shared-finalization tests,
15 real-SIGTERM scenarios and33 provenance controls. Raw logs and source hashes
are retained in evidence.json. These are CPU controls, not native qualification.
