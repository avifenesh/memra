# Invalidated pre-rebase run

This detached box1 run used source `a10d3ca8ea70ed280e13fd637ab3d272eeb16eaf`, whose
merge base was the pre-hardening `3d485a22`. It was stopped during the direct B=1..16 matrix
immediately after a new orchestrator gate required rebasing onto `afb9be7b` or later and rerunning
the complete battery. The release rebuild and `kernel-check` completed, and B=1..11 had reached
zero-bit split comparisons when the process group was terminated; none of those partial results is
a verdict for increment 1.

The raw files are retained only to make the interruption and obsolete source auditable. The valid
evidence must come from a fresh release rebuild and full battery on the post-rebase final source.
