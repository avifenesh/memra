# WP-B day-5 access boundary

Repository avifenesh/memra, lane/spill-b-20260919. Private SSH endpoint is referenced
only from the lead's gitignored LANE-LOCAL.md; it is deliberately absent here.

The initial successful SSH operation fetched B, created `/root/wt-b` at
`fb8708495c5b411963b0e67f31cc2f37c2730ea6` on `scratch/b-hostprefix-day5`, and ran
`git apply --3way research/spill-b-20260919/HOSTPREFIX-PATCH.diff`. Exit 0.
The three touched files were staged only in the remote scratch worktree:
worker.rs, admit_memory.rs and worker/host_glm.rs. Three-way complained that blobs
were missing, then direct application succeeded. Lane runtime files stay unchanged.
This paragraph is a tool-output transcription, NOT a synced raw compiler receipt.

The following build-launch connections exhausted the bounded three-attempt budget,
with more than 30 seconds between attempts (local inspection continued between them):

1. exit 255: `Operation timed out`
2. exit 255: `Connection refused`
3. exit 255: `Connection refused`

These are verbatim SSH error suffixes; address/port-bearing prefixes are intentionally
not recorded in public evidence. The successful setup is not proof that the build ran.
No compiler output was produced or fetched, no server test ran, and no GPU command
or baseline ran. No OOM, compile defect or GPU failure is inferred. No rsync could
be completed. Access restoration is required before native work can resume.

The requested commands remain **UNRUN**:

- `cargo build --release -p memra-server -j 16` with HostPrefix patch
- `cargo test --release -p memra-server --no-run -j 16`
- CPU server prefix-cache tests, then any GPU teeth through the collector
- `cargo build --release -p memra-engine --bin kv_tier_gate -j 16`
- Collector baseline at context 8192 on the approved artifact

Lead subsequently reported interruption of the rental and that a replacement is being
bootstrapped. That is the lead's infrastructure report, not a B driver observation.
B will not connect until the replacement bootstrap build is confirmed; the private
access file will be re-read then. Prior scratch cleanup/data survival cannot be verified.
Do not remove recovered scratch until outputs (if any) are inspected and synced;
B never resets /root/memra-spill or touches other lane worktrees.
