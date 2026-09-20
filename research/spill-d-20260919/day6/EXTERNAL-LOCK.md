# Collector / legacy gate lock handoff

Source premise: B's `research/spill-b-20260919/LEGACY-LOCK-PROPOSAL.md` on
`origin/lane/spill-b-20260919` was read before implementation. The actual two
legacy scripts remain **unchanged** in D's worktree; the exact applyable patch is
`research/spill-d-20260919/LEGACY-EXTERNAL-LOCK.diff`. Lead owns applying it.

## Contract

After the fragment is integrated with `tools/tier-lock-proof.py`:

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$CELL" --external-lock --execute \
  bash tools/kv-host-spill-identity-gate.sh \
    --external-lock @COLLECTOR_LOCK_FD@ "$MODEL" "$SERVER_BIN" "$EV"
# The failure-gate script uses the same prefix/positional argument contract.
```

Collector `--external-lock` replaces exactly one standalone
`@COLLECTOR_LOCK_FD@` argument with its inherited canonical FD. Without that
explicit mode **every child argument is verbatim**, including `--`. The existing
raw command now contains the resolved descriptor; external-lock resumes refuse
because FD identity is ephemeral. Use a fresh output cell.

No new environment read was introduced, MEMRA_* or otherwise. **FLAGS.md
fragment: none required; no experimental flag/decide-by door was added.** This is
a correctness-runner CLI interface, not a runtime numerical/default switch.

The helper checks a regular file, the SAME canonical device/inode, exclusion of
an independently opened descriptor, and successful reassertion by the inherited
open-file description. Missing/closed/foreign/unlocked descriptors refuse. A
boolean or a lock file's mere existence is never accepted. The receipt names the
owner (`collector` or `internal-canonical`), canonical path and mechanism.

The collector retains ownership across boot/probes/teardown, passes the FD only
to its child (not telemetry), and kills only the spawned command process group
on timeout or completion, including children that closed stdout. It does not
explicitly unlock an inherited open-file description on parent close; surviving
FD owners keep exclusion until the kernel closes them. No setsid/daemonization
inside the legacy scripts is allowed. Their stop function targets only the
spawned foreground server PID; the collector owns final whole-group teardown.
No global pkill remains in the fragment.

Internal mode remains the default canonical lock, strengthened to span the
entire gate instead of separate server boots; concurrent admission refuses
immediately. All named engagement/digest/tiny-pool/fault assertions are retained
byte-for-byte. This does not change the model, numeric program, cache budget or
fault settings; B's approved test settings still apply.

## Verification

- `git apply --check LEGACY-EXTERNAL-LOCK.diff` passes against D's current files.
- Tests apply the fragment in a disposable tree, run `bash -n`, run shellcheck
  with source following when available, compare the complete assertion tails,
  and execute the exact patched internal/external prologues without a model.
- Five dedicated lock tests pass on macOS and Linux: valid owner, missing/closed/
  foreign/same-inode-nonowning/unlocked proof refusals, concurrent refusal, child
  lifetime after parent close, failure/success/timeout group teardown, unrelated
  process survival, and hash-bound lock receipt tampering.
- Linux execution used a temporary CPU-only source tree and removed it on exit;
  no model/server binary ran. One initial SSH timeout is preserved, followed by
  a successful second attempt after bounded backoff. Raw logs under
  `day6/external-lock/`; failed connectivity proves no state change.

**Still pending:** lead applies/reviews the legacy-script fragment; complete
native identity/failure model gates under this wrapper. CPU lock proofs do not
stand in for those serving gates. No global flags, kernels or published numbers
changed.

Full Python suite: **57 passed**. One prior aggregate attempt failed the existing
100 ms timeout test because its raw log was empty (no first print). The test's
startup allowance is now 1 s while the child still sleeps 10 s; timeout and raw
preservation assertions are unchanged. Failed output is retained in
`external-lock/python-startup-timeout.log`, not discarded or labeled green.
