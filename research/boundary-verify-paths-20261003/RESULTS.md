Allowlist drift verification checks the currently tracked paths named by the exemption file. It still evaluates the same path and content rules with Git's PCRE candidate predicate, including symlink blobs. The separate whole-tree check remains unconditional and unchanged. Policy, exemptions, expiry, diagnostics, pruning and native gates are unchanged.

On the frozen d5ba tree, both original and repaired verification returned exit 0 and identical output for 627 live exemptions. The original took 616.894 seconds. The repair took 206.672 seconds under one CPU, 2 GiB maximum and zero swap. It evaluated 627 pinned paths out of 200,810 tracked paths. This is one local CPU measurement, not a model or serving claim.

All 52 prior controls remain. The final 60-test suite passes. Five actual implementation controls produce six assertion failures with zero errors or skips: whole-tree drift work, wildcard interpretation, untracked symlink resurrection, narrowing the full check, and an empty scope reverting to whole-tree grep. Edited, deleted, remediated and partially remediated pins retain their existing failures and prune behavior. New unlisted violations still fail the full check. [PROOF.json](PROOF.json) binds current source hashes and the unchanged benchmark program to the final extra controls.

The first comparator was interrupted after test inputs changed. Its incomplete attempt remains private and grants no result. An initial bracket-only mutation control stayed green because Git preserved the literal file match; the actual magic-prefix control corrected that gap. No result cache or blanket exemption was added.

publicity: skipped: maintenance release.
