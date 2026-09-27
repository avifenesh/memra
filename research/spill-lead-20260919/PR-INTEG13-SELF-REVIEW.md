# integ13 self-review (lead, 2026-09-21)

Read in full: the `crates/memra-server/src/worker.rs` diff of C day 14 (`host_tier_entry_class`, `HostTierDraftSource`,
`host_tier_draft_program`, `host_tier_shape_metadata`, `HostTierPrograms::program`, the `bind_tier_image` draft arm,
the four refusal sites, the four new tests), the FLAGS row change, TESTING rows, `HOSTPREFIX-DOOR.md` draft-planes
section, `verify-day14.py`; receipts spot-checked.

## Findings
1. **OFF still byte-identical by construction.** Every changed statement sits inside a `tier.is_some()` block or in a
   pure function only those blocks call; the door unset never reaches them. Local serve-smoke with the door unset and
   the target-card OFF/ON tables agree.
2. **One classifier, four call sites.** `host_tier_entry_class` is the single pure decision (plain, MTP draft, refuse
   GLM planes, refuse the DFlash tail) used at demote, bind, insert and promote, so the surface cannot drift between
   the paths; refusals are typed and name what is missing.
3. **Identity separates spec from plain.** The draft program folds the head's source, plan and draft encodings into
   `artifact`, `serialized_plan` and `numeric` with length framing, so a spec entry and a plain entry of the same
   prompt never share an identity; the other identity fields are unchanged.
4. **Bytes and copies untouched.** The draft plane is bound as its own `Role::Draft` segments with checksums under the
   trunk's geometry rule and a v2 shape blob; `host_plane_from_device`/`plane_up` are the same programs as before.
   Equal demote byte counts OFF/ON on four draft-bearing entries; the 0.2 MB delta is the bound draft plane.
5. **Honest findings.** The `MEMRA_KV_HOST_VERIFY` digest attests the trunk only (stated as a later slice, not
   claimed); the second promote is declined by the protected-share rule identically OFF and ON (C's first verifier
   assertion was wrong and was fixed before commit, recorded).
6. **Nits (not blocking).** The plan identity still uses the plan's `Debug` form (integ11 nit stands). The door's
   ON arm has target-card receipts only for the Qwen3.8 artifact; another family is its own census before promotion.

## Verification this review relied on
integ13 CPU battery (`integration-day12/integ13-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `verify-day14.py`, perf board, diff-check, all rc=0; local 5090 serve-smoke with the door
unset. C's target-card OFF/ON gates under the default spec env. This rig cannot run the model gates.
