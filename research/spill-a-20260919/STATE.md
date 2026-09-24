# WP-A day 35 resumable state

- Lane `lane/spill-a-20260919`; Linux worktree `wt-spill-a`; base `05fbec3b2` (integ54). Pre-registration `a0f915d8a` (DAY35 sections 1 and 2, before any arm or code). The cell's script and reader `bc97178b4`; receipts `cbf52cc16`. Write-up `DAY35.md`.
- F settled on the 5090: `DAY35 F DECISION -> KEEP` (HK minus FK e2e +7.81 / +7.44 over pair noise 5.57 / 5.03; promote in-ms +7.60 / +7.60 over 0.90 / 0.90). The HK arm's source is `rtx5090-day35/hk-revert.patch`; its scratch branch is deleted.
- Next: design M (DAY35 section 2), the demote's two owner-thread KV hashes: M1 hash 1 deferred to the helper (tier rule `d2h_deferred_checksum` first), M2 hash 2 in the Hashing job, the ordering rule; acceptance (a) to (d) and the 5090 A/B as written there.
- Owed after M: the fill's speed on slower CPUs (not pre-registered), the D2D half, the strong-form receipt. Lever 1 is not this lane's.
