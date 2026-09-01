# cx-restoredrill progress

Date: 2026-08-12
Branch: `lane/cx-restoredrill`
Pinned base: `main@d2fba6200`

## Scope and gates

- [x] Confirm the dedicated clean worktree and pinned base.
- [x] Inventory the off-host manifests and identify restore inputs that exist only on the serve box.
- [x] Sample the live public endpoint and record live process/GPU state before the drill.
- [x] Restore the pinned source, verified model artifacts, fresh key material, and serve metadata
  under an isolated `/scratch/restore-drill` root.
- [x] Start a second two-model server on `127.0.0.1:8004` only after verifying VRAM headroom.
- [x] Pass the 21-check protocol battery against the isolated instance and retain timed raw receipts.
- [x] Stop only the isolated instance and prove the live endpoint stayed available before, during,
  and after the drill.
- [x] Record bounded manifest/runbook fixes, the gap list, and an explicit PASS/FAIL verdict for the
  durable-restore checklist row in `RESULTS.md`.

## Constraints held

- Do not stop, restart, reconfigure, or attach the drill to the live `cx-servetest-server`,
  `cx-servetest-relay`, or `cf-tunnel` tmux sessions.
- No merge, tag, origin push, performance-board update, broad formatting, `rustup`, committed key
  material, or verification bypass.
- Preserve unrelated work and commit only this lane's evidence and bounded restore fixes.

## Timeline

- 2026-08-12: Started. The dedicated branch is clean at the required pinned base. This ledger was
  created before inventory, remote mutation, or drill work; no restore result is claimed yet.
- 2026-08-12: Inventory confirmed the committed binary, Q27, Q35, source-script, and secret
  fingerprints against the live box without reading secret contents. The prior setup environment
  is Q27-only, its keyring hash predates the current live keyring, and no single manifest assigns
  both model artifacts plus the pair runtime shape. The live extracted source also has no `.git`;
  GitHub HTTPS does serve the recorded pinned revision through current `main` history.
- 2026-08-12: Added a content-free pair restore manifest, a non-commercial two-alias metadata
  template, and an isolated drill runner. Shell syntax and ShellCheck pass. These inputs are not
  marked proven until the remote drill and cleanup receipts pass.
- 2026-08-12: The upload preflight found that the Vast image lacks `fuser`; no drill root or
  process had been created. Replaced the runner's listener ownership checks with `ss` parsing,
  which identifies the live `127.0.0.1:8002` listener without signaling it.
- 2026-08-12: Attempt 1 stopped during source verification, before binary/model staging, key
  generation, or server boot. The pinned runtime revision predates the separately deployed
  servetest scripts, so the runner's assumption that they lived in that checkout was false.
  Production remained ready on the original PID and listener with all three pane identities
  unchanged. Retained the raw failure, removed its exact secret-free/model-free scratch root,
  and split the manifest into pinned runtime and pinned harness revisions.
- 2026-08-12: Attempt 2 also stopped during source verification, before any drill process or
  secret/model staging. It caught a stale monitor SHA-256 in the new consolidated manifest; the
  blob at the harness revision and the deployed monitor agree on the corrected hash. Production
  again remained ready, and the exact failed scratch root was retained then removed.
- 2026-08-12: Attempt 3 passed end to end. Runtime and harness fetch, binary re-verification,
  full non-reflinked model copies and hashes, fresh config/secrets, isolated boot, both 21-check
  batteries, TERM cleanup, and receipt verification all passed. Public monitor samples were
  24/24 with zero errors before, during, and after; live PID `15511` and all three pane identities
  were unchanged. The isolated root, upload bundle, transfer archive, model copies, and generated
  secrets were removed only after the off-box mirror and all nested manifests verified.
