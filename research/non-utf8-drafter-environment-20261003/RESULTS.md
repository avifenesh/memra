The drafter identity collector received std::env::vars(), so any non-UTF8 export could panic before its irrelevant-name filter ran. The repair accepts OS strings, checks the two ASCII family prefixes before conversion, and returns a named refusal when a relevant key or value is not exact UTF8.

Valid UTF8 identities retain their existing sorted NAME=value bytes, artifact manifest, config text and framing. MEMRA_DSPARK_DRAFT remains excluded because its loaded directory is represented by the artifact manifest. The worker's existing ready-error branch handles collector refusals. Generic launcher sanitizers remain intact.

The pinned original/candidate CPU executables compile the real from_export, host_tier_tail_knobs and streaming SHA256 helper. They use a synthetic config and artifact file, not a model. Original unrelated malformed key/value cases exit101; repaired cases exit0. Both relevant families' malformed keys/values now exit2 with named REFUSED text. Normal, changed, empty and Unicode UTF8 controls preserve baseline bytes. Three source-identical unit methods also check exact file hashes, config binding and missing-artifact refusal. Final source 2e136048b21d9b151a7cb27ae4e52e4714e2ead1 passed 1,093 server CPU tests with zero failures and 28 unchanged ignored tests. Release all-target Clippy passed with warnings denied. The coverage verifier admitted 17 independently asserted CPU edges and refused missing-edge and skipped-mandatory-control copies. The temporary export cleanup guard arms only after successful directory creation.

Scope: CPU OS input admission only. No native model execution or serving qualification. Model math, identity construction, decode defaults, artifacts, native gate requirements and shadow selectors remain unchanged.

Proof: [PROOF.json](PROOF.json). Original source: 75bac45b8acc0cb96190689a2e11a882310ebdbd. Candidate source: 2e136048b21d9b151a7cb27ae4e52e4714e2ead1. Both extracted collectors compile with the same streaming SHA256 implementation. Raw assertion outputs and immutable source/helper/executable hashes are included.

publicity: skipped: maintenance release.
