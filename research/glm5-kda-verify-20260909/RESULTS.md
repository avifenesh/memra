# GLM KDA verify rows result, 2026-09-09

Verdict: **SUPERSEDED MECHANISM, no candidate admitted**. rev: 2026-09-23

The proposed one-launch t-row scan already exists on source `dcfeab7c738912a150ebbfea277112724bb99de4`.
`MECHANISM.md` binds dispatch, kernel body, state ownership and rejection
rollback to file and line. Current and proposed scan launches per KDA layer
are 1 -> 1 at t2/4/7, across 34 KDA layers. Zero launches are removed.
This is a source-level rejection of the premise, not a measured flat A/B.

## Oracle and timing status

| t | Real-input oracle | ABBA x5 timing | Reason |
|---|---|---|---|
| 2 | Not run | Not run | No distinct candidate |
| 4 | Not run | Not run | No distinct candidate |
| 7 | Not run | Not run | No distinct candidate |

No argmax, band or byte-identity PASS is claimed. No failing layer exists.
Weighted saving is **unmeasured**, not a fabricated 0 ms timing result. The
>=0.5 ms/round KEEP bar is unchanged. No door/code was added and then retained;
`MEMRA_GLM5_KDA_VERIFY_ROWS` is recorded as never introduced in Removed doors.
No engine or KERNELS.md change, no serving default, no GPU phase and no rig cargo.

The two prior real-input RESULTS were read before mechanism inspection:
F16 `fbced3d7a` fails layer20 row3 at t4 (argmax 4 vs 1819); shared-expert dual
`538c38544` is exact but saves 0.247601569 ms/round. Their captures are native
short-prime activations, not production DFlash2 rounds. Neither their 42 FFN
layers nor their oracle PASS transfers to 34 KDA layers. The same oracle-first
rule and conditional width weights 48/162, 103/162, 11/162 would apply to a
new candidate; no GPU time is spent manufacturing an identical second arm.

The private profile custody check additionally verifies the profiled launcher
explicitly enabled the existing batched path. Profile timing/context stays in
the private lane report. This result makes no new performance or serving claim.

## Raw source custody

Full source archive is banked in private Darklanes under
`research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/kda-verify-20260909/source-receipts.tar.gz`. SHA256:
`b6686521e945aedcf335c99ee54ddc6316d37b6dc8f7e1157cdb34453c7ab32a`.
The private archive contains complete inspected source blobs and their hash manifest,
pinned to the source above. It contains no newly captured model activations,
oracle output or timing rows. Existing test source is archived as prior art.
`source-manifest.json` retains the public per-file hashes. The compressed bytes
hit the public boundary scanner, so the complete archive is private; no
allowlist or gate bypass was used. Checks and final commit identity are reported
on the draft PR.

publicity: skipped - maintenance research record.
