# Repack JSON declaration repair

Independent review of frozen `86751a19cd577058453cfe70024484d7d28f0883` found that syntax
validation and declaration interpretation used different JSON string rules. An escaped
`pruned\u005fexperts` key became apparent absence; an escaped duplicate key avoided collision
checking; an escaped `memra-expert-overlay\u002dv2` value was classified as a complete repack.
The actual six-case complete-artifact probe and exact fixture bytes are retained under
`red-86751a19/`. That revision has source_GO=false despite its earlier passing ordinary controls.

The repair uses one decoded value tree for all manifest keys and values. The recursive serde
visitor rejects duplicate decoded keys at every object; masks, format, file/tensor names, qtypes
and numeric metadata consume that same tree. JSON escapes, UTF-8 and surrogate pairs use the
standard decoder. Wrong types remain errors. Raw manifest bytes remain retained for artifact
identity, so byte-distinct JSON encodings can share semantic bindings without sharing artifact
identity. The obsolete repack-only JsonObj and safetensors validation hooks are removed; legacy
config parsing is unchanged by this repair.

Shared internal API for the config owner: `crate::strict_json::parse_object(&str)` returns
`Result<serde_json::Map<String, serde_json::Value>, String>`;
`optional_string(&Map, key)` returns an error for a present non-string value and None only for
absence. Intake `strict_json.rs`, its crate-private registration in lib.rs, the memra-gguf Cargo
serde/serde_json edges, corresponding Cargo.lock edges, and the two explicit usize shape-product
annotations in memra-reference. Registry versions remain the already locked serde1.0.228 and
serde_json1.0.150. The annotations preserve the previous inferred type and arithmetic.

The unchanged reviewer probe has the expected repaired six outcomes in the working repair,
and dedicated controls prove escaped valid retained declarations have identical plans/bindings,
while escaped masks/overlay formats and decoded duplicate keys cannot become absence. Exact final
frozen-source validation and independent re-review will be appended; no root/native/model/support
or merge approval is claimed here.

A separate actual config probe confirmed that literal hidden_act=relu refuses but the equivalent
escaped key is accepted as Silu by legacy HfConfig parsing. Its source/log are in config-escape-537/;
the canonical config owner and coordinator have this separate root-activation blocker. It is not
claimed repaired by the manifest consumer change. Existing Qwen3.5 numeric issue #548 also remains
separate: full reference testing still reports68PASS/1FAIL with the same recorded bit vectors.

## Frozen repair validation at 625aad04

Exact source `625aad049d5b5c87689efc047452364eb726c979` passes the unchanged six-case
reviewer probe. Probe code and all nine fixture files are byte-identical to the red input corpus.
The ordinary artifact binds; both literal/escaped masked uniform banks refuse; decoded duplicate
masks and both literal/escaped overlays lacking source_dir refuse at open.

GGUF378PASS/2ignored, CLI13, Step1, inspector7, external3, doctest2, actual runtime
identity/snapshot18, all-target GGUF/CLI Clippy and Linux DOCS_RS engine/server lib/bin/test
Clippy pass. Full reference remains68PASS/1FAIL at #548 with identical actual/expected vectors.
Final source hashes, statuses, raw logs and binary hashes are in green-625aad04/. Intermediate
warning/build/type-inference failures remain separately banked; they are not relabeled.
Independent re-review passed on exact `625aad04`, closing ARCH541-REPACK-JSON-01.
The report, unchanged-probe verification, focused controls and preserve_order decoder controls
are preserved under review-625aad04/. The canonical config bypass remains assigned to #537.
