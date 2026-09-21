# Session C day nineteen: memra#617 (the server refuses an unknown argument at boot), the arena lease handoff scoped, the DFlash tail slice pre-registered

Scope (lead brief, day 19): (1) memra#617, filed by the lead from this lane's day-18 item 6 finding
(`flag_silently_accepted=True` on both cards): `memra-server` refuses an unknown command-line argument at
boot, names the token, exits non-zero before any device work, keeps every documented argument working; a
CPU test; the day-18 `serverdoor` cell shape re-run on the fixed tree on the local RTX 5090; the receipt
commented on #617 (not closed). (2) The arena lease handoff from the `HOSTPREFIX-DOOR.md` review table's
missing list, scoped first: implemented only if it is one bounded change. (3) The DFlash tail slice,
pre-registration only unless a cell fits. (4) The records. Tree: lane merge of main `653c997f4`
(`05b42ede9`, #614's `small_m_tier_max()` fix of memra#427 and the integ24 records; PR #616 with this
lane's day 18 was still OPEN at the start, so main does not carry day 18 yet), then the marker deletions
below, then the fix. Every push of this lane today is in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook prints `UNQUALIFIED DEVELOPMENT ...
no GPU qualification claimed` and appends a `log_skip` row): no qualification is claimed by anything
here; every cell is `executed-not-qualified`. Nothing here is a support state. No commits on main, no
other lane's worktree touched, no PR opened.

## Push mode, stated

`git push origin lane/spill-c-20260919` at `05b42ede9` ran with `MEMRA_RELEASE_QUALIFICATION_MODE=development`,
verbatim from the hook: `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at
05b42ede911c7161ffa08fe76e6f0713b3ae99d5; no GPU qualification claimed` then `pre-push: skip recorded in
.../.git/memra-gate-skips.log`. The same shape at every later push today; each is listed in the Push
section at the end. The range carries engine source (main's `lib.rs` change from #614 and this lane's
`memra-server` change), which is why the mode is needed; no qualification is claimed.

## First action: the merge and the ruling 27 census

`git fetch origin && git merge --no-ff origin/main` merged `653c997f4` cleanly (no conflict, no hand
resolution) as `05b42ede9`. Ruling 27 (integ25: a hand-resolved merge is followed by
`tools/check-conflict-markers.sh` before its commit; a marker line in a tracked file is a push refusal)
was applied even though this merge was not hand-resolved: the census script is on
`origin/lane/spill-integ25-20260921` (PR #616) and not yet on main, so it was run from that branch's copy
out of tree (`git show origin/lane/spill-integ25-20260921:tools/check-conflict-markers.sh | bash -s .`
shape; nothing copied into this lane). It read RED on the merged tree, verbatim:

```
check-conflict-markers: REFUSED, marker lines in tracked files:
research/INDEX.md:569:||||||| parent of 8faa37ca4 (fix(engine): a 16-row prime segment rides the prefill program, not the batched decode/verify tier)
research/tune-data/perf-ci.jsonl:1147:||||||| parent of cdb3b49a1 (data(perf-ci): local-ci --perf rows and stage log for the exact lockstep CPU rows lane; READM
```

Both lines are main's (from the #614 and #604 rebases), the same two the lead removed on the #616 branch
(read from `git diff 653c997f4 origin/lane/spill-integ25-20260921 -- research/INDEX.md
research/tune-data/perf-ci.jsonl`: one deletion each, no other change). The merge push had already gone
(the census is not in main's hook yet, and the merge itself was clean); this lane then deleted exactly
those two lines (`6d09ac60f`), so the lane tree reads `check-conflict-markers: OK (no conflict marker
line in tracked source or docs)` and every `perf-ci.jsonl` line parses (checked with `json.loads` over
the file). The deletions are byte-identical to the lead's, so the eventual merge with #616 is a same-side
deletion on both files, not a conflict.

## Task 1: memra#617, `memra-server` refuses an unknown argument at boot

**Why the token was accepted (from source).** `serve_with` (`crates/memra-server/src/lib.rs`, the
`std::env::args().skip(1).collect()` at the top) consulted argv for exactly two things: `--version` /
`-V` (an `any()` over the tokens) and `auth::run_cli(&args)` (`crates/memra-server/src/auth.rs`), which
returns `None` unless `--gen-key` or `--revoke-key` is present and otherwise reads `--lane`,
`--rate-limit` and `--keys` by `position()` plus the next token. No code path walked the whole argv; a
token none of those lookups matched was never seen, so the boot continued as if argv were empty. The
same class as memra#483 for unknown `MEMRA_*` names.

**The fix (the commit named in the Push section).** New module `crates/memra-server/src/argv.rs`,
`pub fn validate(args: &[String]) -> Result<(), String>`: a single left-to-right walk. `--version` and
`-V` stand alone; `--gen-key` and `--revoke-key` take the next token as their value (a missing value
stays `auth::run_cli`'s usage error, unchanged); `--lane`, `--rate-limit` and `--keys` take a value and
are accepted only beside `--gen-key` or `--revoke-key` (alone they used to boot a server that ignored
them: `memra-server --keys /p` served with no keyring and said nothing, the same silent class); any other
token is `unknown argument "<token>" refused at boot; accepted arguments: <the list>`. `serve_with`
calls it as its first statement after collecting argv, before the version print, before
`validate_stream_prefill_config` (the first environment read), before `auth::run_cli`, the build
identity line and every device statement; a refusal is `[server] FATAL: <message>` on stderr and
`std::process::exit(2)` (the usage exit class `run_cli` already uses). The accepted list is one
`pub const ACCEPTED` printed in every refusal so the message and the tables cannot drift. No new
`MEMRA_*` read, no numeric or serving-path change, no new dependency. The one deployment binary that
links `memra_server::serve_with` (read in the private product repo: it calls `serve_with(wiring)` and
reads no argv of its own) is unaffected.

**Every documented argument, from the parser (not from memory), all still accepted:**

| Argument | Shape | Where declared | Documented |
|---|---|---|---|
| `--version`, `-V` | bare | `lib.rs` `serve_with` | `docs/SERVING.md` "Reading it without a GPU" |
| `--gen-key <tenant>` | one value | `auth.rs` `run_cli` | `docs/SERVING.md` "Lifecycle CLI", `docs/FLAGS.md` `MEMRA_API_KEYS` row, `tools/apikeys-gate.sh` |
| `--revoke-key <prefix>` | one value | `auth.rs` `run_cli` | same |
| `--lane interactive\|batch` | one value, beside `--gen-key` | `auth.rs` `run_cli` | same |
| `--rate-limit N` | one value, beside `--gen-key` | `auth.rs` `run_cli` | same |
| `--keys <path>` | one value, beside `--gen-key` or `--revoke-key` | `auth.rs` `run_cli` | same |

**CPU tests, both green on the local rig (`cargo test --release -p memra-server`, under the CPU quota).**
Unit, `crates/memra-server/src/argv.rs` `argv::tests`: `empty_argv_boots`,
`every_documented_shape_is_accepted` (ten shapes from the table, including `--keys` before
`--revoke-key`), `unknown_flag_is_refused_and_named`, `unknown_token_after_a_valid_one_is_refused`
(`--version --bogus`; a trailing positional after a full `--gen-key` shape), `positional_token_is_refused`,
`key_modifier_without_a_command_is_refused`, `a_missing_value_is_left_to_run_cli`; verbatim `test result:
ok. 7 passed; 0 failed; 0 ignored; 0 measured; 774 filtered out`. Integration, boots the real binary
(`env!("CARGO_BIN_EXE_memra-server")`, `MEMRA_API_KEYS` removed from its environment),
`crates/memra-server/tests/argv_boot.rs`: `bogus_flag_is_refused_before_boot` (`--experts-via-tier`: exit
2, `[server] FATAL:`, the token quoted, the accepted list, no `[server] build:` line so nothing booted,
empty stdout), `bogus_flag_beside_a_valid_one_is_refused`, `positional_token_is_refused`,
`key_modifier_without_a_command_is_refused`, `version_flags_are_accepted` (exit 0, `memra-server ` and
`system_fingerprint ` on stdout), `key_lifecycle_shapes_are_accepted` (`--gen-key acme --keys P` prints
`mk-acme-...`; `--gen-key bulk --lane batch --rate-limit 2 --keys P`; `--revoke-key mk-bulk- --keys P`
prints `[revoke-key] mk-bulk-`), `gen_key_without_a_keyring_is_run_cli_usage_not_an_argv_refusal` (exit
2 with `no keys file`, not `unknown argument`: admitted by argv, refused by `run_cli`); verbatim `test
result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out`. The battery's
`cargo test --release -p memra-server` runs both (integration tests are part of the package's test set).
`docs/SERVING.md` gained one sentence under "Reading it without a GPU" naming the refusal.

### Cell `serverdoor19` (local RTX 5090, pre-registered before the run)

**Question.** On the fixed tree, is the door's flag on `memra-server`'s argv refused at boot
(`flag_refused=True`, the day-18 line read `False`), and does `MEMRA_KV_HOST_CONTRACTS` as an
ENVIRONMENT variable still behave as documented (ON boots and serves; a value outside `0`/`1` refuses
at boot, `HOSTPREFIX-DOOR.md`)?

**Expectation from source.** `argv::validate` refuses `--experts-via-tier` with exit 2 and one stderr line
before the build identity; `kv_host_contracts_door()` (`worker.rs`) is untouched, so `=1` prints
`[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): ...` after model load and serves, and `=on`
refuses with `MEMRA_KV_HOST_CONTRACTS="on" refused: the host tier contracts door takes exactly `1` (on)
or `0` (off, the default); it does not fall back to off`.

**Shape** (`day19-cell.sh serverdoor19`, `day19-replay.py`): the day-18 `serverdoor` boot (same
environment: `MEMRA_COMPAT=openai`, `MEMRA_MODELS=gate=<the Qwen3.5-9B NVFP4 MTP GGUF>`,
`MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`, a lane port behind `tools/port-guard.sh`, readiness polled on
`/v1/models`, one completion at `max_tokens` 16 temperature 0, TERM) run three times in one collector
lock hold on `/tmp/memra-5090.lock` (`--rig rtx5090`): `flag` = `--experts-via-tier` on argv, no door
env; `envon` = no flag, `MEMRA_KV_HOST_CONTRACTS=1`; `envbad` = no flag, `MEMRA_KV_HOST_CONTRACTS=on`.
Binary and artifact SHA-256 banked in `ev/`.

**Rule** (PASS = `flag_refused; env_door_documented`, every clause required): `flag` exit 2, log carries
`[server] FATAL: unknown argument "--experts-via-tier" refused at boot`, no `[server] build:` line,
never ready, zero bracketed door lines (`[experts-via-tier]`, `[expert-host-slru]`, `[expert-gpu-slru]`,
the day-18 correction: the refusal line carries the bare substring and is not a door line), the log is
that one line; `envon` ready, one completion with text, the door ON boot line present, no `FATAL` line,
zero MoE door lines; `envbad` exit non-zero, the `"on"` refusal line present, never ready. Reported beside
the verdict: every exit code and the telemetry regime. N=1 per arm (correctness cell, pass/fail; no
timing is read).

## Task 2: the arena lease handoff, scoped (`HOSTPREFIX-DOOR.md` missing item 1)

**What the handoff is.** Under the door every host-resident KV byte is a governor-charged lease bound to
a program identity: the D2H route (Option B) allocates each destination as a `CudaPinnedLease` from the
`TransferEngine` pool (`tier_transfer.rs` `alloc_host_kind`: `request.bytes.pinned = bytes`,
`governor.reserve`, `PinnedBacking::alloc` with the `PinnedKind` flags, released in
`PinnedAllocation::drop`), and `tier_charge` (`worker.rs`) charges the pageable remainder per entry. The
startup arena (`MEMRA_GLM5_TP_KV_HOST=1`, `PinnedHostArena::reserve`: one `cuMemHostAlloc(bytes,
PORTABLE)`, a best-fit free list, slices as `PinnedHostBuf { region: Some(..) }`, sized by
`MEMRA_KV_HOST_MB`, stored on `hpx.arena` with `hpx.budget = bytes`) is charged by nobody the governor
knows: `tier_charge` refuses `tier fixed-arena lease handoff pending` while `self.arena.is_some()`, so
`host_tier_arena_refusal` refuses the door at boot instead. "Handing the lease over" means the arena's
bytes become governor-known and its slices become the door's destinations, so the two contract rules the
door states (`every resident byte charged once`, `every KV plane a receipted contract lease`) hold on the
arena as they hold on the pageable tier.

**Which rule it completes and what would prove it.** Rule: charged-once plus receipted planes on the
arena. Proof: a CPU conformance test (the governor's `pinned` used equals the arena capacity at boot;
a demote with the arena charges `pinned = 0` plus the pageable remainder and releases on evict; a slice
used as a D2H destination carries the completion checksum `bind_tier_image` checks) and the door's
identity gate, default and plain, on the target card with `MEMRA_GLM5_TP_KV_HOST=1 MEMRA_KV_HOST_MB=<n>`
in both arms (the day-17 `arena-pair` shape, door OFF today, would become the OFF arm).

**Is it one bounded change? No.** Three shapes were read (code walk, every claim at a line in the tree):

- (a) charge the arena's capacity once at boot (one `ResidentCharge` on the governor's `pinned`
  dimension, held on `HostTierContext`), drop the refusal, charge arena entries `pinned = 0`. Touches
  `host_tier_context`, `host_tier_governor`, `tier_charge`, `host_tier_arena_refusal` and its two
  callers, `HostTierContext`, and the demote's pinned/pageable split (`host_demote_prefix_ref`, which
  today subtracts the KV plane bytes because every plane is a `Contract` lease). No new type or env
  read. BUT: with the arena, `host_entry_from_device` takes the `host_plane_from_device` arm (the
  contract route is `Some(tier) if host.arena.is_none() && !is_glm`), so every arena plane is
  `HostPlaneBytes::Pinned`, receipt-less, and `bind_tier_image`'s receipt check runs in its OFF form.
  That breaks the census invariant `HOSTPREFIX-DOOR.md` records ("under ON every KV plane of a resident
  entry is `Contract`; no `Pinned` KV plane exists"), and it makes the door a two-surface door: charged
  and identity-bound but unreceipted on the arena, receipted on the pageable tier. That is a second
  host program under one flag, the shape the one-program law forbids; (a) alone is not the handoff.
- (b) per-entry charging of arena slices (`tier_charge` charges `pinned = sum(sizes)` on the arena arm
  and threads the `ResidentCharge`): the same receipt-less planes as (a), plus a double-count hazard
  against the arena's own budget; rejected for the same reason.
- (c) arena slices as the D2H contract destinations, so the planes are receipted: blocked by the
  transfer engine as it stands. `CudaTransfers::validate` requires `o.host` to be a `CudaPinnedLease`
  whose `PinnedBacking` matches the owner stream's context with `Rc::strong_count == 1`;
  `CudaPinnedLease` is constructible only by `alloc_host_kind` and the `Destination::Host` take;
  `PinnedBacking` owns its tracking `CudaEvent` and frees with `free_host` in `Drop`, which an arena
  `Region` (freed back to the free list) cannot do. (c) needs a new `PinnedBacking` variant or a new
  lease type in `memra-engine` (`tier_transfer.rs`: `alloc_host_kind`, `validate`,
  `PinnedAllocation::drop`), the arena-aware destination allocation in `host_kv_planes_through_contract`,
  the charge-once accounting of (a), and the promote route's `plane_up` reading arena-backed leases.
  That is a new type across two crates plus five functions, and a design decision (does a
  governor-charged arena still need `MEMRA_KV_HOST_MB` as its own budget, or does the governor's
  `pinned` capacity size it), which is the lead's, not this lane's.

**Verdict: wider than one function or one contract rule; stopped at this note.** What it needs, in
order: (1) the lead's ruling on (c)'s budget question (one budget or two); (2) a lane slice landing the
arena-backed lease type in `memra-engine` with its CPU conformance (constructible from a `Region`, the
tracking event, drop back to the free list, `validate` admitting it); (3) the worker changes of (a) and
(c) together, behind the existing door and the existing arena flag, no new env; (4) the identity gate
default and plain on the target card with the arena in both arms, and the day-17 arena pair re-read
under the door. Until then the boot refusal stands and the review table's item 1 stays "engine work,
not started" with this note as its scope. Existing tests that pin the current shape and would move:
`host_tier_arena_refusal_names_the_arena_and_passes_without_it`,
`option_b_contract_route_is_door_only_and_keeps_the_frozen_demote_order` (greps the
`host.arena.is_none()` guard), `host_cache_tenant_share_reservation_evicts_nothing_without_an_arena_and_books_a_copy_failure_wasted`,
and the eight `pinned_arena_*` tests in `pinned_host.rs`.

## Task 3: the DFlash tail slice, pre-registration only (`HOSTPREFIX-DOOR.md` missing item 2)

**Question.** Can the door bind a prefix entry that carries a DFlash draft KV tail (`PrefixEntry.
dspark_draft: Option<DflashKvTail>`, produced only under `MEMRA_DSPARK_SPEC=1` with
`MEMRA_DSPARK_DRAFT=<export dir>`, restored under `MEMRA_DSPARK_PREFIX_RESTORE`) with its own
`ProgramIdentity`, so the tail's f32 planes are receipted contract leases like the trunk's and the
entry stops being refused by name (`host_tier_entry_class` refuses `dflash_tail` today)?

**Rule.** The identity gate, default environment with `MEMRA_DSPARK_SPEC=1` and the drafter, OFF against
ON: `ALL GREEN` both arms, the same verdict lines, equal demote bytes, and under ON one receipt per tail
plane per draft layer (`Role::Tail`), with a promote that restores the tail and re-arms the drafter
(the restore's own verdict line unchanged between arms). A refusal line under ON is a FAIL of the
slice, not of the door.

**Shape.** The day-14 draft-plane pattern: a third entry class beside plain and MTP-draft, its program
naming the drafter's artifact (a byte manifest of the export directory: `config.json` plus the
safetensors, since no GGUF digest covers it), `DflashCfg` as the plan and an f32 tail numeric class;
`HostF32::down` per layer replaced by the contract D2H route; `plane_up` on promote. Engine work before
any cell: the manifest digest, the class, the tail through the route. No flag; the door's own.

**Artifact and whether it exists.** The 27B target (`Qwen3.8-27B` NVFP4 MTP GGUF) and the DFlash2
drafter export (`tiyuvta/Qwen3.8-27B-DFlash2-memra`, the qualified serving route per `docs/MODELS.md`)
are both present on the local rig's model store. On the target card this lane knows the GGUF artifacts
under the read-only artifact dir by the brief's layout; a drafter export directory is not a `.gguf`,
so whether one is staged there is unknown to this lane (not inspected). No gate boots a DFlash drafter
on the card today (`HOSTPREFIX-DOOR.md`), so even with the artifact the cell needs the gate to grow a
drafter arm first. No cell fits today; nothing run.
