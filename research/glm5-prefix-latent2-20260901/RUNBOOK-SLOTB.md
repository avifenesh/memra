# Slot-B qualification runbook: the glm5 cache fix on the serving shape

Executor: the launch coordinator's slot-B window on the glm5 serving box (identity in
the private ops repo). One card-set, the SHIP RECIPE, three server boots (arms), then
the greps. Turnkey: `battery2.py` in this directory; prompts = the banked agent pool
(`l3-ab-prompts.json` class / the box's own prompts.json from the parent window).

## Pins (record in every receipt)

- memra commit of this lane's merged head; binary built via serving/build-artifact.sh
  (the two-pin build); `git log -1` + `strings` checks:
  `strings memra-server | grep -c 'MEMRA_HYPER_SUFFIX_PRIME'` >= 1 and
  `grep -c 'MEMRA_GLM5_SPEC_PREFIX'` >= 1 (rebuild-attribution law).
- Artifact + drafter shas per the launch lane's artifact.lock.
- `NVIDIA_TF32_OVERRIDE=0`. Ship recipe env: 3-card PP3 `MEMRA_PP_STAGES=3
  MEMRA_PP_SPLITS=15,30` + recipe pins + the serving spec shape
  (`MEMRA_GLM5_SPEC=1 MEMRA_GLM5_DFLASH=<pinned> MEMRA_SPEC_PMIN=0.7`, auto K).

## Arms (each: fresh boot, READY receipt, battery2.py, log greps, stop verified)

1. `off`  — cache flags UNSET, `MEMRA_PREFIX_CACHE_MB=2000`:
   `python3 battery2.py out-off off` must PASS; greps:
   - `grep -c 'snapshot failed (latent' serve-off.log` > 0 (the refusal stands),
   - `grep -c '\[suffix-prime\]' serve-off.log` == 0 (flag off, no receipts),
   - `grep -c 'restored=1' serve-off.log` == 0.
   This is the negative control: today's behavior, byte-identical.
2. `on`   — `MEMRA_PREFIX_LATENT=1 MEMRA_HYPER_SUFFIX_PRIME=1 MEMRA_GLM5_SPEC_PREFIX=1
   MEMRA_PREFIX_CACHE_MB=4096`:
   `python3 battery2.py out-on on` must PASS; greps:
   - `grep -c 'snapshot failed (latent' serve-on.log` == 0,
   - `grep -c '\[prefix-cache\].*glm5-boundary' serve-on.log` > 0 (spec captures publish),
   - `grep -c '\[glm5-spec\] RESTORED session' serve-on.log` > 0 (spec restores engaged),
   - `grep -c 'restored=1' serve-on.log` > 0,
   - `grep -c '\[suffix-prime\] ENGAGED' serve-on.log` > 0 (plain-route suffixes, C1b),
   - `grep -c '\[suffix-prime\] TOKENWISE' serve-on.log` == 0 (no silent fallback under
     the serving tick budget — if this fires, finding 3a's door is open on this config:
     STOP and report, do not paper over),
   - `[glm5-acc]` present on spec rows (never-serve-greedy law).
3. `bust` — same env as `on`: `python3 battery2.py out-bust bust` must PASS
   (cache_salt rotates per request; cached_tokens == 0 everywhere). The C2/C3 deltas
   between `on` and `bust` are THE cache numbers for the pricing row.

## C4: entry-digest compare (PR #93 review finding 3c — R16's artifact-level bar)

On the `on` boot, `MEMRA_PREFIX_SPLIT_TRACE=1` is added to the env. After C2:
1. From serve-on.log take the LAST `[prefix-cache-*state*]` digest for the deepened
   entry class (the turn-3+ republish, role=snapshot/why=glm5-boundary or seed).
2. Fresh boot (same env). Replay `out-on/c2-messages-turn3.json` ONCE (cold). Take the
   seed digest for the same rendered prompt.
3. The two `state_sha256` values must be EQUAL — a deepened entry is byte-identical to a
   cold-primed entry of the same depth. Inequality = FAIL, report verbatim.

## zqx round trip

`latentprobe.py` (research/prefix-restore-toolcall-20260828/) on the `on` arm: tool /
recall / bare, cold AND restored, greedy + sampled — restored reps must show
engagement (cached == prompt) and pass content.

## Acceptance (feeds the pricing row + launch gate 7)

- C1 + C1b byte identity, both PASS, with engagement receipts.
- C2 `on`: cached_tokens > 0 from turn 2, per-turn TTFT; the bar vs `bust`:
  restored-turn TTFT ~= bust-turn TTFT x (suffix / total) + restore overhead — the
  "beats cold by roughly the cached fraction" law. Report the full per-turn table.
- C3 `on` hit rows: near-instant (the parent lane's green class), cached == prompt.
- Gate 7 flips: from asserting `cached_tokens == 0` to asserting engagement
  (cached_tokens > 0 on warm turns + the grep set above).
- Numbers to the coordinator: per-turn TTFT (on vs bust), cached_tokens per turn,
  C3 depth table, byte-identity verdicts, spec acc/cycle on restored vs cold rows.

## Discipline

- Serialize behind the box's lane lock; GPUs to 0 MiB between arms; PID-verified stops
  scoped by MEMRA_ADDR (the parent window's co-tenancy incident fix).
- Loop-scored rows excluded from aggregates and reported separately (greedy-instrument
  law). Sampled rows are vendor-default; greedy only where the cell is a byte oracle.
- Any engine-error / 5xx: stop the arm, bank the log, report — never improvise env.
