# Spec-on-cache-hit — fix + gates (lane/spec-on-cache-hit, 2026-08-18/19)

Base: v0.91.0 (022d848148). Mechanism write-up (pre-fix, from source): DIAGNOSIS.md.
Fix binary for every gate below: `target/release/memra-server` built from this lane's
working tree at 022d848148 + the lane diff, sha256 prefix `81c496a338fc6b0f`.

## The fix (PORT-PLAN item 3, scoped to whole-entry restores)

1. **Publication** (`prefix_insert_from_spec_boundary`, worker.rs): a spec session's
   boundary capture now publishes the MTP draft plane (scratch rows `[0..pos)`, sliced
   append-only like the trunk KV — `SpecSession::draft_plane_ref`) and the boundary
   hidden (`SpecBoundaryCapture.last_h`, D2H of trunk hidden row `pos-1` at capture).
   `PREFIX_ENTRY_LAYOUT_VERSION` 1 → 2; plane + hidden bytes counted in `entry.bytes`
   (budget/SLRU stay byte-true); `model_prefix_entry_bytes` (budget derivation)
   unchanged and its pinned geometry test still green. Plane copy failure publishes
   trunk-only (optimization, never a dependency).
2. **Qwen/MTP conversion** (`spec_session_from_restored`, spec.rs + the admit probe,
   worker.rs): a greedy, unconstrained, spec-eligible WHOLE-entry hit re-arms a
   SpecSession instead of downgrading — restored trunk cache + draft plane copied into
   fresh scratch + entry anchor as `last_h`. The engine feeds any prompt suffix ITSELF,
   mirroring prefill_tick's program selection arm-for-arm (eager `decode_step_h` below
   PRIME_MIN_T, one `prime_cache` call at/above), fills the suffix draft rows
   predecessor-paired, and hands back a fully-warm continuation session
   (`next_pred` = argmax of the feed logits; full-cover hits seed from the entry's
   boundary logits). Every failure before trunk mutation hands the cache back → the hit
   serves PLAIN exactly as before; failure mid-feed serves the request cold-plain.
   `cached_tokens` = restored prefix only (suffix rows are computed, not cached).
3. **Gemma conversion** (`gemma_spec_session_from_restored`, gemma_spec.rs + the
   `gspec_carrier` admission arm, worker.rs): the assistant drafter holds NO
   per-session KV (it attends the trunk's), so a restored trunk + a non-empty suffix
   feed (verify-trunk program `gemma4_decode_step_t_h`, the same arm the cold
   sub-PRIME_MIN_T session prime uses; gemma4's monolithic prime refuses pos>0)
   regenerates the drafter seed hidden + pending. Full-cover (empty-suffix) gemma hits
   stay PLAIN by design (no rows to feed for the seed hidden). Solo-admission
   (`n_active == 0`) and every other banked gspec arm unchanged.
4. **Downgrade-on-hit kept** as the fallback for every non-convertible shape (no draft
   plane, sampled, constrained, partial hits, conversion declines).

### Why this does NOT resurrect the rolled-back hazard
The lcprestore NO-GO (6249b0096, default-off in c6cac1e1c) was mid-entry trunk
restores; splitiso (0b0ffa13c6) identified it as a two-programs defect, not a split-
position one. This lane converts ONLY whole-entry hits (restore at exactly `e.pos`
through the shipping `prefix_restore` path); `spec_restore_convertible` refuses
`entry_pos != fed_len`, so a partial restore can never route into a spec session.
`MEMRA_PREFIX_PARTIAL_RESTORE` stays default-off and untouched.

### Second finding (this lane's own gate caught it): the suffix-feed program pair
First fix iteration handed the suffix to the burst prime; qwen r3 (suffix hit) then
byte-DIVERGED from the plain hit at generated token ~8 while cold and full-cover rows
were identical. Cause: the generate path's tokenwise arm routes qwen35-class through
the BATCHED T=1 program (`spec_target_step_h` → `decode_step_t_core`) while
prefill_tick feeds sub-PRIME_MIN_T suffixes through eager `decode_step` — ULP-different
suffix KV rows, near-tie flip downstream. The same "one request, two numerical
programs" family as splitiso. Fix: the engine feeds the suffix mirroring prefill_tick's
arms exactly (see 2 above); r3 went byte-identical. Residual, documented: suffixes
longer than one prefill tick budget chunk differently on the plain path (accepted
prime-config class); gemma's suffix feed (verify-trunk T=n) vs plain (eager tokenwise)
passed identity on the 12B cell but remains a program pair to watch at 31B.

### v1 scope lines (explicit, all fall back to the banked plain path)
- Sampled hits stay plain (continuation seeds by argmax). Follow-up cell: sampled
  restores with suffix >= PRIME_MIN_T, where both paths prime batched.
- Constrained hits stay plain (pool-resume law kept).
- Gemma full-cover hits stay plain (no drafter seed hidden without a fed row).
- Plain-published entries (no draft plane) serve plain for qwen; gemma converts on them
  (needs no plane).

## Gate table

| # | Gate | Status | Result |
|---|---|---|---|
| 1 | `cargo test -p memra-server` (incl. new `spec_restore_conversion_rules_are_pinned`, prefix-cache accounting, pinned budget geometry) | RUN (CPU) | 304 passed / 0 failed |
| 2 | `cargo test -p memra-engine --lib` | RUN (CPU) | 100 passed / 0 failed |
| 3 | qwen real-artifact hit gate | RUN (local 5090, idle-verified) | ALL GREEN |
| 4 | gemma real-artifact hit gate | RUN (local 5090) | ALL GREEN |
| 5 | serve-smoke battery (incl. cache-meter-gate spec-off accounting) | RUN (local 5090) | see cert line below |
| 6 | mixed-load: spec-on-cache-hit under batch coexistence (`tools/spec-cache-mixed-gate.sh`) | RUN (box 2, 96GB) | qwen c1 **FULL PASS** (6/6 engaged, identity, 1.23x floor); c8/c2 = load-policy demotion by design on this card (identity + floors green); gemma bounded by solo-admission (see below) |
| 7 | 31B/27B production-artifact twin of gates 3-4 | RUN (box 2, 96GB) | **ALL GREEN both models** (see 96GB section) |

## Cert lines

- CERT unit: `cargo test -p memra-server` @ lane tree (base 022d848148) → `304 passed; 0 failed`; `cargo test -p memra-engine --lib` → `100 passed; 0 failed` (2026-08-19, local).
- CERT qwen hit-engage+identity: `tools/spec-on-cache-hit-gate.sh qwen /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf target/release/memra-server research/spec-cache-20260818/qwen-gate` (binary 81c496a338fc6b0f, local RTX 5090 24GB, GPU idle-checked before run) → **ALL GREEN**: cold spec engages (acc 0.460, cached 0); full-cover hit = continuation, cached 106/106, spec acc **0.460 == cold exactly**; suffix hit cached 106/119, spec acc 0.429 (−3.2pp vs cold); spec-on text == spec-off text byte-for-byte on ALL three rows; `[prefix-cache] spec restore` in server log. Banked: `research/spec-cache-20260818/qwen-gate/` (12 JSON receipts + both server logs).
- CERT gemma hit-engage+identity: `tools/spec-on-cache-hit-gate.sh gemma /data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf /data/ai-ml/hf-models/gemma4-12b-mtp-gguf/gemma-4-12B-it-qat-assistant-MTP-Q4_0.gguf target/release/memra-server research/spec-cache-20260818/gemma-gate` (same binary/box) → **ALL GREEN**: sampled leader publishes plain seed (cached 0, no spec); suffix hit cached 107/120 with gspec engaged (acc 0.333 — this 12B drafter pair's own scale; server log `[gspec-acc]` x2); full-cover hit cached 107/107 stays plain (documented decline, asserted); spec-on text == spec-off text on both greedy rows. Banked: `research/spec-cache-20260818/gemma-gate/`.
- CERT no-regression: `tools/serve-smoke.sh` (default 9B MTP artifact; same binary/box) → **`serve-smoke: 0 failed`** — cache-meter accounting exact (per-request + /metrics + economics), spec==plain greedy serving exactness, sampled truncation matrix, session-affinity resume exactness, gemma4 arm zero panics, Q35 mixed c=4 cold-prefill 20/20. Banked: `research/spec-cache-20260818/serve-smoke.log`.
- FAILED-RUN receipts kept: the first gate iteration (orphaned-server stop() bug, then the r3 program-pair divergence) is documented above; the fixed gate re-ran from clean state (`rm -rf qwen-gate` between runs).

## What the 96GB validation window needs to close the lane

1. `tools/spec-on-cache-hit-gate.sh` on the PRODUCTION artifacts: qwen
   `mtp-Qwen3.8-27B-NVFP4-*` and gemma `gemma-4-31B-it*-MTP` + its drafter — same
   assertions (hit-engage acc within a few pp of cold, byte identity vs spec-off).
   Special attention: gemma r2 identity at 31B (the verify-trunk vs eager-tokenwise
   suffix-feed program pair passed at 12B but is content/scale-sensitive).
2. `tools/spec-cache-mixed-gate.sh` — spec-on-cache-hit under c8 batch coexistence
   (>= 2 spec-engaged hit rows in the window, byte identity vs spec-off replay, batch
   tier >= 0.9x spec-off). Runs on the rented cloud box at the next lane boundary or the
   owner's box when delegated; NEVER on a prod box.
3. The sold-shape throughput twin: `tools/spec-cache-gate.sh` re-run (its spec-on arms
   now take restored hits on the spec path; assert req/s >= the banked 0.9x floor and
   hit-rate >= 0.95 still hold — the anchor sha check in that gate also re-pins
   exactness under concurrency).
4. VRAM watch: draft plane adds bytes/token to spec-published entries (qwen 27B scale;
   `[prefix-cache] insert` lines show resident MB) — confirm the derived budget still
   holds >= 95% hit rate on the sold shape, or bump `MEMRA_PREFIX_CACHE_MB`.

---

# 96GB window results (box 2, 2026-08-19) — lane rebased onto v0.92.0

Box: rented 2× RTX PRO 6000 Blackwell Server 96GB, box-2. Branch
replayed CLEAN onto v0.92.0 (no conflicts; tip 6ab3e50ff1 + gate-tool commits
f30273199c/9bda7c89ed). Unit gates re-run on box 2: memra-server 307/0,
memra-engine --lib 100/0. Production artifacts sha-verified: qwen trunk
`1facf36c…` + frspec-mixed drafter `a47848be…` (HF, orbench pins); gemma trunk
`5517f9ef…` (NJ served copy, template-injected) + official Q8_0 drafter
`f5a87587…`. Evidence: research/spec-cache-20260818/box2/.

## Cert lines

- CERT qwen production hit-engage+identity:
  `tools/spec-on-cache-hit-gate.sh qwen "<trunk>+<frspec>" memra-server box2/qwen-hit-gate`
  (GPU 0) → **ALL GREEN**: cold spec acc 0.483 cached 0/106; full-cover hit cached
  106/106, acc 0.483 **== cold exactly**; suffix hit cached 106/119 acc 0.348;
  spec-on == spec-off byte identity ALL rows; `[prefix-cache] spec restore` in log.
- CERT gemma-31B production hit-engage+identity:
  `tools/spec-on-cache-hit-gate.sh gemma <trunk-5517f9ef> <official-Q8_0-MTP> memra-server box2/gemma-hit-gate`
  (GPU 1) → **ALL GREEN**: sampled leader plain+cold 0/107; suffix hit cached
  107/120 with gspec ENGAGED acc 0.44 (33/75, [gspec-acc] in log); full-cover
  107/107 stays plain (documented decline, asserted); byte identity both greedy
  rows. **The 31B suffix-feed program pair (verify-trunk T=n vs eager tokenwise)
  flagged at 12B: PASSED identity at 31B on the production mint.**
- CERT sold-shape (`tools/spec-cache-gate.sh "<trunk>+<frspec>"`, c16, 3 passes,
  10% miss salts): req/s on-gate 3.023-3.024 vs off 3.022-3.026 = **1.00x**
  (floor 0.9x) every pass; hit rate **0.987** (floor 0.95) every arm+pass.
  EXACTNESS ANCHOR: FAILED — but NOT this lane (next section).
- CERT mixed coexistence (`tools/spec-cache-mixed-gate.sh`):
  - qwen c1: **FULL PASS** — 6/6 repeat rows spec-engaged under live batch load,
    0 demoted, every engaged row byte-identical to the spec-off replay, batch
    floor 1.23x.
  - qwen c8/c2: repeats take the cache, identity green, batch floor 0.95x/0.90x —
    engaged 0/6: the spec-admission LOW=2 gate demotes (this card's batch tier
    never drains below LOW; the load-policy demotion is the banked design, and
    the c1 window proves the cache-hit path engages when load admits spec).
  - gemma c8 (sampled warm, prod env ranks+trim): cache + identity + floor 1.00x
    green; engaged 0/6 — **gemma gspec is solo-admission by design** (n_active==0),
    so hits under ANY load stay plain; the c1 window caught 1 engaged row in an
    inter-request gap and it was byte-identical. Gate-tool fix banked: greedy
    gemma warm leaders ride gspec which never publishes — sampled warm
    (MEMRA_MIXED_WARM_TEMP) is the correct seed protocol (9bda7c89ed).
- CERT VRAM watch: v2 entries (draft plane + boundary hidden) on the sold shape:
  4860 tok → 310.3MB, 4924 tok → 312.3MB (~65KB/token at 27B); resident 622.5MB
  of the derived 2260MB budget (2× max entry at MEMRA_CTX=32768); hit rate 0.987
  ≥ 0.95 → **derived budget holds, no MEMRA_PREFIX_CACHE_MB bump needed**.

## Finding (owner-level, NOT lane-caused): qwen-27B spec-vs-plain divergence on this card class

The sold-shape anchor tripped: spec-on text != spec-off text for the greedy anchor
prompt — deterministic (same two shas every pass/arm). Bisection + matrix (banked
box2/anchor-*):
- reproduces SINGLE-REQUEST (greedy, 15-token prompt, 48 tokens) on a clean
  **v0.91.0** build → predates this lane;
- drafter-independent: identical sha pair trunk-only (in-file MTP) and with the
  production frspec attach;
- env-independent: every {ctx 8192/32768} × {sessions 4/18} × {prefix-cache
  on/off} spec-on run = `1e9aa0cd…`, every spec-off run = `07785b71…`;
- prompt-dependent near-tie: one token ("only" vs "exactly" at ~token 27), then
  re-convergence; the hit-gate's own prompts are byte-identical.
The spec==plain law is violated for at least one prompt on Q38-27B NVFP4-Q5K on
the RTX PRO 6000 Blackwell (sm_120, 96GB) card class, at v0.91.0+. 5090 release
batteries never ran this artifact/card combo (card-keyed-defaults law). Whether
DE prod (PP2 across two cards) reproduces is UNTESTED — prod untouched. Handed to
the owner as a base-engine finding; it does not gate this lane (all lane-added
paths are byte-identical in every cell above).
