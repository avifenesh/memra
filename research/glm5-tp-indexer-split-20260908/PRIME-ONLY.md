# Prime-only indexer split, 2026-09-09

Baseline: `b4a719ed48cac74c032e2cb649607c1833a83152`, memra draft #395.
Prior pair receipt: [darklanes #527](https://github.com/avifenesh/darklanes/pull/527),
[RESULTS.md](RESULTS.md). This follow-up changes dispatch, not CUDA arithmetic.

`MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME` is default OFF, decide-by 2026-09-22.
Only grouped-prime chunks (`t>1`) can enter the pool split. The shared admission helper
returns false for `t<=1` before reading the flag. Both symmetric decode split dispatches
and their scalar PRE workspace code are deleted. The old flag is no longer read; no
future decode door is retained. CHECK remains a prime-only diagnostic.

The previous pair's prime IDs were identical. Its decode arm was negative at 128k and flat
at 1M: merge cost was about 128 ms/GPU over 159 steps plus 36-40 ms exchange. Shared
candidate/exchange/merge kernels remain because grouped prime uses them.

## Validation

All cargo commands ran remotely on one non-serving B200, using a dedicated source and
target directory, `nohup`, `MEMRA_CUDA_ARCH=100a`, and `flock /tmp/memra-gpu.lock`.
The local rig ran no cargo, tests, benchmark or smoke server.

- Formatting: PASS, `cargo fmt --all -- --check`.
- Release build: PASS in 3m47s, `cargo build --release -p memra-engine --bin glm5-tp2-box-probe -j 16`.
- Clippy: PASS in 27s, `cargo clippy --release --all-targets -j 16 -- -D warnings`.
- CPU target: PASS, 4 passed / 0 failed / 3 GPU tests ignored, `cargo test --release -p memra-engine --test glm5_tp_indexer_split -j 16`.
- Engine library: PASS, 463 passed / 0 failed / 19 ignored, `cargo test --release -p memra-engine --lib -j 16`.
- Single-device GPU merge: PASS, 1 passed / 0 failed.
- GPU range bit-identity: PASS at RP=1 and RP=2, 1 passed / 0 failed in each process.

Raw logs, the release binary SHA-256, changed Rust source hashes, job timestamps and exit 0
are in [receipts/prime-only-20260909](receipts/prime-only-20260909/). The local changed Rust
files match the remotely formatted, built and tested source hashes.

The first clippy attempt failed because the isolated source copy omitted tokenizer test
fixtures under `research/reasoning-schema-20260823`. The missing tracked fixtures were
copied from the same lane before rerunning; no source patch was needed for that failure.

The CPU regression runs the production flag helper in child processes with unset/OFF/ON
prime flag values and the retired flag ON, testing decode before and after the prime latch.
The single-device merge gate stages exact rank-major candidate words and compares every
index against both the replicated GPU selector and the CPU oracle, including ties,
nonfinite values, causal tails, odd pools, ragged k, and 1M pool sizes. Transport is not
simulated as a pass: the existing two-device exchange test remains a separate pending gate
on this follow-up binary, since only one B200 was available.

Required post-deploy pair cell: 1M prime OFF/ON, 1M decode OFF/ON interleaved x3, and
byte-identical IDs in every arm. Confirm retained prime improvement and no decode regression.
Vendor-default sampled requests and eight-turn cache-on continuation remain serving gates.
No full-model performance or default-promotion claim is made for this follow-up binary.

The pre-commit fmt hook uses the matching remote receipt for this commit. Push uses
`MEMRA_SKIP_PERF_CI=1`; hosted CI remains required. The existing lane worktree stays open
with the draft PR; this task's remote build directory and local scratch are removed after
receipt collection.

## Current-main integration

The first push could not start PR CI because `research/INDEX.md` conflicted with current
main. Merge main `9396a817e85ce2c724ef85a3284de6329ccdd7b9`, preserving both index entries. Engine files
merged without conflicts; the prime-only dispatch patch is unchanged.

The integrated v0.137.0 tree passed remote fmt, release build (3m46s), full-workspace clippy
`-D warnings` (3m31s), the CPU target (4 passed), engine library suite (466 passed,
20 ignored), and server suite (652 passed, 4 ignored). Single-device GPU merge and RP=1/2
range bit-identity each passed again, with final exit 0. Raw logs and matching source hashes:
[receipts/prime-only-integration-20260909](receipts/prime-only-integration-20260909/).

The first server run passed 636 tests and failed 16 because the isolated copy omitted tracked
DSV4, Gemma, GLM and OpenRouter schema fixtures. After copying those files from this same
lane, the full server suite passed. The staging-failure log is retained; no source change
was needed. This integration still makes no new pair performance claim.

## The pair cell that flipped the default, 2026-09-10

The OFF default carried `decide-by: 2026-09-22` and named the cell it was waiting on: "the
prime-only pair cell: 1M prime OFF/ON, 1M decode OFF/ON x3 and IDs". That cell ran on
2026-09-10 on the dev pair, vast 50431646, 2x B200 SXM. Same probe binary sha
(`ee2a3a003984...`), same artifact, same prompt sha, and one variable between arms; each arm
writes its own `env.txt` and the OFF/ON pair differs in exactly that variable.

### 1M, n=3 per arm, OFF and ON alternating in wall-clock order

| arm | split | prime_s | decode_tok_s | tape_sha16 |
| --- | --- | --- | --- | --- |
| idx-1m-on-a | ON | 484.1133 | 59.199 | b7b8d994ab601e43 |
| idx-1m-off-b | OFF | 736.3056 | 59.222 | b7b8d994ab601e43 |
| idx-1m-on-b | ON | 484.1200 | 59.285 | b7b8d994ab601e43 |
| idx-1m-off-c | OFF | 736.4605 | 59.423 | b7b8d994ab601e43 |
| idx-1m-on-c | ON | 483.9467 | 59.029 | b7b8d994ab601e43 |
| idx-1m-off-a2 | OFF | 735.8988 | 59.237 | b7b8d994ab601e43 |

Means: prime 736.2216 s OFF (spread 0.562) against 484.0600 s ON (spread 0.173), **-34.25%**.
Decode 59.294 against 59.171 tok/s, **-0.21%, flat**.

`idx-1m-off-a2` is the honest exception to the interleave: the third OFF repetition ran an
hour after the other five because its first attempt was destroyed by a duplicate runner on the
box. It landed within 0.562 s of the two OFF arms taken an hour earlier, which is itself the
evidence that the box did not drift across that hour.

### 128k, n=1 per arm

| arm | split | prime_s | decode_tok_s | tape_sha16 |
| --- | --- | --- | --- | --- |
| idx-128k-off | OFF | 38.4299 | 69.440 | f44054f64a912489 |
| idx-128k-on | ON | 34.4500 | 69.641 | f44054f64a912489 |

Prime **-10.36%**, reproducing the combined door's -10.3%. Decode **+0.29%, flat**.

That decode row is the point of the restriction. The combined door measured 128k decode at
**-4.2%** (69.852 -> 66.947 tok/s), and that is what killed the decode half. With the decode
split deleted and the door confined to prime, 128k decode returns to flat while the prime win
is kept in full.

### Identity

Output tapes are byte-identical between OFF and ON at both contexts
(`b7b8d994ab601e43` at 1M, `f44054f64a912489` at 128k). Two dedicated oracle arms compare the
planes themselves and both pass:

```
[glm5-tp-indexer-split] CHECK: merged idx plane byte-identical to the replicated selection (layer 3, t 4096, width 2051)
```

`idx-1m-check` rc=0 and `idx-128k-check` rc=0. Their own `prime_s` values (1285.4165 s and
79.7361 s) are the cost of running both planes and comparing them, and are **not** perf rows;
each receipt says so in its own `NOTE.md`.

### Engagement gate

The cell's first runner marked every ON arm rc=93 against a gate requiring
`indexer=pool-split .*t=1 `, a decode shape, from this prime-only door. The corrected gate
matches the shape the door actually engages in, `indexer=pool-split ... phase=prime`, and
carries the red half: with the door OFF that line must be ABSENT. All three OFF arms and both
128k arms passed both halves.

### Verdict

Winner at both contexts: prime -34.25% at 1M and -10.36% at 128k, both far past the 3% door
bar, decode flat at both, output and merged planes byte-identical at both. The default flips
**ON** with `=0` as the rollback, and the `FLAGS.md` row moves in this PR. `decide-by:
2026-09-22` is satisfied ahead of its date by exactly the cell it named.

Full lane write-up and per-arm receipts: darklanes
[#585](https://github.com/avifenesh/darklanes/pull/585),
`research/glm5-dev-pair-20260910/LANE.md` section "Cell 3" and `receipts/cell3/`.

## The lane guard matched a shape this door cannot print (2026-09-10)

`run-cell.sh`'s lane-class engagement guard read:

```
grep -E 'indexer=pool-split .*t=1 ' "$out/run.log" || rc=93
```

The door announces once per process from the grouped-prime path, and that line hard-codes
`phase=prime` with `t` at the chunk width:

```
[glm5-tp-indexer-split] engaged indexer=pool-split layer=3 t=4096 pools=1024 rank0=512 \
    candidates=512 exchange=device-signal phase=prime
```

So `t=1 ` is a DECODE shape, and decode ignores this door by construction (decode split dispatch
was deleted when `MEMRA_GLM5_TP_INDEXER_SPLIT` became `MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME`).
Every `split=1` lane row therefore failed `rc=93` while engaging perfectly. It is a guard that
could not pass, which is the same class of defect as a check that cannot fail.

Found on the 256k depth cell (darklanes, dev pair vast 50431646, 2026-09-10): four rows, both ON
arms carrying the engagement line and `Exit status: 0` from the probe itself, and `rc=93` from
the guard alone.

The repair matches what the door prints AND carries the red half, verified against the four
already-banked logs with no GPU time:

| row | repaired guard | old guard |
|---|---|---|
| `idxsplit-on-a-256k` | passes | fails (this was the rc=93) |
| `idxsplit-on-b-256k` | passes | fails (this was the rc=93) |
| `idxsplit-off-a-256k` | red half holds, no announcement | n/a |
| `idxsplit-off-b-256k` | red half holds, no announcement | n/a |

The OFF rows are what make the ON check mean something: with the door off the engine prints
`indexer-state=replicated indexer=runtime-selected` and no `[glm5-tp-indexer-split]` line at all.
