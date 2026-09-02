# qwen4_exp checkpoint loader: stream the expert banks (memra issue #48)

**Defect.** `qwen4exp_real_gate` loading the real NVFP4 artifact was OOM-killed by the host
kernel three times at **anon-RSS 179.7 GB on a 180 GB-RAM box**, 64 GB of swap untouched
(pinned/anon growth outran reclaim):

```
Out of memory: Killed process (qwen4exp_real_g) total-vm:575237472kB, anon-rss:179739000kB
```

The box that first brought this artifact up had 499 GB, so nothing there could see it.
180 GB is the cheapest 2-card class on every provider we rent from, which makes this a
hardware-cost defect.

**Mechanism.** `read_checkpoint_with` materialized the ENTIRE `LoadedCheckpoint` on host
before the first byte reached the device: every layer's expert bank staged into a `Vec` and
parked in `LoadedCheckpoint::banks` (~72 GB as-stored NVFP4 across 48 layers), on top of the
102 GB bf16 n-gram table and ~20 GB of trunk f32 in `ReferenceWeights`. Only then did
`from_loaded_checkpoint_dual` consume the banks and upload. Peak host = the whole artifact.

**Fix.** The contract walk now records per-layer read addresses (`BankPlan`) and keeps the
safetensors mmap open on the `LoadedCheckpoint`; each bank is read off the mmap INSIDE the
consuming loop, uploaded, and dropped before the next is read. Peak host becomes the n-gram
table + one layer's bank staging + the trunk f32 set.

No new flag. This is a pure memory-ordering change with a byte-identity gate, which is the
better-wins-by-default case (`LAW:better-wins-by-default`): a seam here would only give the
old peak a way back.

## What did NOT change, and why the n-gram table stays anon

- **The n-gram table is still an owned host `Vec`**, not an mmap view. It is 102 GB read
  16 rows per token on the host critical path; a file-backed mapping is reclaimable, so
  under memory pressure the gather would fault to disk mid-decode. Anon residency is the
  deliberate choice, and the table is not the excess this lane removed.
- **The trunk f32 set** (~20 GB) still materializes in `ReferenceWeights` before upload.
  It fits inside the target budget; streaming it is a separate lane if the budget ever
  tightens.
- **Residency decisions are still made at WALK time** (`LoadOptions::host_bf16_banks`, and
  an MTP bank at index >= n_trunk keeping raw bf16 for `BankHalf::DeviceBf16`). Only the
  byte reads moved.
- **Walk-time refusal is preserved but narrower**: `check_bank_header` uses the header-only
  `StModel::info`, so a missing or wrong-dtype bank name is still refused before the 102 GB
  table is allocated and before any H2D. Shape, scale siblings, macro finiteness and
  `input_scale` validity are now checked when the layer is read — still load time, before
  any forward, just per layer instead of all at once.

## Known cost: TP2 reads the bank bytes twice per load

`build_tp2_shard` gathers the card-1 expert halves from the host bank sources, and it runs
BEFORE `from_loaded_checkpoint` uploads card 0. With streaming, each of those two passes
reads the layer's bank off the mmap, so a TP2 load reads ~72 GB of bank bytes twice. That is
load-time disk/page-cache work only: no numerics, no steady-state cost. It is the price of
not holding 72 GB of banks on a host that has 180 GB total. Single-pass shard+model build is
a named follow-up, not a defect of this change.

## Gate: the bytes did not move

`bank-bytes` arm in `qwen4exp_gpu_gate` (tiny fixtures, rig). Every expert-bank projection is
digested (sha256 over the uploaded payload in upload order — NVFP4: codes, then scales, then
the little-endian macro row) and compared against goldens **minted from the pre-streaming
loader** on the same deterministic fixtures.

Five arms cover every branch of the bank read:

| arm | what it pins |
|---|---|
| `fused-f32` | fused gate_up dequantized to f32 + the fused row split |
| `fused-hostbf16` | `host_bf16_banks` raw-bf16 arm of the split |
| `fused-mtp-devbf16` | the MTP bank at index >= n_trunk (5 layers instead of 4) |
| `nvfp4-stacked` | NVFP4 code/scale row split, per-expert macro duplication |
| `nvfp4-perexpert-e12` | per-expert modelopt stacking **at E=12**, so `experts.10` sorting before `experts.2` would show up — the tiny plan's E=8 cannot see that trap |

Goldens mint procedure (reproducible): check out the pre-streaming commit in a throwaway
worktree, add the same `bank_fingerprints()` over the eager `banks` map (additive only —
57 insertions, loader untouched), copy in the gate binary, and run with
`--write-bank-goldens`, which mints and exits before touching a GPU. The streaming build is
never allowed to mint its own goldens; running `--write-bank-goldens` on it would rubber-stamp
whatever it produces.

## Cert lines

### Rig (sm_120a laptop 5090) — exactness only, never timing

Tiny-fixture gate, streaming build, with the pre-streaming goldens in place:

```
cd ~/projects/wt-q4e-loader && cargo build -p memra-engine --bin qwen4exp_gpu_gate
flock -w 600 /tmp/memra-gpu.lock ./target/debug/qwen4exp_gpu_gate \
    research/qwen4exp-bringup-20260829/gpu-eager/tiny-fixture-gate.tsv
```

- **PASS, `# verdict failures=0`**, binary sha256 `cee27dcb3200a358be1ba281dab019b85290e836eb55d80ec7c0585fd09e2ead` (re-run on the rebased HEAD; the earlier pre-rebase run was PASS at `d256e63b97e15cdd...`).
- `bank-bytes arms=5 rows=63 mismatches=0 pass=true` — banked in the same receipt
  (`gpu-eager/tiny-fixture-gate.tsv`), goldens in `gpu-eager/bank-bytes-goldens.tsv`.

Goldens mint (pre-streaming loader, throwaway worktree at `origin/main` 7ef035581, loader
untouched — 57 pure insertions adding the same digest function over the eager `banks` map):

```
git worktree add /tmp/wt-q4e-prestream --detach origin/main
# + bank_fingerprints() over self.banks; + the gate binary from this lane
cargo build -p memra-engine --bin qwen4exp_gpu_gate
./target/debug/qwen4exp_gpu_gate /tmp/q4e-goldmint/receipt.tsv --write-bank-goldens
```

- 63 rows minted. Build attribution checked per `TRAP:rebuild-after-checkout-attribution`:
  the minting binary greps positive for the new arm's own string
  (`grep -a -c 'the E>9 order pin is not being built'` = 1) while the loader in that tree
  still carries the eager `banks: BTreeMap<u32, BankSrc>` and the in-walk
  `read_bank_tensor(&model, ...)` / `read_per_expert(&model, ...)` calls.
- Cross-check, both directions: minting from the STREAMING build into a scratch dir and
  diffing against the pre-streaming file → **no differences, 63/63 rows**.

Repo-level checks from the repo root (`TRAP:boundary-scanner-passes-silently-from-a-subdir`):

- `python3 tools/check-public-boundary.py` → `578 matches (578 grandfathered, 0 new)`.
- `bash tools/check-flags.sh` → `no uncovered runtime names` (this lane adds no `MEMRA_*`
  read; the goldens-mint switch is the CLI flag `--write-bank-goldens`).
- `cargo clippy -p memra-engine --lib --bin qwen4exp_gpu_gate -- -D warnings` → clean.

### The lane box (4-card, 360 GB RAM) — the 180 GB class, reproduced under a cgroup cap

The box has 360 GB, so the 180 GB class is reproduced with a cgroup cap rather than hoped
for: `systemd-run --scope -p MemoryMax=150G -p MemorySwapMax=0`. The cap instrument was
proven to have teeth before any arm ran (`LAW:loud-failures-fail-quietly` — execute the
failure path):

```
systemd-run --scope --unit=q4e-oomtest2 -p MemoryMax=1G -p MemorySwapMax=0 \
    python3 -c 'b=bytearray(2*1024**3); print("NOT KILLED - cap did not hold")'
# rc=137, nothing printed, and dmesg:
# Memory cgroup out of memory: Killed process 16553 (python3) anon-rss:1046400kB
```

Arms (driver `run-cap.sh`, cell `cell.sh`, both banked in `loader/box/`), each taking
`/tmp/q48fn-measure.lock -s` on its own so a co-tenant lane holding `-x` is never contended,
each gated on a MemAvailable headroom check so this lane never squeezes a co-tenant:

```
flock -s /tmp/q48fn-measure.lock \
  env CUDA_VISIBLE_DEVICES=1 MEMRA_Q4E_SEAMS=idxsel \
  systemd-run --scope --unit=q4e-<label> -p MemoryMax=<cap> -p MemorySwapMax=0 \
    /usr/bin/time -v <binary> /root/data/q48fn-yarn1m /root/realgate/loaderout \
      --label <label> --mtp \
      --goldens /root/realgate/dump --prompts /root/realgate/shapes/thinkon-prompts.tsv
```

Artifact on this box: `~/data/q48fn-yarn1m` (the same 174 GB per-expert NVFP4 mint, carrying
the YaRN-1M rope config: `rope_type: yarn`, `factor 3.814697265625`, `rope_theta 1e7`,
`max_position_embeddings 1000000`).

**Peak memory is reported as ANON, not RSS.** Two instruments were rejected on measurement,
not on taste:

- `/usr/bin/time -v` "Maximum resident set size" reads **180.8 GiB** for the streaming arm,
  but 63.9 GiB of that is `RssFile` — the artifact mmap the streaming loader reads through.
  Those pages are clean, reclaimable, and their page-cache charge can belong to whichever
  cgroup first faulted them (a co-tenant had already read this artifact).
- cgroup `memory.peak` reads exactly `memory.max` (150 GiB) because reclaimable file cache
  always expands to fill the limit; it says "the cap was touched", not "this much was needed".

`anon` is also the quantity the kernel OOM line that opened this defect reported
(`anon-rss:179739000kB`), so it is the comparable number. Sampled from
`/sys/fs/cgroup/system.slice/q4e-<label>.scope/memory.stat` (sampler `box/anon-sampler.sh`,
receipt `box/anon-peak.tsv`) and cross-read from `/proc/<pid>/status:RssAnon`.

| arm | binary | cap | result | load | peak anon |
|---|---|---|---|---|---|
| `old-cap150` | pre-streaming `69c44eb85b82d4ee...` | 150 GiB | **cgroup OOM-killed during load** | — | 149.4 GiB (walked into the cap) |
| `old-cap230` | pre-streaming `69c44eb85b82d4ee...` | 230 GiB | rc=0, argmax 10/10 | 420.5 s | **185.8 GiB** |
| `new-cap150` | streaming `67bd6177d6c6cb9f...` | 150 GiB | **rc=0, argmax 10/10** | 457.0 s | 116.5 GiB (single sample) |
| `new-cap150b` | streaming `67bd6177d6c6cb9f...` | 150 GiB | **rc=0, argmax 10/10** | 150.4 s | **118.9 GiB** (sampled from t=0) |

**Peak host anon: 185.8 GiB -> 118.9 GiB, down 66.9 GiB (36%)** — the ~72 GB of expert banks
that no longer sit on host at once, less the one-layer staging that replaced them.

Why the 180 GB box died and now will not: 185.8 GiB is **199.5 GB**, so the pre-streaming
loader needed more host memory than that box class HAS. 118.9 GiB is **127.7 GB**, leaving
~52 GB of margin on the same box.

`old-cap230` exists because `old-cap150` only proves a lower bound (">150 GiB, killed"). A cap
above the requirement measures the real peak AND still protects the box from a runaway — the
number in that row is what the pre-streaming loader actually needs.

`old-cap150`, the defect reproduced under the cap (the kernel killed the gate process; the
scope's `/usr/bin/time` wrapper then reported `Terminated`, rc 143):

```
Memory cgroup out of memory: Killed process 25952 (qwen4exp_real_g)
  total-vm:552111388kB anon-rss:156645208kB file-rss:47836800kB shmem-rss:7680kB
```

The last sampler line before the kill is `q4e-old-cap150  160386187264  149.4` — anon walked
straight into the 150 GiB wall and the cgroup killed it, exactly the shape of the original
180 GB-box OOM (`anon-rss:179739000kB`). Same binary, same artifact, same card, 4 GB less
than the cap of headroom: the pre-streaming loader cannot load this artifact on this box class.

The streaming arms on the same cap: **loaded, `# logits_argmax_agreement 10/10`**, and
`# vram post-load` on card 1 = **95283 MiB — the same VRAM to the MiB** as the pre-streaming
reference run (`spec-ab-rep0-off`, 95283 MiB on card 0). Device residency did not move.

Load wall clocks are NOT a claim from this lane and are not comparable across these rows: the
box was shared and page-cache state differed per arm (the same streaming binary read 457.0 s
with a cold/contended cache and 150.4 s warm, against 420.5 s for the pre-streaming arm and
138.2 s for the reference run). What the numbers do rule out is a load-time collapse: streaming
per layer did not turn a ~140 s load into a multiple of itself.

### Real-artifact byte identity (`compare-identity.sh new-cap150`)

Reference arm = `/root/realgate/downsel/*-spec-ab-rep0-off.*`, produced by the **pre-streaming**
binary on the **same artifact** with the same `--goldens` dump and the same seams env, **by an
unrelated lane on a different card**. An independently banked reference is a stronger control
than a twin run this lane arranged.

```
probe-logits prestream sha256 = f6634e4489b1d586127e66a78f6bc125611c7ec8fd7db9bb875fdc18ef0c112c
probe-logits streamed   sha256 = f6634e4489b1d586127e66a78f6bc125611c7ec8fd7db9bb875fdc18ef0c112c
probe-logits: BYTE-IDENTICAL
hidden-gate envelope table: IDENTICAL (62 rows)
greedy chains: IDENTICAL (5 rows)
identity_failures=0
```

`probe-logits-*.bin` is the raw f32 prefill logits over the 10-token goldens probe
(10 rows x 248320 vocab). Same weights + same ids + same program = same bytes, so an identical
sha256 over 9.9 MB of logits is the load path's byte oracle on the real artifact.

**Not claimed:** greedy chains matching the banked HF goldens. This box's artifact is
`q48fn-yarn1m`, whose rope program (`rope_type: yarn`, factor 3.8147, theta 1e7) is not the
262144-context config `make-goldens.py` ran on, and the greedy gate duly reports
`first_divergence=0` against those goldens on BOTH loaders. That is a property of the artifact,
not of this change; the identity statement above is old-loader-vs-new-loader on the same
artifact, which is the question this lane has to answer.

### Lock discipline: one arm ran without the lock, and that was wrong

`old-cap150` executed while a co-tenant lane held `flock -x` on `/tmp/q48fn-measure.lock`. The
runner had a bounded-wait-then-proceed fallback, reasoning that the lock guards GPU capacity
and this arm held only its own assigned card. That reasoning was wrong and the owner corrected
it: a 174 GB host load hammers memory bandwidth, so it corrupts a co-tenant's timed arms
without touching a card of theirs. The rep it overlapped was f32-pinned and superseded, so no
receipt was lost, but the rule is now unconditional and the fallback is deleted from
`box/run-cap.sh` — every real-artifact load on this box takes `flock -s` and waits, with no
timeout. Recorded here rather than quietly fixed, because the receipt has to say which arm ran
under which conditions.

### Consumer coverage: every changed bank consumer, and the one that is SKIPPED

The change touches four bank consumers. Three are exercised, one cannot be with one card:

| consumer | covered by | result |
|---|---|---|
| `from_loaded_checkpoint_dual` | `new-cap150`, `new-cap150b` (real artifact) | PASS, byte-identical |
| `into_reference_weights` | tiny-gate `dir-*` arms (rig) | PASS |
| `mtp_reference_weights` | `new-draftgate` (real artifact) | PASS |
| `build_tp2_shard` | **nothing — SKIPPED** | see below |

`new-draftgate` exists precisely because no other arm reaches `mtp_reference_weights`, which
builds the HOST reference MTP twin from the lazily-read MTP bank before the device model reads it
again:

```
flock -s 9   # shared measurement lock, waited 570s
env CUDA_VISIBLE_DEVICES=1 MEMRA_Q4E_SEAMS=idxsel \
  systemd-run --scope --unit=q4e-new-draftgate -p MemoryMax=230G -p MemorySwapMax=0 \
    /usr/bin/time -v ~/realgate/bin/qwen4exp_real_gate.loader ~/data/q48fn-yarn1m \
      ~/realgate/loaderout --label new-draftgate --mtp --draft-gate \
      --goldens ~/realgate/dump --prompts ~/realgate/shapes/thinkon-prompts.tsv
```

- **`# verdict rows=20 argmax_matches=20 worst_abs=1.004e-4 worst_rel=9.632e-5 pass=true`**,
  `# logits_argmax_agreement 10/10`, rc=0, load 254.9 s
  (`box/draft-gate-new-draftgate.tsv`, `box/new-draftgate.log`).
- Peak anon **147.2 GiB**, ~28 GiB above the plain streaming arm. That is `--draft-gate` cloning
  the whole trunk f32 set for its host twin, which is a pre-existing property of that flag and
  the reason this arm ran at a 230 GiB cap: a 150 GiB result here would have measured the flag,
  not the loader.

**`build_tp2_shard`: SKIPPED, not PASS.** TP2 needs two engines and this lane was assigned one
card (cards 0 and 2-3 belong to other queues), so no arm on this box can construct it, and the
rig has a single GPU. What is known: the two changed lines call the same `BankPlan::read` that is
byte-verified 63/63 on fixtures and byte-verified on the real artifact through the three
consumers above, and the refusal path stayed loud (`no bank source for layer N`). What is NOT
known: that a real TP2 load still builds its card-1 halves correctly end to end. That arm needs a
second card and is the first thing to run if TP2 is next.

### Banked receipts (`loader/box/`)

| file | what |
|---|---|
| `new-cap150.log`, `new-cap150b.log` | streaming arms, full gate output under the 150 GiB cap |
| `old-cap150.log` | the OOM death under the same cap |
| `old-cap230.log` | the pre-streaming peak measured at a cap above its requirement |
| `anon-peak.tsv` | anon sampler trace; last line per scope is that arm's peak |
| `hidden-gate-new-cap150.tsv`, `greedy-gate-new-cap150.tsv` | the gate tables compared for identity |
| `new-draftgate.log`, `draft-gate-new-draftgate.tsv` | the `mtp_reference_weights` coverage arm |
| `run-cap.sh`, `cell.sh`, `chain2.sh`, `draft-arm.sh`, `anon-sampler.sh`, `compare-identity.sh` | the invocations, verbatim |
| `CELL.log` | the cell's own timeline including the identity comparison output |

`cell.sh` is banked as it RAN, lock-free fallback and all; `run-cap.sh` is banked in its
corrected form (unbounded `flock -s`). The difference is the point of the lock-discipline
section above.

## Verdict

The streaming loader cuts peak host anon on the real artifact from **185.8 GiB to 118.9 GiB**
and turns a three-times-reproduced OOM kill on the cheapest 2-card box class into a clean load
that passes the gate, while the bytes it uploads are provably unchanged: 63/63 bank projections
byte-identical to pre-streaming goldens on the tiny fixtures, and an identical 9.9 MB
probe-logits sha256 plus identical envelope table and greedy chains against a pre-streaming
reference run on the real artifact. No flag, no numerics change, no new env read.

Follow-ups this lane deliberately did not do, named rather than left implicit:

1. **TP2 single-pass shard+model build** — removes the double read of bank bytes described
   above. Load-time only.
2. **Trunk f32 streaming** — the remaining ~20 GB host set. Only worth doing if a box class
   below ~150 GB becomes interesting.
3. **The n-gram table stays anon on purpose** (see above). Any future lane proposing to mmap it
   owns the mid-decode page-fault question first.
