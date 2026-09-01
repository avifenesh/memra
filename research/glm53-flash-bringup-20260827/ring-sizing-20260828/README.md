# MEMRA_DSA_INDEX_RING sizing, lane/glm53-ring-sizing (2026-08-28)

The regression this lane closes was measured by lane/glm53-box on the bench box, three arms one
env flag apart on the same binary (`../rebaseline-and-surface-20260828/`, receipts 13 and 14,
FINDINGS section 5):

| prompt tokens | ring ON (default) | ring OFF (`=0`, same binary) | pre-ring binary |
|---|---|---|---|
| 940 .. 4630 | 200 | 200 | 200 |
| 5550 / 6470 / 7300 | 500 | 200 | 200 |
| largest served | **4630** | 7300 | 7312 |

`prefill error: layer 3: DSA k-pool selection failed: indexer tail ring lapped: 5120 rows cannot
cover pools from row 0 ...`, inside a configured `MEMRA_CTX=8192`.

## Receipts in this directory

| file | arm | what it shows |
|---|---|---|
| `01-red-host-sizing-gate.txt` | RED | the host sizing gate against the UNFIXED rule |
| `02-red-gpu-ring-gates.txt` | RED | the full GPU indexer battery on the rig 5090: 10 pre-existing gates green (the drain-loop extraction is behaviour preserving), the two ring-shape arms red |
| `03-green-host-sizing-gate.txt` | GREEN | the same host gate on the fix, plus the whole memra-kv suite |
| `04-green-gpu-ring-gates.txt` | GREEN | the same GPU battery on the fix, 12/12 |
| `05-green-mla-forward-gates.txt` | GREEN | the MLA forward battery, which shares `mla_kpool_indices` |

Rig law: the 5090 is CORRECTNESS ONLY under `flock /tmp/memra-5090.lock`. No timing number in
this directory comes from it, and none is quoted anywhere.

## Reading the red arms

`01` fails at the first configured context it tries: the shipped default books 5120 rows and a
monolithic prime of 8192 tokens needs the whole 8192 live at once.

`02` fails two arms:
  * `gpu_kpool_tail_ring_serves_a_whole_monolithic_prime` (new): one call carrying the whole
    prompt over a ring four times shorter, the shape `prime_cache_hyper` actually runs.
  * `gpu_kpool_tail_ring_wraps_and_matches_the_flat_plane` arm 3a (rewritten): a ring shorter
    than the chunk's live window is SERVED after the fix, and was refused before it.

The red arms run against the liveness rule extracted verbatim into `memra_kv::index_ring_take`,
so the refusal is the shipped one. Its message text is already written for the post-fix rule,
which is why the red output reads `0 rows still owed`: under the old body the rule refuses the
whole call rather than a live window, and the numbers it prints are the fix-era ones. The
refusal itself, and which calls it refuses, are exactly today's.

## Re-running

```
cargo test -p memra-kv the_derived_ring_serves_a_monolithic_prime
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
  cargo test -p memra-engine --test glm5_kpool_indexer_gpu -- --ignored --test-threads=1
```


## The fix, in one paragraph

The true upper bound on a monolithic prime's per-call `t` is the ADMISSION LIMIT: `prime_cache_hyper`
refuses `cache.pos + t > cache.max_ctx` and nothing smaller stands between a request and one forward
call. A ring sized to bound `t` is therefore a ring of `max_ctx` rows, which is the flat plane, which
frees nothing. So `t` was removed from the requirement rather than the constant raised.
`mla_kpool_indices` DRAINS the ring inside the call: `memra_kv::index_ring_take` returns how many
rows fit before the ring must be drained, the append writes them, the pool-key build over the pools
that completes frees them, and the loop continues. `k_norm` and `gate` are still computed ONCE for
the whole call and walked by source-row offset, so the values written, their order, and the ring
addresses they land on are exactly what a single whole-call append produced. That is why 10 of the
12 GPU gates were already green on the red commit.

The only correctness floor left is ONE POOL (4 rows for glm5_next), enforced by the engine because
the state plan does not carry `pool`. 5120 rows is a working-set choice above that floor, kept
unchanged so every banked memory number stays exact.

Measured evidence of the fix, from `04`:

```
monolithic prime, t=64,  ring 16 rows: usable 64  of configured 64  (4x the ring, 4 wraps inside ONE call)
monolithic prime, t=256, ring 16 rows: usable 256 of configured 256 (16x the ring, 16 wraps inside ONE call)
ring 16 vs flat 64: 0/64 queries differ
ring 16 vs flat 64: 0/16 pool keys differ in BITS
```

and from `03`, at `MEMRA_CTX` 8192 / 262144 / 1048576: the derived ring is 5120 rows at all three,
5 MiB per MLA layer at every one of them, and a single call carrying the whole configured context
is admitted.

## The owed serving probe: PAID (box-ctxprobe/, 2026-08-29)

lane/glm53-box's three-arm `ctxprobe` re-ran on serving hardware against a binary built at the
consolidated head f929dda914 (this fix an ancestor). Ring ON serves the full configured window
(7300 tokens at MEMRA_CTX=8192, zero failures); the unfixed binary re-run the same session with
the prefix cache pinned off reproduces the 4630 cap and the `tail ring lapped: 5120` refusal; a
262144-context wall ladder produced zero OOM at any rung. Full verdicts, timing rows, and the
rollback-seam perf caveat: `box-ctxprobe/BOXPROBE.md`.

Adjacent, pre-existing, NOT fixed here and named so nobody reads the fix as broader than it is: a
monolithic prime allocates a `t * n_pools` score plane, which is quadratic in context and is what
the box lane saw as `CUDA_ERROR_OUT_OF_MEMORY` at ~50k tokens (FINDINGS section 5, "ALSO OPEN").
The ring has nothing to do with it and the drain does not address it. Separately, nothing enforces
the admission limit before prefill on the serve side. Both are other lanes' work.
