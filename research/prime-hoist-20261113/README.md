# prime-hoist-20261113: the prime's per-layer transients move into the resident slab and one scan workspace (memra#1258, part 2)

Verdict: open after review round 1. The chunked GDN scan's planes are one workspace per prime,
kept mapped only while the device holds admission's driver-headroom floor: between two calls of a
prime, between primes, and later when the device gets short. The fold that asks the floor at
those points is measured on two cells so far; the rest are owed and listed under "Review round 1".

- **RTX 5090 Laptop, ~5.7k tokens (before the fold):** main's prime fails inside its first call on
  one prompt at margins of 49 and 100 MiB with 1.3 GB unused inside the pool. With the change no
  row fails, the zero-margin row on the boot's own charge included, and the pool grows 704-736 MiB
  per prime against main's 800-992. These rows have not been repeated on the fold.
- **RTX 5090 Laptop, ~8.1k tokens:** unplaced, the per-call draw is lower and the same in both
  boots (second call 1,103-1,105 MiB against 1,616-2,255). Under a 100-150 MiB margin the change
  before the fold moved which prompts fail and was worse on one of them: prompt 8,173 as the first
  placed row of a boot ran out of memory in the prime's second call in 3 fresh boots of 4, where
  main ran it clean in 6 of 6. It held the workspace's 481 MiB across the gap between the prime's
  two calls. The fold drops the workspace there under the floor: 2 fresh boots of 2 clean, with
  main's pool readings. Main's own failures at this length (prompts 8,234 and 8,215, memra#1322)
  passed before the fold and are not run on the fold.
- **GB10:** the first call's draw is 1,201-1,204 MiB against main's 1,474-1,477 at 5.7k, the same in
  every boot on both units before the fold, and 1,201-1,204 against 1,506-1,509 in the fold's one
  boot per arm on main 0db086f87a. Under a full page cache the workspace still parks.
- **Outputs:** identical in every exactness run (before the fold), and the served answer texts are
  equal to main's in the 8.1k and GB10 cells, the fold's GB10 boot included. Exactness on the
  fold's own binary is owed.
- **Parking on a short device was measured and rejected:** it moved a failure to the step after the
  prime at zero margin. Holding the workspace between a prime's two calls on a short device was
  rejected the same way, by the 8,173 rows.
- No timing claim in either direction.

## Question

Part 1 (#1274, `research/prime-tail-cap-20261113/`) stopped tail folds from widening a prime call
past the cap. A second defect remained: the first prime call's draw on the async pool depends on the
pool's state. On main it is bimodal by one 192 MiB GDN `ssnap16` block, and under memory pressure
the call can fail while the bytes it needs are free inside the pool, but not in one place (the
margin sweep's failure at 5,774 tokens). A 4,096-row prime call on Qwen3.8-27B allocated every
per-layer transient fresh and freed it inside the layer, 2,442 allocations per call.

Does moving the prime's per-layer transients into resident memory make the draw smaller and the same
for every prime, without costing capacity on a 24 GB card?

## Change

Five code commits: step 1, step 2, the park floor, and the review fold in two (the fold, and one
line that opens the numeric scope in `prime_abandoned`). Each cell below names the main commit
its builds stand on.

- **Step 1**:
  - the FFN gate/up/down GEMMs write into the prime slab's planes through the W4A8 MMQ `_into` form;
  - the GDN f32 `w` plane is skipped where its bf16 twin is used;
  - `x_in` is dropped after its copy;
  - full-attention primes reserve their dequantized-K workspace at `seq_end`.
- **Step 2**: the chunked GDN scan's planes (`GdnScanWs`: gcum, a, p, u, wb16, kb16 and the
  y16/ssnap16 pair) become one workspace held for a prime.
  - A call of at least half the cap rows takes it, held at `max(cap, rows)` rows.
  - `prime_layers` lends it to the engine for the walk.
  - At the prime's end it is parked on the engine, still mapped, and the next prime takes the same
    bytes back.
  - Accounting:
    - parked bytes count as cached (`pool_cached_bytes`);
    - `pool_trim_to` frees the parked workspace only when the cached blocks fall short of the target;
    - admission charges only the workspace's rows past the call's;
    - the boot calibration leaves the parked bytes out of its measured peak.
- **The park floor**: the workspace stays mapped only while the device's free reading holds a
  floor: the one admission keeps free in front of every request
  (`MEMRA_ADMIT_DRIVER_HEADROOM_MB`, 2,048 MiB).
  - The reading is admission's own: `cuMemGetInfo` on a discrete card, the larger of that and
    `MemAvailable` on unified memory.
  - Under the floor the workspace is dropped at the prime's end. Its bytes are cached blocks again
    and the next prime allocates a new one (`[prime-ws] ... not parked`).
- **Review round 1** (f906e325ce) asks the floor at every point the workspace could stay mapped,
  and releases it on every way out of a prime:
  - **Between two calls of one prime.** A prime of more than one call yields between its calls and
    other sessions run. Under the floor the workspace is dropped at the end of a call that has
    more to come, and the next call allocates again (`[prime-ws] ... not held between calls`).
    Held, its bytes are `used`: no trim frees them and the sessions in the gap cannot take them.
  - **After the prime.** A workspace parked on a roomy device is dropped in front of the next
    admitted request once the device no longer holds the floor (`[prime-ws] ... unparked`,
    `[admit-trim] ... parked prime workspace released`). The drop runs at the top of the
    driver-headroom rung, before the rung's own off switch. The rung's target was left alone: a
    target that counted parked bytes would free the workspace only after the other cached blocks
    ran out, while the drop makes the bytes the cached blocks the gate already counted them as.
  - **Every way out.** A walk that fails, a chunk that fails (whoever called it: the MTP prime
    walker calls `prime_chunk` directly), and a request that is retired between two calls of its
    prime all release the workspace. The release walks every engine the model owns.
  - **The floor is the park's own.** `MEMRA_ADMIT_DRIVER_HEADROOM_MB=0` switches the rung off and
    no longer means "always park": the park keeps a 2,048 MiB default. An engine binary, which
    has no admission, runs on that default too: on a device with less than 2,048 MiB free it
    allocates the workspace per call and keeps nothing mapped between calls or primes.
  - **Sizing.** `fits` and the slab's reuse decision are pure functions of the dimensions, with
    table tests: a scan one row wider than the planes does not take them, and a wider call gets a
    new workspace.

Placement only: the same kernels run on the same operands.

Variants measured on the way and not kept:

- **Resident in the prime slab from boot** (`ds2`): about 480 MB that admission counted as used
  (rest-mapped 19,168 MB against 18,688 on main). The 5090 margin sweeps then could not place their
  margins: at 5.7k they saw at most 56-60 MiB for targets of 75-150, and at 8k 12 of 14 rows were
  served plain where main served 12 of 14 on spec. Rejected: no permanent 480 MB cost.
  Raw: `rtx5090/sweep/main-{warm-1,warm-2,trim-1,trim-8k}` and `s2-{warm,trim,warm-8k,trim-8k}`
  (builds on part 1's head; `s2-warm-VOID-acoff-100323` is void: the laptop left AC power as it
  started and no row finished).
- **Freed to the pool at every prime's end** (`ds2r`): other allocations took its blocks and the next
  prime's draw depended on what they left. On GB10 the first call drew 1,571 MiB in one boot and
  1,603 in another, both above main's 1,474-1,477.
- **Parked, before two fixes** (`ds2q` on GB10):
  - a trim before every prime freed the parked workspace each time, so the pool grew 160-224 MiB
    per prime;
  - the boot calibration counted it in its peak, so the reserve rose from 1,576 to 1,800 MB.
  - Both are fixed in step 2 (`ds2qf`; on the 5090 the `ds2q` directories hold this fixed build).
- **Parked after every prime, whatever the device's free memory** (`ds2q` on the 5090): admission
  counts parked bytes as cached, and only a prime or a trim can use them. After a prime at zero
  margin on the 24 GB card the next step of the request ran out of memory with 30 MB of driver free
  and 480 MB parked. The park floor replaces it.

## First measurements (RTX 5090 Laptop, Qwen3.8-27B NVFP4 + Q5_K, DFlash2; 12 cold primes per boot beside a keeper)

"need" = pool reserved after the call minus pool used before it, MiB, per prime call. Builds on
part 1's head; step 2 here is its first form (workspace resident).

| arm | ~5.7k tokens, call 1 | call 2 | ~8.1k tokens, call 1 | call 2 |
|---|---|---|---|---|
| main (diag6, 2 boots) | 1,599-1,603, 1,794 once per boot at different prompts | equal to call 1 | 1,593-1,810 | up to 2,223 (growth up to 608) |
| main, drain before every call (diag6 mode 2) | 1,599-1,603, 1,794 | 642-768 | 1,593-1,778 | 1,584-1,807 |
| step 1 (diag7, 2 boots) | bimodal 1,409-1,442 / 1,600-1,603, 1,759 once | equal to call 1 | not run | |
| step 2, resident (diag9, 2 boots) | 1,055-1,089, identical per prompt across boots | 1,080-1,090, growth 0 | 1,048-1,167, identical across boots | 1,071-1,168, growth 0 or 96 |

Step 1 alone does not make the draw the same from prime to prime. Step 2 does.

Raw: `rtx5090/spread/diag6-043933`, `diag6-062620`, `diag7-050013`, `diag9-064957`, `diag9-071423`.

## Capacity A/B on the RTX 5090 Laptop (24 GB, sm_120a)

Diagnostic builds on main 202e1411fe (a worker patch adds the `[diag1255]` lines the harness
reads, `tools/diag-worker.py`), Qwen3.8-27B NVFP4 + Q5_K with DFlash2, balanced profile. These
cells read memory, not time.

- `dmain`: main;
- `ds2`: step 2 with the workspace resident in the slab;
- `ds2q`: steps 1+2 with the workspace parked after every prime (the directories hold the fixed
  build, `ds2qf`);
- `ds2h`: `ds2q` plus the park floor (the change as it stood before the review fold).

### Per-call demand at ~5.7k tokens (12 cold primes per boot beside a decoding keeper, 2 boots)

"need" = pool reserved after the prime's first call minus pool used before it. "growth" = how far
the call raised the pool's reserve. MiB. On this card the driver-headroom rung trims the pool before
every prime (11-12 trims per boot in every arm), so each prime starts from about 800 MiB of unused
pool (320 for `ds2`, whose workspace is counted as used).

| arm | need from the 3rd prime | growth after the 1st prime | the two boots | required = cost + reserve (MB) | rest-mapped (MB) |
|---|---|---|---|---|---|
| dmain | 1,600-1,603; 1,794 at prompt 5,756 | 800; 992 at that prompt | identical | 2,353 = 776 + 1,576 | 18,688 |
| ds2 | 1,057-1,087 | 736-768 | identical | 2,353 = 776 + 1,576 | 19,168 |
| ds2q | 1,537-1,600 | 736-768; 832 at one prime of boot 2 | same need; growth differs at that prime | 2,320 = 776 + 1,543 | 18,720 |
| ds2h | 1,538-1,571 | 704-736 | identical but one prime (1,538 / 1,570) | 2,320 = 776 + 1,543 | 18,720 |

- `ds2q`'s and `ds2h`'s need includes the workspace (481 MiB at 4,096 rows): `ds2q` takes the parked
  one back, `ds2h` allocates one. Main's includes the per-layer planes the workspace replaces.
- The 192 MiB step in main's draw (1,794 against 1,600-1,603) is at the same prompt in both boots
  here. In the first measurements above it moved between prompts from boot to boot. No arm with the
  workspace has it.
- `ds2h` did not park after any of the 12 primes of either boot: the device's free reading at the
  prime's end was 1,260-1,292 MB, under the 2,048 MiB floor.

Raw: `rtx5090/spread/dmain-192837`, `ds2-204809`, `ds2q-221812`, `ds2h-031817`.

### Margin sweeps at ~5.7k tokens

One cold DFlash prime at a time beside a decoding keeper. A ballast process leaves the gate's
`required` plus a target margin of effective free memory (driver free plus the pool's cached
bytes); "seen" is the margin the gate's own line reports. "up" = targets 0, 25, ... 150, two reps.
"down" = 150 ... 0, one rep, so that the first rows run on the boot's own charge: after a request
is served plain the gate's fixed residual rises and `required` with it, by about 300 MB in every
arm, and later rows of that boot no longer test the first charge. "drained" = up, with the pool's
cached bytes dropped before each prime; it reads as the plain sweep in every arm, because the rung
trims before every prime here anyway.

| arm | sweep | rows | spec / plain first pass | out-of-memory lines | where |
|---|---|---|---|---|---|
| dmain | up | 14 | 13 / 1 | 1 | inside the prime's first call: prompt 5,774, seen 49 (2,400 MB effective free) |
| dmain | up, drained | 14 | 13 / 1 | 1 | the same row |
| dmain | down | 7 | 7 / 0 | 1 | inside the prime's first call: prompt 5,774, seen 100 (2,451 MB) |
| ds2 | up | 14 | 12 / 2 | 0 | targets of 75-150 seen as 56-60: the margins could not be placed |
| ds2 | up, drained | 14 | 12 / 2 | 0 | the same |
| ds2q | up | 14 | 13 / 1 | 1 | a step after the prime: prompt 5,733, seen 0 (2,317 MB) |
| ds2q | up, drained | 14 | 13 / 1 | 1 | the same row |
| ds2q | down | 7 | 7 / 0 | 0 | seen 150, 122, 100, 76, 49, 49, 50 |
| ds2h | up | 14 | 13 / 1 | 0 | first rep on the boot's own charge: seen 0, 22, 48, 78, 97, 123, 150 |
| ds2h | down | 7 | 6 / 1 | 0 | seen 150, 122, 100, 80, 49, 25; the last row served plain |

The out-of-memory lines, quoted from the serve logs (each request was parked, replayed plain and
answered with status 200):

- `dmain`, up, prompt 5,774: `[admit-oom] step OOM parked session back to queue (model gate, retry
  1/3, 2 active) [dev0 free 17MB pool-res 22208MB pool-used 20879MB]:
  DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, after the prime's first
  `dspark-prime-chunk-before` line and before any `chunk-after`. 1,329 MB sat unused inside the
  pool. In the down sweep the same prompt failed at the same point with 100 MiB of margin (`dev0 free
  36MB pool-res 22240MB pool-used 20879MB`). Main ran its other prompts clean at seen margins of 1,
  20, 121 and 150 on the first charge.
- `ds2q`, up, prompt 5,733, seen 0: the prime completed (`dspark-prime-done`: driver free 94 MB, pool
  reserved 22,048 MB), and a later step of the request failed: `[dev0 free 30MB pool-res 22112MB
  pool-used 21083MB]`. Main ran this prompt clean at a seen margin of 1.
- `ds2h`: none in any cell.

What the rows say:

- **Main's failure is the one this change set out to remove.** The prime's first call fails on one
  prompt at margins of 49 and 100 MiB with more than 1.3 GB unused inside the pool. `ds2q` ran that
  prompt clean at 2,419 MB effective free in its down sweep, and `ds2h` ran it clean at seen 48 on the
  first charge (2,367 MB) and at seen 100.
- **Parking on a short device moves a failure to the step after the prime.** Admission counts the
  parked workspace as cached. Only a prime or a trim can use those 480 MB. After a prime at zero
  margin the driver had 94 MB free, and the request's next step could not take its blocks from the
  parked bytes. This is why the park has a floor. `ds2q`'s down sweep did not test it again: its two
  lowest targets ran at 49-50 MiB, because the ballast could not add its last 25 and 51 MB
  (`BALLAST FAILED` in `ballast.log`).
- **With the park floor every row is clean.** The row at seen 0 on the first charge included. The
  plain rows of `ds2h` are admission's own choice, not failures: the first row of the second rep in
  every arm, and `ds2h`'s last down row, asked for a draft plane that did not fit by 0-3 MiB
  (`[admission] drafter refused: ... draft plane 169MB does not fit (available 2429MB, required
  2431MB with it, 2262MB without ...)`), and admission priced the request plain.
- In the sweeps `ds2h` parked after 3 of 16 and 2 of 8 primes, the ones that ended with the floor
  free. At the others the free reading at the prime's end was 95-563 MB (1,319 MB once).

Raw: `rtx5090/sweep/cap-{dmain,ds2,ds2q}-{warm,trim}-5.7k`, `cap-{dmain,ds2q,ds2h}-desc-5.7k`,
`cap-ds2h-warm-5.7k` (`warm` = up, `trim` = up drained, `desc` = down). Each holds `sweep.log`,
`rows.jsonl`, `serve.log`, `ballast.log` and `guard.log`. `tools/cap-summary.py 5090` prints the
table's inputs.

### ~8.1k tokens: per-call demand (10 cold primes per boot, 2 boots)

At both lengths the prime runs as two calls (4,096 rows, then the rest). At 5.7k the second call
is short; at 8.1k it is about as wide as the first, over twice the context.

| arm | call 1 need from the 3rd prime | growth after the 1st prime | call 2 need | call 2 growth | the two boots |
|---|---|---|---|---|---|
| dmain | 1,586-1,617; 1,778 at prompt 8,097 in boot 2 only | 832-864; 1,024 at that prime | 1,616-1,871 (boot 1), 1,617-2,255 (boot 2) | 0-96; 256 once in boot 1; 224 and 640 in boot 2 | differ |
| ds2h | 1,551-1,554 | 800 | 1,103-1,105; 1,198 once | 0; 96 once | identical, every prime |

- The answer texts are the same in both arms for all ten prompts, in both boots (`text_sha256` per
  row in `rows.jsonl`, the 16 greedy tokens at temperature 0).
- No out-of-memory line in either arm. Each boot trims 9 times in both arms; `ds2h` did not park
  after 10 of its 11 primes per boot.

Raw: `rtx5090/spread/dmain-035913`, `ds2h-053441`.

### ~8.1k tokens: margins

| arm | cell | rows | out-of-memory lines | where |
|---|---|---|---|---|
| dmain | up, 1 rep | 7 | 0 | row 1 served plain by admission, then `required` 2,465 -> 2,777 |
| dmain | down | 7 | 1 | the prime's second call: 3rd row, prompt 8,215, seen 101 (2,569 MB) |
| dmain | one row per fresh boot, 100 MiB, 7 prompts | 7 | 1 | the prime's second call: prompt 8,234, seen 96 |
| ds2h | up, 1 rep | 7 | 0 | row 1 served plain by admission, then `required` 2,432 -> 2,744 |
| ds2h | down | 7 | 1 | the prime's second call: 1st row, prompt 8,173, seen 148 (2,581 MB) |
| ds2h | one row per fresh boot, 100 MiB, 7 prompts | 7 | 1 | the prime's second call: prompt 8,173, seen 98 |

- The up sweeps say nothing about the boot's own charge at this length: in both arms the first row
  asks for a draft plane that does not fit by 0-3 MiB, admission serves it plain, and `required`
  rises by about 310 MB for the rest of the boot. The down sweeps stop testing it at their first
  failure for the same reason. The fresh-boot rows (`tools/b-fresh.sh`) boot the server for every
  row, so each of the seven runs on a boot's own charge.
- **At 8.1k both arms fail in the prime's second call, on different prompts.** On a boot's own
  charge main fails 2 of 10 rows and `ds2h` 2 of 8, every time inside the prime's second call with
  free bytes inside the pool, for example `[dev0 free 31MB pool-res 22368MB pool-used 20962MB]`
  (main) and `[dev0 free 29MB pool-res 22336MB pool-used 20959MB]` (`ds2h`). Before the second
  call both arms have about 220-255 MB of driver free; a passing second call grows the pool by
  0-128 MB.
  - Main failed on prompts 8,234 (seen 96) and 8,215 (seen 101); 8,215 ran clean in its fresh boot.
    `ds2h` ran both clean.
  - `ds2h` failed on prompt 8,173 both times it met it as the first placed row of a boot (seen 98
    and 148); main ran that prompt clean at 99 and 149.
- **So `ds2h` moved which prompts fail, and for prompt 8,173 it is worse than main.** The count of
  failing rows is the same in both arms, but the rows are not: `ds2h`'s two failures are one
  scenario that main passes. The review of this change read that from these rows (this record
  first said the change neither removed nor added to the failure, which the rows do not support).
  "Review round 1" below reproduces the scenario in fresh boots, traces it to the workspace held
  across the gap between the prime's two calls, and measures the fold that drops it there.
- In the rows that pass, the pool's reserve after the prime is 22,112-22,272 MB for `ds2h` (five
  of six at or under 22,208) and 22,176-22,208 MB for main, with 34 MB less effective free placed
  for `ds2h` (its `required` is 33 MB lower).

Raw: `rtx5090/sweep/cap-{dmain,ds2h}-{warm,desc}-8k`, `cap-{dmain,ds2h}-fresh-8k-m100/s401..s407`.
`tools/fresh-table.py 8k 100 dmain ds2h` prints one line per fresh-boot row.

## GB10 (DGX Spark, sm_121a, Qwen3.8-27B NVFP4 + Q5_K, DFlash2; 2 boots per cell, no sweeps: a ballast would starve the shared host)

"need" = pool reserved after the first prime call minus pool used before it, MiB, from the 3rd prime
on. "required" = the first gate pass's cost plus reserve. `unit-a` and `unit-b` are the two units.

| arm | 5.7k need (boot 1 / boot 2) | 5.7k required | 8k need (boot 1 / boot 2) | 8k required | rest-mapped |
|---|---|---|---|---|---|
| dmain | 1,474-1,477, identical | 3,729 MB | 1,457-1,459, identical | 3,841 MB | 18,688 |
| ds2 | 1,217-1,220, identical | 4,241 MB | 1,200-1,203 / 1,041-1,104 | 4,354 MB | 19,136 |
| ds2r | 1,603 / 1,571 | 4,209 MB | not run | | 18,688 |
| ds2q | 1,201-1,204 / 1,346-1,349 | 3,953 MB | 1,216 / up to 1,651 | 4,065 MB | 18,720 |
| ds2qf | 1,201-1,204 / 1,346-1,349 | 3,712 MB | 1,216-1,218 then 1,504-1,507, identical | 3,825 MB | 18,720 |

- ds2qf's required is within 17 MB of main's at both lengths: cost 2,152 / 2,265 MB as main, and the
  reserve is 1,559 MB against main's 1,576.
- The boot-to-boot differences in these cells (5.7k ds2q and ds2qf, 8k ds2) follow admission's
  driver-headroom rung. On GB10 the rung compared the raw `cuMemGetInfo` reading with its 2,048 MiB
  floor before every prime (memra#1257, `research/admit-headroom-unified-20261113/`). A boot that
  settled under the floor trimmed 160-235 MiB of cached blocks before every prime and the next call
  mapped them back; a boot that settled above it trimmed nothing.
  - Plain step 2 at 8k: driver free at the first prime was 4,602 MB in boot 1 and 3,678 MB in boot
    2. Boot 1 settled at about 2,470 MB: no trims, draw 1,200-1,203 from 1,926 MB cached. Boot 2
    settled at about 2,040 MB: 9 trims, draw 1,041-1,104, growth 32-64 per prime.
  - Main at 8k sat at 2,030-2,042 MB in both boots, trimmed in both (10 and 10) and read the same.
    Main at 5.7k ran when the host had 8.4 GB free (no trims). A later main run at 8k with 14 GB free
    (no trims) read 1,905-1,907, identical across its boots: main's own draw moves with the rung too.

With #1257's fix in both arms (the rung reads the larger of the raw reading and `MemAvailable`;
`dmainh` = main + the fix, `ds2qfh` = ds2qf + the fix):

| cell | arm | trims (boot 1 / 2) | need from the 3rd prime (boot 1 / boot 2) | growth inside a prime after the 2nd |
|---|---|---|---|---|
| unit-b, host 10.6 GB driver free | dmainh 5.7k | 0 / 0 | 1,474-1,477, identical | 0 |
| | ds2qfh 5.7k | 0 / 0 | 1,201-1,204, identical | 0 |
| | dmainh 8k | 0 / 0 | 1,905-1,907, identical | 0 |
| | ds2qfh 8k | 0 / 0 | 1,216-1,218 then 1,504-1,507, identical | 0 |
| unit-a, page cache filled before each boot (MemFree after the fill 1.3-11 GB) | ds2qfh 5.7k | 0 / 0 | 1,201-1,204, identical | 0 |
| | ds2qf 5.7k (no fix) | 13 / 12 | 1,346-1,349 / 1,346-1,412 | 768-1,088 |
| | ds2qfh 8k | 0 / 0 | 1,216-1,218 for primes 3-6 in both; then 1,280-1,283 / 1,504-1,507 | 0 |
| | ds2qf 8k (no fix) | 11 / 11 | 1,329-1,363 | 0-64 / 448-768 |

- With the rung fix: 0 trims and growth 0 inside the prime, in every boot of both arms. The
  variant's draw at 5.7k was 1,201-1,204 in all four boots on both units, against main's 1,474-1,477.
- The pool growth that is left happens between primes, and it tracks the host's page cache. At 8k
  both units, unfilled, step from 1,216 to 1,504 at the 7th prime; in the filled cell boot 1 stepped
  to 1,280 at the 8th instead. In each case pool used before the call was unchanged (20,073 MiB) and
  the pool's reserve had grown between the 6th and 7th primes, while the keeper decoded or the
  previous prompt's prefix was published: 288 MB unfilled and in filled boot 2 (raw driver free 1.3
  GB), 64 MB in filled boot 1 (raw driver free 6.3 GB). These cells do not show which of the two
  grows it.
- Without the fix, under the same filled page cache, the variant trimmed 11-13 times a boot and its
  pool grew 448-1,088 MB per prime in three of the four boots.

Raw: `gb10/spread-*` (one directory per arm, length and run; `-fill-` = page cache filled).

### The park floor on GB10 (filled page cache, ~5.7k tokens, 2 boots per cell, both units)

`ds2hh` = `ds2qfh` plus the park floor. The page cache is filled before each boot, so the raw
`cuMemGetInfo` reading sits near admission's 2,048 MiB floor while `MemAvailable` is about 120 GB.
The park reads the larger of the two, as the rung does since memra#1257; a park on the raw reading
would stop parking in exactly this state (the red arm `red-raw` under CPU).

| unit | arm | MemFree after the fill (boot 1 / boot 2) | trims | `[prime-ws]` not-parked lines | need from the 2nd prime (boot 1 / boot 2) | growth inside a prime | texts against dmainh |
|---|---|---|---|---|---|---|---|
| unit-a | dmainh | 13.2 GB / 2.9 GB | 0 / 0 | n/a | 1,474-1,477 / 1,475, then 1,506-1,509 from the 3rd prime | 0 | |
| unit-a | ds2hh | 1.5 GB / 1.7 GB | 0 / 0 | 0 / 0 | 1,201-1,204, identical | 0 | equal, 12 of 12 in both boots |
| unit-b | dmainh | 2.9 GB / 2.4 GB | 0 / 0 | n/a | 1,474-1,477, identical | 0 | |
| unit-b | ds2hh | 1.7 GB / 2.7 GB | 0 / 0 | 0 / 0 | 1,201-1,204, identical | 0 | equal, 12 of 12 in both boots |

- The workspace parks in every boot (no `[prime-ws]` line, `parked-gdn-ws 240MB` on the calibration
  line), with 0 trims and growth 0 inside the prime, and the draw is the always-parked arm's.
- unit-a's first `dmainh` boot ran before the fill reached its target (13.2 GB free when the 300 s
  cap ended); the other seven boots ran with 1.5-2.9 GB free.
- unit-a's second `dmainh` boot shows the between-primes growth again: +32 MiB between the 2nd and
  3rd primes, none inside a prime.

Raw: `gb10/spread-dmainh-5.7k-fill-054207`, `spread-ds2hh-5.7k-fill-061700` (unit-a),
`spread-dmainh-5.7k-fill-060453`, `spread-ds2hh-5.7k-fill-063245` (unit-b).

## Review round 1: the workspace under the floor (2026-11-14)

The review of this change found that the 8.1k rows above do not say what the record first read
from them. Both failures of the change were one scenario, prompt 8,173 as the first placed row of
a boot, which main passes; main's own failures, prompts 8,234 and 8,215, passed with the change.
So the change moved which prompts fail, and for 8,173 it was worse. The cell below reproduces
that on main 0db086f87a, traces it, and measures the fold. It is not complete: the cards were
taken for other work after the rows below, and the cells still owed are listed at the end of this
section.

Arms, all diagnostic builds on main 0db086f87a (`tools/mk-arms2.sh`: the `[diag1255]` lines and an
allocation ring, `tools/diag-ring.py`): `dmain2` = main, `ds2h2` = this change before the fold
(steps 1 and 2 and the park floor), `ds2i` = the fold. The cell is the one above: a cold DFlash
prime beside a decoding keeper, placed by a ballast at the gate's `required` plus a margin, with
the server booted fresh for every row (`tools/b-first.sh`).

### What happened to prompt 8,173

At about 8.1k tokens the prime runs as two calls and yields between them; in the gap the keeper
runs two bursts.

Prompt 8,173 as the first placed row of a boot, 100 MiB (RTX 5090 Laptop, 24 GB):

| arm | fresh boots | boots with an out-of-memory line | margin seen | pool reserved / used after call 1 | pool reserved / used before call 2 | pool growth in the gap |
|---|---|---|---|---|---|---|
| main (`dmain2`) | 6 | 0 | 99 MiB | 22,144 / 20,560 MB | 22,144 / 20,560 MB | 0 MB |
| before the fold (`ds2h2`) | 4 | 3 | 98 MiB | 22,080 / 21,041 MB | 22,144 / 21,041 MB | 64 MB |
| the fold (`ds2i`) | 2 | 0 | 98 MiB | 22,080 / 20,560 MB | 22,080 / 20,560 MB | 0 MB |

Every number in a row is the same in all of that arm's boots. The fold's two boots are two boots,
not a rate.

- **Before the fold** the workspace stayed in the slab across the gap. After the first call the
  pool's used bytes are 21,041 MB against main's 20,560: the workspace's 481 MiB, which no trim
  frees and no other session can take. The keeper's bursts allocate the same blocks in every arm
  (the last 256 allocations of the gap sum to 149 MB, 105 MB of it in 21 MB blocks of the
  speculative step); with the workspace held they no longer fit the cached blocks, and the pool
  grows 64 MB from the driver. The second call then starts with that much less driver memory
  and, in three boots of four, fails with 1.4 GB unused inside the pool:
  `[admit-oom] step OOM parked session back to queue (model gate, retry 1/3, 2 active) [dev0 free
  29MB pool-res 22336MB pool-used 20959MB]: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`.
  The request is replayed and answers with status 200 in all four boots.
- **With the fold** the workspace is dropped at the end of the first call, because the device is
  under the floor (`[prime-ws] dev0 GDN scan workspace not held between calls:
  free_bytes=299827200 floor_bytes=2147483648 workspace_bytes=504102912`). The pool's used bytes
  after the first call are main's, the gap grows nothing, and the second call allocates the
  workspace again from the cached blocks. The same boots show the other two mechanisms: the
  workspace that the boot's unplaced prime had parked is released in front of the placed prime's
  admission (`[prime-ws] ... unparked: free_bytes=1004470272`, `[admit-trim] dev0 parked prime
  workspace released: bytes=504102912 before prime`), and at the prime's end it is not parked
  (`free_bytes=266272768`).

So on this scenario the change before the fold was worse than main, by holding 481 MiB across
the gap between its two calls, and the fold's two boots read as main's. Main's own 8.1k failures
(prompts 8,234 and 8,215, memra#1322) are a different matter: they passed on the head before the
fold in the rows above, and they have not been run on the fold.

Raw: `rtx5090/sweep/cap-{dmain2,ds2h2,ds2i}-first-8k-m100/s401-r*` (one directory per boot;
`tools/first-table.py DIR...` prints one line per boot with the pool around the prime's two calls).

### GB10 on the fold (unit-b, filled page cache, ~5.7k tokens, one boot per arm)

The filled-cache cell of the section above, one boot per lease (`tools/gb10-cap-run-fill3.sh`):
the page cache is filled until MemFree is about 2 GB, then one boot runs 12 cold primes beside the
keeper. Builds of main 0db086f87a and of the fold for `sm_121a`.

| arm | MemFree at the boot | need, first call, primes 1-2 | need from the 3rd prime | growth from the 4th prime | trims | out-of-memory lines |
|---|---|---|---|---|---|---|
| main (`dmain2`) | 1,914 MB | 1,473, 1,475 | 1,506-1,509 | 0 | 0 | 0 |
| the fold (`ds2i`) | 2,073 MB | 1,281, 1,202 | 1,201-1,204 | 0 | 0 | 0 |

- The 12 answer texts are equal row by row between the arms (`text_sha256` in `rows.jsonl`).
- The fold's `serve.log` has no `[prime-ws]` line: under a full page cache the device still holds
  the floor on the reading admission uses, so the workspace parks after every prime, as before the
  fold.
- One boot per arm. The second boot of each arm was not run.

Raw: `gb10/spread-{dmain2,ds2i}-5.7k-fill3/`.

### Cells of the fold that are still owed

Not run when this record was written. None of the statements above rests on them.

| cell | arms | device | what it settles |
|---|---|---|---|
| prompt 8,173 as the first placed row, 100 MiB: boots 3-6 of the fold, 5-6 before the fold | `ds2i`, `ds2h2` | RTX 5090 Laptop | the fold's row above at main's N |
| the same row at 150 MiB (the change failed there at 148, main passed at 149) | `dmain2`, `ds2i` | RTX 5090 Laptop | the review's second failing margin |
| the seven 8.1k prompts, one per fresh boot, 100 MiB | `dmain2`, `ds2i` | RTX 5090 Laptop | main's own failures (8,234, 8,215) on the fold |
| a short cold prime placed after an unplaced 5.7k prime (the workspace parks, then the device gets short) | `dmain2`, `ds2h2`, `ds2i` | RTX 5090 Laptop | the later reclaim (`unparked`) under a placed margin |
| margin sweeps and per-call demand at ~5.7k tokens | `dmain2`, `ds2i` | RTX 5090 Laptop | that the 5.7k rows of this record, measured before the fold, hold on it |
| `run-gen`, `run-spec` K=1..8 and `dspark_q38_gate`, main against the fold | | RTX 5090 Laptop | exactness on the fold's own binary |
| filled page cache, ~5.7k tokens, boot 2 of each arm | `dmain2`, `ds2i` | GB10 | the GB10 row above at two boots |

## Exactness

Each pair of binaries is built from one tree with only the engine files changed, and every
compared byte is in the receipts.

- **RTX 5090, steps 1+2 against main df1d23a764** (`rtx5090/exact/m6-s2f-020029`, balanced profile;
  arm `m6` = main, `s2f` = the change):
  - `run-gen` at 4,106 and 8,200 tokens: `MATCH` / `MATCH` on both arms, the same argmax (16602),
    the same maxdiffs (1.727e-1 / 1.851e-1 and 2.147e-1 / 1.759e-1) and the same 32 tokens.
  - `run-spec` K=1..8 at 8,200: identical `self-consistency: PASS` lines, and `STEP-IDENTITY ... 0
    of 1986560 logits differ` on both.
  - `dspark_q38_gate` at 2,000 / 4,106 / 8,200 / 16,500: `ALL EXACT` on both, with identical
    `spec_sha256` and `plain_sha256` per prompt.
  - `ARMS-EQUAL` (`guard.log`).
- **RTX 5090, step 2's first form against part 1's head** (`rtx5090/exact/1258-s2-072825`): the
  same three checks, `ARMS-EQUAL`.
- **GB10, main against step 2's first form** (`gb10/exact/exact-main`, `exact-s2`): `run-gen` at
  4,106 `MATCH` / `MATCH` with the same argmax and maxdiffs (3.186e-1 / 3.203e-1), `run-spec`
  `STEP-IDENTITY` 0 of 1,986,560 logits differ on both, `dspark_q38_gate` `ALL EXACT` on both at
  2,000 / 4,106 / 8,200: `ARMS-EQUAL`.
- **The park floor and the review fold** change one thing: whether the workspace is kept or
  dropped, at a prime's end and, since the fold, between two calls of a prime and in front of a
  later admission. The runs above are binaries from before the fold, in which an engine binary
  had no floor and always parked (the gate's four prompts in one process cover a first allocation
  and three parked reuses). A dropped workspace is allocated again by the next call, uninitialised,
  as step 2's first allocation and its allocation after a trim already are. Since the fold an
  engine binary keeps the park's own 2,048 MiB floor, so on a device with less than that free it
  allocates per call. `run-gen`, `run-spec` and `dspark_q38_gate` on the fold's own binary are
  owed (see "Review round 1").
- **Served answers, the path with the floor.** The 16 greedy tokens of every request, by sha256:
  - RTX 5090 at ~8.1k, `ds2h` against `dmain`: equal for all ten prompts in both boots. `ds2h` did
    not park after 10 of its 11 primes per boot, so these are primes on a newly allocated
    workspace.
  - GB10 at ~5.7k, `ds2hh` against `dmainh`: equal for all twelve prompts in both boots on both
    units, with the workspace parked.
  - The 5.7k cells on the 5090 predate the hash in the harness and carry no text.
  - GB10 at ~5.7k on the fold, `ds2i` against `dmain2` (main 0db086f87a): equal for all twelve
    prompts in the one boot per arm, with the workspace parked.
  - RTX 5090 at ~8.1k on the fold: the first-row cell's two boots answer with status 200; the
    sweep harness records no answer text, so there is nothing to compare.

## CPU

On the branch head (main 5a6ae2438e plus the five code commits), CPU only:

- The workspace, census and schedule tests: `only_prime_layers_writes_gemm_outputs_into_slabs`,
  `only_prime_layers_lends_the_gdn_scan_workspace`, `the_admission_row_bytes_are_the_allocated_bytes`,
  `the_gdn_scan_workspace_lives_for_one_prime` (the guards that settle the workspace at the end of
  a walk and of a failed chunk, the release over every owned engine, parked bytes in
  `pool_cached_bytes`, the trim order, the slab bytes, the graph key, and the one floor reading
  behind the park, the gap between two calls and the later reclaim),
  `the_gdn_workspace_charges_only_its_rows_past_the_call`,
  `the_workspace_parks_only_above_the_floor` (with the floor's own default),
  `a_scan_fits_only_within_the_planes`, `a_wider_call_gets_a_new_workspace`, with #1276's schedule
  tests.
- memra-server: `calibration_floor_leaves_out_the_parked_gdn_workspace`,
  `the_worker_hands_the_engine_the_park_floor`,
  `a_short_device_gives_its_parked_workspace_back_before_the_rung`,
  `a_request_retired_mid_prime_hands_the_workspace_back`.
- Red arms (tests kept), 23, each failing its test:
  - accounting: `pool_cached_bytes` without the parked bytes; the workspace charged whole; the
    calibration without the parked subtraction;
  - the floor: the decision always true; a zero from the server taken as the floor; the park
    without the decision; the reading on raw `cuMemGetInfo`; the gap between two calls without the
    decision; the later reclaim never dropping; the worker without the floor call;
  - the ways out: a failed walk not releasing; a call with more to come never asking the floor; a
    failed chunk not releasing; a pipeline chunk never reporting its result; the release reaching
    one engine only; a failed prime not releasing; the rung's entry without the release; the
    release after the rung's off switch; the retire seam without the hook; the retire predicate
    without the queued prompt;
  - sizing: `fits` without its row check; reuse without the row check; the slab reusing without
    the sizing decision.
- Engine library suite 793 passed, server library suite 1,276 passed, clippy (engine and
  server, all targets) and fmt green.
- Before the rebase onto 5a6ae2438e the same checks read 792 and 1,276 on main 0db086f87a, with
  the same 23 red arms.

## Not measured

- **Time.** The timed TTFT cells ran once and were void under the rig's thermal rule (throttled
  takes), and were not repeated: this record makes no timing claim, in either direction. The
  capacity cells run on the balanced profile and read memory only.
- **A pipeline (PP) placement, step 2.** No cell, because none can run: the GDN scan workspace
  exists only for plans with a `GatedDeltaNet` mixer, and pipeline placement refuses those plans at
  load. On an RTX 5090 with `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,0` the load of Qwen3.8-27B stops
  with `pipeline placement is unsupported for plan operations [...]; blockers=[GatedDeltaNet,
  FusedAttentionGate]` (`rtx5090/pp-refusal/`: `concat-prime-probe` built from main 63c351a2f4 and
  from memra#1299's head of that day; the refusal is the loader's, before any prime). The release
  still walks every engine the model owns (`prime_gdn_ws_release`), and a census test pins that.
- **A pipeline (PP) placement, step 1.** No cell. On the Step-3.7 serving route (IQ4_XS GGUF, PP-2
  on two RTX PRO 6000) the `_into` route is not taken:
  - a fresh step35 prime, and a continuation with the suffix door at its default, take the batched
    GEMM entry (`step35_prime_cache_batch`), which has its own layer loop and never calls
    `prime_layers`, the one caller of `matmul_prefill_w4a8_into`;
  - `mmq_w4a8_into_route` needs an NVFP4 weight, and that artifact's weights are GGUF quant types.

  It can be taken on a pipeline only with the NVFP4 Step-3.7 artifact on the walk
  (`MEMRA_STEP_GEMM_PRIME_SUFFIX=0`, `MEMRA_STEP_GEMM_PRIME=0` or a vision overlay), on the trunk's
  dense layers. That was not run. Each stage's walk is single-device code on that stage's engine
  and its own slab.
- **RTX PRO 6000, single card.** No cell. On a 96 GB card the margins these sweeps place are not
  the constraint.
- **Prompts past 8k tokens.** The long-prompt footprint cells (7.6k to 18.7k) were not run.
- **The margin sweeps on GB10.** Their ballast would take the shared host's memory down to the
  admission margin.
- **The rest of the review fold's cells.** Listed with their devices at the end of "Review round
  1": the fold is measured on the first-row cell (two boots) and one GB10 boot per arm.
