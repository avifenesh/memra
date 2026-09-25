# WP-C day 44 (2026-09-24): the MoE slot cache door, improvement I9: no pinned duplicate of the bank under the door

`OWED.md` C1 step (b). A new item on day 40's list, found while designing I6 and the fill: under the door the loader
still copies every expert bank into write-combined pinned host memory (`HostBuf::Pinned`, cudarc's `alloc_pinned`),
about 15 GB for the approved artifact, which the door's decode never reads (every miss is served from the bank's
host tier). That duplicate costs three things measured or read on day 40: the install's record pass (55.8 of 60.7 s
on the RTX 5090, a byte compare over write-combined memory plus the per-record SHA-256), the load's copy, and the
host memory a full-bank host tier needs beside it (15 GB pinned plus 15 GB tier on a 60 GB host). Written before any
I9 code; tree at start: the day-43 code commit `ed3c8c5b4`.

## 1. Pre-registration

**The design.**

- (a) **An engine load option, not an environment variable.** `Engine::set_expert_host_mapped(bool)`, default
  false. `run-gen` and `run-spec` set it true when `--experts-via-tier` is parsed, before the model loads; nothing
  else sets it. With it false every loader path is byte for byte today's.
- (b) **The mapped view.** With it true, the stacked-expert loader (`HostExps::load_stacked_from_source`) takes the
  expert tensor's bytes as `HostBuf::Mmap` over the artifact's own mapping: the GGUF shard's `Mmap` becomes an
  `Arc<Mmap>` (`memra-gguf`, no behavior change for any reader: every slice still derefs the same map) and
  `GgufSource` answers a new `TensorSource::map_expert_extent(name)` with that map, the shard's retained `Arc<File>`
  and the tensor's absolute range. No copy, no pinned allocation. A layer the source cannot map while the option is
  on is an error, never a silent copy (the door's installer already refuses split shards and non-GGUF sources).
- (c) **The installer's compare.** Every retained record is still compared byte for byte with the loaded `HostExps`
  and checksummed; with the view both sides are the same mapped bytes, so the compare reads page-cache memory instead
  of write-combined memory. It is kept, not skipped: it is the proof that the bytes the native path would stage are
  the catalog's.
- (d) Unchanged: the numeric program (the same bytes reach the same kernels; the door never stages from `HostExps`),
  the OFF arm's loader (pinned, as today), every other loader path and default.

**Why this is not a substitute.** The door's decode reads only the bank. The pinned copy exists because the legacy
cache stages from it; under the door nothing does. The mapped view keeps `HostExps` complete for every consumer that
reads it (the installer's compare, any refused path's error text) at no memory cost.

**Acceptance.**

- CPU: a `memra-gguf` test that `map_expert_extent` returns the tensor's absolute range and that slicing the returned
  map at it equals `tensor_data`; an engine census that the loader's mapped branch is taken only under the option.
- The cell `mapped` (RTX 5090 first): day 40's shape, three arms, every door arm with `--expert-bank-stages`: OFF (the
  I9 binary); I6 (the day-43 binary, default budget); I9 (the I9 binary, default budget). Order 1 (OFF, I6, I9) x 5,
  order 2 reversed x 5, one collector hold. Integrity as day 43's (one tape across all 30, `MATCH`, one STEADY-STATE
  line, the host trace consistency) plus: the I9 arm's host RSS at `phase=window` (read from `/proc/self/status`
  `VmRSS`/`VmPin` in the stage line, a log field added with I9) shows no pinned expert slab (`VmPin` below 2 GiB).
- **Clauses.** (i) No regression: `median(I9 window) - median(I6 window) <= noise` pooled and both orders (noise as
  day 43). (ii) The install falls: I9's `install_s` below I6's by more than the larger IQR, both orders. (iii) The
  load does not grow: the `loaded` line's time since process start on I9 not above I6's by more than its IQR.
  I9 stays if (i) holds; (ii) and (iii) are readings with registered directions.

**What each card can decide.** The RTX 5090 decides (i) here and reads (ii) and (iii); the target card reads all
three in the ladder sitting (its pinned copy is cached memory there, so (ii) is expected smaller on that host).

## 1a. Amendment, before any I9 build or cell

- **The memory clause.** Section 1's clause "`VmPin` below 2 GiB" rests on the driver's pinned host allocations
  showing in `/proc/self/status` `VmPin`, which I have not confirmed; if they do not, the clause passes on both arms
  and proves nothing. It is replaced before any build: the installer prints a census of the loaded banks' host
  storage, `[experts-via-tier] host expert storage: mmap=N (B bytes) pinned=P (B bytes) paged=Q (B bytes)`, read off
  each retained projection's `HostBuf` variant, and the clause is **I9 prints `pinned=0` and `paged=0` with every
  projection `mmap`** (the I6 arm runs the day-43 binary, which has no census; its banks are pinned by the loader's
  default, `model.rs` `load_stacked_from_source`, read from source). The stage lines additionally carry `VmRSS`, `RssAnon`, `RssFile`,
  `RssShmem`, `VmPin` and `VmLck` from `/proc/self/status` as recorded context, no clause on them.
- **The mapped view's plumbing.** `TensorSource` already exposes `gguf()`, so the view needs no new trait method:
  the loader asks the source's `GgufFile` for the tensor's range, the shard's shared map (`GgufFile::shard_mmap`,
  the shard's `Mmap` made an `Arc<Mmap>`) and its retained inode. Same effect as section 1 (b), less surface.
- **The load reading's end point.** Section 1's (iii) names "the `loaded` line's time since process start"; run-gen
  prints `loaded ...` after the installer, so that line counts the install as load. The reader takes the load's end
  at the `[q8rp] split-plane decode mirrors built` line (the last line of the load, before the installer), stated
  here before any I9 cell ran.

## 3. Results, cell `mapped` (RTX 5090 Laptop GPU, `rtx5090-day44/mapped/`)

One collector hold, 22:33:31Z to 22:50:10Z, 30 runs, tree `23daf81ff` (scripts only past the I9 code), binaries
`run-gen-i6` `771ba66b...` and `run-gen-i9` `878aa1ff...`, the approved artifact, the runner under the 1200% cap.
Regime (`regime.log`, 250 ms, N=3970): SM 172 to 2782 MHz, power 9.3 to 145.7 W, 54 to 77 C. Collector `--validate`
rc=0.

Verbatim (`mapped/reading.log`):

`DAY44 MAPPED CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY44 CLAUSE census i9 mmap=120 (15219032064 bytes) pinned=0 (0) paged=0 (0) rule pinned=0 paged=0 mmap>0 on every i9 run -> PASS`

`DAY44 ARM i6 window_door_ms_per_token pooled=21.97 o1=21.59 o2=21.97 window_s median=1.034 iqr=0.017 install_s median=60.74 iqr=1.45 loaded_s median=4.98 iqr=0.71`

`DAY44 ARM i9 window_door_ms_per_token pooled=21.86 o1=21.53 o2=22.00 window_s median=1.030 iqr=0.018 install_s median=9.84 iqr=0.17 loaded_s median=0.73 iqr=0.04`

`DAY44 CLAUSE (i) no_regression i9_minus_i6 window pooled=-0.004 o1=-0.002 o2=+0.001 noise=0.018 rule <=noise pooled and both orders -> PASS`

`DAY44 READING (ii) install i9_minus_i6 o1=-51.71 o2=-50.47 noise=1.45 -> install_falls`

`DAY44 READING (iii) load i9_minus_i6 o1=-4.53 o2=-3.84 noise=0.71 -> load_not_higher`

`DAY44 MAPPED rig=rtx5090 integrity=ok census=PASS no_regression=PASS`

I9 stays (clause (i) holds). The banks load as 120 views of the artifact's mapping (15,219,032,064 bytes, no pinned
copy), the load falls 4.98 to 0.73 s and the install 60.74 to 9.84 s: the record pass now compares against
page-cache bytes instead of write-combined pinned memory, as registered.
