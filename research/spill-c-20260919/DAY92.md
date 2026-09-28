# WP-C day 92 (2026-09-27): OWED C2, option 1: I25a (the host demand trace behind a gate-set flag) and I25b (the registry's owner check), registered before any code

The lead, relaying the owner's ruling (option 1 only): "I25a: approved by the lead under the flags doctrine. A
22,000-line-per-run trace on the naked door is a diagnostic, so it goes behind a flag that gates and cells set.
Register it text-first as planned; I am telling the owner. I25b is fine. I26: design first." The I24 `promo` sitting
(`DAY91.md` section 2) runs meanwhile on both classes, and nothing waits on I25 or I26. Tree at start: `ba8a5f4cc`.
Gates before a push: the affected crates' tests and clippy.

## 1. I25a: the trace as a gate-set diagnostic

**Today.** `TracedDispatch` (`banked_residency/native.rs`) always traces. Per demanded record it does four things:
- it reads the pre-demand SLRU slot (`resident_slot`, which the trace's `slot` and the stage clock's hit counts use);
- it updates the host-slot occupant vector (the trace's `victim`);
- it formats one `[expert-host-slru] key=...` line into a buffer;
- it writes the buffer to stderr every 64 KiB and at close.

That is 22077 lines per 32-token run. REF does none of it.

**The change** (`memra-engine`: `banked_residency.rs`, `banked_residency/native.rs`, `bin/run_gen.rs`,
`bin/run_spec.rs` only where the CLI lists its keys; `docs/FLAGS.md`).
1. **A new CLI flag** `--expert-bank-trace` (no value; given twice refuses; a door flag, so with
   `MEMRA_EXPERTS_VIA_TIER=0` it refuses like the others) sets `ExpertBankBudget::trace`.
   - It is a CLI flag, not an env read. FLAGS gets a row for it as a flag gates and cells set, beside
     `--expert-bank-stages`.
2. **Without the flag**, `TracedDispatch` reads no pre-demand slot, keeps no occupant vector, allocates no buffer and
   writes nothing.
   - Its demands, fill intake and every answer are unchanged: it passes each call to the adapter as now.
   - The stage clock's owner section then prints `trace=off` in place of `host_hits` and `host_misses`, which only the
     pre-demand read counted. It keeps `inner_demand_ns` and `trace_ns=0`.
3. **With the flag**, every trace line is byte for byte today's, at the same points, and the day-48 and day-84 trace
   censuses keep their strings.
4. **Nothing else changes**: the numeric program, the tape, the host demand the bank sees, the fill, the stage clock's
   other terms.

**What reads the trace, and how each keeps reading it.**
- **The integrity checks** read one host demand sequence from the door arms: the local check, the split's `--check`
  and `day88-read.py`'s integrity. From I25a on, the arms whose sequence a check reads carry `--expert-bank-trace`.
  - A check never counts an untraced arm's empty sequence as a match.
  - `day83-read.py` gets `--traced <arms>`: its host-demand check reads only those arms, and an arm it reads must
    hold a nonempty trace.
- **The next `promo` sitting's cell** (after I26) runs `naked` untraced for timing, beside a traced twin of it
  (`naked-t`) whose sequence the integrity reads with `q22`'s. It registers with that sitting.
- **`tests/bank/day8.rs`** reads a committed trace fixture and does not change.

## 2. I25b: the registry's owner check without a thread handle

**Today.** `ExpertBankProxy::access` checks `self.owner != thread::current().id()`, and `thread::current()` clones
the thread handle. The engine enters `access` four times per ticket (`host_resident_many`, `demand_many`,
`with_bytes_each`, `finish_group`), plus the single-demand and validate paths.

**The change** (`memra-tier`: `bank/owner_proxy.rs`).
- A thread-local copy of the current thread's id, read by `access` and `register`.
- The same comparison, the same `WrongOwner` from another thread, and the same order of refusals. Nothing else.

## 3. The gates and the measurement

- **Code gates.** I25a: memra-engine's `banked_native` tests plus the new ones, and clippy on memra-engine's lib and
  tests. I25b: memra-tier's tests and clippy on all targets. Both: fmt. The new tests:
  - the CLI parse of `--expert-bank-trace`: accepted, twice refused, and with `=0` refused;
  - a traced and an untraced `TracedDispatch` over the same bank and demands give the same demands, leases and bank
    state, the traced one the day-48 lines, the untraced one no line.
  - I25b's owner-check answers are covered by the proxy's `WrongOwner` test and I24's fixture, unchanged.
- **The local queue v22** (`rtx5090-day92/`, the argv of v21).
  - **The check:** `run-gen-p88`, and `run-gen-i25` with `--expert-bank-trace`, in the order p88, i25t, i25t, p88.
    It must give one tape and one host demand sequence, and an untraced `run-gen-i25` run must give the same tape.
  - **The split:** arms `p88s` (traces, as every build before I25a), `i24s`, `i25s` (untraced) and `i25t`
    (`--expert-bank-trace`), each with both clocks; order 1 is p88s, i24s, i25s, i25t, x 5; order 2 is reversed, x 5.
    That is 40 runs, and `--check --traced p88s,i24s,i25t` reads the integrity.
  - **The reading.** DAY89 section 2's sizing rule, cumulative from `p88s` to `i25s`, is the one that counts: the
    naked program is untraced. `i25t` against `i25s` is the trace's clocked cost, deciding nothing. The stderr writes
    it also saves are outside every leaf, so the rule reads the cut conservatively.
  - A `short` reading sends the work to I26's design. A `reaches` reading makes NEED TARGET CARD for the `promo`
    sitting with the traced twin.

## 4. Queue v22 read as registered (2026-09-27, 23:44Z to 23:57Z; `rtx5090-day92/`)

**The check** (`check/reading.log`): `DAY89 GPU CHECK PASS`. I25 traced reads p88's tape and host demand sequence
(`4bdc2610c3534e42`, 22077 lines) in both orders. `DAY92 UNTRACED i25u-a rc=0 MATCH the same tape as p88-a
trace_lines=0 -> PASS`.

**The split** (`split25/reading.log`): `DAY83 SPLIT CHECKS rig=rtx5090 runs=40 integrity=ok`. Verbatim:
- `DAY92 SPLIT rig=rtx5090 p88s generate N=10 door-only leaves summed=314.5 us per token`
- `DAY92 SPLIT rig=rtx5090 i24s generate N=10 door-only leaves summed=264.7 us per token`
- `DAY92 SPLIT rig=rtx5090 i25s generate N=10 door-only leaves summed=233.0 us per token`
- `DAY92 CHANGE rig=rtx5090 generate i25s minus i24s: -31.8 us per token`
- `DAY92 SIZING rig=rtx5090 generate door-only change p88s->i25s=-81.6 us per token, threshold -155 -> short`
- `DAY92 CHANGE rig=rtx5090 generate i25t minus i25s: +36.3 us per token`

**Read as registered.**
- I25 removes 31.8 us per generated token beyond I24: `outer` -22.8 (the untraced door's pre-demand reads) and
  `own_trace` -14.9. From p88 to I25 the door-only work falls 81.6 us per token.
- The trace alone costs 36.3 us per token clocked (`i25t` against `i25s`); its stderr writes add more outside every
  leaf.
- The rule reads `short`. `DAY91.md` section 3 retired it as the 285K predictor, so it decides nothing here; the
  285K question is `DAY94.md`'s.
