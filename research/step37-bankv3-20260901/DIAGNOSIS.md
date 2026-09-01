# step37 NVFP4 bank v2: where the layout and the reader disagreed

**Milestone 1 of the bank-v3 re-derive lane.** Branch `lane/step37-bankv3-20260901`, cut from
`origin/main` at `d1e9eab57`. Pure archaeology + static audit: no box time, no build, no
measurement in this document. Everything below is a citation into this repository's history.

## Verdict, up front

The 2026-08-29 incident (`75bf4ce76`) blamed "the slot-major v2 TP expert-bank layout" and
falsified its bit-identity claim. **The layout was innocent and the claim was true.** The
host permutation is a genuine pure permutation, and every kernel written to read it reads it
correctly. The corruption came from a **defaulted function argument in the prefill grouped-GEMM
tail kernel** that only the v2 layout consumes:

> `crates/memra-engine/cu/moe_f16_grouped.cu`, `moe_kq_sktail_kernel`, lines **1471-1472**:
> ```cuda
> braw0 = kq_fetch<QT>(wrow, (kb + 1) * SKT_BK + bc0, s_cb);
> braw1 = kq_fetch<QT>(wrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb);
> ```
> `kq_fetch` is declared `kq_fetch(wrow, k0v, s_cb, int in_f = 0)`. These two call sites omit
> `in_f` and silently take the default **0**. Eight of the ten `kq_fetch<QT>` call sites in the
> file pass `in_f`; exactly these two do not.

`in_f` is read by **exactly one** branch of `kq_fetch` — `QT_NVFP4_V2`:

```cuda
const int n_slots = in_f >> 5;
const int g = k0v >> 5, sub = (k0v >> 4) & 1;
r.f1 = g_ue4m3_to_float(wrow[(size_t)n_slots * 16 + g * 2 + sub]);   // scale tail
const uint8_t* qsp = wrow + (size_t)g * 16 + sub * 8;                // codes
```

With `in_f == 0`, `n_slots == 0`, and the UE4M3 **scale byte is fetched from `wrow[g*2 + sub]`
— an address inside the packed-codes region** instead of the row's scale tail at `n_slots*16`.
The 4-bit codes are still fetched correctly. So the kernel multiplies the right weights by the
wrong per-16-element scale, for every k-block except `kb == 0` (the `kb=0` windows at lines
1458-1459 do pass `in_f`).

Every other `kq_fetch` branch (`QT_NVFP4`, `QT_Q4_K`, `QT_Q6_K`, `QT_IQ4_XS`, `QT_IQ3_S`)
derives its addressing from `k0v` and superblock constants and **never touches `in_f`**. The
missing argument is therefore a no-op for v1 and every other qtype, and corrupting for v2
alone.

### This is still live on `origin/main` today

`fd0a175ab` deleted the door but deliberately kept `QT_NVFP4_V2` alive ("Survives: ... QT_NVFP4_V2
(EP2 + moe-tp2-repro via MEMRA_TP2_QT)"). The two defaulted call sites are unchanged at
`d1e9eab57`. Any prefill that reaches the deep tail arm with the always-slot-major **EP2** banks
is silently reading wrong scales right now. That is a correctness finding independent of this
lane's perf mandate and is fixed as this lane's first commit.

## Why it looked like "the layout corrupts text"

### It reproduces the whole observed fingerprint, including why the gates passed

`75bf4ce76` recorded three facts that no gross layout mismatch can explain but this mechanism
explains exactly:

1. **"Divergence at token 1 is already in the prime's logits."** The defect is in a
   *prefill/prime* GEMM. Decode never touches this kernel — the decode sweeps go through the
   `qmatvec_nvfp4_dp4a_sel_v2*` family, which is correct (audited below).

2. **"At 613 tokens the first token still agrees and the arms diverge later — the damage is
   MARGIN-dependent, not length-dependent, which is why every gate prompt (613 and up) sat on
   the safe side."** A wrong *scale* on correct *codes* is a bounded perturbation, not garbage
   and not NaN: the substituted byte is an arbitrary code byte reinterpreted as UE4M3, so the
   dot product is off by a modest relative factor. The logits move a little; the argmax flips
   only where the top-1 margin is narrow. The corruption is present in **both** prompt lengths —
   only its visibility is margin-gated. That is precisely why a length-indexed gate ladder could
   not find it, and why the soak, which metered request success rather than answer text, stayed
   green while its own banked replies carried corrupt leading fragments.

3. **The arm-selection geometry.** `moe_kq_sktail_kernel` is the **deep tail** arm, taken by CSR
   groups with `m_e < cross`. `moe_f16g_tail_on()` is **default ON** (`MEMRA_F16G_TAIL=0` is the
   only opt-out) and `cross` defaults to **64** (`moe_f16g_sk_params`, `lib.rs:159-171`). step37
   is 288 experts at top-8, so a prime of `t` tokens builds `8t` pairs over up to 288 CSR groups:
   at `t=25` every group has `m_e` in single digits and the tail arm takes **all** of them; even
   at `t=613` the mean group is ~17 pairs, still far below 64. The corrupt arm was the arm the
   serving prime actually ran.

### Why the "bit-identical to v1" claim survived review

Three independent instruments were all blind to it, and each blindness is a gate-craft lesson:

- **`bank_v2_layout_tests` is a host-side test.** `75bf4ce76` said this itself ("a host-side test
  cannot see a reader mismatch"). It pins the documented slot-major byte map and passes. It is
  correct and it is irrelevant to a reader defect.
- **The host-canonical oracle followed the door.** `run_full_bank_expert_nvfp4` and its three
  siblings (`tp.rs:9978-10154`) branch on `nvfp4_bank_v2_on()` and call `qmatvec_nvfp4_fast_v2`
  when the door is on. So the "oracle" and the path under test **both** moved to v2 together: the
  comparison was v2-vs-v2, never v2-vs-v1. An oracle that is a function of the flag under test
  measures nothing.
- **Neither instrument ran the prefill GEMM.** Both the layout test and the per-expert oracle
  exercise the decode matvec family. The defect lives in a prefill kernel that no v2 gate touched.

### Why the bisect could not name the mechanism

`75bf4ce76` reports `MEMRA_NVFP4_BANK_V2=1` as the sole trigger out of 24 doors, which is
correct — and not separable. `qmatvec_nvfp4_sel_down8_into` **hard-refuses** without the v2 banks
(`lib.rs`: `if !crate::tp::nvfp4_bank_v2_on() { return Err("NVFP4 sel down8 requires the v2
banks") }`), and the gate+up fusion `gu_fused` auto-arms on `nvfp4_bank_v2_on()` with **no door of
its own** (`tp.rs:10712`). Toggling one env var therefore toggled at least three program elements
at once: the bank layout, the `SEL_DOWN8` fused down+combine, and the `_sel_v2_gu` fusion. The
receipt names one door because only one door existed; the mechanism was never isolated to one of
the three. **Design consequence for milestone 5: down8 must be rebuilt as its own independently
armable arm with its own gate, so a future bisect can separate layout from selector.**

## Audit: what was actually correct

Byte maps verified by hand against the v1 reader, for both the codes and the scale bytes, over
the block_nvfp4 geometry (64 elements per 36-byte superblock = 4 bytes of per-16-element UE4M3
scale + 32 bytes of packed e2m1; a "slot" is 32 elements = 16 code bytes + 2 scale bytes).

Host permutation, `tp.rs nvfp4_matrix_v2_permute` at `fd0a175ab^`, emits per row:
`n_slots x 16` code bytes, then `n_slots x 2` scale bytes. For slot `g`, with `sblk = g/2` and
`h = g%2`: codes from `b[4 + 16h .. +16]`, scales `b[2h]` and `b[2h+1]`.
The v1 reader at slot `g` uses `sblk = g>>1`, `whichHalf = g&1`, `s0 = 2*whichHalf`, reading codes
at `b + 4 + s*8` for `s in {s0, s0+1}` and the scale at `b[s]`. **The two agree on every byte**,
and row size is unchanged (`n_slots*18 == (in_f/64)*36` for `in_f % 64 == 0`).

| reader | v2-aware | verdict |
|---|---|---|
| `qmatvec_nvfp4_dp4a_v2` | yes | correct: byte map matches, same dp4a/scale order and same reduce tree as v1 — genuinely bit-identical |
| `qmatvec_nvfp4_dp4a_sel_v2` | yes | correct: same body over `sel[t]*expert_stride` |
| `qmatvec_nvfp4_dp4a_sel_v2_gu` | yes | correct: `_sel_v2` body, blocks `[0,out_f)` on gate and `[out_f,2*out_f)` on up |
| `qmatvec_nvfp4_dp4a_sel_v2_down8` / `_down8_rows` | yes | correct: `nsb <= 32` and `n_sel <= 32/8` are launcher-enforced, `s_dot[8]` cannot overflow, and at the fit-block size the reduce really is the same tree |
| `kq_fetch<QT_NVFP4_V2>` (grouped GEMM) | yes | **body correct, two callers feed it `in_f = 0`** — the defect |
| `dequant_nvfp4v2_f16_kernel` | yes | correct: `in_f` passed at the only call site (`memra_moe_f16g_dequant`) |
| `run_tensor_parallel_routes_nvfp4_prime_grouped` | yes | correct: selects `QT_NVFP4_V2` for `bank_qt`, carrying its own receipt comment about an *earlier* v1-kernel-on-v2-bytes bug |

## The defect class, and the design input it produces

Bank v2 was introduced as a **parallel qtype tag plus a parallel kernel family**. Correctness then
depended on every predicate, every size macro, every dispatch arm and every argument list that
enumerates NVFP4 being updated in lockstep. It has now failed that way **twice**, both times
v2-only, both times invisible to a host-side test:

- `KQ_CB_WORDS(QT)` sized `s_cb[1]` for `QT_NVFP4_V2` while `kq_stage_codebook` staged 16 words
  for both NVFP4 layouts — 15 words of shared-memory overrun per launch. Found by
  compute-sanitizer 2026-08-28 (6816 invalid shared writes in `moe_kq_sktail_kernel<107>`, `qt=7`
  clean), fixed in `068cbc425`. **That fix was already in the incident tree** (`068cbc425` is an
  ancestor of `75bf4ce76`), so it is not the incident's mechanism — it is the first instance of
  the class.
- `kq_fetch`'s `int in_f = 0` default, above: the second instance, in the same kernel, and the one
  that shipped.

Still-unconverted sites of the same class, latent because nothing currently routes a v2 bank
through them, listed so v3 does not walk into them:

- `qmatvec.cu` `expert_dot_g(int qtype, ...)` has **no** `QT_NVFP4_V2` case at all; a v2 bank
  passed with `qtype == QT_NVFP4` silently gets the v1 36-byte body. It backs the whole
  `moe_gate_up_silu8_dev_q8*` / `moe_down8_fma_dev_q8*` / `moe_pairs_matvec_q8*` family.
- `qmatvec_gemm.cu` `StageMeta<GQT_NVFP4>` / `sb_byte_off<GQT_NVFP4>` hardcode the 36-byte
  superblock with no v2 twin.
- `qmatvec.cu:407` `deq_nvfp4` in the Stage-A generic dequant switch: no v2 case.
- `qmatvec_nvfp4_fast_prequant_into` hardcodes the `qmatvec_nvfp4_dp4a` (v1) kernel with no v2
  branch. Dead at `fd0a175ab^`, so it never fired, but it is a v1 reader with a bank-shaped
  signature.
- `nvfp4_matrix_v2_permute` asserts only its **input** extent. Its output is `n_slots*18` bytes
  per row while the struct stores `row_bytes = nvfp4_row_bytes(in_features) = (in_f/64)*36`;
  those agree only for `in_f % 64 == 0`. At `in_f ≡ 32 (mod 64)` the permute would silently
  produce a longer row than the stride every reader uses. step37 is safe (gate/up `in_f` 4096,
  down `local_in` 640, both multiples of 64) but nothing enforces it.

### What "layout and reader as ONE unit" has to mean, concretely

The two failures were not layout errors and not reader errors. They were **geometry-plumbing**
errors: a byte map that is correct in two places, and a piece of geometry needed to evaluate it
that one caller failed to supply. So the one-unit rule is not "write them in the same commit" —
`fd0a175ab^` had them in the same commit. It is:

1. **The layout's geometry travels with the pointer, in one struct, with no defaultable field.**
   The v3 reader must take a descriptor (`{ base, row_bytes, n_slots, scale_tail_off, expert_stride }`)
   whose constructor is the *only* way to obtain it. `int in_f = 0` must be unrepresentable. A
   defaulted scalar that only one layout consumes is the exact hole that shipped.
2. **One source of truth for the byte map, shared by repack and reader.** A single header of
   constants and offset functions, `#include`d by the CUDA side and mirrored by a Rust
   `const fn` with an equality test, so a change cannot land in one and not the other.
3. **No second qtype tag.** A parallel `QT_*` constant is what created the lockstep-predicate
   burden. v3 is a property of the descriptor, not a new tag threaded through every
   `qtype ==` chain in three `.cu` files.
4. **The oracle may not be a function of the flag.** The v1 reader is the oracle and it must stay
   pinned to v1 bytes: unpack(v2) vs unpack(v1) compared on dequantized values, never
   fast_v2-vs-fast_v2.
5. **The gate must run the prefill GEMM.** Both prior instances were in a prefill kernel that no
   v2 gate touched. Decode-only byte identity proved nothing and will prove nothing again.
6. **The short-prompt oracle is mandatory, and it is the margin instrument.** The corruption was
   always present at 613 tokens and merely invisible. A byte gate that samples only long prompts
   is measuring margin width, not correctness.

## Sources

- `fd0a175ab` — door removal (parents carry the full v2 implementation); `75bf4ce76` — the
  serving refusal and the four-arm/first-token bisect; `068cbc425` — the `KQ_CB_WORDS` fix.
- `research/step37-reasoning-effort-20260829/` (incident receipts, `raw/cell11-first-token-ab.txt`),
  `research/step37-bankv2-removal-20260829/RESULTS.md`, `research/perf-chain-20260831/RESULTS.md`
  (cell 1: the doors were worth -21.53% wall / 157.60 vs 120.25 decode).
- Live tree at `d1e9eab57`: `crates/memra-engine/cu/moe_f16_grouped.cu:1458-1472`,
  `crates/memra-engine/src/lib.rs:159-208`.
- The 140-era serving env is the bench box's `agentic8.sh` `ENVV=` line; it carries
  `MEMRA_NVFP4_BANK_V2=1` and `MEMRA_SEL_DOWN8=1` alongside `MEMRA_MOE_DIRECT=1`,
  `MEMRA_ROUTES_PRESTAGE=1`, `MEMRA_STEP_NVFP4_DEV_ROUTES=1` and no `MEMRA_TCOL_FFN`.
