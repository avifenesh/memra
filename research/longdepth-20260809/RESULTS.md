# Step-3.7-Flash long-generation corruption — results

Date: 2026-08-09

Branch: `lane/cx-longdepth`

Scored context: `MEMRA_CTX=262144`

Baseline runtime: `40b1f613` (12-cell run), with the completed spec-off 12K rerun at
`31473b35`

Fixed runtime: `585d46c4` (`3e2af07a` sampler fix plus `585d46c4` omitted-field defaults)

## Verdict

**The cross-lingual token soup is a GPU Gumbel-sampler bug, not a 262K-context, SWA, MTP,
RoPE, or KV-wrap bug.** It reproduces with speculation disabled, at both 131K and 262K context
caps, and as early as completion token 281. More generated tokens increase exposure to the
sampler defect; there is no late positional threshold in the retained foreign-token evidence.

That verdict is specifically about arbitrary cross-lingual token injection. A post-fix 12K run
still had an ordinary HTML parse error, and the requested `MEMRA_FAST=0` onset cross-check found
a separate numeric quality delta. At the causal token 3642, the default Stage-B row assigned
`td` 18.78% probability and sampled it, while the Stage-A f32-dequant oracle assigned it 7.51%
and sampled the grammatically correct `dd` under the identical Philox draw. This is not the old
sampler bug: both rows were finite and both raw argmaxes were `dd`. It also is not isolated to an
attention accumulator: `MEMRA_FAST=0` changes the broad matvec numeric class while retaining the
same Step35 SWA/KV/FA/RoPE path. The exact anatomy is reported below; no unsafe runtime-default
change is presented as a small fix.

The CUDA uniform conversion admitted `u == 1.0f`. For a high Philox `u32`, conversion to `f32`
rounds the integer to `2^32`; the old expression then evaluates to exactly one:

```text
u = ((float)v + 1.0f) * 2^-32
G = -log(-log(u)) = +inf
```

That vocabulary lane wins Gumbel-max regardless of its model probability, injecting an arbitrary
token. The two first visible failures in the plain-device controls match this mechanism exactly:

| seed | output index (0-based) | selected token | Philox `u32` | old `f32 u` |
|---:|---:|---|---:|---:|
| 2026080901 | 504 | 94712, `工作了` | 4294967240 (`0xffffffc8`) | 1.0 |
| 2026080902 | 281 | 57066, `这让` | 4294967187 (`0xffffff93`) | 1.0 |

There are 128 `u32` values that hit this rounding case. With 128896 independent vocabulary draws
per sampled token, the IID exposure estimate is 0.003834 per token, or one event per 260.8 sampled
tokens. That agrees with an issue that looks progressively worse in long output while actually
starting early. `philox_receipt.py` recomputes both exact counters from the frozen response token
arrays and proves the first fixed-stream divergences also land on `u == 1.0f`; its raw output is
`raw/root-cause-philox-receipts.txt`.

The fix clamps only this rounded endpoint to the greatest `f32` below one:

```cpp
return fminf(u, 0x1.fffffep-1f);
```

All ordinary Philox-to-float results retain their previous bits. The new `sample-check` arm uses
the two live seeds, stream positions, and token ids and requires every perturbed logit to remain
finite.

## Context-cap result — lowering to 131K is not a mitigation

The imported live responses settle the original 131K-versus-262K uncertainty:

| context cap | completion tokens | first forbidden reasoning character | first forbidden content character |
|---:|---:|---|---|
| 131072 | 8687 | char 261, `ا` | char 65, `給` |
| 262144 | 9632 | char 896, `牌` | char 575, `刘` |
| 262144 control | 4000 | none | no content emitted |

The originally reported 8.7K and 9.6K values were total response lengths, not onset positions.
Exact examples are `canvas اجتماع`, `<meta charset給了UTF-8">`, `move the mouse,牌照?`, and
`font-size刘备? 14px`. Full JSON, extracted content, hashes, and the steering receipt are under
`raw/orchestrator-live/`.

**Serving guidance: keep the required 262144 context. The fault follows sampled-token exposure,
not the configured context ceiling.** StepFun's current
[model card](https://huggingface.co/stepfun-ai/Step-3.7-Flash) also declares a 256K window.

## Controlled matrix

All 24 requests used the same artifact hashes and rendered prompt hash:

```text
trunk shard 1  b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259
trunk shard 2  e7e0caaaf0057fabc8bf9b71cbe41322f9945a44df7240bb58e6b7c375e7ffec
trunk shard 3  ccbd3df81b4f4cb8e73d899734944bcbdefcf436faec9203353419c6750c0590
MTP draft      469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57
rendered prompt dbbf222b94ff45035026482e87b1a9c2353650598b3bd294858d0fb23bc52564
```

Notation below: `NL@i` is the first forbidden non-Latin token, `P@i` is a parser failure, and
`none` means neither signal. The event listed first is the cell's first-corruption-token index.
`n=` is the actual returned completion length. Natural EOS may end before the requested depth.

| spec | temp | requested | repetition 1 | repetition 2 |
|---|---:|---:|---|---|
| off | 0 | 2048 | none (`n=2048`) | none (`n=2048`) |
| on | 0 | 2048 | none (`n=2048`) | none (`n=2048`) |
| off | 0.7 | 2048 | **NL@504** (`工作了`, `n=2048`) | **NL@281** (`这让`, `n=2048`) |
| on | 0.7 | 2048 | **NL@1525**, P@1568 (`導演`, `n=2050`) | none (`n=2048`) |
| off | 0 | 6144 | none (`n=6144`) | none (`n=6144`) |
| on | 0 | 6144 | none (`n=6146`) | none (`n=6146`) |
| off | 0.7 | 6144 | **NL@504** (`工作了`, `n=6144`) | **NL@281**, P@3221 (`这让`, `n=6144`) |
| on | 0.7 | 6144 | **NL@1525**, P@1568 (`導演`, `n=6144`) | **P@2346**, NL@3838 (`实的`, `n=6146`) |
| off | 0 | 12288 | **P@11478** (`n=12288`) | **P@11478** (`n=12288`) |
| on | 0 | 12288 | **P@11478** (`n=12290`) | **P@11478** (`n=12290`) |
| off | 0.7 | 12288 | **NL@504**, P@7194 (`工作了`, EOS `n=7196`) | **NL@281**, P@3221 (`这让`, `n=12288`) |
| on | 0.7 | 12288 | **NL@1525**, P@1568 (`導演`, `n=12290`) | **P@2346**, NL@3838 (`实的`, EOS `n=11222`) |

The requested-depth prefixes are deterministic within a seed. In particular, the spec-off
sampled failures remain at 504/281 in the 2K, 6K, and 12K requests. All six greedy spec-on/off
pairs are exact token-prefix matches through the requested depth; the deterministic 12K parser
failure is the same model typo in both modes:

```html
<dt>Headroom</td><dd>30% minimum.</dd>
```

The four sampled failure families are quoted below. Repeated cells reuse the same seed and exact
prefix, so these are the unique first-positive excerpts across the matrix:

```text
main{margin-left工作了 environment effectively. The operations team must monitor...
.header .brand { font-weight这让: 700; font-size: 1.25rem; ...
<dt導演</dt><dd>Operations Director</dd>
<td>Yes</实的><td>CISO</td><td>sec@atlas</td>
```

Secondary parser examples after those first corruptions include
`<td>Data Team</tdauthor missions><td>Quarterly</td>` and the parser's exact
`saw </tr> while <tbody> was open`. The complete byte offsets, token ids/text, excerpts, and
tokenizer-span validation are in each `rep*/detector.json`.

Forced-spec output also exposed a separate response-budget defect: some `MaxNew` responses exceed
the requested cap by one or two tokens (`6144 -> 6146`, `12288 -> 12290`, and a diagnostic
`2048 -> 2049/2050`). This is retained as a separate API-contract issue; it does not explain the
foreign tokens and was not folded into the sampler fix.

## Isolation controls

| control | result | implication |
|---|---|---|
| spec disabled, GPU sampler, `temp=0.7 top_p=1` | NL at 504/281 for both seeds | MTP is not required |
| host sampler, spec off, `top_p=1`, 2K N=2 | both clean | failure is in the device sampler path |
| host sampler, spec off, `top_p=1`, 6K N=2 | no NL in either; one clean EOS, one ordinary P@4390 | separates token injection from model HTML mistakes |
| GPU sampler, `top_p=0.9`, spec off/on, 2K N=2 each | all four clean | excluding low-probability tail ids mitigates the old bug |
| greedy, spec off/on, through 12K N=2 | exact token-prefix match; no NL | trunk/MTP/KV/RoPE do not spontaneously corrupt long greedy output |

StepFun's current
[API documentation](https://platform.stepfun.ai/docs/en/api-reference/chat/chat-completion-create)
gives omitted defaults `temperature=0.5` and `top_p=0.9`.
Memra previously applied generic `1.0/1.0` defaults to Step35 chat requests. `585d46c4` now applies
the provider defaults only when those fields are omitted; explicit caller values, including
`temperature=0` and `top_p=1`, remain authoritative. This is a defense-in-depth serving fix. The
Gumbel endpoint repair remains necessary because full-vocabulary sampling is a valid explicit
request.

## Suspect disposition

| suspect | anatomy and controlled evidence | verdict |
|---|---|---|
| Step35 SWA at rolled positions | first foreign tokens are at 281/504 sampled tokens; greedy remains exact through 12K; exact Philox lanes explain those ids. The later fast/oracle parse probe keeps the same `fa_decode_kvmod`, last-512 view, RoPE, and KV arithmetic in both arms | falsified for token soup; not the variable changed by the secondary oracle |
| MTP drafter geometry (`swa=true`, window 512) | spec-off runs reproduce both deterministic bad ids; forced-spec uses the same CUDA uniform helper | falsified as token-soup root |
| RoPE precision at 262K | the controlled prompt is 368 tokens and the deepest returned absolute position is about 12658, far below the configured cap; positions are unchanged by the cap; 131K also fails | falsified as token-soup root |
| KV rolling wrap | `Cache::new_inner` allocates each full-attention slab for `max_ctx`; decode appends at monotonic `kvl.len` and SWA takes a last-512 view (`off = len - win`), with no modulo/ring overwrite | falsified as token-soup root |
| CUDA Gumbel endpoint | exact bad token id + seed + stream position maps to a Philox lane whose old `f32 u` is 1.0 and Gumbel is `+inf` | confirmed token-soup root |
| Stage-B activation-quantized matvec numeric class | exact forced prefix at post-fix parse cause: default samples `td`, `MEMRA_FAST=0` samples `dd`; the narrower `MEMRA_MMVQ=0` arm is worse and raw-argmaxes `td` | confirmed secondary HTML-quality variable; no small fix isolated |

This also corrects the phrase “long-depth corruption”: length controls cumulative sampling
opportunities, not SWA/RoPE/KV positional arithmetic.

## Fixed verification

All fixed cells deliberately kept the harder explicit `temperature=0.7, top_p=1` configuration.
The first-seen RunPod rig ran the forced-spec 12K verification as required.

| rig | spec | requested | repetition 1 | repetition 2 | forbidden NL |
|---|---|---:|---|---|---:|
| box1 | off | 2048 | clean (`n=2048`) | P@1770, natural EOS (`n=1772`) | 0/2 |
| box1 | off | 12288 | P@3650 (`n=12288`) | P@1770, natural EOS (`n=1772`) | 0/2 |
| first-seen RunPod | forced on | 12288 | P@12184 (`n=12288`) | clean (`n=12288`) | 0/2 |

Across these six responses, **none of 42456 server-reported completion token ids produced a
forbidden non-Latin codepoint in visible output**. The pod's two forced-spec requests both reached
the full 12288-token depth; its server
log records 745 `[spec-acc]` rounds. The remaining parser positives are coherent model-generated
HTML mistakes, not arbitrary vocabulary injection:

```html
<dt>Lockout</dt><td>5 failures / 15 minutes</dd>
<dt>At rest</dt><td>AES-256</dd>
```

The natural-EOS failure is exactly `generation stopped with <main> open`. Therefore the
cross-lingual serving corruption is fixed, while perfect long-form HTML validity is not claimed.

## Post-fix fast-path cross-check at the remaining parse error

The SOTA-sweep steering required a fast-versus-oracle comparison if any 12K detector result
remained positive after the clamp. The probe teacher-forced the exact 368-token rendered prompt
and the fixed run's recorded completion tape so every arm traversed the same prefix. It dumped
the full 128896-logit row before a named zero-based token and then replayed the actual CUDA
Gumbel kernel with seed `2026080901`, temperature `0.7`, and the same stream position.

The parser first reports at token 3650 (`</`, id 1718), but token-span receipts locate the causal
source choice eight tokens earlier:

```html
<dt>Lockout</dt><td>5 failures / 15 minutes</dd>
                    ^ token 3642: `td`, id 5333
```

At parser-visible token 3650, both arms raw-argmax and sample `</` with very wide margins; it is
only where the parser notices the already-created mismatch. At causal token 3642:

| arm | raw argmax | `dd` probability | `td` probability | exact CUDA sample |
|---|---|---:|---:|---|
| default Stage-B/MMVQ | `dd` by 1.02514 logits | 81.2218% | 18.7782% | **`td`** |
| `MEMRA_FAST=0` Stage-A oracle | `dd` by 1.75711 logits | 92.4851% | 7.5149% | **`dd`** |
| `MEMRA_MMVQ=0` dp4a control | **`td`** by 0.68293 logits | 27.3764% | 72.6236% | **`td`** |

Default versus Stage-A at that row has `max_abs=2.21351` and `rms_rel=0.10199` over the full
vocabulary. The identical Gumbel field changes no sampler arithmetic; it exposes the distribution
difference. Disabling MMVQ does not repair it and moves farther from the oracle, so this is not a
small MMVQ reduction-order fix. The source path also shows `MEMRA_FAST=0` does not select a
different attention implementation: both arms append the same quantized KV, take the same
monotonic last-512 SWA view, and call `fa_decode_kvmod`; the changed variable is the broad
Stage-B activation-quantized projection/FFN matvec class feeding that shared attention path.

The oracle is therefore anatomy, not a shippable default. It measured 19.37 tok/s versus 24.68
tok/s for the default on this exact forced window, changes many numeric sites at once, and has not
passed the required local-5090 default-flip campaign. The Gumbel endpoint was a one-line
correctness defect with exact receipts and is fixed; the residual long-form syntax-quality delta
needs a dedicated numeric-quality lane rather than a speculative accumulator patch here.

Setup failures in this late diagnostic are retained verbatim: `cargo: command not found` in the
non-login SSH shell (rerun with the installed toolchain binaries directly), `env: ‘-u’: No such
file or directory` in the first oracle command (unset options had followed an assignment),
`no bin target named logit-sample-probe` (Cargo reported `logit_sample_probe`), and
`./target/release/tok-span: No such file or directory` (the built target is `tok_span`). Each
failed block released the flock immediately; none loaded a model or supports a model conclusion.

## Gates

Fixed revision `585d46c480f3dcd83314ea9cc080626f7bbf490c`:

- box1 direct sampler gate: `gumbel open interval (live receipt seeds): OK`, then
  `=== sample-check ALL GREEN ===`;
- first-seen RunPod direct sampler gate: same result under the service CUDA 13.1 runtime;
- model-backed `kernel-check`: `ALL GREEN: kernels match CPU reference.`;
- `run-gen`: prefill/decode argmax MATCH and batched-prime/tokenwise argmax MATCH;
- `run-spec`: K=1..8 all `self-consistency: PASS`, final `=== SELF-CONSISTENCY PASS ===`;
- local server unit suite: 132 passed, 0 failed;
- restored RunPod service: fixed commit, `/health` and `/readyz` HTTP 200, worker idle.

The first bare-shell pod sampler attempt is preserved with the exact failure
`CublasError(CUBLAS_STATUS_NOT_INITIALIZED)`. Both GPUs were empty; rerunning with the pod's
required `/root/serve-env.sh` CUDA 13.1 library path passed. No model conclusion is based on the
failed environment setup.

## Receipt map

| path | contents |
|---|---|
| `raw/orchestrator-live/` | 131K/262K live responses, content extracts, hashes, steering |
| `raw/20260808T223925Z/` | frozen 12-cell baseline matrix and detector records |
| `raw/20260808T233600Z/` | authoritative completed spec-off sampled 12K N=2 rerun |
| `raw/20260808T234300Z/` | host-sampler and StepFun-`top_p` diagnostics |
| `raw/20260809T000200Z-fixverify-box1/` | post-fix spec-off cells, sampler gate, full exactness battery |
| `raw/20260809T000200Z-fixverify-pod/` | mandatory first-seen-rig spec-on verification and restored-health receipt |
| `raw/20260809T004500Z-accum-oracle/` | exact forced-prefix fast/Stage-A/dp4a logit rows, CUDA samples, token spans, timings, and retained harness failures |
| `raw/root-cause-philox-receipts.txt` | exact CPU reconstruction of bad Philox lanes and fixed divergences |
| `raw/local-cargo-test-memra-server.log` | complete 132-test server suite |

The original 12K detector attempts that failed because the terminal EOS token had no text are also
retained. Their exact tokenizer error is preserved; repaired analysis accepts exactly one terminal
EOS id only when `stop_reason=Eos`. The native API now emits a terminal full-token snapshot so
speculative round coalescing cannot truncate research token receipts.

No origin push, merge, tag, release, or `rustup` command was performed.
