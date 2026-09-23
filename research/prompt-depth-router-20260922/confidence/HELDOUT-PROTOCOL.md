# Conditional held-out Qwen code validation

This corpus and decision rule were written before opening the six-scenario
fixed-C development result. The earlier simple-helper scenarios and this
corpus have no shared helper functions. Keep the same pinned
Qwen3.8-27B NVFP4+Q5_K model, embedded full MTP head, rebuilt confidence
research binary, desktop RTX 5090, 0.7/20/0.95 sampler, default thinking,
fresh cache per request and 8,192 output-token cap.

## Frozen task families

Each scenario asks for one straightforward four-function Python module,
then an ordinary-language explanation on separate, paired requests. Use
the same 256, 1,024, 4,096 and 16,384 user-token prompt targets and the
same bounded reference-filler procedure as the completed prefix study.

1. Dictionary utilities: `get_or_default`, `count_keys`, `merged_copy`,
   `invert_unique` (the input values are unique).
2. Sequence summaries: `minimum_or_none`, `maximum_or_none`,
   `mean_or_zero`, `count_above`.
3. Coordinate utilities: `translate_point`, `midpoint`,
   `manhattan_distance`, `in_rectangle`.
4. Text cleaning: `normalize_space`, `initials`, `nonempty_lines`,
   `trim_prefix`.
5. Clock utilities: `minutes_to_hours_parts`, `seconds_to_minutes_parts`,
   `clamp_hour`, `format_hhmm`.
6. Set utilities: `union_copy`, `intersection_copy`, `only_left`,
   `is_subset`.

The separate qualification task is four boolean helpers: `xor_bool`,
`nand_bool`, `all_true` and `any_true`. Generate and hash all user prompts,
template token counts, seeds (`20760000` through `20760005`) and arm orders
before inspecting the development grid's rates. Qualify all eight
prose/code length cells on the boolean task at the fixed output cap;
if format coverage or the exact loop screen fails, retain that failure
and stop without held-out throughput.

## Selection and controls

Use the completed development grid only to select the single **positive**
confidence arm with the highest pooled code-request output tok/s across all
four lengths, compared with K=3/C=0. If every positive-C arm is at or below
K=3/C=0, do not score the held-out corpus with this synchronous cutoff;
use a separate phase diagnostic to locate cost instead. Do not select a
different C after opening held-out results.

For a selected arm, run each held-out scenario under all three controls:
K=3/C=0, K=3/selected C, and K=2/C=0. Register a balanced forward/reverse
arm order before generation. Match prompt, sampler, seed, output budget,
artifact, binary and GPU. Repeat the target-identity gate at K=2 and K=3;
record actual offered draft lengths and confidence-shortened rounds. The
primary comparison is pooled generated tokens / complete native request
seconds on the requested-code cells, with six whole-scenario paired changes
per length, output-length and latency ratios, format coverage and matched
loop exclusions. Prose remains a diagnostic.

A winner versus K=3/C=0 that loses to fixed K=2/C=0 has not established a
confidence-specific speed benefit. Any held-out gain here remains native,
single-GPU and synthetic evidence; it does not qualify a live cutoff
learner or a vendor-default HTTP serving policy.
