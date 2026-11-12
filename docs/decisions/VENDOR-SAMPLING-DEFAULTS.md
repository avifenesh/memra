# An omitted sampling field takes the model vendor's published value (2026-08-19)

**Status:** adopted 2026-08-19 on an owner ruling. The per-mode arm (`non_thinking_sampling`)
came later on the same law. Code: the `default_*` keys of a `[models.*]` block in
`MEMRA_MODEL_METADATA`, validated at boot, and the shared resolver `resolve_sampler_config` with
`ModelSamplingDefaults::for_mode` in `crates/memra-server/src/lib.rs`. Tests:
`vendor_sampling_defaults_are_identical_on_every_surface`,
`vendor_defaults_leave_the_pure_temp_sampled_spec_regime`. Flag row: `MEMRA_MODEL_METADATA` in
[docs/FLAGS.md](../FLAGS.md).

The rationale below moved out of the serving doc on 2026-11-12. The old text, with the key table
and TOML examples, is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#per-model-vendor-sampling-defaults)
and its section
["Per-mode arms"](../archive/SERVING-2026-11-12.md#per-mode-arms-non_thinking_sampling).

## Question

A client that omits `temperature`, `top_p`, `top_k`, `min_p` or a penalty gets something. Before
2026-08-19 it got the API-standard default (temperature 1.0, every filter and penalty off), and an
architecture that publishes its own API defaults (`step35`) got those. Which value should an
omitting client get?

## Chosen

The model vendor's own published recommendation for that model. Seven optional keys
(`default_temperature`, `default_top_p`, `default_top_k`, `default_min_p`,
`default_presence_penalty`, `default_frequency_penalty`, `default_repetition_penalty`):

- each key substitutes for exactly one omitted field; an explicit client value always wins, and an
  explicit `temperature: 0` is still true greedy;
- all four generation surfaces resolve through the one resolver, so they cannot disagree;
- an undeclared key leaves the API-standard default. Where the vendor recommends nothing, declare
  nothing;
- the TOML comment cites the vendor's source, so a later cleanup does not "simplify" a deliberate
  number;
- a model with no key is byte-identical to before the keys existed. Architectures that publish
  their own API defaults still supply them from the engine (`step35`: StepFun's 0.5 / 0.9); a
  metadata key outranks that default, field by field.

Per-mode arm: where a vendor publishes two recommendations (Qwen3.8: thinking 1.0 / 0.95 / 20,
non-thinking 0.7 / 0.80 / 20 plus `presence_penalty` 1.5), `non_thinking_sampling` declares the
second. The request's resolved thinking mode picks the arm, whichever spelling turned thinking off.
`/v1/completions`, which has no thinking control, always takes the primary arm.

## Rejected

- **Greedy as the served default.** Owner ruling, 2026-08-19: "we don't have to serve greedy, we
  measure greedy but we serve what the user chooses", "we default to what are the
  recommendations", "greedy can create issues". Greedy stays the measurement mode and stays
  reachable by an explicit `temperature: 0`.
- **A house value where the vendor published none.** The number belongs to the vendor; an
  invented one is a claim nobody measured.
- **`default_temperature = 0.0`.** Fatal at boot. A zero default is greedy-by-default in a config
  file: it applies silently, deployment-wide, to every client that omits the field, which is the
  hazard the keys exist to remove.
- **Blending the two arms.** A field the vendor left out of the non-thinking arm falls to the API
  standard, not to the thinking arm's value. The arms are separate vendor programs; a blend serves
  numbers no vendor published. An empty arm table is refused at boot.
- **Inferring non-thinking from the template default.** A deployment whose unset case should be
  non-thinking says so with `default_reasoning_effort = "none"`, which resolves the request to
  thinking-off upstream, and the arm follows. `ThinkMode::Default` keeps the primary arm.

## Measurement

No A/B decided this: the owner's ruling did, and the values are the vendors' published ones. What
is measured is the contract and its cost:

- **The contract.** `vendor_sampling_defaults_are_identical_on_every_surface` proves the four
  surfaces resolve the same values; boot validation refuses out-of-range values, a zero
  temperature and an empty arm.
- **The throughput profile moves.** Vendor recommendations include truncation filters (`top_p` < 1,
  `top_k` > 0), so the default request leaves the pure-temperature sampled-speculation regime
  (`Sampler::is_spec_sampling`). The MTP route still drafts a filtered request on its captured
  sampled graph with the filter inside it (`MEMRA_SPEC_GRAPH_FILTERED`, default on), and a
  penalised request on the eager chain. Exactness is unchanged on every route: the verify emits the
  target's own filtered draw. `vendor_defaults_leave_the_pure_temp_sampled_spec_regime` pins this
  so the shift stays measured. A client opts back into the pure-temperature regime with an explicit
  `top_p: 1.0` and `top_k: 0`.
- **The non-thinking arm's penalty.** Qwen's non-thinking shape carries `presence_penalty` 1.5,
  which ran on the host penalty sampler and stopped Qwen3.5-9B NVFP4+MTP on one RTX 5090 from
  scaling past c=1. The device penalty arm fixed it: aggregate 109.6 to 527.2 tok/s at c=4 and
  115.5 to 758.0 at c=8 (three boots per arm, interleaved), greedy byte-identical across all six
  boots
  ([`research/devpenalty-qwen35-20260909/RESULTS.md`](../../research/devpenalty-qwen35-20260909/RESULTS.md)),
  and on GB10 later ([GB10-DEVICE-PENALTY-DEFAULT.md](GB10-DEVICE-PENALTY-DEFAULT.md)).

## Operator consequence

`[models.*]` blocks are parsed with `deny_unknown_fields`, so a binary older than these keys
refuses to boot on a config that carries them. Roll out the binary first, then the config.

## What would change it

A vendor that revises its recommendation changes the declared value, with the new citation.
