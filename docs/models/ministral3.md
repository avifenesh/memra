# Ministral 3 8B Instruct

NativeReference text decoder at `mistralai/Ministral-3-8B-Instruct-2512@5b26027e7b19eeb4b7352e1fed3926375dd2cb4d`.

The pack preserves 34 layers, 4096 hidden units, 32/8 query/KV heads, 128-wide
heads, SwiGLU, full causal attention, and separate input/output embeddings.
Pixtral and the multimodal projector are excluded. Image inputs are not supported.
YaRN uses factor 16 and original context 16384 with unit rotary amplitude. Queries
are multiplied after rotation by `1 + 0.1 * ln(1 + floor(position / 16384))`.
The configured 262144 context is a checkpoint fact, not a measured serving limit.

Tekken has a dedicated case-sensitive Unicode splitter. The Mistral v7 renderer
preserves the artifact default system prompt, adjacent-turn aggregation, tool
JSON, `[TOOL_CALLS]name[ARGS]JSON`, and tool results. Raw argument strings remain
unchanged when assistant history is rendered. Malformed generated calls remain
content; they never become executable calls.

The publisher recommends temperature below 0.1 for daily use. The pack chooses
sampled temperature 0.05 and top-p 1.0 when callers omit them. Explicit request
values win. No new environment flag is introduced.

The source uses per-tensor FP8 weights with static activation scales. This pack's
native eager serving path refuses that storage program rather than dropping its
activation scales. The weight-only GPTQ NVFP4 artifact minted from a recorded BF16 expansion of the
pinned FP8 weights passed its tensor census, generation smoke, sampled tool round trips
and eight-turn continuation at c=1 and c=4. The BF16
expansion is calibration material, not the serving deliverable.

Evidence and remaining gates: `research/ministral3-onboarding-20260909/GATES.md`.
NativeReference is bring-up evidence only. It does not grant production admission.
