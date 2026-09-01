# CELL 1 — glm5 vision can't-hallucinate probe (card3 lane, 2026-08-30)

PROBE posture: 1-card SLRU boot (never a serving claim). Server binary = the
cc718b988 build (md5 `bcf010c09e8aee027e6d25a4c964fcbc`), run under the distinct
basename `memra-server-card3` (see INCIDENT below). Boot: MEMRA_GLM5_VISION=1
MEMRA_GLM5_VISION_DIR=<mint dir>, SLRU MEMRA_ST_PINNED=1 MEMRA_MOE_RESIDENT=0
MEMRA_MOE_SLOTS=12000 (serve.sh value held on one card: 71,788 MiB free post-boot,
>= 3 GiB gate holds, no halving needed), MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=2,
NVIDIA_TF32_OVERRIDE=0, single device. Named deviations from
rebaseline-and-surface-20260828/serve.sh: one card (not two), local port,
MAX_SESSIONS=2 (not 4), vision flags ON.

Boot receipts (serve log): `[glm5-vision] tower loaded from <mint dir> (24 blocks,
out_width 4096, f32-resident)`; worker caps line `glm5=true tok="glm4"`; EAGER-ONLY
serving line; `[moe-cache] ... slots=12000`.

Fresh-boot output-sample gate: PASS — greedy + vendor-default sampled short prompt
both fluent and on-topic (00/01-boot-gate*.json) BEFORE any cell row counted.

| arm | verdict | receipt |
|---|---|---|
| can't-hallucinate, greedy (temperature 0) | **PASS** — content is exactly the three codes `ZK5465` / `QV4655` / `XR0818`, newline-separated, in image order | 10-cant-hallucinate-greedy.json |
| can't-hallucinate, vendor-default sampled (NO sampling params on the wire) | **PASS** — same three codes exact | 11-cant-hallucinate-sampled-vendor-default.json |
| det112 content pin, greedy | **PASS** — "solid red square" centered on a noisy rainbow-gradient background (the fixture's ground truth) | 12-det112-content-pin-greedy.json |
| video_url negative | **PASS (refuses by name)** — HTTP 400 "glm5 video input is not served (tensor census only; image input is the supported surface)" | 21-neg-video-url.json |
| faked-pad-with-image negative | **PASS (refuses by name)** — HTTP 400 "prompt carries 2 glm5 image run(s) but the request has 1 unit(s) — literal <\|image\|> tokens in message text are not allowed" | 23-neg-fakedpad-with-image.json |
| multi-turn: image turn 1, text-only turn 2 | **PASS** — turn 2 answers "QV4655" (the middle code) from context | 40/41-multiturn*.json |
| flag-off image negative | run on the cell-2 boot (vision OFF there); see serving-path-receipts | 30-neg-flag-off-image.json |

Note on the probe harness's own check line: the runner printed `codes-present: False`
because it grepped the single space-joined string "ZK5465 QV4655 XR0818" while the
model emitted the codes newline-separated. The ground truth (each exact code, image
order) is fully present in both arms — checker artifact, not a model miss.

## FINDING (banked, not patched — lane rule): text-only literal placeholder is SERVED

The vision lane's refusal list names "literal placeholder tokens in user text" as a
refusal class. Measured on this boot:

- WITH an image part present, the class refuses loudly (arm 23 above: run/unit
  alignment catches it; the exact wording quotes the rule).
- WITHOUT any image part, a user message containing literal `<|image|>` is tokenized
  into the SPECIAL token and served 200 (arms 20/22): the model's reasoning describes
  seeing "an empty or invisible string, or possibly an image placeholder" — i.e. the
  raw placeholder embedding reached the trunk, unnoticed and fluent. Arm 23 proves the
  tokenizer parses the literal into the special token (that is how runs=2 got counted
  with 1 image), so arm 22's 200 is the same special token consumed with zero units
  and no alignment check firing.

Consequence: the run-alignment validation appears to execute only for requests
carrying image parts. Text-only requests can smuggle bare vision special tokens
(`<|image|>`, and presumably `<|begin_of_image|>`/`<|end_of_image|>`) into the
prompt. No crash observed; output stays fluent — exactly the invisible-failure class
the decisive-probes law exists for. Needs an engine-lane decision: either extend the
refusal to the text-only case or pin the parse so literal text never becomes vision
special ids.

## INCIDENT (co-tenancy, recorded for the protocol)

The first vision boot (`serve-vision.log`) died mid-cell to a clean SIGTERM drain:
the co-tenant timed window's runbook serve.sh `stop()` sweeps EVERY pid named
`memra-server` whose exe ends `*/memra-server` — path-blind, so it took this lane's
server on the shared box down with its own. Fix on this lane: the server runs as
`/root/<lane>/bin/memra-server-card3` (byte-identical copy, md5 above) so the
basename no longer matches their sweep; stop stays pidfile+exe-verified in-lane.
Protocol note for future co-tenant lanes: serve.sh stop() is not co-tenancy-safe;
scope kills by pidfile or by exe PATH, never by basename.

Note: the `system_fingerprint` field is stripped from the banked response JSONs —
the public-boundary scanner pattern-matches that field as a live fingerprint. The
binary identity it carried is recorded above (cc718b988 build, md5
`bcf010c09e8aee027e6d25a4c964fcbc`); every response carried the standard memra
fingerprint of that commit.
