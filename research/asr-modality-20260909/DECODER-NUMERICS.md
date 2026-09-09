# Whisper cached decoder numerical stages

Same pinned checkpoint, fixture and CPU conditions as `ENCODER-NUMERICS.md`.
Twelve forced decoder steps on the two-second synthetic clip, self and cross KV cached,
absolute positions, tied output logits. Bounded numeric and cache evidence only.
This forced token loop is not the Whisper transcription policy, and it carries no
statement about transcript quality, throughput or serving.

## FP32

Native FP32 versus HF FP32, same encoder input: worst step max abs **1.430511474609375e-5**,
limit **1e-3**. All 12 raw argmax tokens match HF. Receipt: `stage3-decoder-f32.json`.

## FP16, measured apples to apples

The first FP16 attempt compared an HF F16 capture built on the F16 encoder oracle against an
F32 control built on the F32 encoder oracle. Those two references consume different encoder
states, so the resulting floor measured encoder FP16 error as well as decoder rounding.

The gate now runs on one encoder input. HF FP16 was recaptured on the FP32 encoder oracle,
and the native FP16 run was fed the exact bytes the HF FP16 decoder consumed, so the
encoder-input hash is identical on both sides.

| Quantity | Fixture max abs |
| --- | ---: |
| Native F16 versus HF F16 | 0.015625 |
| Reference's own noise, HF F16 versus HF F32 | 0.0345668793 |
| Native F16 versus the FP32 truth | 0.0345668793 |
| Native F32 versus HF F32 | 0.0000143051 |

All 12 argmax tokens match the FP32 truth, and HF F16 and HF F32 agree on every argmax.

0.015625 is one FP16 ULP at these logit magnitudes. Two distinct FP16 rounding programs
cannot be required to agree more closely than their shared distance from FP32, so the gate
is the same shape the owner set for the encoder: bound the native-versus-reference delta at
fixture level by the same-fixture HF F16 versus HF F32 delta. Native error against the FP32
truth is also required to be no larger than the reference's own error against that truth,
and here the two are equal to the bit. Both bounds come from reference captures only.
Native F16 exceeds the per-step floor on 9 of 12 steps by 1.07x to 1.76x, which is the
expected triangle-inequality band for two FP16 programs; that count is recorded in the receipt.

Receipt: `stage3-decoder-f16.json`.

## Red arms

The decoder gate was exercised failing before its passes were accepted:
contaminated encoder-oracle pairing rejected by field name, a single logit element moved by
0.5 fails the gate, an F16 run without its FP32 control is refused, and an F32 capture
offered to the F16 gate is refused on numeric class.
