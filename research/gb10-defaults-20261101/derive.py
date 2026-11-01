#!/usr/bin/env python3
"""GB10 (DGX Spark) serving defaults: the arithmetic behind the unified-memory derivations.

Run: python3 research/gb10-defaults-20261101/derive.py > research/gb10-defaults-20261101/derive.txt

Inputs and where they come from:
- Qwen3.8-27B prefix entry geometry: 29,696 B/token trunk KV + 156,893,184 B recurrent state
  (the target card's boot line; worker.rs `derived_prefix_budget_context_term_is_the_served_context_set_or_unset`).
- First Spark run: MEMRA_CTX=262144, ~70 GB free after weights, 48-prompt agentic working set,
  32768 MB prefix cache set by hand, 90 s streaming TTFT missed by 4 concurrent 14k-42k prompts.
- Reference prefill rate: RTX PRO 6000 (188 SMs) pp512 4118 tok/s, 27B NVFP4 (docs/PERFORMANCE.md, pro6000wk).
- GB10: 48 SMs (sm_121).
- Serving constants: SPEC_SHRINK_RESERVE 1.5 GiB, MEMRA_ADMIT_DRIVER_HEADROOM_MB 2048, 90 s shipped window.
"""

MIB = 1 << 20
GIB = 1 << 30

# Prefix cache.
bpt, recurrent, ctx = 29_696, 156_893_184, 262_144
entry = bpt * ctx + recurrent
two_entries = 2 * entry
print(f"entry at ctx {ctx}: {entry} B = {entry / 1e9:.2f} GB")
print(f"discrete default (2 entries): {two_entries} B = {two_entries / 1e9:.2f} GB")

reserve, headroom = 1536 * MIB, 2048 * MIB
for free_gib in (70, 95):
    free = free_gib * GIB
    clamp = free - reserve - headroom
    share = max(two_entries, clamp // 2)
    print(
        f"unified, boot free {free_gib} GiB: clamp {clamp // MIB} MiB, "
        f"budget {min(share, clamp) // MIB} MiB ({min(share, clamp) / entry:.1f} full-ctx entries)"
    )

# Agentic working set: 48 prompts of 14k-42k tokens, mean ~28k.
for tokens in (14_000, 28_000, 42_000):
    e = bpt * tokens + recurrent
    print(f"entry at {tokens} tokens: {e / 1e9:.2f} GB")

# Streaming TTFT.
ref_tok_s, ref_sms, concurrent, sms = 4118, 188, 4, 48
rate = ref_tok_s * sms / ref_sms / concurrent
print(f"GB10 prefill rate per concurrent prefill: {ref_tok_s} x {sms}/{ref_sms} / {concurrent} = {rate:.1f} tok/s")
for c in (262_144, 131_072, 65_536, 32_768, 8_192):
    num = c * 1000 * ref_sms * concurrent
    den = ref_tok_s * sms
    ms = max(-(-num // den), 90_000)
    print(f"served ctx {c}: streaming TTFT default {ms} ms ({ms / 1000:.0f} s)")
# The failing cell: 4 prompts of up to 42k tokens at the per-prefill rate.
print(f"42k-token prompt in the c4 cell: {42_000 / rate:.0f} s (> 90 s shipped window)")

# Non-streaming deadline (whole response): streaming TTFT + 8192 open output tokens at 4 tok/s.
# 4 tok/s floors the measured per-stream decode under concurrent prefill (GB10 agentic c4 cell,
# 2026-11-01, per-user median 4.59 tok/s); 8192 is admission's open-output charge.
open_tokens, floor_tok_s = 8192, 4
gen_ms = -(-open_tokens * 1000 // floor_tok_s)
for c in (262_144, 131_072, 32_768, 8_192):
    num = c * 1000 * ref_sms * concurrent
    den = ref_tok_s * sms
    ttft = max(-(-num // den), 90_000)
    print(f"served ctx {c}: non-streaming default {ttft + gen_ms} ms ({(ttft + gen_ms) / 1000:.0f} s)")
