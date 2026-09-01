#!/usr/bin/env python3
"""Fold the MEASURED decode-kernel census into a per-token ledger for the served config.

  usage: census-ledger.py census.txt <measured tok/s>

Every GEMV number here is measured on the serving box by `decode-kernel-census`, at the shapes
the step37 TP2 decode tick actually issues, and converted to ms/token over 45 layers. The
routed-expert term is the one line the census cannot measure directly (the NVFP4 sweep is not
in it), so it is derived from the census's own q8 twin at the same stacked gate+up shape by
byte ratio, and the derivation is printed rather than asserted.

The point of the ledger is the RESIDUAL: what is left after every GEMV is accounted for. No
weight-precision door (W8, W4, BF16_MMV) can touch the residual, so the residual is the honest
ceiling on what that class of door can ever buy.
"""
import re
import sys

census, tps = sys.argv[1], float(sys.argv[2])
rows = {}
for line in open(census):
    m = re.match(r"\[(.+?)\]\s+([\d.]+) us/call\s+([\d.]+) TB/s.*?([\d.]+) ms/token", line.strip())
    if m:
        rows[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

TP = 2
NMOE = 42
NVFP4_B, Q8_B = 0.5 + 1 / 16, 34 / 32
NVFP4_TBS = 0.79   # the receipted achieved rate of the NVFP4 sel gate/up sweep


def ms(key):
    return rows[key][2]


def tbs(key):
    return rows[key][1]


print("MEASURED GEMV LEDGER, per card per token, at the SERVED step37 TP2 config")
print("  (q8 rows are the doors that are already ON: MEMRA_STEP_TP_W8 + MEMRA_W8_HYBRID)")
served = []
served.append(("qkv fused, q8 (banked)", ms("q8_0 qkv-equivalent 4096->5152")))
served.append(("o_proj, q8 (banked)", ms("q8_0 o_proj-equivalent 4096->4096")))
# SHEXP_OVERLAP splits the shared-expert down rows across the pair and only the HI half reaches
# the mirrored launcher, so the critical path is the bf16 half: half the full-shape bf16 time.
served.append(("shexp down, bf16 lo half is the critical path", ms("shexp down 1280->4096") / 2))
# HEAD_SPLIT does the same to the lm head: rank1's half is q8, rank0's half is bf16 through
# the view launcher, and they run concurrently, so the bf16 half sets the time.
served.append(("lm head, bf16 lo half is the critical path", ms("head 4096->64448 (HEAD_SPLIT half)")))
q8_stack = ms("q8_0 expert gate+up stacked 4096->20480")
gu = q8_stack / TP * (NVFP4_B / Q8_B) * (tbs("q8_0 expert gate+up stacked 4096->20480") / NVFP4_TBS) * (NMOE / 45.0)
served.append(("routed experts gate+up, NVFP4 (derived, see below)", gu))
served.append(("routed experts down, NVFP4 (half the rows)", gu / 2))
tot = sum(v for _, v in served)
for k, v in served:
    print("  %-48s %6.2f ms" % (k, v))
print("  %-48s %6.2f ms" % ("GEMV TOTAL", tot))
print()
print("  expert derivation: census q8 stacked gate+up = %.2f ms/token full width at %.2f TB/s;"
      % (q8_stack, tbs("q8_0 expert gate+up stacked 4096->20480")))
print("    halve for TP2, scale bytes %.4f/%.4f for NVFP4, scale rate %.2f/%.2f, scale 42/45 MoE layers."
      % (NVFP4_B, Q8_B, tbs("q8_0 expert gate+up stacked 4096->20480"), NVFP4_TBS))
print()
token_ms = 1000.0 / tps
print("MEASURED TOKEN at %.2f tok/s = %.2f ms" % (tps, token_ms))
print("  GEMV accounted            %6.2f ms  (%.0f%%)" % (tot, 100 * tot / token_ms))
print("  RESIDUAL, not weight GEMV %6.2f ms  (%.0f%%)" % (token_ms - tot, 100 * (token_ms - tot) / token_ms))
print()
print("WHAT A WEIGHT-PRECISION DOOR CAN STILL BUY, from the census's own q8/bf16 pairs:")
head_win = ms("head 4096->64448 (HEAD_SPLIT half)") - ms("q8_0 head-equivalent 4096->64448")
shexp_win = (ms("shexp down 1280->4096") - ms("q8_0 shexp-down-equivalent 1280->4096")) / 2
print("  lm head lo half bf16 -> q8      %6.2f ms   (MEMRA_W8_VIEW)" % head_win)
print("  shexp down lo half bf16 -> q8   %6.2f ms   (MEMRA_W8_VIEW)" % shexp_win)
print("  W4-class attention projection   %6.2f ms   (NOT IMPLEMENTED, new numeric class)"
      % ((ms("q8_0 qkv-equivalent 4096->5152") + ms("q8_0 o_proj-equivalent 4096->4096")) * 0.41))
best = token_ms - head_win - shexp_win - (ms("q8_0 qkv-equivalent 4096->5152") + ms("q8_0 o_proj-equivalent 4096->4096")) * 0.41
print("  every one of them, stacked      %6.2f ms/token -> %.1f tok/s" % (best, 1000.0 / best))
print("  90 tok/s needs                   11.11 ms/token")
