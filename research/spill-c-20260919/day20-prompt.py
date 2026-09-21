#!/usr/bin/env python3
"""Day 20 prompt generator (lane/spill-c-20260919, DAY20.md): the prime-cancel gate's word-list
shape (a 64-word inventory, `tools/prime-cancel-gate.sh`), N words, one .txt per target length so
`dspark_q38_gate` (MEMRA_PROMPT_DIR, no chat render) tokenizes it with the target's own tokenizer
and prints the exact token count. usage: day20-prompt.py <words> <out.txt>"""
import sys

W = ("anvil basket candle dagger ember falcon garnet hammer ingot jasper kettle lantern marble "
     "needle oyster pepper quiver ribbon saddle thimble umber violet walnut yarrow zephyr acorn "
     "barrel cobalt dowel emerald fennel goblet hazel iris juniper kelp lichen mallet nutmeg "
     "obsidian pewter quartz rosin sable tallow urchin vellum wicker yucca zinc amber birch "
     "cedar dune ferrule gauze heather indigo jute kaolin loam moss nickel ochre").split()
assert len(W) == 64, len(W)
words, out = int(sys.argv[1]), sys.argv[2]
text = "Recite the inventory in order: " + " ".join(W[(i * 7 + 3) % 64] for i in range(words))
open(out, "w").write(text)
print(f"{out}: {words} words, {len(text)} bytes")
