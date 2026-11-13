#!/usr/bin/env python3
"""render_readme.py: memra#1266. Fill README.tmpl.md's placeholders from sections/*.md (a missing
section reads as pending) into the research README."""
import pathlib
L = pathlib.Path.home() / ".cache/memra-1266"
t = (L / "README.tmpl.md").read_text()
# Round 2's sections first: their keys contain the older ones (R2_AG35 holds AG3, R2_GAG5 GAG).
for key, name in [("R2_GRIDERS", "r2-griders"), ("R2_RIDERS", "r2-riders"), ("R2_AG35", "r2-ag35"),
                  ("R2_GAG5", "r2-gag5"), ("R2_GPAD", "r2-gpad"), ("R2_PAD", "r2-pad"),
                  ("R2_GNC5", "r2-gnc5")]:
    f = L / "sections" / f"{name}.md"
    t = t.replace(key, f.read_text().rstrip("\n") if f.exists() else "PENDING")
for key, name in [("VERDICT_PARAGRAPH", "verdict"), ("AG3", "ag3"), ("AG2", "ag2"), ("AG4", "ag4"), ("GAG", "gag"),
                  ("NORMAL_TABLES", "normal"), ("BENCH_TABLE", "bench"), ("GATES_TABLE", "gates"), ("CPU_TABLE", "cpu"),
                  ("GB10_M3", "gb10_m3"), ("GB10_F3", "gb10_f3")]:
    f = L / "sections" / f"{name}.md"
    body = f.read_text().rstrip("\n") if f.exists() else "PENDING"
    t = t.replace(key, body)
(L / "wt/research/pooled-verify-memcap-20261113/README.md").write_text(t)
print("rendered")
