Oracle dump (14 arrays, npz + flat twins) regenerates deterministically:
  oracle-venv/bin/python tools/dspark_q38_oracle.py <cum1000_export> oracle.npz <SpecForge@2590f48e3a93>
(fixed seeds; census 62/62). Arrays not banked (27MB, mostly the 248320-vocab
base_logits stand-in); shas of the box-2 run in MANIFEST.sha256. Parity verdict
ALL PASS banked in WIRING-RESULTS.md + driver log.
