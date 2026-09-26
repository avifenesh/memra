#!/usr/bin/env python3
"""B6: the five G2 sizes D's day-10 campaign did not cover, through D's own worker (OWED 22).

Loads research/spill-d-20260919/run-g2.py as a module, replaces only its SIZES with the unrun
registered sizes from CELLS-ENVELOPE.md (16 KiB, 256 KiB, 4 MiB, 64 MiB, 1 GiB), and runs its
worker unchanged: calibration to >= 500 ms per visit, then 5 AB plus 5 BA scored visits per size,
600/600 W checked on every sample, one uninterrupted inherited lock. Invoke it only as the
collector's --execute child with --lock-fd @COLLECTOR_LOCK_FD@ (D's --launch path re-executes
run-g2.py itself and would run D's sizes).
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent / "spill-d-20260919" / "run-g2.py"
UNRUN = [16384, 262144, 4194304, 67108864, 1073741824]

spec = importlib.util.spec_from_file_location("run_g2", D)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
if "--launch" in sys.argv:
    print("REFUSED: run this only as the collector's --execute child (see the docstring)", file=sys.stderr)
    sys.exit(2)
G.SIZES = UNRUN
out = Path(sys.argv[sys.argv.index("--out") + 1])
sys.argv[0] = str(D)
try:
    G.main()
except (ValueError, OSError, AssertionError) as error:
    print("REFUSED: " + str(error), file=sys.stderr)
    sys.exit(2)
(out / "b6-wrapper.json").write_text(json.dumps({"wrapper_sha256": G.B.digest(Path(__file__)),
                                                 "worker": str(D.relative_to(HERE.parents[1])),
                                                 "worker_sha256": G.B.digest(D), "sizes": UNRUN}, indent=1) + "\n")
