#!/usr/bin/env python3
"""tsrun.py CMD...: run CMD with stdout+stderr merged, prefix every line with unix time (6 decimals),
forward SIGTERM/SIGINT to the child, exit with its code."""
import signal, subprocess, sys, time

p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(s, lambda n, f: p.send_signal(n))
for line in p.stdout:
    sys.stdout.write(f"{time.time():.6f} {line.decode(errors='replace')}")
    sys.stdout.flush()
sys.exit(p.wait())
