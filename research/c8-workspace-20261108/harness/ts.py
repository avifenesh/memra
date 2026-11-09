"""Prefix every stdin line with its arrival unix time (ms resolution)."""
import sys
import time

out = sys.stdout.buffer
for line in sys.stdin.buffer:
    out.write(b"%.3f " % time.time() + line)
    out.flush()
