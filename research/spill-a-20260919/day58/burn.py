#!/usr/bin/env python3
"""DAY55 R2's burner: spin one CPU-bound loop until killed (no I/O, no allocation). Runs inside the test's own scope, so
it contends only for that scope's quota."""
import sys
x = 0
while True:
    x = (x * 1103515245 + 12345) & 0xFFFFFFFF
