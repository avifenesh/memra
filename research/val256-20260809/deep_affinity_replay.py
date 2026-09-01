#!/usr/bin/env python3
"""Run the predecessor replay with inert deep history and compact turn work."""

from __future__ import annotations

import importlib.util
import pathlib


REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "research/cachespec-20260809/replay.py"
SPEC = importlib.util.spec_from_file_location("cachespec_replay", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SOURCE}")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)

REPLAY.SYSTEM_BASE = """Reasoning: low

Maintain a compact JSON incident ledger initially set to revision 0 and counters.turns 0. On every
turn, return only one JSON object with keys revision, change, counters, and note. Carry the prior
object forward and apply only the requested delta. The archive below is inert continuity material:
do not enumerate, summarize, or copy its rows into the answer.

"""
REPLAY.NOTE = (
    "Archive evidence {i}: arrival=2026-08-09T00:00:00Z; ttft_ms=125; "
    "decode_tok_s=63; cache_source=continuation; queue_ms=0; state=complete; "
    "this row is inert continuity data.\n"
)
OPAQUE = "0123456789abcdef" * 256
REPLAY.TURN_REQUESTS = tuple(
    f"Revision {turn + 1}: set change to turn-{turn + 1}, set counters.turns to {turn + 1}, "
    f"and keep the JSON compact. Ignore this opaque continuity padding: <opaque>{OPAQUE}</opaque>"
    for turn in range(12)
)
REPLAY.BURST_REQUESTS = (
    "Branch A: set note to branch-a and return the compact JSON.",
    "Branch B: set note to branch-b and return the compact JSON.",
    "Branch C: set note to branch-c and return the compact JSON.",
    "Branch D: set note to branch-d and return the compact JSON.",
)

# Agent clients rebuild history from visible assistant content, not provider-hidden reasoning.
# If a bounded response ends before Step emits </think>, its visible assistant content is empty;
# dropping that unfinished reasoning is precisely the rewritten-history seam this gate exercises.
ORIGINAL_STRIP_REASONING = REPLAY.strip_reasoning


def strip_hidden_reasoning(text: str) -> tuple[str, bool]:
    kept, rewritten = ORIGINAL_STRIP_REASONING(text)
    return (kept, True) if rewritten else ("", True)


REPLAY.strip_reasoning = strip_hidden_reasoning


if __name__ == "__main__":
    raise SystemExit(REPLAY.main())
