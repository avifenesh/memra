# PRE-REGISTERED rubrics and comparison rule, sampled quality cell (gemm-prime suffix door)

Written BEFORE any generation exists (commit timestamp is the receipt). Built from the
prompts alone: turn 4 = agentic8.json idx2 (arena orchestrator final pass), turn 8 =
agentic8.json idx6 (agnix GitHub issue #1099 marketplace listing task). No model output
had been produced for this cell when this file was committed.

## Scoring scale

Six items per turn, 1 point each (0.5 allowed for partial credit), score in [0, 6].
Judged blind: outputs shuffled and stripped of arm labels and timing before reading.
The model is a thinking model; the emitted reasoning is judged when content is empty,
content preferred when present.

## Turn 4 rubric (arena orchestrator "final pass")

The prompt asks for a final pass on an implementation in "this repo": fix real bugs,
harden the phase engine (edge cases, timeouts, degraded workers), polish the web
spectator view, concrete reviewable changes only with no rewrites, run detected
tests/build, end with a concise summary. No repo is attached to the conversation.

1. ON-TASK: responds to the arena-orchestrator final-pass request, not to earlier
   turns' topics (learning guide, try_notify tool).
2. GROUNDING: handles the absent repo honestly: asks for it, states assumptions, or
   frames output as plan/checklist; does NOT assert fabricated results as fact
   (e.g. "ran the tests, all green" with invented files).
3. COVERAGE: touches all three named axes: real-bug fixing, phase-engine hardening
   (edge cases / timeouts / degraded workers), spectator-view polish.
4. VERIFY: addresses the tests/build instruction incl. detection (package.json /
   Makefile / Cargo.toml or equivalent).
5. CONSTRAINT: respects "concrete, reviewable changes only - no rewrites".
6. SHAPE: ends with (or clearly produces) a concise summary of final changes.

## Turn 8 rubric (agnix issue #1099, JetBrains Marketplace listing)

The prompt: handle "[BUG] JetBrains Marketplace listing: Documentation link returns
404, Source Code link references stale repository URL" (agent-sh/agnix#1099);
investigate, triage or implement on a branch, summarize.

1. ON-TASK: addresses the marketplace-listing issue, not prior turns.
2. DECOMPOSITION: identifies BOTH defects: documentation link 404 AND stale
   source-code repository URL.
3. MECHANISM: names a plausible concrete place those links live for a JetBrains
   plugin (plugin.xml urls, gradle intellij config, marketplace admin page, README).
4. PROCESS: follows "investigate and handle": triages it as a real bug needing a
   change, describes branch/PR workflow.
5. GROUNDING: does not assert unverifiable specifics as fact (e.g. invented merged
   PR numbers); states what it would check (the live issue, the repo).
6. SHAPE: includes the required summary of what was done.

## Disqualifiers (counted separately, never silently dropped)

- EMPTY: no emitted text (reasoning + content both empty).
- LOOP: greedy-loop-shaped degenerate repetition dominating the output (a known
  sampling artifact; counted, not allowed to dominate the verdict).
- TRUNC: finish_reason=length AND cut mid-instruction; recorded per row.
  With max_tokens=1024 on a thinking model truncation may be common; it is a
  per-arm count, and scores are still assigned to what was emitted.

## Pre-registered comparison rule

Per evaluated turn: COLD self-spread = (max - min) of COLD's 8 rubric scores.

- WARM-GEMM is INDISTINGUISHABLE from COLD if |median(GEMM) - median(COLD)| <=
  COLD self-spread AND median(GEMM) >= median(COLD) - 1.0, on both turns.
- An arm DEGRADES if its median falls below COLD's median by more than the COLD
  self-spread on either turn, or its disqualifier count exceeds COLD's by >= 3
  on either turn.
- Outcome mapping (from the tasking):
  (1) GEMM indistinguishable from COLD -> door adds no quality harm.
  (2) both warm arms degrade -> continuation itself, not the door.
  (3) GEMM degrades and WALK does not -> the door is the problem, stays OFF.

TTFT and accept rates are banked but do not enter the quality verdict.
