# Agent Knowledge Base Index

Learning guides synthesized from web research, designed for RAG retrieval by AI agents working on this repository.

## Available Learning Guides

| Topic | File | Sources | Generated | Status |
|-------|------|---------|-----------|--------|
| **README craft for memra** — first-impression evidence, credibility without overclaiming, narrow-scope positioning, benchmark presentation, 20-README comparison, review checklist | [readme-craft-inference-engine.md](readme-craft-inference-engine.md) | 47 | 2026-08-17 | **Authoritative** |

### Why the 2026-07-16 guide is superseded

It contains fabricated evidence: an invented mistral.rs benchmark row, an invented head-to-head table with fake competitor versions and driver numbers, and a `## What llama.cpp is NOT` section that does not exist in llama.cpp's README. It also claims llama.cpp publishes `llama-bench` output with uncertainty bars (it publishes no numbers at all) and credits vLLM with badges (it has none). Roughly 40% of its examples come from ecosystems memra is not in, and several of its recommendations — an OS × accelerator support matrix, PyPI download badges, contributor avatars, and titling the project "RTX 50-series" — are actively wrong for a two-card, single-maintainer Rust/CUDA engine. The audit is [Part 0 of the new guide](readme-craft-inference-engine.md#part-0--audit-of-the-existing-guide), which also lists the six ideas worth keeping (all absorbed).

## Trigger Phrases

### Before editing README.md — always
- "edit the README" / "update the README" / "add to the README" → readme-craft-inference-engine.md **§9 Review Checklist**
- "README review" / "check the README" / "README diff" → readme-craft-inference-engine.md §9

### First impressions and structure
- "README best practices" → readme-craft-inference-engine.md
- "what do developers see first" / "above the fold" / "first screen" → §1
- "README structure" / "section order" / "TOC" / "badges" → §6, §7.1
- "how long is too long for a README" → §4.3, §6.3, §9 item S1

### Credibility and claims
- "how do we look credible without overclaiming" → §2
- "single maintainer credibility" / "small project vs big project" → §2.1, §2.4
- "what reads as marketing" → §2.3
- "byte-exactness as a differentiator" / "determinism claim" → §2.3
- "the word exact" / "exactness gate wording" → §2.3, §9 item C3

### Scope and positioning
- "narrow scope" / "how to frame limited hardware support" → §3
- "why shouldn't I use memra" / "look elsewhere section" → §3.3, §8.3 A1
- "positioning against vLLM / SGLang / llama.cpp / TensorRT-LLM" → §3.2, §6.1

### Numbers and benchmarks
- "performance claims in README" / "benchmark presentation" → §4
- "benchmarking crimes" / "how to report a measurement" → §4.1
- "competitor column" / "COMPETITOR-SETUP" / "fair comparison" → §4.6, §9 gate X
- "what was not measured" → §4.4, §9 item N6
- "benchmarks that damaged credibility" → §4.5
- "how much perf goes in the README" → §4.3

### Onboarding
- "quick start" / "time to first success" / "install section" → §5
- "does Docker matter" / "curl | sh" / "one-liner install" → §5.3
- "CUDA prerequisites disclosure" / "driver floor" / "glibc" → §5.3
- "expected output in examples" → §5.1, §9 item Q3

### Anti-patterns
- "README anti-patterns" → §7
- "hardware disclosure in docs" → §6.1 (flash-attention gating pattern), §8.3 A7
- "format vs model support" → §7.2 (format-as-support), §9 item C2

## Usage Guidelines

1. **Run §9 before committing any README diff.** The checklist is the deliverable; the rest of the guide is its justification.
2. Cite the named evidence, not the guide's authority — e.g. "Heiser crime 4.3" or "NN/g's 10-second finding", so the claim can be checked.
3. Competitor README observations are dated 2026-08-17. **Re-fetch before quoting a competitor as precedent.**
4. Source metadata with quality scores, self-evaluation, and stated gaps: `resources/readme-craft-inference-engine-sources.json`.

## Maintenance

- Guides are versioned by generation date; competitor structure claims carry a fetch date.
- To refresh: re-run `/learn` with `--depth=deep` and re-fetch the 20 READMEs in the sources file.

## File Structure

```
agent-knowledge/
├── CLAUDE.md                          # This index
├── AGENTS.md                          # OpenCode/Codex-compatible index (same content)
├── readme-craft-inference-engine.md   # Authoritative README guide + review checklist
└── resources/
    └── {topic-slug}-sources.json      # Source metadata, quality scores, stated gaps
```

---

*Created with the `/learn` skill. All content synthesized from publicly available sources with full attribution; every competitor README claim was fetched live and dated.*
