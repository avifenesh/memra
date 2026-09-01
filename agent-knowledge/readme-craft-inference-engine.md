# Learning Guide: Writing memra's README — a narrow, single-maintainer inference engine against entrenched competitors

> **Policy update, 2026-08-23:** the owner moved Memra to a shorter front-door README. Detailed
> performance, model, serving, flag, testing, and architecture material belongs in its focused
> document; `tools/update-perf-board.py` no longer writes a README sample table. This guide remains
> the research record behind that decision, but Part 8's target length and Part 9's PERF-SAMPLES
> checks are superseded by `CLAUDE.md` and the current README structure.

**Generated**: 2026-08-17
**Depth**: deep — 47 sources analyzed (20 live competitor/peer READMEs, 18 peer-reviewed or standards-body sources, 9 practitioner/primary sources)
**Supersedes**: `readme-technical-inference-repos.md` (2026-07-16, 20 sources). See [Part 0](#part-0--audit-of-the-existing-guide) for the audit and the replace/keep decision.
**Scope**: `github.com/avifenesh/memra` — Rust + CUDA inference engine, MIT, one maintainer, tuned for RTX PRO 6000 Blackwell (`sm_120a`) and RTX 5090, compile-gated `sm_90a` H100 lane. Competitors are vLLM, SGLang, llama.cpp, TensorRT-LLM — every one of them larger by orders of magnitude.

All README observations were fetched live on 2026-08-17 from `raw.githubusercontent.com` and are dated as such. READMEs change; re-fetch before quoting a competitor as precedent.

---

## How an agent should use this guide

| You are about to… | Read |
|---|---|
| Add or edit anything in `README.md` | [Part 9 — Review checklist](#part-9--review-checklist-run-this-against-the-diff) first, then the relevant Part |
| Add a number | [Part 4](#part-4--numbers-where-they-go-and-how-to-make-them-read-as-true) + checklist items N1–N9 |
| Add a feature bullet | [Part 2](#part-2--how-a-solo-maintainer-earns-credibility-without-overclaiming) + checklist items C1–C6 |
| Reorder or restructure | [Part 6](#part-6--twenty-readmes-what-each-does-in-its-first-screen) + [Part 1](#part-1--what-a-developer-decides-in-the-first-1530-seconds) |
| Argue about scope | [Part 3](#part-3--framing-a-deliberately-narrow-scope-as-the-advantage) |
| Touch the quick start | [Part 5](#part-5--what-converts-reading-into-cloning-into-serving) |

**Hard rule for agents:** the checklist in Part 9 is the deliverable of this guide. Writing to the README without running it is the failure mode the previous guide had.

---

## TL;DR — the twelve rules that survived the evidence

1. **You get 10 seconds, not 30.** Nielsen Norman Group, summarizing Liu, White & Dumais (SIGIR '10) over 205,873 pages and >2 billion dwell times: "Users often leave Web pages in 10–20 seconds," the "first 10 seconds of the page visit are critical," and only after ~30 seconds does the abandonment curve flatten. The instruction that follows is literal: "communicate your value proposition within 10 seconds."
2. **The first screen is judged before it is read.** Tuch et al. (2012, *IJHCS* 70(11)) found visual complexity and prototypicality shape aesthetic judgement "within the first 50 ms of exposure," and detectably "even within 17ms." Low complexity + high prototypicality rated most appealing. A dense 90-line block loses before a word of it is parsed.
3. **Front-load information-carrying words.** NN/g's F-pattern study (232 users, confirmed 2017): "The first two paragraphs must state the most important information," and "Start subheads, paragraphs, and bullet points with information-carrying words."
4. **Named limits are the cheapest credibility a small project can buy.** SIGPLAN's Empirical Evaluation Checklist lists "Fails to acknowledge limitations" as a first-category defect, alongside "Claims not appropriately scoped." Heiser's benchmarking crimes 1.1 and 1.2 make omitted regressions and unjustified subsetting *crimes*, not tact.
5. **Publish the arm that lost.** mistral.rs's README ships four head-to-head tables in which it loses several BF16 rows to vLLM. That is the highest-trust move observed in any inference README in this survey, and it costs nothing but nerve.
6. **A narrow scope wins when you say what to use instead.** ripgrep's README has a literal `## Why shouldn't I use ripgrep?` heading whose first disqualifier is that ripgrep conforms to no standard: "The best tool for this job is good old grep." No large project does this. It is available to memra and unavailable to vLLM.
7. **Every number carries its conditions inside the same visual unit.** MLPerf Inference rules: "Results that cannot be replicated are not valid results," and "SUT parameters and configuration must be uniquely and specifically named." Heiser 5.1/5.3: state the platform, and never give ratios without absolutes.
8. **Same hardware + different software = 2x.** NVIDIA's own rebuttal to AMD's MI300X-vs-H100 numbers: "The results shared did not use optimized software, and the H100, if benchmarked properly, is 2x faster." Any competitor column without a documented competitor configuration is worth zero.
9. **One command, early, that produces visible output.** llama.cpp's README opens with `## Quick start` *before* `## Description`. Ollama's first content heading is `Download`. `uv` puts install before any example. TGI calls Docker "The easiest way of getting started."
10. **The README routes; the docs hold.** vLLM's entire Getting Started is one install line and three links. SGLang's README contains no code block at all. TensorRT-LLM's contains no command anywhere.
11. **Do not copy a 2,000-contributor README.** SGLang's credibility engine is "over 400,000 GPUs worldwide" and a 30-name adopter list. vLLM's is "over 2000 contributors." Neither is available to you; imitating the *shape* without the substance reads as costume.
12. **Reproducibility artifacts correlate with adoption, measurably.** Papers with Code's ML Code Completeness Checklist (>200 repos, validated on NeurIPS 2019): repos satisfying all five items — including "README file including table of results accompanied by precise commands to run/produce those results" — had a "median of 196 and mean of 2,664 stars," leading the field.

---

## Part 0 — Audit of the existing guide

**File**: `agent-knowledge/readme-technical-inference-repos.md`, 747 lines, generated 2026-07-16, 20 sources.
**Verdict: replace it.** Do not incrementally patch it. It contains fabricated evidence, and a guide about not fabricating numbers that fabricates numbers cannot be repaired by editing around the fabrications — every remaining example becomes suspect.

### 0.1 Where it is factually wrong

Each of these was checked against the live README on 2026-08-17.

| # | Old guide says | Reality | Severity |
|---|---|---|---|
| A1 | Lines 72–84 present a "**mistral.rs** benchmark pattern (exemplary)" with a table row `Qwen3.6-27B \| GB10 \| UQFF Q8 \| 14,520 \| 182` | mistral.rs's benchmark block covers **Gemma 4 E4B and Gemma 4 26B-A4B** across GB10 / B200 / H100 SXM, in four tables (Q8 prefill, Q8 decode vs llama.cpp GGUF Q8_0; BF16 prefill, BF16 decode vs vLLM BF16). No such row exists. | **Fabricated evidence** |
| A2 | Lines 331–359, "Example 2: Performance Claim Table (mistral.rs style)" — invents `Llama-3-8B \| B200 (1x) \| 14,520 \| 14,105 \| +2.9%`, a llama.cpp build `b1234 (2024-12-15)`, `vLLM v0.8.1`, `Driver 560.28.03` | Invented throughout, and labelled in a way that reads as quotation. A reader copying this into a README would publish fake numbers. | **Fabricated evidence** |
| A3 | Line 66: llama.cpp "shows literal `llama-bench` output with build hash, uncertainty bars"; line 694 lists it as exemplary for "reproducible benchmarks" | llama.cpp's README has **no benchmark tables, no throughput figures, no comparison charts**. Its only performance claim is the prose phrase "state-of-the-art performance," plus a link to a *troubleshooting* doc. | **Wrong** |
| A4 | Lines 150–156 quote a `## What llama.cpp is NOT` section ("Not a training framework… Not a hosted service… Not a model hub") as an "Explicit scope boundaries (llama.cpp)" pattern | No such section exists. llama.cpp has no scope or limitations section at all. | **Fabricated evidence** |
| A5 | Lines 49–56 quote llama.cpp's quick start as `llama-cli -hf ggml-org/gemma-3-1b-it-GGUF` with "For detailed build instructions, see docs/build.md" | Current commands are `llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF` and `llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF`, preceded by a four-option install bullet list. | Outdated / misquoted |
| A6 | Lines 183–193 quote vLLM as having `## Supported Models` with "200+ architectures on Hugging Face" | vLLM's headings are About / Getting Started / Contributing / Citation / Contact Us / Media Kit. "200+ model architectures" is a bullet inside About. | Misquoted |
| A7 | Sources JSON credits vLLM with badges; line 216 implies version/downloads badges | vLLM's README has **zero badges** — no CI, PyPI, license, or Discord shield. | Wrong |
| A8 | Line 69 cites "InfluxDB: `<10ms` for last-value queries on X hardware" as a trust tier | Unverifiable in this pass, and InfluxDB is not a peer of memra in any dimension. | Unsupported |
| A9 | Lines 132 and 658–665 tell memra to title itself "memra: LLM inference for sm_120 (RTX 50-series)" / "memra-inference — RTX 50-series LLM Inference" | Understates and misstates the project: the flagship target is the **RTX PRO 6000 Blackwell** (also `sm_120a`), with RTX 5090 second and a compile-gated `sm_90a` H100 lane. "RTX 50-series" erases the primary card. | Wrong, and actively harmful if applied |

### 0.2 Where it does not fit a narrow single-maintainer Rust/CUDA engine

| Old guide advice | Why it misfits memra |
|---|---|
| A 7-row OS × arch × accelerator support matrix with ✅/🟡/⚠️ legend (lines 96–103, 368–412) | memra ships two tuned cards and one compile-gated lane. A wide matrix manufactures the *impression* of breadth memra explicitly refuses to claim, and most rows would carry the same value. The existing 4-row `Family / Supported on / Tuning now` table is the correct shape and should not be replaced by an OS matrix. |
| `pip install`, `check_install.py`, `cp310-cu118` wheel tags, PyTorch C++ extension ABI discussion (lines 493–516) | memra distributes a Rust binary through a release installer that selects published `sm_120a` / `sm_90a` / `sm_89` prebuilts and verifies a release checksum. None of the Python packaging surface exists here. |
| "Downloads/PyPI" and "Test coverage" badges as adoption signals (lines 213–218) | No PyPI. A download badge on a single-maintainer engine advertises a small number; per §2.3 that is a self-inflicted wound. |
| "Contributor avatars … Helps community feel seen" (line 230) | On a one-maintainer repo this renders as an empty room. Inverted signal. |
| Redis/Docker-verified builds, PyTorch binary selector, Rust-lang delegation, Nerfstudio GIFs, Meta-Llama gated access, alpaca-lora (lines 198, 250–255, 588–600, 690–723) | ~40% of the guide's examples are drawn from ecosystems memra is not in. None of the *inference-engine-specific* problems — per-device defaults, exactness gates, competitor-column fairness, format-vs-model support — are addressed anywhere in 747 lines. |
| Both checklists (lines 604–646) | Generic. Neither references a single memra artifact (`tools/update-perf-board.py`, `research/tune-data/current-board.json`, `docs/COMPETITOR-SETUP.md`, the `PERF-SAMPLES` markers, `docs/PERFORMANCE.md`), and neither is expressed as a check you can run against a diff. This is the mechanical reason it "is not being re-checked when adding to the README": it produces no verdict, so there is nothing to fail. |

### 0.3 What to keep (already absorbed into this guide)

Six ideas from the old guide are sound and survive, restated here on real evidence:

1. **README as router, not manual** — hub-and-spoke. Now supported by the actual vLLM / SGLang / TensorRT-LLM structures in [Part 6](#part-6--twenty-readmes-what-each-does-in-its-first-screen).
2. **A falsifiability hierarchy for claims** — the concept was right; the ladder is rebuilt in [§4.2](#42-the-falsifiability-ladder-rebuilt-on-verified-examples) from verified examples only.
3. **Limitations stated inline with the feature, not in a ghetto** — confirmed by ExLlamaV3, PowerInfer, and flash-attention ([§2.2](#22-what-reads-as-trustworthy-verified-patterns)).
4. **"One reproducible number is worth more than ten vague claims."** Keep the sentence.
5. **Generated content for anything that goes stale** — memra already implements the strongest version of this via the `PERF-SAMPLES` block generated by `tools/update-perf-board.py`. Keep and extend; see checklist item N9.
6. **The two-tier split table** (what stays in README vs what moves to docs) — retained and re-cut for memra in [§6.3](#63-the-two-tier-split-for-memra).

### 0.4 Disposition

- Mark `readme-technical-inference-repos.md` superseded at the top of the file and in both indexes (done as part of this run).
- Do not cite it. Do not copy any code block out of it — three of its four "exemplary" blocks are invented.
- Delete it once nothing references it. Its only remaining value is as the record of what a plausible-sounding but unverified guide looks like.

---

## Part 1 — What a developer decides in the first 15–30 seconds

### 1.1 The timing evidence

| Finding | Source | Number |
|---|---|---|
| Page abandonment is front-loaded and follows a Weibull distribution with negative aging — the longer someone stays, the less likely they leave | NN/g, summarizing Liu, White & Dumais, SIGIR '10 | 205,873 pages, ≥10,000 visits each, >2 billion dwell times; "99% of web pages have a negative aging effect" |
| The decision window | NN/g | "Users often leave Web pages in 10–20 seconds"; "first 10 seconds of the page visit are critical"; curve flattens "after people have stayed on a page for about 30 seconds" |
| Reward for surviving | NN/g | Pages that clear ~30 s often retain readers "2 minutes or more, which is an eternity on the web" |
| Aesthetic judgement precedes reading | Tuch, Presslaber, Stoecklin, Opwis & Bargas-Avila, *IJHCS* 70(11), 2012 (Google Research) | Visual complexity + prototypicality affect ratings "within the first 50 ms"; detectable "even within 17ms" (exposures tested: 17/33/50/500/1000 ms) |
| Reading shape | NN/g F-pattern, 232 users, replicated 2017 | Two horizontal sweeps + a vertical scan down the left edge; "The first two paragraphs must state the most important information" |

**Consequence for a README.** The 10-second budget is spent on: the repo name and description in GitHub's chrome, the badge row, the first heading, the first table, and the left edge of the first ~30 rendered lines. Everything after that is spent only if those elements bought it.

### 1.2 The mechanics of the GitHub repo page

Verified from GitHub's own docs (`about-readmes`):

- "A README is often the first item a visitor will see when visiting your repository," and "GitHub will automatically surface your README to repository visitors."
- Precedence if several exist: `.github/`, then repo root, then `docs/`.
- **Rendering truncates past 500 KiB.**
- GitHub auto-generates an outline: "GitHub will automatically generate a table of contents based on section headings," reachable via the "Outline" menu icon. A hand-written jump bar is therefore *additive* — it is visible without a click, which the Outline is not.
- Relative links are rewritten branch-aware; "we recommend using relative links to refer to other files within your repository" so they survive a clone. memra complies (all `docs/…` links are relative; all verified to exist on 2026-08-17, including `docs/decisions/`).

### 1.3 Which specific elements move the decision — measured

| Element | Evidence | Effect |
|---|---|---|
| **Number of links to other repositories in the README** | Fan et al., *EMSE* 2020 (arXiv:2010.02472), 1,149 academic AI repos, top-20% vs bottom-70% by stars, 21 features tested | One of the three strongest discriminators of popular vs unpopular; groups differed significantly on 11 of 21 features |
| **Number of images in the README** | same | Also among the three strongest discriminators |
| **Presence of a license** | same | Third strongest discriminator |
| Lists, images, external links; contribution guidelines and references | Venigalla & Chimalakonda (arXiv:2206.10772), 1,950 READMEs, ten languages | Popular projects' READMEs are "well organised using lists and images, and comprise links to external sources"; contribution guidelines and references "associated with higher popularity" |
| **A results table with the exact commands that produce it** | Papers with Code ML Code Completeness Checklist, >200 repos, validated on NeurIPS 2019 | Repos with all five checklist items led with "median of 196 and mean of 2,664 stars" |
| Star count as a gate on adoption | Borges & Valente, *JSS* 2018 (arXiv:1811.07643), 791 developers surveyed + top-5,000 repos | "three out of four developers consider the number of stars before using or contributing to a GitHub project" — and the paper explicitly warns about "the risks faced when selecting projects by GitHub stars" |
| Content categories readers look for | Prana et al. (arXiv:1802.06997), 4,226 sections from 393 repos, hand-annotated | "What" and "How" dominate; **purpose and status information is frequently missing** — i.e. the gap is *why this exists* and *is it alive* |
| Machine-checked trust signals third parties compute about you | OpenSSF Scorecard checks | License at top level and OSI-named; `Maintained` scores highest at roughly weekly commits over 90 days; `CI-Tests`, `Signed-Releases`, `Security-Policy`, `Packaging` all separately scored |
| Launch-day traffic is a spike, not a trend | arXiv:2511.04453, 138 AI-tool launches 2024–25 | Mean +121 stars in 24 h, +189 in 48 h, +289 in a week; posting time dominated, and the "Show HN" tag "shows no statistical advantage after controlling for other factors" |
| Stars ≠ conversion | arXiv:2607.02453, 15 agent frameworks, 808,042 stars | AutoGPT: 111,967 stars in a month, <9 contributors per 1,000 stars; LangChain: 41. "headline popularity is unreliable" |

**Read against memra.** Two of the three measured discriminators (links out, license) memra already satisfies. **Images is the one it does not.** memra's README has six shields and no figure. This is not an argument for decoration — it is an argument for exactly one artifact that is unfakeable and legible in 2 seconds: a plot of the interleaved memra-vs-llama.cpp medians generated by the same script that writes the `PERF-SAMPLES` block, or a terminal capture of `run-gen` streaming with the tok/s line visible. Generated, dated, and regenerable; not a logo.

### 1.4 What the first 10 seconds must answer

Open Source Guides states the four questions a README must answer: "What does this project do?", "Why is this project useful?", "How do I get started?", "Where can I get more help, if I need it?" For an inference engine facing entrenched competition, that set is insufficient. The real first-screen question set is six:

1. **What is it?** (inference engine, Rust + CUDA, OpenAI-compatible)
2. **Will it run on my card?** — the disqualifier. Answer it before anything else, because a reader on an A100 should leave in 5 seconds and not resent you.
3. **Is it faster than what I run now, and under what conditions?**
4. **Is it alive?** (Prana et al.'s missing "status")
5. **What is the one command?**
6. **Why should I believe the numbers?**

memra's `What / Tuned for / Format / Shape / Author / Licence` table answers 1, 2, 4 (partly) and 6 (via Author) in one scannable unit. It is the strongest element in the file and it currently sits **below** a six-line blockquote. See [§8.1](#81-the-first-screen-specific-recommended-change).

---

## Part 2 — How a solo maintainer earns credibility without overclaiming

### 2.1 The asymmetry, stated plainly

You cannot borrow the four credibility engines the big projects run on:

| Big-project credibility engine | Verbatim example (fetched 2026-08-17) | Available to memra? |
|---|---|---|
| Institutional origin | vLLM: PagedAttention bibtex + UC Berkeley lineage | No |
| Contributor mass | vLLM: "over 2000 contributors" | No |
| Deployment scale | SGLang: "generating trillions of tokens in production each day"; "over 400,000 GPUs worldwide"; "has become the de facto industry standard" | No |
| Vendor authority | TensorRT-LLM: NVIDIA's name, dated first-party news items | No |

What is available, and what the evidence says actually converts, is **falsifiability**. A single-maintainer project's advantage is that it can make claims small enough to be checked, and can afford to publish losses that a vendor's legal and marketing review would strip.

### 2.2 What reads as trustworthy: verified patterns

| # | Pattern | Verified instance | Why it works |
|---|---|---|---|
| T1 | **Publish the arm that lost** | mistral.rs's four benchmark tables show it losing multiple BF16 rows to vLLM (e.g. 26B-A4B prefill 592.2 vs 3878.6 on GB10) | Selective reporting is Heiser crime 1.1 ("Not evaluating potential performance degradation"). Voluntarily reporting the loss is the only cheap proof you did not subset. |
| T2 | **A heading that tells people not to use it** | ripgrep: `## Why shouldn't I use ripgrep?` — four disqualifiers, first being non-portability: "The best tool for this job is good old grep" | Costs a paragraph, buys the reader's whole model of your honesty. Zero large inference projects do this. |
| T3 | **Limits inline with the feature they limit** | ExLlamaV3: "expect that some things may be a little broken at first"; "`n_group>1` currently not supported"; "though it still needs some work to achieve the same efficiency on Ampere GPUs". flash-attention: "Requirements: H100 / H800 GPU, CUDA >= 12.3", "Sliding window attention is currently a work in progress", "Note: Does not support backward pass." | The reader discovers the limit while still trusting you, not after wasting an afternoon. |
| T4 | **Refuse the compatibility claim your architecture cannot honour** | PowerInfer: "supports inference with llama.cpp's model weights for compatibility purposes, but there will be no performance gain"; "Now we only support models with ReLU/ReGLU/Squared ReLU activation function"; on 70B, "This insufficient retraining has resulted in the model's inability to regain its original performance" | Explicitly disclaiming a *quality regression in your own artifact* is the strongest honesty signal in this survey. |
| T5 | **Status banner when status changed** | TGI opens with a caution admonition: "now in maintenance mode," and points readers to vLLM, SGLang, llama.cpp / MLX | Answers Prana et al.'s missing "status" in the first line. |
| T6 | **Generated content for anything that drifts** | mistral.rs's supported-model reference is the "single source of truth," generated from the loader registry so it does not drift. memra's `PERF-SAMPLES` block already does this. | Removes the reader's fear that the numbers are from a good day two versions ago. |
| T7 | **Reject-on-unbenchmarked-claims as a stated policy** | tinygrad's contributing rules: speedup claims "must be benchmarked" | Publishing your own evidence bar tells a reader the numbers upstream of it were held to it. |
| T8 | **Complexity budget stated as a refusal** | llm.c: for a PR trading 500 complex lines for 2% speed, "I may reject the PR because the complexity is not worth it" | Signals that the codebase a reader is about to trust is governed, not accreted. |

### 2.3 What reads as marketing

| Anti-signal | Named source of the objection | Verbatim / concrete form |
|---|---|---|
| Superlatives without a number | SIGPLAN checklist §1: "Claims not explicit", "Claims not appropriately scoped" | "blazing fast", "state-of-the-art", "production-ready", "world-class" |
| Speedup ratios without absolutes | Heiser 5.3 "Relative numbers only" | "2.3x" with no tok/s next to it |
| Speedup without a stated baseline configuration | Heiser 4.3 "Unfair benchmarking of competitors"; 4.4 "Inflating gains by not comparing against the state of the art" | Any competitor column whose build/flags are not published |
| Adoption numbers you cannot substantiate | Schaeffer, Kazdan & Denisov-Blanch 2025 (arXiv:2506.13681): min-p's "community adoption claims (49k GitHub repositories, 1.1M GitHub stars) were found to be unsubstantiated, leading to their removal," and "the revised adoption claim remains misleading" | Star counts, "used by", "trusted by" |
| Attributing gains to the mechanism you want credit for | Lipton & Steinhardt, ICML 2018 (arXiv:1807.03341): "failure to identify the sources of empirical gains", e.g. "emphasizing unnecessary modifications to neural architectures when gains actually stem from hyper-parameter tuning" | "our fused kernel gives 1.3x" when the win was a launch-config change |
| Language borrowed for connotation | same: "misuse of language, e.g., by choosing terms of art with colloquial connotations" | "exact", "verified", "certified", "guaranteed" used loosely — see below |
| A win over a weak baseline | Dacrema, Cremonesi & Jannach, RecSys 2019 (arXiv:1907.06902): of 18 methods only 7 reproduced, and 6 of those "can often be outperformed with comparably simple heuristic methods" | Beating an untuned competitor and not saying so |

**The "exact" trap, specific to memra.** memra's differentiator language includes *exactness gates* — "8/8 byte-identical", "zero differing logits at T=1..4, K=1/3/8". That is a genuine and rare claim; it is also exactly the kind of term Lipton & Steinhardt flag as easy to erode. Protect it: the word *exact* in memra's README must always be attached to (a) the comparison target (plain decode, same request), (b) the count of requests or logits compared, and (c) the tolerance (bit-identical, or a named cosine floor as in the vision oracle's "images min-cos 0.9997, video 0.99999"). memra currently does this. It is one careless sentence away from not doing it, at which point the whole differentiator is worth nothing.

**Why byte-exactness is a legible differentiator right now.** Thinking Machines' "Defeating Nondeterminism in LLM Inference" established for a wide audience that serving nondeterminism is not floating-point noise but a *batch-invariance* failure — "our forward pass lacks 'batch invariance'", such that a request's output depends on the batch it landed in. Their demonstration: 1000 temperature-0 completions of one prompt produced **80 unique outputs**, first diverging at token 103; with batch-invariant kernels "all of our 1000 completions are identical," at a cost of 26 s → 42 s. That article is the reason a reader in 2026 understands, without explanation, why "speculative/graphed/batched serving proven byte-identical to plain decode per request" is a hard thing to say. memra's README should state the property in one sentence and link the gate, not explain the background — the background is now common knowledge and explaining it wastes the 10-second budget.

### 2.4 The single-maintainer signals that *do* scale

| Signal | Mechanism | Where memra stands |
|---|---|---|
| Named author, reachable | A person is more falsifiable than an org | Present in the table and the blockquote |
| Release cadence | OpenSSF `Maintained`: highest score at ~weekly commits over 90 days | Strong; README says "`main` runs ahead" of the tag, which pre-empts the staleness question |
| Refusal to version-pin prose | "A version number in prose is stale the day after it is written, so this file does not repeat one" | Keep this sentence. It is a maintenance-discipline signal in one line |
| Decision record | `docs/decisions/` — "why a default, format, target or arm was chosen — and what was rejected, with the measurement that settled it" | Rare, and worth more prominence than a row in a docs table |
| Adversarial results published anyway | The `MEMRA_SERVE_SPEC=0` result: spec decoding "cost 4x" on the shared-prefix serving shape, and "the arm expected to win lost" | Keep verbatim. This is the highest-value paragraph in the current README |
| Hardware-validation issue template | `.github/ISSUE_TEMPLATE/hardware-validation.md` | Converts a reader's card into evidence; underused, mentioned only in Contributing |

---

## Part 3 — Framing a deliberately narrow scope as the advantage

### 3.1 The two framings, and which one converts

There are exactly two ways to say "narrow", and they perform very differently:

- **Apologetic** — "only supports X for now", "limited to", "does not yet". Reads as an incomplete version of a general project. Invites the comparison you lose.
- **Consequential** — narrowness is the *cause* of a property the reader wants. Reads as a design, and the general projects cannot copy it without giving up their breadth.

memra's consequential form already exists in the file and is the best sentence in it:

> "Neither is tuned at the other's expense: where a mechanism wins on one and loses on the other it becomes a per-device default, so a naked command runs at full speed on whichever card it lands on."

That is the whole argument: *breadth forces compromise defaults; we don't have breadth, so we don't compromise.* It is currently on line 32–34, below the fold. It belongs in the first screen.

### 3.2 Real precedents, with the phrasing they used

| Project | The narrowness | README phrasing (verbatim, fetched 2026-08-17) | What the narrowness bought |
|---|---|---|---|
| **ripgrep** | One job, no POSIX conformance | `## Why shouldn't I use ripgrep?` → "The best tool for this job is good old grep" | Permission to be faster and defaults-opinionated (gitignore-aware by default) |
| **llm.c** (Karpathy) | Two languages, one model family | "LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch or 107MB of cPython"; "I'd like this repo to only maintain C and CUDA code"; ports "should be done in separate repos" | Readability as the product; a notable-forks list instead of a portability burden |
| **PowerInfer** | One consumer GPU + CPU, ReLU-family models only | "PowerInfer is a CPU/GPU LLM inference engine leveraging **activation locality** for your device"; "Now we only support models with ReLU/ReGLU/Squared ReLU activation function" | A defensible 11x over llama.cpp on Falcon-40B that a general engine structurally cannot claim |
| **KTransformers** | CPU–GPU heterogeneous MoE only | "a research project focused on efficient inference and fine-tuning of large language models" "through CPU-GPU heterogeneous computing"; "Heterogeneous expert placement (hot experts on GPU, cold experts on CPU)" | Positions as a *complement* ("Clean Python API for SGLang and other frameworks"), not a competitor — sidesteps the comparison entirely |
| **ExLlamaV3** | Consumer GPUs, one quant format | "an inference library for running local LLMs on modern consumer GPUs"; `## What's missing?` | Quantization cost framing that wins on a different axis: a rival method needs "around **720 GPU-hours**" (~$850) vs minutes-to-hours on one GPU |
| **nano-vllm** | ~1,200 lines, readability | "A lightweight vLLM implementation built from scratch"; "Clean implementation in ~ 1,200 lines of Python code" | Beat vLLM on one honest small-scale cell (1434.13 vs 1361.84 tok/s, RTX 4070 Laptop 8GB, Qwen3-0.6B, 256 seqs) |
| **tinygrad** | 15 primitive ops; "intentionally tiny and hackable" | "For something between PyTorch and karpathy/micrograd"; a new backend needs only "~25 low level ops" | Hackability as the product; ~25-op backend porting cost is the payoff |
| **luminal** | 15 primitives, AOT-only | core "is and always will be minimal"; "No indirections or abstractions, compatability layers, docker containers, or virtual environments"; "The best heuristic is no heuristic" | A claim general frameworks can't make: search finds Flash-Attention-class optimizations without hand-written kernels |
| **esbuild** | One job, no cache needed | "Extreme speed without needing a cache" | 10–100x framing against an entire tool category |
| **uv** | Python packaging only | "An extremely fast Python package and project manager, written in Rust" | "10-100x faster" than pip, hyperlinked to `BENCHMARKS.md` |

Two structural lessons from the table:

1. **KTransformers' move is the one memra should consider hardest.** By framing itself as a complement with a clean integration API, it never has to win a head-to-head against SGLang. memra's analogue is *not* to become a plugin — it is to make the scope sentence do the same work: memra is what you run on these two cards, not what you run instead of vLLM on a fleet.
2. **Everyone in this table names the alternative.** ripgrep names grep. llm.c names its forks. ExLlamaV3 names TabbyAPI. PowerInfer names llama.cpp as its baseline. Naming the alternative is what makes narrowness read as a choice.

### 3.3 The phrasing pattern that works

The observed shape is a three-part sentence:

> **[Use it]** if `<the reader's exact situation>`. **[Because]** `<the property narrowness causes>`. **[Look elsewhere]** if `<the disqualifier>` — `<named project>` is better at that.

memra has all three parts, split across lines 36–38 and buried mid-paragraph. Give the third part a heading. That single change is the highest-yield credibility edit available, per T2.

### 3.4 What *not* to do with narrowness

- Do not call it "focused" or "opinionated" without stating the consequence. Those words are cost-free and therefore worthless.
- Do not list the hardware you don't support as a matrix of ⚠️ rows. Two sentences beat a 7-row table with one populated column (see [§0.2](#02-where-it-does-not-fit-a-narrow-single-maintainer-rustcuda-engine)).
- Do not promise the narrowness is temporary. "Tensor parallel, P2P and 3-stage pipeline parallel are being built now — named here as unfinished rather than listed as features" is the correct form: present tense, no date, explicitly not a feature claim.
- Do not let the roadmap outgrow the shipped list. luminal's roadmap is longer than its results; the effect is a project that reads as promise-weighted. memra's current ratio is healthy — keep unfinished work to one short paragraph.

---

## Part 4 — Numbers: where they go, and how to make them read as true

### 4.1 The standards, in one place

Three bodies have written down what a performance claim must carry. They agree, and none of them is optional reading for whoever edits memra's Speed section.

**Heiser, "Systems Benchmarking Crimes"** — the crimes that apply directly to a README:

| Crime | Name (verbatim) | README form |
|---|---|---|
| 1.1 | "Not evaluating potential performance degradation" | Showing the win, not the regression |
| 1.2 | "Benchmark sub-setting without strong justification" | Three rows from a board of twenty, unlabelled as a subset |
| 1.3 | "Selective data set hiding deficiencies" | Stopping the concurrency axis right before the shed |
| 2.1 | "Pretending micro-benchmarks represent overall performance" | tg128 as a stand-in for serving |
| 2.4 | "No indication of significance of data" | No N, no range, no spread |
| 2.5 | "Arithmetic mean for averaging across benchmark scores" | Use the geometric mean for normalised ratios |
| 4.1 / 4.2 | "No proper baseline" / "Only evaluate against yourself" | Version-over-version only |
| 4.3 | "Unfair benchmarking of competitors" | Tuning yours, not theirs — or not saying how theirs was configured |
| 5.1 | "Missing specification of evaluation platform" | No card, driver, CUDA, OS |
| 5.2 | "Missing sub-benchmark results" | Headline only |
| 5.3 | "Relative numbers only" | Ratios with no absolutes |

Heiser's own closing prescription is the one memra already implements: "Make your benchmarking rig part of your regression testing suite."

**SIGPLAN Empirical Evaluation Checklist** (Berger, Blackburn, Hauswirth, Hicks; ACM SIGPLAN EC, Oct 2018) — 7 categories, 22 items. The four categories that govern a README:

- *Clearly Stated Claims*: "Claims not explicit", "Claims not appropriately scoped", "Fails to acknowledge limitations"
- *Suitable Comparison*: "Fails to compare against appropriate baseline", "Comparison is unfair"
- *Adequate Data Analysis*: "Insufficient number of trials", "Inappropriate summary statistics", "No data distribution reported"
- *Appropriate Presentation of Results*: "Misleading summary of results", "Inappropriately truncated axes", "Ratios plotted incorrectly", "Inappropriate level of precision"

Note "Inappropriate level of precision." A README reporting `0.156 s` and `238–245 tok/s` is honest; one reporting `0.1563 s` from three runs is not.

**MLPerf Inference rules** — the reporting discipline:

- "Results that cannot be replicated are not valid results."
- "The same system and framework must be used for a suite result or set of benchmark results reported in a single context."
- "SUT parameters and configuration must be uniquely and specifically named in the submission results."
- Prohibitions worth internalizing: "Truncating output tokens to boost performance or meet accuracy is not permitted"; sorting samples across dataset boundaries is disallowed; "Hard coding the total number of queries" is disallowed.
- And the framing that matters for a serving engine: latency-bounded throughput, not batch-1 latency, is the industry metric — NVIDIA, defending its own numbers, notes "Industry-standard benchmarks like MLPerf also measure performance with this fixed response time metric."

### 4.2 The falsifiability ladder, rebuilt on verified examples

| Tier | Form | Verified instance | Why the reader believes it |
|---|---|---|---|
| 1 (highest) | **Generated table + published raw runs + documented competitor setup + frozen-reference labelling** | memra's `PERF-SAMPLES` block: generated by `tools/update-perf-board.py` from `research/tune-data/current-board.json`, with per-run logs in `research/tune-data/` and competitor config in `docs/COMPETITOR-SETUP.md` | Nothing is hand-typed; the competitor's configuration is inspectable; the reference is dated and declared frozen |
| 2 | **Head-to-head table with stated workload + a link to a full report, including rows you lose** | mistral.rs: "Mean tokens per second across prompt lengths and decode depths from 128 to 16384 tokens.", "Decode uses 256 generated tokens.", `releases/v0.8.2/report.md` with "commands, model revisions, host metadata" | Conditions precede numbers; losses present; report is versioned |
| 3 | **Table bound to a named model + explicit hardware, never a bare headline** | KTransformers: 227.85 tok/s total / 87.58 tok/s output for DeepSeek-R1-0528 (FP8) on 8×L20 + Xeon Gold 6454S at "8-way concurrency" | You can tell whether it applies to you |
| 4 | **Single cell, fully specified, modest delta** | nano-vllm: RTX 4070 Laptop 8GB, Qwen3-0.6B, 256 sequences, I/O "Randomly sampled between 100–1024 tokens", 1434.13 vs vLLM 1361.84 tok/s on identical 133,966 output tokens | Small and checkable beats large and vague |
| 5 | **Conditional efficiency claim** | ExLlamaV3: "roughly memory-bound latency under optimal conditions (4bpw, RTX 4090)" | The hedge is doing real work |
| 6 | **Dated news item carrying its own conditions** | TensorRT-LLM: "TensorRT LLM can run Llama 4 at over 40,000 tokens per second on B200 GPUs!" — old items collapsed into `<details>` | Staleness is visible rather than hidden |
| 7 | **Ratio in a bullet, hyperlinked to methodology** | uv: "10-100x faster" where the phrase itself links to `BENCHMARKS.md` | One click to the evidence; no conditions on the page |
| 8 | **Chart image with a one-line condition and nothing else** | uv's hero bar chart, captioned only "with a warm cache" | Weak — no hardware, versions, or run counts inline |
| 9 | **Chart image with no caption, no numbers, no methodology link** | esbuild's README: an SVG bar chart whose alt text is "Bar chart with benchmark results", no caveats, no methodology link on the page | Works only because esbuild's speed is folklore by now. Not available to a new project |
| 10 (lowest) | **Prose superlative** | llama.cpp: "state-of-the-art performance," no tables anywhere | Works only from a position of total incumbency |

**The asymmetry to internalize:** llama.cpp and esbuild can operate at tiers 9–10 because their reputations precede the README. A new narrow project starts at tier 1 or it starts nowhere. memra is already at tier 1 — the job is not to improve the tier, it is to *not regress* when someone adds a number in a hurry.

### 4.3 How much goes above the fold

Observed distribution across 20 READMEs: **the median inference engine puts zero numbers in the README.** vLLM, SGLang, llama.cpp, Ollama, TGI, burn, candle: none. TensorRT-LLM: only inside dated news bullets. That is not a model to copy — it is what breadth costs. If your claim applies to 40 GPUs and 200 architectures, no table is true, so you write none.

memra's claim applies to two cards. A table is exactly the right instrument. The budget:

| Position | Contents | Line budget |
|---|---|---|
| First screen | Nothing but a **single anchor figure** with its card and condition inline — e.g. decode p50 on the flagship, one number, one card, one path | 1 line |
| First scroll | **Metrics table**: 5–7 rows max, each row carrying its own conditions, plus the *not measured* line | 12–15 lines |
| Second scroll | **Generated competitor samples** table + its methodology footnote + one counter-intuitive published result | 15–20 lines |
| Everything else | `docs/PERFORMANCE.md` | 0 |

That is ~35–40 lines of numbers in a ~270-line README: about 14%. memra's Speed section is currently ~90 lines (~33% of the file). See [§8.2](#82-the-speed-section-what-to-move).

### 4.4 The four things that must accompany every number

Derived from Heiser 5.1/2.4, SIGPLAN §4, MLPerf, and the memra house protocol:

1. **Rig** — card (and which of the two), driver/CUDA where it can change the answer.
2. **Workload** — model, quantization, path (safetensors/GGUF), context length, concurrency, prompt shape.
3. **N and spread** — number of reps and the observed range or median-of-medians. memra does this well already ("rep medians 138–141", "range 71–80"); the TTFT row currently has no N.
4. **What was not measured** — the single most under-used credibility instrument in the entire survey. Nobody does it. Heiser 1.1 and 1.2 and SIGPLAN's "Fails to acknowledge limitations" all demand it. Two lines of "not measured: multi-tenant mixed-model, >2-card, sustained >1 h at c=32, non-flagship models on `sm_90a`" buys more trust than a fourth favourable row.

### 4.5 Case file — benchmark presentations that damaged credibility

| Case | What happened | The transferable lesson |
|---|---|---|
| **AMD MI300X vs NVIDIA H100 (Dec 2023)** | AMD showed MI300X inference numbers against an H100 running vLLM v0.2.2.2 in FP16. NVIDIA's rebuttal: "The results shared did not use optimized software, and the H100, if benchmarked properly, is 2x faster" — re-running with TensorRT-LLM v0.5.0/v0.6.1 and FP8 (incl. FP8 KV cache), publishing both command lines | **Same hardware, different software config, 2x.** A competitor column is a claim about *their* software, not their silicon. If you cannot document how you configured theirs, you have not measured anything. memra's `docs/COMPETITOR-SETUP.md` is the mitigation; a competitor number added without updating it is a defect |
| **min-p sampling (arXiv:2506.13681, 2025)** | The paper's human evals "omitted data, conducted statistical tests incorrectly, and described qualitative feedback inaccurately"; benchmark advantage vanished "when controlling for the number of hyperparameters"; adoption claims of "49k GitHub repositories, 1.1M GitHub stars" were "unsubstantiated, leading to their removal", and the revised claim "remains misleading" | An unsubstantiated *adoption* number retroactively poisons the *technical* numbers. Never put a popularity figure in a README that also carries measurements |
| **MemPalace (arXiv:2604.21284, 2026)** | 47,000+ stars in two weeks; independent replication attributed the headline 96.6% Recall@5 to verbatim storage and the default embedding model rather than the advertised architecture. The critique names the pattern: "marketing velocity exceeds scientific rigor" | Lipton & Steinhardt's "failure to identify the sources of empirical gains" is not an academic nicety. If a memra win came from a launch-config change, say so — otherwise the first person who bisects it publishes that you didn't |
| **Deno excluding Bun from a SQLite benchmark (Aug 2022)** | Deno omitted Bun, characterizing it as a "demo" | Exclusions get read as fear. If a comparison omits a competitor, say why, in the footnote — Heiser 1.2 |
| **PrinceJS self-correction (Nov 2025)** | After HN feedback, the author retracted "fastest" claims: the load generator had been "too slow for Bun", switched to `oha`, republished | The recovery move is public, specific, and fast. A retraction with the tool named costs less credibility than a quiet edit |
| **Mojo's 35,000x / 68,000x Python claims (2023–24)** | The best-engaged follow-up post's own title is self-refuting: "Making CRC calculations in Mojo 18x faster than Python and 3x slower than Python" | A ratio against a deliberately weak baseline invites someone to publish the ratio that goes the other way. Heiser 4.4 |
| **Intel's commissioned benchmarks (2018, 2019)** | Two separate HN front-page stories (618 pts / 161 pts) about published comparisons slanted against AMD | First-party comparisons attract adversarial re-measurement. Assume yours will be re-run |
| **Ollama / llama.cpp license issue (#3185, 202 pts, 68 comments)** | An attribution/licence complaint against an upstream dependency became a top HN thread | Provenance is a trust surface. memra credits upstream nowhere in the README — see checklist item C6 |

### 4.6 The competitor column, done correctly

memra's current footnote is, as far as this survey found, better than any competitor's:

> *"Measured 2026-08-02 on the RTX 5090 Laptop — same-session interleaved medians, same exact prompts; memra at its naked defaults, llama.cpp at its swept best (docs/COMPETITOR-SETUP.md). The llama.cpp column is a frozen reference recorded through 2026-08-03 (benching stopped that day)."*

Four things it gets right that should be preserved verbatim in spirit:

1. **Interleaved same-session** — pre-empts box-drift, which is the standard silent invalidator of cross-run claims.
2. **"memra at its naked defaults, llama.cpp at its swept best"** — the exact inverse of Heiser 4.3. **Say why this matters, because the reader's prior is the opposite.** A one-clause addition — "(the handicap runs against us, deliberately)" — converts a footnote a skeptic skims into the sentence that wins them.
3. **Frozen reference, with the date benching stopped** — makes staleness a stated property rather than a discoverable flaw.
4. **Absolutes beside every ratio** — Heiser 5.3 satisfied.

One gap: the footnote gives the date and rig but no **N**. Add the rep count.

---

## Part 5 — What converts reading into cloning into serving

### 5.1 The shape of a quick start that works

Observed placement across 20 READMEs:

| Project | First user-facing command | Position |
|---|---|---|
| llama.cpp | `llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF` | `## Quick start` is the **first section**, before `## Description` |
| Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` | `## Download` is the first content heading; no badges, no feature list first |
| uv | standalone installer curl | Install precedes every example |
| mistral.rs | `curl \| sh` / `irm \| iex` | Third section, after benchmarks and "Why mistral.rs?" |
| TGI | `docker run … ghcr.io/…:3.3.5 --gpus all --shm-size 1g -p 8080:80` | Docker first, called "The easiest way of getting started" |
| vLLM | `uv pip install vllm` | First item under Getting Started; **no run example at all** |
| llm.c | `./dev/download_starter_pack.sh` → `make train_gpt2fp32cu` → `./train_gpt2fp32cu` | Immediately after framing |
| SGLang / TensorRT-LLM | none | **No code block anywhere in the README** |
| memra | release installer `curl \| sh`, then `run-gen` | `## Install` at ~line 44, `## Quick start` at ~line 71 |

**The pattern that converts is: install → run → see output, with nothing between the three.** memra satisfies this. Two refinements:

1. **Show the expected output.** Both makeareadme ("show the expected output if you can") and PLOS's Ten Simple Rules (Rule 3: "Include a quickstart guide" — users "may abandon your tool" otherwise) call for it, and llm.c/tinygrad/candle all do it (candle: `cargo run` prints `Tensor[[2, 4], f32]`; tinygrad: MNIST "gets 98% in ~5 seconds"). memra's `run-gen` block shows a command and no result. **One line of expected output — the first tokens plus the tok/s line — closes the loop between the Speed table and the reader's own terminal.** This is the single highest-value addition to memra's quick start, because it is the moment the reader can check your headline number themselves.
2. **State the time-to-first-success.** No inference engine in the survey does. burn states a compile time ("under 5 seconds"); tinygrad states "~5 seconds" for MNIST; nerfstudio-style "expected: 15 min" framing exists elsewhere. For memra: the release-installer path to first token is short and the source path is not. Saying "prebuilt: first token in under a minute; source build: expect a full CUDA 13.1 compile" sets expectation and removes the worst abandonment cause.

### 5.2 What to remove from a quick start

| Remove | Why | Evidence |
|---|---|---|
| Anything that is not needed for the first token | Every optional flag is a decision the reader must make before they have any reason to care | vLLM ships **one** line; llama.cpp ships two |
| Multiple model choices | Choice is friction at step one | llama.cpp names exactly one model; nano-vllm names one; memra correctly names a path, not a menu |
| Auth, aliases, drafters, tuning | Post-success concerns | memra already defers these to `docs/SERVING.md` — keep it that way |
| Build-from-source as the first path | The install path with the shortest failure surface goes first | memra: release installer first, `cargo build` second. Correct |
| Env-var soup | Each `MEMRA_*` in the quick start is a concept to learn | memra's quick start uses `MEMRA_CHAT`, `MODEL`, `MEMRA_MODELS` — three. That is near the ceiling; do not add a fourth |

### 5.3 Do Docker and one-liners matter for a CUDA project?

**Yes for the one-liner, conditionally for Docker.** The evidence is specific.

**Why the one-liner matters more here than anywhere else** — the actual friction a CUDA project imposes, taken verbatim from vLLM's own GPU installation page (the most thorough disclosure of CUDA install hazard in the survey):

- "vLLM contains pre-compiled C++ and CUDA (12.9) binaries"; alternate builds for 12.8 and 13.0
- Compiling kernels "introduces binary incompatibility with other CUDA versions and PyTorch versions" — "even for the same PyTorch version with different building configurations"
- "NVIDIA Blackwell GPUs (B200, GB200) require a minimum of CUDA 12.8"
- ROCm wheels exist only for Python 3.12; otherwise the installer "**will silently fall back**" to the CUDA wheel and fails with "`libcudart.so: cannot open shared object file`"
- Source builds need "GCC/G++ ≥ 11.3" because "PyTorch's C++20 headers are not compatible with GCC 10 or GCC < 11.3"
- CUDA 13 images need "an R580 or newer driver"; minimum host kernel Linux 4.15
- "vLLM does not support Windows natively"

Every one of those is a place a reader's first attempt dies. memra's structural advantage is that it is a **Rust binary with no Python ABI surface at all**, and its install section already states the four things that matter: Linux x86_64, glibc ≥ 2.35, driver ≥ 580, CUDA runtime libs, **and explicitly "They do not require `nvcc`."** That last clause is worth more than a Docker image, and it should be more prominent than it currently is — it is the sentence that tells a reader the CUDA-install death spiral does not apply.

On `curl | sh` specifically: the objection is well-rehearsed and weak. Tournoij's analysis is the standard rebuttal — "You're not running some random shell script from a random author, you're running it from a software vendor who you _already trust_ to run software", and "There is no fundamental difference between `curl .. | sh` versus cloning a repo and building it from source." He concedes package managers are "more secure due to checksums, signing, and auditing." memra's installer already "verifies the release checksum," which answers the strongest form of the objection. Ollama, mistral.rs and uv all ship `curl | sh`; it is now the dominant pattern.

**Docker: worth it, but not as the headline.** TGI calls it "The easiest way of getting started" — because TGI is Python-heavy and Docker is genuinely the shortest path there. memra is not, so a Docker image is a convenience for CI and multi-tenant deployment, not the front door. Recommended posture: mention it in one line under Install if an image exists, pointing at `docs/SERVING.md` for the `--gpus`/`--shm-size`/driver-floor detail. Do not lead with it, and do not add it to the first screen — llama.cpp lists Docker as option 2 of 4, which is the right prominence.

**What a CUDA project must disclose regardless of packaging** (all four already present in memra's Install; keep them together and do not let them drift apart): glibc floor, driver floor, whether `nvcc` is needed, and how architecture is selected (`MEMRA_CUDA_ARCH` as the documented override).

### 5.4 The install-instruction drift problem

Gao, Treude & Zahedi (arXiv:2312.03250) studied 1,163 README commits across 400 repositories touching installation sections, deriving six change categories: "pre-installation instructions, installation instructions, post-installation instructions, help information updates" plus "document presentation, and external resource management." Their template-augmented documents were judged "generally of better quality."

The operational reading: **installation sections drift more than any other section**, and the drift is usually in *pre*- and *post*-installation (prerequisites and verification), not the command itself. memra has strong pre-installation (glibc/driver/nvcc/arch) and **no post-installation verification step**. A `kernel-check` invocation with its expected output would close that gap — memra already ships the binary; the README never tells anyone to run it.

---

## Part 6 — Twenty READMEs: what each does in its first screen

All fetched 2026-08-17. "First screen" = everything before the first substantive content heading.

### 6.1 The comparison

| Project | First screen contains | Numbers in README? | Quick start | Scope / limits statement | For memra |
|---|---|---|---|---|---|
| **vLLM** | Logo, h3 tagline "Easy, fast, and cheap LLM serving for everyone", 6-link bar, one 🔥 news line. **No badges, no H1** | **None.** Only "over 2000 contributors", "200+ model architectures", "State-of-the-art serving throughput" | `uv pip install vllm` + 3 doc links; **no run example** | **None.** Scope implied by enumeration; problems routed to Issues/forum/Slack | **Reject** the numberless posture (it's what breadth costs) and the badge-free minimalism (memra needs the arch badge). **Take** the link-bar-in-first-screen and the total delegation of depth |
| **SGLang** | Logo, 6 badges (PyPI, downloads, license, issues, DeepWiki), 6-link bar, then `## News` before any description | Speedups only inside News headlines (25x, 3.8x/4.8x, 2.7x, 7x MLA); the `Benchmark and Performance` section has **no table** — one sentence: "Learn more in the release blogs" | **No code block anywhere** | None. Hedges only: "and more", "etc.", "most Hugging Face models" | **Reject** everything. News-before-description, unrestated speedups, scale-as-credibility ("over 400,000 GPUs worldwide"), a Benchmark section with no benchmark — all unavailable and all anti-patterns for a small project |
| **llama.cpp** | H1, cover image, bold tagline "LLM inference in C/C++", 5 badges, 7-link slash bar. **No news section** | **None.** "state-of-the-art performance" in prose | `## Quick start` is the **first section**; two commands (`llama cli -hf …`, `llama serve -hf …`) then screenshots | None. Goal statement only: "to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware" | **Take** Quick-start-before-Description, and the 18-row backend table with two "[In Progress]" markers as the model for honest per-target status. **Reject** the numberless prose claim |
| **TensorRT-LLM** | Centered title, h4 tagline, 8 badges (docs, python, cuda, torch, release, license), 5-link nav | Only in dated news bullets, each carrying its condition: "over 40,000 tokens per second on B200 GPUs"; older items collapsed into `<details>` | **None anywhere** | Via **Deprecation Policy** (3-month migration period) and **Telemetry** disclosure (on by default, opt-out documented) | **Take** the dated-item pattern for anything that will go stale, and the `<details>` archive. **Take** the idea of a written deprecation policy. **Reject** the no-command README |
| **Ollama** | Logo, H1, tagline "Start building with open models." then `## Download` immediately. **No badges** | **None** | `curl -fsSL … \| sh` is the first command | **None.** No requirements, no GPU matrix, no known issues | **Reject** the omissions. **Take** the ruthless first screen: tagline → download, nothing between |
| **TGI** (Rust) | Caution admonition: "now in maintenance mode" + pointers to vLLM/SGLang/llama.cpp/MLX; then video thumbnail, H1, 2 badges, tagline "A Rust, Python and gRPC server for text generation inference." | Essentially none; "~2x latency" for speculation | Docker first — "The easiest way of getting started" — then two curl examples (`/generate_stream`, `/v1/chat/completions`) | Caveats inline: CPU "subpar", AMD only MI210/MI250, Nix x86_64-only, unlisted architectures "on a best-effort basis". **No license section** | **Take** the status admonition pattern and the two-curl-examples shape (memra already does the second). **Take** "best-effort basis" as honest language for the unqualified tail |
| **mistral.rs** (Rust) | Banner GIF carrying tagline "Fast, flexible LLM inference.", 6-link bar, stars badge | **Yes — the best in the survey.** `<details>` "v0.8.2 CUDA benchmarks": conditions stated first, 4 head-to-head tables vs llama.cpp and vLLM, **including rows it loses**, plus `releases/v0.8.2/report.md` | Third section; `curl\|sh`, then a commented block of first-run commands, then port 1234 endpoints | Caveats scattered: Windows prebuilds CPU-only, cuTile needs `tileiras`, UI on by default, "not affiliated with Mistral AI" | **Take**: conditions-before-numbers, publishing losses, a linked versioned report, and the loader-registry-generated model list as "single source of truth". **Reject** hiding benchmarks in a collapsed `<details>` — memra's numbers are its argument |
| **ExLlamaV3** | Logo + name, one-line tagline "an inference library for running local LLMs on modern consumer GPUs", then "Headline features" bullets | Comparative cost framing (720 GPU-hours ≈ $850 vs minutes-hours on one GPU); conditional latency claim; a bpw quality chart; details deferred to `doc/exl3.md` | `## How to?` with three methods | **`## What's missing?` as a top-level heading.** Plus inline: "expect that some things may be a little broken at first", "`n_group>1` currently not supported" | **Take the `What's missing?` heading pattern.** **Take** the practice of winning on a different axis (cost, not just tok/s) |
| **PowerInfer** | Tagline "a CPU/GPU LLM inference engine leveraging **activation locality** for your device" | Per-rig tables with input length stated; baseline is always llama.cpp; explicit metric definition ("total prompting + generation time / total tokens generated") | — | The most honest limits section in the survey (see [§2.2](#22-what-reads-as-trustworthy-verified-patterns) T4), including a self-reported quality regression on 70B | **Take** the metric-definition line and the per-rig separation. **Take** the willingness to disclaim your own artifact's regression |
| **KTransformers** | Logo, h3 tagline, emoji nav bar, then a long reverse-chron "Updates" list | Tables bound to named model + explicit hardware + concurrency; softened scope ("in benchmarked MoE SFT workloads") | — | Soft — hedged scoping and maturity notes inside changelog entries | **Take** the complement-not-competitor positioning move ([§3.2](#32-real-precedents-with-the-phrasing-they-used)). **Reject** the changelog-as-limitations pattern |
| **nano-vllm** | Logo, Trendshift badge, title, tagline "A lightweight vLLM implementation built from scratch" | One fully specified cell vs vLLM (RTX 4070 Laptop 8GB, Qwen3-0.6B, 256 seqs, 100–1024 token I/O, identical 133,966 output tokens) | pip install → HF download → ~7-line Python snippet | Scope is the claim: "~ 1,200 lines of Python code" | **Take** the discipline: one cell, fully specified, modest delta, identical token count on both sides |
| **candle** (Rust) | H1, 5 badges, tagline "a minimalist ML framework for Rust with a focus on performance … and ease of use", then online demos | None | `## Get started`: install pointer, matmul snippet, **printed expected output** `Tensor[[2, 4], f32]`, and a one-line diff to switch to CUDA | `## FAQ` with "Why should I use Candle?" and a **Common Errors** section carrying verbatim error strings | **Take** the printed expected output and the **Common Errors section with literal error text** — memra has none, and its failure modes (arch mismatch, driver floor, missing runtime libs) are highly predictable |
| **burn** (Rust) | Logo, 7 badges, tagline "both a tensor library and a deep learning framework, optimized for numerical computing, training and inference." | One compile-time figure ("under 5 seconds"); throughput deferred entirely to an external `burn-bench` | Sixth section | `## Status`: active development, breaking changes, "no guarantees at this stage"; per-feature Beta labels; a boxed `recursion_limit` warning | **Take** the explicit Status section and per-feature maturity labels. **Reject** deferring *all* numbers externally — that works for a framework, not for an engine whose thesis is speed |
| **luminal** (Rust) | Banner "inference at the speed of light", 4 badges, then a Rust snippet | "~80% of theoretical max performance" for Q8 Llama 3 8B on H100; "fastest ML framework" stated as a goal, not a fact | Two commands, Llama 3 8B on CUDA | Roadmap as the limits list: no ROCm, "no public benchmarking suite yet", `hl_ops` covers "the most used ~80% of the pytorch api" | **Take** "stated as a goal, not a present fact" as the correct grammar for unfinished ambition. **Reject** letting the roadmap outweigh the results |
| **tinygrad** | Logo, positioning line "For something between PyTorch and karpathy/micrograd", 3 links, 3 badges | MNIST "98% in ~5 seconds"; 15 primitives; "~25 low level ops" per backend | Install from source | Contribution refusals as de-facto scope: "No code golf!", speedup claims "must be benchmarked", big diffs "won't be reviewed or merged" | **Take** the between-X-and-Y positioning sentence and the published evidence bar for contributions |
| **llm.c** | Tagline "LLMs in simple, pure C/CUDA with no need for 245MB of PyTorch or 107MB of cPython" | "~7% faster than PyTorch Nightly"; "~1,000 lines" | Exact command sequence, three variants (1 GPU fp32, manual, CPU) | "I'd like this repo to only maintain C and CUDA code"; CPU mode: "you won't go too far"; a stated complexity budget | **Take** the tagline structure (what it is + what it does not need) and the explicit language boundary |
| **ripgrep** (Rust) | Title, tagline paragraph that names competitors (ag/ack/grep), 3 badges, licence line | 5 tables (Tool/Command/Line count/Time), hardware disclosed (i9-12900K), corpora named, ripgrep bolded at `1.00x`; **tables ordered favourable → unfavourable, ending in "performance cliffs"**; preceded by "a single benchmark is never enough!" | After the "why" sections | **`## Why should I use ripgrep?` and `## Why shouldn't I use ripgrep?`** and `## Is it really faster than everything else?` ("Generally, yes," + 5 technical reasons) | **Take all three headings.** The favourable→unfavourable table ordering is the single most transferable presentation idea in the survey |
| **uv** (Rust) | H1, 3 badges, tagline, hero bar chart, italic caption naming the workload ("with a warm cache"), then Highlights | Chart (image) + "10-100x faster" hyperlinked to `BENCHMARKS.md` | Install before any example | None | **Take** the caption-under-the-chart discipline. **Reject** numbers-only-in-an-image |
| **esbuild** | Wordmark, 5-link bar | Bar chart image, alt text "Bar chart with benchmark results", **no numbers in text, no caveats, no methodology link** | Deferred to the site | None on the README | **Reject.** This only works from incumbency |
| **flash-attention** | — | Charts with conditions stated ("Head dimension 64 or 128, hidden dimension 2048", batch "16k / seqlen"), plus the honest pre-qualifier "speedup depends on memory bandwidth - we see more speedup on slower GPU memory" and "mostly on A100 GPUs" for test coverage | pip | **The reference for per-generation gating**: "Requirements: H100 / H800 GPU, CUDA >= 12.3"; "bf16 requires Ampere, Ada, or Hopper GPUs"; a struck-through obsolete restriction replaced by the version it changed in; "Sliding window attention is currently a work in progress"; "Note: Does not support backward pass." | **Take the per-generation feature gating format wholesale.** This is the closest existing model for how memra should express `sm_120a` vs `sm_90a` capability differences |

### 6.2 The five patterns worth stealing, ranked

1. **ripgrep's `Why shouldn't I use X?` heading** — the highest trust-per-line in the survey, and structurally unavailable to any project that wants to be everything.
2. **ripgrep's favourable→unfavourable table ordering, ending in the cliffs** — proves you did not subset (Heiser 1.2/1.3) by construction rather than by assertion.
3. **flash-attention's per-generation gating grammar** — exactly the instrument memra needs for `sm_120a` / `sm_90a` / `sm_89` capability differences.
4. **mistral.rs's conditions-before-numbers + linked versioned report + published losses.**
5. **candle's Common Errors section with literal error strings** — the cheapest reduction in support load and abandonment for a CUDA project.

### 6.3 The two-tier split for memra

| Stays in README | Lives in docs |
|---|---|
| One-line what/for-whom/on-what-card | `docs/MODELS.md` — full roster, per-card targets, drafter flavours, per-family architecture notes |
| The `What / Tuned for / Format / Shape / Author / Licence` table | `docs/PERFORMANCE.md` — full boards, methodology, thermal regime, N, open cells |
| The per-device-defaults sentence (the thesis) | `docs/SERVING.md` — request fields, capability gates, auth, cache semantics, admission, PP-2, runbooks |
| Install: 4 prerequisites + `nvcc`-not-required + one command | `docs/FLAGS.md` — the env-var catalog |
| Quick start: one generation with expected output, one server + one curl | `docs/COMPETITOR-SETUP.md` — how the competitor was built and swept |
| 5–7 metric rows with conditions + the *not measured* line | `docs/decisions/` — why a default/format/target/arm was chosen, and what was rejected |
| Generated competitor samples table + footnote | `research/` — raw per-run receipts |
| One counter-intuitive result that went against the project | `ARCHITECTURE.md`, `ARCHITECTURE-H100.md` |
| `Why you shouldn't use memra` + `What's missing` | `docs/RELEASING.md`, `CONTRIBUTING.md`, `docs/TESTING.md` |
| Docs table + Request a model + Contributing + Licence | GitHub Releases as the changelog |

---

## Part 7 — Anti-patterns, named

Each of these has a name so it can be cited in a review comment.

### 7.1 Structural

| Name | Description | Source of objection |
|---|---|---|
| **Badge racecar** | More badges than the reader can parse; each one dilutes the rest | Art of README: badges are "easy to abuse" and add visual noise. Tuch et al.: visual complexity degrades first impression within 50 ms. Counter-evidence: vLLM ships zero badges and is the most-used engine in the world |
| **News wall** | A reverse-chronological changelog above the description, so the first screen tells a returning reader what changed instead of telling a new reader what it is | SGLang puts `## News` before `## About`. GitHub Releases already is the changelog — memra correctly says so |
| **Feature bullet inflation** | A bullet list where each item is an unmeasured capability claim | SIGPLAN "Claims not appropriately scoped" |
| **Roadmap-heavy** | The list of what's coming is longer than the list of what works | luminal |
| **Empty-room signalling** | Contributor avatars, Discord badges, "join the community" on a project with one maintainer and no community | Inverted signal; the old guide recommended this |
| **Broken relative links** | Links that die on clone or after a file move | standard-readme: "Must not contain broken links." memra passes as of 2026-08-17 |
| **Sub-fold thesis** | The sentence that is the entire argument for the project sits below the first screen | memra's per-device-defaults sentence, currently line 32 |
| **Off-ramp above the fold** | A link to something other than using this project placed in the highest-value real estate | memra's hosted-instance blockquote |

### 7.2 Claim anti-patterns

| Name | Description | Source |
|---|---|---|
| **Naked superlative** | "blazing fast", "state-of-the-art", "production-ready" with no number and no condition | SIGPLAN §1 |
| **Ratio without absolute** | "2.3x" with no tok/s | Heiser 5.3 |
| **Ghost baseline** | A competitor number whose configuration is not published | Heiser 4.3; the AMD/NVIDIA case |
| **Self-only comparison** | Version-over-version wins presented as competitive standing | Heiser 4.2 |
| **Subset without a label** | Three good rows from a twenty-row board, not declared a subset | Heiser 1.2 |
| **Truncated axis** | Concurrency sweep stopping just before the shed | Heiser 1.3; SIGPLAN "Inappropriately truncated axes" |
| **Micro-as-macro** | tg128 presented as serving performance | Heiser 2.1 |
| **Precision theatre** | Four significant figures from three runs | SIGPLAN "Inappropriate level of precision" |
| **Format-as-support** | "supports GGUF" / "supports safetensors" as if a container implied a working model | memra's own README states the correct rule: "Support is specific to a model, quantization and drafter — never to a format." **This is a house rule with the force of a hard prohibition.** Loading is not support: each family has its own tensor census, quantization arithmetic, topology and gates |
| **Popularity in a numbers document** | Stars/downloads/"used by" adjacent to measurements | min-p case (arXiv:2506.13681) |
| **Unattributed gain** | Crediting the mechanism you're proud of for a win that came from elsewhere | Lipton & Steinhardt |
| **Eroded term of art** | "exact", "verified", "certified" used loosely | Lipton & Steinhardt, "misuse of language" |
| **Stale number with no date** | A table that was true once | TensorRT-LLM's mitigation: date every item, collapse old ones |
| **Undated competitor column** | A competitor number with no record of when it was taken or whether it is still being maintained | memra's mitigation — "frozen reference recorded through 2026-08-03" — is the correct form |

### 7.3 Onboarding anti-patterns

| Name | Description | Source |
|---|---|---|
| **Quick start that isn't** | A "quick start" requiring a source build, a conversion step, or a model hunt first | PLOS Rule 3: users "may abandon your tool" |
| **Output-free example** | A command with no expected output, so the reader cannot tell success from silence | makeareadme: "show the expected output if you can"; candle and llm.c both do it |
| **Undisclosed floors** | Missing glibc/driver/toolkit/OS minimums | vLLM's install page is the catalogue of what happens when they surprise you |
| **Silent-fallback packaging** | An installer that quietly installs the wrong artifact | vLLM ROCm: "**will silently fall back**" → `libcudart.so: cannot open shared object file` |
| **No verification step** | Nothing to run that says "your install is good" | Gao et al.: post-installation is one of the six drift categories; memra ships `kernel-check` and never tells anyone to run it |
| **Unattributed upstream** | No credit to the projects you build on | The Ollama/llama.cpp licence thread (202 pts). memra's README credits no upstream; llama.cpp and burn both carry Acknowledgements sections |
| **Missing status** | No signal whether the project is alive, experimental, or abandoned | Prana et al.: purpose and status are the most commonly missing categories. TGI's maintenance-mode admonition is the model |

---

## Part 8 — Concrete changes for memra's README

Current state: ~270 lines, jump bar, six badges, depth already moved to `docs/MODELS.md`, `docs/PERFORMANCE.md`, `docs/SERVING.md`. All relative links verified live. The file is already in the top decile of inference-engine READMEs on evidence discipline. What follows is ranked by expected effect per line changed.

### 8.1 The first screen: specific recommended change

**Problem.** The 10-second budget currently opens on a six-line blockquote whose content is (a) licence sentiment, (b) an off-ramp to a hosted commercial instance. The two elements that actually answer the reader's disqualifying question — the spec table and the per-device-defaults sentence — are second and fourth.

**Recommended order:**

1. H1 + one-line tagline. Structure it llm.c-style: *what it is + the constraint that makes it different.* e.g. "Inference engine in Rust + CUDA, tuned per device for RTX PRO 6000 Blackwell and RTX 5090 — no `nvcc`, no Python, no account."
2. **Badges, cut to four:** CI, licence, arch (`sm_120a + sm_90a`), tuned-for. Drop the standalone Rust and CUDA version shields — the CUDA version belongs in Install where it is actionable, and edition-2024 is not a reader-relevant fact. (Rationale: Tuch et al. on visual complexity; Art of README on badge abuse. Four is enough to signal maintained + legal + scope.)
3. **The spec table**, unchanged. It is the strongest element in the file.
4. **The thesis sentence**, promoted from line 32: per-device defaults, no compromise, naked command runs at full speed on whichever card it lands on.
5. **`Use it` / `Look elsewhere`** — keep, one line each.
6. Jump bar.
7. **The hosted-instance and lab links move** out of the first screen — into a short note after Quick start ("No card? …") and into the Author row of the table, which already carries them. Rationale: the first screen's job is to convert a reader who *has* the hardware; a paid off-ramp there reads as the project's purpose being lead generation, which is precisely the reading a solo-maintainer project can least afford.

Expected first screen after the change answers all six questions from [§1.4](#14-what-the-first-10-seconds-must-answer) in ~22 lines.

### 8.2 The Speed section: what to move

Currently ~90 lines (~33% of the file), containing nine distinct things. Target ~45 lines.

| Keep in README | Move to |
|---|---|
| Metrics table (7 rows) + **a new "not measured" line** | — |
| The exactness row, with its gate (`8/8 byte-identical; zero differing logits at T=1..4, K=1/3/8`) | — |
| Generated `PERF-SAMPLES` table + footnote + **rep count added** | — |
| The two counter-intuitive published results (prefix-cache depth 3.1x swing; spec costing 4x on cache-carried shapes) — **compress to one, keep the "the arm expected to win lost" sentence** | the other → `docs/PERFORMANCE.md` |
| One sentence on vision, with the parity-oracle floor | full ViT/oracle detail → `docs/MODELS.md` |
| One sentence that the trim moves acceptance, never output | trim mechanics, vocab arithmetic → `docs/MODELS.md` |
| — | **Drafter-flavours table (3 rows × 6 HF links)** → `docs/MODELS.md`. It is a post-adoption tuning choice; in the README it consumes ~12 lines of the highest-attention region for a decision no first-time reader makes |
| — | GDN-hybrid prefix-cache architecture note → `docs/MODELS.md` |
| — | Step-3.7-Flash / Qwen3.6 tuning status → keep one clause in `Which models run`, detail to `docs/MODELS.md` |

### 8.3 Additions, ranked by value per line

| # | Add | Lines | Why |
|---|---|---|---|
| A1 | **`## Why you shouldn't use memra`** — 3–4 bullets naming vLLM/SGLang/llama.cpp/TensorRT-LLM for the cases they win (broad coverage, datacenter fleets, non-NVIDIA, non-Linux, breadth of quantization formats) | 6 | ripgrep T2. Highest trust-per-line available. Converts the existing buried "Look elsewhere" clause into a checkable structure |
| A2 | **A `not measured` line under the metrics table** | 2 | Heiser 1.1/1.2, SIGPLAN "Fails to acknowledge limitations". Nobody in the survey does this |
| A3 | **Expected output in the quick start** — first tokens + the tok/s line | 4 | candle/llm.c/tinygrad precedent; PLOS Rule 3; closes the loop between your table and their terminal |
| A4 | **One generated figure** (interleaved medians plot or a `run-gen` terminal capture), produced by the same tooling as `PERF-SAMPLES` | 2 + asset | Images are one of the three measured discriminators of popular repos (Fan et al.). memra has none |
| A5 | **`kernel-check` verification step** with expected output, right after Install | 5 | Gao et al.'s post-installation drift category; the binary already ships |
| A6 | **`## Common problems`** — 4 literal error strings and their fixes (arch mismatch, driver < 580, glibc < 2.35, missing CUDA runtime libs) | 8 | candle's Common Errors pattern; these are memra's four predictable first-run failures |
| A7 | **A `sm_120a` vs `sm_90a` vs `sm_89` capability line** in flash-attention's gating grammar | 3 | The compile-gated H100 lane is currently a parenthetical; a reader on an H100 cannot tell what they get |
| A8 | **One clause on the deliberate handicap** in the competitor footnote — "(the handicap runs against us, deliberately)" | 1 | The reader's prior is that you tuned yours and not theirs. Say the opposite out loud |
| A9 | **Acknowledgements** — one line naming upstream projects the engine builds on or measures against | 2 | The Ollama/llama.cpp thread; llama.cpp and burn both do it |
| A10 | **A status line** — one sentence on maturity posture, alongside the existing "`main` runs ahead" note | 1 | Prana et al.: status is the most commonly missing README category |

Net effect: ~34 lines added, ~45 moved out. README lands at ~260 lines with a materially stronger first screen and three new trust instruments.

### 8.4 Do not do

- Do not add an OS × arch × accelerator support matrix. Two cards and a gated lane do not need a grid.
- Do not add downloads, stars, or "used by" anywhere.
- Do not add a News/Changelog section. Releases is the changelog and the README correctly says so.
- Do not restate a version number in prose. The existing sentence explaining why is worth keeping.
- Do not soften "Support is specific to a model, quantization and drafter — never to a format." That sentence is load-bearing and matches a standing house prohibition against inferring family support from loader or format success.
- Do not lead with Docker.
- Do not turn the unfinished-work paragraph (TP, P2P, 3-stage PP) into a roadmap section.

---

## Part 9 — REVIEW CHECKLIST: run this against the diff

Run before committing any change to `README.md`. Every item is answerable yes/no from the diff plus at most one file read. **Any NO blocks the commit or must be recorded in the commit body as a conscious exception.**

### Gate 0 — Triggers (which gates apply)

| If the diff touches… | Run |
|---|---|
| Any figure, unit, or ratio | N1–N9 |
| Any capability, feature, or support statement | C1–C6 |
| The first 30 rendered lines | F1–F6 |
| Install or Quick start | Q1–Q6 |
| A competitor name or column | X1–X5 |
| Structure, headings, or length | S1–S5 |
| Anything at all | Z1–Z4 |

### N — Numbers

- **N1.** Does every figure state, within 2 lines or in a linked doc named on the same line: **card**, **model + quantization**, **path** (safetensors/GGUF), **concurrency or batch**, and **context shape**?
- **N2.** Does it state **N** (reps) and a **spread** (range, or median-of-medians)? *(Heiser 2.4; SIGPLAN "No data distribution reported")*
- **N3.** Is every ratio accompanied by **both absolute numbers**? *(Heiser 5.3)*
- **N4.** Is the precision justified by the run count? No 4 significant figures from 3 runs. *(SIGPLAN "Inappropriate level of precision")*
- **N5.** If the change adds a favourable row, does the section still contain at least one **unfavourable or neutral** result? *(Heiser 1.1; ripgrep's favourable→unfavourable ordering)*
- **N6.** Is the **`not measured`** line present and still accurate after this change?
- **N7.** If a sweep is shown, does it extend past the point where the curve turns, or say where it stops and why? *(Heiser 1.3)*
- **N8.** Are raw receipts reachable — `research/tune-data/` or `docs/PERFORMANCE.md` — from within the same section?
- **N9.** If the number is inside `<!-- PERF-SAMPLES -->`, was it produced by `tools/update-perf-board.py` from `research/tune-data/current-board.json` and **not hand-edited**?
  ```bash
  # must produce no diff inside the markers
  python tools/update-perf-board.py && git diff --exit-code README.md
  ```

### C — Claims

- **C1.** No superlative without an adjacent number.
  ```bash
  git diff -U0 -- README.md | grep -nEi '^\+.*\b(blazing|lightning|fastest|world.?class|state.of.the.art|production.ready|revolutionary|seamless|effortless|unmatched|industry.leading|cutting.edge|best.in.class)\b'
  ```
- **C2.** No support claim keyed to a **format** rather than a (model, quantization, drafter) triple.
  ```bash
  git diff -U0 -- README.md | grep -nEi '^\+.*supports? (gguf|safetensors|nvfp4|fp8|awq|gptq)\b'
  ```
- **C3.** Does every use of **exact / byte-identical / verified / certified / guaranteed / parity** name the comparison target, the count compared, and the tolerance?
  ```bash
  git diff -U0 -- README.md | grep -nEi '^\+.*\b(exact|byte.identical|bit.identical|verified|certified|guarantee[ds]?|parity|proven)\b'
  ```
- **C4.** Is any new capability that is unfinished phrased as unfinished — present tense, no date, explicitly **not** a feature? *(luminal's "stated as a goal, not a present fact")*
- **C5.** Is any performance gain attributed to the mechanism that was actually measured to cause it? *(Lipton & Steinhardt)*
- **C6.** No popularity, adoption, or download figure anywhere in the file.
  ```bash
  git diff -U0 -- README.md | grep -nEi '^\+.*\b(stars?|downloads|users|adopted by|trusted by|used by [0-9])\b'
  ```

### F — First screen (first 30 rendered lines)

- **F1.** Do the first 30 lines answer: **what it is**, **which cards**, **is it alive**, **the one command or where it is**? *(NN/g 10-second budget)*
- **F2.** Is the per-device-defaults thesis sentence still within the first screen?
- **F3.** Badge count ≤ 5, and every badge resolves?
- **F4.** Is the highest-value real estate free of off-ramps to anything that is not using this repo?
- **F5.** Does line 1 of prose lead with information-carrying words, not a preamble? *(NN/g F-pattern)*
- **F6.** Is the first screen free of a News/changelog block? *(Releases is the changelog)*

### Q — Quick start and install

- **Q1.** Is the shortest-failure-surface install path still first (release installer before `cargo build`)?
- **Q2.** Are all four floors present and together: **Linux x86_64**, **glibc ≥ 2.35**, **driver ≥ 580**, **CUDA runtime libs** — plus the explicit **"do not require `nvcc`"**?
- **Q3.** Does the first runnable command still produce visible output within one screen, and is **expected output shown**?
- **Q4.** Are there ≤ 3 environment variables in the quick start?
- **Q5.** Is the verification step (`kernel-check`) present with its expected output?
- **Q6.** Were the commands actually run on a target rig at this commit? If not, the diff says so in the commit body.

### X — Competitor claims

- **X1.** Is the competitor's build and sweep documented in `docs/COMPETITOR-SETUP.md`, and was that file updated in this diff if the column changed?
- **X2.** Does the footnote state the **measurement date**, the **rig**, **interleaved same-session**, and **N**?
- **X3.** If the competitor reference is frozen, does the footnote say so and give the date benching stopped?
- **X4.** Is the handicap direction stated explicitly (memra at naked defaults vs competitor at swept best)? *(Heiser 4.3, inverted — say it out loud)*
- **X5.** If a competitor was omitted from a comparison, is the reason given? *(Heiser 1.2)*

### S — Structure

- **S1.** Total README length ≤ 300 lines.
  ```bash
  wc -l README.md
  ```
- **S2.** Is Speed ≤ ~50 lines and ≤ ~20% of the file?
  ```bash
  awk '/^## Speed/{f=1} /^## Which models run/{f=0} f' README.md | wc -l
  ```
- **S3.** Does `## Why you shouldn't use memra` (or its equivalent heading) still exist?
- **S4.** Is every new detail block one that a **first-time** reader needs? If it is a post-adoption tuning choice, it belongs in `docs/`.
- **S5.** Does every `##` heading still appear in the jump bar, and vice versa?

### Z — Always

- **Z1.** All relative links resolve.
  ```bash
  grep -oE '\]\(([^)#][^)]*)\)' README.md | sed -E 's/^\]\(//; s/\)$//' \
    | grep -v '^http' | while read -r p; do [ -e "${p%%#*}" ] || echo "BROKEN: $p"; done
  ```
- **Z2.** No version number in prose. *(the README states why; keep it true)*
  ```bash
  git diff -U0 -- README.md | grep -nE '^\+.*v?[0-9]+\.[0-9]+\.[0-9]+'
  ```
- **Z3.** Is the status/maturity signal still accurate ("`main` runs ahead of the tag", plus the maturity line)?
- **Z4.** If this diff changed a claim that `docs/PERFORMANCE.md`, `docs/MODELS.md`, `docs/SERVING.md`, or `docs/COMPETITOR-SETUP.md` also states, was that file updated in the same commit?

### One-shot verdict block

Paste into the commit body:

```
README review (agent-knowledge/readme-craft-inference-engine.md §9)
Gates run:  N C F Q X S Z      (mark n/a where untriggered)
NO items:   <none | id: reason>
Rig/commit for any new number:  <card> / <commit> / N=<reps>
Docs updated in this commit:    <files | none required>
```

---

## Sources

### Peer-reviewed and standards bodies

| # | Source | Type | Used for |
|---|---|---|---|
| 1 | Liu, White & Dumais, SIGIR '10, via [NN/g "How Long Do Users Stay on Web Pages?"](https://www.nngroup.com/articles/how-long-do-users-stay-on-web-pages/) | HCI study, 205,873 pages | §1.1 the 10-second budget |
| 2 | [NN/g "F-Shaped Pattern For Reading Web Content"](https://www.nngroup.com/articles/f-shaped-pattern-reading-web-content-discovered/) | Eyetracking, 232 users | §1.1 front-loading |
| 3 | Tuch, Presslaber, Stoecklin, Opwis & Bargas-Avila, *IJHCS* 70(11) 2012, [Google Research](https://research.google/pubs/the-role-of-visual-complexity-and-prototypicality-regarding-first-impression-of-websites-working-towards-understanding-aesthetic-judgments/) | Controlled experiment, 17–1000 ms | §1.1 pre-reading judgement |
| 4 | Fan et al., *EMSE* 2020, [arXiv:2010.02472](https://arxiv.org/abs/2010.02472) | 1,149 AI repos, 21 features | §1.3 measured discriminators |
| 5 | Venigalla & Chimalakonda, [arXiv:2206.10772](https://arxiv.org/abs/2206.10772) | 1,950 READMEs | §1.3 lists/images/links |
| 6 | Prana et al., [arXiv:1802.06997](https://arxiv.org/abs/1802.06997) | 4,226 sections, 393 repos | §1.3 missing purpose/status |
| 7 | Borges & Valente, *JSS* 2018, [arXiv:1811.07643](https://arxiv.org/abs/1811.07643) | 791 devs + top-5,000 repos | §1.3 stars as a gate |
| 8 | Gao, Treude & Zahedi, [arXiv:2312.03250](https://arxiv.org/abs/2312.03250) | 1,163 README commits, 400 repos | §5.4 install drift |
| 9 | Treude, Middleton & Atapattu, [arXiv:2007.10744](https://arxiv.org/abs/2007.10744) | 10-dimension quality framework | §0.3 doc quality |
| 10 | Lipton & Steinhardt, ICML 2018, [arXiv:1807.03341](https://arxiv.org/abs/1807.03341) | Position paper | §2.3 overclaiming modes |
| 11 | Dacrema, Cremonesi & Jannach, RecSys 2019, [arXiv:1907.06902](https://arxiv.org/abs/1907.06902) | 18 methods, 7 reproducible | §2.3 weak baselines |
| 12 | Schaeffer, Kazdan & Denisov-Blanch, [arXiv:2506.13681](https://arxiv.org/abs/2506.13681) | Critical replication | §2.3, §4.5 unsubstantiated adoption claims |
| 13 | [arXiv:2604.21284](https://arxiv.org/abs/2604.21284) (MemPalace critique) | Critical replication | §4.5 "marketing velocity exceeds scientific rigor" |
| 14 | [arXiv:2511.04453](https://arxiv.org/abs/2511.04453) | 138 launches | §1.3 launch-day spike |
| 15 | [arXiv:2607.02453](https://arxiv.org/abs/2607.02453) | 15 frameworks, 808k stars | §1.3 stars ≠ conversion |
| 16 | Heiser, ["Systems Benchmarking Crimes"](https://gernot-heiser.org/benchmarking-crimes.html) | Standards essay | §4.1, §7.2 |
| 17 | Berger, Blackburn, Hauswirth & Hicks, [SIGPLAN Empirical Evaluation Checklist](https://raw.githubusercontent.com/SIGPLAN/empirical-evaluation/master/checklist/checklist.yml) (7 categories, 22 items) | ACM SIGPLAN | §4.1, §7.2 |
| 18 | [MLPerf Inference Rules](https://raw.githubusercontent.com/mlcommons/inference_policies/master/inference_rules.adoc) | MLCommons | §4.1 reporting discipline |
| 19 | Lee, ["Ten simple rules for documenting scientific software"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006561), *PLoS Comput Biol* 14(12) 2018 | Peer-reviewed | §5.1, §7.3 |
| 20 | [arXiv:2101.08903](https://arxiv.org/abs/2101.08903) | 171 novice developers | §5 newcomer "finding a way to start" |
| 21 | Larios-Vargas et al., [arXiv:2005.12574](https://arxiv.org/abs/2005.12574) | 16 interviews + 115 devs, 26 factors | §1 selection is ad-hoc |
| 22 | [arXiv:1802.08391](https://arxiv.org/abs/1802.08391), [2502.18440](https://arxiv.org/abs/2502.18440), [2603.00331](https://arxiv.org/abs/2603.00331), [2607.15780](https://arxiv.org/abs/2607.15780), [2603.00489](https://arxiv.org/abs/2603.00489), [2607.21079](https://arxiv.org/abs/2607.21079) | README corpus/linting/maintenance studies (arXiv cs.SE sweep, 50 hits) | Background; §7 drift |

### Standards, guides, primary practitioner sources

| # | Source | Used for |
|---|---|---|
| 23 | [GitHub Docs — About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) | §1.2 mechanics, 500 KiB limit, Outline, relative links |
| 24 | [Art of README](https://github.com/hackergrrl/art-of-readme) | Cognitive funnel; "as short as it can be without being any shorter"; badge abuse |
| 25 | [standard-readme spec](https://github.com/RichardLitt/standard-readme/blob/main/spec.md) | Section order; "Must not contain broken links"; 120-char short description |
| 26 | [makeareadme.com](https://www.makeareadme.com/) | Section guidance; "show the expected output if you can" |
| 27 | [Open Source Guides — Starting a Project](https://opensource.guide/starting-a-project/) | The four README questions |
| 28 | [Changelog — "Top ten reasons why I won't use your open source project"](https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project) | Adoption blockers; unclear licensing |
| 29 | [liw.fi README review criteria](https://liw.fi/readme-review/) | Blurb; usage-not-manual; legal status |
| 30 | [Papers with Code — ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code) | §1.3 five items → median 196 stars |
| 31 | [OpenSSF Scorecard checks](https://github.com/ossf/scorecard/blob/main/docs/checks.md) | §1.3 machine-checked trust signals |
| 32 | [NVIDIA — H100 + TensorRT-LLM inference performance](https://developer.nvidia.com/blog/achieving-top-inference-performance-with-the-nvidia-h100-tensor-core-gpu-and-nvidia-tensorrt-llm/) | §4.5 the MI300X 2x software-config case |
| 33 | [Thinking Machines — Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | §2.3 why byte-exactness is legible; 1000 completions → 80 unique outputs |
| 34 | [vLLM GPU installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) | §5.3 the catalogue of CUDA install hazard |
| 35 | Tournoij, ["Curl to shell isn't so bad"](https://www.arp242.net/curl-to-sh.html) | §5.3 the `curl \| sh` objection and its limits |
| 36 | HN Algolia sweeps: README stories (>100 pts, 22 items); misleading-benchmark stories (>80 pts); `Ollama llama.cpp` (263 hits, incl. issue #3185 at 202 pts / 68 comments); `Bun benchmark` (1,008 hits); Mojo speed claims | §4.5 case file; §7 practitioner objections |

### READMEs analyzed live (2026-08-17)

| # | Project | Raw source |
|---|---|---|
| 37 | vLLM | `raw.githubusercontent.com/vllm-project/vllm/main/README.md` |
| 38 | SGLang | `sgl-project/sglang/main/README.md` |
| 39 | llama.cpp | `ggml-org/llama.cpp/master/README.md` |
| 40 | TensorRT-LLM | `NVIDIA/TensorRT-LLM/main/README.md` |
| 41 | Ollama | `ollama/ollama/main/README.md` |
| 42 | text-generation-inference (Rust) | `huggingface/text-generation-inference/main/README.md` |
| 43 | mistral.rs (Rust) | `EricLBuehler/mistral.rs/master/README.md` |
| 44 | ExLlamaV3 | `turboderp-org/exllamav3/master/README.md` |
| 45 | PowerInfer | `SJTU-IPADS/PowerInfer/main/README.md` |
| 46 | KTransformers | `kvcache-ai/ktransformers/main/README.md` |
| 47 | nano-vllm | `GeeeekExplorer/nano-vllm/main/README.md` |
| 48 | candle (Rust) | `huggingface/candle/main/README.md` |
| 49 | burn (Rust) | `tracel-ai/burn/main/README.md` |
| 50 | luminal (Rust) | `luminal-ai/luminal/main/README.md` |
| 51 | tinygrad | `tinygrad/tinygrad/master/README.md` |
| 52 | llm.c | `karpathy/llm.c/master/README.md` |
| 53 | ripgrep (Rust) | `BurntSushi/ripgrep/master/README.md` |
| 54 | uv (Rust) | `astral-sh/uv/main/README.md` |
| 55 | esbuild | `evanw/esbuild/main/README.md` |
| 56 | flash-attention | `Dao-AILab/flash-attention/main/README.md` |

---

*Synthesized from 47 distinct evaluated sources across 56 fetches. Source metadata with quality scores: [`resources/readme-craft-inference-engine-sources.json`](resources/readme-craft-inference-engine-sources.json). This guide supersedes `readme-technical-inference-repos.md`; see [Part 0](#part-0--audit-of-the-existing-guide).*
