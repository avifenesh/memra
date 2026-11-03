# Impact map

Qualification measures the delta. For any change, the impact map answers three questions:

1. What can the change affect: which model families, serving routes and devices, and how (output bytes, speed, memory, serving policy)?
2. What has already been measured for exactly that code?
3. What still needs measuring, and what is the smallest model x route x device run set that covers it?

The push gate (`tools/change_impact.py`, `tools/cell_evidence.py`) reads the map. The path rules in `tools/change-impact.toml` still decide a file's class. For Rust and CUDA program files, the map then narrows the answer from the items the diff actually changed.

## Files

| file | what it is |
|---|---|
| `map.json` | Machine-readable map, consumed by the gate. Every Rust and CUDA item (function, method, type, const, kernel) with its span, comment-free token hash, effect kinds and reach (the set of family x route x device cells that can execute it), plus the labeled predicate regions. Generated; do not hand-edit. |
| `annotations.json` | The semantic labels that the generator cannot derive mechanically: predicate regions (which families, routes and devices take each branch arm), file defaults, entry points, verified output-neutral items, dynamic edges, and the family / route / device tables. Each label has file:line evidence. |
| `models.json` | Families -> models, with the formats, routes and device classes each model has been measured on. The plan picks models from here. |
| `ledger.json` | The measurement ledger: measured cells (model, route, device, test) at a commit, with the receipt path and the cell fingerprint at that commit. |
| `history.md` | Real past diffs replayed through the delta query: the chosen test set for each of the last 40 main commits and for the open stacked PRs. |
| `findings.md` | Mergeable and reusable components, components that need decoupling, and defects found while mapping. Each has file:line evidence, a severity and a one-line fix. Nothing in it is fixed by this change. |
| `subsystems/*.md` | One write-up per slice of the repo: hot paths per family and route, device-specific parts, effect kinds, where the impact set is wide and why. |
| `scheduler.md` | How runs find a quiet window on each device class. |

## How the map is built

`python3 tools/impact_map/build.py --run-scip` regenerates `map.json` on the CPU. No kernel is compiled and no GPU is used. The steps:

1. **Items.** `rust_items.py` lexes every Rust file into items with line spans, `cfg` attributes, a test flag and a token hash. Comments and whitespace are excluded from the hash, so a comment-only or rustfmt-only edit keeps it. The scanner is checked against rust-analyzer's SCIP index: on 5ba8a3b11a, each of the 16,861 function definitions in the index lands in a scanned item with the same name. `cu_items.py` does the same for CUDA. Of the `__global__` kernels it finds, only 9 are absent from a real sm_120a fatbin, and all 9 are flag- or arch-gated (CUTLASS units, the FA3 stub, Hopper wgmma) plus one candidate dead kernel.
2. **References.** A rust-analyzer SCIP index (built with `DOCS_RS=1`, so no nvcc runs) gives precise symbol references. They become item -> item edges, recorded with every call line. A reference to a trait method fans out to every impl of that method. Code under an inactive cfg (CUTLASS, Hopper) gets name-based edges. Kernel edges come from Rust string literals that name a `__global__` symbol (`Engine::func("...")`). FFI edges link `extern "C"` declarations to CUDA host launchers, and launch edges link launchers to their kernels.
3. **Per-arch liveness.** `buildrs_capture.py` compiles `crates/memra-engine/build.rs` standalone and runs it once per build variant (120a, 120a + MEMRA_CUTLASS, 100a, 90a, 89) with a logging stand-in for nvcc. That captures the exact nvcc command lines, including stub substitution and per-arch `-D` flags. `cu_liveness.py` then runs only the preprocessor step of each command and reads the linemarkers. A CUDA source line is live for a build when the preprocessor emits it. This is how `#if __CUDA_ARCH__`, `MEMRA_PORTABLE_CUDA` and the stub units are attributed exactly. For example, `wgmma_common.cuh` has 15 lines live on every build and 54 on 90a.
4. **Reach.** Every served root starts with every cell: an item that no served item calls (mains, callbacks, entry points). Reach flows from caller to callee. An edge whose call line sits strictly inside an annotated region carries that region's family / route / device restriction. A region reachable only behind a default-OFF door carries none, because no default launch executes it. A trait edge to an impl carries the families of the impl's self type. Rust `cfg` flags and CUDA liveness restrict devices. Diagnostic and gate binaries are not served roots. An item that no root reaches takes its file default, or every cell: unknown is wide, never empty.

The annotations come from a bounded multi-agent pass. Agents only add what the scripts cannot decide:

- what a predicate means;
- which families, routes and devices take each arm;
- effect kinds;
- defects.

A script checks every claim against the tree: its quote must appear on the cited line. Claims that narrow reach (that exclude a family, route, device, or every default launch) were then checked by independent skeptic agents told to refute them, with "uncertain" counting as refuted. A refuted claim is widened back to every cell. The asymmetry is deliberate: a wrong exclusion lets a regression ship, while a wrong inclusion only costs a run.

## What the first build measured (code of 372a4cf284, main at 5ba8a3b11a)

| step | result |
|---|---|
| Rust items | 13,007 non-test functions, 29,240 nodes in total with types, consts and CUDA. Every one of the 16,861 function definitions in the SCIP index (tests included) lands in a scanned item of the same name |
| CUDA kernels | 1,042 `__global__` kernels scanned. All but 9 are present in a real sm_120a fatbin, and the 9 are flag- or arch-gated. 7 fatbin kernels are macro-generated; macro bodies are expanded into edges |
| Graph | 107,552 edges. Kernel names built with `format!` add 1,363 pattern edges, and macro-body launches are expanded. Kernels with no served caller dropped from 109 to 8, which are listed as dead-code candidates |
| Per-arch liveness | 5 build variants x 43 nvcc units, preprocessed in 25 s |
| Annotation | 22 slice agents and 3 scout agents (family, route and device tables) produced 1,710 labeled regions. The citation check pinned every region's quote; 19 regions span no item and were dropped. Regions over several items became 6,392 item-anchored regions |
| Skeptic pass on narrowing claims | 2,220 claims (regions, file defaults, output-neutral items, engine-type restrictions): 863 hold, 1,357 refuted. Refuted claims that narrowed families got a second check on the family restriction alone: 568 of 1,008 hold. The final map uses 2,017 regions as claimed and 1,487 at family level only, and widens 2,544 back to every cell. 146 of 180 output-neutral items and 9 of 10 engine-type restrictions hold |
| Skeptic pass on defects | 287 reports: 280 confirmed, 5 refuted, 2 unsure (`findings.md`) |
| Kernel traces | 6 short runs on an RTX 5090 (Qwen3.8-27B, Ornith-1.5-35B-A3B, Qwen3.5-9B and Gemma-4-31B; plain and MTP). Before the skeptic pass, 11 executed kernels were outside their run's reach, all from regions that labeled batched-verify paths as `batch` only. After the skeptic pass, 0. Executed kernels are also fed back as reach (`observed.json`) |
| Precision | 17,032 items. 77% of items (75% of lines) reach every family: the server, tokenizer, engine core and shared kernels. 15% reach exactly one family; 8% reach several. 12% are GPU-only and 2% run on a narrower device set |

The wide share is mostly real sharing, not missing labels. The server, the tokenizer and the `Engine` launch helpers run for every model. One exception is called out in `findings.md`: 459 items write a process-global static, which makes their change reach the static's whole reach.

## The delta query

`python3 tools/impact_map/delta.py --base origin/main --head HEAD` prints the reached cells and the run plan. `change_impact.py` and `cell_evidence.py need` print the same plan under the target cells.

- Each changed Rust or CUDA file is scanned at the base and at the head. An item has changed when its token hash moved. Every token outside an item counts too: inner attributes, `mod x;` and its cfg, extern-block headers, file-scope CUDA constants, and anything the scanners do not place. Those tokens form one synthetic `<outside>` item that the map never contains. A file in which nothing moved reaches no cell.
- An item counts as test-only only when every side that has it is a test item. A production item moved into `#[cfg(test)]` is a production change.
- A changed item the map knows takes the item's reach.
  - Its touched lines are matched against the item's regions only when its base copy is laid out exactly as at the map commit: same token hash, same token-line layout, same start line.
  - A change entirely inside narrowing arms then takes only those arms' cells.
  - The first line of a region (usually the branch condition) never narrows, because a changed condition moves every arm.
  - An item that writes process-global state keeps its readers' reach in every case.
- These items keep the path rule's answer in the gate:
  - an item the map does not know (new, renamed, or an added item whose id the map had before);
  - an item the map places nowhere by default (behind a build-time door);
  - `<outside>`;
  - a file under `cu/` the scanner does not read.
- Effects select tests:
  - `numeric`: calibrated `run-gen` argmax, plus `run-spec` K=1..8 for speculative routes;
  - `speed`: the interleaved perf A/B;
  - `memory`: the device memory qualification;
  - `policy`: serve-smoke;
  - CUDA or FFI: kernel-check.
- The plan picks one model per reached family and device from `models.json`, preferring the owner's own models. Reached cells that no available model covers are listed, never dropped.

**When the map is older than the base.** The commit that adds a new call carries its own reach into that push. A later edit of the callee alone would not. So before narrowing anything, the query scans what changed between the map commit and the base. Every item changed or added there contributes its identifiers. A changed item named by one of them takes the cells of that new or changed caller as well.

Narrowing stops in two cases:

- **The arch table moved** (`build.rs`, `cuda_arch.rs`, the toolchain or the root manifest) since the map: device narrowing stops.
- **The map's commit is not an ancestor of the base:** nothing narrows.

A change that only removes an edge leaves the map wider than the truth, never narrower. Edges the static graph cannot see (string-keyed or env-selected dispatch) are recorded in `annotations.json` as dynamic edges, which the generator adds as unconditional edges.

## The ledger

`ledger.json` keeps one row per (model, route, device, test): its latest receipt, preferring one that still holds. The row records the commit and receipt path, the cell fingerprint at that commit, and whether the fingerprint equals the map commit's.

The seed is mechanical. Records come from three sources, in order of trust:

1. `research/release-qualification/*/` records, which name their exact commit;
2. receipts that name a commit resolving in this repo (2,599 distinct commits among 171k receipt files);
3. the commit that added a receipt, as an upper bound.

Model, device and test kind come from patterns over the receipt's path and text, so a row says what a receipt mentions, not what it proves. The authoritative entries going forward are `tools/cell_evidence.py` records, which bind their commit, device and raw logs.

At the map commit no seeded row still holds. Two reasons cover it:

- The last receipts on main predate #1021, a tokenizer change that reaches every cell.
- The GB10 receipts measured branch commits whose GB10 changes are not on main.

`python3 tools/impact_map/ledger.py status` re-checks every row against any ref.

## In the push gate

The map in force is the push base's, like the rules, so a change cannot narrow its own requirements. Only the change that introduces the map uses the head's.

Safety rules:

- A push that changes `map.json` must leave a loadable map, or the gate refuses it.
- A base map that cannot be loaded leaves the path rules alone in force.
- `map.json`, `annotations.json` and the non-test code under `tools/impact_map/` are gate-class: a change to them needs the full evidence a rules change needs.
- No environment variable replaces the map. What-if replays (`history.py`) pass a map explicitly and are never a push decision.

For each program file of class kernel, numeric or serving that is not flagged for device-keyed defaults:

- **Token-identical edit:** the file is reclassified `comment` and reaches no cell. An edit only to test items makes it `tests`.
- **Every changed item placed by the map:**
  - the file's build targets are intersected with the targets of the devices those items reach (not when the arch table moved since the map);
  - the perf cell is dropped when no changed item has a speed effect;
  - a numeric file whose items reach only families that no battery roster model belongs to is reported UNCOVERED for those families, as #1030 already did for pack-local changes. The battery cannot see such a change; those families' own gates are the evidence.
- **Otherwise:** the path rule's answer stands.

**Delta against the last measured commit.** `cell_evidence.py` admits a record measured at an older commit when four things hold:

- its tested commit is an ancestor of the candidate;
- the record still verifies against its own commit;
- it is the newest such record for its target and device class;
- the change impact from that commit to the candidate reaches no cell of the record's target in a scope the record's kinds cover.

That change impact is judged with the push base's rules and map. A rules or map edit inside the pushed range therefore cannot relax it. A perf record also needs the push base to be equivalent to the record's base on that target.

## Refreshing

- Run `build.py` after merges that move many items. Regions are re-anchored through their items: by token hash when the item is unchanged, otherwise by searching for the evidence quote inside the item. A region that cannot be re-anchored is dropped, which only widens the map.
- When a subsystem is rewritten, re-annotate its slice: `gen_facts.py` writes the agent fact sheets, `ingest.py` turns agent output plus skeptic verdicts into `annotations.json`, then `assemble.py` builds the map.
- `tools/impact_map/test_delta.py` covers the delta logic on a synthetic repository.
