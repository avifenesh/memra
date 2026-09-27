# OWED 20: above-RAM expert-bank artifact candidates (owner pick)

Written 2026-09-26 for the owner to pick on text. Nothing here is downloaded or pinned; the lead
pins the pick. The need: B3's storage-bound steady state was only reachable with a balloon
(regime iii) because the pinned 35B bank (15,600,713,728 bytes) fits every box's page cache. An
artifact whose expert bank exceeds host RAM makes the cold, warm and bounded regimes collapse
into one real one: every decode step past the resident set reads storage, with no balloon.

Reference RAM: the local 5090 rig has 65,390,510,080 bytes (MemTotal 63,857,920 kB) and
90.4 GB free on `/data`. A rented box's RAM varies by offer; "above RAM" is checked against
the chosen box at acceptance.

Sizes are the published file sizes. Expert-bank figures marked "est." scale the IQ4_XS bank
fraction (0.857) and are replaced by the tensor census at pin time.

| # | Candidate | File bytes | Expert bank | Above RAM on | Support state today | What it proves | What it does not |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3.6-35B-A3B BF16, same repo and revision as the pinned IQ4_XS (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238`, two shards) | 71,065,942,560 | about 61 GB est. | the local 5090 rig (bank near RAM, total above it); a box with 64 GB or less | not a memra support state: BF16 routed experts are a different numeric program and have no qualified GGUF path for this family | the same router and topology as every B3 number, so a storage-bound result compares directly with the balloon result | nothing on a box with 128 GB or more; needs its own qualification (tensor census, oracle, argmax) before any number counts; fills most of the 5090 rig's free `/data` |
| 2 | Qwen3.6-35B-A3B Q8_0 or UD-Q8_K_XL, same repo and revision | 37,801,097,504 / 39,099,447,584 | about 32 / 34 GB est. | none of the current rigs | not qualified for this family (IQ4_XS is the card's path) | a 2x larger bank on the same model | not above RAM anywhere we run: still needs the balloon, so it does not close OWED 20 |
| 3 | Step-3.7-Flash IQ4_XS GGUF (three shards, the model card's qualified PP-2 artifact) | about 105 GB (from the Step lane's shape census; 108.7 GB with the Q8_0 MTP head) | most of the file | the local 5090 rig and any box with under about 128 GB | Supported on the qualified GGUF PP-2 path; single-card expert spill is not a qualified surface | a real supported model whose bank exceeds RAM on common boxes; the spill arms measured on a model someone actually serves | does not fit the 5090 rig's free `/data` (90.4 GB); needs a PP-2 box, or a single-card spill qualification first; not comparable with the 35B B3 numbers |
| 4 | Hy3 all-expert ModelOpt W4A16 artifact (`docs/models/hy3.md`, 99 shards, sealed 108-file manifest) | 180,826,481,152 tensor payload | most of the payload | any box with under about 192 GB, which is every box offered so far | NativeQualified on four cards; spill is the Hy3 lane's subject, with `HostExps.layouts` and the five-arm study | the exact storage-to-compute pipeline CLAUDE.md names for large expert banks, on the model that needs it most | the Hy3 lane owns the artifact, the research machine and its campaign; F measuring there needs the owner to split the lane; a 181 GB stage per box |
| 5 | No new artifact: keep regime (iii), the touched balloon, as the storage-bound regime | 0 | 15.6 GB, cache bounded below half of it | any | the pinned IQ4_XS, qualified | already measured on BOX27 (bounded scored: worker16 beats every arm) | the balloon is a model of memory pressure, not a real above-RAM workload; readahead and reclaim behavior can differ from a true oversize bank |

Recommendation, for the owner to overrule: candidate 3 if the aim is a serving-real storage-bound
number, since it is a supported model and its bank exceeds RAM on the boxes we rent. Candidate 1
is the only one that compares directly with the existing B3 numbers, but it creates a new
unqualified program whose qualification is the larger part of the work. Candidate 2 does not
meet the requirement. Candidate 4 belongs to the Hy3 lane.
