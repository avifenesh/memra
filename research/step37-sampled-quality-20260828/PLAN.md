# Sampled-quality cell: does MEMRA_STEP_GEMM_PRIME_SUFFIX ship ON for Step-3.7-Flash?

Bytes are settled (gemm-suffix lane VERDICT.md: every prime path is m-dependent under a
decomposition fork). This cell answers the owner's remaining question: do the warm paths
produce WORSE ANSWERS, sampled, vendor-default?

## Box and pins

dev box (devbox2) <devbox2, see darklanes research/cloud-devbox2-20260828> (<devbox2-ip>, 2x RTX PRO 6000 Blackwell Server, SPOT).
Source /home/ubuntu/memra @ lane/step37-main-merge-20260828 tip 8695bdef4a.
Binary /home/ubuntu/memra/target/release/memra-server md5 f45c3623d958ca085eefd3207987812a
(verified before first boot; printed into every results header).
Model /root/models/step37-flash-nvfp4 (ephemeral NVMe). GPU lock /root/gemmprime.lock.

Serving env: ENVV from /root/agentic8.sh + MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3
MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1.
Vendor-default sampled: NO temperature/top_p in any evaluated payload. max_tokens=1024.

## Conversation

8-turn growing conversation: turn 1 user = /root/curve-1000.json content, turns 2..8 user
= /root/agentic8.json strings 1..7 (idx 0..6). Canonical assistant replies A1..A7 are
generated ONCE (cold, fresh session per turn, vendor-default sampled, max_tokens 1024,
reply text = content if non-empty else reasoning, truncated to 4000 chars per the
agentic8.sh convention) and REUSED as the fixed transcript by every arm and sample.
Evaluated turns: 4 (mid context) and 8 (deep context), 8 samples per arm per turn.

## Arms

- COLD: fresh session_id, full conversation in one request, first requests on a fresh
  boot. Validity: no rewind, eng_fresh>0, eng_suffix=0, walk_suffix=0.
- WARM-GEMM (MEMRA_STEP_GEMM_PRIME_SUFFIX=1): conversation replayed sequentially on one
  session_id (prefix turns max_tokens=64, replies discarded; the transcript stays the
  canonical one). Validity: rewound=True on the evaluated turn and [gemm-prime] ENGAGED
  base>0 present, walk_suffix=0.
- WARM-WALK (MEMRA_STEP_GEMM_PRIME_SUFFIX=0, committed default): same replay. Validity:
  rewound=True and [gemm-prime] WALK base>0 present, eng_suffix=0.

Each warm sample replays the conversation fresh on its own session_id (no sample inherits
another sample's sampled replies; the fixed transcript keeps the answered conversation
identical across arms and samples).

Known instrument property, stated up front: with a fixed transcript the warm suffix is
(canonical reply + next user msg) rather than the pure new-user-msg suffix real serving
sees, because the server's own discarded prefix replies diverge from the canonical text at
the rewind point. The mechanism under test (suffix prime at base>0 through the batched vs
walk entry) is exercised on every warm evaluated turn; suffix lengths are banked per row.

## Interleaving and boots

Cycles of one sample per arm per turn; each cycle boots a door=0 server (COLD t4, COLD t8,
WALK t4, WALK t8) and a door=1 server (GEMM t4, GEMM t8), boot order alternating by cycle
parity. Cold rows are always the first requests of their boot. Cycle count x samples per
cycle = 8 total samples per arm per turn (granularity chosen after measuring boot time;
recorded in RESULTS).

## Hygiene

- /health 200 guard before every generation; empty output = disqualified row, kept.
- Engagement receipts parsed from the server log per request (gs-drive.py pattern).
- ILLEGAL / #87 counts scanned per boot; any nonzero is a launch-blocking finding.
- Raw generations, receipts, rubric, scores banked here and committed as they land
  (SPOT box; nothing lives only on the box).
- Blind judging: outputs shuffled and stripped to text-only before scoring; mapping
  file read only after all scores are written. RUBRIC.md committed before any
  generation existed.
