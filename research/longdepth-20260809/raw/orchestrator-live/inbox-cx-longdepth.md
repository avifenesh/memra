STEERING (2026-08-09, orchestrator): the ctx question is ANSWERED — corruption reproduces
at BOTH MEMRA_CTX=131072 (7 corrupt lines @ 8.7k completion tokens) and 262144 (3 @ 9.6k),
same prompt, temp 0.7, live pod. Receipts: /tmp/repro-131k-content.txt + /tmp/repro2-content.txt
on the orchestrator machine (grep -P Hebrew/CJK/Cyrillic classes). DROP the ctx axis from
your matrix; it is DEPTH. Focus: depth {2k, 6k, 12k} x spec {on, off} x temp {0, 0.7}.
Suspect order unchanged: SWA window=512 rolling at depth, MTP drafter at depth, KV wrap.
Also check: does corruption onset move with reasoning length included vs content-only depth
(i.e. is the trigger TOTAL sequence position or generated-token count)?
Owner doctrine update: 256k serving is MANDATORY (a model with a 256k window must serve
256k — anything less is itself a bug). Your fix must hold at 262144.

STEERING 2 (owner order: hardware time is THE resource): box1 pair is IDLE and lock-free
RIGHT NOW. Stop extended planning; start the depth matrix measurements on box1 immediately.
Plan refinements can land between measurement blocks. batchdraft lane also uses box1 —
flock arbitrates, take it in blocks, release between cells.
