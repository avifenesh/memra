# The price of a per-session recurrent checkpoint on the RTX 5090 (memra#1093 item 5, 2026-11-09)

Verdict: at the device default, arming a checkpoint for every recurrent session buys nothing on
this card and costs memory. With the prefix cache on, the prefix cache's on-grid spec restore
serves the next turn before any pool probe runs, so an armed session parks unused. With the
prefix cache off (the bound for an evicted or missing entry), the checkpoint is worth 0.47 to 0.67
s of TTFT per later turn. It costs about 480 MB of device memory per parked session and +0.08 s on
the first turn's cold prime. The glm5 case named in the issue is not measured: no glm5_next
checkpoint is on this rig.

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head (the MTP spec route), no DFlash
drafter, RTX 5090 Laptop, context 8,192, two sessions, prefix cache 800 MB (derived) or off. Binary:
PR #1133's head bb4f8c5b7d (sha256 891c478fe0188685). Under it a recurrent spec session with no
on-grid checkpoint is freed at retire, so "no checkpoint" means no pool resume.

## The cell

`price_cell.py`. The same conversations go through two arms, one binary, eight boots in the order
anon named named anon anon named named anon, three conversations per boot:

- Each conversation is three `/v1/completions` turns of markerless text (a 30-note first prompt of
  814 tokens; turn N+1 = turn N's prompt + its completion + a follow-on), greedy,
  `max_tokens` 128, streamed (TTFT = first chunk).
- **anon** sends no session id: the engine arms no turn checkpoint for a markerless anonymous
  prompt. This is the default case the issue names.
- **named** sends a `user` id per conversation: the engine arms the checkpoint at the raw guard
  window's grid point, so this arm prices "a checkpoint for every recurrent session" with no code
  change.
- After each request `/metrics` gives `spec_pool_entries` and `cuda_driver_free_bytes`, and the
  server log gives the path.

n = 12 per cell (4 boots x 3 conversations).

## Prefix cache on, the device default (`prefix-on/`)

| arm | turn 1 TTFT | turn 2 TTFT | turn 3 TTFT | turns 2-3 served by |
|---|---:|---:|---:|---|
| anon | 0.705 (0.587-0.852) | 0.310 (0.288-0.325) | 0.239 (0.227-0.250) | prefix-cache spec restore (768, then 928) |
| named | 0.714 (0.610-0.847) | 0.312 (0.292-0.324) | 0.240 (0.230-0.246) | prefix-cache spec restore (768, then 928) |

The named session parks (its checkpoint is on the grid) but is never resumed: the restore claims
the turn first. The anon turn-1 session is freed (`spec-park SKIPPED (grid law)`). Driver free
memory after the first conversation: 4,722 MB named against 5,202 MB anon. After that both pools
hold their cap of 2 sessions, since the restored turn-2 and turn-3 sessions park in both arms.

## Prefix cache off, the bound (`prefix-off/`, MEMRA_PREFIX_CACHE_MB=0)

| arm | turn 1 TTFT | turn 2 TTFT | turn 3 TTFT | turns 2-3 served by | driver free after turn 3 (MB) |
|---|---:|---:|---:|---|---|
| anon | 0.691 (0.668-0.731) | 0.785 (0.762-0.863) | 0.909 (0.863-1.006) | cold prime (every session freed) | 6,162-6,194, 0 parked |
| named | 0.771 (0.666-0.895) | 0.311 (0.301-0.328) | 0.239 (0.232-0.248) | affinity rewind to 768, then 928 | 6,034 / 5,554 / 5,074 with 1 / 2 / 2 parked |

What the checkpoint buys here: 0.47 s on turn 2 and 0.67 s on turn 3. What it costs: +0.08 s on
turn 1 (the prime stops at the checkpoint; `research/spec-pool-grid-20261109/stop-cost/` measures
0.11 s for that stop at a fixed length) and about 480 MB per parked session. Of that, about 158 MB
is the recurrent snapshot (a 32-token entry of this model is 157.9 MB); the rest is the session's
trunk KV at its cap and the draft scratch.

## Reading

On the RTX 5090 at the default, the reuse the issue counted as lost to cold primes is not lost
for this shape: the prefix cache serves it. A per-session checkpoint pays off only where no
prefix entry serves the turn (the cache off, or its entry evicted under memory pressure), and
there it trades about 480 MB per parked session for 0.5 to 0.7 s per later turn. Not armed by
default; no code change follows from this receipt.
