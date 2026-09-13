# Physical one-row replay qualification

Reuse the 48 prior oracle prompts strictly as a numerical qualification set;
they are no longer fresh policy evaluation data. Keep the same locked model,
4096+512 head IDs, K4, zero confidence, frozen updates and eager greedy path.

At sampled reached states, preserve the exact draft hidden vector. Construct a
separate same-width Q8_0 head from original row bytes. Replace slot4096 with each
top16 outside candidate, the missing verifier target if absent from that list,
the current proposal (duplicate-winner tie fixture), and the original slot ID
(restoration). Use the normal native matmul and GPU argmax. Live state and target
verification never consume a shadow proposal.

Compare actual winners and score-based predictions; refuse every non-ambiguous
disagreement or failed restoration. Record margins, maximum score errors, actual
delta-correctness, ambiguity and all shadow candidate IDs. Require positive,
negative and neutral changes, plus tie cases. Compare complete 128-token output
against plain greedy and replay-OFF controls on all 48 prompts. A request with
replay enabled without the parent diagnostic must refuse.

This qualifies one-row scoring only. It does not validate changed trajectories,
learned policy benefit or serving performance. Replay cost is not a candidate
probe implementation cost. Archive source/binary hashes, raw outputs, telemetry,
predicted/actual interventions and failures before continuing to bounded discovery.
