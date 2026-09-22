# Predeclared request-level C replay

This exploratory policy is frozen before reading the fixed-C grid's scored
throughput. Every candidate request was actually run at C=0, C=0.15,
C=0.30 and C=0.30 with zero-draft rounds, always under fixed K=3. Because
the requests use fresh caches and fixed prompts, selecting one measured arm
cannot change the next request's prompt. The replay may inspect an unchosen
arm for the matched loop screen and final control comparison; its controller
receives only the chosen arm's committed output count, time and
drafted/accepted counts.

Use scenarios 0 and 1 for full-information calibration. For each code prompt
length, compute the C=0 reference rate from those calibration requests;
choose the initial C with the highest mean length-normalized calibration
rate. Score only scenarios 2–5, in scenario and turn order. The controller
persists across all 16 held-out code requests.

Every sixth evaluation request probes one adjacent setting in the ordered
ladder `off, c015, c030, c030zero`. Recent selected-arm acceptance below
0.65 probes higher C; above 0.82 probes lower C. Otherwise probe direction
alternates, reversing at an edge. Adopt a probed C if its chosen request's
length-normalized output tok/s exceeds the incumbent's recent mean by more
than 2%. This intentionally simple rule uses actual cycle yield for the
decision; acceptance only chooses which direction to probe. All probe
requests count in the evaluation.

Compare pooled **native request** output tok/s with C=0 and the one
calibrated fixed setting on exactly the same held-out prompts, plus each
scenario's paired change, output lengths and format coverage. Exclude a
loop-flagged request from all arms in the primary comparison; a chosen
loop produces no learner update. The replay's CPU policy time and
calibration workload are outside this native-request ratio and must be
priced before a serving claim. A single probe can be noisy, and four
scenarios cannot establish a general policy gain. This replay does not
switch C inside a live model process or demonstrate warm-session KV reuse.
The live learner design and its full controls are in
[LEARNER-DESIGN.md](LEARNER-DESIGN.md).
