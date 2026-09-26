# Vast grader correction

The first launch on the accepted Vast host was stopped during its
qualifier after two completed instruction arms. Review found that the
math grader could accept the first numeric prefix after `####` instead
of requiring a complete final answer line. Its two results and one
incomplete arm were excluded. The grader, protocol, source manifest
and on-host grader check were updated before restarting every native
arm from empty results. The restart has its own run metadata and
source hashes; the stopped launch is a diagnostic receipt only.

The earlier v9 code test ran on an RTX PRO 6000 Blackwell, while this
complete transfer rerun runs on an RTX 5090. Each rerun comparison is
paired on that one 5090, but a difference between v9 code and rerun
non-code rates mixes workload and hardware effects. Interpret each
topic's within-GPU fixed-versus-learned result separately.

The frozen transfer scorer checks two learned options across two
topics with marginal paired intervals. Its `transfer_win` field is
an exploratory gate, without a familywise multiple-comparison
adjustment. The prepared v11 final study freezes one primary learned
option per topic on validation before testing fresh final prompts.
