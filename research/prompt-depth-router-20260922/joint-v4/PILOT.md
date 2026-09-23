# Development pilot: do recent tokens predict acceptance?

The hosted CPU pilot trained a small logistic model on the v3
calibration conversations and tested it on six v3 C=0 monitor
conversations. Those records are **development input for v4**,
not fresh v4 heldout data. Only eligible accepted-prefix positions
were labeled: a proposal after the first rejection received no
target-acceptance label.

| Features | Evaluation conditional log loss |
|---|---:|
| Draft confidence and position | 0.37255 |
| Plus last committed token identity/class | 0.37243 |
| Plus previous 4/16 committed-token classes | 0.37274 |
| Plus prior completed-round acceptance/cost | 0.37246 |

The fit used 7,821 training labels from three conversations and
28,858 evaluation labels from six other conversations. Extra
features lowered training loss but changed evaluation loss by less
than 0.0002 in mixed directions. This **does not establish useful
history signal** for acceptance on these topics, nor does it
establish that token history cannot help a *cost-aware* action
policy. The many round labels share only nine conversation-level
clusters; they are not 36,679 independent requests.

`PILOT.json` retains the source/archive hashes, split, trained
coefficients, per-conversation and per-output-phase scores. Hosted
job `107237781802` passed in workflow `35877693084`. These are
acceptance-calibration diagnostics, not a C/K/D decision or an
end-to-end speed result. The fresh code workload in
`workloads-v4/manifest.json` stays unseen by this fit.
