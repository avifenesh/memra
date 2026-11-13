| metric | what | memra | vendor CPU floor | band | verdict |
|---|---|---|---|---|---|
| ring | median rel L2, n=3969 | 0.000e+00 | 0.000e+00 | <= 1.000e-04 | PASS |
| main_x | median rel L2, n=339 | 3.074e-09 | 0.000e+00 | <= 1.000e-04 | PASS |
| attn_in0 | median rel L2, n=1695 | 0.000e+00 | 0.000e+00 | <= 1.000e-04 | PASS |
| attn_out0 | median rel L2, n=1695 | 4.777e-03 | 4.725e-03 | <= 9.550e-03 | PASS |
| ffn_out0 | median rel L2, n=1695 | 0.0243 | 0.0242 | <= 4.860e-02 | PASS |
| h0 | median rel L2, n=1695 | 0.0229 | 0.0228 | <= 4.562e-02 | PASS |
| attn_in1 | median rel L2, n=1695 | 0.0219 | 0.0217 | <= 4.358e-02 | PASS |
| attn_out1 | median rel L2, n=1695 | 0.0533 | 0.0537 | <= 1.076e-01 | PASS |
| ffn_out1 | median rel L2, n=1695 | 0.0561 | 0.0563 | <= 1.127e-01 | PASS |
| h1 | median rel L2, n=1695 | 0.0479 | 0.0481 | <= 9.627e-02 | PASS |
| attn_in2 | median rel L2, n=1695 | 0.0499 | 0.0499 | <= 9.999e-02 | PASS |
| attn_out2 | median rel L2, n=1695 | 0.0760 | 0.0758 | <= 1.517e-01 | PASS |
| ffn_out2 | median rel L2, n=1695 | 0.0673 | 0.0678 | <= 1.358e-01 | PASS |
| h2 | median rel L2, n=1695 | 0.0665 | 0.0666 | <= 1.333e-01 | PASS |
| head_x | median rel L2, n=1695 | 0.0687 | 0.0686 | <= 1.372e-01 | PASS |
| pre0_absmax | median of the per-row max abs difference | 3.517e-04 | 3.521e-04 | <= 8.042e-04 | PASS |
| route0_agree | expert-set agreement, mean | 0.9980 | 0.9982 | >= 0.9882 | PASS |
| pre1_absmax | median of the per-row max abs difference | 5.065e-03 | 5.532e-03 | <= 1.116e-02 | PASS |
| route1_agree | expert-set agreement, mean | 0.9831 | 0.9835 | >= 0.9735 | PASS |
| pre2_absmax | median of the per-row max abs difference | 2.006e-04 | 2.084e-04 | <= 5.167e-04 | PASS |
| route2_agree | expert-set agreement, mean | 0.9797 | 0.9792 | >= 0.9692 | PASS |
| draft_token_agree | agreement with the vendor GPU draft, n=1695 | 0.9558 | 0.9528 | >= 0.9425 | PASS |
| draft_token_agree_same_ctx | agreement with the vendor GPU draft, n=1634 | 0.9712 | 0.9729 | >= 0.9629 | PASS |
| conf_absdiff | mean absolute difference | 0.0996 | 0.0973 | <= 1.947e-01 | PASS |
| conf_sigmoid_absdiff | mean absolute difference | 5.342e-03 | 5.532e-03 | <= 1.116e-02 | PASS |
| kl_top32 | mean KL(vendor GPU || x), n=1634 | 6.159e-03 | 6.515e-03 | <= 1.303e-02 | PASS |
| kl_full | mean KL(vendor GPU || x), n=191 | 8.013e-03 | 0.0126 | <= 2.512e-02 | PASS |
| markov_bias_rel_l2 | median rel L2, n=191 | 1.090e-07 | 3.581e-08 | <= 1.001e-04 | PASS |
