| text | tokens | positions | native KL | schedule floor KL | native / floor | native KL, prefix | length floor KL | native / floor | top-1: native, schedule; native, length (prefix) | pass (per floor) |
|---|---|---|---|---|---|---|---|---|---|---|
| tf_en_lighthouse | 2,048 | all: 2,048 | 2.23e-02 | 1.76e-02 | 1.26 | - | - | - | 97.4%, 97.9% | PASS (in) |
| tf_en_lighthouse | 2,048 | entropy > 1 nat: 159 | 1.06e-01 | 7.35e-02 | 1.44 | - | - | - | 81.1%, 83.0% | PASS (in) |
| tf_en_lighthouse | 2,048 | top entropy decile: 205 | 1.14e-01 | 7.95e-02 | 1.44 | - | - | - | 81.5%, 84.4% | PASS (in) |
| tflong_he_haifa | 31,859 | all: 31,859 | 1.66e-02 | 1.68e-02 | 0.99 | 1.64e-02 | 1.66e-02 | 0.98 | 94.8%, 95.0%; 94.9%, 94.9% | PASS (in, in) |
| tflong_he_haifa | 31,859 | entropy > 1 nat: 11,387 | 3.34e-02 | 3.38e-02 | 0.99 | 3.27e-02 | 3.28e-02 | 1.00 | 87.4%, 87.6%; 87.7%, 87.8% | PASS (in, in) |
| tflong_he_haifa | 31,859 | top entropy decile: 3,186 | 2.99e-02 | 3.10e-02 | 0.96 | 2.95e-02 | 2.97e-02 | 0.99 | 81.6%, 81.9%; 81.5%, 82.4% | PASS (in, in) |
| tf128k_en_moby | 131,072 | all: 131,072 | 1.34e-02 | 1.32e-02 | 1.02 | 1.33e-02 | 9.01e-03 | 1.47 | 99.1%, 99.1%; 99.1%, 99.2% | PASS (in, in) |
| tf128k_en_moby | 131,072 | entropy > 1 nat: 1,322 | 3.08e-01 | 2.94e-01 | 1.05 | 3.15e-01 | 1.99e-01 | 1.58 | 71.5%, 71.5%; 71.4%, 73.3% | PASS (in, in) |
| tf128k_en_moby | 131,072 | top entropy decile: 13,108 | 1.27e-01 | 1.24e-01 | 1.02 | 1.26e-01 | 8.54e-02 | 1.47 | 90.6%, 90.7%; 90.8%, 91.6% | PASS (in, in) |
| tf64k_he_cities | 131,072 | all: 131,072 | 1.49e-02 | 1.44e-02 | 1.04 | 1.50e-02 | 1.40e-02 | 1.07 | 95.5%, 95.5%; 95.4%, 95.5% | PASS (in, in) |
| tf64k_he_cities | 131,072 | entropy > 1 nat: 43,069 | 3.23e-02 | 3.17e-02 | 1.02 | 3.23e-02 | 3.07e-02 | 1.05 | 88.0%, 88.1%; 88.0%, 88.2% | PASS (in, in) |
| tf64k_he_cities | 131,072 | top entropy decile: 13,108 | 3.23e-02 | 3.17e-02 | 1.02 | 3.21e-02 | 3.00e-02 | 1.07 | 82.1%, 82.5%; 82.0%, 82.5% | PASS (in, in) |
| tf256k_en_moby | 262,144 | all: 262,144 | 1.31e-02 | 1.41e-02 | 0.93 | 1.11e-02 | 8.79e-03 | 1.27 | 99.0%, 98.9%; 99.1%, 99.1% | PASS (in, in) |
| tf256k_en_moby | 262,144 | entropy > 1 nat: 3,057 | 2.65e-01 | 2.83e-01 | 0.94 | 2.71e-01 | 1.88e-01 | 1.44 | 70.6%, 69.2%; 69.4%, 72.0% | PASS (in, out) |
| tf256k_en_moby | 262,144 | top entropy decile: 26,215 | 1.23e-01 | 1.32e-01 | 0.93 | 1.06e-01 | 8.19e-02 | 1.29 | 89.8%, 89.5%; 91.0%, 91.4% | PASS (in, in) |
