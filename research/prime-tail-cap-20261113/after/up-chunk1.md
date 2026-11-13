| prompt | path | allocs | bytes allocated MiB | pool res before -> after MiB | growth MiB | first diff vs first prime | OOM |
|---|---|---|---|---|---|---|---|
| 4060 | spec | 2443 | 130759 | 19200 -> 20736 | 1536 | - | False |
| 4106 | spec | 2413 | 130801 | 19200 -> 20736 | 1536 | #0: (416153600, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 4167 | spec | 2442 | 131872 | 19232 -> 20768 | 1536 | #0: (419430400, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 4153 | spec | 2442 | 131872 | 19264 -> 20992 | 1728 | #0: (419430400, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 4598 | spec | 2442 | 131872 | 19264 -> 20800 | 1536 | #0: (419430400, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 5745 | spec | 2442 | 131872 | 19296 -> 20864 | 1568 | #0: (419430400, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 8116 | spec | 2442 | 131872 | 19392 -> 20928 | 1536 | #0: (419430400, 'crates/memra-engine/src/dflash.rs:5594') vs (415744000, 'crates/memra-engine/src/dflash.rs:5594') | False |
