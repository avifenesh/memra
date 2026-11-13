| prompt | path | allocs | bytes allocated MiB | pool res before -> after MiB | growth MiB | first diff vs first prime | OOM |
|---|---|---|---|---|---|---|---|
| 8168 | spec | 2442 | 131872 | 19328 -> 20896 | 1568 | - | False |
| 5731 | spec | 2442 | 131872 | 19328 -> 21056 | 1728 | - | False |
| 4644 | spec | 2442 | 131872 | 19296 -> 20832 | 1536 | - | False |
| 4153 | spec | 2442 | 131872 | 19296 -> 20832 | 1536 | - | False |
| 4108 | spec | 2494 | 133763 | 19296 -> 21152 | 1856 | #0: (420659200, 'crates/memra-engine/src/dflash.rs:5594') vs (419430400, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 4088 | spec | 2443 | 131688 | 19328 -> 20864 | 1536 | #0: (418611200, 'crates/memra-engine/src/dflash.rs:5594') vs (419430400, 'crates/memra-engine/src/dflash.rs:5594') | False |
| 4070 | spec | 2443 | 131218 | 19328 -> 20864 | 1536 | #0: (416768000, 'crates/memra-engine/src/dflash.rs:5594') vs (419430400, 'crates/memra-engine/src/dflash.rs:5594') | False |
