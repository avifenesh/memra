# Split-boundary map reduction

Prompt construction: `lcprestore`.

Target prompt token SHA-256 (canonical JSON): `not-recorded`.

## Pass/fail table

| Split | Output | Restored SHA-256 | Cold SHA-256 |
|---:|:---:|---|---|
| 64 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 128 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 192 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 256 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 320 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 384 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 448 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 57f67b8a0d6fb0d1e4d7fb261b518bda69465d3efe4b2dfeffa934ecf8ec0ed3 |
| 512 | FAIL | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 | 719a43f41b407364130580b2f12a8c09e78da460dc25ada2f1781dd436780079 |
| 576 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | d65710436ccfe6351880b48abd9bc3118b5ab6b66e0c229b3973bd70fcb985f4 |
| 640 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 5f59c049f4131b5908f77ead6d38347afe7c4f885117e7af09fce34554328e17 |
| 704 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 768 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 832 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 896 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 960 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1024 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 8f88259caeeb310ca268a31cef48e12a9580ade7a598a13585557b1dd040905c |
| 1088 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1152 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1216 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1280 | FAIL | 2682c50e5ea808430b239cf026617f3064f70b04f3a005853507c501d7ed8cc7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1344 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1408 | FAIL | e5cb66977f51ee89b7691f73b53ded4f43f1a7d3aa66e3b2250e7decc6e7baa0 | c72270bc85c8d10675db25fe0aa0e4eb7081afa1e484c8649e6b4fb8297638c7 |
| 1472 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1536 | PASS | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 |
| 1600 | FAIL | 544ff7f590f57b6ac1379266c80046eed627a5512084fe9555cbb1cf1112418b | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1664 | FAIL | 9c98f2e4f54eafe93539d851f9c9010bdb5b79931cc6eb5a7dfe2b9495b06a79 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1728 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 5f59c049f4131b5908f77ead6d38347afe7c4f885117e7af09fce34554328e17 |
| 1792 | FAIL | 8fc4413efc4e0d7fcd470fa1e5d6e26e773a74a086f07103a3b589212de82ca4 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1856 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1920 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1984 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2048 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2 |
| 2112 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2176 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2240 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2304 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2368 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2432 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2496 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2560 | PASS | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 |
| 2624 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2688 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2752 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 77c4c87dfd2a46c04f7ce1904c274d1b317a058ea140e3d9457b3e29cc4c0470 |
| 2816 | FAIL | 8fc4413efc4e0d7fcd470fa1e5d6e26e773a74a086f07103a3b589212de82ca4 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2880 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2944 | FAIL | 28911536683b6fde47ba23d72ef811c2f6c37b920f707459c4d9f80e2bd4a61a | 64fd4f1e99e7ad3288493b885cddf56158aa751ff484fdab6b69349619db2d91 |
| 3008 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3072 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3136 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3200 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3264 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3328 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3392 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3456 | PASS | 8f88259caeeb310ca268a31cef48e12a9580ade7a598a13585557b1dd040905c | 8f88259caeeb310ca268a31cef48e12a9580ade7a598a13585557b1dd040905c |
| 3520 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3584 | PASS | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 | bf81e8cb4ffc94c306d31d47159bb6a2ef9eb65b519bf41f122e5ae82f1fe525 |
| 3648 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3712 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3776 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3840 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3904 | FAIL | 5f59c049f4131b5908f77ead6d38347afe7c4f885117e7af09fce34554328e17 | fb827a99f0c3e7f2b57c7bb8f974855cf2493ebe4566a035aeef20b222f13973 |
| 3968 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4032 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4096 | FAIL | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | 223618bfd84e4f30bb454fb7383f139753011e918926af620cf047dda7c136c2 |
| 4160 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4224 | FAIL | 5ccc7095db081405372a1e9f29f3bf37591fcc87941b2dca8aba4cd81533cb29 | 8f88259caeeb310ca268a31cef48e12a9580ade7a598a13585557b1dd040905c |
| 4288 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4352 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4374 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |

## Correlation counts

| Feature | Value | Pass | Fail |
|---|---|---:|---:|
| `prefix_min_eligible` | `True` | 51 | 18 |
| `cold_prefill_execution` | `gemma4_prime-monolithic-t4860` | 51 | 18 |
| `restored_suffix_execution` | `decode_step-tokenwise-t1` | 51 | 18 |
| `worker_prefill_tick` | `1024` | 51 | 18 |
| `worker_solo_prefill_tick` | `8192` | 51 | 18 |
| `decode_batch_provenance` | `eager-b1-width-row-null` | 51 | 18 |
| `split_mod_block_q64` | `0` | 50 | 18 |
| `split_mod_block_q64` | `22` | 1 | 0 |
| `split_mod_bk32` | `0` | 50 | 18 |
| `split_mod_bk32` | `22` | 1 | 0 |
| `split_mod_sp_m16` | `0` | 50 | 18 |
| `split_mod_sp_m16` | `6` | 1 | 0 |
| `suffix_mod_block_q64` | `38` | 1 | 0 |
| `suffix_mod_block_q64` | `60` | 50 | 18 |
| `global_first_suffix_class` | `fa_decode_kvmod-scalar-sp16` | 6 | 1 |
| `global_first_suffix_class` | `fa_decode_rows-sp32` | 45 | 17 |
| `swa_first_suffix_class` | `fa_decode_kvmod-vec-sp16` | 11 | 4 |
| `swa_first_suffix_class` | `fa_decode_rows_w-sp64` | 40 | 14 |
| `big_rig_fallback_split_ladder` | `16` | 20 | 11 |
| `big_rig_fallback_split_ladder` | `64` | 31 | 7 |
| `big_rig_fallback_split_live` | `False` | 40 | 14 |
| `big_rig_fallback_split_live` | `True` | 11 | 4 |
| `actual_global_partition_keys` | `16` | 6 | 1 |
| `actual_global_partition_keys` | `32` | 45 | 17 |
| `actual_swa_partition_keys` | `16` | 11 | 4 |
| `actual_swa_partition_keys` | `64` | 40 | 14 |
| `global_plane_page_offset` | `0` | 50 | 18 |
| `global_plane_page_offset` | `3072` | 1 | 0 |
| `swa_plane_page_offset` | `0` | 51 | 18 |

Exact discriminators: `none`.

Sampled transitions: `[{"left_split": 384, "left_verdict": "PASS", "right_split": 448, "right_verdict": "FAIL"}, {"left_split": 640, "left_verdict": "FAIL", "right_split": 704, "right_verdict": "PASS"}, {"left_split": 960, "left_verdict": "PASS", "right_split": 1024, "right_verdict": "FAIL"}, {"left_split": 1024, "left_verdict": "FAIL", "right_split": 1088, "right_verdict": "PASS"}, {"left_split": 1216, "left_verdict": "PASS", "right_split": 1280, "right_verdict": "FAIL"}, {"left_split": 1280, "left_verdict": "FAIL", "right_split": 1344, "right_verdict": "PASS"}, {"left_split": 1344, "left_verdict": "PASS", "right_split": 1408, "right_verdict": "FAIL"}, {"left_split": 1408, "left_verdict": "FAIL", "right_split": 1472, "right_verdict": "PASS"}, {"left_split": 1536, "left_verdict": "PASS", "right_split": 1600, "right_verdict": "FAIL"}, {"left_split": 1792, "left_verdict": "FAIL", "right_split": 1856, "right_verdict": "PASS"}, {"left_split": 1984, "left_verdict": "PASS", "right_split": 2048, "right_verdict": "FAIL"}, {"left_split": 2048, "left_verdict": "FAIL", "right_split": 2112, "right_verdict": "PASS"}, {"left_split": 2688, "left_verdict": "PASS", "right_split": 2752, "right_verdict": "FAIL"}, {"left_split": 2816, "left_verdict": "FAIL", "right_split": 2880, "right_verdict": "PASS"}, {"left_split": 2880, "left_verdict": "PASS", "right_split": 2944, "right_verdict": "FAIL"}, {"left_split": 2944, "left_verdict": "FAIL", "right_split": 3008, "right_verdict": "PASS"}, {"left_split": 3840, "left_verdict": "PASS", "right_split": 3904, "right_verdict": "FAIL"}, {"left_split": 3904, "left_verdict": "FAIL", "right_split": 3968, "right_verdict": "PASS"}, {"left_split": 4032, "left_verdict": "PASS", "right_split": 4096, "right_verdict": "FAIL"}, {"left_split": 4096, "left_verdict": "FAIL", "right_split": 4160, "right_verdict": "PASS"}, {"left_split": 4160, "left_verdict": "PASS", "right_split": 4224, "right_verdict": "FAIL"}, {"left_split": 4224, "left_verdict": "FAIL", "right_split": 4288, "right_verdict": "PASS"}]`.

Targeted candidates: `65,385,447,510,511,641,703,961,1023,1025,1087,1217,1279,1281,1343,1345,1407,1409,1471,1537,1599,1793,1855,1985,2047,2049,2111,2689,2751,2817,2879,2881,2943,2945,3007,3841,3903,3905,3967,4033,4095,4097,4159,4161,4223,4225,4287`.

Named-boundary candidates: `65,510,511,1023,1025,2047,2049`.
