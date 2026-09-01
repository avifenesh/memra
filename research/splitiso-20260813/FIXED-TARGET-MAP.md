# Split-boundary map reduction

Prompt construction: `fixed-target`.

Target prompt token SHA-256 (canonical JSON): `21ef4227fcb0993c341e03c4df6bf01b27f6012021c881fc4a8f451364495397`.

## Pass/fail table

| Split | Output | Restored SHA-256 | Cold SHA-256 |
|---:|:---:|---|---|
| 64 | FAIL | 7dd042dd320db978d838991068a8da374ed7294b4ba54d1774629af427816dd3 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 65 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 127 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 128 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 192 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 256 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 320 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 321 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 383 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 384 | FAIL | c72270bc85c8d10675db25fe0aa0e4eb7081afa1e484c8649e6b4fb8297638c7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 385 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 447 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 448 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 510 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 511 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 512 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 513 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 576 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 640 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 704 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 768 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 832 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 896 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 960 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1023 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1024 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1025 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1088 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1152 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1216 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1280 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1344 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1408 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1409 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1471 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1472 | FAIL | 5905248c3ab566e353d828d760a6bf040cbc866822d95ab752baabde7b7f813f | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1473 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1535 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1536 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1600 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1664 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1665 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1727 | FAIL | d243e34dd841d450550ad4b2c9c09b51ada3635371dce425bb80d13601ecc1d0 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1728 | FAIL | 5905248c3ab566e353d828d760a6bf040cbc866822d95ab752baabde7b7f813f | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1729 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1791 | FAIL | c72270bc85c8d10675db25fe0aa0e4eb7081afa1e484c8649e6b4fb8297638c7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1792 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1793 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1855 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1856 | FAIL | c72270bc85c8d10675db25fe0aa0e4eb7081afa1e484c8649e6b4fb8297638c7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1857 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1919 | FAIL | d243e34dd841d450550ad4b2c9c09b51ada3635371dce425bb80d13601ecc1d0 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1920 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 1984 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2047 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2048 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2049 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2112 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2176 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2240 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2241 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2303 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2304 | FAIL | c72270bc85c8d10675db25fe0aa0e4eb7081afa1e484c8649e6b4fb8297638c7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2305 | FAIL | 337fdde03ad60802a6ffcc9c40c790c52e32b9ac5fa8c90b27178f8dc37044d7 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2367 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2368 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2432 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2496 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2560 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2624 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2688 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2752 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2816 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2880 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 2944 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3008 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3072 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3073 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3135 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3136 | FAIL | d243e34dd841d450550ad4b2c9c09b51ada3635371dce425bb80d13601ecc1d0 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3137 | FAIL | 77976a7a8fdea5d0e6beb03bc7d52cb8d33d4220891ee0f91634fe7cf8cead0d | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3199 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3200 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3264 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3328 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3392 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3456 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3520 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3584 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3648 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3712 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3713 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3775 | FAIL | 8fc4413efc4e0d7fcd470fa1e5d6e26e773a74a086f07103a3b589212de82ca4 | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3776 | FAIL | 77976a7a8fdea5d0e6beb03bc7d52cb8d33d4220891ee0f91634fe7cf8cead0d | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3777 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3839 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3840 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3904 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 3968 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4032 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4096 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4160 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4224 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4288 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4352 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |
| 4374 | PASS | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df | eb3e68a9f0448bba8e01933cf1cddf2cea114cdbbb41b122b8219482b27211df |

## Correlation counts

| Feature | Value | Pass | Fail |
|---|---|---:|---:|
| `prefix_min_eligible` | `True` | 92 | 14 |
| `cold_prefill_execution` | `gemma4_prime-monolithic-t4860` | 92 | 14 |
| `restored_suffix_execution` | `decode_step-tokenwise-t1` | 92 | 14 |
| `worker_prefill_tick` | `1024` | 92 | 14 |
| `worker_solo_prefill_tick` | `8192` | 92 | 14 |
| `decode_batch_provenance` | `eager-b1-width-row-null` | 92 | 14 |
| `split_mod_block_q64` | `0` | 60 | 8 |
| `split_mod_block_q64` | `1` | 16 | 2 |
| `split_mod_block_q64` | `22` | 1 | 0 |
| `split_mod_block_q64` | `62` | 1 | 0 |
| `split_mod_block_q64` | `63` | 14 | 4 |
| `split_mod_bk32` | `0` | 60 | 8 |
| `split_mod_bk32` | `1` | 16 | 2 |
| `split_mod_bk32` | `22` | 1 | 0 |
| `split_mod_bk32` | `30` | 1 | 0 |
| `split_mod_bk32` | `31` | 14 | 4 |
| `split_mod_sp_m16` | `0` | 60 | 8 |
| `split_mod_sp_m16` | `1` | 16 | 2 |
| `split_mod_sp_m16` | `14` | 1 | 0 |
| `split_mod_sp_m16` | `15` | 14 | 4 |
| `split_mod_sp_m16` | `6` | 1 | 0 |
| `suffix_mod_block_q64` | `38` | 1 | 0 |
| `suffix_mod_block_q64` | `59` | 16 | 2 |
| `suffix_mod_block_q64` | `60` | 60 | 8 |
| `suffix_mod_block_q64` | `61` | 14 | 4 |
| `suffix_mod_block_q64` | `62` | 1 | 0 |
| `global_first_suffix_class` | `fa_decode_kvmod-scalar-sp16` | 12 | 2 |
| `global_first_suffix_class` | `fa_decode_rows-sp32` | 80 | 12 |
| `swa_first_suffix_class` | `fa_decode_kvmod-vec-sp16` | 23 | 2 |
| `swa_first_suffix_class` | `fa_decode_rows_w-sp64` | 69 | 12 |
| `big_rig_fallback_split_ladder` | `16` | 47 | 8 |
| `big_rig_fallback_split_ladder` | `64` | 45 | 6 |
| `big_rig_fallback_split_live` | `False` | 69 | 12 |
| `big_rig_fallback_split_live` | `True` | 23 | 2 |
| `actual_global_partition_keys` | `16` | 12 | 2 |
| `actual_global_partition_keys` | `32` | 80 | 12 |
| `actual_swa_partition_keys` | `16` | 23 | 2 |
| `actual_swa_partition_keys` | `64` | 69 | 12 |
| `global_plane_page_offset` | `0` | 60 | 8 |
| `global_plane_page_offset` | `3072` | 2 | 0 |
| `global_plane_page_offset` | `3584` | 14 | 4 |
| `global_plane_page_offset` | `512` | 16 | 2 |
| `swa_plane_page_offset` | `0` | 62 | 8 |
| `swa_plane_page_offset` | `2048` | 30 | 6 |

Exact discriminators: `none`.

Sampled transitions: `[{"left_split": 64, "left_verdict": "FAIL", "right_split": 65, "right_verdict": "PASS"}, {"left_split": 383, "left_verdict": "PASS", "right_split": 384, "right_verdict": "FAIL"}, {"left_split": 384, "left_verdict": "FAIL", "right_split": 385, "right_verdict": "PASS"}, {"left_split": 1471, "left_verdict": "PASS", "right_split": 1472, "right_verdict": "FAIL"}, {"left_split": 1472, "left_verdict": "FAIL", "right_split": 1473, "right_verdict": "PASS"}, {"left_split": 1665, "left_verdict": "PASS", "right_split": 1727, "right_verdict": "FAIL"}, {"left_split": 1728, "left_verdict": "FAIL", "right_split": 1729, "right_verdict": "PASS"}, {"left_split": 1729, "left_verdict": "PASS", "right_split": 1791, "right_verdict": "FAIL"}, {"left_split": 1791, "left_verdict": "FAIL", "right_split": 1792, "right_verdict": "PASS"}, {"left_split": 1855, "left_verdict": "PASS", "right_split": 1856, "right_verdict": "FAIL"}, {"left_split": 1856, "left_verdict": "FAIL", "right_split": 1857, "right_verdict": "PASS"}, {"left_split": 1857, "left_verdict": "PASS", "right_split": 1919, "right_verdict": "FAIL"}, {"left_split": 1919, "left_verdict": "FAIL", "right_split": 1920, "right_verdict": "PASS"}, {"left_split": 2303, "left_verdict": "PASS", "right_split": 2304, "right_verdict": "FAIL"}, {"left_split": 2305, "left_verdict": "FAIL", "right_split": 2367, "right_verdict": "PASS"}, {"left_split": 3135, "left_verdict": "PASS", "right_split": 3136, "right_verdict": "FAIL"}, {"left_split": 3137, "left_verdict": "FAIL", "right_split": 3199, "right_verdict": "PASS"}, {"left_split": 3713, "left_verdict": "PASS", "right_split": 3775, "right_verdict": "FAIL"}, {"left_split": 3776, "left_verdict": "FAIL", "right_split": 3777, "right_verdict": "PASS"}]`.

Targeted candidates: `1666,1726,1730,1790,1858,1918,2306,2366,3138,3198,3714,3774`.

Named-boundary candidates: ``.
