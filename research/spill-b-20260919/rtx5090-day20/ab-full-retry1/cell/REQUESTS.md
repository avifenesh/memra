| idx | role | tenant | turn | prompt | cold digest[:16] | slru cached | slru computed | slru events | lru cached | lru computed | lru events |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 0 | seed1 | cohort-1 | - | 1248 | c1e188654046e892 | 0 | 1248 | insert 1248 | 0 | 1248 | insert 1248 |
| 1 | seed2 | cohort-1 | - | 1248 | c1e188654046e892 | 1248 | 0 | hit 1248 | 1248 | 0 | hit 1248 |
| 2 | seed1 | cohort-2 | - | 1344 | 3ca9a6e466c2dfb8 | 0 | 1344 | insert 1344 | 0 | 1344 | insert 1344 |
| 3 | seed2 | cohort-2 | - | 1344 | 3ca9a6e466c2dfb8 | 1344 | 0 | hit 1344 | 1344 | 0 | hit 1344 |
| 4 | seed1 | cohort-3 | - | 1440 | 1e5e661ff29ee077 | 0 | 1440 | insert 1440 | 0 | 1440 | insert 1440 |
| 5 | seed2 | cohort-3 | - | 1440 | 1e5e661ff29ee077 | 1440 | 0 | hit 1440 | 1440 | 0 | hit 1440 |
| 6 | seed1 | cohort-4 | - | 1536 | 57541e978a9b2efb | 0 | 1536 | insert 1536 | 0 | 1536 | insert 1536 |
| 7 | seed2 | cohort-4 | - | 1536 | 57541e978a9b2efb | 1536 | 0 | hit 1536 | 1536 | 0 | hit 1536 |
| 8 | loop | grow | 1 | 10912 | 006fdb36c6ff0391 | 0 | 10912 | insert 10912, evict 1248 (Protected), evict 1344 (Protected) | 0 | 10912 | insert 10912, evict 1248 (Protected), evict 1344 (Protected) |
| 9 | loop | grow | 2 | 11072 | 70efa8d085a5531a | 10912 | 160 | hit 10912, insert 11072, evict 1440 (Probation), evict 1536 (Protected), demote x1 | 10912 | 160 | hit 10912, insert 11072, evict 1440 (Protected), evict 1536 (Protected) |
| 10 | loop | grow | 3 | 11232 | 4b3ad3a092aaa00f | 11072 | 160 | hit 11072, insert 11232, evict 10912 (Probation), demote x1 | 11072 | 160 | hit 11072, insert 11232, evict 10912 (Protected) |
| 11 | return | cohort-1 | 3 | 1408 | f1a33345e94b7e62 | 0 | 1408 | insert 1408, evict 11232 (Probation) | 0 | 1408 | insert 1408, evict 11072 (Protected) |
| 12 | loop | grow | 4 | 11392 | 24a97a2867768b2d | 11072 | 320 | hit 11072, insert 11392, evict 1408 (Probation) | 11232 | 160 | hit 11232, insert 11392, evict 1408 (Probation) |
| 13 | loop | grow | 5 | 11552 | 22f023976ebc22fd | 11392 | 160 | hit 11392, insert 11552, evict 11072 (Probation), demote x1 | 11392 | 160 | hit 11392, insert 11552, evict 11232 (Protected) |
| 14 | loop | grow | 6 | 11712 | 24a97a2867768b2d | 11552 | 160 | hit 11552, insert 11712, evict 11392 (Probation), demote x1 | 11552 | 160 | hit 11552, insert 11712, evict 11392 (Protected) |
| 15 | return | cohort-2 | 6 | 1504 | fa08252a46f69194 | 0 | 1504 | insert 1504, evict 11712 (Probation) | 0 | 1504 | insert 1504, evict 11552 (Protected) |
| 16 | loop | grow | 7 | 11872 | 8d11831c91ef4dc9 | 11552 | 320 | hit 11552, insert 11872, evict 1504 (Probation) | 11712 | 160 | hit 11712, insert 11872, evict 1504 (Probation) |
| 17 | loop | grow | 8 | 12032 | 24a97a2867768b2d | 11872 | 160 | hit 11872, insert 12032, evict 11552 (Probation), demote x1 | 11872 | 160 | hit 11872, insert 12032, evict 11712 (Protected) |
| 18 | loop | grow | 9 | 12192 | c95fce823d87df23 | 12032 | 160 | hit 12032, insert 12192, evict 11872 (Probation), demote x1 | 12032 | 160 | hit 12032, insert 12192, evict 11872 (Protected) |
| 19 | return | cohort-3 | 9 | 1600 | 7e744a72e7b79cb1 | 0 | 1600 | insert 1600, evict 12192 (Probation) | 0 | 1600 | insert 1600, evict 12032 (Protected) |
| 20 | loop | grow | 10 | 12352 | 9022af0570f69478 | 12032 | 320 | hit 12032, insert 12352, evict 1600 (Probation) | 12192 | 160 | hit 12192, insert 12352, evict 1600 (Probation) |
| 21 | loop | grow | 11 | 12512 | b6b5b760960efe97 | 12352 | 160 | hit 12352, insert 12512, evict 12032 (Probation), demote x1 | 12352 | 160 | hit 12352, insert 12512, evict 12192 (Protected) |
| 22 | loop | grow | 12 | 12672 | d5b3015fc5c4f9f7 | 12512 | 160 | hit 12512, insert 12672, evict 12352 (Probation), demote x1 | 12512 | 160 | hit 12512, insert 12672, evict 12352 (Protected) |
| 23 | return | cohort-4 | 12 | 1696 | 66333786858cc554 | 0 | 1696 | insert 1696, evict 12672 (Probation) | 0 | 1696 | insert 1696, evict 12512 (Protected) |
| 24 | final | cohort-1 | 12 | 1568 | f1a33345e94b7e62 | 0 | 1568 | insert 1568 | 0 | 1568 | insert 1568 |
| 25 | final | cohort-2 | 12 | 1664 | 777b8b408b746f8f | 0 | 1664 | insert 1664, evict 1696 (Probation) | 0 | 1664 | insert 1664, evict 12672 (Probation) |
| 26 | final | cohort-3 | 12 | 1760 | 4b23c43d6dad7932 | 0 | 1760 | insert 1760, evict 1568 (Probation) | 0 | 1760 | insert 1760 |
| 27 | final | cohort-4 | 12 | 1856 | d130dbcf85337f3d | 0 | 1856 | insert 1856, evict 1664 (Probation) | 1696 | 160 | hit 1696, insert 1856 |
