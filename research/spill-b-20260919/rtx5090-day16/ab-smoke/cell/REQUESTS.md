| idx | role | tenant | turn | prompt | cold digest[:16] | slru cached | slru computed | slru events | lru cached | lru computed | lru events |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 0 | seed1 | cohort-1 | - | 1250 | 035b948dc4d492de | 0 | 1250 | insert 1250 | 0 | 1250 | insert 1250 |
| 1 | seed2 | cohort-1 | - | 1250 | 035b948dc4d492de | 1250 | 0 | hit 1250 | 1250 | 0 | hit 1250 |
| 2 | seed1 | cohort-2 | - | 1350 | 83f2152f545205ba | 0 | 1350 | insert 1350 | 0 | 1350 | insert 1350 |
| 3 | seed2 | cohort-2 | - | 1350 | 83f2152f545205ba | 1350 | 0 | hit 1350 | 1350 | 0 | hit 1350 |
| 4 | seed1 | cohort-3 | - | 1450 | f1d5e9804c3c2119 | 0 | 1450 | insert 1450 | 0 | 1450 | insert 1450 |
| 5 | seed2 | cohort-3 | - | 1450 | f1d5e9804c3c2119 | 1450 | 0 | hit 1450 | 1450 | 0 | hit 1450 |
| 6 | seed1 | cohort-4 | - | 1550 | 1aa4fad6f6cf0cf5 | 0 | 1550 | insert 1550 | 0 | 1550 | insert 1550 |
| 7 | seed2 | cohort-4 | - | 1550 | 1aa4fad6f6cf0cf5 | 1550 | 0 | hit 1550 | 1550 | 0 | hit 1550 |
| 8 | loop | grow | 1 | 11000 | 9745f43fe07640e1 | 0 | 11000 | insert 11000, evict 1250 (Protected), evict 1350 (Protected) | 0 | 11000 | insert 11000, evict 1250 (Protected), evict 1350 (Protected) |
| 9 | loop | grow | 2 | 11150 | 66d394ced7203380 | 11000 | 150 | hit 11000, insert 11150, demote x1 | 11000 | 150 | hit 11000, insert 11150 |
| 10 | loop | grow | 3 | 11300 | 24a97a2867768b2d | 11150 | 150 | hit 11150, insert 11300, demote x1 | 11150 | 150 | hit 11150, insert 11300 |
| 11 | return | cohort-1 | 3 | 1400 | 2ac0f4d0262cd82c | 0 | 1400 | insert 1400, evict 11300 (Probation) | 0 | 1400 | insert 1400, evict 11150 (Protected) |
| 12 | loop | grow | 4 | 11450 | 876c0bf0f375834e | 11150 | 300 | hit 11150, insert 11450 | 11300 | 150 | hit 11300, insert 11450 |
| 13 | loop | grow | 5 | 11600 | 7abc2552932a6a15 | 11450 | 150 | hit 11450, insert 11600, demote x1 | 11450 | 150 | hit 11450, insert 11600 |
| 14 | loop | grow | 6 | 11750 | 24a97a2867768b2d | 11600 | 150 | hit 11600, insert 11750, demote x1 | 11600 | 150 | hit 11600, insert 11750 |
| 15 | return | cohort-2 | 6 | 1500 | 606f05730107bce9 | 0 | 1500 | insert 1500, evict 11750 (Probation) | 0 | 1500 | insert 1500, evict 11600 (Protected) |
| 16 | loop | grow | 7 | 11900 | 22f023976ebc22fd | 11600 | 300 | hit 11600, insert 11900 | 11750 | 150 | hit 11750, insert 11900 |
| 17 | loop | grow | 8 | 12050 | 01062972f3feb195 | 11900 | 150 | hit 11900, insert 12050, demote x1 | 11900 | 150 | hit 11900, insert 12050 |
| 18 | loop | grow | 9 | 12200 | 22f023976ebc22fd | 12050 | 150 | hit 12050, insert 12200, demote x1 | 12050 | 150 | hit 12050, insert 12200 |
| 19 | return | cohort-3 | 9 | 1600 | adbf88716a2d1aef | 0 | 1600 | insert 1600, evict 12200 (Probation) | 0 | 1600 | insert 1600, evict 12050 (Protected) |
| 20 | loop | grow | 10 | 12350 | 4415b7e361fc6f6b | 12050 | 300 | hit 12050, insert 12350 | 12200 | 150 | hit 12200, insert 12350 |
| 21 | loop | grow | 11 | 12500 | b6b5b760960efe97 | 12350 | 150 | hit 12350, insert 12500, demote x1 | 12350 | 150 | hit 12350, insert 12500 |
| 22 | loop | grow | 12 | 12650 | a60e5ad8da5b0484 | 12500 | 150 | hit 12500, insert 12650, demote x1 | 12500 | 150 | hit 12500, insert 12650 |
| 23 | return | cohort-4 | 12 | 1700 | d130dbcf85337f3d | 0 | 1700 | insert 1700, evict 12650 (Probation) | 0 | 1700 | insert 1700, evict 12500 (Protected) |
| 24 | final | cohort-1 | 12 | 1550 | f1a33345e94b7e62 | 0 | 1550 | insert 1550 | 0 | 1550 | insert 1550 |
| 25 | final | cohort-2 | 12 | 1650 | 3ca9a6e466c2dfb8 | 0 | 1650 | insert 1650, evict 1700 (Probation) | 0 | 1650 | insert 1650, evict 12650 (Probation) |
| 26 | final | cohort-3 | 12 | 1750 | 7f6850027a758030 | 0 | 1750 | insert 1750, evict 1550 (Probation) | 0 | 1750 | insert 1750 |
| 27 | final | cohort-4 | 12 | 1850 | d130dbcf85337f3d | 0 | 1850 | insert 1850, evict 1650 (Probation) | 1700 | 150 | hit 1700, insert 1850 |
