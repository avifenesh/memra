| idx | role | tenant | turn | prompt | cold digest[:16] | slru cached | slru computed | slru events | lru cached | lru computed | lru events |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 0 | seed1 | cohort-1 | - | 7800 | 7680ef522093d785 | 0 | 7800 | insert 7800 | 0 | 7800 | insert 7800 |
| 1 | seed2 | cohort-1 | - | 7800 | 7680ef522093d785 | 7800 | 0 | hit 7800 | 7800 | 0 | hit 7800 |
| 2 | seed1 | cohort-2 | - | 8000 | d3441c016fa4f364 | 0 | 8000 | insert 8000 | 0 | 8000 | insert 8000 |
| 3 | seed2 | cohort-2 | - | 8000 | d3441c016fa4f364 | 8000 | 0 | hit 8000 | 8000 | 0 | hit 8000 |
| 4 | seed1 | cohort-3 | - | 8200 | 7e3fdb9b0a70022f | 0 | 8200 | insert 8200 | 0 | 8200 | insert 8200 |
| 5 | seed2 | cohort-3 | - | 8200 | 7e3fdb9b0a70022f | 8200 | 0 | hit 8200 | 8200 | 0 | hit 8200 |
| 6 | seed1 | cohort-4 | - | 8400 | 0243029b1aabb886 | 0 | 8400 | insert 8400 | 0 | 8400 | insert 8400 |
| 7 | seed2 | cohort-4 | - | 8400 | 0243029b1aabb886 | 8400 | 0 | hit 8400 | 8400 | 0 | hit 8400 |
| 8 | loop | grow | 1 | 27300 | 6eb8e535cc2c3bbe | 0 | 27300 | insert 27300, evict 7800 (Protected), evict 8000 (Protected) | 0 | 27300 | insert 27300, evict 7800 (Protected), evict 8000 (Protected) |
| 9 | loop | grow | 2 | 27600 | 24a97a2867768b2d | 27300 | 300 | hit 27300, insert 27600, evict 8200 (Probation), evict 8400 (Protected), demote x1 | 27300 | 300 | hit 27300, insert 27600, evict 8200 (Protected), evict 8400 (Protected) |
| 10 | loop | grow | 3 | 27900 | 65eeb1ef8f716af1 | 27600 | 300 | hit 27600, insert 27900, evict 27300 (Probation), demote x1 | 27600 | 300 | hit 27600, insert 27900, evict 27300 (Protected) |
| 11 | return | cohort-1 | 3 | 8100 | 7632408d9a818cfc | 0 | 8100 | insert 8100, evict 27900 (Probation) | 0 | 8100 | insert 8100, evict 27600 (Protected) |
| 12 | loop | grow | 4 | 28200 | 22f023976ebc22fd | 27600 | 600 | hit 27600, insert 28200, evict 8100 (Probation) | 27900 | 300 | hit 27900, insert 28200, evict 8100 (Probation) |
| 13 | loop | grow | 5 | 28500 | 50163e8c56428f00 | 28200 | 300 | hit 28200, insert 28500, evict 27600 (Probation), demote x1 | 28200 | 300 | hit 28200, insert 28500, evict 27900 (Protected) |
| 14 | loop | grow | 6 | 28800 | 94d76ac2f6c4aa98 | 28500 | 300 | hit 28500, insert 28800, evict 28200 (Probation), demote x1 | 28500 | 300 | hit 28500, insert 28800, evict 28200 (Protected) |
| 15 | return | cohort-2 | 6 | 8300 | bb95b75c9882e0cf | 0 | 8300 | insert 8300, evict 28800 (Probation) | 0 | 8300 | insert 8300, evict 28500 (Protected) |
| 16 | loop | grow | 7 | 29100 | 0d6d7198ce6e8bc4 | 28500 | 600 | hit 28500, insert 29100, evict 8300 (Probation) | 28800 | 300 | hit 28800, insert 29100, evict 8300 (Probation) |
| 17 | loop | grow | 8 | 29400 | 0cab0347a07327d1 | 29100 | 300 | hit 29100, insert 29400, evict 28500 (Probation), demote x1 | 29100 | 300 | hit 29100, insert 29400, evict 28800 (Protected) |
| 18 | loop | grow | 9 | 29700 | 553f5eef1c29307f | 29400 | 300 | hit 29400, insert 29700, evict 29100 (Probation), demote x1 | 29400 | 300 | hit 29400, insert 29700, evict 29100 (Protected) |
| 19 | return | cohort-3 | 9 | 8500 | c309d824900a712c | 0 | 8500 | insert 8500, evict 29700 (Probation) | 0 | 8500 | insert 8500, evict 29400 (Protected) |
| 20 | loop | grow | 10 | 30000 | d3cbc5d1ae768103 | 29400 | 600 | hit 29400, insert 30000, evict 8500 (Probation) | 29700 | 300 | hit 29700, insert 30000, evict 8500 (Probation) |
| 21 | loop | grow | 11 | 30300 | b6b5b760960efe97 | 30000 | 300 | hit 30000, insert 30300, evict 29400 (Probation), demote x1 | 30000 | 300 | hit 30000, insert 30300, evict 29700 (Protected) |
| 22 | loop | grow | 12 | 30600 | d5b3015fc5c4f9f7 | 30300 | 300 | hit 30300, insert 30600, evict 30000 (Probation), demote x1 | 30300 | 300 | hit 30300, insert 30600, evict 30000 (Protected) |
| 23 | return | cohort-4 | 12 | 8700 | 58fcbed59f3d168f | 0 | 8700 | insert 8700, evict 30600 (Probation) | 0 | 8700 | insert 8700, evict 30300 (Protected) |
| 24 | final | cohort-1 | 12 | 8400 | 24a97a2867768b2d | 0 | 8400 | insert 8400 | 0 | 8400 | insert 8400 |
| 25 | final | cohort-2 | 12 | 8600 | e31d97b3c8ee4728 | 0 | 8600 | insert 8600, evict 8700 (Probation) | 0 | 8600 | insert 8600, evict 30600 (Probation) |
| 26 | final | cohort-3 | 12 | 8800 | 8861c44b40974cb7 | 0 | 8800 | insert 8800, evict 8400 (Probation) | 0 | 8800 | insert 8800 |
| 27 | final | cohort-4 | 12 | 9000 | 16eff1da2acb0d16 | 0 | 9000 | insert 9000, evict 8600 (Probation) | 8700 | 300 | hit 8700, insert 9000 |
