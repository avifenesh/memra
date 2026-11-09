434 probe lines from 1 logs
| kind | rows <= | calls | members max | peak MB median | peak MB max | peak bytes/row max (excl. slab growth) | slab growth MB max | retained MB max | slab after MB max |
|---|---|---|---|---|---|---|---|---|---|
| batch | 128 | 2 | 3 | 30 | 30 | 372119 | 0 | 2 | 1527 |
| chunk | 4096 | 1 | 1 | 1942 | 1942 | 287684 | 763 | 856 | 1527 |
| prime | 64 | 56 | 1 | 9 | 75 | 3645444 | 0 | 1 | 1527 |
| prime | 128 | 5 | 1 | 27 | 70 | 729092 | 0 | 3 | 1527 |
| prime | 256 | 13 | 1 | 73 | 73 | 285636 | 0 | 5 | 1527 |
| prime | 512 | 18 | 1 | 119 | 146 | 289732 | 0 | 11 | 1527 |
| prime | 1024 | 333 | 1 | 292 | 297 | 289732 | 0 | 25 | 1527 |
| prime | 2048 | 5 | 1 | 585 | 2231 | 716532 | 763 | 1688 | 763 |
| prime | 4096 | 1 | 1 | 1124 | 1124 | 285636 | 0 | 81 | 1527 |
