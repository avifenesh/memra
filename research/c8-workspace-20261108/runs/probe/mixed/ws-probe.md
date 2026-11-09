63 probe lines from 1 logs
| kind | rows <= | calls | members max | peak MB median | peak MB max | peak bytes/row max (excl. slab growth) | slab growth MB max | retained MB max | slab after MB max |
|---|---|---|---|---|---|---|---|---|---|
| batch | 128 | 15 | 4 | 42 | 57 | 444356 | 0 | 3 | 1527 |
| batch | 256 | 19 | 6 | 86 | 114 | 444356 | 0 | 5 | 1527 |
| batch | 512 | 8 | 6 | 127 | 171 | 444356 | 0 | 8 | 1527 |
| chunk | 64 | 1 | 1 | 15 | 15 | 363050 | 0 | 1 | 1527 |
| chunk | 512 | 1 | 1 | 137 | 137 | 285636 | 0 | 10 | 1527 |
| chunk | 4096 | 1 | 1 | 1942 | 1942 | 287684 | 763 | 856 | 1527 |
| prime | 64 | 11 | 1 | 7 | 18 | 379789 | 0 | 1 | 1527 |
| prime | 512 | 4 | 1 | 146 | 146 | 285636 | 0 | 11 | 1527 |
| prime | 2048 | 3 | 1 | 585 | 2231 | 716532 | 763 | 1688 | 763 |
