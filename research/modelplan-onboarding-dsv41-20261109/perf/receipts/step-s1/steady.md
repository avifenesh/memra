
### c1: calls by how long the card has been busy (8 runs of 2.31 to 2.36 s)
| arm | call | first 0.15 s: median ms (calls) | to 0.5 s | from 0.5 s: the steady figure | steady / burst |
|---|---|---|---|---|---|
| E | decode c1 | 4.531 (66) | 4.758 (147) | 4.843 (745) | 1.069 |
| G | decode c1 | 4.434 (62) | 4.705 (150) | 4.800 (746) | 1.083 |
| H | decode c1 | 4.587 (60) | 4.761 (149) | 4.875 (749) | 1.063 |
| P | decode c1 | 4.631 (63) | 4.785 (146) | 4.912 (749) | 1.061 |

### c8: calls by how long the card has been busy (8 runs of 2.87 to 9.92 s)
| arm | call | first 0.15 s: median ms (calls) | to 0.5 s | from 0.5 s: the steady figure | steady / burst |
|---|---|---|---|---|---|
| E | decode c8 | 10.874 (26) | 11.351 (62) | 11.375 (390) | 1.046 |
| G | decode c8 | 10.851 (22) | 11.247 (64) | 11.253 (392) | 1.037 |
| H | decode c8 | 38.479 (6) | 39.507 (18) | 39.953 (454) | 1.038 |
| P | decode c8 | 38.923 (6) | 40.099 (18) | 40.467 (454) | 1.040 |

ids of c1 (distinct digests over all runs and rounds):
     16 "gen_sha256":["e691399ef0ca6c29a9c30b5dc292cd9d2d38a19bf9053b6b6860f4d7ce924329"]

ids of c8 (distinct digests over all runs and rounds):
      2 "gen_sha256":["40d4a9c9b7bfc9a7941c159b11a29825d408c0fdb9c51cdd04c9a3f6172189b8","646b78c2edc8ef3f9639e274397e68
      2 "gen_sha256":["84c3951eb7ce21a281ae21c66591aef312157dd99436ca956f88a7de4b43b377","646b78c2edc8ef3f9639e274397e68
     12 "gen_sha256":["c0287c9fd93da26cd4c88461c3eb1aeff6f5f49afa4dcaa64ce45631c35fe995","94d9d2f81eee758fff2dc7588d8862
