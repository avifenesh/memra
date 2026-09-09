Plain baseline: 80.635647 HTTP tok/s; 81.450887 decode-interval tok/s; 12.277337 ms per plain token interval (client wall, includes sampler/transport, not isolated GPU).

| Vendor K | Observed accepted/round | HTTP ms/round | Break-even accepted/round | Engine ms/round | Decode break-even accepted/round |
|---|---:|---:|---:|---:|---:|
| 2 | 1.256039 | 34.209755 | 1.758526 | 33.865826 | 1.758402 |
| 4 | 1.444976 | 37.555889 | 2.028343 | 37.218905 | 2.031513 |
| 6 | 1.850000 | 39.588602 | 2.192253 | 39.210672 | 2.193744 |
| auto | 1.492683 | 35.894493 | 1.894376 | 35.549697 | 1.895554 |

Break-even uses a+1 tokens/round and ignores first-anchor/final-cap boundary corrections. The HTTP calculation compares HTTP with HTTP; the decode screening calculation uses engine round wall and the plain client interval. Neither is a hardware ceiling.

| Verify width | Current synchronized verify median ms | N current rounds | Prior tally GPU span ms | Prior N |
|---|---:|---:|---:|---:|
| 2 | 24.339360 | 4170 | 25.605873 | 10 |
| 3 | 28.227006 | 4405 | 29.379530 | 3 |
| 4 | 31.874907 | 2219 | 32.953999 | 3 |
| 5 | 39.189727 | 1112 | 40.268890 | 1 |
| 6 | 43.341594 | 178 | 44.319680 | 2 |
| 7 | 49.555427 | 389 | 50.458073 | 2 |
