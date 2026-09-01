# What the measured binary contains, versus what was committed

`hybrid_forward-as-measured.patch` (sha256 `8d1ca90e1dc715c83ff62efecc289e89562663c5b177c1f880cf7aee520ac9ad`)
is the exact working-tree change that built `/root/memra-server.gsuffix` (md5 `0216d7011fb3`)
on top of `e3faf5a17c98f534654572608d708b40d1edb8c6`. Build receipt: `gs-build.txt`.

It differs from the committed code in ONE line, deliberately: the patch has
`MEMRA_STEP_GEMM_PRIME_SUFFIX` defaulting ON, and the commit defaults it OFF pending these
receipts. That delta cannot reach any measurement, because every arm of `gs-battery.sh` names
the door explicitly (`=0` on the off arm, `=1` on the on and canary arms) rather than leaning
on a default. The `seq_end` fix is identical in both and is not behind any door.
