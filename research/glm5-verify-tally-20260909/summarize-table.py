#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('summary',type=Path);a=p.parse_args()
data=json.loads(a.summary.read_text())
for t,block in sorted(data.items(),key=lambda x:int(x[0])):
    print(f'## t={t}, N={block["rounds"]} verify rounds\n')
    print('| Kernel | Launches/round | us/round | Mean us | Weight MB/round | Effective GB/s | % of 8 TB/s |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row in block['ranked'][:15]:
        bw=[f'{row["weight_bytes_per_round"]/1e6:.3f}',f'{row["effective_GBs"]:.1f}',f'{row["bw_pct"]:.2f}'] if 'weight_bytes_per_round' in row else ['n/a']*3
        print(f'| `{row["kernel"]}` | {row["launches_per_round"]:.3f} | {row["total_us_per_round"]:.3f} | {row["mean_us"]:.3f} | '+ ' | '.join(bw)+' |')
    print(f'\nVerify launches/round: {block["total_launches_per_round"]:.3f}; trunk launches/layer/round: {block["trunk_launches_per_layer"]:.3f}.')
    print(f'Kernel sum {block["total_kernel_us_per_round"]:.3f} us/round; GPU span {block["gpu_span_us_per_round"]:.3f} us/round; GPU gaps {block["gpu_gaps_us_per_round"]:.3f} us/round; NVTX host span {block["nvtx_host_us_per_round"]:.3f} us/round.\n')
