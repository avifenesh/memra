"""Which indices torch.topk(sorted=False) keeps among ties at the cutoff on this device, at the shapes the vendor
indexer uses (rows x width, bf16 scores, k = 512 or fewer). memra's reference assumes the lowest index wins."""
import json, sys
import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
g = torch.Generator().manual_seed(0)
out = []
for rows, width, k in [(4, 64, 6), (8, 1024, 512), (8, 4096, 512), (4, 16384, 512), (2, 131072, 512), (1, 262144, 512)]:
    # few distinct values so that many entries tie at the k-th value
    vals = torch.randint(0, 8, (rows, width), generator=g).to(torch.bfloat16)
    x = vals.to(dev)
    got = x.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values.cpu()
    lowest = []
    for r in range(rows):
        v = vals[r].float()
        kth = v.sort(descending=True).values[k - 1]
        above = (v > kth).nonzero().flatten()
        ties = (v == kth).nonzero().flatten()[: k - above.numel()]
        lowest.append(torch.cat([above, ties]).sort().values)
    lowest = torch.stack(lowest)
    out.append({"rows": rows, "width": width, "k": k, "lowest_index_wins": bool(torch.equal(got, lowest)),
                "rows_matching": int((got == lowest).all(-1).sum())})
print(json.dumps({"device": dev, "torch": torch.__version__, "cases": out}, indent=1))
