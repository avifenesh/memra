# Hardware cards

One card per hardware target: the build, the defaults keyed on that device, the models measured
there and what does not transfer. Pick the card for the GPU you have.

| Card | Build target | Push-gated | Prebuilt | Maintained |
|---|---|---|---|---|
| [RTX PRO 6000 Blackwell](rtx-pro-6000-blackwell.md) | `sm_120a` | yes | yes | yes, the primary optimized target |
| [RTX 5090 and RTX 50-series](rtx-5090.md) | `sm_120a` | yes | yes | yes, the primary optimized target |
| [DGX Spark (GB10)](dgx-spark.md) | `sm_121a` | yes | no | yes |
| [H100](h100.md) | `sm_90a` | no | no | no; source build, gates run on Hopper directly |
| [B200](b200.md) | `sm_100a` | no | no | no; best effort |
| [Ada](ada.md) | `sm_89` | no | no | no; compatibility reports |

A device is maintained when its build target is push-gated, so every push that reaches the target
is proven on it (`tools/change-impact.toml`, `tools/change_impact.py`). The build auto-detects the
target from the card's compute capability; `MEMRA_CUDA_ARCH` overrides it
([INSTALLATION.md](../INSTALLATION.md)). A rig label in a receipt names the hardware and its
measurement conditions; conditions per measurement are in [PERFORMANCE.md](../PERFORMANCE.md).
