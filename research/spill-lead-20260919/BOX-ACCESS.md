# Rented 5090 box — access + rules for lane sessions (2026-09-19)

Owner-approved development rental (marketplace GPU, interruptible; NOT production). Details of the box are
private receipt metadata; the connection below uses the operator's existing SSH identity only.

- Connect **via the provider's SSH proxy** (the direct-IP path is flaky). The proxy host/port is a
  deployment fact and lives ONLY in the gitignored `LANE-LOCAL.md` at the repo root (public-boundary
  rule); use `ssh -o ConnectTimeout=25 -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 -p <port> root@<proxy-host>`.
  Bounded retries only (≤3, 30 s backoff); a failed connection proves no state change. Filter the
  banner lines (`Welcome|Have fun`) from captured output.
- Checkout on the box: `/root/memra-spill` (branch `lane/spill-integ2-20260919`; `git fetch` + checkout
  your lane branch in a SEPARATE worktree under `/root/wt-<lane>` — never reset `/root/memra-spill`).
  Toolchain: `/root/.cargo/bin/cargo` (stable), nvcc 13.1 at `/usr/local/cuda`; `MEMRA_NVCC`/`MEMRA_CUDA_ARCH=120a`.
- Receipts: `/root/spill-receipts/` (bootstrap + first hour). Artifacts: `/root/artifacts/` (Qwen3.8-27B
  NVFP4+Q5K GGUF, sha256-verified) — read-only for lanes; never delete.
- **GPU work only through the collector** `python3 tools/tier-battery.py --rig rtx5090 --timeout N --out <dir> --execute <cmd>`
  (it takes `/tmp/memra-5090.lock`). Never run a GPU binary bare. Check `pgrep -fa "run-gen|run-spec|qwen4exp|storage-bench|pp-transport"`
  (beware self-match) and `nvidia-smi --query-compute-apps=pid,process_name --format=csv` before starting; if busy, wait.
- Compiles (`cargo build --release -j 16`) are allowed while another lane's GPU cell runs (correctness
  cells, not timing); cap `-j` at 16 so the box stays responsive.
- The container root is **overlay**; NVMe ancestry is unproven. Label every storage number
  "overlay, development, not spill speed". `O_DIRECT` on overlayfs may be refused — record verbatim.
- Spot box: it can vanish. Sync results back to your lane (`rsync` under `research/spill-<x>-20260919/rented-5090-20260919/`)
  before reporting; nothing counts until it is committed and pushed.
- Never: touch `/root/artifacts`, `/root/memra-spill` checkout, other lanes' worktrees or receipts; embed
  credentials; run anything on any other host.

## Transport (added 2026-09-20 after intermittent connect timeouts from the operator Mac)
The lead keeps a persistent SSH ControlMaster to BOX2. Lanes MUST reuse it instead of opening new
TCP connections (each new connection has been timing out intermittently):
`ssh -o ControlPath=$HOME/.ssh/cm/box2 -o ControlMaster=no -o BatchMode=yes root@<proxy-host-from-LANE-LOCAL>`
(the host/port matter only when the socket is absent; with the socket present the connection is
reused). Check first: `ssh -O check -o ControlPath=$HOME/.ssh/cm/box2 root@<proxy-host>`. For
`scp`/`rsync`: `-o ControlPath=$HOME/.ssh/cm/box2`. If the socket is gone, report it to the lead
(who re-establishes it with `ControlMaster=auto ControlPersist=yes`) rather than looping retries.
A direct-IP path also exists in LANE-LOCAL.md as a fallback.

## Proxy vs direct path (2026-09-20)

The provider's ssh proxy has refused connections while the instance stayed `running`; the direct
port (both in `LANE-LOCAL.md`) kept working. The lead re-establishes the master over whichever path
answers. Because the `ControlPath` is a fixed socket (no `%h`), lanes change nothing: `ssh -O check`
then `ssh -o ControlPath=$HOME/.ssh/cm/box2 -o ControlMaster=no root@<either-host> <cmd>` multiplexes
over the live master regardless of the hostname given.

## Rig change 2026-09-20: BOX3 replaces BOX2

The bid-market 5090 (BOX2) was host-stopped four times in one day; the owner moved the lane to a
fixed-price spot **RTX PRO 6000 Blackwell** (BOX3) — the memra target card class, 96 GB, 600 W cap, a
real block-device root (no overlayfs; NVMe ancestry still provider-claimed, hidden by virtio). Rig profile
`pro-single` in `tools/tier-battery.py` / `tools/tier-rig-bootstrap.sh`; lock `/tmp/memra-gpu.lock` per the
lock table. Socket `~/.ssh/cm/box3`; same rules (`ssh -O check` first, never a fresh connection; the lead's
keeper re-establishes within ~30 s). Host, id, price live in `LANE-LOCAL.md` only. BOX2 receipts stay valid as
5090 evidence; **no timing is compared across the two boxes**, and every cell records its own power envelope.
