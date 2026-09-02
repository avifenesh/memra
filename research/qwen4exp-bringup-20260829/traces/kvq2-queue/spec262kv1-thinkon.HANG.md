# spec262kv1-thinkon HANG receipt 2026-09-01T23:52:54Z
invocation: MEMRA_Q4E_SEAMS=idxsel qwen4exp_real_gate.downsel q48fn-yarn1m --label r3spec262kv1-thinkon --mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1 --spec-k 5 --ladder 262144 --ladder-ids ladder-ids.txt --ladder-chunk 2048 --ladder-decode 36 --ladder-kv-dev1 --ladder-spec 5 --ladder-spec-shape thinkon (CUDA_VISIBLE_DEVICES=2,3; 4x RTX PRO 6000 box, 360 GB RAM)
state at kill: 113 min after lock acquire, receipt files untouched since post-load (22:05:58); nvidia-smi dmon card2 sm=100% mem=0% for 3 consecutive samples; card3 memory flat 27187 MiB for ~90 min; host thread 1 R (poll loop, 107 CPU-min) + 3 S; no OOM (dmesg clean for this pid)
prior: same route never produced a row in 45 min on the 2-card box before it was reclaimed (2026-09-01 AM)
== nvidia-smi topo -m
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	PHB	PHB	0-119	0		N/A
GPU1	PHB	 X 	PHB	PHB	0-119	0		N/A
GPU2	PHB	PHB	 X 	PHB	0-119	0		N/A
GPU3	PHB	PHB	PHB	 X 	0-119	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
== p2p status
 	[4mGPU0	GPU1	GPU2	GPU3	[0m
 GPU0	X	OK	OK	OK	
 GPU1	OK	X	OK	OK	
 GPU2	OK	OK	X	OK	
 GPU3	OK	OK	OK	X	

Legend:

