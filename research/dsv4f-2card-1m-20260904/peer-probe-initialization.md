# Peer probe initialization ordering

2026-09-05. The standalone `peer-copy-direction-probe` uploaded its expected
pattern and destination poison from pageable `std::vector` buffers with
`cuMemcpyHtoD_v2`, then ran the tested copy on non-blocking streams. That does
not guarantee input readiness: CUDA's documented synchronous pageable H2D may
return after staging, while DMA to device is still in flight. The default
stream upload is not implicitly ordered with these non-blocking streams.

Primary reference, read today:
https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html

The fix synchronizes each initializing context after its H2D. It does not add
a fence inside or change any of the four copy programs being tested. The old
initialization can be explicitly selected with `--unsafe-init` for diagnostics.
The DSV4 runtime's separate pooled-allocation peer probe already synchronizes
its uploads; that code was inspected and does not need this fix.

On two RTX PRO 6000 Blackwell Max-Q cards, the old probe reported intermittent
1 MiB mismatches, first in pull-D2D and later in producer-issued peer copy.
Those observations cannot qualify or reject the transport because their input
precondition was not guaranteed. In an interleaved five-repeat control using
one binary, unsafe initialization failed 3/5 runs, ordered initialization
passed 5/5. Two earlier ordered five-repeat passes were also green.
Every ordered run checked 24 cells: four programs, both directions and
16 KiB/1 MiB/64 MiB. No timing or power verdict follows from this test.

Corrected binary SHA256:
`d1aa185d11f1cb0f1abd17528fb1461ce40c561c88538f3853548483ae515dec`.
Original binary: `6ec26fb784a0f0f99d261ccced841ec97ad97e98fa61cff15f4a194470b9ee5a`.
The full transport gate was restored before model staging proceeded.
