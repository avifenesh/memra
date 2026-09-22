# Native preparation status

The initial remote checkout was verified clean at
`5967429fb9c02a9fbc813d94dd063405cf5a53d3`. CPU-only build and staging ran with
`CUDA_VISIBLE_DEVICES` empty and build jobs capped at eight.

After an access outage and coordinator-managed recovery, no tmux session survived.
`preflight-001/build.log` has no matching build.exit: this build attempt is interrupted,
not passing. No cause is assigned from that partial log.

`preflight-001/stage.exit` is zero. The staged official Qwen3-0.6B artifact was rehashed
after recovery against its frozen revision and weight hash. The source manifest and
completed staging log were copied off the host.

CPU build attempt002 resumes in a fresh receipt namespace. GPU work has not run in this
lane. The assigned storage is overlay-backed scratch; no physical NVMe ancestry is claimed.

## Recovered CPU build

`preflight-002/build.exit` is zero and its complete build log is retained locally. Real CUDA
13.1 fatbins and the requested release binaries compiled successfully in 4m26s. This is a
CPU compilation result, not GPU correctness evidence.

The remote checkout then advanced cleanly to
`b47d87e5a407178abcdec13a58064f43a804b898`, and build attempt003 was launched with
GPU visibility disabled so the interruption-control fix has a consistent source/build record.
Access failed again before completion could be read back. Build003 is unverified. No GPU
lease, CUDA model execution, or native qualification case has been started by this lane.
