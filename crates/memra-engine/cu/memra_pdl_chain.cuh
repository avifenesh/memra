// memra_pdl_chain.cuh: programmatic dependent launch for the DSv4 decode chain.
//
// A kernel launched through `memra_chain_launch` carries the programmatic stream serialization
// attribute while `memra_pdl_chain_on` is set, so its blocks may be scheduled while the kernel
// before it on the stream is still running. Every such kernel opens with
// `MEMRA_PDL_CHAIN_ENTRY()`: `griddepcontrol.wait` holds each thread until the preceding grid
// has completed and its writes are visible, so no load or store of the kernel's own body can
// overlap its predecessor. The body, its reads and its arithmetic are unchanged: the attribute
// moves only when blocks are dispatched, never what they compute. `launch_dependents` then lets
// the next kernel's launch start while this one runs. Without the attribute (the switch off, or
// a predecessor that is not a kernel) both instructions are no-ops.
//
// A kernel may run a prologue before the entry that reads only checkpoint constants: weights and
// scales that no kernel writes. It opens with `MEMRA_PDL_PRE_WAIT(p, ...)` naming those pointer
// parameters, and those loads may overlap the predecessor (memra #710: a GEMV fetches its first
// weight chunks while the kernel before it finishes). Everything that reads an activation or
// writes memory stays after the entry.
//
// tools/check-pdl-chain.py refuses a kernel launched through the chain launcher whose body does
// not open with the entry, or with a pre-wait declaration whose prologue touches a pointer
// parameter it did not name, names a non-const pointer, or has no entry after it.
#pragma once
#include <cuda_runtime.h>
#include <utility>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MEMRA_PDL_CHAIN_ENTRY()                                  \
    do {                                                         \
        asm volatile("griddepcontrol.wait;" ::: "memory");       \
        asm volatile("griddepcontrol.launch_dependents;" ::: ); \
    } while (0)
#else
#define MEMRA_PDL_CHAIN_ENTRY() \
    do {                        \
    } while (0)
#endif

// Declares the pointer parameters a pre-entry prologue may read. Expands to nothing: the
// declaration is for tools/check-pdl-chain.py.
#define MEMRA_PDL_PRE_WAIT(...) \
    do {                        \
    } while (0)

extern "C" int memra_pdl_chain_on;

template <class... K> struct MemraChainLaunch {
    void (*kernel)(K...);
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attr[1];
    template <class... A> void operator()(A&&... args) {
        if (memra_pdl_chain_on) {
            attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
        }
        // Errors reach the caller through cudaGetLastError, as with a <<<>>> launch.
        (void)cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...);
    }
};

template <class... K>
MemraChainLaunch<K...> memra_chain_launch(void (*kernel)(K...), dim3 grid, dim3 block,
                                          size_t smem, cudaStream_t stream) {
    MemraChainLaunch<K...> l{};
    l.kernel = kernel;
    l.cfg.gridDim = grid;
    l.cfg.blockDim = block;
    l.cfg.dynamicSmemBytes = smem;
    l.cfg.stream = stream;
    l.cfg.attrs = nullptr;
    l.cfg.numAttrs = 0;
    return l;
}
