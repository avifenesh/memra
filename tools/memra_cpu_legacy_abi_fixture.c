/* Symbol-admission fixture only. It must never execute a numerical request. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
struct memra_cpu_expert_v2;
uint32_t memra_cpu_experts_abi_version(void) { return 2; }
int memra_cpu_moe_token_v2(const struct memra_cpu_expert_v2 *experts, int32_t count,
                         const float *input, float *output, int32_t threads,
                         char *error, size_t capacity) {
    (void)experts; (void)count; (void)input; (void)output; (void)threads;
    (void)error; (void)capacity;
    abort(); /* A fallback is a test failure, not a fake successful inference. */
}
void memra_cpu_expert_cache_stats_v2(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) {
    *a = *b = *c = *d = 0;
}
void memra_cpu_expert_profile_stats_v2(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) {
    *a = *b = *c = *d = 0;
}
