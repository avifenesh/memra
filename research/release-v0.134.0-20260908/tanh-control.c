#include <math.h>
#include <stdio.h>
static unsigned calls;
float tanhf(float x) { ++calls; return (float)tanh((double)x); }
__attribute__((destructor)) static void report(void) { fprintf(stderr, "TANHF_F64_CONTROL calls=%u\n", calls); }
