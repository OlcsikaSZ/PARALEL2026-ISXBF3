#ifndef MINMAX_H
#define MINMAX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute min and max of a nonnegative integer array.
// OpenCL implementation is hidden behind this call.
// Returns 0 on success, nonzero on error.
int array_minmax(const int* in, size_t n, int* out_min, int* out_max);

// Sequential reference (for checking correctness)
void array_minmax_seq(const int* in, size_t n, int* out_min, int* out_max);

// Generate test input: nonnegative ints in [0, range).
// If range==0, it picks a default range (n*2+1).
void generate_int_array(int* out, size_t n, int range, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif
