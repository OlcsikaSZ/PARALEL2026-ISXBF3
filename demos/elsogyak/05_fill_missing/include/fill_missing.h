#ifndef FILL_MISSING_H
#define FILL_MISSING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel for missing elements.
   Input is specified as nonnegative integers, so -1 is safe. */
#define MISSING_VALUE (-1)

/* OpenCL-accelerated implementation (hidden behind a plain C API).
   - in:  input array (length n)
   - out: output array (length n)
   Returns 0 on success, nonzero on error. */
int fill_missing(const int* in, int* out, size_t n);

/* Sequential reference implementation for validation. */
void fill_missing_seq(const int* in, int* out, size_t n);

/* Generates a nonnegative integer array with scattered missing values.
   Constraints:
   - Missing values are never placed at the ends.
   - No two missing values are adjacent, so each missing has both neighbors known.
   missing_prob: approx probability (0..1) for an element to be missing (internal clamped).
   seed: deterministic seed for reproducibility.
   Returns number of missing values inserted. */
size_t generate_with_missing(int* a, size_t n, float missing_prob, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif
