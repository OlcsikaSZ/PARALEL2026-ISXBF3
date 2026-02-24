#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <stddef.h>

/**
 * Adds two real vectors: out[i] = a[i] + b[i]
 * 
 * OpenCL is used internally, but this API is intentionally "clean":
 * caller does not need to know anything about OpenCL.
 *
 * Returns 0 on success, non-zero on error.
 */
int vector_add(const float* a, const float* b, float* out, size_t n);

#endif
