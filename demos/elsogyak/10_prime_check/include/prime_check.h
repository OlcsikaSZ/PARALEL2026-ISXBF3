#ifndef PRIME_CHECK_H
#define PRIME_CHECK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum prime_mode_t {
    PRIME_MODE_SINGLE_DIVISOR = 0,   // each work-item checks one divisor
    PRIME_MODE_RANGE_DIVISOR  = 1,   // each work-item checks a range of divisors
    PRIME_MODE_PREPRIMES_HOST = 2,   // host precomputes primes up to sqrt(n), GPU checks
    PRIME_MODE_PREPRIMES_GPU  = 3    // GPU builds primes up to sqrt(n), then checks
} prime_mode_t;

// Returns 0 on success, non-zero on error.
// is_prime_out is set to 1 if n is prime, else 0.
int prime_is_prime(uint64_t n, prime_mode_t mode, int* is_prime_out);

// Sequential reference primality test (trial division by odd numbers).
int prime_is_prime_seq(uint64_t n);

// Generate test numbers: if want_prime=1 -> returns a prime (probabilistic),
// else returns a composite (probabilistic). Useful for quick demos.
uint64_t generate_test_number(int want_prime, uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif
