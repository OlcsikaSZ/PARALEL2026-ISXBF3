#include "prime_check.h"
#include <stdlib.h>
#include <math.h>

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int prime_is_prime_seq(uint64_t n) {
    if (n < 2ULL) return 0;
    if (n == 2ULL) return 1;
    if ((n & 1ULL) == 0ULL) return 0;

    uint64_t r = (uint64_t)floor(sqrt((long double)n));
    for (uint64_t d = 3ULL; d <= r; d += 2ULL) {
        if (n % d == 0ULL) return 0;
    }
    return 1;
}

uint64_t generate_test_number(int want_prime, uint32_t seed) {
    // Generate numbers in a moderate range so the demo runs quickly.
    // Not cryptographic. Just for testing.
    uint32_t s = seed ? seed : 0xC0FFEEu;
    while (1) {
        uint64_t x = (uint64_t)(xorshift32(&s) % 50000000u) + 2ULL; // [2, 50M)
        if ((x & 1ULL) == 0ULL) x += 1ULL;
        int p = prime_is_prime_seq(x);
        if (want_prime) {
            if (p) return x;
        } else {
            if (!p) return x;
        }
    }
}
