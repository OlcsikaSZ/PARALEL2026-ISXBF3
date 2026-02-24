#include "fill_missing.h"
#include <stdlib.h>

void fill_missing_seq(const int* in, int* out, size_t n) {
    if (!in || !out || n == 0) return;
    for (size_t i = 0; i < n; ++i) out[i] = in[i];

    /* Assumption: every missing has both neighbors known. */
    for (size_t i = 1; i + 1 < n; ++i) {
        if (in[i] == MISSING_VALUE) {
            int left  = in[i - 1];
            int right = in[i + 1];
            out[i] = (left + right) / 2; /* integer average */
        }
    }
}

static unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

size_t generate_with_missing(int* a, size_t n, float missing_prob, unsigned int seed) {
    if (!a || n == 0) return 0;
    if (missing_prob < 0.0f) missing_prob = 0.0f;
    if (missing_prob > 0.9f) missing_prob = 0.9f; /* avoid degenerate */

    unsigned int st = (seed ? seed : 1u);
    /* Fill with some nonnegative integers */
    for (size_t i = 0; i < n; ++i) {
        a[i] = (int)(xorshift32(&st) % 10000u);
    }

    /* Insert missing values: not at ends, not adjacent. */
    size_t missing_count = 0;
    for (size_t i = 1; i + 1 < n; ++i) {
        /* don't place if neighbor already missing */
        if (a[i - 1] == MISSING_VALUE || a[i] == MISSING_VALUE || a[i + 1] == MISSING_VALUE)
            continue;

        /* random float in [0,1) */
        float r = (float)(xorshift32(&st) & 0xFFFFFFu) / (float)0x1000000u;
        if (r < missing_prob) {
            a[i] = MISSING_VALUE;
            missing_count++;
        }
    }
    return missing_count;
}
