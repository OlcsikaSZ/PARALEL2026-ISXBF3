#include "minmax.h"
#include <stdlib.h>

void array_minmax_seq(const int* in, size_t n, int* out_min, int* out_max) {
    if (!in || n == 0 || !out_min || !out_max) return;
    int mn = in[0];
    int mx = in[0];
    for (size_t i = 1; i < n; ++i) {
        int v = in[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    *out_min = mn;
    *out_max = mx;
}

static unsigned int lcg(unsigned int* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

void generate_int_array(int* out, size_t n, int range, unsigned int seed) {
    if (!out || n == 0) return;
    if (range <= 0) {
        // avoid tiny ranges that trivialize min/max
        if (n > (size_t)1000000000u) range = 2147483647;
        else range = (int)(n * 2u + 1u);
        if (range <= 0) range = 2147483647;
    }
    unsigned int st = seed ? seed : 123456789u;
    for (size_t i = 0; i < n; ++i) {
        unsigned int r = lcg(&st);
        out[i] = (int)(r % (unsigned int)range);
    }
}
