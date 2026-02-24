#include "vector_add.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void vector_add_seq(const float* a, const float* b, float* out, size_t n)
{
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

int main(void)
{
    const size_t N = 1u << 20; // ~1M elems

    float* a = (float*)malloc(sizeof(float) * N);
    float* b = (float*)malloc(sizeof(float) * N);
    float* out = (float*)malloc(sizeof(float) * N);
    float* ref = (float*)malloc(sizeof(float) * N);

    if (!a || !b || !out || !ref) {
        fprintf(stderr, "[ERROR] malloc failed\n");
        return 1;
    }

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        // random-ish values in [-1, 1]
        a[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
        b[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }

    int err = vector_add(a, b, out, N);
    if (err != 0) {
        fprintf(stderr, "[ERROR] vector_add failed (code=%d)\n", err);
        return 2;
    }

    vector_add_seq(a, b, ref, N);

    // Verify (float, so use tolerance)
    float max_abs_err = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < N; ++i) {
        float e = fabsf(out[i] - ref[i]);
        if (e > max_abs_err) {
            max_abs_err = e;
            worst_i = i;
        }
    }

    const float tol = 1e-6f;
    if (max_abs_err > tol) {
        printf("NOT OK  max_abs_err=%g at i=%lu (out=%g, ref=%g)\n",
               max_abs_err, (unsigned long)worst_i, out[worst_i], ref[worst_i]);
        return 3;
    }

    printf("OK  N=%lu  max_abs_err=%g\n", (unsigned long)N, max_abs_err);

    free(a);
    free(b);
    free(out);
    free(ref);
    return 0;
}
