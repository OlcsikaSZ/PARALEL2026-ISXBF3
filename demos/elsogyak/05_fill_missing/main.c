#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fill_missing.h"

static int check_equal(const int* a, const int* b, size_t n, size_t* worst_i) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (worst_i) *worst_i = i;
            return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    size_t N = 1u << 20; /* default: 1,048,576 */
    float miss_p = 0.02f;
    unsigned int seed = 1234u;

    if (argc >= 2) {
        long v = atol(argv[1]);
        if (v > 3) N = (size_t)v;
    }
    if (argc >= 3) {
        double p = atof(argv[2]);
        if (p >= 0.0 && p <= 0.9) miss_p = (float)p;
    }

    int* in  = (int*)malloc(N * sizeof(int));
    int* out = (int*)malloc(N * sizeof(int));
    int* ref = (int*)malloc(N * sizeof(int));
    if (!in || !out || !ref) {
        fprintf(stderr, "malloc failed\n");
        free(in); free(out); free(ref);
        return 1;
    }

    size_t missing_count = generate_with_missing(in, N, miss_p, seed);

    int err = fill_missing(in, out, N);
    if (err != 0) {
        fprintf(stderr, "fill_missing(OpenCL) failed with code %d\n", err);
        free(in); free(out); free(ref);
        return 2;
    }

    fill_missing_seq(in, ref, N);

    size_t worst_i = 0;
    int ok = check_equal(out, ref, N, &worst_i);

    if (!ok) {
        printf("NOT OK  mismatch at i=%lu  out=%d  ref=%d  in[i]=%d\n",
               (unsigned long)worst_i, out[worst_i], ref[worst_i], in[worst_i]);
        /* show neighbors for debugging */
        if (worst_i > 0 && worst_i + 1 < N) {
            printf("neighbors: in[i-1]=%d  in[i+1]=%d\n", in[worst_i-1], in[worst_i+1]);
        }
    } else {
        printf("OK  N=%lu  missing=%lu  (p=%.3f)\n",
               (unsigned long)N, (unsigned long)missing_count, (double)miss_p);
    }

    free(in);
    free(out);
    free(ref);
    return ok ? 0 : 3;
}
