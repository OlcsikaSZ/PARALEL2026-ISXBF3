#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include "prime_check.h"

static void usage(const char* exe) {
    printf("Usage:\n");
    printf("  %s <n> [mode]\n", exe);
    printf("\nModes:\n");
    printf("  0 = each work-item checks one divisor\n");
    printf("  1 = each work-item checks a divisor range\n");
    printf("  2 = host precomputes primes up to sqrt(n), GPU checks primes table\n");
    printf("  3 = GPU builds primes up to sqrt(n), then checks primes table\n");
    printf("\nExamples:\n");
    printf("  %s 9999991 0\n", exe);
    printf("  %s 9999991 1\n", exe);
    printf("  %s 9999991 2\n", exe);
    printf("  %s 9999991 3\n", exe);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    uint64_t n = 0;
#if defined(_MSC_VER)
    n = _strtoui64(argv[1], NULL, 10);
#else
    n = (uint64_t)strtoull(argv[1], NULL, 10);
#endif

    int mode_i = 1;
    if (argc >= 3) mode_i = atoi(argv[2]);
    if (mode_i < 0 || mode_i > 3) { usage(argv[0]); return 1; }

    int ref = prime_is_prime_seq(n);

    int out = 0;
    int rc = prime_is_prime(n, (prime_mode_t)mode_i, &out);
    if (rc != 0) {
        printf("OpenCL ERROR\n");
        return 2;
    }

    if (out == ref) {
        printf("OK  n=%" PRIu64 "  mode=%d  is_prime=%d\n", n, mode_i, out);
        return 0;
    } else {
        printf("NOT OK  n=%" PRIu64 "  mode=%d  out=%d  ref=%d\n", n, mode_i, out, ref);
        return 3;
    }
}
