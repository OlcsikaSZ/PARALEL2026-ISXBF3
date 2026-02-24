#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "minmax.h"

static int parse_ul(const char* s, unsigned long* out) {
    char* end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!s || *s == '\0' || (end && *end != '\0')) return -1;
    *out = v;
    return 0;
}

int main(int argc, char** argv) {
    unsigned long N_ul = 1048576ul;
    int range = 1000000;

    if (argc >= 2) parse_ul(argv[1], &N_ul);
    if (argc >= 3) range = atoi(argv[2]);
    size_t N = (size_t)N_ul;

    int* a = (int*)malloc(sizeof(int) * N);
    if (!a) { fprintf(stderr, "alloc failed\n"); return 1; }

    generate_int_array(a, N, range, 123u);

    int mn_cl = 0, mx_cl = 0;
    int mn_ref = 0, mx_ref = 0;

    if (array_minmax(a, N, &mn_cl, &mx_cl) != 0) {
        fprintf(stderr, "array_minmax(OpenCL) failed\n");
        free(a);
        return 2;
    }

    array_minmax_seq(a, N, &mn_ref, &mx_ref);

    if (mn_cl != mn_ref || mx_cl != mx_ref) {
        printf("NOT OK  N=%lu  cl(min=%d,max=%d)  ref(min=%d,max=%d)\n",
               (unsigned long)N, mn_cl, mx_cl, mn_ref, mx_ref);
        free(a);
        return 3;
    }

    printf("OK  N=%lu  min=%d  max=%d\n", (unsigned long)N, mn_cl, mx_cl);
    free(a);
    return 0;
}
