#include <stdio.h>
#include <stdlib.h>
#include "kernel_loader.h"

char* load_kernel_source(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Nem sikerult megnyitni a kernel fajlt: %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char* src = (char*)malloc((size_t)size + 1);
    if (!src) {
        fclose(f);
        return NULL;
    }

    size_t read_bytes = fread(src, 1, (size_t)size, f);
    src[read_bytes] = '\0';

    fclose(f);
    return src;
}