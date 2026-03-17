#include <stdio.h>
#include <stdlib.h>
#include "kernel_loader.h"

char* load_kernel_source(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Nem sikerult megnyitni a kernel fajlt");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char* src = (char*)malloc(size + 1);
    if (!src) {
        fclose(f);
        return NULL;
    }

    fread(src, 1, size, f);
    src[size] = '\0';

    fclose(f);
    return src;
}