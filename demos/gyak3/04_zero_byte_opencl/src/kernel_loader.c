#include <stdio.h>
#include <stdlib.h>
#include "kernel_loader.h"

char *load_text_file(const char *filename) {
    FILE *f;
    long size;
    char *buffer;
    size_t read_bytes;

    f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(f);
        return NULL;
    }

    size = ftell(f);
    if (size < 0) {
        perror("ftell");
        fclose(f);
        return NULL;
    }

    rewind(f);

    buffer = (char *)malloc((size_t)size + 1);
    if (!buffer) {
        fprintf(stderr, "malloc failed\n");
        fclose(f);
        return NULL;
    }

    read_bytes = fread(buffer, 1, (size_t)size, f);
    fclose(f);

    if (read_bytes != (size_t)size) {
        fprintf(stderr, "fread failed\n");
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    return buffer;
}