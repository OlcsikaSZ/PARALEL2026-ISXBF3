#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* const path, int* error_code)
{
    FILE* source_file = fopen(path, "rb");
    if (source_file == NULL) {
        *error_code = -1;
        return NULL;
    }

    fseek(source_file, 0, SEEK_END);
    long file_size = ftell(source_file);
    rewind(source_file);

    if (file_size <= 0) {
        fclose(source_file);
        *error_code = -2;
        return NULL;
    }

    char* source_code = (char*)malloc((size_t)file_size + 1);
    if (!source_code) {
        fclose(source_file);
        *error_code = -3;
        return NULL;
    }

    size_t read_bytes = fread(source_code, 1, (size_t)file_size, source_file);
    fclose(source_file);

    source_code[read_bytes] = 0;
    *error_code = 0;
    return source_code;
}
