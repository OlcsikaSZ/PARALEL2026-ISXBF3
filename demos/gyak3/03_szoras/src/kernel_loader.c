#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* const path, int* error_code)
{
    FILE* source_file = fopen(path, "rb");
    char* source_code;
    long file_size;

    if (source_file == NULL) {
        *error_code = -1;
        return NULL;
    }

    fseek(source_file, 0, SEEK_END);
    file_size = ftell(source_file);
    rewind(source_file);

    source_code = (char*)malloc((size_t)file_size + 1);
    if (source_code == NULL) {
        fclose(source_file);
        *error_code = -2;
        return NULL;
    }

    fread(source_code, sizeof(char), (size_t)file_size, source_file);
    source_code[file_size] = 0;
    fclose(source_file);

    *error_code = 0;
    return source_code;
}