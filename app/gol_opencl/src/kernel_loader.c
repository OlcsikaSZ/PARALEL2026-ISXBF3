#include "kernel_loader.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* const path, int* error_code)
{
    FILE* source_file;
    char* source_code;
    int file_size;

    source_file = fopen(path, "rb");
    if (source_file == NULL) {
        *error_code = -1;
        return NULL;
    }

    fseek(source_file, 0, SEEK_END);
    file_size = ftell(source_file);
    rewind(source_file);
    source_code = (char*)malloc(file_size + 1);
    fread(source_code, sizeof(char), file_size, source_file);
    source_code[file_size] = 0;

    *error_code = 0;
    return source_code;
}
