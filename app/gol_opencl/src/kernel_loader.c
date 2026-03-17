#include "../include/kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

char* load_kernel_source(const char* path, int* error_code) {
    FILE* fp = NULL;
    long file_size_long = 0;
    size_t file_size = 0;
    size_t read_count = 0;
    char* source = NULL;

    if (error_code == NULL) {
        return NULL;
    }

    *error_code = 0;

    if (path == NULL) {
        *error_code = -5;  // invalid argument
        return NULL;
    }

    fp = fopen(path, "rb");
    if (!fp) {
        *error_code = -1;  // file open error
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    file_size_long = ftell(fp);
    if (file_size_long < 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    if ((unsigned long)file_size_long > (unsigned long)(SIZE_MAX - 1)) {
        *error_code = -4;  // invalid file size / read error
        fclose(fp);
        return NULL;
    }

    file_size = (size_t)file_size_long;

    source = (char*)malloc(file_size + 1);
    if (!source) {
        *error_code = -3;  // allocation error
        fclose(fp);
        return NULL;
    }

    read_count = fread(source, 1, file_size, fp);
    if (read_count != file_size) {
        *error_code = -4;  // read error
        free(source);
        fclose(fp);
        return NULL;
    }

    source[file_size] = '\0';

    if (fclose(fp) != 0) {
        *error_code = -2;  // close error
        free(source);
        return NULL;
    }

    return source;
}
