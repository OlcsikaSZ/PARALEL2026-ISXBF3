#include "../include/kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>

// Load an OpenCL kernel file into a null-terminated heap buffer.
char* load_kernel_source(const char* path, int* error_code) {
    FILE* fp = NULL;
    long file_size_long = 0;
    size_t file_size = 0;
    size_t read_count = 0;
    char* source = NULL;

    // The error output pointer is mandatory for reporting failures.
    if (error_code == NULL) {
        return NULL;
    }

    *error_code = 0;

    // Reject a missing file path before touching the filesystem.
    if (path == NULL) {
        *error_code = -5;  // invalid argument
        return NULL;
    }

    // Open the kernel file in binary mode to preserve the exact source bytes.
    fp = fopen(path, "rb");
    if (!fp) {
        *error_code = -1;  // file open error
        return NULL;
    }

    // Seek to the end so the file size can be measured once.
    if (fseek(fp, 0, SEEK_END) != 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    // Read the source length so enough memory can be allocated.
    file_size_long = ftell(fp);
    if (file_size_long < 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    // Rewind to the beginning before reading the file contents.
    if (fseek(fp, 0, SEEK_SET) != 0) {
        *error_code = -2;  // seek/tell/close error
        fclose(fp);
        return NULL;
    }

    // Guard against overflow when reserving space for the terminator byte.
    if ((unsigned long)file_size_long > (unsigned long)(SIZE_MAX - 1)) {
        *error_code = -4;  // invalid file size / read error
        fclose(fp);
        return NULL;
    }

    file_size = (size_t)file_size_long;

    // Allocate a writable buffer for the full source plus
    source = (char*)malloc(file_size + 1);
    if (!source) {
        *error_code = -3;  // allocation error
        fclose(fp);
        return NULL;
    }

    // Read the entire file in one pass.
    read_count = fread(source, 1, file_size, fp);
    if (read_count != file_size) {
        *error_code = -4;  // read error
        free(source);
        fclose(fp);
        return NULL;
    }

    // Null-terminate the buffer so OpenCL can consume it as C text.
    source[file_size] = '\0';

    // Fail cleanly if the file could not be closed.
    if (fclose(fp) != 0) {
        *error_code = -2;  // close error
        free(source);
        return NULL;
    }

    return source;
}
