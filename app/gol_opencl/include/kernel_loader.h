#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

/*
 * Loads the kernel source file into memory.
 * Returns:
 *   - on success: a null-terminated char* buffer (must be freed with free())
 *   - on failure: NULL
 *
 * error_code meanings:
 *   0   = success
 *  -1   = file open error
 *  -2   = seek / tell / close error
 *  -3   = memory allocation error
 *  -4   = read error or invalid file size
 *  -5   = invalid argument
 */
char* load_kernel_source(const char* path, int* error_code);

#endif
