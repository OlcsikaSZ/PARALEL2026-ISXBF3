#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

/**
 * Load the OpenCL kernel source code from a file.
 * 
 * path: Path of the source file
 * error_code: 0 on successful file loading
 * 
 * Returns by a dynamically allocated string
 */
char* load_kernel_source(const char* const path, int* error_code);

#endif
