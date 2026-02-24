#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reads a text file into a NUL-terminated buffer. Caller must free(*out_src).
int load_text_file(const char* path, char** out_src, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif
