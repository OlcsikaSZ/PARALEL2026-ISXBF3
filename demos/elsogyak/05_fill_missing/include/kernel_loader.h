#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Loads a text file into a null-terminated buffer.
   Returns 0 on success. Caller must free(*out_src) with free(). */
int load_text_file(const char* path, char** out_src, size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif
