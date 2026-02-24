#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Read text file into a newly allocated null-terminated buffer.
// Returns NULL on error. Caller must free().
char* load_text_file(const char* path, size_t* out_size);

#ifdef __cplusplus
}
#endif

#endif
