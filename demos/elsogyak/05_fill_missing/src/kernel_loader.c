#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

int load_text_file(const char* path, char** out_src, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }

    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return -1; }

    buf[sz] = '\0';
    if (out_src) *out_src = buf; else free(buf);
    if (out_len) *out_len = (size_t)sz;
    return 0;
}
