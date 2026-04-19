#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>

char* load_kernel_source(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);

    char* buffer = (char*)malloc((size_t)size + 1u);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t got = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    if (got != (size_t)size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    return buffer;
}

cl_program build_program_from_file(cl_context context,
                                   cl_device_id device,
                                   const char* path,
                                   const char* options) {
    cl_int err = CL_SUCCESS;
    char* source = load_kernel_source(path);
    if (!source) {
        fprintf(stderr, "Nem sikerult beolvasni a kernel forrast: %s\n", path);
        return NULL;
    }

    const char* sources[] = { source };
    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource hiba: %d\n", err);
        free(source);
        return NULL;
    }

    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        (void)clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1u);
        if (log) {
            (void)clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "OpenCL build log (%s):\n%s\n", path, log);
            free(log);
        }
        clReleaseProgram(program);
        free(source);
        return NULL;
    }

    free(source);
    return program;
}
