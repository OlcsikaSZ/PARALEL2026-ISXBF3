#include <CL/cl.h>
#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

cl_context create_context(cl_device_id* device)
{
    cl_platform_id platform;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    if (err != CL_SUCCESS)
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, device, NULL);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    return context;
}

cl_program load_program(cl_context context, cl_device_id device, const char* filename)
{
    FILE* f = fopen(filename, "r");
    if (!f) { perror("File open"); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);

    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    free(source);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    return program;
}

cl_kernel create_kernel(cl_program program, const char* kernel_name)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    return kernel;
}