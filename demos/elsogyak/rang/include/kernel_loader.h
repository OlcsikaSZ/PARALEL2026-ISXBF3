#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <CL/cl.h>

cl_context create_context(cl_device_id* device);
cl_program load_program(cl_context context, cl_device_id device, const char* filename);
cl_kernel create_kernel(cl_program program, const char* kernel_name);

#endif