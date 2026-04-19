#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

char* load_kernel_source(const char* path);
cl_program build_program_from_file(cl_context context,
                                   cl_device_id device,
                                   const char* path,
                                   const char* options);

#endif
