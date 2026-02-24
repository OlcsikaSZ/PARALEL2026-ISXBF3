#include "vector_add.h"
#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>


#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR (-1001)
#endif
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CL(err, msg) \
    do { \
        if ((err) != CL_SUCCESS) { \
            fprintf(stderr, "[OpenCL ERROR] %s (code=%d)\n", (msg), (int)(err)); \
            return 1; \
        } \
    } while (0)

static cl_device_id pick_device(cl_platform_id* out_platform, cl_int* out_err)
{
    cl_uint n_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        *out_err = (err == CL_SUCCESS) ? CL_PLATFORM_NOT_FOUND_KHR : err;
        return NULL;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platforms);
    err = clGetPlatformIDs(n_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        *out_err = err;
        return NULL;
    }

    cl_device_id chosen = NULL;
    cl_platform_id chosen_platform = NULL;

    // Prefer GPU, fall back to CPU.
    for (cl_uint p = 0; p < n_platforms && !chosen; ++p) {
        cl_uint n_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
        if (err == CL_SUCCESS && n_devices > 0) {
            err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &chosen, NULL);
            if (err == CL_SUCCESS) {
                chosen_platform = platforms[p];
                break;
            }
        }
    }
    for (cl_uint p = 0; p < n_platforms && !chosen; ++p) {
        cl_uint n_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 0, NULL, &n_devices);
        if (err == CL_SUCCESS && n_devices > 0) {
            err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_CPU, 1, &chosen, NULL);
            if (err == CL_SUCCESS) {
                chosen_platform = platforms[p];
                break;
            }
        }
    }

    free(platforms);

    if (!chosen) {
        *out_err = CL_DEVICE_NOT_FOUND;
        return NULL;
    }

    *out_platform = chosen_platform;
    *out_err = CL_SUCCESS;
    return chosen;
}

int vector_add(const float* a, const float* b, float* out, size_t n)
{
    if (!a || !b || !out || n == 0) return 2;

    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = pick_device(&platform, &err);
    CHECK_CL(err, "No suitable OpenCL device found");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext failed");

    // Build program
    int load_err = 0;
    char* kernel_code = load_kernel_source("kernels/vector_add.cl", &load_err);
    if (load_err != 0 || kernel_code == NULL) {
        fprintf(stderr, "[ERROR] Could not load kernel source file.\n");
        clReleaseContext(context);
        return 3;
    }

    const char* sources[] = { kernel_code };
    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    free(kernel_code);
    CHECK_CL(err, "clCreateProgramWithSource failed");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // print build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = 0;
            fprintf(stderr, "[OpenCL BUILD LOG]\n%s\n", log);
            free(log);
        }
        fprintf(stderr, "[OpenCL ERROR] clBuildProgram failed (code=%d)\n", (int)err);
        clReleaseProgram(program);
        clReleaseContext(context);
        return 4;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK_CL(err, "clCreateKernel failed");

    const size_t bytes = n * sizeof(float);
    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer(a) failed");
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer(b) failed");
    cl_mem buf_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CHECK_CL(err, "clCreateBuffer(out) failed");

    // Queue
#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties props[] = { 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_CL(err, "clCreateCommandQueue failed");

    // H2D
    err = clEnqueueWriteBuffer(queue, buf_a, CL_FALSE, 0, bytes, a, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueWriteBuffer(a) failed");
    err = clEnqueueWriteBuffer(queue, buf_b, CL_FALSE, 0, bytes, b, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueWriteBuffer(b) failed");

    // Args
    int n_int = (int)n;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_out);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n_int);
    CHECK_CL(err, "clSetKernelArg failed");

    // Launch
    size_t local = 256;
    size_t global = ((n + local - 1) / local) * local;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueNDRangeKernel failed");

    // D2H
    err = clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, bytes, out, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueReadBuffer(out) failed");

    // Cleanup
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_out);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
