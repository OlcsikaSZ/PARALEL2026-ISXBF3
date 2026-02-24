#include "fill_missing.h"
#include "kernel_loader.h"

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR (-1001)
#endif

static const char* cl_errstr(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_PLATFORM_NOT_FOUND_KHR: return "CL_PLATFORM_NOT_FOUND_KHR";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        default: return "CL_ERROR";
    }
}

static cl_device_id pick_device(cl_platform_id* out_platform, cl_int* out_err) {
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        if (out_err) *out_err = (err == CL_SUCCESS) ? CL_PLATFORM_NOT_FOUND_KHR : err;
        return NULL;
    }

    cl_platform_id* plats = (cl_platform_id*)calloc(nplat, sizeof(cl_platform_id));
    if (!plats) { if (out_err) *out_err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) { free(plats); if (out_err) *out_err = err; return NULL; }

    cl_device_id dev = NULL;
    cl_platform_id chosen = NULL;

    /* Prefer GPU, then CPU */
    for (cl_uint i = 0; i < nplat && !dev; ++i) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev) == CL_SUCCESS && ndev > 0) {
            cl_device_id* ds = (cl_device_id*)calloc(ndev, sizeof(cl_device_id));
            if (!ds) break;
            if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, ndev, ds, NULL) == CL_SUCCESS) {
                dev = ds[0];
                chosen = plats[i];
            }
            free(ds);
        }
    }
    for (cl_uint i = 0; i < nplat && !dev; ++i) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, 0, NULL, &ndev) == CL_SUCCESS && ndev > 0) {
            cl_device_id* ds = (cl_device_id*)calloc(ndev, sizeof(cl_device_id));
            if (!ds) break;
            if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, ndev, ds, NULL) == CL_SUCCESS) {
                dev = ds[0];
                chosen = plats[i];
            }
            free(ds);
        }
    }

    free(plats);
    if (!dev) { if (out_err) *out_err = CL_DEVICE_NOT_FOUND; return NULL; }
    if (out_platform) *out_platform = chosen;
    if (out_err) *out_err = CL_SUCCESS;
    return dev;
}

static int build_log(cl_program prog, cl_device_id dev) {
    size_t n = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
    char* log = (char*)malloc(n + 1);
    if (!log) return -1;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log, NULL);
    log[n] = '\0';
    fprintf(stderr, "OpenCL build log:\n%s\n", log);
    free(log);
    return 0;
}

int fill_missing(const int* in, int* out, size_t n) {
    if (!in || !out || n == 0) return -1;

    cl_int err = CL_SUCCESS;
    cl_platform_id plat = NULL;
    cl_device_id dev = pick_device(&plat, &err);
    if (!dev) {
        fprintf(stderr, "OpenCL: no device (%s / %d)\n", cl_errstr(err), (int)err);
        return -2;
    }

    cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)plat, 0 };
    cl_context ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
    if (!ctx || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext failed (%s / %d)\n", cl_errstr(err), (int)err);
        return -3;
    }

    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, 0, &err);
    if (!q || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueueWithProperties failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseContext(ctx);
        return -4;
    }

    char* src = NULL;
    size_t src_len = 0;
    if (load_text_file("kernels/fill_missing.cl", &src, &src_len) != 0) {
        fprintf(stderr, "Failed to load kernel file: kernels/fill_missing.cl\n");
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -5;
    }

    const char* sources[] = { src };
    const size_t lens[] = { src_len };
    cl_program prog = clCreateProgramWithSource(ctx, 1, sources, lens, &err);
    if (!prog || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource failed (%s / %d)\n", cl_errstr(err), (int)err);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -6;
    }

    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram failed (%s / %d)\n", cl_errstr(err), (int)err);
        build_log(prog, dev);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -7;
    }

    cl_kernel k = clCreateKernel(prog, "fill_missing", &err);
    if (!k || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -8;
    }

    size_t bytes = n * sizeof(int);
    cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void*)in, &err);
    if (!bin || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer(in) failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseKernel(k);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -9;
    }

    cl_mem bout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (!bout || err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer(out) failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseMemObject(bin);
        clReleaseKernel(k);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -10;
    }

    cl_int missing = (cl_int)MISSING_VALUE;
    cl_uint N = (cl_uint)n;

    err  = clSetKernelArg(k, 0, sizeof(cl_mem), &bin);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &bout);
    err |= clSetKernelArg(k, 2, sizeof(cl_uint), &N);
    err |= clSetKernelArg(k, 3, sizeof(cl_int), &missing);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseMemObject(bout);
        clReleaseMemObject(bin);
        clReleaseKernel(k);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -11;
    }

    size_t gws = n;
    err = clEnqueueNDRangeKernel(q, k, 1, NULL, &gws, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseMemObject(bout);
        clReleaseMemObject(bin);
        clReleaseKernel(k);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -12;
    }

    err = clEnqueueReadBuffer(q, bout, CL_TRUE, 0, bytes, out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer failed (%s / %d)\n", cl_errstr(err), (int)err);
        clReleaseMemObject(bout);
        clReleaseMemObject(bin);
        clReleaseKernel(k);
        clReleaseProgram(prog);
        free(src);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return -13;
    }

    clReleaseMemObject(bout);
    clReleaseMemObject(bin);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    free(src);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
