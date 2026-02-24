#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "minmax.h"
#include "kernel_loader.h"

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
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        default: return "CL_ERROR";
    }
}

static cl_device_id pick_device(cl_int* out_err) {
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS || nplat == 0) {
        if (out_err) *out_err = (err == CL_SUCCESS) ? CL_PLATFORM_NOT_FOUND_KHR : err;
        return NULL;
    }
    cl_platform_id* plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplat);
    if (!plats) { if (out_err) *out_err = CL_OUT_OF_HOST_MEMORY; return NULL; }
    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) { free(plats); if (out_err) *out_err = err; return NULL; }

    cl_device_id dev = NULL;

    // Prefer GPU, then CPU.
    for (cl_uint p = 0; p < nplat && !dev; ++p) {
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev);
        if (err == CL_SUCCESS && ndev > 0) {
            cl_device_id* devs = (cl_device_id*)malloc(sizeof(cl_device_id) * ndev);
            if (!devs) break;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, ndev, devs, NULL) == CL_SUCCESS) dev = devs[0];
            free(devs);
        }
    }
    for (cl_uint p = 0; p < nplat && !dev; ++p) {
        cl_uint ndev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, 0, NULL, &ndev);
        if (err == CL_SUCCESS && ndev > 0) {
            cl_device_id* devs = (cl_device_id*)malloc(sizeof(cl_device_id) * ndev);
            if (!devs) break;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, ndev, devs, NULL) == CL_SUCCESS) dev = devs[0];
            free(devs);
        }
    }

    free(plats);
    if (out_err) *out_err = dev ? CL_SUCCESS : CL_DEVICE_NOT_FOUND;
    return dev;
}

// Strategy: single kernel launch.
// - Choose a small-ish number of work-groups (<= compute units), each thread scans multiple elements.
// - Kernel does local reduction to 1 min/max per group.
// - Host finishes reduction over group results (small).
static int compute_group_layout(cl_device_id dev, size_t n,
                                size_t* out_local, size_t* out_groups, size_t* out_global) {
    size_t max_wg = 0;
    cl_uint cu = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL) != CL_SUCCESS) return -1;
    if (clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL) != CL_SUCCESS) cu = 1;

    // Pick a conservative local size (power of two) not exceeding max_wg.
    size_t local = 256;
    while (local > max_wg) local >>= 1;
    if (local < 32) local = (max_wg >= 32) ? 32 : max_wg;

    // Minimize "cores": keep number of work-groups small, but enough to hide latency.
    // Aim for up to compute units, but not more than needed.
    size_t groups = (size_t)cu;
    if (groups < 1) groups = 1;

    // Each work-item will scan a strided sequence; even 1 group will cover full N.
    // But for very large N, 1 group might be slow; we keep it at CU groups.
    size_t global = groups * local;

    *out_local = local;
    *out_groups = groups;
    *out_global = global;
    return 0;
}

int array_minmax(const int* in, size_t n, int* out_min, int* out_max) {
    if (!in || n == 0 || !out_min || !out_max) return -1;

    cl_int err = CL_SUCCESS;
    cl_device_id dev = pick_device(&err);
    if (!dev) {
        fprintf(stderr, "OpenCL device error: %s (%d)\n", cl_errstr(err), (int)err);
        return -2;
    }

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateContext: %s\n", cl_errstr(err)); return -3; }

    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue: %s\n", cl_errstr(err)); clReleaseContext(ctx); return -4; }

    char* src = NULL; size_t src_len = 0;
    if (load_text_file("kernels/minmax.cl", &src, &src_len) != 0) {
        fprintf(stderr, "Failed to load kernel source.\n");
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        return -5;
    }

    const char* srcs[] = { src };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateProgramWithSource: %s\n", cl_errstr(err)); free(src); clReleaseCommandQueue(q); clReleaseContext(ctx); return -6; }

    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        char* log = (char*)malloc(log_sz + 1);
        if (log) {
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
            log[log_sz] = '\0';
            fprintf(stderr, "Build log:\n%s\n", log);
            free(log);
        }
        fprintf(stderr, "clBuildProgram: %s (%d)\n", cl_errstr(err), (int)err);
        clReleaseProgram(prog); free(src);
        clReleaseCommandQueue(q); clReleaseContext(ctx);
        return -7;
    }
    free(src);

    cl_kernel krn = clCreateKernel(prog, "minmax_group", &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateKernel: %s\n", cl_errstr(err)); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -8; }

    size_t local = 0, groups = 0, global = 0;
    if (compute_group_layout(dev, n, &local, &groups, &global) != 0) {
        fprintf(stderr, "Failed to compute group layout.\n");
        clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
        return -9;
    }

    // buffers
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, (void*)in, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer(d_in): %s\n", cl_errstr(err)); clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -10; }

    cl_mem d_mins = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * groups, NULL, &err);
    cl_mem d_maxs = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * groups, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "clCreateBuffer(d_out): %s\n", cl_errstr(err)); clReleaseMemObject(d_in); clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -11; }

    // args: (in, n, out_mins, out_maxs, local scratch mins, local scratch maxs)
    err  = clSetKernelArg(krn, 0, sizeof(cl_mem), &d_in);
    err |= clSetKernelArg(krn, 1, sizeof(cl_uint), &(cl_uint){ (cl_uint)n });
    err |= clSetKernelArg(krn, 2, sizeof(cl_mem), &d_mins);
    err |= clSetKernelArg(krn, 3, sizeof(cl_mem), &d_maxs);
    err |= clSetKernelArg(krn, 4, sizeof(int) * local, NULL);
    err |= clSetKernelArg(krn, 5, sizeof(int) * local, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "clSetKernelArg: %s\n", cl_errstr(err)); clReleaseMemObject(d_maxs); clReleaseMemObject(d_mins); clReleaseMemObject(d_in); clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -12; }

    err = clEnqueueNDRangeKernel(q, krn, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueNDRangeKernel: %s (%d)\n", cl_errstr(err), (int)err); clReleaseMemObject(d_maxs); clReleaseMemObject(d_mins); clReleaseMemObject(d_in); clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -13; }

    // read back group results
    int* h_mins = (int*)malloc(sizeof(int) * groups);
    int* h_maxs = (int*)malloc(sizeof(int) * groups);
    if (!h_mins || !h_maxs) {
        fprintf(stderr, "Host alloc failed.\n");
        free(h_mins); free(h_maxs);
        clReleaseMemObject(d_maxs); clReleaseMemObject(d_mins); clReleaseMemObject(d_in);
        clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
        return -14;
    }

    err = clEnqueueReadBuffer(q, d_mins, CL_TRUE, 0, sizeof(int) * groups, h_mins, 0, NULL, NULL);
    err|= clEnqueueReadBuffer(q, d_maxs, CL_TRUE, 0, sizeof(int) * groups, h_maxs, 0, NULL, NULL);
    if (err != CL_SUCCESS) { fprintf(stderr, "clEnqueueReadBuffer: %s\n", cl_errstr(err)); free(h_mins); free(h_maxs); clReleaseMemObject(d_maxs); clReleaseMemObject(d_mins); clReleaseMemObject(d_in); clReleaseKernel(krn); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return -15; }

    // final reduction on CPU over small arrays
    int mn = h_mins[0];
    int mx = h_maxs[0];
    for (size_t g = 1; g < groups; ++g) {
        if (h_mins[g] < mn) mn = h_mins[g];
        if (h_maxs[g] > mx) mx = h_maxs[g];
    }
    *out_min = mn;
    *out_max = mx;

    free(h_mins); free(h_maxs);
    clReleaseMemObject(d_maxs); clReleaseMemObject(d_mins); clReleaseMemObject(d_in);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}
