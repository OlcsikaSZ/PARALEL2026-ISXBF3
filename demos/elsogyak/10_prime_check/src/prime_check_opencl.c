#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "prime_check.h"
#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef CL_PLATFORM_NOT_FOUND_KHR
#define CL_PLATFORM_NOT_FOUND_KHR (-1001)
#endif

#define KERNEL_PATH "kernels/prime_check.cl"

static void print_build_log(cl_program prog, cl_device_id dev) {
    size_t log_sz = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
    char* log = (char*)malloc(log_sz + 1);
    if (!log) return;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log, NULL);
    log[log_sz] = '\0';
    fprintf(stderr, "OpenCL build log:\n%s\n", log);
    free(log);
}

static cl_int pick_device(cl_platform_id* out_plat, cl_device_id* out_dev) {
    cl_uint nplat = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplat);
    if (err != CL_SUCCESS || nplat == 0) return (err == CL_SUCCESS) ? CL_PLATFORM_NOT_FOUND_KHR : err;

    cl_platform_id* plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplat);
    if (!plats) return CL_OUT_OF_HOST_MEMORY;
    err = clGetPlatformIDs(nplat, plats, NULL);
    if (err != CL_SUCCESS) { free(plats); return err; }

    cl_device_id best = NULL;
    cl_platform_id best_plat = NULL;

    // Prefer GPU then CPU
    for (cl_uint i = 0; i < nplat && !best; ++i) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 0, NULL, &ndev) == CL_SUCCESS && ndev) {
            cl_device_id* devs = (cl_device_id*)malloc(sizeof(cl_device_id) * ndev);
            if (!devs) { free(plats); return CL_OUT_OF_HOST_MEMORY; }
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, ndev, devs, NULL);
            best = devs[0];
            best_plat = plats[i];
            free(devs);
        }
    }
    for (cl_uint i = 0; i < nplat && !best; ++i) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, 0, NULL, &ndev) == CL_SUCCESS && ndev) {
            cl_device_id* devs = (cl_device_id*)malloc(sizeof(cl_device_id) * ndev);
            if (!devs) { free(plats); return CL_OUT_OF_HOST_MEMORY; }
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_CPU, ndev, devs, NULL);
            best = devs[0];
            best_plat = plats[i];
            free(devs);
        }
    }

    free(plats);
    if (!best) return CL_DEVICE_NOT_FOUND;

    *out_plat = best_plat;
    *out_dev = best;
    return CL_SUCCESS;
}

static uint64_t isqrt_u64(uint64_t n) {
    long double r = sqrt((long double)n);
    uint64_t x = (uint64_t)r;
    while ((x+1) > 0 && (x+1) * (x+1) <= n) x++;
    while (x * x > n) x--;
    return x;
}

// Host sieve primes up to limit (inclusive). Returns malloc'd primes array and count.
static int sieve_primes_host(uint32_t limit, uint32_t** primes_out, uint32_t* count_out) {
    if (limit < 2) {
        *primes_out = NULL; *count_out = 0; return 0;
    }
    uint8_t* is_comp = (uint8_t*)calloc((size_t)limit + 1, 1);
    if (!is_comp) return -1;
    uint32_t cap = (uint32_t)(limit / 10 + 16);
    uint32_t* primes = (uint32_t*)malloc(sizeof(uint32_t) * cap);
    if (!primes) { free(is_comp); return -1; }

    uint32_t cnt = 0;
    for (uint32_t p = 2; p <= limit; ++p) {
        if (!is_comp[p]) {
            if (cnt == cap) {
                cap = cap * 2 + 16;
                uint32_t* tmp = (uint32_t*)realloc(primes, sizeof(uint32_t) * cap);
                if (!tmp) { free(primes); free(is_comp); return -1; }
                primes = tmp;
            }
            primes[cnt++] = p;
            if ((uint64_t)p * (uint64_t)p <= (uint64_t)limit) {
                for (uint32_t k = p * p; k <= limit; k += p) is_comp[k] = 1;
            }
        }
    }
    free(is_comp);
    *primes_out = primes;
    *count_out = cnt;
    return 0;
}

// GPU sieve (marking) to generate primes up to limit. Compaction is done on host.
// Returns primes array (malloc) and count.
static int sieve_primes_gpu(cl_context ctx, cl_command_queue q, cl_program prog,
                            cl_device_id dev, uint32_t limit,
                            uint32_t** primes_out, uint32_t* count_out) {
    if (limit < 2) { *primes_out = NULL; *count_out = 0; return 0; }

    cl_int err;
    size_t nflags = (size_t)limit + 1;
    cl_mem flags = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nflags, NULL, &err);
    if (err != CL_SUCCESS) return -1;

    // init flags to 0 (meaning "prime until proven composite")
    uint8_t* zeros = (uint8_t*)calloc(nflags, 1);
    if (!zeros) { clReleaseMemObject(flags); return -1; }
    err = clEnqueueWriteBuffer(q, flags, CL_TRUE, 0, nflags, zeros, 0, NULL, NULL);
    free(zeros);
    if (err != CL_SUCCESS) { clReleaseMemObject(flags); return -1; }

    cl_kernel k_mark = clCreateKernel(prog, "sieve_mark", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(flags); return -1; }

    err |= clSetKernelArg(k_mark, 0, sizeof(cl_mem), &flags);
    err |= clSetKernelArg(k_mark, 1, sizeof(cl_uint), &limit);
    if (err != CL_SUCCESS) { clReleaseKernel(k_mark); clReleaseMemObject(flags); return -1; }

    // We mark composites for each p in parallel: each work-item corresponds to a candidate p.
    // This is not the most efficient sieve, but it's simple and shows OpenCL usage.
    size_t g = (size_t)limit + 1;
    size_t l = 256;
    if (l > g) l = g ? g : 1;

    err = clEnqueueNDRangeKernel(q, k_mark, 1, NULL, &g, &l, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k_mark); clReleaseMemObject(flags); return -1; }
    clFinish(q);

    uint8_t* host_flags = (uint8_t*)malloc(nflags);
    if (!host_flags) { clReleaseKernel(k_mark); clReleaseMemObject(flags); return -1; }
    err = clEnqueueReadBuffer(q, flags, CL_TRUE, 0, nflags, host_flags, 0, NULL, NULL);
    clReleaseKernel(k_mark);
    clReleaseMemObject(flags);
    if (err != CL_SUCCESS) { free(host_flags); return -1; }

    uint32_t cap = (uint32_t)(limit / 10 + 16);
    uint32_t* primes = (uint32_t*)malloc(sizeof(uint32_t) * cap);
    if (!primes) { free(host_flags); return -1; }

    uint32_t cnt = 0;
    for (uint32_t i = 2; i <= limit; ++i) {
        if (host_flags[i] == 0) {
            if (cnt == cap) {
                cap = cap * 2 + 16;
                uint32_t* tmp = (uint32_t*)realloc(primes, sizeof(uint32_t) * cap);
                if (!tmp) { free(primes); free(host_flags); return -1; }
                primes = tmp;
            }
            primes[cnt++] = i;
        }
    }
    free(host_flags);
    *primes_out = primes;
    *count_out = cnt;
    return 0;
}

// Common: build program and create context/queue
typedef struct ocl_env_t {
    cl_platform_id plat;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
} ocl_env_t;

static int ocl_init(ocl_env_t* e) {
    memset(e, 0, sizeof(*e));
    cl_int err = pick_device(&e->plat, &e->dev);
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL: no device/platform (err=%d)\n", err); return -1; }

    e->ctx = clCreateContext(NULL, 1, &e->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL: clCreateContext failed (%d)\n", err); return -1; }

#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties props[] = {0};
    e->q = clCreateCommandQueueWithProperties(e->ctx, e->dev, props, &err);
#else
    e->q = clCreateCommandQueue(e->ctx, e->dev, 0, &err);
#endif
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL: create queue failed (%d)\n", err); return -1; }

    size_t src_sz = 0;
    char* src = load_text_file(KERNEL_PATH, &src_sz);
    if (!src) { fprintf(stderr, "Cannot load kernel: %s\n", KERNEL_PATH); return -1; }

    const char* sources[] = { src };
    e->prog = clCreateProgramWithSource(e->ctx, 1, sources, &src_sz, &err);
    free(src);
    if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL: create program failed (%d)\n", err); return -1; }

    err = clBuildProgram(e->prog, 1, &e->dev, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL: build failed (%d)\n", err);
        print_build_log(e->prog, e->dev);
        return -1;
    }
    return 0;
}

static void ocl_free(ocl_env_t* e) {
    if (e->prog) clReleaseProgram(e->prog);
    if (e->q) clReleaseCommandQueue(e->q);
    if (e->ctx) clReleaseContext(e->ctx);
}

// Mode 0: each work-item checks one divisor. We only check odd divisors.
static int prime_single_divisor(ocl_env_t* e, uint64_t n, int* is_prime_out) {
    if (n < 2ULL) { *is_prime_out = 0; return 0; }
    if (n == 2ULL) { *is_prime_out = 1; return 0; }
    if ((n & 1ULL) == 0ULL) { *is_prime_out = 0; return 0; }

    uint64_t r = isqrt_u64(n);
    uint64_t first = 3ULL;
    if (r < first) { *is_prime_out = 1; return 0; }

    // count of odd divisors: 3,5,7,...,r
    uint64_t count = ((r - first) / 2ULL) + 1ULL;

    cl_int err;
    cl_mem result = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) return -1;
    cl_int one = 1;
    err = clEnqueueWriteBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &one, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); return -1; }

    cl_kernel k = clCreateKernel(e->prog, "check_single_divisor", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); return -1; }

    err |= clSetKernelArg(k, 0, sizeof(cl_ulong), &n);
    cl_ulong first_ul = (cl_ulong)first;
    err |= clSetKernelArg(k, 1, sizeof(cl_ulong), &first_ul);
    cl_ulong step = 2;
    err |= clSetKernelArg(k, 2, sizeof(cl_ulong), &step);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &result);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); return -1; }

    size_t g = (size_t)count;
    size_t l = 256;
    if (l > g) l = g ? g : 1;

    err = clEnqueueNDRangeKernel(e->q, k, 1, NULL, &g, &l, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); return -1; }
    clFinish(e->q);

    cl_int out = 0;
    err = clEnqueueReadBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &out, 0, NULL, NULL);
    clReleaseKernel(k);
    clReleaseMemObject(result);
    if (err != CL_SUCCESS) return -1;

    *is_prime_out = (out != 0);
    return 0;
}

// Mode 1: each work-item checks a range [start, end] of divisors (odd only)
static int prime_range_divisor(ocl_env_t* e, uint64_t n, int* is_prime_out) {
    if (n < 2ULL) { *is_prime_out = 0; return 0; }
    if (n == 2ULL) { *is_prime_out = 1; return 0; }
    if ((n & 1ULL) == 0ULL) { *is_prime_out = 0; return 0; }

    uint64_t r = isqrt_u64(n);
    if (r < 3ULL) { *is_prime_out = 1; return 0; }

    // Choose small number of work-items: ~ 4 * compute units, clamped
    cl_uint cu = 1;
    clGetDeviceInfo(e->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    size_t work_items = (size_t)cu * 4;
    if (work_items < 1) work_items = 1;
    if (work_items > 65535) work_items = 65535;

    // total odd divisors count
    uint64_t total = ((r - 3ULL) / 2ULL) + 1ULL;
    uint64_t chunk = (total + (uint64_t)work_items - 1ULL) / (uint64_t)work_items;
    if (chunk < 1ULL) chunk = 1ULL;

    cl_int err;
    cl_mem result = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) return -1;
    cl_int one = 1;
    err = clEnqueueWriteBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &one, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); return -1; }

    cl_kernel k = clCreateKernel(e->prog, "check_range_divisor", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); return -1; }

    err |= clSetKernelArg(k, 0, sizeof(cl_ulong), &n);
    cl_ulong r_ul = (cl_ulong)r;
    err |= clSetKernelArg(k, 1, sizeof(cl_ulong), &r_ul);
    cl_ulong chunk_ul = (cl_ulong)chunk;
    err |= clSetKernelArg(k, 2, sizeof(cl_ulong), &chunk_ul);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &result);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); return -1; }

    size_t g = work_items;
    size_t l = 256;
    if (l > g) l = g;

    err = clEnqueueNDRangeKernel(e->q, k, 1, NULL, &g, &l, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); return -1; }
    clFinish(e->q);

    cl_int out = 0;
    err = clEnqueueReadBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &out, 0, NULL, NULL);
    clReleaseKernel(k);
    clReleaseMemObject(result);
    if (err != CL_SUCCESS) return -1;

    *is_prime_out = (out != 0);
    return 0;
}

// Mode 2/3: check divisibility only by primes up to sqrt(n)
static int prime_preprimes(ocl_env_t* e, uint64_t n, int gpu_sieve, int* is_prime_out) {
    if (n < 2ULL) { *is_prime_out = 0; return 0; }
    if (n == 2ULL) { *is_prime_out = 1; return 0; }
    if ((n & 1ULL) == 0ULL) { *is_prime_out = 0; return 0; }

    uint64_t r64 = isqrt_u64(n);
    if (r64 < 3ULL) { *is_prime_out = 1; return 0; }
    if (r64 > 0xFFFFFFFFULL) {
        fprintf(stderr, "Limit too large for prime table (sqrt(n) > 2^32-1)\n");
        return -1;
    }
    uint32_t limit = (uint32_t)r64;

    uint32_t* primes = NULL;
    uint32_t pcount = 0;
    int rc = gpu_sieve ? sieve_primes_gpu(e->ctx, e->q, e->prog, e->dev, limit, &primes, &pcount)
                       : sieve_primes_host(limit, &primes, &pcount);
    if (rc != 0) return -1;

    // remove 2 because n is odd already; but harmless if included
    if (pcount == 0) { *is_prime_out = 1; free(primes); return 0; }

    cl_int err;
    cl_mem primes_buf = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(cl_uint) * (size_t)pcount, primes, &err);
    free(primes);
    if (err != CL_SUCCESS) return -1;

    cl_mem result = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(primes_buf); return -1; }
    cl_int one = 1;
    err = clEnqueueWriteBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &one, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); clReleaseMemObject(primes_buf); return -1; }

    cl_kernel k = clCreateKernel(e->prog, "check_primes_table", &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(result); clReleaseMemObject(primes_buf); return -1; }

    err |= clSetKernelArg(k, 0, sizeof(cl_ulong), &n);
    err |= clSetKernelArg(k, 1, sizeof(cl_mem), &primes_buf);
    err |= clSetKernelArg(k, 2, sizeof(cl_uint), &pcount);
    err |= clSetKernelArg(k, 3, sizeof(cl_mem), &result);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); clReleaseMemObject(primes_buf); return -1; }

    // Minimal work-items: again ~ 4 * CU, each checks strided subset of primes
    cl_uint cu = 1;
    clGetDeviceInfo(e->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    size_t g = (size_t)cu * 4;
    if (g < 1) g = 1;
    if (g > (size_t)pcount) g = (size_t)pcount;
    size_t l = 256;
    if (l > g) l = g;

    err = clEnqueueNDRangeKernel(e->q, k, 1, NULL, &g, &l, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseKernel(k); clReleaseMemObject(result); clReleaseMemObject(primes_buf); return -1; }
    clFinish(e->q);

    cl_int out = 0;
    err = clEnqueueReadBuffer(e->q, result, CL_TRUE, 0, sizeof(cl_int), &out, 0, NULL, NULL);
    clReleaseKernel(k);
    clReleaseMemObject(result);
    clReleaseMemObject(primes_buf);
    if (err != CL_SUCCESS) return -1;

    *is_prime_out = (out != 0);
    return 0;
}

int prime_is_prime(uint64_t n, prime_mode_t mode, int* is_prime_out) {
    if (!is_prime_out) return -1;
    ocl_env_t e;
    if (ocl_init(&e) != 0) return -1;

    int rc = -1;
    switch (mode) {
        case PRIME_MODE_SINGLE_DIVISOR:
            rc = prime_single_divisor(&e, n, is_prime_out);
            break;
        case PRIME_MODE_RANGE_DIVISOR:
            rc = prime_range_divisor(&e, n, is_prime_out);
            break;
        case PRIME_MODE_PREPRIMES_HOST:
            rc = prime_preprimes(&e, n, 0, is_prime_out);
            break;
        case PRIME_MODE_PREPRIMES_GPU:
            rc = prime_preprimes(&e, n, 1, is_prime_out);
            break;
        default:
            fprintf(stderr, "Unknown mode %d\n", (int)mode);
            rc = -1;
    }

    ocl_free(&e);
    return rc;
}
