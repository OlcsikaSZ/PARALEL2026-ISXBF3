#include "include/kernel_loader.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_BATCH 131072u
#define DEFAULT_BITS 32u

typedef unsigned long long u64;

static u64 rng_state = 0x123456789abcdef0ULL;

static u64 splitmix64_next(void) {
    u64 z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30ULL)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27ULL)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31ULL);
}

static u64 addmod_u64(u64 a, u64 b, u64 mod) {
    a %= mod;
    b %= mod;
    if (a >= mod - b) return a - (mod - b);
    return a + b;
}

static u64 mulmod_u64(u64 a, u64 b, u64 mod) {
    u64 result = 0ULL;
    a %= mod;
    while (b > 0ULL) {
        if (b & 1ULL) result = addmod_u64(result, a, mod);
        a = addmod_u64(a, a, mod);
        b >>= 1ULL;
    }
    return result;
}

static u64 powmod_u64(u64 a, u64 e, u64 mod) {
    u64 result = 1ULL;
    a %= mod;
    while (e > 0ULL) {
        if (e & 1ULL) result = mulmod_u64(result, a, mod);
        a = mulmod_u64(a, a, mod);
        e >>= 1ULL;
    }
    return result;
}

static int is_prime_mr_u64(u64 n) {
    if (n < 2ULL) return 0;
    if ((n % 2ULL) == 0ULL) return n == 2ULL;
    if ((n % 3ULL) == 0ULL) return n == 3ULL;

    static const u64 bases[] = {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL};
    u64 d = n - 1ULL;
    unsigned int s = 0u;
    while ((d & 1ULL) == 0ULL) {
        d >>= 1ULL;
        ++s;
    }

    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); ++i) {
        u64 a = bases[i] % n;
        if (a == 0ULL) continue;
        u64 x = powmod_u64(a, d, n);
        if (x == 1ULL || x == n - 1ULL) continue;
        int witness = 1;
        for (unsigned int r = 1; r < s; ++r) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1ULL) {
                witness = 0;
                break;
            }
        }
        if (witness) return 0;
    }
    return 1;
}

static u64 make_candidate(unsigned int bits) {
    if (bits < 4u) bits = 4u;
    if (bits > 63u) bits = 63u;
    u64 x = splitmix64_next();
    if (bits < 64u) {
        u64 mask = (1ULL << bits) - 1ULL;
        x &= mask;
    }
    x |= (1ULL << (bits - 1u));
    x |= 1ULL;
    return x;
}

static cl_device_id pick_device(void) {
    cl_uint platform_count = 0;
    clGetPlatformIDs(0, NULL, &platform_count);
    if (platform_count == 0) return NULL;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platform_count);
    clGetPlatformIDs(platform_count, platforms, NULL);
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < platform_count && !dev; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count) == CL_SUCCESS && device_count > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices, NULL);
            dev = devices[0];
            free(devices);
        }
    }
    for (cl_uint i = 0; i < platform_count && !dev; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &device_count) == CL_SUCCESS && device_count > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, device_count, devices, NULL);
            dev = devices[0];
            free(devices);
        }
    }
    free(platforms);
    return dev;
}

int main(int argc, char** argv) {
    unsigned int bits = DEFAULT_BITS;
    size_t batch = DEFAULT_BATCH;
    if (argc > 1) bits = (unsigned int)strtoul(argv[1], NULL, 10);
    if (argc > 2) batch = (size_t)strtoull(argv[2], NULL, 10);
    if (bits < 4u) bits = 4u;
    if (bits > 63u) bits = 63u;

    printf("=== 05 - Prímszám generálás OpenCL ===\n");
    printf("Bitek száma: %u\n", bits);
    printf("Batch méret: %zu\n\n", batch);

    u64* candidates = (u64*)malloc(sizeof(u64) * batch);
    unsigned int* cpu_results = (unsigned int*)malloc(sizeof(unsigned int) * batch);
    unsigned int* gpu_results = (unsigned int*)malloc(sizeof(unsigned int) * batch);
    if (!candidates || !cpu_results || !gpu_results) {
        fprintf(stderr, "Memóriafoglalási hiba.\n");
        free(candidates); free(cpu_results); free(gpu_results);
        return 1;
    }

    for (size_t i = 0; i < batch; ++i) {
        candidates[i] = make_candidate(bits);
    }

    clock_t c0 = clock();
    u64 cpu_prime = 0ULL;
    size_t cpu_tests = 0;
    for (size_t i = 0; i < batch; ++i) {
        cpu_results[i] = (unsigned int)is_prime_mr_u64(candidates[i]);
        ++cpu_tests;
        if (cpu_results[i]) {
            cpu_prime = candidates[i];
            break;
        }
    }
    clock_t c1 = clock();
    double cpu_sec = (double)(c1 - c0) / CLOCKS_PER_SEC;

    cl_device_id device = pick_device();
    if (!device) {
        fprintf(stderr, "Nincs OpenCL eszköz.\n");
        free(candidates); free(cpu_results); free(gpu_results);
        return 1;
    }
    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("GPU eszköz: %s\n", device_name);

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program program = build_program_from_file(context, device, "kernels/prime_batch.cl", NULL);
    if (!program) {
        free(candidates); free(cpu_results); free(gpu_results);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "prime_test_batch", &err);

    cl_mem cand_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(u64) * batch, candidates, &err);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(unsigned int) * batch, NULL, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &cand_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);

    size_t global = batch;
    clock_t g0 = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Kernel launch hiba: %d\n", err);
        free(candidates); free(cpu_results); free(gpu_results);
        clReleaseMemObject(out_buf);
        clReleaseMemObject(cand_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(unsigned int) * batch, gpu_results, 0, NULL, NULL);
    clock_t g1 = clock();
    double gpu_sec = (double)(g1 - g0) / CLOCKS_PER_SEC;

    u64 gpu_prime = 0ULL;
    size_t gpu_tests = batch;
    size_t mismatches = 0;
    for (size_t i = 0; i < batch; ++i) {
        if (gpu_results[i] != cpu_results[i]) {
            ++mismatches;
        }
        if (gpu_prime == 0ULL && gpu_results[i]) {
            gpu_prime = candidates[i];
            gpu_tests = i + 1u;
        }
    }

    printf("\n[CPU] Első talált prím: %llu\n", cpu_prime);
    printf("[CPU] Vizsgált jelöltek: %zu\n", cpu_tests);
    printf("[CPU] Idő: %.3f s\n", cpu_sec);
    if (cpu_sec > 0.0) printf("[CPU] Sebesség: %.2f jelölt/s\n", (double)cpu_tests / cpu_sec);

    printf("\n[GPU] Első talált prím a batch-ben: %llu\n", gpu_prime);
    printf("[GPU] Batch méret: %zu\n", batch);
    printf("[GPU] Idő: %.3f s\n", gpu_sec);
    if (gpu_sec > 0.0) printf("[GPU] Sebesség: %.2f jelölt/s\n", (double)batch / gpu_sec);

    printf("\nEltérések száma: %zu\n", mismatches);
    if (mismatches == 0 && cpu_sec > 0.0 && gpu_sec > 0.0) {
        printf("Gyorsulás (CPU/GPU): %.2fx\n", cpu_sec / gpu_sec);
    }
    printf("Megjegyzés: ez a generátor 64 bit alatti (legfeljebb 63 bites) prímszámokra van belőve.\n");

    clReleaseMemObject(out_buf);
    clReleaseMemObject(cand_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(candidates);
    free(cpu_results);
    free(gpu_results);
    return 0;
}
