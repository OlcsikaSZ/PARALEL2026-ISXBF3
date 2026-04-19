#include "include/kernel_loader.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_COUNT 200000u
#define DEFAULT_START 1000000000001ULL

typedef unsigned long long u64;

static u64 addmod_u64(u64 a, u64 b, u64 mod) {
    a %= mod;
    b %= mod;
    if (a >= mod - b) {
        return a - (mod - b);
    }
    return a + b;
}

static u64 mulmod_u64(u64 a, u64 b, u64 mod) {
    u64 result = 0;
    a %= mod;
    while (b > 0) {
        if (b & 1ULL) {
            result = addmod_u64(result, a, mod);
        }
        a = addmod_u64(a, a, mod);
        b >>= 1ULL;
    }
    return result;
}

static u64 powmod_u64(u64 a, u64 e, u64 mod) {
    u64 result = 1ULL;
    a %= mod;
    while (e > 0) {
        if (e & 1ULL) {
            result = mulmod_u64(result, a, mod);
        }
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
    size_t count = DEFAULT_COUNT;
    u64 start = DEFAULT_START;
    if (argc > 1) count = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) start = (u64)strtoull(argv[2], NULL, 10);
    if ((start & 1ULL) == 0ULL) ++start;

    printf("=== 04 - Miller-Rabin prímteszt OpenCL ===\n");
    printf("Darabszam: %zu\n", count);
    printf("Kezdo odd szam: %llu\n\n", start);

    u64* numbers = (u64*)malloc(sizeof(u64) * count);
    unsigned int* cpu_results = (unsigned int*)malloc(sizeof(unsigned int) * count);
    unsigned int* gpu_results = (unsigned int*)malloc(sizeof(unsigned int) * count);
    if (!numbers || !cpu_results || !gpu_results) {
        fprintf(stderr, "Memoriafoglalasi hiba.\n");
        free(numbers); free(cpu_results); free(gpu_results);
        return 1;
    }

    for (size_t i = 0; i < count; ++i) {
        numbers[i] = start + (u64)(2ULL * (u64)i);
    }

    clock_t c0 = clock();
    size_t cpu_prime_count = 0;
    for (size_t i = 0; i < count; ++i) {
        cpu_results[i] = (unsigned int)is_prime_mr_u64(numbers[i]);
        cpu_prime_count += cpu_results[i] ? 1u : 0u;
    }
    clock_t c1 = clock();
    double cpu_sec = (double)(c1 - c0) / CLOCKS_PER_SEC;

    cl_device_id device = pick_device();
    if (!device) {
        fprintf(stderr, "Nincs OpenCL eszkoz.\n");
        free(numbers); free(cpu_results); free(gpu_results);
        return 1;
    }

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("GPU eszkoz: %s\n", device_name);

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program program = build_program_from_file(context, device, "kernels/miller_rabin.cl", NULL);
    if (!program) {
        free(numbers); free(cpu_results); free(gpu_results);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "miller_rabin_batch", &err);

    cl_mem nums_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(u64) * count, numbers, &err);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(unsigned int) * count, NULL, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &nums_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
    size_t global = count;
    clock_t g0 = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Kernel launch hiba: %d\n", err);
        free(numbers); free(cpu_results); free(gpu_results);
        clReleaseMemObject(out_buf);
        clReleaseMemObject(nums_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    clFinish(queue);
    clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(unsigned int) * count, gpu_results, 0, NULL, NULL);
    clock_t g1 = clock();
    double gpu_sec = (double)(g1 - g0) / CLOCKS_PER_SEC;

    size_t gpu_prime_count = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        gpu_prime_count += gpu_results[i] ? 1u : 0u;
        if (gpu_results[i] != cpu_results[i]) {
            ++mismatches;
            if (mismatches <= 5) {
                printf("Eltérés: n=%llu CPU=%u GPU=%u\n",
                       numbers[i], cpu_results[i], gpu_results[i]);
            }
        }
    }

    printf("\n[CPU] Prímek száma: %zu\n", cpu_prime_count);
    printf("[CPU] Idő: %.3f s\n", cpu_sec);
    if (cpu_sec > 0.0) printf("[CPU] Sebesség: %.2f szám/s\n", (double)count / cpu_sec);

    printf("\n[GPU] Prímek száma: %zu\n", gpu_prime_count);
    printf("[GPU] Idő: %.3f s\n", gpu_sec);
    if (gpu_sec > 0.0) printf("[GPU] Sebesség: %.2f szám/s\n", (double)count / gpu_sec);

    printf("\nEltérések száma: %zu\n", mismatches);
    if (mismatches == 0 && cpu_sec > 0.0 && gpu_sec > 0.0) {
        printf("Gyorsulás (CPU/GPU): %.2fx\n", cpu_sec / gpu_sec);
    }
    printf("Megjegyzés: ez a változat 64 bites számokra készült. 128+ bitnél már más liga van, ott big integer kell.\n");

    clReleaseMemObject(out_buf);
    clReleaseMemObject(nums_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(numbers);
    free(cpu_results);
    free(gpu_results);
    return 0;
}
