#define __USE_MINGW_ANSI_STDIO 1
#define _CRT_SECURE_NO_WARNINGS
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernel_loader.h"

#ifdef _WIN32
#include <windows.h>
double now_ms() {
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER counter;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart * 1000.0 / freq.QuadPart);
}
#else
#include <sys/time.h>
double now_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

#define CHECK_CL(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "%s hiba: %d\n", (msg), (err)); \
        exit(1); \
    }

void cpu_counting_sort(const unsigned int* input, unsigned int* output, size_t n, unsigned int max_val) {
    unsigned int* hist = (unsigned int*)calloc(max_val + 1, sizeof(unsigned int));
    if (!hist) {
        fprintf(stderr, "Nem sikerult memoriat foglalni a CPU histogramhoz.\n");
        exit(1);
    }

    for (size_t i = 0; i < n; i++) {
        hist[input[i]]++;
    }

    size_t idx = 0;
    for (unsigned int v = 0; v <= max_val; v++) {
        for (unsigned int c = 0; c < hist[v]; c++) {
            output[idx++] = v;
        }
    }

    free(hist);
}

void cpu_rebuild_from_histogram(const unsigned int* hist, unsigned int* output, size_t n, unsigned int max_val) {
    size_t idx = 0;
    for (unsigned int v = 0; v <= max_val; v++) {
        for (unsigned int c = 0; c < hist[v]; c++) {
            if (idx >= n) {
                fprintf(stderr, "Hiba: tulindexeles a CPU rebuild kozben.\n");
                exit(1);
            }
            output[idx++] = v;
        }
    }

    if (idx != n) {
        fprintf(stderr, "Hiba: a CPU rebuild utan idx=%zu, vart=%zu\n", idx, n);
        exit(1);
    }
}

int arrays_equal(const unsigned int* a, const unsigned int* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    size_t n = 1000000;
    unsigned int max_val = 255;
    int rebuild_on_cpu = 1; /* 1 = CPU rebuild, 0 = GPU rebuild */

    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) max_val = (unsigned int)atoi(argv[2]);
    if (argc > 3) rebuild_on_cpu = atoi(argv[3]);

    printf("OpenCL leszamlalo rendezes\n");
    printf("Elemszam: %zu\n", n);
    printf("Max ertek: %u\n", max_val);
    printf("Visszaepites: %s\n", rebuild_on_cpu ? "CPU" : "GPU");

    unsigned int* input = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* output_gpu = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* output_cpu = (unsigned int*)malloc(n * sizeof(unsigned int));
    unsigned int* histogram = (unsigned int*)calloc(max_val + 1, sizeof(unsigned int));
    unsigned int* prefix = (unsigned int*)calloc(max_val + 2, sizeof(unsigned int));

    if (!input || !output_gpu || !output_cpu || !histogram || !prefix) {
        fprintf(stderr, "Host memoriafoglalasi hiba.\n");
        return 1;
    }

    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < n; i++) {
        input[i] = rand() % (max_val + 1);
    }

    double cpu_start = now_ms();
    cpu_counting_sort(input, output_cpu, n, max_val);
    double cpu_end = now_ms();

    cl_int err;

    cl_uint num_platforms;
    CHECK_CL(clGetPlatformIDs(0, NULL, &num_platforms), "Platformok lekerdezese");

    if (num_platforms == 0) {
        fprintf(stderr, "Nincs OpenCL platform.\n");
        return 1;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    CHECK_CL(clGetPlatformIDs(num_platforms, platforms, NULL), "Platformlista lekerese");

    cl_platform_id platform = platforms[0];
    free(platforms);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU nem elerheto, CPU eszkozre valtas.\n");
        CHECK_CL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL), "CPU device lekerese");
    }

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Eszkoz: %s\n", device_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "Context letrehozas");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL(err, "Command queue letrehozas");

    char* source = load_kernel_source("kernels.cl");
    if (!source) {
        fprintf(stderr, "Nem sikerult betolteni a kernels.cl fajlt.\n");
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CHECK_CL(err, "Program letrehozas");
    free(source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        CHECK_CL(err, "Program build");
    }

    cl_kernel kernel_hist = clCreateKernel(program, "count_histogram_atomic", &err);
    CHECK_CL(err, "Histogram kernel letrehozas");

    cl_kernel kernel_fill = NULL;
    if (!rebuild_on_cpu) {
        kernel_fill = clCreateKernel(program, "fill_output_from_prefix", &err);
        CHECK_CL(err, "Fill kernel letrehozas");
    }

    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "Input buffer letrehozas");

    cl_mem d_hist = clCreateBuffer(context, CL_MEM_READ_WRITE, (max_val + 1) * sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "Histogram buffer letrehozas");

    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(cl_uint), NULL, &err);
    CHECK_CL(err, "Output buffer letrehozas");

    cl_mem d_prefix = NULL;
    if (!rebuild_on_cpu) {
        d_prefix = clCreateBuffer(context, CL_MEM_READ_ONLY, (max_val + 2) * sizeof(cl_uint), NULL, &err);
        CHECK_CL(err, "Prefix buffer letrehozas");
    }

    double h2d_start = now_ms();
    CHECK_CL(clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, n * sizeof(cl_uint), input, 0, NULL, NULL),
             "Input masolas GPU-ra");
    CHECK_CL(clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0, (max_val + 1) * sizeof(cl_uint), histogram, 0, NULL, NULL),
             "Histogram nullazasa GPU-n");
    double h2d_end = now_ms();

    CHECK_CL(clSetKernelArg(kernel_hist, 0, sizeof(cl_mem), &d_input), "Kernel arg 0");
    CHECK_CL(clSetKernelArg(kernel_hist, 1, sizeof(cl_mem), &d_hist), "Kernel arg 1");
    CHECK_CL(clSetKernelArg(kernel_hist, 2, sizeof(cl_uint), &n), "Kernel arg 2");

    size_t local_size = 256;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    cl_event evt_hist;
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_hist, 1, NULL, &global_size, &local_size, 0, NULL, &evt_hist),
             "Histogram kernel futtatasa");
    CHECK_CL(clFinish(queue), "Kernel finish");

    cl_ulong hist_start_ns, hist_end_ns;
    clGetEventProfilingInfo(evt_hist, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &hist_start_ns, NULL);
    clGetEventProfilingInfo(evt_hist, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &hist_end_ns, NULL);
    double hist_ms = (hist_end_ns - hist_start_ns) / 1e6;
    clReleaseEvent(evt_hist);

    double d2h_hist_start = now_ms();
    CHECK_CL(clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, (max_val + 1) * sizeof(cl_uint), histogram, 0, NULL, NULL),
             "Histogram visszaolvasasa");
    double d2h_hist_end = now_ms();

    double rebuild_start = now_ms();

    if (rebuild_on_cpu) {
        cpu_rebuild_from_histogram(histogram, output_gpu, n, max_val);
    } else {
        prefix[0] = 0;
        for (unsigned int i = 0; i <= max_val; i++) {
            prefix[i + 1] = prefix[i] + histogram[i];
        }

        CHECK_CL(clEnqueueWriteBuffer(queue, d_prefix, CL_TRUE, 0, (max_val + 2) * sizeof(cl_uint), prefix, 0, NULL, NULL),
                 "Prefix masolas GPU-ra");

        unsigned int max_value_plus_one = max_val + 1;
        CHECK_CL(clSetKernelArg(kernel_fill, 0, sizeof(cl_mem), &d_prefix), "Fill arg 0");
        CHECK_CL(clSetKernelArg(kernel_fill, 1, sizeof(cl_mem), &d_output), "Fill arg 1");
        CHECK_CL(clSetKernelArg(kernel_fill, 2, sizeof(cl_uint), &max_value_plus_one), "Fill arg 2");

        size_t fill_global = max_value_plus_one;
        size_t fill_local = 64;
        if (fill_global % fill_local != 0) {
            fill_global = ((fill_global + fill_local - 1) / fill_local) * fill_local;
        }

        cl_event evt_fill;
        CHECK_CL(clEnqueueNDRangeKernel(queue, kernel_fill, 1, NULL, &fill_global, &fill_local, 0, NULL, &evt_fill),
                 "Fill kernel futtatasa");
        CHECK_CL(clFinish(queue), "Fill finish");

        CHECK_CL(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, n * sizeof(cl_uint), output_gpu, 0, NULL, NULL),
                 "Output visszaolvasasa");

        clReleaseEvent(evt_fill);
    }

    double rebuild_end = now_ms();

    int ok = arrays_equal(output_cpu, output_gpu, n);

    printf("\n--- Eredmenyek ---\n");
    printf("CPU counting sort ido:          %.3f ms\n", cpu_end - cpu_start);
    printf("Host->Device masolas:           %.3f ms\n", h2d_end - h2d_start);
    printf("GPU histogram kernel ido:       %.3f ms\n", hist_ms);
    printf("Device->Host histogram ido:     %.3f ms\n", d2h_hist_end - d2h_hist_start);
    printf("Visszaepites ido (%s):          %.3f ms\n",
           rebuild_on_cpu ? "CPU" : "GPU",
           rebuild_end - rebuild_start);

    double total_pipeline =
        (h2d_end - h2d_start) +
        hist_ms +
        (d2h_hist_end - d2h_hist_start) +
        (rebuild_end - rebuild_start);

    printf("Teljes OpenCL pipeline ido:     %.3f ms\n", total_pipeline);
    printf("Helyes eredmeny:                %s\n", ok ? "IGEN" : "NEM");

    if (!ok) {
        fprintf(stderr, "Az eredmeny hibas!\n");
    }

    clReleaseMemObject(d_input);
    clReleaseMemObject(d_hist);
    clReleaseMemObject(d_output);
    if (d_prefix) clReleaseMemObject(d_prefix);

    clReleaseKernel(kernel_hist);
    if (kernel_fill) clReleaseKernel(kernel_fill);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(input);
    free(output_gpu);
    free(output_cpu);
    free(histogram);
    free(prefix);

    return ok ? 0 : 1;
}