#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIST_SIZE 101
#define DEFAULT_DATA_SIZE 10000000UL
#define DEFAULT_LOCAL_SIZE 256UL

static double event_elapsed_ms(cl_event event)
{
    cl_ulong start = 0;
    cl_ulong end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    return (double)(end - start) / 1000000.0;
}

static void print_build_log(cl_program program, cl_device_id device_id)
{
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size + 1);
        if (log != NULL) {
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            printf("\n--- Build log ---\n%s\n", log);
            free(log);
        }
    }
}

static int check_error(cl_int err, const char* step)
{
    if (err != CL_SUCCESS) {
        printf("[HIBA] %s sikertelen. Hibakod: %d\n", step, err);
        return 0;
    }
    return 1;
}

static void fill_random_data(int* data, size_t n, unsigned int seed)
{
    size_t i;
    srand(seed);
    for (i = 0; i < n; ++i) {
        data[i] = rand() % HIST_SIZE;
    }
}

static void cpu_histogram(const int* data, size_t n, unsigned int* histogram)
{
    size_t i;
    for (i = 0; i < HIST_SIZE; ++i) {
        histogram[i] = 0;
    }
    for (i = 0; i < n; ++i) {
        ++histogram[data[i]];
    }
}

static int verify_histogram(const unsigned int* a, const unsigned int* b)
{
    int i;
    for (i = 0; i < HIST_SIZE; ++i) {
        if (a[i] != b[i]) {
            printf("[ELTERES] Ertek: %d, CPU: %u, OpenCL: %u\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

static void save_results(const char* filename,
                         size_t n,
                         size_t local_size,
                         const char* kernel_name,
                         double write_ms,
                         double kernel_ms,
                         double read_ms,
                         double total_gpu_ms,
                         double cpu_ms,
                         int ok,
                         const unsigned int* histogram)
{
    FILE* fp = fopen(filename, "w");
    int i;
    if (fp == NULL) {
        printf("[FIGYELEM] Nem sikerult letrehozni: %s\n", filename);
        return;
    }

    fprintf(fp, "Feladat: gyakorisagok szamitasa OpenCL-lel\n");
    fprintf(fp, "Elemszam: %lu\n", (unsigned long)n);
    fprintf(fp, "Local work size: %lu\n", (unsigned long)local_size);
    fprintf(fp, "Kernel: %s\n", kernel_name);
    fprintf(fp, "Host->Device masolas: %.3f ms\n", write_ms);
    fprintf(fp, "Kernel ido: %.3f ms\n", kernel_ms);
    fprintf(fp, "Device->Host masolas: %.3f ms\n", read_ms);
    fprintf(fp, "GPU osszesen (masolas+kernel): %.3f ms\n", total_gpu_ms);
    fprintf(fp, "CPU ido: %.3f ms\n", cpu_ms);
    fprintf(fp, "Ellenorzes: %s\n", ok ? "SIKERES" : "HIBAS");
    fprintf(fp, "\nGyakorisagok:\n");
    for (i = 0; i < HIST_SIZE; ++i) {
        fprintf(fp, "%d -> %u\n", i, histogram[i]);
    }

    fclose(fp);
}

static void append_csv(const char* filename,
                       size_t n,
                       size_t local_size,
                       const char* kernel_name,
                       double write_ms,
                       double kernel_ms,
                       double read_ms,
                       double total_gpu_ms,
                       double cpu_ms,
                       int ok)
{
    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        printf("[FIGYELEM] Nem sikerult megnyitni: %s\n", filename);
        return;
    }
    fprintf(fp, "%lu,%lu,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
            (unsigned long)n,
            (unsigned long)local_size,
            kernel_name,
            write_ms,
            kernel_ms,
            read_ms,
            total_gpu_ms,
            cpu_ms,
            ok ? "OK" : "FAIL");
    fclose(fp);
}

int main(int argc, char* argv[])
{
    size_t data_size = DEFAULT_DATA_SIZE;
    size_t local_work_size = DEFAULT_LOCAL_SIZE;
    int use_local_kernel = 1;
    int kernel_loader_error = 0;
    int ok;
    int i;
    unsigned int seed = 12345U;

    cl_int err;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint n_platforms = 0;
    cl_uint n_devices = 0;
    cl_context context = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_command_queue queue = NULL;
    cl_mem input_buffer = NULL;
    cl_mem histogram_buffer = NULL;
    cl_event write_event = NULL;
    cl_event kernel_event = NULL;
    cl_event read_event = NULL;

    char* kernel_code;
    const char* kernel_name;

    int* host_input;
    unsigned int host_histogram[HIST_SIZE];
    unsigned int cpu_reference[HIST_SIZE];
    unsigned int zero_histogram[HIST_SIZE];

    clock_t cpu_start;
    clock_t cpu_end;
    double cpu_ms;
    double write_ms;
    double kernel_ms;
    double read_ms;
    double total_gpu_ms;

    size_t global_work_size;
    size_t n_work_groups;

    if (argc >= 2) {
        data_size = (size_t)strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        local_work_size = (size_t)strtoull(argv[2], NULL, 10);
    }
    if (argc >= 4) {
        use_local_kernel = atoi(argv[3]);
    }

    kernel_name = use_local_kernel ? "histogram_local_atomic" : "histogram_global_atomic";

    printf("OpenCL gyakorisagszamitas\n");
    printf("Elemszam: %lu\n", (unsigned long)data_size);
    printf("Local work size: %lu\n", (unsigned long)local_work_size);
    printf("Kernel valtozat: %s\n\n", kernel_name);

    host_input = (int*)malloc(data_size * sizeof(int));
    if (host_input == NULL) {
        printf("[HIBA] Nem sikerult memoriat foglalni a bemeneti tombnek.\n");
        return 1;
    }

    fill_random_data(host_input, data_size, seed);

    cpu_start = clock();
    cpu_histogram(host_input, data_size, cpu_reference);
    cpu_end = clock();
    cpu_ms = ((double)(cpu_end - cpu_start) * 1000.0) / (double)CLOCKS_PER_SEC;

    memset(zero_histogram, 0, sizeof(zero_histogram));
    memset(host_histogram, 0, sizeof(host_histogram));

    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (!check_error(err, "clGetPlatformIDs")) goto cleanup;

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[INFO] GPU nem erheto el, probalkozas CPU eszkozzel...\n");
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &n_devices);
        if (!check_error(err, "clGetDeviceIDs (CPU fallback)")) goto cleanup;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!check_error(err, "clCreateContext")) goto cleanup;

    kernel_code = load_kernel_source("kernels/histogram.cl", &kernel_loader_error);
    if (kernel_loader_error != 0 || kernel_code == NULL) {
        printf("[HIBA] Nem sikerult betolteni a kernel forrast.\n");
        goto cleanup;
    }

    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, NULL, &err);
    free(kernel_code);
    if (!check_error(err, "clCreateProgramWithSource")) goto cleanup;

    err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[HIBA] clBuildProgram sikertelen.\n");
        print_build_log(program, device_id);
        goto cleanup;
    }

    kernel = clCreateKernel(program, kernel_name, &err);
    if (!check_error(err, "clCreateKernel")) goto cleanup;

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!check_error(err, "clCreateCommandQueue")) goto cleanup;

    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size * sizeof(int), NULL, &err);
    if (!check_error(err, "clCreateBuffer(input_buffer)")) goto cleanup;

    histogram_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, HIST_SIZE * sizeof(unsigned int), NULL, &err);
    if (!check_error(err, "clCreateBuffer(histogram_buffer)")) goto cleanup;

    err = clEnqueueWriteBuffer(queue,
                               input_buffer,
                               CL_FALSE,
                               0,
                               data_size * sizeof(int),
                               host_input,
                               0,
                               NULL,
                               &write_event);
    if (!check_error(err, "clEnqueueWriteBuffer(input_buffer)")) goto cleanup;

    err = clEnqueueWriteBuffer(queue,
                               histogram_buffer,
                               CL_FALSE,
                               0,
                               HIST_SIZE * sizeof(unsigned int),
                               zero_histogram,
                               0,
                               NULL,
                               NULL);
    if (!check_error(err, "clEnqueueWriteBuffer(histogram_buffer)")) goto cleanup;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    if (!check_error(err, "clSetKernelArg(0)")) goto cleanup;
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &histogram_buffer);
    if (!check_error(err, "clSetKernelArg(1)")) goto cleanup;

    if (use_local_kernel) {
        err = clSetKernelArg(kernel, 2, HIST_SIZE * sizeof(unsigned int), NULL);
        if (!check_error(err, "clSetKernelArg(2 - local memory)")) goto cleanup;
        err = clSetKernelArg(kernel, 3, sizeof(int), &data_size);
        if (!check_error(err, "clSetKernelArg(3)")) goto cleanup;
    } else {
        err = clSetKernelArg(kernel, 2, sizeof(int), &data_size);
        if (!check_error(err, "clSetKernelArg(2)")) goto cleanup;
    }

    n_work_groups = (data_size + local_work_size - 1) / local_work_size;
    global_work_size = n_work_groups * local_work_size;

    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 1,
                                 NULL,
                                 &global_work_size,
                                 &local_work_size,
                                 0,
                                 NULL,
                                 &kernel_event);
    if (!check_error(err, "clEnqueueNDRangeKernel")) goto cleanup;

    err = clEnqueueReadBuffer(queue,
                              histogram_buffer,
                              CL_FALSE,
                              0,
                              HIST_SIZE * sizeof(unsigned int),
                              host_histogram,
                              0,
                              NULL,
                              &read_event);
    if (!check_error(err, "clEnqueueReadBuffer")) goto cleanup;

    err = clFinish(queue);
    if (!check_error(err, "clFinish")) goto cleanup;

    write_ms = event_elapsed_ms(write_event);
    kernel_ms = event_elapsed_ms(kernel_event);
    read_ms = event_elapsed_ms(read_event);
    total_gpu_ms = write_ms + kernel_ms + read_ms;

    ok = verify_histogram(cpu_reference, host_histogram);

    printf("CPU ido: %.3f ms\n", cpu_ms);
    printf("Host->Device masolas: %.3f ms\n", write_ms);
    printf("Kernel ido: %.3f ms\n", kernel_ms);
    printf("Device->Host masolas: %.3f ms\n", read_ms);
    printf("GPU osszesen: %.3f ms\n", total_gpu_ms);
    printf("Ellenorzes: %s\n\n", ok ? "SIKERES" : "HIBAS");

    printf("Elso nehany gyakorisag:\n");
    for (i = 0; i < 10; ++i) {
        printf("%d -> %u\n", i, host_histogram[i]);
    }

    save_results("results/results.txt",
                 data_size,
                 local_work_size,
                 kernel_name,
                 write_ms,
                 kernel_ms,
                 read_ms,
                 total_gpu_ms,
                 cpu_ms,
                 ok,
                 host_histogram);

    append_csv("results/results.csv",
               data_size,
               local_work_size,
               kernel_name,
               write_ms,
               kernel_ms,
               read_ms,
               total_gpu_ms,
               cpu_ms,
               ok);

cleanup:
    if (write_event != NULL) clReleaseEvent(write_event);
    if (kernel_event != NULL) clReleaseEvent(kernel_event);
    if (read_event != NULL) clReleaseEvent(read_event);
    if (input_buffer != NULL) clReleaseMemObject(input_buffer);
    if (histogram_buffer != NULL) clReleaseMemObject(histogram_buffer);
    if (queue != NULL) clReleaseCommandQueue(queue);
    if (kernel != NULL) clReleaseKernel(kernel);
    if (program != NULL) clReleaseProgram(program);
    if (context != NULL) clReleaseContext(context);
    free(host_input);

    return 0;
}