#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "kernel_loader.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_DATA_SIZE 10000000UL
#define DEFAULT_LOCAL_SIZE 256UL
#define MAX_RANDOM_VALUE 1000

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
        data[i] = rand() % (MAX_RANDOM_VALUE + 1);
    }
}

static void cpu_reference(const int* data,
                          size_t n,
                          unsigned long long* sum,
                          unsigned long long* sum_sq,
                          double* mean,
                          double* variance,
                          double* stddev)
{
    size_t i;
    unsigned long long local_sum = 0ULL;
    unsigned long long local_sum_sq = 0ULL;

    for (i = 0; i < n; ++i) {
        unsigned long long x = (unsigned long long)data[i];
        local_sum += x;
        local_sum_sq += x * x;
    }

    *sum = local_sum;
    *sum_sq = local_sum_sq;
    *mean = (double)local_sum / (double)n;
    *variance = ((double)local_sum_sq / (double)n) - ((*mean) * (*mean));
    if (*variance < 0.0) {
        *variance = 0.0;
    }
    *stddev = sqrt(*variance);
}

static void save_results(const char* filename,
                         size_t n,
                         size_t local_size,
                         size_t num_groups,
                         double write_ms,
                         double kernel_ms,
                         double read_ms,
                         double total_gpu_ms,
                         double cpu_ms,
                         unsigned long long gpu_sum,
                         unsigned long long gpu_sum_sq,
                         double gpu_mean,
                         double gpu_variance,
                         double gpu_stddev,
                         int ok)
{
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("[FIGYELEM] Nem sikerult letrehozni: %s\n", filename);
        return;
    }

    fprintf(fp, "Feladat: szoras szamitasa OpenCL-lel\n");
    fprintf(fp, "Elemszam: %lu\n", (unsigned long)n);
    fprintf(fp, "Local work size: %lu\n", (unsigned long)local_size);
    fprintf(fp, "Work-groupok szama: %lu\n", (unsigned long)num_groups);
    fprintf(fp, "Host->Device masolas: %.3f ms\n", write_ms);
    fprintf(fp, "Kernel ido: %.3f ms\n", kernel_ms);
    fprintf(fp, "Device->Host masolas: %.3f ms\n", read_ms);
    fprintf(fp, "GPU osszesen (masolas+kernel): %.3f ms\n", total_gpu_ms);
    fprintf(fp, "CPU ido: %.3f ms\n", cpu_ms);
    fprintf(fp, "GPU sum: %llu\n", gpu_sum);
    fprintf(fp, "GPU sum_sq: %llu\n", gpu_sum_sq);
    fprintf(fp, "GPU atlag: %.10f\n", gpu_mean);
    fprintf(fp, "GPU variancia: %.10f\n", gpu_variance);
    fprintf(fp, "GPU szoras: %.10f\n", gpu_stddev);
    fprintf(fp, "Ellenorzes: %s\n", ok ? "SIKERES" : "HIBAS");

    fclose(fp);
}

static void append_csv(const char* filename,
                       size_t n,
                       size_t local_size,
                       size_t num_groups,
                       double write_ms,
                       double kernel_ms,
                       double read_ms,
                       double total_gpu_ms,
                       double cpu_ms,
                       double gpu_mean,
                       double gpu_variance,
                       double gpu_stddev,
                       int ok)
{
    FILE* fp = fopen(filename, "a");
    if (fp == NULL) {
        printf("[FIGYELEM] Nem sikerult megnyitni: %s\n", filename);
        return;
    }

    fprintf(fp, "%lu,%lu,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.10f,%.10f,%.10f,%s\n",
            (unsigned long)n,
            (unsigned long)local_size,
            (unsigned long)num_groups,
            write_ms,
            kernel_ms,
            read_ms,
            total_gpu_ms,
            cpu_ms,
            gpu_mean,
            gpu_variance,
            gpu_stddev,
            ok ? "OK" : "FAIL");

    fclose(fp);
}

int main(int argc, char* argv[])
{
    size_t data_size = DEFAULT_DATA_SIZE;
    size_t local_work_size = DEFAULT_LOCAL_SIZE;
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
    cl_mem partial_sum_buffer = NULL;
    cl_mem partial_sum_sq_buffer = NULL;
    cl_event write_event = NULL;
    cl_event kernel_event = NULL;
    cl_event read_sum_event = NULL;
    cl_event read_sum_sq_event = NULL;

    char* kernel_code;
    int kernel_loader_error = 0;

    int* host_input = NULL;
    unsigned long long* partial_sums = NULL;
    unsigned long long* partial_sum_sqs = NULL;

    unsigned long long cpu_sum = 0ULL;
    unsigned long long cpu_sum_sq = 0ULL;
    double cpu_mean = 0.0;
    double cpu_variance = 0.0;
    double cpu_stddev = 0.0;

    unsigned long long gpu_sum = 0ULL;
    unsigned long long gpu_sum_sq = 0ULL;
    double gpu_mean = 0.0;
    double gpu_variance = 0.0;
    double gpu_stddev = 0.0;

    double cpu_ms = 0.0;
    double write_ms = 0.0;
    double kernel_ms = 0.0;
    double read_ms = 0.0;
    double total_gpu_ms = 0.0;

    size_t global_work_size;
    size_t num_groups;
    size_t i;
    int ok = 0;

    clock_t cpu_start;
    clock_t cpu_end;

    if (argc >= 2) {
        data_size = (size_t)strtoull(argv[1], NULL, 10);
    }
    if (argc >= 3) {
        local_work_size = (size_t)strtoull(argv[2], NULL, 10);
    }

    num_groups = (data_size + local_work_size - 1) / local_work_size;
    global_work_size = num_groups * local_work_size;

    printf("OpenCL szoras szamitas\n");
    printf("Elemszam: %lu\n", (unsigned long)data_size);
    printf("Local work size: %lu\n", (unsigned long)local_work_size);
    printf("Work-groupok szama: %lu\n\n", (unsigned long)num_groups);

    host_input = (int*)malloc(data_size * sizeof(int));
    partial_sums = (unsigned long long*)malloc(num_groups * sizeof(unsigned long long));
    partial_sum_sqs = (unsigned long long*)malloc(num_groups * sizeof(unsigned long long));

    if (host_input == NULL || partial_sums == NULL || partial_sum_sqs == NULL) {
        printf("[HIBA] Nem sikerult memoriat foglalni.\n");
        goto cleanup;
    }

    fill_random_data(host_input, data_size, seed);

    cpu_start = clock();
    cpu_reference(host_input,
                  data_size,
                  &cpu_sum,
                  &cpu_sum_sq,
                  &cpu_mean,
                  &cpu_variance,
                  &cpu_stddev);
    cpu_end = clock();
    cpu_ms = ((double)(cpu_end - cpu_start) * 1000.0) / (double)CLOCKS_PER_SEC;

    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (!check_error(err, "clGetPlatformIDs")) goto cleanup;

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[INFO] GPU nem erheto el, probalkozas CPU OpenCL eszkozzel...\n");
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &n_devices);
        if (!check_error(err, "clGetDeviceIDs (CPU fallback)")) goto cleanup;
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (!check_error(err, "clCreateContext")) goto cleanup;

    kernel_code = load_kernel_source("kernels/stddev.cl", &kernel_loader_error);
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

    kernel = clCreateKernel(program, "reduce_sum_and_sumsq", &err);
    if (!check_error(err, "clCreateKernel")) goto cleanup;

    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!check_error(err, "clCreateCommandQueue")) goto cleanup;

    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  data_size * sizeof(int), NULL, &err);
    if (!check_error(err, "clCreateBuffer(input_buffer)")) goto cleanup;

    partial_sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        num_groups * sizeof(cl_ulong), NULL, &err);
    if (!check_error(err, "clCreateBuffer(partial_sum_buffer)")) goto cleanup;

    partial_sum_sq_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                           num_groups * sizeof(cl_ulong), NULL, &err);
    if (!check_error(err, "clCreateBuffer(partial_sum_sq_buffer)")) goto cleanup;

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

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    if (!check_error(err, "clSetKernelArg(0)")) goto cleanup;

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &partial_sum_buffer);
    if (!check_error(err, "clSetKernelArg(1)")) goto cleanup;

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &partial_sum_sq_buffer);
    if (!check_error(err, "clSetKernelArg(2)")) goto cleanup;

    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &data_size);
    if (!check_error(err, "clSetKernelArg(3)")) goto cleanup;

    err = clSetKernelArg(kernel, 4, local_work_size * sizeof(cl_ulong), NULL);
    if (!check_error(err, "clSetKernelArg(4 - local sums)")) goto cleanup;

    err = clSetKernelArg(kernel, 5, local_work_size * sizeof(cl_ulong), NULL);
    if (!check_error(err, "clSetKernelArg(5 - local sumsq)")) goto cleanup;

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
                              partial_sum_buffer,
                              CL_FALSE,
                              0,
                              num_groups * sizeof(cl_ulong),
                              partial_sums,
                              0,
                              NULL,
                              &read_sum_event);
    if (!check_error(err, "clEnqueueReadBuffer(partial_sum_buffer)")) goto cleanup;

    err = clEnqueueReadBuffer(queue,
                              partial_sum_sq_buffer,
                              CL_FALSE,
                              0,
                              num_groups * sizeof(cl_ulong),
                              partial_sum_sqs,
                              0,
                              NULL,
                              &read_sum_sq_event);
    if (!check_error(err, "clEnqueueReadBuffer(partial_sum_sq_buffer)")) goto cleanup;

    err = clFinish(queue);
    if (!check_error(err, "clFinish")) goto cleanup;

    for (i = 0; i < num_groups; ++i) {
        gpu_sum += partial_sums[i];
        gpu_sum_sq += partial_sum_sqs[i];
    }

    gpu_mean = (double)gpu_sum / (double)data_size;
    gpu_variance = ((double)gpu_sum_sq / (double)data_size) - (gpu_mean * gpu_mean);
    if (gpu_variance < 0.0) {
        gpu_variance = 0.0;
    }
    gpu_stddev = sqrt(gpu_variance);

    write_ms = event_elapsed_ms(write_event);
    kernel_ms = event_elapsed_ms(kernel_event);
    read_ms = event_elapsed_ms(read_sum_event) + event_elapsed_ms(read_sum_sq_event);
    total_gpu_ms = write_ms + kernel_ms + read_ms;

    ok = (gpu_sum == cpu_sum) &&
         (gpu_sum_sq == cpu_sum_sq) &&
         (fabs(gpu_stddev - cpu_stddev) < 1e-9);

    printf("CPU sum: %llu\n", cpu_sum);
    printf("CPU sum_sq: %llu\n", cpu_sum_sq);
    printf("CPU atlag: %.10f\n", cpu_mean);
    printf("CPU variancia: %.10f\n", cpu_variance);
    printf("CPU szoras: %.10f\n\n", cpu_stddev);

    printf("GPU sum: %llu\n", gpu_sum);
    printf("GPU sum_sq: %llu\n", gpu_sum_sq);
    printf("GPU atlag: %.10f\n", gpu_mean);
    printf("GPU variancia: %.10f\n", gpu_variance);
    printf("GPU szoras: %.10f\n\n", gpu_stddev);

    printf("CPU ido: %.3f ms\n", cpu_ms);
    printf("Host->Device masolas: %.3f ms\n", write_ms);
    printf("Kernel ido: %.3f ms\n", kernel_ms);
    printf("Device->Host masolas: %.3f ms\n", read_ms);
    printf("GPU osszesen: %.3f ms\n", total_gpu_ms);
    printf("Ellenorzes: %s\n", ok ? "SIKERES" : "HIBAS");

    save_results("results/results.txt",
                 data_size,
                 local_work_size,
                 num_groups,
                 write_ms,
                 kernel_ms,
                 read_ms,
                 total_gpu_ms,
                 cpu_ms,
                 gpu_sum,
                 gpu_sum_sq,
                 gpu_mean,
                 gpu_variance,
                 gpu_stddev,
                 ok);

    append_csv("results/results.csv",
               data_size,
               local_work_size,
               num_groups,
               write_ms,
               kernel_ms,
               read_ms,
               total_gpu_ms,
               cpu_ms,
               gpu_mean,
               gpu_variance,
               gpu_stddev,
               ok);

cleanup:
    if (write_event != NULL) clReleaseEvent(write_event);
    if (kernel_event != NULL) clReleaseEvent(kernel_event);
    if (read_sum_event != NULL) clReleaseEvent(read_sum_event);
    if (read_sum_sq_event != NULL) clReleaseEvent(read_sum_sq_event);
    if (input_buffer != NULL) clReleaseMemObject(input_buffer);
    if (partial_sum_buffer != NULL) clReleaseMemObject(partial_sum_buffer);
    if (partial_sum_sq_buffer != NULL) clReleaseMemObject(partial_sum_sq_buffer);
    if (queue != NULL) clReleaseCommandQueue(queue);
    if (kernel != NULL) clReleaseKernel(kernel);
    if (program != NULL) clReleaseProgram(program);
    if (context != NULL) clReleaseContext(context);

    free(host_input);
    free(partial_sums);
    free(partial_sum_sqs);

    return 0;
}