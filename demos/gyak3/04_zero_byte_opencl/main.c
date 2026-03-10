#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "kernel_loader.h"

#define CHECK_CL(err, msg) \
    do { \
        if ((err) != CL_SUCCESS) { \
            fprintf(stderr, "%s failed with error %d\n", (msg), (int)(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);
    return (double)(counter.QuadPart * 1000.0 / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

static unsigned long long get_event_time_ms(cl_event event) {
    cl_ulong start = 0, end = 0;
    cl_int err;

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    CHECK_CL(err, "clGetEventProfilingInfo START");

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    CHECK_CL(err, "clGetEventProfilingInfo END");

    return (unsigned long long)((end - start) / 1000000ULL);
}

static unsigned char *read_file_binary(const char *filename, size_t *out_size, double *read_time_ms) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(f);
        return NULL;
    }

    long size = ftell(f);
    if (size < 0) {
        perror("ftell");
        fclose(f);
        return NULL;
    }
    rewind(f);

    unsigned char *buffer = (unsigned char *)malloc((size_t)size);
    if (!buffer) {
        fprintf(stderr, "malloc failed for input buffer\n");
        fclose(f);
        return NULL;
    }

    {
        double t0 = now_ms();
        size_t read_bytes = fread(buffer, 1, (size_t)size, f);
        double t1 = now_ms();

        fclose(f);

        if (read_bytes != (size_t)size) {
            fprintf(stderr, "fread failed: expected %ld bytes, got %lu\n",
                    size, (unsigned long)read_bytes);
            free(buffer);
            return NULL;
        }

        *out_size = (size_t)size;
        *read_time_ms = t1 - t0;
    }

    return buffer;
}

static void append_csv_result(const char *csv_file,
                              const char *input_file,
                              size_t file_size,
                              size_t local_size,
                              size_t global_size,
                              unsigned long long zero_count,
                              double read_ms,
                              unsigned long long write_ms,
                              unsigned long long kernel_ms,
                              unsigned long long readback_ms,
                              double total_ms) {
    FILE *f = fopen(csv_file, "a");
    if (!f) {
        perror("fopen csv");
        return;
    }

    fprintf(f,
            "%s,%lu,%lu,%lu,%llu,%.3f,%llu,%llu,%llu,%.3f\n",
            input_file,
            (unsigned long)file_size,
            (unsigned long)local_size,
            (unsigned long)global_size,
            zero_count,
            read_ms,
            write_ms,
            kernel_ms,
            readback_ms,
            total_ms);

    fclose(f);
}

static void ensure_csv_header(const char *csv_file) {
    FILE *f = fopen(csv_file, "r");
    if (f) {
        fclose(f);
        return;
    }

    f = fopen(csv_file, "w");
    if (!f) {
        perror("fopen csv header");
        return;
    }

    fprintf(f, "file,file_size,local_size,global_size,zero_count,file_read_ms,host_to_device_ms,kernel_ms,device_to_host_ms,total_ms\n");
    fclose(f);
}

int main(int argc, char **argv) {
    const char *input_file;
    size_t local_size = 256;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_file> [local_size]\n", argv[0]);
        return EXIT_FAILURE;
    }

    input_file = argv[1];

    if (argc >= 3) {
        local_size = (size_t)strtoul(argv[2], NULL, 10);
        if (local_size == 0) {
            fprintf(stderr, "Invalid local_size\n");
            return EXIT_FAILURE;
        }
    }

    ensure_csv_header("results/measurements.csv");

    {
        double total_t0 = now_ms();

        size_t file_size = 0;
        double file_read_ms = 0.0;
        unsigned char *host_data = read_file_binary(input_file, &file_size, &file_read_ms);

        cl_int err;
        cl_uint num_platforms = 0;
        cl_platform_id *platforms = NULL;
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        char *kernel_source = NULL;
        const char *sources[1];
        cl_program program;
        cl_kernel kernel;
        size_t global_size;
        size_t num_groups;
        uint32_t *host_partial = NULL;
        cl_mem data_buffer;
        cl_mem partial_buffer;
        cl_event write_event, kernel_event, read_event;
        unsigned long long zero_count = 0;
        unsigned long long write_ms, kernel_ms, readback_ms;
        double total_t1, total_ms;
        size_t i;

        if (!host_data) {
            return EXIT_FAILURE;
        }

        err = clGetPlatformIDs(0, NULL, &num_platforms);
        CHECK_CL(err, "clGetPlatformIDs count");

        if (num_platforms == 0) {
            fprintf(stderr, "No OpenCL platforms found\n");
            free(host_data);
            return EXIT_FAILURE;
        }

        platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
        if (!platforms) {
            fprintf(stderr, "malloc failed for platforms\n");
            free(host_data);
            return EXIT_FAILURE;
        }

        err = clGetPlatformIDs(num_platforms, platforms, NULL);
        CHECK_CL(err, "clGetPlatformIDs list");

        platform = platforms[0];
        free(platforms);

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "GPU not available, falling back to CPU\n");
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            CHECK_CL(err, "clGetDeviceIDs CPU");
        }

        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        CHECK_CL(err, "clCreateContext");

        queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_CL(err, "clCreateCommandQueue");

        kernel_source = load_text_file("kernels/count_zero_bytes.cl");
        if (!kernel_source) {
            fprintf(stderr, "Failed to load kernel source\n");
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            free(host_data);
            return EXIT_FAILURE;
        }

        sources[0] = kernel_source;
        program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
        CHECK_CL(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            char *log = NULL;

            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            log = (char *)malloc(log_size + 1);
            if (log) {
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                log[log_size] = '\0';
                fprintf(stderr, "Build log:\n%s\n", log);
                free(log);
            }
            CHECK_CL(err, "clBuildProgram");
        }

        kernel = clCreateKernel(program, "count_zero_bytes", &err);
        CHECK_CL(err, "clCreateKernel");

        global_size = ((file_size + local_size - 1) / local_size) * local_size;
        num_groups = global_size / local_size;

        host_partial = (uint32_t *)malloc(num_groups * sizeof(uint32_t));
        if (!host_partial) {
            fprintf(stderr, "malloc failed for host_partial\n");
            free(kernel_source);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            free(host_data);
            return EXIT_FAILURE;
        }

        data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, file_size, NULL, &err);
        CHECK_CL(err, "clCreateBuffer data_buffer");

        partial_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(uint32_t), NULL, &err);
        CHECK_CL(err, "clCreateBuffer partial_buffer");

        err = clEnqueueWriteBuffer(queue, data_buffer, CL_TRUE, 0, file_size, host_data, 0, NULL, &write_event);
        CHECK_CL(err, "clEnqueueWriteBuffer");

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_buffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_ulong), &file_size);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &partial_buffer);
        err |= clSetKernelArg(kernel, 3, local_size * sizeof(uint32_t), NULL);
        CHECK_CL(err, "clSetKernelArg");

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event);
        CHECK_CL(err, "clEnqueueNDRangeKernel");

        err = clEnqueueReadBuffer(queue, partial_buffer, CL_TRUE, 0,
                                  num_groups * sizeof(uint32_t), host_partial, 0, NULL, &read_event);
        CHECK_CL(err, "clEnqueueReadBuffer partial_buffer");

        clFinish(queue);

        for (i = 0; i < num_groups; ++i) {
            zero_count += host_partial[i];
        }

        write_ms = get_event_time_ms(write_event);
        kernel_ms = get_event_time_ms(kernel_event);
        readback_ms = get_event_time_ms(read_event);

        total_t1 = now_ms();
        total_ms = total_t1 - total_t0;

        printf("Input file          : %s\n", input_file);
        printf("File size           : %lu bytes\n", (unsigned long)file_size);
        printf("Local size          : %lu\n", (unsigned long)local_size);
        printf("Global size         : %lu\n", (unsigned long)global_size);
        printf("Zero byte count     : %llu\n", zero_count);
        printf("File read time      : %.3f ms\n", file_read_ms);
        printf("Host -> Device time : %llu ms\n", write_ms);
        printf("Kernel time         : %llu ms\n", kernel_ms);
        printf("Device -> Host time : %llu ms\n", readback_ms);
        printf("Total time          : %.3f ms\n", total_ms);

        append_csv_result("results/measurements.csv",
                          input_file,
                          file_size,
                          local_size,
                          global_size,
                          zero_count,
                          file_read_ms,
                          write_ms,
                          kernel_ms,
                          readback_ms,
                          total_ms);

        clReleaseEvent(write_event);
        clReleaseEvent(kernel_event);
        clReleaseEvent(read_event);
        clReleaseMemObject(data_buffer);
        clReleaseMemObject(partial_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        free(host_partial);
        free(kernel_source);
        free(host_data);
    }

    return EXIT_SUCCESS;
}