#define _CRT_SECURE_NO_WARNINGS
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernel_loader.h"

typedef struct {
    int x;
    int y;
} int2_host;

typedef enum {
    INPUT_RANDOM = 0,
    INPUT_SORTED,
    INPUT_REVERSED,
    INPUT_NEARLY_SORTED
} input_type_t;

static double now_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}

static int cmp_int(const void* a, const void* b) {
    int x = *(const int*)a;
    int y = *(const int*)b;
    return (x > y) - (x < y);
}

static void fill_random(int* a, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        a[i] = rand() % 1000000;
    }
}

static void make_input(int* a, size_t n, input_type_t type) {
    fill_random(a, n, 12345u);

    if (type == INPUT_RANDOM) return;

    qsort(a, n, sizeof(int), cmp_int);

    if (type == INPUT_SORTED) return;

    if (type == INPUT_REVERSED) {
        for (size_t i = 0; i < n / 2; ++i) {
            int t = a[i];
            a[i] = a[n - 1 - i];
            a[n - 1 - i] = t;
        }
        return;
    }

    if (type == INPUT_NEARLY_SORTED) {
        srand(54321u);
        size_t swaps = n / 100 + 1;
        for (size_t i = 0; i < swaps; ++i) {
            size_t x = (size_t)(rand() % (int)n);
            size_t y = (size_t)(rand() % (int)n);
            int t = a[x];
            a[x] = a[y];
            a[y] = t;
        }
    }
}

static const char* input_type_name(input_type_t t) {
    switch (t) {
        case INPUT_RANDOM: return "random";
        case INPUT_SORTED: return "sorted";
        case INPUT_REVERSED: return "reversed";
        case INPUT_NEARLY_SORTED: return "nearly_sorted";
        default: return "unknown";
    }
}

static void quicksort_cpu_range(int* a, int left, int right) {
    int lstack[64];
    int rstack[64];
    int top = 0;

    lstack[top] = left;
    rstack[top] = right;
    top++;

    while (top > 0) {
        top--;
        int l = lstack[top];
        int r = rstack[top];

        while (l < r) {
            int i = l;
            int j = r;
            int pivot = a[l + (r - l) / 2];

            while (i <= j) {
                while (a[i] < pivot) i++;
                while (a[j] > pivot) j--;
                if (i <= j) {
                    int tmp = a[i];
                    a[i] = a[j];
                    a[j] = tmp;
                    i++;
                    j--;
                }
            }

            if ((j - l) < (r - i)) {
                if (i < r) {
                    lstack[top] = i;
                    rstack[top] = r;
                    top++;
                }
                r = j;
            } else {
                if (l < j) {
                    lstack[top] = l;
                    rstack[top] = j;
                    top++;
                }
                l = i;
            }
        }
    }
}

static void cpu_sort_ranges(int* a, const int2_host* ranges, int num_ranges) {
    for (int i = 0; i < num_ranges; ++i) {
        if (ranges[i].x < ranges[i].y) {
            quicksort_cpu_range(a, ranges[i].x, ranges[i].y);
        }
    }
}

static int is_range_sorted(const int* a, int left, int right) {
    for (int i = left + 1; i <= right; ++i) {
        if (a[i - 1] > a[i]) return 0;
    }
    return 1;
}

static int verify_ranges(const int* gpu, const int* cpu, const int2_host* ranges, int num_ranges) {
    for (int r = 0; r < num_ranges; ++r) {
        int left = ranges[r].x;
        int right = ranges[r].y;

        if (!is_range_sorted(gpu, left, right)) {
            fprintf(stderr, "Hiba: a(z) %d. intervallum nincs rendezve: [%d, %d]\n", r, left, right);
            return 0;
        }

        for (int i = left; i <= right; ++i) {
            if (gpu[i] != cpu[i]) {
                fprintf(stderr, "Hiba: elteres a(z) %d. intervallumban, idx=%d, GPU=%d, CPU=%d\n",
                        r, i, gpu[i], cpu[i]);
                return 0;
            }
        }
    }
    return 1;
}

static int build_ranges(size_t n, int block_size, int2_host** out_ranges) {
    int num_ranges = (int)((n + (size_t)block_size - 1) / (size_t)block_size);
    int2_host* ranges = (int2_host*)malloc((size_t)num_ranges * sizeof(int2_host));
    if (!ranges) return 0;

    for (int i = 0; i < num_ranges; ++i) {
        int left = i * block_size;
        int right = left + block_size - 1;
        if (right >= (int)n) right = (int)n - 1;

        ranges[i].x = left;
        ranges[i].y = right;
    }

    *out_ranges = ranges;
    return num_ranges;
}

static void append_csv_header_if_needed(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (f) {
        fclose(f);
        return;
    }

    f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "input_type,n,block_size,num_ranges,local_size,host_to_device_ms,kernel_ms,device_to_host_ms,total_gpu_ms,cpu_ms,correct\n");
    fclose(f);
}

static void append_csv_result(const char* filename,
                              input_type_t input_type,
                              size_t n,
                              int block_size,
                              int num_ranges,
                              size_t local_size,
                              double h2d_ms,
                              double kernel_ms,
                              double d2h_ms,
                              double total_gpu_ms,
                              double cpu_ms,
                              int correct) {
    FILE* f = fopen(filename, "a");
    if (!f) return;

    fprintf(f, "%s,%lu,%d,%d,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
            input_type_name(input_type),
            (unsigned long)n,
            block_size,
            num_ranges,
            (unsigned long)local_size,
            h2d_ms,
            kernel_ms,
            d2h_ms,
            total_gpu_ms,
            cpu_ms,
            correct);

    fclose(f);
}

static void print_build_log(cl_program program, cl_device_id device) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Build log:\n%s\n", log);
            free(log);
        }
    }
}

int main(int argc, char** argv) {
    size_t n = 65536;
    input_type_t input_type = INPUT_RANDOM;
    int block_size = 1024;
    size_t local_size = 64;
    const char* csv_file = "results.csv";

    // argv:
    // 1: n
    // 2: input_type (0..3)
    // 3: block_size
    // 4: local_size
    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) input_type = (input_type_t)atoi(argv[2]);
    if (argc > 3) block_size = atoi(argv[3]);
    if (argc > 4) local_size = (size_t)atoll(argv[4]);

    if (block_size < 2) {
        fprintf(stderr, "A block_size legyen legalabb 2.\n");
        return 1;
    }

    int* data = (int*)malloc(n * sizeof(int));
    int* cpu_ref = (int*)malloc(n * sizeof(int));
    int* gpu_out = (int*)malloc(n * sizeof(int));
    if (!data || !cpu_ref || !gpu_out) {
        fprintf(stderr, "Memoriafoglalasi hiba.\n");
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    int2_host* ranges = NULL;
    int num_ranges = build_ranges(n, block_size, &ranges);
    if (num_ranges <= 0 || !ranges) {
        fprintf(stderr, "Intervallumepitesi hiba.\n");
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    make_input(data, n, input_type);
    memcpy(cpu_ref, data, n * sizeof(int));
    memcpy(gpu_out, data, n * sizeof(int));

    int cpu_repeats = 20;
    double cpu_sum = 0.0;

    for (int rep = 0; rep < cpu_repeats; ++rep) {
        memcpy(cpu_ref, data, n * sizeof(int));

        double cpu_start = now_ms();
        cpu_sort_ranges(cpu_ref, ranges, num_ranges);
        double cpu_end = now_ms();

        cpu_sum += (cpu_end - cpu_start);
    }

    double cpu_ms = cpu_sum / cpu_repeats;

    cl_int err;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs hiba: %d\n", err);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU nem talalhato, probalkozas CPU-val...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs hiba: %d\n", err);
            free(ranges);
            free(data);
            free(cpu_ref);
            free(gpu_out);
            return 1;
        }
    }

    char device_name[256] = {0};
    cl_uint cu_count = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu_count), &cu_count, NULL);

    printf("Device: %s, CU=%u\n", device_name, cu_count);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext hiba: %d\n", err);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue hiba: %d\n", err);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    char* source = load_kernel_source("kernel/quicksort_ranges.cl");
    if (!source) {
        fprintf(stderr, "Kernel forras betoltese sikertelen.\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    const char* sources[] = { source };
    program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource hiba: %d\n", err);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram hiba: %d\n", err);
        print_build_log(program, device);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    kernel = clCreateKernel(program, "quicksort_ranges", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel hiba: %d\n", err);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
    cl_mem d_ranges = clCreateBuffer(context, CL_MEM_READ_ONLY, (size_t)num_ranges * sizeof(int2_host), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer hiba: %d\n", err);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    cl_event ev_write1, ev_write2, ev_kernel, ev_read;
    double h2d_ms = 0.0, kernel_ms = 0.0, d2h_ms = 0.0, total_gpu_ms = 0.0;

    double gpu_total_start = now_ms();

    err = clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, n * sizeof(int), gpu_out, 0, NULL, &ev_write1);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer(data) hiba: %d\n", err);
        clReleaseMemObject(d_ranges);
        clReleaseMemObject(d_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    err = clEnqueueWriteBuffer(queue, d_ranges, CL_TRUE, 0,
                               (size_t)num_ranges * sizeof(int2_host), ranges, 0, NULL, &ev_write2);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer(ranges) hiba: %d\n", err);
        clReleaseEvent(ev_write1);
        clReleaseMemObject(d_ranges);
        clReleaseMemObject(d_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    clWaitForEvents(1, &ev_write1);
    clWaitForEvents(1, &ev_write2);
    {
        cl_ulong s1, e1, s2, e2;
        clGetEventProfilingInfo(ev_write1, CL_PROFILING_COMMAND_START, sizeof(s1), &s1, NULL);
        clGetEventProfilingInfo(ev_write1, CL_PROFILING_COMMAND_END, sizeof(e1), &e1, NULL);
        clGetEventProfilingInfo(ev_write2, CL_PROFILING_COMMAND_START, sizeof(s2), &s2, NULL);
        clGetEventProfilingInfo(ev_write2, CL_PROFILING_COMMAND_END, sizeof(e2), &e2, NULL);
        h2d_ms = (double)(e1 - s1) / 1e6 + (double)(e2 - s2) / 1e6;
        clReleaseEvent(ev_write1);
        clReleaseEvent(ev_write2);
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_ranges);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &num_ranges);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg hiba: %d\n", err);
        clReleaseMemObject(d_ranges);
        clReleaseMemObject(d_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    size_t global_size = (size_t)num_ranges;
    size_t current_local = local_size;

    if (current_local > global_size) current_local = global_size;
    if (current_local == 0) current_local = 1;

    size_t rounded_global =
        ((global_size + current_local - 1) / current_local) * current_local;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
                                 &rounded_global, &current_local, 0, NULL, &ev_kernel);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel hiba: %d (global=%lu, local=%lu)\n",
                err, (unsigned long)rounded_global, (unsigned long)current_local);
        clReleaseMemObject(d_ranges);
        clReleaseMemObject(d_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    clWaitForEvents(1, &ev_kernel);
    {
        cl_ulong s, e;
        clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
        clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
        kernel_ms = (double)(e - s) / 1e6;
        clReleaseEvent(ev_kernel);
    }

    err = clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, n * sizeof(int), gpu_out, 0, NULL, &ev_read);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer hiba: %d\n", err);
        clReleaseMemObject(d_ranges);
        clReleaseMemObject(d_data);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(ranges);
        free(data);
        free(cpu_ref);
        free(gpu_out);
        return 1;
    }

    clWaitForEvents(1, &ev_read);
    {
        cl_ulong s, e;
        clGetEventProfilingInfo(ev_read, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
        clGetEventProfilingInfo(ev_read, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
        d2h_ms = (double)(e - s) / 1e6;
        clReleaseEvent(ev_read);
    }

    total_gpu_ms = now_ms() - gpu_total_start;

    int correct = verify_ranges(gpu_out, cpu_ref, ranges, num_ranges);

    append_csv_header_if_needed(csv_file);
    append_csv_result(csv_file, input_type, n, block_size, num_ranges, local_size,
                      h2d_ms, kernel_ms, d2h_ms, total_gpu_ms, cpu_ms, correct);

    printf("Input: %s\n", input_type_name(input_type));
    printf("N: %lu\n", (unsigned long)n);
    printf("Block size: %d\n", block_size);
    printf("Num ranges: %d\n", num_ranges);
    printf("Requested local size: %lu\n", (unsigned long)local_size);
    printf("CPU ido: %.3f ms\n", cpu_ms);
    printf("Host->Device: %.3f ms\n", h2d_ms);
    printf("Kernel ido: %.3f ms\n", kernel_ms);
    printf("Device->Host: %.3f ms\n", d2h_ms);
    printf("Teljes GPU ido: %.3f ms\n", total_gpu_ms);
    printf("Helyes eredmeny: %s\n", correct ? "igen" : "nem");

    clReleaseMemObject(d_ranges);
    clReleaseMemObject(d_data);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(source);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(ranges);
    free(data);
    free(cpu_ref);
    free(gpu_out);

    return correct ? 0 : 2;
}