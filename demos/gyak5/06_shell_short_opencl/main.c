#define _CRT_SECURE_NO_WARNINGS
#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernel_loader.h"

#define MAX_SOURCE_SIZE (1 << 20)

typedef enum {
    INPUT_RANDOM = 0,
    INPUT_SORTED,
    INPUT_REVERSED,
    INPUT_NEARLY_SORTED
} input_type_t;

typedef enum {
    GAP_SHELL = 0,
    GAP_KNUTH,
    GAP_CIURA
} gap_type_t;

static double now_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}

static void fill_random(int* a, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        a[i] = rand() % 100000;
    }
}

static int cmp_int(const void* x, const void* y) {
    int a = *(const int*)x;
    int b = *(const int*)y;
    return (a > b) - (a < b);
}

static void make_input(int* a, size_t n, input_type_t type) {
    fill_random(a, n, 12345u);

    if (type == INPUT_RANDOM) {
        return;
    }

    qsort(a, n, sizeof(int), cmp_int);

    if (type == INPUT_SORTED) {
        return;
    }

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
        size_t swaps = n / 100 + 1; // kb. 1% zavarás
        for (size_t i = 0; i < swaps; ++i) {
            size_t x = (size_t)(rand() % (int)n);
            size_t y = (size_t)(rand() % (int)n);
            int t = a[x];
            a[x] = a[y];
            a[y] = t;
        }
    }
}

static int is_sorted(const int* a, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (a[i - 1] > a[i]) return 0;
    }
    return 1;
}

static void shell_sort_cpu(int* a, size_t n, gap_type_t gap_type) {
    int gaps[128];
    int gap_count = 0;

    if (gap_type == GAP_SHELL) {
        for (size_t gap = n / 2; gap > 0; gap /= 2) {
            gaps[gap_count++] = (int)gap;
            if (gap == 1) break;
        }
    } else if (gap_type == GAP_KNUTH) {
        int h = 1;
        while (h < (int)n / 3) h = 3 * h + 1;
        while (h >= 1) {
            gaps[gap_count++] = h;
            h /= 3;
        }
    } else if (gap_type == GAP_CIURA) {
        int base[] = {701, 301, 132, 57, 23, 10, 4, 1};
        int base_count = (int)(sizeof(base) / sizeof(base[0]));

        // Előbb a nagyobb, n-hez igazított elemek
        int ext[128];
        int ext_count = 0;

        int x = 1750;
        while (x < (int)n) {
            x = (int)(x * 2.25);
        }
        while (x > (int)n) {
            x = (int)(x / 2.25);
        }
        while (x > 701) {
            ext[ext_count++] = x;
            x = (int)(x / 2.25);
        }

        for (int i = 0; i < ext_count; ++i) {
            if (ext[i] > 0) gaps[gap_count++] = ext[i];
        }
        for (int i = 0; i < base_count; ++i) {
            if (base[i] < (int)n) gaps[gap_count++] = base[i];
        }
    }

    for (int g = 0; g < gap_count; ++g) {
        int gap = gaps[g];
        for (int i = gap; i < (int)n; ++i) {
            int temp = a[i];
            int j = i;
            while (j >= gap && a[j - gap] > temp) {
                a[j] = a[j - gap];
                j -= gap;
            }
            a[j] = temp;
        }
    }
}

static int build_gaps(size_t n, gap_type_t gap_type, int* gaps, int max_gaps) {
    int count = 0;

    if (gap_type == GAP_SHELL) {
        for (size_t gap = n / 2; gap > 0 && count < max_gaps; gap /= 2) {
            gaps[count++] = (int)gap;
            if (gap == 1) break;
        }
    } else if (gap_type == GAP_KNUTH) {
        int tmp[128];
        int tcount = 0;
        int h = 1;
        while (h < (int)n / 3 && tcount < 128) {
            tmp[tcount++] = h;
            h = 3 * h + 1;
        }
        if (tcount < 128) tmp[tcount++] = h;

        for (int i = tcount - 1; i >= 0 && count < max_gaps; --i) {
            if (tmp[i] < (int)n) gaps[count++] = tmp[i];
        }
    } else if (gap_type == GAP_CIURA) {
        int base[] = {701, 301, 132, 57, 23, 10, 4, 1};
        int base_count = (int)(sizeof(base) / sizeof(base[0]));
        int ext[128];
        int ext_count = 0;

        int x = 701;
        while (x < (int)n && ext_count < 128) {
            x = (int)(x * 2.25);
            if (x < (int)n) ext[ext_count++] = x;
        }

        for (int i = ext_count - 1; i >= 0 && count < max_gaps; --i) {
            gaps[count++] = ext[i];
        }
        for (int i = 0; i < base_count && count < max_gaps; ++i) {
            if (base[i] < (int)n) gaps[count++] = base[i];
        }
    }

    return count;
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

static const char* gap_type_name(gap_type_t t) {
    switch (t) {
        case GAP_SHELL: return "shell";
        case GAP_KNUTH: return "knuth";
        case GAP_CIURA: return "ciura";
        default: return "unknown";
    }
}

static void append_csv_header_if_needed(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (f) {
        fclose(f);
        return;
    }

    f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "input_type,n,gap_seq,local_size,host_to_device_ms,kernel_ms,device_to_host_ms,total_gpu_ms,cpu_ms,correct\n");
    fclose(f);
}

static void append_csv_result(
    const char* filename,
    input_type_t input_type,
    size_t n,
    gap_type_t gap_type,
    size_t local_size,
    double h2d_ms,
    double kernel_ms,
    double d2h_ms,
    double total_gpu_ms,
    double cpu_ms,
    int correct
) {
    FILE* f = fopen(filename, "a");
    if (!f) return;

    fprintf(f, "%s,%lu,%s,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
        input_type_name(input_type),
        (unsigned long)n,
        gap_type_name(gap_type),
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
    gap_type_t gap_type = GAP_SHELL;
    size_t local_size = 64;
    const char* csv_file = "results.csv";

    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) input_type = (input_type_t)atoi(argv[2]); // 0..3
    if (argc > 3) gap_type = (gap_type_t)atoi(argv[3]);     // 0..2
    if (argc > 4) local_size = (size_t)atoll(argv[4]);

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

    make_input(data, n, input_type);
    memcpy(cpu_ref, data, n * sizeof(int));
    memcpy(gpu_out, data, n * sizeof(int));

    double cpu_start = now_ms();
    shell_sort_cpu(cpu_ref, n, gap_type);
    double cpu_end = now_ms();
    double cpu_ms = cpu_end - cpu_start;

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
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU nem talalhato, probalkozas CPU-val...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs hiba: %d\n", err);
            return 1;
        }
    }

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

    cl_uint cu_count = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu_count), &cu_count, NULL);

    printf("Device: %s, CU=%u\n", device_name, cu_count);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateContext hiba: %d\n", err);
        return 1;
    }

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateCommandQueue hiba: %d\n", err);
        clReleaseContext(context);
        return 1;
    }

    char* source = load_kernel_source("kernel/shell_sort.cl");
    if (!source) {
        fprintf(stderr, "Kernel forras betoltese sikertelen.\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    const char* sources[] = { source };
    program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateProgramWithSource hiba: %d\n", err);
        free(source);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram hiba: %d\n", err);
        print_build_log(program, device);
        free(source);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    kernel = clCreateKernel(program, "shell_pass", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateKernel hiba: %d\n", err);
        free(source);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clCreateBuffer hiba: %d\n", err);
        free(source);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    cl_event ev_write, ev_kernel, ev_read;
    double h2d_ms = 0.0, kernel_ms_total = 0.0, d2h_ms = 0.0, total_gpu_ms = 0.0;

    double gpu_total_start = now_ms();

    err = clEnqueueWriteBuffer(queue, d_data, CL_TRUE, 0, n * sizeof(int), gpu_out, 0, NULL, &ev_write);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer hiba: %d\n", err);
        clReleaseMemObject(d_data);
        free(source);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    clWaitForEvents(1, &ev_write);
    {
        cl_ulong t1, t2;
        clGetEventProfilingInfo(ev_write, CL_PROFILING_COMMAND_START, sizeof(t1), &t1, NULL);
        clGetEventProfilingInfo(ev_write, CL_PROFILING_COMMAND_END, sizeof(t2), &t2, NULL);
        h2d_ms = (double)(t2 - t1) / 1e6;
        clReleaseEvent(ev_write);
    }

    int gaps[128];
    int gap_count = build_gaps(n, gap_type, gaps, 128);

    for (int gi = 0; gi < gap_count; ++gi) {
        int gap = gaps[gi];

        size_t global_size = (size_t)gap;
        size_t current_local = local_size;

        if (current_local > global_size) {
            current_local = global_size;
        }
        if (current_local == 0) {
            current_local = 1;
        }

        // Opcionálisan lehetne felkerekíteni global_size-ot local_size többszörösére,
        // de itt nem muszáj, mert a kernel úgyis ellenőrzi a gid >= gap feltételt.
        // Egyes driverek szeretik a többszöröst, de ez az egyszerűség kedvéért most marad így.

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &n);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &gap);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clSetKernelArg hiba: %d\n", err);
            break;
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &current_local, 0, NULL, &ev_kernel);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clEnqueueNDRangeKernel hiba: %d (gap=%d, global=%lu, local=%lu)\n",
                err, gap, (unsigned long)global_size, (unsigned long)current_local);
            break;
        }

        clWaitForEvents(1, &ev_kernel);

        cl_ulong t1, t2;
        clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_START, sizeof(t1), &t1, NULL);
        clGetEventProfilingInfo(ev_kernel, CL_PROFILING_COMMAND_END, sizeof(t2), &t2, NULL);
        kernel_ms_total += (double)(t2 - t1) / 1e6;
        clReleaseEvent(ev_kernel);
    }

    err = clEnqueueReadBuffer(queue, d_data, CL_TRUE, 0, n * sizeof(int), gpu_out, 0, NULL, &ev_read);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer hiba: %d\n", err);
    } else {
        clWaitForEvents(1, &ev_read);
        cl_ulong t1, t2;
        clGetEventProfilingInfo(ev_read, CL_PROFILING_COMMAND_START, sizeof(t1), &t1, NULL);
        clGetEventProfilingInfo(ev_read, CL_PROFILING_COMMAND_END, sizeof(t2), &t2, NULL);
        d2h_ms = (double)(t2 - t1) / 1e6;
        clReleaseEvent(ev_read);
    }

    double gpu_total_end = now_ms();
    total_gpu_ms = gpu_total_end - gpu_total_start;

    int correct = 1;
    if (!is_sorted(gpu_out, n)) {
        correct = 0;
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (gpu_out[i] != cpu_ref[i]) {
                correct = 0;
                fprintf(stderr, "Elteres a %lu indexnel: GPU=%d, CPU=%d\n",
    (unsigned long)i, gpu_out[i], cpu_ref[i]);
                break;
            }
        }
    }

    append_csv_header_if_needed(csv_file);
    append_csv_result(
        csv_file,
        input_type,
        n,
        gap_type,
        local_size,
        h2d_ms,
        kernel_ms_total,
        d2h_ms,
        total_gpu_ms,
        cpu_ms,
        correct
    );

    printf("Input: %s\n", input_type_name(input_type));
    printf("N: %lu\n", (unsigned long)n);
    printf("Gap sequence: %s\n", gap_type_name(gap_type));
    printf("Local size: %lu\n", (unsigned long)local_size);
    printf("CPU ido: %.3f ms\n", cpu_ms);
    printf("Host->Device: %.3f ms\n", h2d_ms);
    printf("Kernel osszesen: %.3f ms\n", kernel_ms_total);
    printf("Device->Host: %.3f ms\n", d2h_ms);
    printf("Teljes GPU ido: %.3f ms\n", total_gpu_ms);
    printf("Helyes eredmeny: %s\n", correct ? "igen" : "nem");

    clReleaseMemObject(d_data);
    free(source);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(data);
    free(cpu_ref);
    free(gpu_out);

    return correct ? 0 : 2;
}