#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

static void die_cl(const char* where, cl_int err) {
    fprintf(stderr, "[OpenCL ERROR] %s failed, code=%d\n", where, err);
    exit(1);
}

static cl_device_id pick_device(cl_platform_id* out_platform) {
    cl_int err;
    cl_uint n_platforms = 0;
    err = clGetPlatformIDs(0, NULL, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) die_cl("clGetPlatformIDs(count)", err);

    cl_platform_id* plats = (cl_platform_id*)calloc(n_platforms, sizeof(cl_platform_id));
    if (!plats) {
        fprintf(stderr, "Platform allocation failed.\n");
        exit(1);
    }

    err = clGetPlatformIDs(n_platforms, plats, NULL);
    if (err != CL_SUCCESS) die_cl("clGetPlatformIDs(list)", err);

    cl_device_id chosen = NULL;
    cl_platform_id chosen_plat = NULL;

    for (cl_uint p = 0; p < n_platforms && !chosen; ++p) {
        cl_uint n_dev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &n_dev);
        if (err == CL_SUCCESS && n_dev > 0) {
            err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &chosen, NULL);
            if (err == CL_SUCCESS) chosen_plat = plats[p];
        }
    }

    if (!chosen) {
        for (cl_uint p = 0; p < n_platforms && !chosen; ++p) {
            cl_uint n_dev = 0;
            err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, 0, NULL, &n_dev);
            if (err == CL_SUCCESS && n_dev > 0) {
                err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_CPU, 1, &chosen, NULL);
                if (err == CL_SUCCESS) chosen_plat = plats[p];
            }
        }
    }

    free(plats);

    if (!chosen) {
        fprintf(stderr, "No OpenCL GPU/CPU device found.\n");
        exit(1);
    }

    *out_platform = chosen_plat;
    return chosen;
}

static void print_device_info(cl_device_id dev) {
    char name[256];
    char vendor[256];
    cl_uint cu = 0;
    cl_ulong gmem = 0;

    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);

    printf("Device: %s (%s), CU=%u, GlobalMem=%.2f MB\n",
           name, vendor, cu, (double)gmem / (1024.0 * 1024.0));
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

static void append_csv_row(const char* out_path,
                           int rows, int cols, int iters, int wrap,
                           size_t lx, size_t ly,
                           double h2d_ms, double kernel_ms, double d2h_ms, double total_ms,
                           int tiled)
{
    int exists = file_exists(out_path);
    FILE* f = fopen(out_path, "a");
    if (!f) {
        fprintf(stderr, "Could not open output file: %s\n", out_path);
        return;
    }

    if (!exists) {
        fprintf(f, "rows,cols,iters,wrap,lx,ly,h2d_ms,kernel_ms,d2h_ms,total_ms,tiled\n");
    }

    fprintf(f, "%d,%d,%d,%d,%u,%u,%.6f,%.6f,%.6f,%.6f,%d\n",
            rows, cols, iters, wrap,
            (unsigned)lx, (unsigned)ly,
            h2d_ms, kernel_ms, d2h_ms, total_ms,
            tiled);

    fclose(f);
}

static size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) return value;
    size_t rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

static void usage(const char* argv0) {
    printf("Usage: %s [--rows N] [--cols N] [--iters N] [--seed N] [--wrap 0|1] [--tiled 0|1] [--lx N] [--ly N] [--csv] [--out FILE] [--repeat N] [--warmup N]\n", argv0);
    printf("Defaults: rows=1024 cols=1024 iters=500 seed=time wrap=0 tiled=0 lx=16 ly=16 repeat=1 warmup=0\n");
}

int main(int argc, char** argv) {
    int rows = 1024;
    int cols = 1024;
    int iters = 500;
    unsigned int seed = (unsigned int)time(NULL);
    int wrap = 0;
    int tiled = 0;
    int csv = 0;
    int lx_arg = 16;
    int ly_arg = 16;
    const char* out_path = NULL;
    int repeat = 1;
    int warmup = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--wrap") && i + 1 < argc) wrap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tiled") && i + 1 < argc) tiled = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lx") && i + 1 < argc) lx_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ly") && i + 1 < argc) ly_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")) csv = 1;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else {
            printf("Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (rows <= 0 || cols <= 0 || iters <= 0) {
        fprintf(stderr, "rows/cols/iters must be > 0\n");
        return 1;
    }

    if (lx_arg <= 0 || ly_arg <= 0) {
        fprintf(stderr, "lx/ly must be > 0\n");
        return 1;
    }

    if (repeat <= 0 || warmup < 0) {
        fprintf(stderr, "repeat must be > 0 and warmup must be >= 0\n");
        return 1;
    }

    const size_t n = (size_t)rows * (size_t)cols;
    unsigned char* h_grid = (unsigned char*)malloc(n);
    unsigned char* h_tmp  = (unsigned char*)malloc(n);

    if (!h_grid || !h_tmp) {
        fprintf(stderr, "Host allocation failed (n=%u)\n", (unsigned)n);
        free(h_grid);
        free(h_tmp);
        return 1;
    }

    srand(seed);
    for (size_t k = 0; k < n; ++k) {
        h_grid[k] = (unsigned char)(rand() & 1);
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device = pick_device(&platform);
    print_device_info(device);

    size_t max_wg = 0;
    size_t max_wi[3] = {0, 0, 0};

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wi), max_wi, NULL);

    if ((size_t)lx_arg > max_wi[0] || (size_t)ly_arg > max_wi[1] ||
        (size_t)lx_arg * (size_t)ly_arg > max_wg) {
        fprintf(stderr,
                "Invalid local size lx=%d ly=%d for this device (max_wi=%ux%u, max_wg=%u)\n",
                lx_arg, ly_arg,
                (unsigned)max_wi[0], (unsigned)max_wi[1], (unsigned)max_wg);
        free(h_grid);
        free(h_tmp);
        return 1;
    }

    size_t lx = (size_t)lx_arg;
    size_t ly = (size_t)ly_arg;
    size_t gx = round_up((size_t)cols, lx);
    size_t gy = round_up((size_t)rows, ly);

    size_t global[2] = { gx, gy };
    size_t local[2]  = { lx, ly };

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) die_cl("clCreateContext", err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device,
        (cl_queue_properties[]){ CL_QUEUE_PROPERTIES, (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE, 0 },
        &err
    );
    if (!queue || err != CL_SUCCESS) die_cl("clCreateCommandQueueWithProperties", err);

    int loader_err = 0;
    const char* kernel_path = tiled ? "kernels/gol_tiled.cl" : "kernels/gol_naive.cl";
    const char* kernel_name = tiled ? "gol_step_tiled" : "gol_step";

    const char* src = load_kernel_source(kernel_path, &loader_err);
    if (loader_err != 0 || !src) {
        fprintf(stderr, "Kernel source load failed. Did you run from project root?\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(h_grid);
        free(h_tmp);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    if (!program || err != CL_SUCCESS) die_cl("clCreateProgramWithSource", err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char* log = (char*)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = 0;
            fprintf(stderr, "Build failed:\n%s\n", log);
            free(log);
        }
        die_cl("clBuildProgram", err);
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS) die_cl("clCreateKernel", err);

    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(cl_uchar), NULL, &err);
    if (!d_a || err != CL_SUCCESS) die_cl("clCreateBuffer(d_a)", err);

    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(cl_uchar), NULL, &err);
    if (!d_b || err != CL_SUCCESS) die_cl("clCreateBuffer(d_b)", err);

    double sum_h2d_ms = 0.0;
    double sum_kernel_ms = 0.0;
    double sum_d2h_ms = 0.0;
    double sum_total_ms = 0.0;

    for (int run = 0; run < warmup + repeat; ++run) {
        cl_ulong h2d_ns = 0, kernel_ns = 0, d2h_ns = 0;

        cl_mem cur = d_a;
        cl_mem next = d_b;

        cl_event ev_h2d;
        err = clEnqueueWriteBuffer(queue, cur, CL_FALSE, 0, n * sizeof(cl_uchar), h_grid, 0, NULL, &ev_h2d);
        if (err != CL_SUCCESS) die_cl("clEnqueueWriteBuffer", err);

        clWaitForEvents(1, &ev_h2d);
        {
            cl_ulong s = 0, e = 0;
            clGetEventProfilingInfo(ev_h2d, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
            clGetEventProfilingInfo(ev_h2d, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
            h2d_ns += (e - s);
            clReleaseEvent(ev_h2d);
        }

        for (int t = 0; t < iters; ++t) {
            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cur);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &next);
            err |= clSetKernelArg(kernel, 2, sizeof(int), &rows);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &cols);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &wrap);

            if (tiled) {
                size_t tile_bytes = (lx + 2) * (ly + 2) * sizeof(unsigned char);
                err |= clSetKernelArg(kernel, 5, tile_bytes, NULL);
            }

            if (err != CL_SUCCESS) die_cl("clSetKernelArg", err);

            cl_event ev_k;
            err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &ev_k);
            if (err != CL_SUCCESS) die_cl("clEnqueueNDRangeKernel", err);

            clWaitForEvents(1, &ev_k);
            {
                cl_ulong s = 0, e = 0;
                clGetEventProfilingInfo(ev_k, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
                clGetEventProfilingInfo(ev_k, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
                kernel_ns += (e - s);
                clReleaseEvent(ev_k);
            }

            cl_mem tmp = cur;
            cur = next;
            next = tmp;
        }

        cl_event ev_d2h;
        err = clEnqueueReadBuffer(queue, cur, CL_TRUE, 0, n * sizeof(cl_uchar), h_tmp, 0, NULL, &ev_d2h);
        if (err != CL_SUCCESS) die_cl("clEnqueueReadBuffer", err);

        {
            cl_ulong s = 0, e = 0;
            clGetEventProfilingInfo(ev_d2h, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
            clGetEventProfilingInfo(ev_d2h, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
            d2h_ns += (e - s);
            clReleaseEvent(ev_d2h);
        }

        double h2d_ms = (double)h2d_ns / 1e6;
        double ker_ms = (double)kernel_ns / 1e6;
        double d2h_ms = (double)d2h_ns / 1e6;
        double total_ms = h2d_ms + ker_ms + d2h_ms;

        if (run >= warmup) {
            sum_h2d_ms += h2d_ms;
            sum_kernel_ms += ker_ms;
            sum_d2h_ms += d2h_ms;
            sum_total_ms += total_ms;
        }
    }

    double h2d_ms = sum_h2d_ms / (double)repeat;
    double ker_ms = sum_kernel_ms / (double)repeat;
    double d2h_ms = sum_d2h_ms / (double)repeat;
    double total_ms = sum_total_ms / (double)repeat;

    if (csv) {
        printf("%d,%d,%d,%d,%u,%u,%.6f,%.6f,%.6f,%.6f,%d\n",
               rows, cols, iters, wrap,
               (unsigned)lx, (unsigned)ly,
               h2d_ms, ker_ms, d2h_ms, total_ms, tiled);
    } else {
        printf("GoL OpenCL (%s)\n", tiled ? "tiled" : "naive");
        printf("Grid: %dx%d, iters=%d, wrap=%d\n", rows, cols, iters, wrap);
        printf("Local: %ux%u, Global: %ux%u\n",
               (unsigned)lx, (unsigned)ly,
               (unsigned)gx, (unsigned)gy);
        printf("Warmup: %d, Repeat: %d\n", warmup, repeat);
        printf("H2D:   %.3f ms\n", h2d_ms);
        printf("Kernel:%.3f ms (sum over iters)\n", ker_ms);
        printf("D2H:   %.3f ms\n", d2h_ms);
        printf("Total: %.3f ms\n", total_ms);
        printf("Per-iter kernel: %.6f ms\n", ker_ms / (double)iters);
    }

    if (out_path) {
        append_csv_row(out_path, rows, cols, iters, wrap, lx, ly,
                       h2d_ms, ker_ms, d2h_ms, total_ms, tiled);
    }

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_grid);
    free(h_tmp);
    return 0;
}
