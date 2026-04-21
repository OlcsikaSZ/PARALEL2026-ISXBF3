#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

typedef enum RunMode {
    MODE_GPU = 0,
    MODE_CPU_SEQ = 1
} RunMode;

// Return the current time in milliseconds using a high-resolution timer.
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
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

// Abort the program when an OpenCL call fails.
static void die_cl(const char* where, cl_int err) {
    fprintf(stderr, "[OpenCL ERROR] %s failed, code=%d\n", where, err);
    exit(1);
}

// Select the first available GPU, otherwise fall back to a CPU device.
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

    // Try to find a GPU device first.
    for (cl_uint p = 0; p < n_platforms && !chosen; ++p) {
        cl_uint n_dev = 0;
        err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &n_dev);
        if (err == CL_SUCCESS && n_dev > 0) {
            err = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &chosen, NULL);
            if (err == CL_SUCCESS) chosen_plat = plats[p];
        }
    }

    // Fall back to a CPU device if no GPU was found.
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

// Print basic information about the selected OpenCL device.
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

// Check whether the given file already exists.
static int file_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

// Convert the selected run mode to the corresponding CSV label.
static const char* mode_to_csv_name(RunMode mode, int tiled) {
    if (mode == MODE_CPU_SEQ) return "cpu_seq";
    return tiled ? "gpu_tiled" : "gpu_naive";
}

// Append one benchmark result row to a CSV file.
static void append_csv_row(const char* out_path,
                           RunMode mode,
                           int rows, int cols, int iters, int wrap,
                           size_t lx, size_t ly,
                           double h2d_ms, double kernel_ms, double d2h_ms,
                           double total_ms, double wall_total_ms,
                           int tiled)
{
    int exists = file_exists(out_path);
    FILE* f = fopen(out_path, "a");
    if (!f) {
        fprintf(stderr, "Could not open output file: %s\n", out_path);
        return;
    }

    // Write the CSV header when the file is created for the first time.
    if (!exists) {
        fprintf(f, "mode,rows,cols,iters,wrap,lx,ly,h2d_ms,kernel_ms,d2h_ms,total_ms,wall_total_ms,tiled\n");
    }

    fprintf(f, "%s,%d,%d,%d,%d,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            mode_to_csv_name(mode, tiled),
            rows, cols, iters, wrap,
            (unsigned)lx, (unsigned)ly,
            h2d_ms, kernel_ms, d2h_ms, total_ms, wall_total_ms,
            tiled);

    fclose(f);
}

// Round a value up to the next valid multiple.
static size_t round_up(size_t value, size_t multiple) {
    if (multiple == 0) return value;
    size_t rem = value % multiple;
    return rem == 0 ? value : value + (multiple - rem);
}

// Wrap a coordinate into the valid grid range.
static int wrap_coord_cpu(int v, int maxv) {
    int r = v % maxv;
    return (r < 0) ? (r + maxv) : r;
}

// Read one cell from the CPU grid with optional wrap-around.
static unsigned char read_cell_cpu(const unsigned char* grid,
                                   int x,
                                   int y,
                                   int rows,
                                   int cols,
                                   int wrap)
{
    if (wrap) {
        x = wrap_coord_cpu(x, rows);
        y = wrap_coord_cpu(y, cols);
        return grid[(size_t)x * (size_t)cols + (size_t)y];
    }
    if (x < 0 || x >= rows || y < 0 || y >= cols) return 0;
    return grid[(size_t)x * (size_t)cols + (size_t)y];
}

// Compute one CPU reference step of the Game of Life.
static void gol_cpu_step(const unsigned char* in,
                         unsigned char* out,
                         int rows,
                         int cols,
                         int wrap)
{
    for (int x = 0; x < rows; ++x) {
        for (int y = 0; y < cols; ++y) {
            int sum = 0;
            // Sum the eight neighboring cells around the current cell.
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    sum += (int)read_cell_cpu(in, x + dx, y + dy, rows, cols, wrap);
                }
            }

            const size_t idx = (size_t)x * (size_t)cols + (size_t)y;
            const unsigned char alive = in[idx];
            // Apply the standard Conway birth and survival rules.
            if (alive) out[idx] = (sum == 2 || sum == 3) ? 1u : 0u;
            else       out[idx] = (sum == 3) ? 1u : 0u;
        }
    }
}

// Run the sequential CPU reference implementation and measure its wall-clock time.
static void run_cpu_seq(const unsigned char* initial,
                        unsigned char* result,
                        int rows,
                        int cols,
                        int iters,
                        int wrap,
                        int repeat,
                        int warmup,
                        double* avg_wall_total_ms)
{
    const size_t n = (size_t)rows * (size_t)cols;
    unsigned char* cpu_a = (unsigned char*)malloc(n);
    unsigned char* cpu_b = (unsigned char*)malloc(n);
    if (!cpu_a || !cpu_b) {
        fprintf(stderr, "CPU benchmark allocation failed.\n");
        free(cpu_a);
        free(cpu_b);
        exit(1);
    }

    double wall_sum = 0.0;

    for (int run = 0; run < warmup + repeat; ++run) {
        memcpy(cpu_a, initial, n);
        double start_ms = now_ms();

        for (int t = 0; t < iters; ++t) {
            gol_cpu_step(cpu_a, cpu_b, rows, cols, wrap);
            unsigned char* tmp = cpu_a;
            cpu_a = cpu_b;
            cpu_b = tmp;
        }

        double elapsed_ms = now_ms() - start_ms;
        if (run >= warmup) wall_sum += elapsed_ms;
    }

    memcpy(result, cpu_a, n);
    *avg_wall_total_ms = wall_sum / (double)repeat;

    free(cpu_a);
    free(cpu_b);
}

// Compare the GPU result against the CPU reference implementation.
static int validate_against_cpu(const unsigned char* initial,
                                const unsigned char* gpu_result,
                                int rows,
                                int cols,
                                int iters,
                                int wrap)
{
    const size_t n = (size_t)rows * (size_t)cols;
    unsigned char* cpu_a = (unsigned char*)malloc(n);
    unsigned char* cpu_b = (unsigned char*)malloc(n);
    if (!cpu_a || !cpu_b) {
        fprintf(stderr, "CPU validation allocation failed.\n");
        free(cpu_a);
        free(cpu_b);
        return -1;
    }

    // Copy the initial grid before running the CPU simulation.
    memcpy(cpu_a, initial, n);
    // Run the same number of iterations on the CPU for validation.
    for (int t = 0; t < iters; ++t) {
        gol_cpu_step(cpu_a, cpu_b, rows, cols, wrap);
        unsigned char* tmp = cpu_a;
        cpu_a = cpu_b;
        cpu_b = tmp;
    }

    int mismatch_index = -1;
    // Search for the first mismatch between CPU and GPU results.
    for (size_t i = 0; i < n; ++i) {
        if (cpu_a[i] != gpu_result[i]) {
            mismatch_index = (int)i;
            break;
        }
    }

    if (mismatch_index >= 0) {
        int mx = mismatch_index / cols;
        int my = mismatch_index % cols;
        fprintf(stderr,
                "Validation FAILED at cell (%d,%d): CPU=%u GPU=%u\n",
                mx, my,
                (unsigned)cpu_a[mismatch_index],
                (unsigned)gpu_result[mismatch_index]);
        free(cpu_a);
        free(cpu_b);
        return 0;
    }

    free(cpu_a);
    free(cpu_b);
    return 1;
}

// Print the command-line usage and default parameter values.
static void usage(const char* argv0) {
    printf("Usage: %s [--rows N] [--cols N] [--iters N] [--seed N] [--wrap 0|1] [--mode gpu|cpu_seq] [--tiled 0|1] [--lx N] [--ly N] [--validate 0|1] [--csv] [--out FILE] [--repeat N] [--warmup N]\n", argv0);
    printf("Defaults: rows=1024 cols=1024 iters=500 seed=time wrap=0 mode=gpu tiled=0 lx=16 ly=16 validate=0 repeat=1 warmup=0\n");
}

int main(int argc, char** argv) {
    int rows = 1024;
    int cols = 1024;
    int iters = 500;
    unsigned int seed = (unsigned int)time(NULL);
    int wrap = 0;
    int tiled = 0;
    int validate = 0;
    int csv = 0;
    int lx_arg = 16;
    int ly_arg = 16;
    const char* out_path = NULL;
    int repeat = 1;
    int warmup = 0;
    RunMode mode = MODE_GPU;

    // Parse command-line arguments and override defaults.
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--wrap") && i + 1 < argc) wrap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tiled") && i + 1 < argc) tiled = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--validate") && i + 1 < argc) validate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lx") && i + 1 < argc) lx_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ly") && i + 1 < argc) ly_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* mode_arg = argv[++i];
            if (!strcmp(mode_arg, "gpu")) mode = MODE_GPU;
            else if (!strcmp(mode_arg, "cpu_seq")) mode = MODE_CPU_SEQ;
            else {
                fprintf(stderr, "Unknown mode: %s\n", mode_arg);
                usage(argv[0]);
                return 1;
            }
        }
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

    // Validate the main numeric input parameters.
    if (rows <= 0 || cols <= 0 || iters <= 0) {
        fprintf(stderr, "rows/cols/iters must be > 0\n");
        return 1;
    }

    // Validate the requested local work-group size.
    if (lx_arg <= 0 || ly_arg <= 0) {
        fprintf(stderr, "lx/ly must be > 0\n");
        return 1;
    }

    // Validate the repeat and warmup configuration.
    if (repeat <= 0 || warmup < 0) {
        fprintf(stderr, "repeat must be > 0 and warmup must be >= 0\n");
        return 1;
    }

    // Reject tiled execution when the program is running in CPU sequential mode.
    if (mode == MODE_CPU_SEQ && tiled) {
        fprintf(stderr, "CPU sequential mode does not use the tiled kernel flag.\n");
        return 1;
    }

    const size_t n = (size_t)rows * (size_t)cols;
    unsigned char* h_grid = (unsigned char*)malloc(n);
    unsigned char* h_tmp  = (unsigned char*)malloc(n);

    // Allocate host buffers for the input and output grids.
    if (!h_grid || !h_tmp) {
        fprintf(stderr, "Host allocation failed (n=%u)\n", (unsigned)n);
        free(h_grid);
        free(h_tmp);
        return 1;
    }

    // Fill the initial grid with random 0/1 cell states.
    srand(seed);
    for (size_t k = 0; k < n; ++k) {
        h_grid[k] = (unsigned char)(rand() & 1);
    }

    // Execute the sequential CPU benchmark path and optionally write its result to CSV.
    if (mode == MODE_CPU_SEQ) {
        double cpu_wall_total_ms = 0.0;
        run_cpu_seq(h_grid, h_tmp, rows, cols, iters, wrap, repeat, warmup, &cpu_wall_total_ms);

        printf("Mode: cpu_seq\n");
        printf("Execution device: Host CPU (sequential reference, single-threaded)\n");
        printf("Rows x Cols: %d x %d\n", rows, cols);
        printf("Iterations: %d\n", iters);
        printf("Wrap: %d\n", wrap);
        printf("Repeat / Warmup: %d / %d\n", repeat, warmup);
        printf("CPU sequential total wall time: %.3f ms\n", cpu_wall_total_ms);
        printf("CPU sequential time per iteration: %.6f ms\n", cpu_wall_total_ms / (double)iters);

        if (csv && out_path) {
            append_csv_row(out_path, mode, rows, cols, iters, wrap, 1u, 1u,
                           0.0, cpu_wall_total_ms, 0.0,
                           cpu_wall_total_ms, cpu_wall_total_ms, 0);
        }

        free(h_grid);
        free(h_tmp);
        return 0;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device = pick_device(&platform);
    print_device_info(device);

    size_t max_wg = 0;
    size_t max_wi[3] = {0, 0, 0};

    // Query device limits for valid work-group dimensions.
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wi), max_wi, NULL);

    // Reject unsupported local sizes before launching the kernel.
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
    size_t gx = round_up((size_t)rows, lx);
    size_t gy = round_up((size_t)cols, ly);

    size_t global[2] = { gx, gy };
    size_t local[2]  = { lx, ly };

    // Create an OpenCL context for the selected device.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) die_cl("clCreateContext", err);

    // Create a profiling-enabled command queue for timing measurements.
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device,
        (cl_queue_properties[]){ CL_QUEUE_PROPERTIES, (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE, 0 },
        &err
    );
    if (!queue || err != CL_SUCCESS) die_cl("clCreateCommandQueueWithProperties", err);

    int loader_err = 0;
    const char* kernel_path = tiled ? "kernels/gol_tiled.cl" : "kernels/gol_naive.cl";
    const char* kernel_name = tiled ? "gol_step_tiled" : "gol_step";

    // Choose and load the requested kernel source file.
    const char* src = load_kernel_source(kernel_path, &loader_err);
    if (loader_err != 0 || !src) {
        fprintf(stderr, "Kernel source load failed. Did you run from project root?\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(h_grid);
        free(h_tmp);
        return 1;
    }

    // Create an OpenCL program object from the kernel source.
    cl_program program = clCreateProgramWithSource(context, 1, &src, NULL, &err);
    if (!program || err != CL_SUCCESS) die_cl("clCreateProgramWithSource", err);

    // Build the program and print the compiler log on failure.
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

    // Create the kernel object used for one simulation step.
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (!kernel || err != CL_SUCCESS) die_cl("clCreateKernel", err);

    // Allocate the current state buffer on the device.
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(cl_uchar), NULL, &err);
    if (!d_a || err != CL_SUCCESS) die_cl("clCreateBuffer(d_a)", err);

    // Allocate the next state buffer on the device.
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(cl_uchar), NULL, &err);
    if (!d_b || err != CL_SUCCESS) die_cl("clCreateBuffer(d_b)", err);

    double sum_h2d_ms = 0.0;
    double sum_kernel_ms = 0.0;
    double sum_d2h_ms = 0.0;
    double sum_total_ms = 0.0;
    double sum_wall_total_ms = 0.0;

    // Execute warmup and repeated benchmark runs.
    for (int run = 0; run < warmup + repeat; ++run) {
        cl_ulong h2d_ns = 0, kernel_ns = 0, d2h_ns = 0;
        double wall_start_ms = now_ms();

        cl_mem cur = d_a;
        cl_mem next = d_b;

        cl_event ev_h2d;
        // Copy the initial grid from host memory to the device.
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

        // Run the requested number of Game of Life iterations on the GPU.
        for (int t = 0; t < iters; ++t) {
            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cur);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &next);
            err |= clSetKernelArg(kernel, 2, sizeof(int), &rows);
            err |= clSetKernelArg(kernel, 3, sizeof(int), &cols);
            err |= clSetKernelArg(kernel, 4, sizeof(int), &wrap);

            // Allocate local tile memory for the tiled kernel version.
            if (tiled) {
                size_t tile_bytes = (lx + 2) * (ly + 2) * sizeof(unsigned char);
                err |= clSetKernelArg(kernel, 5, tile_bytes, NULL);
            }

            if (err != CL_SUCCESS) die_cl("clSetKernelArg", err);

            cl_event ev_k;
            // Launch one kernel execution over the padded global grid.
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

            // Swap device buffers so the next step reads the new state.
            cl_mem tmp = cur;
            cur = next;
            next = tmp;
        }

        cl_event ev_d2h;
        // Copy the final grid back from the device to the host.
        err = clEnqueueReadBuffer(queue, cur, CL_TRUE, 0, n * sizeof(cl_uchar), h_tmp, 0, NULL, &ev_d2h);
        if (err != CL_SUCCESS) die_cl("clEnqueueReadBuffer", err);

        {
            cl_ulong s = 0, e = 0;
            clGetEventProfilingInfo(ev_d2h, CL_PROFILING_COMMAND_START, sizeof(s), &s, NULL);
            clGetEventProfilingInfo(ev_d2h, CL_PROFILING_COMMAND_END, sizeof(e), &e, NULL);
            d2h_ns += (e - s);
            clReleaseEvent(ev_d2h);
        }

        err = clFinish(queue);
        if (err != CL_SUCCESS) die_cl("clFinish", err);

        double wall_total_ms = now_ms() - wall_start_ms;
        double h2d_ms = (double)h2d_ns / 1e6;
        double ker_ms = (double)kernel_ns / 1e6;
        double d2h_ms = (double)d2h_ns / 1e6;
        double total_ms = h2d_ms + ker_ms + d2h_ms;

        // Accumulate only the measured runs after warmup.
        if (run >= warmup) {
            sum_h2d_ms += h2d_ms;
            sum_kernel_ms += ker_ms;
            sum_d2h_ms += d2h_ms;
            sum_total_ms += total_ms;
            sum_wall_total_ms += wall_total_ms;
        }
    }

    // Compute the average timing values over all measured runs.
    double h2d_ms = sum_h2d_ms / (double)repeat;
    double ker_ms = sum_kernel_ms / (double)repeat;
    double d2h_ms = sum_d2h_ms / (double)repeat;
    double total_ms = sum_total_ms / (double)repeat;
    double wall_total_ms = sum_wall_total_ms / (double)repeat;

    // Validate the GPU output against the CPU reference if requested.
    if (validate) {
        int validation_ok = validate_against_cpu(h_grid, h_tmp, rows, cols, iters, wrap);
        if (validation_ok < 0) {
            fprintf(stderr, "Validation could not be completed due to allocation failure.\n");
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            free(h_grid);
            free(h_tmp);
            return 2;
        }
        if (validation_ok == 0) {
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_b);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            free(h_grid);
            free(h_tmp);
            return 2;
        }
        printf("Validation OK (CPU reference matched GPU result).\n");
    }

    // Print the result either as CSV or as a readable report.
    printf("Mode: %s\n", mode_to_csv_name(mode, tiled));
    printf("Rows x Cols: %d x %d\n", rows, cols);
    printf("Iterations: %d\n", iters);
    printf("Wrap: %d\n", wrap);
    printf("Tiled: %d\n", tiled);
    printf("Local size: %u x %u\n", (unsigned)lx, (unsigned)ly);
    printf("Repeat / Warmup: %d / %d\n", repeat, warmup);
    printf("Host->Device: %.3f ms\n", h2d_ms);
    printf("Kernel total: %.3f ms\n", ker_ms);
    printf("Device->Host: %.3f ms\n", d2h_ms);
    printf("Profiled GPU total: %.3f ms\n", total_ms);
    printf("Wall total: %.3f ms\n", wall_total_ms);
    printf("Kernel per iteration: %.6f ms\n", ker_ms / (double)iters);

    // Save the measured result row to a CSV file when requested.
    if (csv && out_path) {
        append_csv_row(out_path, mode, rows, cols, iters, wrap, lx, ly,
                       h2d_ms, ker_ms, d2h_ms, total_ms, wall_total_ms, tiled);
    }

    // Release all allocated OpenCL objects.
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Free the host-side grid buffers.
    free(h_grid);
    free(h_tmp);
    return 0;
}
