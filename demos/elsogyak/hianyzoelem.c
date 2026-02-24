#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1000000
#define MISSING_COUNT 100000

// =====================================================
// OpenCL kernel
// =====================================================
const char* kernelSource =
"__kernel void fill_missing(                \n"
"   __global int* arr,                      \n"
"   const unsigned int n)                   \n"
"{                                          \n"
"   int i = get_global_id(0);               \n"
"                                           \n"
"   if (i > 0 && i < n-1)                   \n"
"   {                                       \n"
"       if (arr[i] == -1)                   \n"
"           arr[i] = (arr[i-1] + arr[i+1]) / 2; \n"
"   }                                       \n"
"}                                          \n";


// =====================================================
// Bemenet generálása
// =====================================================
void generate_input(int* arr, size_t n, int missing_count)
{
    srand((unsigned int)time(NULL));

    for (size_t i = 0; i < n; i++)
        arr[i] = rand() % 100;

    // egyenletes elosztás
    size_t step = n / missing_count;

    for (int i = 0; i < missing_count; i++)
    {
        size_t idx = 1 + i * step;

        if (idx < n - 1)
            arr[idx] = -1;
    }
}


// =====================================================
// CPU verzió
// =====================================================
void fill_missing_cpu(int* arr, size_t n)
{
    for (size_t i = 1; i < n - 1; i++)
    {
        if (arr[i] == -1)
            arr[i] = (arr[i - 1] + arr[i + 1]) / 2;
    }
}


// =====================================================
// OpenCL verzió
// =====================================================
void fill_missing(int* arr, size_t n)
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;

    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS)
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    buffer = clCreateBuffer(context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            n * sizeof(int),
                            arr,
                            &err);

    program = clCreateProgramWithSource(context,
                                        1,
                                        &kernelSource,
                                        NULL,
                                        &err);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "fill_missing", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(unsigned int), &n);

    size_t globalSize = n;

    clEnqueueNDRangeKernel(queue,
                           kernel,
                           1,
                           NULL,
                           &globalSize,
                           NULL,
                           0,
                           NULL,
                           NULL);

    clFinish(queue);

    clEnqueueReadBuffer(queue,
                        buffer,
                        CL_TRUE,
                        0,
                        n * sizeof(int),
                        arr,
                        0,
                        NULL,
                        NULL);

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


// =====================================================
// Ellenőrzés
// =====================================================
int check_result(int* a, int* b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i])
            return 0;
    return 1;
}


// =====================================================
// MAIN
// =====================================================
int main()
{
    int* arr_gpu = (int*)malloc(SIZE * sizeof(int));
    int* arr_cpu = (int*)malloc(SIZE * sizeof(int));

    generate_input(arr_gpu, SIZE, MISSING_COUNT);

    for (size_t i = 0; i < SIZE; i++)
        arr_cpu[i] = arr_gpu[i];

    // ================= GPU mérés =================
    clock_t start_gpu = clock();
    fill_missing(arr_gpu, SIZE);
    clock_t end_gpu = clock();
    double gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    // ================= CPU mérés =================
    clock_t start_cpu = clock();
    fill_missing_cpu(arr_cpu, SIZE);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // ================= Ellenőrzés =================
    if (check_result(arr_gpu, arr_cpu, SIZE))
        printf("Helyes eredmeny!\n");
    else
        printf("HIBA az eredmenyben!\n");

    printf("CPU ido: %f masodperc\n", cpu_time);
    printf("GPU ido: %f masodperc\n", gpu_time);

    free(arr_gpu);
    free(arr_cpu);

    return 0;
}