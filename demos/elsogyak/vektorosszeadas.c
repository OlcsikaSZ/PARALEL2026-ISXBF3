#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#define TOLERANCE 1e-6
#define VECTOR_SIZE 1024

// =======================
// OpenCL kernel forrás
// =======================
const char* kernelSource =
"__kernel void vector_add(                     \n"
"   __global const float* A,                   \n"
"   __global const float* B,                   \n"
"   __global float* C,                         \n"
"   const unsigned int n)                      \n"
"{                                             \n"
"   int i = get_global_id(0);                  \n"
"   if (i < n)                                 \n"
"       C[i] = A[i] + B[i];                    \n"
"}                                             \n";


// =====================================================
// CPU szekvenciális verzió (ellenőrzéshez)
// =====================================================
void add_vectors_cpu(const float* A,
                     const float* B,
                     float* C,
                     size_t n)
{
    for (size_t i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}


// =====================================================
// OpenCL-es verzió (nem látszik kívülről!)
// =====================================================
void add_vectors(const float* A,
                 const float* B,
                 float* C,
                 size_t n)
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferC;

    cl_int err;

    // Platform
    err = clGetPlatformIDs(1, &platform, NULL);

    // Device (GPU, ha nincs, akkor CPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS)
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    // Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Command queue
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Memória buffer
    bufferA = clCreateBuffer(context,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             n * sizeof(float),
                             (void*)A, &err);

    bufferB = clCreateBuffer(context,
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             n * sizeof(float),
                             (void*)B, &err);

    bufferC = clCreateBuffer(context,
                             CL_MEM_WRITE_ONLY,
                             n * sizeof(float),
                             NULL, &err);

    // Program
    program = clCreateProgramWithSource(context,
                                        1,
                                        &kernelSource,
                                        NULL,
                                        &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Kernel
    kernel = clCreateKernel(program, "vector_add", &err);

    // Kernel argumentumok
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // Kernel futtatás
    size_t globalSize = n;
    err = clEnqueueNDRangeKernel(queue,
                                 kernel,
                                 1,
                                 NULL,
                                 &globalSize,
                                 NULL,
                                 0,
                                 NULL,
                                 NULL);

    clFinish(queue);

    // Eredmény visszaolvasás
    clEnqueueReadBuffer(queue,
                        bufferC,
                        CL_TRUE,
                        0,
                        n * sizeof(float),
                        C,
                        0,
                        NULL,
                        NULL);

    // Felszabadítás
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


// =====================================================
// Eredmény ellenőrzés
// =====================================================
int check_result(const float* C1,
                 const float* C2,
                 size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (fabs(C1[i] - C2[i]) > TOLERANCE)
            return 0;
    }
    return 1;
}


// =====================================================
// MAIN
// =====================================================
int main()
{
    size_t n = VECTOR_SIZE;

    float* A = (float*)malloc(n * sizeof(float));
    float* B = (float*)malloc(n * sizeof(float));
    float* C_gpu = (float*)malloc(n * sizeof(float));
    float* C_cpu = (float*)malloc(n * sizeof(float));

    // Inicializálás
    for (size_t i = 0; i < n; i++)
    {
        A[i] = (float)i;
        B[i] = (float)(2 * i);
    }

    // GPU verzió
    add_vectors(A, B, C_gpu, n);

    // CPU verzió
    add_vectors_cpu(A, B, C_cpu, n);

    // Ellenőrzés
    if (check_result(C_gpu, C_cpu, n))
        printf("Helyes eredmény!\n");
    else
        printf("HIBA az eredményben!\n");

    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);

    return 0;
}