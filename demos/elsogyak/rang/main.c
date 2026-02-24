#include "kernel_loader.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 20  // Teszteléshez kisebb méret

int main()
{
    int* arr = (int*)malloc(SIZE*sizeof(int));
    int* rank_cpu = (int*)malloc(SIZE*sizeof(int));
    int* rank_gpu = (int*)malloc(SIZE*sizeof(int));

    // Feltöltés random adatokkal
    srand((unsigned int)time(NULL));
    for (int i=0; i<SIZE; i++) arr[i] = rand()%100;

    // ==================== KIÍRÁS: bemeneti tömb ====================
    printf("Bemeneti tomb:\n");
    for(int i=0;i<SIZE;i++)
        printf("%d ", arr[i]);
    printf("\n\n");

    // ==================== CPU rang számítás ====================
    clock_t start_cpu = clock();
    for (int i = 0; i < SIZE; i++)
    {
        int r=0;
        for (int j=0; j<SIZE; j++)
            if(arr[j]<arr[i]) r++;
        rank_cpu[i]=r;
    }
    clock_t end_cpu = clock();
    printf("CPU ido: %f\n", (double)(end_cpu-start_cpu)/CLOCKS_PER_SEC);

    printf("CPU rangok:\n");
    for(int i=0;i<SIZE;i++)
        printf("%d ", rank_cpu[i]);
    printf("\n\n");

    // ==================== GPU rang számítás ====================
    cl_device_id device;
    cl_context context = create_context(&device);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    cl_program program = load_program(context, device, "kernels/sample.cl");
    cl_kernel kernel = create_kernel(program, "compute_rank");

    // Buffer létrehozása
    cl_mem d_arr  = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE*sizeof(int), arr, NULL);
    cl_mem d_rank = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SIZE*sizeof(int), rank_gpu, NULL);

    // Kernel argumentumok
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_arr);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_rank);
    unsigned int n = SIZE;
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);

    size_t globalSize = SIZE;
    clock_t start_gpu = clock();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clFinish(queue);
    clock_t end_gpu = clock();

    clEnqueueReadBuffer(queue, d_rank, CL_TRUE, 0, SIZE*sizeof(int), rank_gpu, 0, NULL, NULL);

    printf("GPU ido: %f\n", (double)(end_gpu-start_gpu)/CLOCKS_PER_SEC);

    printf("GPU rangok:\n");
    for(int i=0;i<SIZE;i++)
        printf("%d ", rank_gpu[i]);
    printf("\n\n");

    // ==================== Ellenőrzés ====================
    int ok=1;
    for(int i=0;i<SIZE;i++)
        if(rank_cpu[i]!=rank_gpu[i]) ok=0;
    printf("Eredmeny helyes? %s\n", ok?"Igen":"Nem");

    // ==================== Takarítás ====================
    clReleaseMemObject(d_arr);
    clReleaseMemObject(d_rank);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(arr);
    free(rank_cpu);
    free(rank_gpu);
}