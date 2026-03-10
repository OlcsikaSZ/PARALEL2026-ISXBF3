#define HIST_SIZE 101

__kernel void histogram_global_atomic(__global const int* input,
                                      __global unsigned int* histogram,
                                      const int n)
{
    const int gid = get_global_id(0);
    if (gid < n) {
        const int value = input[gid];
        atom_inc(&histogram[value]);
    }
}

__kernel void histogram_local_atomic(__global const int* input,
                                     __global unsigned int* histogram,
                                     __local unsigned int* local_hist,
                                     const int n)
{
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int local_size = get_local_size(0);

    for (int i = lid; i < HIST_SIZE; i += local_size) {
        local_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        const int value = input[gid];
        atom_inc(&local_hist[value]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < HIST_SIZE; i += local_size) {
        if (local_hist[i] > 0) {
            atom_add(&histogram[i], local_hist[i]);
        }
    }
}