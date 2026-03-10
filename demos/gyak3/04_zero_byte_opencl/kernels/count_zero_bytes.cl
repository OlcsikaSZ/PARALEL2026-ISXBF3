__kernel void count_zero_bytes(__global const uchar *data,
                               const ulong n,
                               __global uint *partial_sums,
                               __local uint *local_sums) {
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t gsize = get_global_size(0);
    size_t group = get_group_id(0);

    uint local_count = 0;

    for (ulong i = gid; i < n; i += gsize) {
        if (data[i] == (uchar)0) {
            local_count++;
        }
    }

    local_sums[lid] = local_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[group] = local_sums[0];
    }
}