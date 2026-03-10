__kernel void reduce_sum_and_sumsq(__global const int* input,
                                   __global ulong* partial_sums,
                                   __global ulong* partial_sum_sqs,
                                   const int n,
                                   __local ulong* local_sums,
                                   __local ulong* local_sum_sqs)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint global_size = get_global_size(0);
    const uint local_size = get_local_size(0);

    ulong sum = 0UL;
    ulong sum_sq = 0UL;

    for (uint i = gid; i < (uint)n; i += global_size) {
        ulong x = (ulong)input[i];
        sum += x;
        sum_sq += x * x;
    }

    local_sums[lid] = sum;
    local_sum_sqs[lid] = sum_sq;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sums[lid] += local_sums[lid + stride];
            local_sum_sqs[lid] += local_sum_sqs[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[group_id] = local_sums[0];
        partial_sum_sqs[group_id] = local_sum_sqs[0];
    }
}