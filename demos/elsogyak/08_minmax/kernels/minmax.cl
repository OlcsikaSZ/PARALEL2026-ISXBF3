__kernel void minmax_group(__global const int* in,
                           const uint n,
                           __global int* out_mins,
                           __global int* out_maxs,
                           __local int* lmins,
                           __local int* lmaxs)
{
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const uint grp = get_group_id(0);

    // Initialize with first element this thread touches (if any), else neutral extremes
    int mn = 2147483647;
    int mx = 0;

    for (uint i = gid; i < n; i += gsz) {
        int v = in[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }

    lmins[lid] = mn;
    lmaxs[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);

    // parallel reduction in local memory
    for (uint stride = lsz >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            int o_mn = lmins[lid + stride];
            int o_mx = lmaxs[lid + stride];
            if (o_mn < lmins[lid]) lmins[lid] = o_mn;
            if (o_mx > lmaxs[lid]) lmaxs[lid] = o_mx;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out_mins[grp] = lmins[0];
        out_maxs[grp] = lmaxs[0];
    }
}
