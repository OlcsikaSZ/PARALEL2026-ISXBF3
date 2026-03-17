__kernel void count_histogram_atomic(
    __global const uint* input,
    __global uint* histogram,
    const uint n
) {
    uint gid = get_global_id(0);
    if (gid < n) {
        uint val = input[gid];
        atomic_inc(&histogram[val]);
    }
}

__kernel void fill_output_from_prefix(
    __global const uint* prefix,
    __global uint* output,
    const uint max_value_plus_one
) {
    uint v = get_global_id(0);

    if (v < max_value_plus_one) {
        uint start = prefix[v];
        uint end   = prefix[v + 1];

        for (uint i = start; i < end; i++) {
            output[i] = v;
        }
    }
}