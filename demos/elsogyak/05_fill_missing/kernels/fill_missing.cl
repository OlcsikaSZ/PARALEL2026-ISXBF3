__kernel void fill_missing(__global const int* in,
                           __global int* out,
                           const uint n,
                           const int missing_value)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    int v = in[i];

    /* Keep ends unchanged; assumption says missing never occurs at ends. */
    if (v == missing_value && i > 0 && (i + 1) < n) {
        int left  = in[i - 1];
        int right = in[i + 1];
        out[i] = (left + right) / 2; /* integer average */
    } else {
        out[i] = v;
    }
}
