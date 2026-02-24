// OpenCL C kernels for prime check & simple sieve.
__kernel void check_single_divisor(ulong n, ulong first, ulong step, __global int* is_prime_flag) {
    // Each work-item checks exactly one divisor d = first + gid*step.
    // We assume n is odd and first is odd (3).
    if (*is_prime_flag == 0) return; // quick exit (not atomic, but OK for speed)
    size_t gid = get_global_id(0);
    ulong d = first + (ulong)gid * step;
    if (d < 3) return;
    // If divisor divides n -> composite
    if (n % d == 0) {
        // Set to 0 (composite). Races are fine: 0 wins.
        *is_prime_flag = 0;
    }
}

__kernel void check_range_divisor(ulong n, ulong r, ulong chunk, __global int* is_prime_flag) {
    // Each work-item checks a range of odd divisors.
    // divisor index k corresponds to d = 3 + 2*k
    if (*is_prime_flag == 0) return;
    size_t gid = get_global_id(0);
    ulong start_k = (ulong)gid * chunk;
    ulong end_k = start_k + chunk;

    // total k values:
    // total = ((r-3)/2)+1
    ulong total = ((r - 3UL) / 2UL) + 1UL;
    if (start_k >= total) return;
    if (end_k > total) end_k = total;

    for (ulong k = start_k; k < end_k; ++k) {
        if (*is_prime_flag == 0) return;
        ulong d = 3UL + 2UL * k;
        if (n % d == 0UL) {
            *is_prime_flag = 0;
            return;
        }
    }
}

__kernel void check_primes_table(ulong n, __global const uint* primes, uint pcount, __global int* is_prime_flag) {
    // Work-items stride through primes list: idx = gid, gid+gsize, ...
    if (*is_prime_flag == 0) return;
    size_t gid = get_global_id(0);
    size_t gsize = get_global_size(0);

    for (uint idx = (uint)gid; idx < pcount; idx += (uint)gsize) {
        if (*is_prime_flag == 0) return;
        uint p = primes[idx];
        if (p < 2u) continue;
        if ((ulong)p == n) continue;
        if (n % (ulong)p == 0UL) {
            *is_prime_flag = 0;
            return;
        }
    }
}

// Simple sieve marking kernel: each work-item corresponds to candidate p.
// flags[i] == 1 means composite, 0 means "prime until proven composite".
// This is not the fastest sieve, but it demonstrates OpenCL prime generation.
__kernel void sieve_mark(__global uchar* flags, uint limit) {
    size_t gid = get_global_id(0);
    uint p = (uint)gid;
    if (p < 2u || p > limit) return;
    // Mark composites for p (skip if already known composite)
    if (flags[p] != 0) return;

    // Avoid overflow: start at p*p
    ulong pp = (ulong)p * (ulong)p;
    if (pp > (ulong)limit) return;

    for (uint k = (uint)pp; k <= limit; k += p) {
        flags[k] = (uchar)1;
    }
}
