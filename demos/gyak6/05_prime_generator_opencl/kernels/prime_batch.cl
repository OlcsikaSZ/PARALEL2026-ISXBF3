static ulong addmod_u64(ulong a, ulong b, ulong mod) {
    a %= mod;
    b %= mod;
    if (a >= mod - b) return a - (mod - b);
    return a + b;
}

static ulong mulmod_u64(ulong a, ulong b, ulong mod) {
    ulong result = 0ul;
    a %= mod;
    while (b > 0ul) {
        if (b & 1ul) result = addmod_u64(result, a, mod);
        a = addmod_u64(a, a, mod);
        b >>= 1ul;
    }
    return result;
}

static ulong powmod_u64(ulong a, ulong e, ulong mod) {
    ulong result = 1ul;
    a %= mod;
    while (e > 0ul) {
        if (e & 1ul) result = mulmod_u64(result, a, mod);
        a = mulmod_u64(a, a, mod);
        e >>= 1ul;
    }
    return result;
}

static int is_prime_mr_u64(ulong n) {
    if (n < 2ul) return 0;
    if ((n % 2ul) == 0ul) return n == 2ul;
    if ((n % 3ul) == 0ul) return n == 3ul;

    const ulong bases[7] = {2ul, 325ul, 9375ul, 28178ul, 450775ul, 9780504ul, 1795265022ul};
    ulong d = n - 1ul;
    uint s = 0u;
    while ((d & 1ul) == 0ul) {
        d >>= 1ul;
        ++s;
    }

    for (int i = 0; i < 7; ++i) {
        ulong a = bases[i] % n;
        if (a == 0ul) continue;
        ulong x = powmod_u64(a, d, n);
        if (x == 1ul || x == n - 1ul) continue;
        int witness = 1;
        for (uint r = 1u; r < s; ++r) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1ul) {
                witness = 0;
                break;
            }
        }
        if (witness) return 0;
    }
    return 1;
}

__kernel void prime_test_batch(__global const ulong* candidates,
                               __global uint* out_results) {
    size_t gid = get_global_id(0);
    out_results[gid] = (uint)is_prime_mr_u64(candidates[gid]);
}
