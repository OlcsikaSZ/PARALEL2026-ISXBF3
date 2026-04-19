#define PASSWORD_LEN 6
#define TARGET_COUNT 4

typedef struct {
    uint a;
    uint b;
    uint c;
    uint d;
} md5_digest_t;

__constant uint md5_r[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

__constant uint md5_k[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu,
    0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
    0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
    0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
    0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
    0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u,
    0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u,
    0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
    0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u
};

static uint leftrotate32(uint x, uint c) {
    return rotate(x, c);
}

static md5_digest_t md5_6chars_private(const char password[PASSWORD_LEN]) {
    uchar block[64] = {0};
    uint w[16];
    for (int i = 0; i < PASSWORD_LEN; ++i) {
        block[i] = (uchar)password[i];
    }
    block[PASSWORD_LEN] = (uchar)0x80;
    ulong bit_len = (ulong)PASSWORD_LEN * 8ul;
    for (int i = 0; i < 8; ++i) {
        block[56 + i] = (uchar)((bit_len >> (8ul * (ulong)i)) & 0xfful);
    }
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint)block[i * 4 + 0]
             | ((uint)block[i * 4 + 1] << 8)
             | ((uint)block[i * 4 + 2] << 16)
             | ((uint)block[i * 4 + 3] << 24);
    }

    uint a0 = 0x67452301u;
    uint b0 = 0xefcdab89u;
    uint c0 = 0x98badcfeu;
    uint d0 = 0x10325476u;
    uint a = a0;
    uint b = b0;
    uint c = c0;
    uint d = d0;

    for (int i = 0; i < 64; ++i) {
        uint f;
        uint g;
        if (i < 16) {
            f = (b & c) | ((~b) & d);
            g = (uint)i;
        } else if (i < 32) {
            f = (d & b) | ((~d) & c);
            g = (uint)((5 * i + 1) & 15);
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (uint)((3 * i + 5) & 15);
        } else {
            f = c ^ (b | (~d));
            g = (uint)((7 * i) & 15);
        }
        uint temp = d;
        d = c;
        c = b;
        b = b + leftrotate32(a + f + md5_k[i] + w[g], md5_r[i]);
        a = temp;
    }

    md5_digest_t out;
    out.a = a0 + a;
    out.b = b0 + b;
    out.c = c0 + c;
    out.d = d0 + d;
    return out;
}

static void index_to_password(ulong index, __global const char* charset, int charset_len, char out[PASSWORD_LEN]) {
    for (int i = PASSWORD_LEN - 1; i >= 0; --i) {
        out[i] = charset[index % (ulong)charset_len];
        index /= (ulong)charset_len;
    }
}

static int digest_equal(md5_digest_t a, md5_digest_t b) {
    return a.a == b.a && a.b == b.b && a.c == b.c && a.d == b.d;
}

__kernel void md5_crack(__global const char* charset,
                        int charset_len,
                        ulong start_offset,
                        __global const md5_digest_t* targets,
                        __global int* found_flags,
                        __global char* found_passwords) {
    ulong gid = get_global_id(0);
    ulong idx = start_offset + gid;
    char password[PASSWORD_LEN];
    index_to_password(idx, charset, charset_len, password);
    md5_digest_t got = md5_6chars_private(password);

    for (int t = 0; t < TARGET_COUNT; ++t) {
        if (digest_equal(got, targets[t])) {
            if (atomic_cmpxchg(&found_flags[t], 0, 1) == 0) {
                int base = t * (PASSWORD_LEN + 1);
                for (int i = 0; i < PASSWORD_LEN; ++i) {
                    found_passwords[base + i] = password[i];
                }
                found_passwords[base + PASSWORD_LEN] = '\0';
            }
        }
    }
}
