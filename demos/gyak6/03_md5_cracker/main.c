#include "include/kernel_loader.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PASSWORD_LEN 6
#define TARGET_COUNT 4
#define MAX_FOUND 4
#define DEFAULT_CPU_LIMIT 2000000ULL
#define DEFAULT_GPU_GLOBAL 1048576ULL
#define DEFAULT_GPU_CHUNK_PASSES 32ULL

static const char* DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

static const char* TARGET_HASHES[TARGET_COUNT] = {
    "af5b3d17aa1e2ff2a0f83142d692d701",
    "3e918e9c9f594bda6b0cf358391c3b1a",
    "a3d11119e7c6af230e2dac2474ef2466",
    "bf4ab447496f2d3d5a6c77c2cd12f996"
};

static const char* TARGET_NAMES[TARGET_COUNT] = { "user1", "user2", "user3", "user4" };

typedef struct {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
} md5_digest_t;

static const uint32_t md5_r[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

static const uint32_t md5_k[64] = {
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

static uint32_t leftrotate32(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32u - c));
}

static int hex_to_bytes16(const char* hex, unsigned char out[16]) {
    for (int i = 0; i < 16; ++i) {
        unsigned int value = 0;
        if (sscanf(hex + i * 2, "%2x", &value) != 1) {
            return 0;
        }
        out[i] = (unsigned char)value;
    }
    return 1;
}

static md5_digest_t digest_from_hex(const char* hex) {
    unsigned char bytes[16];
    md5_digest_t d = {0, 0, 0, 0};
    if (!hex_to_bytes16(hex, bytes)) {
        return d;
    }
    d.a = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) | ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
    d.b = (uint32_t)bytes[4] | ((uint32_t)bytes[5] << 8u) | ((uint32_t)bytes[6] << 16u) | ((uint32_t)bytes[7] << 24u);
    d.c = (uint32_t)bytes[8] | ((uint32_t)bytes[9] << 8u) | ((uint32_t)bytes[10] << 16u) | ((uint32_t)bytes[11] << 24u);
    d.d = (uint32_t)bytes[12] | ((uint32_t)bytes[13] << 8u) | ((uint32_t)bytes[14] << 16u) | ((uint32_t)bytes[15] << 24u);
    return d;
}

static md5_digest_t md5_6chars(const unsigned char msg[PASSWORD_LEN]) {
    uint32_t w[16] = {0};
    unsigned char block[64] = {0};
    memcpy(block, msg, PASSWORD_LEN);
    block[PASSWORD_LEN] = 0x80u;
    const uint64_t bit_len = (uint64_t)PASSWORD_LEN * 8ULL;
    memcpy(block + 56, &bit_len, sizeof(bit_len));
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t)block[i * 4 + 0]
             | ((uint32_t)block[i * 4 + 1] << 8u)
             | ((uint32_t)block[i * 4 + 2] << 16u)
             | ((uint32_t)block[i * 4 + 3] << 24u);
    }

    uint32_t a0 = 0x67452301u;
    uint32_t b0 = 0xefcdab89u;
    uint32_t c0 = 0x98badcfeu;
    uint32_t d0 = 0x10325476u;

    uint32_t a = a0;
    uint32_t b = b0;
    uint32_t c = c0;
    uint32_t d = d0;

    for (int i = 0; i < 64; ++i) {
        uint32_t f = 0;
        uint32_t g = 0;
        if (i < 16) {
            f = (b & c) | ((~b) & d);
            g = (uint32_t)i;
        } else if (i < 32) {
            f = (d & b) | ((~d) & c);
            g = (uint32_t)((5 * i + 1) & 15);
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (uint32_t)((3 * i + 5) & 15);
        } else {
            f = c ^ (b | (~d));
            g = (uint32_t)((7 * i) & 15);
        }
        uint32_t temp = d;
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

static void index_to_password(uint64_t index, const char* charset, int charset_len, char out[PASSWORD_LEN + 1]) {
    for (int i = PASSWORD_LEN - 1; i >= 0; --i) {
        out[i] = charset[index % (uint64_t)charset_len];
        index /= (uint64_t)charset_len;
    }
    out[PASSWORD_LEN] = '\0';
}

static int digest_equal(md5_digest_t a, md5_digest_t b) {
    return a.a == b.a && a.b == b.b && a.c == b.c && a.d == b.d;
}

static uint64_t ipow_u64(uint64_t base, int exp) {
    uint64_t result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

static cl_device_id pick_device(cl_platform_id* out_platform) {
    cl_uint platform_count = 0;
    clGetPlatformIDs(0, NULL, &platform_count);
    if (platform_count == 0) {
        return NULL;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platform_count);
    clGetPlatformIDs(platform_count, platforms, NULL);

    cl_device_id chosen = NULL;
    for (cl_uint i = 0; i < platform_count && !chosen; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count) == CL_SUCCESS && device_count > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices, NULL);
            chosen = devices[0];
            *out_platform = platforms[i];
            free(devices);
            break;
        }
    }

    for (cl_uint i = 0; i < platform_count && !chosen; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &device_count) == CL_SUCCESS && device_count > 0) {
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * device_count);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, device_count, devices, NULL);
            chosen = devices[0];
            *out_platform = platforms[i];
            free(devices);
            break;
        }
    }

    free(platforms);
    return chosen;
}

static double elapsed_seconds(clock_t start, clock_t end) {
    return (double)(end - start) / (double)CLOCKS_PER_SEC;
}

static void run_cpu(const char* charset, int charset_len, uint64_t limit, const md5_digest_t targets[TARGET_COUNT]) {
    char password[PASSWORD_LEN + 1];
    int found[TARGET_COUNT] = {0, 0, 0, 0};
    clock_t t0 = clock();

    uint64_t tested = 0;
    for (uint64_t idx = 0; idx < limit; ++idx) {
        index_to_password(idx, charset, charset_len, password);
        md5_digest_t got = md5_6chars((const unsigned char*)password);
        ++tested;
        for (int t = 0; t < TARGET_COUNT; ++t) {
            if (!found[t] && digest_equal(got, targets[t])) {
                found[t] = 1;
                printf("[CPU] Talalat: %s -> %s\n", TARGET_NAMES[t], password);
            }
        }
    }

    clock_t t1 = clock();
    double sec = elapsed_seconds(t0, t1);
    printf("[CPU] Vizsgalt jeloltek: %" PRIu64 "\n", tested);
    printf("[CPU] Ido: %.3f s\n", sec);
    if (sec > 0.0) {
        printf("[CPU] Becsult sebesseg: %.2f jelszo/s\n", (double)tested / sec);
    }
}

static void run_gpu(const char* charset, int charset_len, uint64_t total_space, uint64_t global_size, uint64_t passes) {
    cl_platform_id platform = NULL;
    cl_device_id device = pick_device(&platform);
    if (!device) {
        fprintf(stderr, "Nem talalhato OpenCL eszkoz.\n");
        return;
    }

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("[GPU] Eszkoz: %s\n", device_name);

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program program = build_program_from_file(context, device, "kernels/md5_cracker.cl", NULL);
    if (!program) {
        fprintf(stderr, "Kernel build hiba.\n");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return;
    }

    cl_kernel kernel = clCreateKernel(program, "md5_crack", &err);

    md5_digest_t targets[TARGET_COUNT];
    for (int i = 0; i < TARGET_COUNT; ++i) {
        targets[i] = digest_from_hex(TARGET_HASHES[i]);
    }

    cl_mem charset_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        (size_t)charset_len, (void*)charset, &err);
    cl_mem target_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(targets), targets, &err);
    int found_flags[MAX_FOUND] = {0, 0, 0, 0};
    char found_passwords[MAX_FOUND][PASSWORD_LEN + 1];
    memset(found_passwords, 0, sizeof(found_passwords));
    cl_mem found_flag_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sizeof(found_flags), found_flags, &err);
    cl_mem found_pass_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sizeof(found_passwords), found_passwords, &err);

    uint64_t tested = 0;
    uint64_t offset = 0;
    uint64_t max_to_test = global_size * passes;
    if (max_to_test > total_space) {
        max_to_test = total_space;
    }

    cl_event ev;
    clock_t wall0 = clock();
    while (offset < max_to_test) {
        uint64_t current_offset = offset;
        uint64_t chunk = global_size;
        if (current_offset + chunk > max_to_test) {
            chunk = max_to_test - current_offset;
        }
        size_t global = (size_t)chunk;

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &charset_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_int), &charset_len);
        clSetKernelArg(kernel, 2, sizeof(cl_ulong), &current_offset);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &target_buf);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &found_flag_buf);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &found_pass_buf);

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, &ev);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clEnqueueNDRangeKernel hiba: %d\n", err);
            break;
        }
        clWaitForEvents(1, &ev);
        clReleaseEvent(ev);
        tested += chunk;
        offset += chunk;
    }
    clock_t wall1 = clock();

    clEnqueueReadBuffer(queue, found_flag_buf, CL_TRUE, 0, sizeof(found_flags), found_flags, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, found_pass_buf, CL_TRUE, 0, sizeof(found_passwords), found_passwords, 0, NULL, NULL);

    for (int i = 0; i < TARGET_COUNT; ++i) {
        if (found_flags[i]) {
            printf("[GPU] Talalat: %s -> %s\n", TARGET_NAMES[i], found_passwords[i]);
        } else {
            printf("[GPU] Nincs talalat a tesztelt tartomanyban: %s\n", TARGET_NAMES[i]);
        }
    }

    double sec = elapsed_seconds(wall0, wall1);
    printf("[GPU] Vizsgalt jeloltek: %" PRIu64 "\n", tested);
    printf("[GPU] Ido: %.3f s\n", sec);
    if (sec > 0.0) {
        printf("[GPU] Becsult sebesseg: %.2f jelszo/s\n", (double)tested / sec);
    }
    printf("[GPU] Megjegyzes: a teljes jelszoter brutalisan nagy, ezert alapbol csak meresi/teszt tartomanyt futtatunk.\n");

    clReleaseMemObject(found_pass_buf);
    clReleaseMemObject(found_flag_buf);
    clReleaseMemObject(target_buf);
    clReleaseMemObject(charset_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char** argv) {
    const char* charset = DEFAULT_CHARSET;
    const int charset_len = (int)strlen(charset);
    const uint64_t total_space = ipow_u64((uint64_t)charset_len, PASSWORD_LEN);
    uint64_t cpu_limit = DEFAULT_CPU_LIMIT;
    uint64_t gpu_global = DEFAULT_GPU_GLOBAL;
    uint64_t gpu_passes = DEFAULT_GPU_CHUNK_PASSES;

    if (argc > 1) {
        cpu_limit = (uint64_t)strtoull(argv[1], NULL, 10);
    }
    if (argc > 2) {
        gpu_global = (uint64_t)strtoull(argv[2], NULL, 10);
    }
    if (argc > 3) {
        gpu_passes = (uint64_t)strtoull(argv[3], NULL, 10);
    }

    md5_digest_t targets[TARGET_COUNT];
    for (int i = 0; i < TARGET_COUNT; ++i) {
        targets[i] = digest_from_hex(TARGET_HASHES[i]);
    }

    printf("=== 03 - MD5 jelszo visszafejtes ===\n");
    printf("Karakterkeszlet merete: %d\n", charset_len);
    printf("Jelszohossz: %d\n", PASSWORD_LEN);
    printf("Teljes jelszoter: %" PRIu64 "\n", total_space);
    printf("CPU limit: %" PRIu64 "\n", cpu_limit);
    printf("GPU global size: %" PRIu64 ", passes: %" PRIu64 "\n", gpu_global, gpu_passes);
    printf("Megjegyzes: a 4 megadott hash nem biztos, hogy beleesik a default tesztelt tartomanyba.\n\n");

    run_cpu(charset, charset_len, cpu_limit < total_space ? cpu_limit : total_space, targets);
    printf("\n");
    run_gpu(charset, charset_len, total_space, gpu_global, gpu_passes);
    return 0;
}
