#include "ggml.h"
#include "llama-util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

int main(int argc, const char ** argv) {
    uint32_t dst_mat_width = 0;
    uint32_t dst_mat_height = 0;
    uint32_t mul_dim = 0;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-w" || arg == "--width") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            dst_mat_width = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--height") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            dst_mat_height = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--mul-dims") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            mul_dim = std::stoi(argv[i]);
        }
    }

    if (dst_mat_width == 0 || dst_mat_height == 0 || mul_dim == 0) {
        invalid_param = true;
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        exit(1);
    }

    size_t ctx_size = 1024 * 1024 * 1024;
#ifdef GGML_USE_VULKAN
    // Init backend
    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
#endif

#ifdef GGML_USE_VULKAN
    llama_ctx_buffer buf;
    buf.resize(ctx_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ buf.addr,
        /* no_alloc   =*/ 0
    };
#else
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };
#endif
    struct ggml_context * ctx = ggml_init(params);

    printf("Creating new tensors\n");

    constexpr float src0_value = 2.0f;
    constexpr float src1_value = 2.0f;
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mul_dim, dst_mat_width);
    ggml_set_f32(m11, src0_value);

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mul_dim, dst_mat_height);
    ggml_set_f32(m12, src1_value);

    printf("\n------ Test 1 - Matrix Mult via F32 code ------------------------------------------------------------------------------\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm12 = ggml_mul_mat(ctx, m11, m12);

    // printf("Creating compute graph\n");
    struct ggml_cgraph gf = ggml_build_forward(m11xm12);

    ggml_graph_compute(ctx, &gf);

    GGML_ASSERT(m11xm12->type == GGML_TYPE_F32);

    const float expect_value = src0_value * src1_value * float(mul_dim);
    for (int j = 0; j < m11xm12->ne[1]; j++) {
        for (int k = 0; k < m11xm12->ne[0]; k++) {
            const float value = ((float *) m11xm12->data)[j*m11xm12->ne[0]+k];
            if (value != expect_value) {
                fprintf(stderr, "Expect '%.2f', but result[%d][%d] is '%.2f'\n", expect_value, k, j, value);
            }
        }
    }

    return 0;
}
