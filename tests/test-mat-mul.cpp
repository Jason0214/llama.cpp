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

int main(int, const char **) {
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

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 72);
    ggml_set_f32(m11, 1.0f);

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 64);
    ggml_set_f32(m12, 2.0f);

    printf("\n------ Test 1 - Matrix Mult via F32 code ------------------------------------------------------------------------------\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm12 = ggml_mul_mat(ctx, m11, m12);

    // printf("Creating compute graph\n");
    struct ggml_cgraph gf = ggml_build_forward(m11xm12);

    ggml_graph_compute(ctx, &gf);

    GGML_ASSERT(m11xm12->type == GGML_TYPE_F32);

    for (int j = 0; j < m11xm12->ne[1]; j++) {
        for (int k = 0; k < m11xm12->ne[0]; k++) {
            const float value = ((float *) m11xm12->data)[j*m11xm12->ne[0]+k];
            if (value != 256.0f) {
                fprintf(stderr, "k %d, j %d : %.2f\n", k, j, value);
            }
        }
    }

    return 0;
}
