#pragma once
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_init_vulkan(void);

void * ggml_vk_host_malloc(size_t size);
void   ggml_vk_host_free(void * ptr);

bool   ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_vk_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

#ifdef  __cplusplus
}
#endif
