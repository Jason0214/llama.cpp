#pragma once
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_init_vulkan(void);

void   ggml_vk_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_vk_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void ggml_vk_transform_tensor(struct ggml_tensor * tensor);
void ggml_vk_load_data(const char * fname, struct ggml_tensor * tensors, size_t offset);

#ifdef  __cplusplus
}
#endif
