#include "ggml-vk.h"
#include "llama.h"
#include "llama-util.h"

#include <vulkan/vulkan.hpp>

#include <cstring>

#define APP_NAME "llama.cpp"

void ggml_init_vulkan(void) {
    vk::Result result = vk::Result::eSuccess;

    const auto app_info = vk::ApplicationInfo()
                         .setPApplicationName(APP_NAME)
                         .setApplicationVersion(0)
                         .setPEngineName(APP_NAME)
                         .setEngineVersion(0)
                         .setApiVersion(VK_API_VERSION_1_2); // TODO, check loader version in build time?

    // No explicit layer required.
    const uint32_t enabled_layer_count = 0;
    // No instance extensions required.
    const uint32_t enabled_ext_count = 0;
    const auto inst_info = vk::InstanceCreateInfo()
                               .setPApplicationInfo(&app_info)
                               .setEnabledLayerCount(enabled_layer_count)
                               .setPpEnabledLayerNames(nullptr)
                               .setEnabledExtensionCount(enabled_ext_count)
                               .setPpEnabledExtensionNames(nullptr);
}
