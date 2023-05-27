#include "ggml-vk.h"
#include "llama.h"
#include "llama-util.h"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define APP_NAME "llama.cpp"

#define ARRAY_LEN(x) sizeof(x) / sizeof(x[0])

static vk::Instance       inst   = {};
static vk::Device         device = {};
static vk::PhysicalDevice gpu    = {};

static vk::PhysicalDeviceProperties gpu_props = {};
static vk::PhysicalDeviceMemoryProperties gpu_mem_props = {};

static uint32_t compute_queue_idx = UINT32_MAX;
static uint32_t transfer_queue_idx = UINT32_MAX;
static bool     is_same_queue = false;

void ggml_init_vulkan(void) {
    vk::Result result = vk::Result::eSuccess;

    const auto app_info = vk::ApplicationInfo()
                         .setPApplicationName(APP_NAME)
                         .setApplicationVersion(0)
                         .setPEngineName(APP_NAME)
                         .setEngineVersion(0)
                         .setApiVersion(VK_API_VERSION_1_1); // TODO, check loader version in build time?

    // No explicit layer required.
    const uint32_t enabled_layer_count = 0;

#if defined(__APPLE__)
    const char * enabled_inst_exts[] = { "VK_KHR_portability_enumeration" };
    const uint32_t enabled_ext_count = ARRAY_LEN(enabled_inst_exts);
#else
    const char ** enabled_inst_exts = nullptr;
    const uint32_t enabled_ext_count = 0;
#endif

    const auto inst_info = vk::InstanceCreateInfo()
                               .setPApplicationInfo(&app_info)
                               .setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR)
                               .setEnabledLayerCount(enabled_layer_count)
                               .setPpEnabledLayerNames(nullptr)
                               .setEnabledExtensionCount(enabled_ext_count)
                               .setPpEnabledExtensionNames(enabled_inst_exts);

    result = vk::createInstance(&inst_info, nullptr, &inst);
    if (result == vk::Result::eErrorIncompatibleDriver) {
        throw std::runtime_error("Cannot find a compatible Vulkan installable client driver (ICD).\n");
    } else if (result == vk::Result::eErrorLayerNotPresent) {
        throw std::runtime_error("Cannot find a specified vulkan layer.\n");
    } else if (result == vk::Result::eErrorExtensionNotPresent) {
        throw std::runtime_error("Cannot find a specified extension library.\n");
    } else if (result != vk::Result::eSuccess) {
        throw std::runtime_error("vkCreateInstance failed.\n");
    }

    uint32_t gpu_count = 0;
    result = inst.enumeratePhysicalDevices(&gpu_count, static_cast<vk::PhysicalDevice *>(nullptr));
    GGML_ASSERT(result == vk::Result::eSuccess);

    if (gpu_count <= 0) {
        throw std::runtime_error("vkEnumeratePhysicalDevices reported zero accessible devices.\n");
    }

    {
        std::unique_ptr<vk::PhysicalDevice[]> physical_devices(new vk::PhysicalDevice[gpu_count]);
        result = inst.enumeratePhysicalDevices(&gpu_count, physical_devices.get());
        GGML_ASSERT(result == vk::Result::eSuccess);

        uint32_t selected_gpu_idx = UINT32_MAX;

        // Find the first reported dGPU. If no dGPU, choose iGPU. CPU is not going to be selected.
        // TODO: Probably sort by some critiria if multiple dGPU/iGPU present.

        const vk::PhysicalDeviceType preferred_device_types[] = {
            vk::PhysicalDeviceType::eDiscreteGpu,
            vk::PhysicalDeviceType::eIntegratedGpu,
        };
        for (uint32_t pass = 0; pass < ARRAY_LEN(preferred_device_types); pass++) {
            for (uint32_t i = 0; i < gpu_count; i++) {
                const auto & physicalDeviceProperties = physical_devices[i].getProperties();
                if (physicalDeviceProperties.deviceType == preferred_device_types[pass]) {
                    selected_gpu_idx = i;
                    goto gpu_select_done;
                }
            }
        }
gpu_select_done:
        if (selected_gpu_idx == UINT32_MAX) {
            throw std::runtime_error("Fail to find a dGPU or iGPU Vulkan device\n!");
        }

        gpu = physical_devices[selected_gpu_idx];

        gpu_props = gpu.getProperties();
        gpu.getMemoryProperties(&gpu_mem_props);

        fprintf(stderr, "Selected Vulkan physical device '%s' type '%s'\n",
            gpu_props.deviceName.data(),
            vk::to_string(gpu_props.deviceType).c_str());
    }

    std::vector<const char *> enabled_device_exts = {};
    {
        uint32_t device_exts_count= 0;
        result = gpu.enumerateDeviceExtensionProperties(nullptr, &device_exts_count, nullptr);
        GGML_ASSERT(result == vk::Result::eSuccess);

        if (device_exts_count > 0) {
            std::unique_ptr<vk::ExtensionProperties[]> device_exts(
                new vk::ExtensionProperties[device_exts_count]);
            result = gpu.enumerateDeviceExtensionProperties(
                nullptr, &device_exts_count, device_exts.get());

            GGML_ASSERT(result == vk::Result::eSuccess);
            for (uint32_t i = 0; i < device_exts_count; i++) {
                // Allow non-conformante device
                if (!strcmp("VK_KHR_portability_subset", device_exts[i].extensionName)) {
                    enabled_device_exts.push_back("VK_KHR_portability_subset");
                }
            }
        }
    }

    {
        uint32_t queue_family_count = 0;
        gpu.getQueueFamilyProperties(&queue_family_count, nullptr);

        std::unique_ptr<vk::QueueFamilyProperties[]> queue_families(
            new vk::QueueFamilyProperties[queue_family_count]);

        gpu.getQueueFamilyProperties(&queue_family_count, queue_families.get());

        const auto select_queue = [&] (const vk::QueueFlagBits target, const vk::QueueFlags preferred_not) -> uint32_t {
            GGML_ASSERT((target & preferred_not) == static_cast<vk::QueueFlags>(0));
            for (uint32_t pass = 0; pass < 3; pass++) {
                for (uint32_t i = 0; i < queue_family_count; i++) {
                    // Looking for single purpose queue
                    if (pass == 0 && queue_families[i].queueFlags == target) {
                        return i;
                    }
                    if (pass == 1 && queue_families[i].queueFlags & target &&
                        !(queue_families[i].queueFlags & preferred_not)) {
                        return i;
                    }
                    if (pass == 2 && queue_families[i].queueFlags & target) {
                        return i;
                    }
                }
            }
            return UINT32_MAX;
        };

        compute_queue_idx = select_queue(vk::QueueFlagBits::eCompute,
            /*preferred_not=*/vk::QueueFlagBits::eGraphics);
        if (compute_queue_idx == UINT32_MAX) {
            throw std::runtime_error("Fail to find a compute capable queue on the selected gpu!\n");
        }

        // TODO: using a single queue or multiple queu may be workload dependent and device dependent.
        // For now prefer single queue.
        if (queue_families[compute_queue_idx].queueFlags & vk::QueueFlagBits::eTransfer) {
            transfer_queue_idx = compute_queue_idx;
            is_same_queue = true;
        } else {
            transfer_queue_idx = select_queue(vk::QueueFlagBits::eTransfer,
                /*preferred_not=*/vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
            is_same_queue = transfer_queue_idx == compute_queue_idx;
        }
        if (transfer_queue_idx == UINT32_MAX) {
            throw std::runtime_error("Fail to find a transfer capable queue on the selected gpu!\n");
        }

        fprintf(stderr, "Selected queue '%u' with '%s' to dispatch compute workload.\n",
            compute_queue_idx, vk::to_string(queue_families[compute_queue_idx].queueFlags).c_str());
        fprintf(stderr, "Selected queue '%u' with '%s' to dispatch transfer workload.\n",
            transfer_queue_idx, vk::to_string(queue_families[transfer_queue_idx].queueFlags).c_str());
    }

}
