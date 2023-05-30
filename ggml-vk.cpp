#include "ggml-vk.h"
#include "llama.h"
#include "ggml.h"

#include <vulkan/vulkan.hpp>

#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define APP_NAME "llama.cpp"

#define ARRAY_LEN(x) sizeof(x) / sizeof(x[0])

static vk::Instance       inst   = {};
static vk::PhysicalDevice gpu    = {};
static vk::Device         device = {};

static vk::PhysicalDeviceProperties gpu_props = {};
static vk::PhysicalDeviceMemoryProperties gpu_mem_props = {};

static uint32_t compute_queue_idx = UINT32_MAX;
static uint32_t transfer_queue_idx = UINT32_MAX;
static bool     is_same_queue = false;

static std::vector<uint32_t> preferred_host_acs_mem_type_idxs;

void init_host_acs_mem_preferences() {
    preferred_host_acs_mem_type_idxs.clear();
    for (uint32_t i = 0; i < gpu_mem_props.memoryTypeCount; i++) {
        if (gpu_mem_props.memoryTypes[i].propertyFlags &
            // Performance of CPU read on non-cached memory is catastrophic
            (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached)) {
            preferred_host_acs_mem_type_idxs.push_back(i);
        }
    }

    if (preferred_host_acs_mem_type_idxs.empty()) {
        throw std::runtime_error("No host visible and cached memory type found!\n");
    }

    const auto compare = [&](uint32_t mem_idx_lhs, uint32_t mem_idx_rhs) -> bool {
        if ((gpu_mem_props.memoryTypes[mem_idx_lhs].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
            !(gpu_mem_props.memoryTypes[mem_idx_rhs].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            return true;
        }
        return false;
    };

    std::sort(preferred_host_acs_mem_type_idxs.begin(), preferred_host_acs_mem_type_idxs.end(), compare);
}

void print_mem_type_preference() {
    fprintf(stderr, "Host accessible memory allocation preferences:\n");
    for (const uint32_t mem_type_idx : preferred_host_acs_mem_type_idxs) {
        const vk::MemoryHeap & heap = gpu_mem_props.memoryHeaps[gpu_mem_props.memoryTypes[mem_type_idx].heapIndex];
        fprintf(stderr, "\t memory property '%s',\n"
                        "\t\t heap size '0x%" PRIx64 "', heap property '%s'.\n",
            vk::to_string(gpu_mem_props.memoryTypes[mem_type_idx].propertyFlags).c_str(),
            static_cast<uint64_t>(heap.size),
            vk::to_string(heap.flags).c_str());
    }
}

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

        init_host_acs_mem_preferences();
        print_mem_type_preference();
    }

    bool has_vk_amd_allocation_behavior = false;
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
                if (!strcmp(VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME, device_exts[i].extensionName)) {
                    enabled_device_exts.push_back(VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME);
                    has_vk_amd_allocation_behavior = true;
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

        // TODO: Performance choises, using a single queue or multiple queues?
        // It may be workload dependent and device dependent. Prefer single queue for now.
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

    {
        constexpr uint32_t queue_count = 1;
        const float queue_priorities[queue_count] = { 1.0f };

        const vk::DeviceQueueCreateInfo queues[] = {
            vk::DeviceQueueCreateInfo()
                .setQueueFamilyIndex(compute_queue_idx)
                .setQueueCount(queue_count)
                .setPQueuePriorities(queue_priorities),
            vk::DeviceQueueCreateInfo()
                .setQueueFamilyIndex(transfer_queue_idx)
                .setQueueCount(queue_count)
                .setPQueuePriorities(queue_priorities),
        };
        uint32_t queue_family_count = ARRAY_LEN(queues);
        if (is_same_queue) {
            queue_family_count = 1;
        }

        auto device_info =
            vk::DeviceCreateInfo()
                .setQueueCreateInfoCount(queue_family_count)
                .setPQueueCreateInfos(queues)
                // No required device layers.
                .setEnabledLayerCount(0)
                .setPpEnabledLayerNames(nullptr)
                // No required device extensions.
                .setEnabledExtensionCount(enabled_device_exts.size())
                .setPpEnabledExtensionNames(enabled_device_exts.data())
                // No required features.
                .setPEnabledFeatures(nullptr);

        const auto mem_alloc_info = vk::DeviceMemoryOverallocationCreateInfoAMD()
            .setOverallocationBehavior(vk::MemoryOverallocationBehaviorAMD::eAllowed);
        if (has_vk_amd_allocation_behavior) {
            device_info.setPNext(&mem_alloc_info);
        }

        result = gpu.createDevice(&device_info, nullptr, &device);
        GGML_ASSERT(result == vk::Result::eSuccess);
    }
}

struct alignas(GGML_MEM_ALIGN) vk_host_buffer_private {
    const uint64_t magic = 0xdeadbeef;
    bool is_gpu_mem = false;
    vk::MemoryPropertyFlags flags = {};
    vk::DeviceMemory gpu_mem = {};
    uint32_t data_offset = 0;
};

static inline size_t calc_alloc_size(const size_t requested_size) {
    return requested_size + sizeof(vk_host_buffer_private) + static_cast<size_t>(GGML_MEM_ALIGN - 1);
}

static inline vk_host_buffer_private * get_vk_private(void * data_ptr) {
    return reinterpret_cast<vk_host_buffer_private *>(reinterpret_cast<uintptr_t>(data_ptr) - sizeof(vk_host_buffer_private));
}

static inline void * get_data_addr(vk_host_buffer_private * vk_private) {
    return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(vk_private) + sizeof(vk_host_buffer_private));
}

void * ggml_vk_host_malloc(size_t size) {
    size = calc_alloc_size(size);

    void * addr = nullptr;
    vk::Result result = vk::Result::eSuccess;

    vk::DeviceMemory gpu_mem = {};
    vk::MemoryPropertyFlags gpu_mem_flags = {};
    for (const uint32_t mem_type_idx : preferred_host_acs_mem_type_idxs) {
        const auto mem_alloc_info = vk::MemoryAllocateInfo()
            .setAllocationSize(size)
            .setMemoryTypeIndex(mem_type_idx);

        result = device.allocateMemory(&mem_alloc_info, nullptr, &gpu_mem);
        if (result == vk::Result::eSuccess) {
            result = device.mapMemory(gpu_mem, 0, size, vk::MemoryMapFlags{}, &addr);
            if (result == vk::Result::eSuccess) {
                GGML_ASSERT(addr);
                gpu_mem_flags = gpu_mem_props.memoryTypes[mem_type_idx].propertyFlags;
                break;
            } else {
                device.freeMemory(gpu_mem);
                gpu_mem = vk::DeviceMemory{};
            }
        }

        fprintf(stderr, "Try allocate gpu memory from type index '%d' failed!\n", mem_type_idx);
        if (result == vk::Result::eErrorOutOfHostMemory || result == vk::Result::eErrorOutOfDeviceMemory) {
            fprintf(stderr, "Out of device memory!\n");
        } else {
            fprintf(stderr, "Unknown error '%s'!\n", vk::to_string(result).c_str());
        }
    }

    if (!addr) {
        fprintf(stderr, "Not able to allocate from gpu memory. Falls back to system memory."
                        "May introduce additional CPU copies.\n");
        addr = malloc(size);
        GGML_ASSERT(addr);
        if (!addr) {
            // Out of memory
            return nullptr;
        }
    }

    constexpr uintptr_t align_mask = GGML_MEM_ALIGN - 1u;
    void * aligned_addr = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(addr) + align_mask) & ~align_mask);
    vk_host_buffer_private * vk_private = new (aligned_addr) vk_host_buffer_private;
    vk_private->is_gpu_mem = gpu_mem != vk::DeviceMemory{};
    vk_private->flags = gpu_mem_flags;
    vk_private->gpu_mem = gpu_mem;
    vk_private->data_offset = reinterpret_cast<uintptr_t>(get_data_addr(vk_private)) - reinterpret_cast<uintptr_t>(aligned_addr);

    return get_data_addr(vk_private);
}

void ggml_vk_host_free(void * ptr) {
    if (ptr) {
        vk_host_buffer_private * vk_private = get_vk_private(ptr);

        GGML_ASSERT(vk_private->magic == 0xdeadbeaf);

        if (vk_private->is_gpu_mem) {
            device.unmapMemory(vk_private->gpu_mem);
            device.freeMemory(vk_private->gpu_mem);
        } else {
            free(reinterpret_cast<uint8_t*>(ptr) - vk_private->data_offset);
        }
    }
}

bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    return false;
}

size_t ggml_vk_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    return 0;
}

void ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {
    return ;
}
