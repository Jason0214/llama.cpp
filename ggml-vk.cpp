#include "ggml-vk.h"
#include "llama.h"
#include "ggml.h"

#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>

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
static std::vector<uint32_t> preferred_host_acs_mem_type_idxs;

static uint64_t gpu_local_size = 0;

static uint32_t compute_queue_idx = UINT32_MAX;
static uint32_t transfer_queue_idx = UINT32_MAX;
static bool     is_same_queue = false;

static vk::CommandPool compute_cmd_pool = {};
static vk::CommandPool transfer_cmd_pool = {};

static vk::DescriptorPool descript_pool = {};

static vk::PipelineCache pipeline_cache = {};

static constexpr uint32_t max_tensors_per_op = 4;

static void init_host_acs_mem_preferences() {
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

static uint32_t select_device_local_mem(const vk::MemoryPropertyFlags required_flags) {
    for (uint32_t i = 0; i < gpu_mem_props.memoryTypeCount; i++) {
        if ((required_flags & gpu_mem_props.memoryTypes[i].propertyFlags) == required_flags) {
            return i;
        }
    }
    throw std::runtime_error("No available device local memories are found!");
}

static void print_mem_type_preference() {
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
    const char * enabled_layers[] = { "VK_LAYER_KHRONOS_validation" };
    const uint32_t enabled_layer_count = ARRAY_LEN(enabled_layers);

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
                               .setPpEnabledLayerNames(enabled_layers)
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

        for (uint32_t i = 0; i < gpu_mem_props.memoryHeapCount; i++) {
            if (gpu_mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                gpu_local_size = std::max(gpu_local_size, gpu_mem_props.memoryHeaps[i].size);
            }
        }

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
                } else if (!strcmp("VK_KHR_portability_subset", device_exts[i].extensionName)) {
                    enabled_device_exts.push_back("VK_KHR_portability_subset");
                } else if (!strcmp(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME, device_exts[i].extensionName)) {
                    enabled_device_exts.push_back(VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME);
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

    {
        const auto pipeline_cache_info = vk::PipelineCacheCreateInfo();
        result = device.createPipelineCache(&pipeline_cache_info, nullptr, &pipeline_cache);
        GGML_ASSERT(result == vk::Result::eSuccess);
    }

    {
        const auto compute_cmd_pool_info =
            vk::CommandPoolCreateInfo()
                .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                .setQueueFamilyIndex(compute_queue_idx);
        result = device.createCommandPool(
            &compute_cmd_pool_info, nullptr, &compute_cmd_pool);
        GGML_ASSERT(result == vk::Result::eSuccess);
        if (is_same_queue) {
            transfer_cmd_pool = compute_cmd_pool;
        } else {
            const auto transfer_cmd_pool_info =
                vk::CommandPoolCreateInfo()
                    .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                    .setQueueFamilyIndex(transfer_queue_idx);
            result = device.createCommandPool(
                &transfer_cmd_pool_info, nullptr, &transfer_cmd_pool);
            GGML_ASSERT(result == vk::Result::eSuccess);
        }
    }

    {
        const vk::DescriptorPoolSize pool_sizes[] = {
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(max_tensors_per_op)
        };

        const auto descript_pool_info = vk::DescriptorPoolCreateInfo()
            .setMaxSets(1) // At most 1 command in flight
            .setPoolSizeCount(ARRAY_LEN(pool_sizes))
            .setPPoolSizes(pool_sizes);

        const vk::Result result = device.createDescriptorPool(&descript_pool_info, nullptr, &descript_pool);
        GGML_ASSERT(result == vk::Result::eSuccess);
    }
}

struct vk_host_buffer_private {
    const uint64_t magic = 0xdeadbeef;
    bool is_gpu_mem = false;
    vk::MemoryPropertyFlags flags = {};
    vk::DeviceMemory gpu_mem = {};
    uint64_t data_offset = 0;
};

static inline size_t calc_alloc_size(const size_t requested_size) {
    static_assert(sizeof(vk_host_buffer_private) <= GGML_MEM_ALIGN, "");
    return requested_size + sizeof(vk_host_buffer_private) + static_cast<size_t>(GGML_MEM_ALIGN - 1);
}

template<bool check=true>
static inline vk_host_buffer_private * get_vk_private(const void * data_ptr) {
    vk_host_buffer_private * ret = reinterpret_cast<vk_host_buffer_private *>(reinterpret_cast<uintptr_t>(data_ptr) - sizeof(vk_host_buffer_private));
    if (check) {
        GGML_ASSERT(ret->magic == 0xdeadbeef);
    }
    return ret;
}

static inline ggml_object * to_object(const ggml_tensor * tensor) {
    return reinterpret_cast<ggml_object *>(reinterpret_cast<uintptr_t>(tensor) - GGML_OBJECT_SIZE);
}

static inline vk_host_buffer_private * get_tensor_vk_private(const ggml_tensor * tensor) {
    ggml_object * obj = to_object(tensor);
    const void * mem_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tensor) - obj->offs);
    return get_vk_private(mem_buffer);
}

void * ggml_vk_host_malloc(size_t size) {
    if (device == vk::Device{}) {
        throw std::runtime_error("Vulkan device not initialized!");
    }

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
    void * data_addr = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(addr) + sizeof(vk_host_buffer_private) + align_mask) & ~align_mask);
    vk_host_buffer_private * vk_private = new (get_vk_private<false>(data_addr)) vk_host_buffer_private{};
    vk_private->is_gpu_mem = gpu_mem != vk::DeviceMemory{};
    vk_private->flags = gpu_mem_flags;
    vk_private->gpu_mem = gpu_mem;
    vk_private->data_offset = reinterpret_cast<uintptr_t>(data_addr) - reinterpret_cast<uintptr_t>(addr);

    return data_addr;
}

void ggml_vk_host_free(void * ptr) {
    if (ptr) {
        vk_host_buffer_private * vk_private = get_vk_private(ptr);
        if (vk_private->is_gpu_mem) {
            device.unmapMemory(vk_private->gpu_mem);
            device.freeMemory(vk_private->gpu_mem);
        } else {
            free(reinterpret_cast<uint8_t*>(ptr) - vk_private->data_offset);
        }
    }
}

size_t ggml_vk_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    (void)src0;
    (void)src1;
    (void)dst;
    return 0;
}

bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->n_dims == 2 &&
        src1->n_dims == 2 &&
        dst->n_dims == 2) {
            return true;
        }
    return false;
}

enum class KernelType {
    MAT_MUL_F32,
};

struct KernelPipeline {
    vk::PipelineLayout pipeline_layout = {};
    vk::DescriptorSetLayout descript_layout = {};
    vk::DescriptorSet descript_set = {}; // Only 1 set is used. At most one cmd in flight.
    vk::Pipeline pipeline = {};
    static constexpr uint32_t tile_rows_per_th = 8;
    static constexpr uint32_t tile_cols_per_th = 8;
    static constexpr uint32_t local_size_x = 8;
    static constexpr uint32_t local_size_y = 8;
    bool valid = false;
};

static const KernelPipeline & get_pipeline(const KernelType kernel_type) {
    vk::Result result = vk::Result::eSuccess;
    switch(kernel_type) {
        case KernelType::MAT_MUL_F32: {
            static KernelPipeline mat_mul_f32_pipeline = {};
            if (mat_mul_f32_pipeline.valid) {
                return mat_mul_f32_pipeline;
            }

            const std::string mat_mul_f32_comp = std::string(R"MatMulFp32Header(
                #version 430
                #extension GL_ARB_compute_shader : enable
                #extension GL_EXT_control_flow_attributes : enable

            )MatMulFp32Header") +
            "layout (local_size_x = " + std::to_string(mat_mul_f32_pipeline.local_size_x) + ", "
            "local_size_y = " + std::to_string(mat_mul_f32_pipeline.local_size_y) + ", local_size_z = 1) in; \n" +
            "const uint tileXLen = " + std::to_string(mat_mul_f32_pipeline.tile_rows_per_th) + ";\n" +
            "const uint tileYLen = " + std::to_string(mat_mul_f32_pipeline.tile_cols_per_th) + ";\n" +
            std::string(R"MatMulF32Body(
                uvec2 invId = uvec2(gl_GlobalInvocationID.xy);

                layout (push_constant, std430) uniform constants {
                    uint K;
                    uint M;
                    uint N;
                } tensorDims;

                // src0 K x M  -- GGML specific layout, transposed
                //             -- it makes tile-based mat mul less efficient
                //             -- because row elements' cache locality are not utilized.
                // src1 K x N
                // dst  M x N
                layout (set=0, binding=0) readonly buffer src0Buffer { float src0[]; };
                layout (set=0, binding=1) readonly buffer src1Buffer { float src1[]; };
                layout (set=0, binding=2) writeonly buffer dstBuffer { float dst[]; };

                float src0Local[tileXLen];
                float dstLocal[tileXLen][tileYLen];

                void main() {
                    // Each thread going to compute an entire tile
                    [[unroll]] for (uint i = 0; i < tileXLen; i++) {
                        [[unroll]] for (uint j = 0; j < tileYLen; j++) {
                            dstLocal[i][j] = 0;
                        }
                    }

                    // Loop over mul dimension
                    [[unroll]] for (uint k = 0; k < tensorDims.K; k++) {
                        uint src0X = k;

                        // Load each tile col of src0 data into faster memory.
                        [[unroll]] for (uint i = 0; i < tileXLen; i++) {
                            uint src0Y = invId.x * tileXLen + i;
                            src0Local[i] = src0[src0Y * tensorDims.K + src0X];
                        }

                        uint src1X = k;
                        // Loop over tile col of src1 data.
                        [[unroll]] for (uint j = 0; j < tileYLen; j++) {
                            uint src1Y = invId.y * tileYLen + j;
                            float src1Local = src1[src1Y * tensorDims.K + src1X];

                            [[unroll]] for (uint i = 0; i < tileXLen; i++) {
                                dstLocal[i][j] = src0Local[i] * src1Local + dstLocal[i][j];
                            }
                        }
                    }

                    [[unroll]] for (uint j = 0; j < tileYLen; j++) {
                        [[unroll]] for (uint i = 0; i < tileXLen; i++) {
                            uint dstX = invId.x * tileXLen + i;
                            uint dstY = invId.y * tileYLen + j;
                            if (dstX < tensorDims.M && dstY < tensorDims.N) {
                                dst[dstY * tensorDims.M + dstX] = dstLocal[i][j];
                            }
                        }
                    }
                }
            )MatMulF32Body");

            vk::ShaderModule shader_module = {};
            {
                shaderc::Compiler compiler;
                shaderc::CompileOptions options;
                shaderc::SpvCompilationResult spv_result = compiler.CompileGlslToSpv(mat_mul_f32_comp, shaderc_shader_kind::shaderc_glsl_compute_shader, "mat_mul_f32_comp", options);
                if (spv_result.GetCompilationStatus() != shaderc_compilation_status_success) {
                    throw std::runtime_error("Fail to compile Mat Mul F32 compute shader: " + spv_result.GetErrorMessage());
                }
                std::vector<uint32_t> comp_spirv = {};
                comp_spirv.assign(spv_result.cbegin(), spv_result.cend());
                const auto shader_info = vk::ShaderModuleCreateInfo()
                    .setCodeSize(comp_spirv.size() * sizeof(uint32_t))
                    .setPCode(comp_spirv.data());

                result = device.createShaderModule(&shader_info, nullptr, &shader_module);
                GGML_ASSERT(result == vk::Result::eSuccess);
            }

            {
                const vk::DescriptorSetLayoutBinding bindings[] = {
                    vk::DescriptorSetLayoutBinding()
                        .setBinding(0)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eCompute),
                    vk::DescriptorSetLayoutBinding()
                        .setBinding(1)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eCompute),
                    vk::DescriptorSetLayoutBinding()
                        .setBinding(2)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eCompute),
                };

                const auto descript_layout_info = vk::DescriptorSetLayoutCreateInfo()
                    .setFlags({})
                    .setBindings(bindings)
                    .setBindingCount(ARRAY_LEN(bindings));
                result = device.createDescriptorSetLayout(&descript_layout_info, nullptr, &mat_mul_f32_pipeline.descript_layout);
                GGML_ASSERT(result == vk::Result::eSuccess);
            }

            {
                const auto descrip_alloc_info = vk::DescriptorSetAllocateInfo()
                    .setDescriptorPool(descript_pool)
                    .setDescriptorSetCount(1)
                    .setPSetLayouts(&mat_mul_f32_pipeline.descript_layout);

                result = device.allocateDescriptorSets(&descrip_alloc_info, &mat_mul_f32_pipeline.descript_set);
                GGML_ASSERT(result == vk::Result::eSuccess);
            }

            {
                const auto push_const_info = vk::PushConstantRange()
                    .setOffset(0)
                    .setSize(sizeof(uint32_t) * 3)
                    .setStageFlags(vk::ShaderStageFlagBits::eCompute);

                const auto pipeline_layout_info = vk::PipelineLayoutCreateInfo()
                    .setSetLayoutCount(1)
                    .setPSetLayouts(&mat_mul_f32_pipeline.descript_layout)
                    .setPPushConstantRanges(&push_const_info)
                    .setPushConstantRangeCount(1);

                result = device.createPipelineLayout(&pipeline_layout_info, nullptr, &mat_mul_f32_pipeline.pipeline_layout);
                GGML_ASSERT(result == vk::Result::eSuccess);
            }

            {
                const auto shader_stage_info = vk::PipelineShaderStageCreateInfo()
                                   .setStage(vk::ShaderStageFlagBits::eCompute)
                                   .setModule(shader_module)
                                   .setPName("main");

                const auto pipeline_info = vk::ComputePipelineCreateInfo()
                    .setStage(shader_stage_info)
                    .setLayout(mat_mul_f32_pipeline.pipeline_layout);

                result = device.createComputePipelines(
                    pipeline_cache, 1, &pipeline_info, nullptr, &mat_mul_f32_pipeline.pipeline);
                GGML_ASSERT(result == vk::Result::eSuccess);

                device.destroyShaderModule(shader_module, nullptr);
            }

            mat_mul_f32_pipeline.valid = true;
            return mat_mul_f32_pipeline;
        }
        default:
            throw std::runtime_error("Unknown vulkan kernel type!");
    }
}


template<bool transfer_src>
struct AutoTensorBuffer {
    AutoTensorBuffer(const struct ggml_tensor * tensor) {
        const vk::DeviceSize buffer_size = to_object(tensor)->size - sizeof(struct ggml_tensor);
        const auto buffer_info = vk::BufferCreateInfo()
            .setFlags(vk::BufferCreateFlags{})
            .setSize(buffer_size)
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                (transfer_src ? vk::BufferUsageFlagBits::eTransferSrc
                              : vk::BufferUsageFlagBits::eTransferDst))
            .setSharingMode(vk::SharingMode::eExclusive);
        const auto result = device.createBuffer(&buffer_info, nullptr, &vk_buffer);
        GGML_ASSERT(result == vk::Result::eSuccess);

        vk_host_buffer_private* vk_buffer_private = get_tensor_vk_private(tensor);
        const vk::DeviceSize offset = vk_buffer_private->data_offset + to_object(tensor)->offs + sizeof(ggml_tensor);
        device.bindBufferMemory(vk_buffer, vk_buffer_private->gpu_mem, offset);
    }

    vk::DescriptorBufferInfo write_descriptor() {
        return vk::DescriptorBufferInfo().setBuffer(vk_buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
    }

    ~AutoTensorBuffer() {
        device.destroyBuffer(vk_buffer, nullptr);
    }

    AutoTensorBuffer(const AutoTensorBuffer &) = delete;
    AutoTensorBuffer & operator=(const AutoTensorBuffer &) = delete;

    vk::Buffer vk_buffer = {};
};


void ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {
    (void)wdata;
    (void)wsize;
#define CHECK_IS_GPU_MEM(tensor) \
    if (!get_tensor_vk_private(tensor)->is_gpu_mem || tensor->data != tensor + 1) { \
        throw std::runtime_error(#tensor " is not allocated on GPU. Functionality not implemented!"); \
    }

    CHECK_IS_GPU_MEM(src0);
    CHECK_IS_GPU_MEM(src1);
    CHECK_IS_GPU_MEM(dst);

    const KernelPipeline & kernel_pipeline = get_pipeline(KernelType::MAT_MUL_F32);

    vk::CommandBuffer cmdbuf = {};
    const auto cmd_alloc_info = vk::CommandBufferAllocateInfo()
                                    .setCommandPool(compute_cmd_pool)
                                    .setLevel(vk::CommandBufferLevel::ePrimary)
                                    .setCommandBufferCount(1);
    auto result = device.allocateCommandBuffers(&cmd_alloc_info, &cmdbuf);
    GGML_ASSERT(result == vk::Result::eSuccess);

    const auto cmd_buf_begin_info = vk::CommandBufferBeginInfo();
    cmdbuf.begin(cmd_buf_begin_info);

    // iGpu device local memory has similar performance as host memories?
    const bool prefer_device_local = (gpu_props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) &&
        (to_object(src0)->size + to_object(src1)->size + to_object(dst)->size) < gpu_local_size;

    AutoTensorBuffer<true> src0_buffer(src0);
    AutoTensorBuffer<true> src1_buffer(src1);
    AutoTensorBuffer<false> dst_buffer(dst);

    if (prefer_device_local) {
        throw std::runtime_error("Not implemented!");
    } else {
        const vk::DescriptorBufferInfo buffer_infos [] = {
            src0_buffer.write_descriptor(),
            src1_buffer.write_descriptor(),
            dst_buffer.write_descriptor(),
        };

        vk::WriteDescriptorSet write_descripts[ARRAY_LEN(buffer_infos)] = {};
        for (uint32_t i = 0; i < ARRAY_LEN(write_descripts); i++) {
            write_descripts[i] = vk::WriteDescriptorSet()
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setPBufferInfo(&buffer_infos[i])
                .setDstBinding(i)
                .setDstSet(kernel_pipeline.descript_set);
        };
        device.updateDescriptorSets(ARRAY_LEN(write_descripts), write_descripts, 0, nullptr);

        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, kernel_pipeline.pipeline);
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                  kernel_pipeline.pipeline_layout,
                                  0, 1, &kernel_pipeline.descript_set, 0, nullptr);

        if (src0->ne[0] > UINT32_MAX || src1->ne[1] > UINT32_MAX || src0->ne[1] > UINT32_MAX) {
            throw std::runtime_error("Larger than UIN32_MAX tensor dimenstion not implemented!");
        }
        struct MatMulDims {
            uint32_t K;
            uint32_t M;
            uint32_t N;
        } mat_mul_dims = {
            // GGML matrix layout always puts higher rank into the inner dimension
            .K=static_cast<uint32_t>(src0->ne[0]),
            .M=static_cast<uint32_t>(dst->ne[0]),
            .N=static_cast<uint32_t>(dst->ne[1]),
        };

        cmdbuf.pushConstants(kernel_pipeline.pipeline_layout,
                             vk::ShaderStageFlagBits::eCompute,
                             0,
                             sizeof(MatMulDims),
                             &mat_mul_dims);

        const auto div_round_up = [] (uint32_t x, uint32_t y) -> uint32_t { return (x + y - 1) / y; };
        cmdbuf.dispatch(
            div_round_up(dst->ne[0], kernel_pipeline.local_size_x * kernel_pipeline.tile_rows_per_th),
            div_round_up(dst->ne[1], kernel_pipeline.local_size_y * kernel_pipeline.tile_cols_per_th),
            1);
    }
    cmdbuf.end();

    const vk::Queue compute_queue = device.getQueue(compute_queue_idx, 0);
    const auto submit_info = vk::SubmitInfo()
        .setWaitSemaphoreCount(0)
        .setCommandBufferCount(1)
        .setPCommandBuffers(&cmdbuf)
        .setSignalSemaphoreCount(0);
    compute_queue.submit(submit_info, {});
    compute_queue.waitIdle();
}
