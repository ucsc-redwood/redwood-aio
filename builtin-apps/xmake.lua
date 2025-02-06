
local source_files = {
    "cifar-dense/dense_appdata.cpp",
    "cifar-sparse/sparse_appdata.cpp",
    "tree/tree_appdata.cpp",
    "tree/omp/tree_kernel.cpp",
}

target("builtin-apps")
    set_kind("static")

    add_files(source_files)

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

-- vukan 

    add_deps("vk-backend")

    add_files({
        "cifar-sparse/vulkan/vk_dispatcher.cpp",
        "cifar-dense/vulkan/vk_dispatcher.cpp",
        "tree/vulkan/vk_dispatcher.cpp",
    })

    add_packages("vulkan-hpp", "vulkan-memory-allocator")
---
    add_packages("spdlog")
    add_packages("cli11")
    add_packages("glm")
    add_packages("cli11")
target_end()

if not is_plat("android") then

target("builtin-apps-cuda")
    set_kind("static")

    add_files(source_files)

    add_files({
        -- "common/cuda/cu_mem_resource.cu",
        "cifar-dense/cuda/cu_dense_kernel.cu",
        "cifar-dense/cuda/cu_kernels.cu",
        "cifar-sparse/cuda/cu_dispatcher.cu",
        "cifar-sparse/cuda/cu_kernels.cu",

        "tree/cuda/01_morton.cu",
        "tree/cuda/02_sort.cu",
        "tree/cuda/03_unique.cu",
        "tree/cuda/04_radix_tree.cu",
        "tree/cuda/05_edge_count.cu",
        "tree/cuda/06_prefix_sum.cu",
        "tree/cuda/07_octree.cu",
        "tree/cuda/im_storage.cu",
        "tree/cuda/kernel.cu",
    })

    add_packages("spdlog")
    add_packages("glm")
    add_packages("cub")

    add_cugencodes("native")


target_end()

end
