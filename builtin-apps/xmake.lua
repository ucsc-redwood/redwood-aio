-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Builtin Apps (Tree, CIFAR-Sparse, CIFAR-Dense) Static Library
-- ----------------------------------------------------------------

target("builtin-apps")
    set_kind("static")
    add_rules("common_flags")
    set_group("static-libs")
    
    add_headerfiles({
        -- common headers
        "affinity.hpp",
        "app.hpp",
        "base_appdata.hpp",
        "conf.hpp",
        "resources_path.hpp",
        
        -- cifar-sparse headers
        "cifar-sparse/arg_max.hpp",
        "cifar-sparse/csr.hpp",
        "cifar-sparse/sparse_appdata.hpp",
        "cifar-sparse/omp/sparse_kernel.hpp",
        
        -- cifar-dense headers
        "cifar-dense/arg_max.hpp",
        "cifar-dense/dense_appdata.hpp",
        "cifar-dense/omp/dense_kernel.hpp",
        
        -- tree headers
        "tree/tree_appdata.hpp",
        "tree/omp/tree_kernel.hpp",
        "tree/omp/func_brt.hpp",
        "tree/omp/func_edge.hpp",
        "tree/omp/func_morton.hpp",
        "tree/omp/func_octree.hpp",
        "tree/omp/func_sort.hpp",
    })

    add_files({
        "conf.cpp",

        -- cifar-dense implementations
        "cifar-dense/dense_appdata.cpp",
        "cifar-dense/omp/dense_kernel.cpp",

        -- cifar-sparse implementations
        "cifar-sparse/sparse_appdata.cpp",
        "cifar-sparse/omp/sparse_kernel.cpp",
        
        -- tree implementations
        "tree/tree_appdata.cpp",
        "tree/omp/tree_kernel.cpp",
    })

target_end()

-- ----------------------------------------------------------------------------
-- Vulkan Static Library
-- ----------------------------------------------------------------------------

target("builtin-apps-vulkan")
    set_kind("static")
    set_group("static-libs")
    add_rules("common_flags", "vulkan_config")

    add_deps("kiss-vk")

    -- add_includedirs("$(projectdir)")
    
    add_headerfiles({
        -- App specific headers
        "cifar-sparse/vulkan/vk_dispatcher.hpp",
        "cifar-dense/vulkan/vk_dispatcher.hpp",
        "tree/vulkan/vk_dispatcher.hpp",
        "tree/vulkan/tmp_storage.hpp",
    })

    add_files({
        -- App specific implementations
        "cifar-sparse/vulkan/vk_dispatcher.cpp",
        "cifar-dense/vulkan/vk_dispatcher.cpp",
        "tree/vulkan/vk_dispatcher.cpp",
    })

target_end()

-- ----------------------------------------------------------------------------
-- CUDA Static Library
-- ----------------------------------------------------------------------------

-- if not is_plat("android") then
if has_config("cuda") then

target("builtin-apps-cuda")
    set_kind("static")
    set_group("static-libs")
    add_rules("common_flags")

    add_headerfiles({
        -- Common CUDA headers
        "common/cuda/cu_mem_resource.cuh",
        "common/cuda/helpers.cuh",
        
        -- CIFAR sparse CUDA headers
        "cifar-sparse/cuda/cu_kernels.cuh",
        "cifar-sparse/cuda/cu_dispatcher.cuh",
        
        -- CIFAR dense CUDA headers
        "cifar-dense/cuda/cu_dense_kernel.cuh",
        "cifar-dense/cuda/cu_kernels.cuh",

        -- Tree CUDA headers
        "tree/cuda/01_morton.cuh",
        "tree/cuda/02_sort.cuh",
        "tree/cuda/03_unique.cuh",
        "tree/cuda/04_radix_tree.cuh",
        "tree/cuda/05_edge_count.cuh",
        "tree/cuda/06_prefix_sum.cuh",
        "tree/cuda/07_octree.cuh",
        "tree/cuda/agents/prefix_sum_agent.cuh",
        "tree/cuda/agents/unique_agent.cuh",
        "tree/cuda/common/helper_cuda.hpp",
        "tree/cuda/common/helper_functions.hpp",
        "tree/cuda/common/helper_math.hpp",
        "tree/cuda/common/helper_string.hpp",
        "tree/cuda/common/helper_timer.hpp",
        "tree/cuda/common.cuh",
        "tree/cuda/func_morton.cuh",
        "tree/cuda/kernel.cuh",
    })

    add_files({
        -- Common CUDA implementations
        "common/cuda/cu_mem_resource.cu",
        
        -- CIFAR dense CUDA implementations
        "cifar-dense/cuda/cu_dense_kernel.cu",
        "cifar-dense/cuda/cu_kernels.cu",
        
        -- CIFAR sparse CUDA implementations
        "cifar-sparse/cuda/cu_dispatcher.cu",
        "cifar-sparse/cuda/cu_kernels.cu",

        -- Tree CUDA implementations
        "tree/cuda/01_morton.cu",
        "tree/cuda/02_sort.cu",
        "tree/cuda/03_unique.cu",
        "tree/cuda/04_radix_tree.cu",
        "tree/cuda/05_edge_count.cu",
        "tree/cuda/06_prefix_sum.cu",
        "tree/cuda/07_octree.cu",
        "tree/cuda/kernel.cu",
    })
    
    -- Best CUDA library
    add_packages("cub")
    add_cugencodes("native")
target_end()

end
-- end
