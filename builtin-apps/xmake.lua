target("builtin-apps")
    set_kind("static")
    
    add_headerfiles({
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
        "cifar-dense/dense_appdata.cpp",
        "cifar-sparse/sparse_appdata.cpp",
        "cifar-sparse/omp/sparse_kernel.cpp",
        "tree/tree_appdata.cpp",
        "tree/omp/tree_kernel.cpp",
    })

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("spdlog")
    add_packages("cli11")
    add_packages("glm")
target_end()

-- ----------------------------------------------------------------------------
-- Vulkan Static Library
-- ----------------------------------------------------------------------------

target("builtin-apps-vulkan")
    set_kind("static")
    add_deps("kiss-vk")
    
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

    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    add_packages("spdlog")
    add_packages("cli11")
    add_packages("glm")
target_end()

-- ----------------------------------------------------------------------------
-- CUDA Static Library
-- ----------------------------------------------------------------------------

if not is_plat("android") then
target("builtin-apps-cuda")
    set_kind("static")

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
        "tree/cuda/cu_dispatcher.cuh",
        "tree/cuda/func_morton.cuh",
        "tree/cuda/im_storage.cuh",
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
        "tree/cuda/cu_dispatcher.cu",
        "tree/cuda/im_storage.cu",
        "tree/cuda/kernel.cu",
    })

    add_packages("spdlog")
    add_packages("glm")
    add_packages("cub")

    add_cugencodes("native")
target_end()
end
