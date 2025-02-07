add_requires("gtest")

-- ----------------------------------------------------------------
-- Tree
-- ----------------------------------------------------------------

target("test-omp-tree")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_omp_tree.cpp")

    add_deps("builtin-apps")

    add_packages("gtest")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end
    add_packages("vulkan-hpp", "vulkan-memory-allocator")
    
    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

if not is_plat("android") then
target("test-cu-tree")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_cu_tree.cu")

    add_deps("builtin-apps")
    add_deps("builtin-apps-cuda")

    add_packages("gtest")

    -- Add CUDA flags
    add_cugencodes("native")

    add_cuflags("-Xcompiler", "-fopenmp", {force = true})  -- Enable OpenMP for the CUDA compiler
    add_ldflags("-fopenmp", {force = true})  -- Link against OpenMP library
    add_packages("openmp")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")
target_end()
end

target("test-vk-tree")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_vk_tree.cpp")

    add_deps("builtin-apps")

    add_packages("gtest")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()

-- ----------------------------------------------------------------
-- VK
-- ----------------------------------------------------------------

target("test-vk-sort")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_vk_sort.cpp")

    add_deps("builtin-apps")

    add_packages("benchmark")
    add_packages("gtest")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()


target("test-vk-prefix-sum")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_vk_prefix_sum.cpp")

    add_deps("builtin-apps")

    add_packages("benchmark")
    add_packages("gtest")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()



target("test-vk-prefix-sum-v2")
    set_kind("binary")
    set_group("test")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("test_vk_prefix_sum_v2.cpp")

    add_deps("builtin-apps")

    add_packages("benchmark")
    add_packages("gtest")

    add_packages("cli11")
    add_packages("spdlog")
    add_packages("glm")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    if is_plat("android") then
      on_run(run_on_android)
    end
target_end()
