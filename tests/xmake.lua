add_requires("gtest")

-- ----------------------------------------------------------------
-- VK
-- ----------------------------------------------------------------

target("test-vk-sort")
    set_kind("binary")
    set_group("play")

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
    set_group("play")

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
    set_group("play")

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
