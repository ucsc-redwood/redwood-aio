-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- VK
-- ----------------------------------------------------------------

target("test-tree-vk")
    set_kind("binary")
    set_group("play")

    add_includedirs("$(projectdir)/builtin-apps/")
    add_files("main.cpp")

    add_deps("builtin-apps")


    add_packages("benchmark")
    -- add_packages("cli11")
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
