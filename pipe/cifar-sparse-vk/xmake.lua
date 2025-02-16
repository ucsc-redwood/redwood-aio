-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("pipe-cifar-sparse-vk")
    add_rules("common_flags", "vulkan_config", "run_on_android")
    set_kind("binary")
    add_files("main.cpp")

    add_includedirs("$(projectdir)/builtin-apps")
    add_includedirs("$(projectdir)")

    add_deps("builtin-apps", "builtin-apps-vulkan")
    add_packages("spdlog")

    add_packages("vulkan-hpp", "vulkan-memory-allocator")

    -- add_packages("concurrentqueue")

    -- Add openmp support
    if is_plat("android") then
        add_cxxflags("-fopenmp -static-openmp")
        add_ldflags("-fopenmp -static-openmp")
    else
        add_packages("openmp")
    end