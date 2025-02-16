target("pipe-cifar-sparse-vk")
    set_kind("binary")
    add_files("main.cpp")

    add_includedirs("$(projectdir)/builtin-apps")
    add_includedirs("$(projectdir)")

    add_deps("builtin-apps")
    add_deps("builtin-apps-vulkan")

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

    add_rules("run_on_android")